"""Off-plan construction monitor: Sentinel-1 backscatter change per plan unit.
python src/monitor.py cities/<city> [--demo]   |   python src/monitor.py --index"""
import sys, os, re, json, math, base64, random, glob
from datetime import date, timedelta

DEFAULTS = dict(gee_project=os.environ.get('GEE_PROJECT', 'dahanga'), baseline_start='2025-01-01', baseline_end='2025-12-31',
                erode_m=10, outside_buffer_m=150, max_unit_ha=1.0, cell_m=50, min_unit_m2=200, min_reliable_ha=0.05,
                z_threshold=3.0, persistence=3, min_sigma_db=0.6)
LU = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'docs', 'landuse.json'), encoding='utf8'))

def bbox(g):
    polys = g['coordinates'] if g['type'] == 'MultiPolygon' else [g['coordinates']]
    xs, ys = zip(*[c for p in polys for c in p[0]]); return min(xs), min(ys), max(xs), max(ys)

def area_m2(g):
    tot = 0
    for poly in (g['coordinates'] if g['type'] == 'MultiPolygon' else [g['coordinates']]):
        for n, r in enumerate(poly):
            k = 111320 * math.cos(math.radians(r[0][1]))
            a = abs(sum(r[t][0]*k*r[t+1][1]*110570 - r[t+1][0]*k*r[t][1]*110570 for t in range(len(r)-1))) / 2
            tot += a if n == 0 else -a
    return tot

def load_plan(path):
    """Validate the standard plan file: GeoJSON, WGS84, polygons, property `land_use` from docs/landuse.json."""
    try: plan = json.load(open(path, encoding='utf8'))
    except Exception as e: sys.exit(f'ERROR: not valid GeoJSON / ፋይሉ ትክክለኛ GeoJSON አይደለም: {e}')
    if plan.get('type') != 'FeatureCollection' or not plan.get('features'):
        sys.exit('ERROR: need a FeatureCollection with at least one feature / ቢያንስ አንድ feature ያለው FeatureCollection ያስፈልጋል')
    bad, feats = {}, []
    for i, f in enumerate(plan['features']):
        g = f.get('geometry') or {}
        if g.get('type') not in ('Polygon', 'MultiPolygon'): sys.exit(f'ERROR: feature {i+1} must be Polygon/MultiPolygon, got {g.get("type")}')
        x0, y0, x1, y1 = bbox(g)
        if not (33 <= x0 and x1 <= 48 and 3 <= y0 and y1 <= 15):
            sys.exit(f'ERROR: feature {i+1} is outside Ethiopia. File must be WGS84 lon/lat (EPSG:4326) / መጋጠሚያው EPSG:4326 መሆን አለበት')
        p = f.get('properties') or {}
        k = re.sub(r'[\s\-]+', '_', str(p.get('land_use', '')).strip().lower())
        if k not in LU or LU[k]['class'] == 'outside_plan': bad[str(p.get('land_use'))] = bad.get(str(p.get('land_use')), 0) + 1; continue
        feats.append({'type': 'Feature', 'geometry': g, 'properties': {'plot_id': f'P{i+1:04d}', 'land_use': k, 'name': str(p.get('name', ''))[:60],
                      'zone_class': LU[k]['class'], 'area_ha': round(area_m2(g) / 1e4, 3)}})
    if bad: sys.exit(f'ERROR: unknown/missing land_use {bad}\nAllowed / የተፈቀዱ: ' + ', '.join(k for k in LU if LU[k]['class'] != 'outside_plan'))
    return {'type': 'FeatureCollection', 'features': feats}

def cells(ee, geom, box, cell_m, props):
    x0, y0, x1, y1 = box
    dx = cell_m / (111320 * math.cos(math.radians((y0 + y1) / 2))); dy = cell_m / 110570
    return [ee.Feature(ee.Geometry.Rectangle([x0 + i*dx, y0 + j*dy, x0 + (i+1)*dx, y0 + (j+1)*dy]).intersection(geom, 1),
                       {**props, 'unit_id': f"{props['plot_id']}-{i}-{j}"})
            for i in range(int((x1 - x0) / dx) + 1) for j in range(int((y1 - y0) / dy) + 1)]

def fetch_gee(cfg, plan):
    import ee
    k = json.loads(base64.b64decode(os.environ['GEE_JSON_KEY']))
    ee.Initialize(ee.ServiceAccountCredentials(k['client_email'], key_data=json.dumps(k)), project=cfg['gee_project'])
    units = []
    for f in plan['features']:
        p = f['properties']; keep = {q: p[q] for q in ('plot_id', 'land_use', 'zone_class')}
        g = ee.Geometry(f['geometry']).buffer(-cfg['erode_m'], 1)                  # drop edge pixels
        if p['area_ha'] <= cfg['max_unit_ha']: units.append(ee.Feature(g, {**keep, 'unit_id': p['plot_id']}))
        else: units += cells(ee, g, bbox(f['geometry']), cfg['cell_m'], keep)      # big polygon -> 50 m cells
    bx = [bbox(f['geometry']) for f in plan['features']]
    X0, Y0, X1, Y1 = min(b[0] for b in bx), min(b[1] for b in bx), max(b[2] for b in bx), max(b[3] for b in bx)
    R = cfg['outside_buffer_m']
    if R:                                                                          # ring around the plan = outside boundary
        union = ee.FeatureCollection([ee.Feature(ee.Geometry(f['geometry'])) for f in plan['features']]).geometry().dissolve(1)
        ex, ey = R / (111320 * math.cos(math.radians((Y0 + Y1) / 2))), R / 110570
        units += cells(ee, union.buffer(R, 1).difference(union, 1), (X0 - ex, Y0 - ey, X1 + ex, Y1 + ey), cfg['cell_m'],
                       {'plot_id': 'OUT', 'land_use': 'outside_plan', 'zone_class': 'outside_plan'})
    s1 = (ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(ee.Geometry.Rectangle([X0 - .01, Y0 - .01, X1 + .01, Y1 + .01]))
          .filterDate(cfg['baseline_start'], str(date.today())).filter(ee.Filter.eq('instrumentMode', 'IW'))
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')))
    h = s1.aggregate_histogram('relativeOrbitNumber_start').getInfo()
    print('latest S1 image used:', ee.Date(s1.aggregate_max('system:time_start')).format('YYYY-MM-dd').getInfo())
    s1 = s1.filter(ee.Filter.eq('relativeOrbitNumber_start', int(max(h, key=h.get)))).sort('system:time_start')  # one track only
    lin = lambda im: ee.Image(10).pow(im.select(['VV', 'VH']).divide(10)).set('system:index', im.get('system:index'))
    stack = s1.map(lin).toBands(); out = []
    for i in range(0, len(units), 200):
        fc = ee.FeatureCollection(units[i:i+200]).map(lambda f: f.set('a', f.geometry().area(1))).filter(ee.Filter.gte('a', cfg['min_unit_m2']))
        for f in stack.reduceRegions(fc, ee.Reducer.mean(), 10).getInfo()['features']:
            g, pr = f.get('geometry'), f['properties']
            if not g or g.get('type') not in ('Polygon', 'MultiPolygon') or not g.get('coordinates'): continue
            ds = {}
            for key, v in pr.items():
                m = re.search(r'(\d{8})T.*_(VV|VH)$', key)
                if m and v: ds.setdefault(m.group(1), {})[m.group(2)] = 10 * math.log10(v)
            days = sorted(d for d in ds if len(ds[d]) == 2)
            out.append(dict(unit_id=pr['unit_id'], props={q: pr[q] for q in ('plot_id', 'land_use', 'zone_class')}, geometry=g,
                            area_ha=area_m2(g) / 1e4, dates=[f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in days],
                            vv=[ds[d]['VV'] for d in days], vh=[ds[d]['VH'] for d in days]))
    return out

def fake(cfg, plan):
    rnd = random.Random(7); t = date.fromisoformat(cfg['baseline_start']); days = []
    while t <= date.today(): days.append(t); t += timedelta(days=6)
    b1 = date.fromisoformat(cfg['baseline_end']); out = []
    for f in plan['features']:
        p = f['properties']; base = rnd.uniform(-14, -9)
        s = rnd.choice([d for d in days if d > b1 + timedelta(days=30)]) if rnd.random() < .10 else None
        vv, vh = [], []
        for d in days:
            seas = .8 * math.sin(2 * math.pi * d.timetuple().tm_yday / 365); jump = 3.5 if s and d >= s else 0
            vv.append(base + seas + jump + rnd.gauss(0, .7)); vh.append(base - 6 + seas * .8 + jump * .6 + rnd.gauss(0, .7))
        out.append(dict(unit_id=p['plot_id'], props={q: p[q] for q in ('plot_id', 'land_use', 'zone_class')}, geometry=f['geometry'],
                        area_ha=p['area_ha'], dates=[str(d) for d in days], vv=vv, vh=vh))
    return out

def detect(u, cfg):
    import numpy as np
    d = np.array(u['dates'], dtype='datetime64[D]'); b0, b1 = np.datetime64(cfg['baseline_start']), np.datetime64(cfg['baseline_end'])
    base = (d >= b0) & (d <= b1)
    if base.sum() < 8: return None
    z = []
    for x in (u['vv'], u['vh']):                      # robust z-score vs the unit's own baseline period
        x = np.asarray(x, float); m = np.median(x[base])
        z.append((x - m) / max(1.4826 * np.median(np.abs(x[base] - m)), cfg['min_sigma_db']))
    zc = (z[0] + z[1]) / 2
    hit = (np.abs(zc) >= cfg['z_threshold']) & (d > b1); run = 0
    for i in range(len(d)):
        run = run + 1 if hit[i] and (run == 0 or np.sign(zc[i]) == np.sign(zc[i-1])) else (1 if hit[i] else 0)
        if run >= cfg['persistence']:                 # persistent change, not a one-off
            j = i - run + 1
            return dict(detected_date=str(d[j]), direction='increase' if zc[j] > 0 else 'decrease', z=round(float(zc[j:i+1].mean()), 1))
    return None

def write_index():
    out = [{'id': os.path.basename(os.path.dirname(m)), 'name': json.load(open(m, encoding='utf8'))['name']} for m in sorted(glob.glob('docs/data/*/meta.json'))]
    json.dump(out, open('docs/data/cities.json', 'w'), ensure_ascii=False)

def main(cdir, demo):
    cid = os.path.basename(os.path.normpath(cdir)); cfg = dict(DEFAULTS, city_id=cid, name=cid.replace('_', ' ').title())
    if os.path.exists(f'{cdir}/config.json'): cfg.update(json.load(open(f'{cdir}/config.json', encoding='utf8')))   # optional overrides
    plan = load_plan(f'{cdir}/plan.geojson')
    units = fake(cfg, plan) if demo else fetch_gee(cfg, plan)
    alerts = []
    for u in units:
        r = detect(u, cfg)
        if not r: continue
        props = {**u['props'], **r, 'unit_id': u['unit_id'], 'area_ha': round(u['area_ha'], 3),
                 'confidence': 'low' if u['area_ha'] < cfg['min_reliable_ha'] or r['direction'] == 'decrease' else 'medium',
                 'alert_type': 'activity' if u['props']['zone_class'] == 'buildable' else 'violation_candidate',
                 'dates': u['dates'], 'vv': [round(x, 2) for x in u['vv']], 'vh': [round(x, 2) for x in u['vh']]}
        alerts.append({'type': 'Feature', 'properties': props, 'geometry': u['geometry']})
    out = f"docs/data/{cid}"; os.makedirs(out, exist_ok=True)
    json.dump({'type': 'FeatureCollection', 'features': alerts}, open(f'{out}/alerts.geojson', 'w'))
    json.dump(plan, open(f'{out}/plan.geojson', 'w'))
    json.dump({'name': cfg['name'], 'demo': demo, 'generated': str(date.today()), 'baseline_end': cfg['baseline_end'],
               'units': len(units), 'alerts': len(alerts)}, open(f'{out}/meta.json', 'w'), ensure_ascii=False)
    write_index(); print(len(units), 'units,', len(alerts), 'alerts')

if __name__ == '__main__':
    write_index() if '--index' in sys.argv else main(sys.argv[1], '--demo' in sys.argv)
