"""Accuracy evaluation: compare alerts with ground truth (field / drone / VHR image checks).
python src/evaluate.py cities/<city> [--reviews reviews_<city>.csv]
Reads cities/<city>/ground_truth.csv (id,lat,lon,label,event_month,source,confidence,observer,notes; lines starting with # ignored),
docs/data/<city>/{alerts,plan}.geojson + meta.json. Writes docs/data/<city>/accuracy.json. Also called automatically by monitor.py."""
import sys, os, csv, json, math, statistics as st
from datetime import date

TOL_M, LAG_MIN, LAG_MAX = 30.0, -45, 120      # match distance; "timely" = detected from 45 d before to 120 d after event month start

def polys(g): return g['coordinates'] if g['type'] == 'MultiPolygon' else [g['coordinates']]

def pip(pt, g):                                 # even-odd rule (outer ring + holes)
    x, y = pt
    for poly in polys(g):
        ins = False
        for r in poly:
            j = len(r) - 1
            for i in range(len(r)):
                xi, yi, xj, yj = r[i][0], r[i][1], r[j][0], r[j][1]
                if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi: ins = not ins
                j = i
        if ins: return True
    return False

def dist_m(pt, g):
    if pip(pt, g): return 0.0
    k = 111320 * math.cos(math.radians(pt[1])); px, py = pt[0] * k, pt[1] * 110570; best = 1e18
    for poly in polys(g):
        for r in poly:
            for a, b in zip(r, r[1:]):
                ax, ay, bx, by = a[0] * k, a[1] * 110570, b[0] * k, b[1] * 110570
                dx, dy = bx - ax, by - ay; L = dx * dx + dy * dy
                t = 0 if L == 0 else max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / L))
                best = min(best, math.hypot(px - ax - t * dx, py - ay - t * dy))
    return best

def wilson(k, n, z=1.96):                       # [percent, low, high] 95% interval
    if n == 0: return None
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round(100 * p, 1), round(100 * (c - w) / d, 1), round(100 * (c + w) / d, 1)]

def run(cdir, out=None, reviews=None):
    cid = os.path.basename(os.path.normpath(cdir)); gt = f'{cdir}/ground_truth.csv'
    if not os.path.exists(gt): return None
    out = out or f'docs/data/{cid}'
    A = json.load(open(f'{out}/alerts.geojson', encoding='utf8'))['features']
    meta = json.load(open(f'{out}/meta.json', encoding='utf8')); plan = json.load(open(f'{out}/plan.geojson', encoding='utf8'))['features']
    b_end = date.fromisoformat(meta['baseline_end'])
    rows = list(csv.DictReader(l for l in open(gt, encoding='utf8') if l.strip() and not l.lstrip().startswith('#')))
    def near(pt):
        hits = []
        for a in A:
            d = dist_m(pt, a['geometry'])
            if d <= TOL_M: hits.append((d, a['properties']))
        return hits
    pos, neg, skipped = [], [], []
    for r in rows:
        try: pt = (float(r['lon']), float(r['lat'])); lab = r['label'].strip().lower()
        except Exception: skipped.append((r.get('id'), 'bad row')); continue
        zone = next((f['properties']['land_use'] for f in plan if pip(pt, f['geometry'])), 'outside_plan')
        hits = near(pt)
        if lab == 'new_construction':
            ev = date.fromisoformat(r['event_month'].strip()[:7] + '-01')
            if ev <= b_end: skipped.append((r['id'], 'event before monitoring start')); continue
            delays = [(date.fromisoformat(h[1]['detected_date']) - ev).days for h in hits]
            best = min(delays, key=abs) if delays else None
            pos.append(dict(id=r['id'], zone=zone, detected=bool(hits), delay_days=best, timely=best is not None and LAG_MIN <= best <= LAG_MAX))
        elif lab in ('no_change', 'non_construction_change'):
            neg.append(dict(id=r['id'], zone=zone, label=lab, false_alarm=bool(hits)))
        else: skipped.append((r['id'], f'unknown label {lab}'))
    det, tim, fa = sum(p['detected'] for p in pos), sum(p['timely'] for p in pos), sum(n['false_alarm'] for n in neg)
    res = dict(city=cid, demo=bool(meta.get('demo')), n_pos=len(pos), n_neg=len(neg), skipped=skipped, match_tolerance_m=TOL_M,
               recall_any=wilson(det, len(pos)), recall_timely=wilson(tim, len(pos)), false_alarm=wilson(fa, len(neg)),
               median_delay_days=st.median([p['delay_days'] for p in pos if p['delay_days'] is not None]) if det else None,
               recall_by_land_use={z: [sum(p['detected'] for p in pos if p['zone'] == z), sum(1 for p in pos if p['zone'] == z)] for z in sorted({p['zone'] for p in pos})},
               missed=[p['id'] for p in pos if not p['detected']], false_alarm_ids=[n['id'] for n in neg if n['false_alarm']],
               warning=None if len(pos) >= 30 and len(neg) >= 60 else 'Small sample: intervals are wide. Aim for >=30 new_construction and >=60 no_change/non_construction_change places.')
    if reviews and os.path.exists(reviews):
        rv = [l for l in csv.DictReader(open(reviews, encoding='utf8')) if l.get('review') in ('confirmed', 'false alarm')]
        ok = sum(1 for l in rv if l['review'] == 'confirmed'); res['precision_from_reviews'] = wilson(ok, len(rv)); res['n_reviewed'] = len(rv)
    json.dump(res, open(f'{out}/accuracy.json', 'w'), ensure_ascii=False, indent=1)
    print(f"accuracy [{cid}]{' (DEMO data: meaningless)' if res['demo'] else ''}: new constructions {len(pos)} -> found {det} (recall {res['recall_any']}), timely {res['recall_timely']}; "
          f"unchanged places {len(neg)} -> false alarms {fa} ({res['false_alarm']}); median delay {res['median_delay_days']} d; skipped {len(skipped)}")
    return res

if __name__ == '__main__':
    rv = sys.argv[sys.argv.index('--reviews') + 1] if '--reviews' in sys.argv else None
    run(sys.argv[1], reviews=rv)
