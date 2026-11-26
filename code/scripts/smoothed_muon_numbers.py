"""Numbers for the text of results/smoothed_muon_ce.tex (on the cluster, with the optsplit env):
    SMOOTH=ts10m_ce_smooth ORDERS='[012]' python scripts/smoothed_muon_numbers.py
Validation loss, JS between runs at equal step and equal training loss, floors, and the curvature over training.
ORDERS: the data orders of the smoothed-Muon runs that are done (glob, default all)."""
import glob
import json
import os
from itertools import combinations

import numpy as np

from optsplit.analysis.distance import loss_curve, step_curve
from optsplit.analysis.io import load_runs

ROOT = os.environ['OPTSPLIT_ROOT'] + '/runs/'
SM = os.environ.get('SMOOTH', 'ts10m_ce_smooth')
R = {'adamw': load_runs(['ts10m_main/adamw_d*']), 'muon': load_runs(['ts10m_main/muon_d*']),
     'smuon': load_runs([f"{SM}/smuon_d{os.environ.get('ORDERS', '*')}"])}
by = {o: {r.cfg.train.data_seed: r for r in rs} for o, rs in R.items()}

print('== validation loss')
final = {}
for o, rs in by.items():
    v = {d: r.series('eval', 'val_loss')[1][-1] for d, r in rs.items()}
    final[o] = v
    print(o, ' '.join(f'd{d}:{x:.4f}' for d, x in sorted(v.items())),
          f'mean {np.mean(list(v.values())):.4f} min {min(v.values()):.4f} max {max(v.values()):.4f}')
target = np.mean(list(final['adamw'].values()))
for o, rs in by.items():
    s, v = zip(*(r.series('eval', 'val_loss') for r in rs.values()))
    m = np.mean(v, 0)
    hit = np.asarray(s[0])[np.argmax(m <= target)] if (m <= target).any() else None
    print(f'{o} mean val reaches AdamW final mean {target:.4f} at step {hit}')
for o, rs in by.items():  # training loss 1.81: the step where each run reaches it
    st = []
    for r in rs.values():
        s, l = r.series('eval', 'train_loss')
        i = np.argmax(l <= 1.81)
        st.append(s[i - 1] + (s[i] - s[i - 1]) * (l[i - 1] - 1.81) / (l[i - 1] - l[i]))
    print(f'{o} reaches training loss 1.81 at steps', ' '.join(f'{x:.0f}' for x in st))

STEPS = [250, 500, 1000, 2000, 3000, 5000, 7500, 10000]
LOSSES = [3.0, 2.5, 2.2, 2.0, 1.81, 1.7, 1.6, 1.5, 1.45]


def at(c, key, xs):
    x, y = np.asarray(c[key], float), np.asarray(c['js'], float)
    o = np.argsort(x)
    return [float(np.interp(v, x[o], y[o], left=np.nan, right=np.nan)) for v in xs]


def table(curve, key, xs, title):
    print(f'== JS (bits) at equal {title}: ' + ' '.join(map(str, xs)))
    for a, b in (('adamw', 'muon'), ('smuon', 'muon'), ('smuon', 'adamw')):
        rows = [at(curve(by[a][d], by[b][d]), key, xs) for d in sorted(by[a].keys() & by[b].keys())]
        print(f'{a} vs {b} (same order, mean of {len(rows)}): ' + ' '.join(f'{x:.3f}' for x in np.nanmean(rows, 0)))
        print('   per order: ' + ' | '.join(' '.join(f'{x:.3f}' for x in r) for r in rows))
    for o in by:
        rows = [at(curve(by[o][s], by[o][t]), key, xs) for s, t in combinations(sorted(by[o]), 2)]
        print(f'floor {o} (mean of {len(rows)} pairs): ' + ' '.join(f'{x:.3f}' for x in np.nanmean(rows, 0)))


table(step_curve, 'step_a', STEPS, 'step')
table(loss_curve, 'loss', LOSSES, 'training loss')

print('== curvature over training (traj_metrics), mean +- half-range over d0-2')
for k in ('lambda_max', 'hess_trace', 'update_align', 'stable_rank', 'mlp_active'):
    print(k)
    for o, suite in (('adamw', 'ts10m_traj'), ('muon', 'ts10m_traj'), ('smuon', SM)):
        vals = {}
        for d in range(3):
            for f in glob.glob(f'{ROOT}{suite}/{o}_d{d}/traj_metrics/*.json'):
                j = json.load(open(f))
                if k in j:
                    vals.setdefault(j['step'], []).append(j[k])
        steps = [s for s in sorted(vals) if len(vals[s]) == 3 and (s % 500 == 0 or s in (25, 50, 100, 150, 200, 250, 300, 400, 600, 800, 1250)) and s > 0]
        print(f'  {o}: ' + ' '.join(f'{s}:{np.mean(vals[s]):.4g}±{(max(vals[s]) - min(vals[s])) / 2:.2g}' for s in steps))
