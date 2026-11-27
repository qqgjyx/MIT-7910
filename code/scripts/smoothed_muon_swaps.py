"""Swap numbers for results/smoothed_muon_ce.tex (on the cluster, with the optsplit env):
    python scripts/smoothed_muon_swaps.py
For every swap of data order 0, the whole-model metrics at step 10,000 and their fraction of the way from the parent's
own run (0) to the new optimizer's own run (1) (the update metrics at step 9998); then the LaTeX rows of the table
of the Hessian trace at step 10,000."""
import json
import os

ROOT = os.environ['OPTSPLIT_ROOT'] + '/runs/'
OWN = {'adamw': 'ts10m_traj/adamw_d0', 'muon': 'ts10m_traj/muon_d0', 'smuon': 'ts10m_ce_smooth/smuon_d0'}
SUITE = {('adamw', 'muon'): 'ts10m_forks_dense', ('muon', 'adamw'): 'ts10m_forks_dense'}
DIRS = [('adamw', 'muon'), ('muon', 'adamw'), ('muon', 'smuon'), ('smuon', 'muon'), ('adamw', 'smuon'),
        ('smuon', 'adamw')]
STEPS = [50, 100, 200, 400, 700, 1000, 1500, 2500, 4000]
KEYS = ['stable_rank', 'spectral', 'mlp_active', 'update_align', 'lambda_max', 'hess_trace']
NAME = {'adamw': 'AdamW', 'muon': 'Muon', 'smuon': 'S-Muon'}


def final(run):
    """Metrics at step 10,000; those of the update (no step after 10,000) at step 9998."""
    ps = [f'{ROOT}{run}/traj_metrics/{s:07d}.json' for s in (9998, 10000)]
    if not all(os.path.exists(p) for p in ps):
        return None
    return {**json.load(open(ps[0])), **json.load(open(ps[1]))}


own = {o: final(r) for o, r in OWN.items()}
print('own runs at step 10000: ' + ' | '.join(f'{o}: ' + ' '.join(f'{k} {own[o][k]:.4g}' for k in KEYS) for o in own))
rows = {}
for a, b in DIRS:
    suite = SUITE.get((a, b), 'ts10m_ce_smooth_forks')
    print(f'== {a} -> {b} ({suite}): value (fraction) at step 10000')
    for k in KEYS:
        cells = []
        for s in STEPS:
            m = final(f'{suite}/{a}_to_{b}_at{s}')
            if m is None:
                cells.append(f'{s}:missing')
                continue
            f = (m[k] - own[a][k]) / (own[b][k] - own[a][k])
            cells.append(f'{s}:{m[k]:.4g}({f:.2f})')
            if k == 'hess_trace':
                rows.setdefault((a, b), []).append(f'{m[k]:.0f}')
        print(f'  {k}: ' + ' '.join(cells))
print('== trace table rows (LaTeX): swap & parent & 9 fork steps & new')
for (a, b), cells in rows.items():
    print(f'{NAME[a]} $\\to$ {NAME[b]} & {own[a]["hess_trace"]:.0f} & ' + ' & '.join(cells)
          + f' & {own[b]["hess_trace"]:.0f} \\\\')
