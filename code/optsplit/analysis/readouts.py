"""Distance between AdamW and Muon runs under several readouts of the stored outputs, at equal step and at equal
training loss: the cross pairs (same data order) and both floors (same optimizer, two data orders).
Readouts: 'clamp' JS of the stored predictions (as Fig 1; max(z,0)/sum under mse); 'soft' JS of softmax(x/T), T
fitted per checkpoint to minimise the cross-entropy of the probe targets; 'l2' the mean over tokens of
|x_a - x_b| / rms(|x_a|, |x_b|); 'disagree' top-1 disagreement. x is z under mse and the centred log-probabilities
(the logits up to a constant) under ce. Also writes each run's training loss and whole-model metrics.
    python -m optsplit.analysis.readouts ts10m_mse_main --out OUT.json"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from optsplit.analysis.distance import js_divergence, loss_points
from optsplit.analysis.io import load_runs
from optsplit.analysis.split import by_seed

torch.set_num_threads(8)


def fit_t(x, y):
    """Temperature minimising the mean cross-entropy of softmax(x/T) at y (golden section on log T)."""
    ce = lambda lt: F.cross_entropy(x / np.exp(lt), y).item()
    a, b, g = -8.0, 4.0, (5 ** 0.5 - 1) / 2
    for _ in range(30):
        c, d = b - g * (b - a), a + g * (b - a)
        a, b = (a, d) if ce(c) < ce(d) else (c, b)
    lt = (a + b) / 2
    return float(np.exp(lt)), ce(lt)


class View:
    """The readouts of one run at one step."""

    def __init__(self, run, step, T=None):
        d = torch.load(run.path / 'preds' / f'{step:07d}.pt', map_location='cpu')
        x = d['z'].float() if 'z' in d else d['logp'].float()
        if 'z' not in d:
            x = x - x.mean(-1, keepdim=True)
        self.clamp = run.logp(step).float()
        self.T, self.ce = fit_t(x, run.targets.long()) if T is None else (T, None)
        self.soft = F.log_softmax(x / self.T, -1)
        self.x, self.top = x, x.argmax(-1)


def dist(u, v):
    l2 = ((u.x - v.x).norm(dim=-1) / ((u.x.pow(2).sum(-1) + v.x.pow(2).sum(-1)) / 2).sqrt()).mean().item()
    return dict(clamp=js_divergence(u.clamp, v.clamp), soft=js_divergence(u.soft, v.soft), l2=l2,
                disagree=1 - (u.top == v.top).float().mean().item())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('suite')
    p.add_argument('--out', required=True)
    p.add_argument('--tol', type=float, default=0.02)
    args = p.parse_args()
    A = by_seed(load_runs(f'{args.suite}/adamw_d*'))
    B = by_seed(load_runs(f'{args.suite}/muon_d*'))
    pairs = [('cross', A[s], B[s]) for s in sorted(A.keys() & B.keys())]
    pairs += [(f'floor_{g}', G[s], G[t]) for g, G in (('adamw', A), ('muon', B)) for s, t in combinations(sorted(G), 2)]
    runs = list(A.values()) + list(B.values())
    steps = sorted(set.intersection(*(set(r.pred_steps) for r in runs)))
    out = dict(steps=steps, step={}, loss={}, runs={})
    temps = {}
    for step in steps:  # equal step
        views = {r.id: View(r, step) for r in runs}
        for r in runs:
            temps[r.id, step] = (views[r.id].T, views[r.id].ce)
        for kind, a, b in pairs:
            out['step'].setdefault(f'{kind}|{a.id}|{b.id}', []).append(dist(views[a.id], views[b.id]))
        print(step, {k: round(v[-1]['soft'], 4) for k, v in list(out['step'].items())[:2]}, flush=True)
    for kind, a, b in pairs:  # equal training loss, matched as distance.loss_curve
        sa, la = loss_points(a)
        sb, lb = loss_points(b)
        j = np.abs(lb[None, :] - la[:, None]).argmin(1)
        ok = np.abs(lb[j] - la) <= args.tol * np.abs(la)
        rows = []
        for s, t, l in zip(sa[ok], sb[j[ok]], la[ok]):
            rows.append(dict(step_a=int(s), step_b=int(t), loss=float(l), **dist(View(a, int(s), temps[a.id, int(s)][0]), View(b, int(t), temps[b.id, int(t)][0]))))
        out['loss'][f'{kind}|{a.id}|{b.id}'] = rows
        print('loss-matched', kind, a.id, b.id, len(rows), flush=True)
    for r in runs:
        tm = {}
        d = r.path / 'traj_metrics'
        for f in sorted(d.glob('*.json')) if d.exists() else []:
            m = json.loads(f.read_text())
            tm[int(f.stem)] = {k: m.get(k) for k in ('stable_rank', 'mlp_active', 'eff_rank', 'feature_erank')}
        out['runs'][r.id] = dict(
            optimizer=r.optimizer, data_seed=r.data_seed,
            train=[list(map(float, v)) for v in r.series('eval', 'train_loss')],
            val=[list(map(float, v)) for v in r.series('eval', 'val_loss')],
            T=[temps[r.id, s][0] for s in steps], soft_ce=[temps[r.id, s][1] for s in steps], traj_metrics=tm)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    print('wrote', args.out)


if __name__ == '__main__':
    main()
