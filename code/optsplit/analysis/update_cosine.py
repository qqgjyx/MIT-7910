"""Cosine between consecutive one-step updates, cos(theta_{t+1} - theta_t, theta_{t+2} - theta_{t+1}) over all
parameters (`update_align` of metrics/trajectory.py), at every checkpoint triple t, t+1, t+2 of the runs: needs the
checkpoints only, so it serves runs without whole-model metrics (the calibration pilots of smoothed Muon).

    python -m optsplit.analysis.update_cosine 'ts10m_mse_smooth_pilot/*' [--level 0.65] [--from-step 1000] [--out f.json]

Prints, per run, the cosine at each triple, its mean over the triples from --from-step on, and its value at training
loss --level (linear in the loss between the triples, the loss of a triple interpolated from the eval records); then
the same for the block matrices alone (the parameters Muon updates) and for the rest (its auxiliary AdamW), and the
matrices' share of the squared norm of the first update of each triple: the cosine over all parameters is close to
the mean of the two parts' cosines weighted by their shares."""

import argparse
import json

import numpy as np
import torch

from optsplit.analysis.io import load_runs
from optsplit.analysis.rank_metrics import at_loss
from optsplit.metrics.trajectory import hidden


def cosines(run):
    """{t: (cosine over all parameters, over the block matrices, over the rest, share of the matrices in the squared
    norm of the update of t)} over the triples of the run's checkpoints."""
    have = set(run.ckpt_steps)
    cos = lambda u, v: torch.nn.functional.cosine_similarity(u, v, dim=0).item()

    def load(s):
        state = torch.load(run.ckpt_path(s), map_location='cpu')['model']
        mats = {id(v) for v in hidden(state)}
        # float64: on the CPU the float32 sums over 14M entries are off by 0.002 (not on the GPU, in trajectory.py)
        return [torch.cat([v.flatten().double() for v in vs]) for vs in
                (state.values(), hidden(state), [v for v in state.values() if id(v) not in mats])]

    out = {}
    for t in sorted(s for s in have if s + 1 in have and s + 2 in have and s - 1 not in have):
        a, b, c = load(t), load(t + 1), load(t + 2)
        out[t] = (*(cos(y - x, z - y) for x, y, z in zip(a, b, c)),
                  ((b[1] - a[1]).norm() ** 2 / (b[0] - a[0]).norm() ** 2).item())
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', nargs='+')
    p.add_argument('--level', type=float, default=0.65, help='training loss of the comparison')
    p.add_argument('--from-step', type=int, default=1000, help='first triple of the mean over training')
    p.add_argument('--out')
    args = p.parse_args()
    res = {}
    for run in load_runs(args.runs):
        cos = cosines(run)
        if not cos:
            continue
        steps = np.array(sorted(cos))
        loss = np.interp(steps, *run.series('eval', 'train_loss'))
        late = steps >= args.from_step
        res[run.id] = dict(steps=steps.tolist(), train_loss=loss.tolist(),
                           final_val_loss=run.series('eval', 'val_loss')[1][-1])
        print(f"{run.id}: final val loss {res[run.id]['final_val_loss']:.4f}")
        for i, part in enumerate(('all', 'matrices', 'rest', 'share of the matrices')):
            v = np.array([cos[t][i] for t in steps])
            res[run.id][part] = dict(cosine=v.tolist(), mean=v[late].mean(),
                                     at_level=at_loss(np.c_[loss, v], [args.level])[0])
            print(f"  {part}: mean from step {args.from_step} {v[late].mean():.3f}, at loss {args.level} "
                  f"{res[run.id][part]['at_level']:.3f}\n    " + ' '.join(f'{t}:{c:.3f}' for t, c in zip(steps, v)),
                  flush=True)
    if args.out:
        json.dump(res, open(args.out, 'w'))


if __name__ == '__main__':
    main()
