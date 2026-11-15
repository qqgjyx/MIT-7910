"""Cosine between consecutive one-step updates, cos(theta_{t+1} - theta_t, theta_{t+2} - theta_{t+1}) over all
parameters (`update_align` of metrics/trajectory.py), at every checkpoint triple t, t+1, t+2 of the runs: needs the
checkpoints only, so it serves runs without whole-model metrics (the calibration pilots of smoothed Muon).

    python -m optsplit.analysis.update_cosine 'ts10m_mse_smooth_pilot/*' [--level 0.65] [--from-step 1000] [--out f.json]

Prints, per run, the cosine at each triple, its mean over the triples from --from-step on, and its value at training
loss --level (linear in the loss between the triples, the loss of a triple interpolated from the eval records)."""

import argparse
import json

import numpy as np
import torch

from optsplit.analysis.io import load_runs
from optsplit.analysis.rank_metrics import at_loss


def cosines(run):
    """{t: cosine} over the triples of the run's checkpoints."""
    have = set(run.ckpt_steps)
    load = lambda s: torch.cat([v.flatten().float() for v in torch.load(run.ckpt_path(s), map_location='cpu')['model'].values()])
    out = {}
    for t in sorted(s for s in have if s + 1 in have and s + 2 in have and s - 1 not in have):
        a, b, c = load(t), load(t + 1), load(t + 2)
        out[t] = torch.nn.functional.cosine_similarity(b - a, c - b, dim=0).item()
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
        v = np.array([cos[t] for t in steps])
        loss = np.interp(steps, *run.series('eval', 'train_loss'))
        late = steps >= args.from_step
        res[run.id] = dict(steps=steps.tolist(), cosine=v.tolist(), train_loss=loss.tolist(), mean=v[late].mean(),
                           at_level=at_loss(np.c_[loss, v], [args.level])[0],
                           final_val_loss=run.series('eval', 'val_loss')[1][-1])
        print(f"{run.id}: mean from step {args.from_step} {res[run.id]['mean']:.3f}, at loss {args.level} "
              f"{res[run.id]['at_level']:.3f}, final val loss {res[run.id]['final_val_loss']:.4f}\n  "
              + ' '.join(f'{t}:{c:.3f}' for t, c in zip(steps, v)), flush=True)
    if args.out:
        json.dump(res, open(args.out, 'w'))


if __name__ == '__main__':
    main()
