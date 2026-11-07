"""Next step (iii), swap timescale: after a swap A->B at step k, how fast do the update cosine, the weight spectra
and the loss rate move to B's? For each fork in ts10m_forks_dense, at its checkpoints k, k+1, k+2, k+5, ...:
stable rank and sigma_1/sigma_median of the block matrices, the cosine of consecutive updates where t, t+1, t+2
exist, and the paired training-loss gap to the A parent (same batches, logged every 10 steps).

The spectra are also given as a fraction of the way from the A parent to the B parent at the same step
(parents interpolated in log step between their checkpoints): 0 = still A, 1 = at B.

    python -m optsplit.analysis.swap_timescale 'ts10m_forks_dense/*' --parents ts10m_main ts10m_parent_late \
        --out <dir>

Writes <dir>/swap_timescale.json and <dir>/swap_timescale.pdf.

    python -m optsplit.analysis.swap_timescale --figure <dir>/swap_timescale.json --out <fig.pdf>

redraws the report figure from the json.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from optsplit.config import root_dir, run_dir
from optsplit.metrics.trajectory import hidden, matrix_stats

WINDOWS = ((0, 10), (10, 30), (30, 100), (100, 300), (300, 1000))


def ckpt_steps(name):
    return sorted(int(p.stem) for p in (run_dir(name) / 'ckpt').glob('*.pt'))


def stats(name, steps=None):
    d = run_dir(name) / 'ckpt'
    all_steps = ckpt_steps(name)
    load = lambda s: hidden(torch.load(d / f'{s:07d}.pt', map_location='cpu')['model'])
    out = {}
    for s in steps if steps is not None else all_steps:
        W = load(s)
        rec = matrix_stats(W)
        if s + 1 in all_steps and s + 2 in all_steps:
            W1, W2 = load(s + 1), load(s + 2)
            a = torch.cat([(b - c).flatten() for b, c in zip(W1, W)])
            b = torch.cat([(c - e).flatten() for c, e in zip(W2, W1)])
            rec['update_align'] = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
        out[s] = rec
    return out


def train_loss(name):
    recs = [json.loads(l) for l in open(run_dir(name) / 'metrics.jsonl')]
    return {r['step']: r['loss'] for r in recs if r.get('kind') == 'train'}


def interp(curve, s, key):
    """curve {step: rec}; value of key at step s, linear in log(1 + step)."""
    xs = sorted(t for t in curve if key in curve[t])
    return float(np.interp(np.log1p(s), np.log1p(xs), [curve[t][key] for t in xs]))


def config(name):
    import yaml
    return yaml.safe_load(open(run_dir(name) / 'config.yaml'))


def analyse(forks, parent_suites):
    parents = {}
    for suite in parent_suites:
        for d in sorted((root_dir() / 'runs' / suite).iterdir()):
            opt = config(f'{suite}/{d.name}')['optimizer']
            parents.setdefault(opt, {}).update(stats(f'{suite}/{d.name}'))
            print(f'parent {suite}/{d.name}', flush=True)
    loss_p = {}
    for opt in parents:
        loss_p[opt] = {}
        for suite in parent_suites:
            for d in (root_dir() / 'runs' / suite).iterdir():
                if config(f'{suite}/{d.name}')['optimizer'] == opt:
                    loss_p[opt].update(train_loss(f'{suite}/{d.name}'))
    res = {}
    for f in forks:
        a, b = f.split('/')[-1].split('_at')[0].split('_to_')
        k = int(f.split('_at')[-1])
        st = {s: v for s, v in stats(f).items() if s >= k}
        for s, v in st.items():
            for key in ('stable_rank', 'sv_spread'):
                va, vb = interp(parents[a], s, key), interp(parents[b], s, key)
                v[f'{key}_frac'] = (v[key] - va) / (vb - va)
        lf = train_loss(f)
        gap = {s - k: lf[s] - loss_p[a][s] for s in lf if s in loss_p[a]}
        gap_ba = {s - k: loss_p[b][s] - loss_p[a][s] for s in lf if s in loss_p[a] and s in loss_p[b]}
        rates = {}
        for lo, hi in WINDOWS:  # least-squares slope of the paired gap, per 1000 steps
            pts = [(t, g) for t, g in gap.items() if lo <= t <= hi]
            if len(pts) >= 2:
                t, g = np.array(pts).T
                rates[f'{lo}-{hi}'] = float(np.polyfit(t, g, 1)[0] * 1000)
        res[f] = dict(a=a, b=b, k=k, stats={s - k: v for s, v in st.items()}, gap=gap, gap_parents=gap_ba,
                      gap_rate=rates)
        print(f'fork {f}', flush=True)
    return res


def half_time(st, key):
    for t in sorted(st):
        if key in st[t] and st[t][key] >= 0.5:
            return t
    return None


def figure(res, out, ks=(400, 1000, 2500, 6000), horizon=3000):
    """Report figure: rows AdamW to Muon and Muon to AdamW; columns the loss gap to the old rule's run (the
    parents' gap at the same step dotted), the stable rank as a fraction of the way to the new rule's run, and the
    update cosine; one line per swap step in `ks`."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    label = {'adamw': 'AdamW', 'muon': 'Muon'}
    fig, axes = plt.subplots(2, 3, figsize=(10, 4.6), sharex=True)
    for row, a in zip(axes, ('adamw', 'muon')):
        fs = sorted((r for r in res.values() if r['a'] == a and r['k'] in ks), key=lambda r: r['k'])
        b = fs[0]['b']
        for i, r in enumerate(fs):
            c = plt.get_cmap('viridis')(0.85 * i / max(1, len(fs) - 1))
            pts = sorted((int(t), g) for t, g in r['gap'].items() if 0 < int(t) <= horizon)
            row[0].plot(*zip(*pts), color=c, lw=1.1, label=f'swap at {r["k"]}')
            pts = sorted((int(t), g) for t, g in r['gap_parents'].items() if 0 < int(t) <= horizon)
            row[0].plot(*zip(*pts), color=c, lw=0.8, ls=':')
            for ax, key in zip(row[1:], ('stable_rank_frac', 'update_align')):
                pts = sorted((max(int(t), 1), v[key]) for t, v in r['stats'].items()
                             if key in v and int(t) <= horizon)
                ax.plot(*zip(*pts), color=c, lw=1.1, marker='.', ms=3)
        row[0].axhline(0, color='0.6', lw=0.6)
        row[0].set_title(f'{label[a]} to {label[b]}: train loss minus the {label[a]} run', fontsize=9)
        row[1].set_title(f'stable rank of $W$, fraction of the way to {label[b]}', fontsize=9)
        row[1].axhline(0, color='0.6', lw=0.6)
        row[1].axhline(1, color='0.6', lw=0.6)
        row[2].set_title('cosine of consecutive updates', fontsize=9)
        row[2].set_ylim(0.5, 1.0)
        row[0].legend(fontsize=7, frameon=False)
        for ax in row:
            ax.set_xscale('log')
    for ax in axes[1]:
        ax.set_xlabel('steps after the swap')
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('forks', nargs='*')
    ap.add_argument('--parents', nargs='+')
    ap.add_argument('--out', required=True)
    ap.add_argument('--figure', help='swap_timescale.json to draw the report figure from (to --out)')
    args = ap.parse_args()
    if args.figure:
        figure(json.loads(Path(args.figure).read_text()), args.out)
        return
    forks = []
    for p in args.forks:
        if '*' in p:
            suite = p.split('/')[0]
            forks += [f'{suite}/{d.name}' for d in sorted((root_dir() / 'runs' / suite).iterdir()) if '_to_' in d.name]
        else:
            forks.append(p)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = analyse(forks, args.parents)
    (out / 'swap_timescale.json').write_text(json.dumps(res))
    figure(res, out / 'swap_timescale.pdf')
    print('fork  k  t_half(stable rank)  t_half(sv_spread)  cos(k)  gap rate per 1000 steps by window')
    for f, r in sorted(res.items(), key=lambda x: (x[1]['a'], x[1]['k'])):
        c0 = r['stats'].get(0, {}).get('update_align')
        print(f"{r['a']}->{r['b']} {r['k']} {half_time(r['stats'], 'stable_rank_frac')} "
              f"{half_time(r['stats'], 'sv_spread_frac')} {c0} {r['gap_rate']}")


if __name__ == '__main__':
    main()
