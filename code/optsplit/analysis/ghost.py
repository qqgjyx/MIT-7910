"""Next step (ii), control A: weight spectra of the open-loop ghosts (optim.Ghost) against the closed-loop runs.
For each ghost <run>__ghost_<rule>, at every checkpoint step: stable rank, effective rank, spectral norm and
sigma_1/sigma_median of the block matrices (as in metrics/trajectory.py), and the cosine of consecutive updates
where the checkpoints t, t+1, t+2 exist; the same for the parent run and for a reference run of the ghost's rule.

    python -m optsplit.analysis.ghost ts10m_ii/adamw_ghost_d0:ts10m_traj/muon_d0 ... --out <dir>

Each argument is <parent run>:<reference run of the ghost's rule>. Writes <dir>/ghost.json and figures/ghost.pdf.

    python -m optsplit.analysis.ghost --figure <dir>/ghost.json --out <fig.pdf>

draws the report figure from ghost.json: rows AdamW and Muon parents, mean over the data orders given (band: range).
"""

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from optsplit.analysis.io import Run
from optsplit.analysis.rank_metrics import COLORS, NAMES
from optsplit.metrics.trajectory import ckpt_series

KEYS = ('stable_rank', 'eff_rank', 'spectral', 'sv_spread', 'update_align')


def figure(path, out):
    res = json.loads(Path(path).read_text())
    cols = (('stable_rank', 'stable rank of $W$'), ('sv_spread', r'$\sigma_1/\sigma_{\rm median}$ of $W$'),
            ('update_align', 'cosine of consecutive updates'))
    color, label = COLORS, NAMES
    fig, axes = plt.subplots(2, 3, figsize=(10, 4.6), sharex=True)
    for i, (row, (run, ghost)) in enumerate(zip(axes, (('adamw', 'muon'), ('muon', 'adamw')))):
        keys = [k for k in res if k.split('/')[-1].startswith(f'{run}_ghost_d')]
        for ax, (k, title) in zip(row, cols):
            for role, c, ls, lab in (('parent', color[run], '-', f'{label[run]} run'),
                                     ('reference', color[ghost], '-', f'{label[ghost]} run'),
                                     ('ghost', color[ghost], '--', f'{label[ghost]} rule on {label[run]} gradients')):
                steps = sorted({int(s) for r in keys for s, v in res[r][role].items() if k in v and int(s) > 0})
                steps = [s for s in steps if all(k in res[r][role].get(str(s), {}) for r in keys)]
                y = np.array([[res[r][role][str(s)][k] for s in steps] for r in keys])
                ax.plot(steps, y.mean(0), color=c, ls=ls, lw=1.3, label=lab)
                ax.fill_between(steps, y.min(0), y.max(0), color=c, alpha=0.2, lw=0)
            ax.set_xscale('log')
            if k == 'sv_spread':
                ax.set_yscale('log')
                ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
                ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
                ax.set_yticks([2, 4, 8, 16])
            ax.set_title(title, fontsize=9)
        row[1].legend(*row[0].get_legend_handles_labels(), fontsize=7, frameon=False, loc='upper left')
    for ax in axes[1]:
        ax.set_xlabel('step')
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('pairs', nargs='*')
    ap.add_argument('--out', required=True)
    ap.add_argument('--figure', help='ghost.json to draw the report figure from (to --out)')
    args = ap.parse_args()
    if args.figure:
        figure(args.figure, args.out)
        return
    out = Path(args.out)
    (out / 'figures').mkdir(parents=True, exist_ok=True)
    res = {}
    for pair in args.pairs:
        parent, ref = pair.split(':')
        rule = Run(ref).optimizer
        ghost = f'{parent}__ghost_{rule}'
        res[parent] = {'parent': ckpt_series(parent), 'ghost': ckpt_series(ghost), 'reference': ckpt_series(ref),
                       'names': {'parent': parent, 'ghost': ghost, 'reference': ref}}
        print(f'done {pair}', flush=True)
    (out / 'ghost.json').write_text(json.dumps(res))

    fig, axes = plt.subplots(len(res), len(KEYS), figsize=(3.2 * len(KEYS), 2.8 * len(res)), squeeze=False)
    for row, (parent, r) in zip(axes, res.items()):
        for ax, k in zip(row, KEYS):
            for role, style in (('parent', 'k-'), ('ghost', 'C3-'), ('reference', 'C0--')):
                pts = sorted((s, v[k]) for s, v in r[role].items() if k in v and s > 0)
                ax.plot(*zip(*pts), style, label=r['names'][role], lw=1.2)
            ax.set_xscale('log')
            ax.set_title(k, fontsize=9)
            ax.set_xlabel('step')
        row[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out / 'figures' / 'ghost.pdf')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
