"""Dose-response of Muon's update U S^p V^T (suite ts10m_mse_svp): per run, the step at which the training loss
reaches a level, the stable rank and sigma_1 / sigma_median of the block matrices there, and the final validation loss;
and the training-loss curves, one colour per p, with plain Muon and AdamW as references.

    python -m optsplit.analysis.dose [--suite ts10m_mse_svp] [--level 0.65] --out <dir>

Needs the whole-model metrics of the runs (python -m optsplit.metrics.trajectory). Writes <dir>/dose.txt (every run),
<dir>/dose.json and <dir>/figures/dose_loss.pdf.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from optsplit.analysis import ckpt_figs  # noqa: F401 (the figures' font sizes)
from optsplit.analysis.io import Run, load_runs
from optsplit.analysis.plots import panel_labels
from optsplit.analysis.rank_metrics import at_loss

REFS = {'muon': ('Muon (Newton–Schulz)', dict(color='k', ls='--')), 'adamw': ('AdamW', dict(color='0.55', ls=':'))}


def row(run, level):
    steps, loss = run.series('eval', 'train_loss')
    traj = [json.loads(f.read_text()) for f in sorted((run.path / 'traj_metrics').glob('*.json'))]
    at = lambda v: float(at_loss(np.c_[[t['train_loss'] for t in traj], v], [level])[0]) if len(traj) > 1 else np.nan
    return dict(run=run.id, p=run.cfg.optim.muon.get('sv_power'), lr=run.lr, data_seed=run.data_seed, done=run.done,
                step=float(at_loss(np.c_[loss, steps], [level])[0]),
                stable_rank=at([t['stable_rank'] for t in traj]), sv_spread=at([t['sv_spread'] for t in traj]),
                final_step=int(steps[-1]), train=loss[-1], val=run.series('eval', 'val_loss')[1][-1])


def color(p):
    """p < 0 red, p > 0 blue, darker further from 0 (down to -1, up to 0.75); p = 0 black."""
    return 'k' if p == 0 else plt.get_cmap('Reds')(0.3 - 0.65 * p) if p < 0 else plt.get_cmap('Blues')(0.35 + 0.8 * p)


def curves(by_lr, refs, level, out, max_loss, offset, ticks):
    """One panel per learning rate: training loss on the probe against the step, a thin line at `level`."""
    fig, axes = plt.subplots(1, len(by_lr), figsize=(7.1, 2.9), sharey=True, layout='constrained', squeeze=False)
    handles = {}
    for ax, (lr, runs) in zip(axes[0], by_lr.items()):
        for run in sorted(runs, key=lambda r: r.cfg.optim.muon.sv_power):
            p = run.cfg.optim.muon.sv_power
            handles[f'$p={p:g}$'], = ax.plot(*run.series('eval', 'train_loss'), lw=0.9, color=color(p))
        for name, run in refs[lr].items():
            handles[REFS[name][0]], = ax.plot(*run.series('eval', 'train_loss'), lw=1, **REFS[name][1])
        ax.axhline(level, color='k', lw=0.4)
        ax.set_yscale('function', functions=(lambda y: np.log(np.maximum(y - offset, 1e-9)),
                                             lambda v: np.exp(v) + offset))
        ax.set_yticks(ticks, labels=[f'{t:g}' for t in ticks])
        ax.set_xlim(0, None)
        ax.set_xlabel('step')
        ax.set_title(f'learning rate {lr:g}')
    axes[0, 0].set_ylim(min(np.nanmin(line.get_ydata()) for ax in axes[0] for line in ax.lines) - 0.003, max_loss)
    axes[0, 0].set_ylabel('training loss')
    fig.legend(handles.values(), handles.keys(), loc='outside lower center', ncol=6, frameon=False)
    if len(by_lr) > 1:
        panel_labels(axes[0])
    fig.savefig(out)
    print(f'wrote {out}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--suite', default='ts10m_mse_svp')
    ap.add_argument('--main', default='ts10m_mse_main', help='suite with plain muon_d* and adamw_d0')
    ap.add_argument('--sweep', default='ts10m_mse_lrsweep', help='suite with plain Muon at the other learning rates')
    ap.add_argument('--level', type=float, default=0.65)
    ap.add_argument('--max-loss', type=float, default=0.9)
    ap.add_argument('--offset', type=float, default=0.5, help='loss axis on log(L - offset)')
    ap.add_argument('--ticks', type=float, nargs='+', default=[0.9, 0.8, 0.7, 0.65, 0.6, 0.57])
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out)
    (out / 'figures').mkdir(parents=True, exist_ok=True)

    started = lambda pats: [r for r in load_runs(pats) if r.records_of('eval')]
    runs = started(f'{args.suite}/*_d0')
    main_lr = Run(f'{args.main}/muon_d0').lr
    by_lr = {lr: [r for r in runs if r.lr == lr] for lr in sorted({r.lr for r in runs}, key=lambda x: x != main_lr)}
    refs = {}
    for lr in by_lr:  # plain Muon at this rate (the main run, or the sweep's), and AdamW
        muon = [r for r in [Run(f'{args.main}/muon_d0')] + load_runs(f'{args.sweep}/muon_*') if r.lr == lr]
        refs[lr] = {'muon': muon[0], 'adamw': Run(f'{args.main}/adamw_d0')} if muon else {'adamw': Run(f'{args.main}/adamw_d0')}

    rows = [row(r, args.level) for lr in by_lr for r in sorted(by_lr[lr], key=lambda r: r.cfg.optim.muon.sv_power)]
    rows += [row(r, args.level) for r in {r.id: r for lr in refs for r in refs[lr].values()}.values()]
    rows += [row(r, args.level) for r in started([f'{args.suite}/*_d[1-9]', f'{args.main}/muon_d[1-9]'])]  # other orders
    lines = [f'{"run":<44} {"lr":>7} {"p":>6} {"step@" + format(args.level, "g"):>9} {"st.rank":>8} {"s1/smed":>8} '
             f'{"last":>6} {"train":>7} {"val":>7}']
    for r in rows:
        lines.append(f'{r["run"]:<44} {r["lr"]:>7g} {str(r["p"]):>6} {r["step"]:>9.0f} {r["stable_rank"]:>8.1f} '
                     f'{r["sv_spread"]:>8.2f} {r["final_step"]:>6} {r["train"]:>7.4f} {r["val"]:>7.4f}'
                     + ('' if r['done'] else '  (not done)'))
    (out / 'dose.txt').write_text('\n'.join(lines) + '\n')
    (out / 'dose.json').write_text(json.dumps(rows, indent=1))
    print('\n'.join(lines))
    curves(by_lr, refs, args.level, out / 'figures' / 'dose_loss.pdf', args.max_loss, args.offset, args.ticks)


if __name__ == '__main__':
    main()
