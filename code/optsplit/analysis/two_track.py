"""Two-track Newton-Schulz against the exact SVD (suite ts10m_mse_tt): a table and the validation-loss curves.

    python -m optsplit.analysis.two_track --out <dir>

Per method (exact U S^p V^T at p = -0.3, two-track with 12, 8 and 5 steps, LC with 5 and 6 steps, plain Muon,
exact p = 0; exact and two-track 12 at p = -0.5 in the table only): the matrix products per step, the time per training step measured in a pilot (scripts/two_track_pilot.sh:
the methods one after another on one GPU, two runs side by side; LC in a second pilot, beside Muon and two-track 12),
the error of the update against the exact one on the run's own momentum (`two_track`
records), the final validation loss per data order and the step at which the training loss reaches a level.
Writes <dir>/two_track.txt, <dir>/two_track.json and <dir>/figures/two_track_val.png.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from optsplit.analysis import ckpt_figs  # noqa: F401 (the figures' font sizes)
from optsplit.analysis.io import Run
from optsplit.analysis.plots import panel_labels
from optsplit.analysis.rank_metrics import at_loss

# label, run (data order {d}), pilot run (data orders 0, 1), matrix products per step, error key, line style
P1, P2 = 'ts10m_mse_tt_pilot', 'ts10m_mse_tt_pilot2'
METHODS = [
    ('exact SVD, $p=-0.3$', 'ts10m_mse_svp/muon_p-0.3_lr0.005_d{d}', f'{P1}/svd_p-0.3', 'SVD', 'svd32',
     dict(color='#e34948')),
    ('two-track, $K=12$', 'ts10m_mse_tt/muon_p-0.3_k12_d{d}', f'{P1}/tt12', '47', '12', dict(color='#2a78d6')),
    ('two-track, $K=8$', 'ts10m_mse_tt/muon_p-0.3_k8_d{d}', f'{P1}/tt8', '31', '8', dict(color='#008300')),
    ('two-track, $K=5$', 'ts10m_mse_tt/muon_p-0.3_k5_d{d}', f'{P1}/tt5', '19', '5', dict(color='#eda100')),
    ('LC5', 'ts10m_mse_tt/muon_p-0.3_lc5_d{d}', f'{P2}/lc5', '15', 'lc5', dict(color='#6250d6', ls='-.')),
    ('LC6', 'ts10m_mse_tt/muon_p-0.3_lc6_d{d}', f'{P2}/lc6', '18', 'lc6', dict(color='#1baf7a', ls='-.')),
    ('Muon (Newton–Schulz)', 'ts10m_mse_main/muon_d{d}', f'{P1}/muon', '15', None, dict(color='k', ls='--')),
    ('exact SVD, $p=0$', 'ts10m_mse_svp/muon_p0_lr0.005_d{d}', f'{P1}/svd_p0', 'SVD', None, dict(color='0.55')),
    # p = -0.5, table only (style None): timed as p = -0.3, same products
    ('exact SVD, $p=-0.5$', 'ts10m_mse_svp/muon_p-0.5_lr0.005_d{d}', f'{P1}/svd_p-0.3', 'SVD', None, None),
    ('two-track p=-0.5, $K=12$', 'ts10m_mse_tt/muon_p-0.5_k12_d{d}', f'{P1}/tt12', '47', '12', None),
]
ORDERS = (0, 1, 2)


def step_time(run):
    """Median seconds per training step, from the clock of the train records (evaluations fall out of the median)."""
    steps, t = run.series('train', 'time')
    return float(np.median(np.diff(t) / np.diff(steps))) if len(t) > 2 else np.nan


def update_error(runs, key):
    """Relative error of the update against the exact one: median and largest over the matrices and logged steps."""
    err = [e for run in runs for rec in run.records_of('two_track') for e in rec[f'err_{key}']]
    return (float(np.median(err)), float(np.max(err))) if err else (np.nan, np.nan)


def row(label, pattern, pilot, products, key, level):
    runs = {d: Run(pattern.format(d=d)) for d in ORDERS}
    runs = {d: r for d, r in runs.items() if r.done}
    timed = [Run(f'{pilot}_d{d}') for d in (0, 1)]
    hit = lambda r: float(at_loss(np.c_[r.series('eval', 'train_loss')[1], r.series('eval', 'train_loss')[0]], [level])[0])
    # the exact method's own error (float32 against float64 SVD) is logged by the pilot's exact runs
    err = update_error(timed if key == 'svd32' else runs.values(), key) if key else (np.nan, np.nan)
    return dict(method=label.replace('$', ''), products=products,
                time=float(np.mean([step_time(r) for r in timed if r.records_of('train')] or [np.nan])),
                err_median=err[0], err_max=err[1],
                val={d: float(r.series('eval', 'val_loss')[1][-1]) for d, r in runs.items()},
                step={d: hit(r) for d, r in runs.items()}), runs


def curves(drawn, out, offset, ticks, zoom):
    """Validation loss against the step: the whole run, and the steps from `zoom` on. Data order 0 thick, others thin."""
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), layout='constrained')
    handles = {}
    for label, runs, style in drawn:
        for d, run in runs.items():
            steps, val = run.series('eval', 'val_loss')
            width = dict(lw=1.0) if d == 0 else dict(lw=0.5, alpha=0.6)
            line, = axes[0].plot(steps, val, **style, **width)
            axes[1].plot(steps[steps >= zoom], val[steps >= zoom], **style, **width)
            if d == 0:
                handles[label] = line
    axes[0].set_yscale('function', functions=(lambda y: np.log(np.maximum(y - offset, 1e-9)),
                                              lambda v: np.exp(v) + offset))
    axes[0].set_yticks(ticks, labels=[f'{t:g}' for t in ticks])
    axes[0].set_ylim(min(np.nanmin(line.get_ydata()) for line in axes[0].lines) - 0.003, ticks[0])
    axes[0].set_xlim(0, None)
    axes[0].set_title('whole run')
    axes[1].set_xlim(zoom, None)
    axes[1].set_title(f'from step {zoom}')
    for ax in axes:
        ax.set_xlabel('step')
        ax.set_ylabel('validation loss')
    fig.legend(handles.values(), handles.keys(), loc='outside lower center', ncol=4, frameon=False)
    panel_labels(axes)
    fig.savefig(out, dpi=300)
    print(f'wrote {out}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--level', type=float, default=0.65)
    ap.add_argument('--offset', type=float, default=0.5, help='loss axis of the whole run on log(L - offset)')
    ap.add_argument('--ticks', type=float, nargs='+', default=[1.0, 0.9, 0.8, 0.7, 0.65, 0.6, 0.57])
    ap.add_argument('--zoom', type=int, default=5000)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out)
    (out / 'figures').mkdir(parents=True, exist_ok=True)

    rows, drawn = [], []
    for label, pattern, pilot, products, key, style in METHODS:
        r, runs = row(label, pattern, pilot, products, key, args.level)
        rows.append(r)
        if style:
            drawn.append((label, runs, style))
    mean = lambda v: f'{np.mean(list(v.values())):.4f}' if len(v) == len(ORDERS) else '-'
    cell = lambda v, d, fmt: format(v[d], fmt) if d in v else '-'
    lines = [f'{"method":<22} {"products":>8} {"ms/step":>8} {"err med":>8} {"err max":>8} '
             + ' '.join(f'{"val d" + str(d):>7}' for d in ORDERS) + f' {"mean":>7} '
             + ' '.join(f'{"step d" + str(d):>7}' for d in ORDERS)]
    for r in rows:
        lines.append(f'{r["method"]:<22} {r["products"]:>8} {1e3 * r["time"]:>8.0f} {r["err_median"]:>8.3f} '
                     f'{r["err_max"]:>8.3f} ' + ' '.join(f'{cell(r["val"], d, ".4f"):>7}' for d in ORDERS)
                     + f' {mean(r["val"]):>7} ' + ' '.join(f'{cell(r["step"], d, ".0f"):>7}' for d in ORDERS))
    (out / 'two_track.txt').write_text('\n'.join(lines) + '\n')
    (out / 'two_track.json').write_text(json.dumps(rows, indent=1))
    print('\n'.join(lines))
    curves(drawn, out / 'figures' / 'two_track_val.png', args.offset, args.ticks, args.zoom)


if __name__ == '__main__':
    main()
