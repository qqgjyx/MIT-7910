"""Two-track Newton-Schulz against the exact SVD (suites ts10m_mse_tt, ts10m_ce_tt): a table and the validation-loss curves.

    python -m optsplit.analysis.two_track --out <dir>
    python -m optsplit.analysis.two_track --loss ce --level 1.81 --offset 1 --ticks 2 1.75 1.5 1.4 1.35 1.3 --out <dir>

Per method (exact U S^p V^T at p = -0.3, two-track with 12, 8 and 5 steps, LC with 5, 6, 8 and 12 steps, plain Muon,
exact p = 0; exact and two-track 12 at p = -0.5 in the table only): the matrix products per step, the time per training step measured in a pilot (scripts/two_track_pilot.sh:
the methods one after another on one GPU, two runs side by side; LC in a second and third pilot, beside Muon and
two-track 12),
the error of the update against the exact one on the run's own momentum (`two_track`
records), the final validation loss per data order, the step at which the training loss reaches a level, and the GPU
time (step x time per step) at which the validation loss reaches plain Muon's final one on the same data order.
With --loss ce, all methods are timed in one pilot (ts10m_ce_tt_pilot), the exact runs log their own error against a
float64 SVD, and LC12 also runs on a schedule shortened to plain Muon's GPU time (`LC12, equal GPU time`). Methods
without runs are left out.
Writes <dir>/two_track.txt, <dir>/two_track.json and <dir>/figures/two_track_val.png (two_track_ce.txt, ... with ce).
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
METHODS = [
    ('exact SVD, $p=-0.3$', '{svp}/muon_p-0.3_lr{lr}_d{d}', '{p1}/svd_p-0.3', 'SVD', 'svd32',
     dict(color='#e34948')),
    ('two-track, $K=12$', '{tt}/muon_p-0.3_k12_d{d}', '{p1}/tt12', '47', '12', dict(color='#2a78d6')),
    ('two-track, $K=8$', '{tt}/muon_p-0.3_k8_d{d}', '{p1}/tt8', '31', '8', dict(color='#008300')),
    ('two-track, $K=5$', '{tt}/muon_p-0.3_k5_d{d}', '{p1}/tt5', '19', '5', dict(color='#eda100')),
    ('LC5', '{tt}/muon_p-0.3_lc5_d{d}', '{p2}/lc5', '15', 'lc5', dict(color='#6250d6', ls='-.')),
    ('LC6', '{tt}/muon_p-0.3_lc6_d{d}', '{p2}/lc6', '18', 'lc6', dict(color='#1baf7a', ls='-.')),
    ('LC8', '{tt}/muon_p-0.3_lc8_d{d}', '{p3}/lc8', '24', 'lc8', dict(color='#d55181', ls='-.')),
    ('LC12', '{tt}/muon_p-0.3_lc12_d{d}', '{p3}/lc12', '36', 'lc12', dict(color='#7a4a1e', ls='-.')),
    ('LC12, equal GPU time', '{tt}/muon_p-0.3_lc12_eqtime_d{d}', '{p3}/lc12', '36', 'lc12', dict(color='#7a4a1e', ls=':')),
    ('Muon (Newton–Schulz)', '{main}/muon_d{d}', '{p1}/muon', '15', None, dict(color='k', ls='--')),
    ('exact SVD, $p=0$', '{svp}/muon_p0_lr{lr}_d{d}', '{p1}/svd_p0', 'SVD', None, dict(color='0.55')),
    # p = -0.5, table only (style None): timed as p = -0.3, same products
    ('exact SVD, $p=-0.5$', '{svp}/muon_p-0.5_lr{lr}_d{d}', '{p1}/svd_p-0.3', 'SVD', None, None),
    ('two-track p=-0.5, $K=12$', '{tt}/muon_p-0.5_k12_d{d}', '{p1}/tt12', '47', '12', None),
]
ORDERS = (0, 1, 2)
SUITES = {'mse': dict(svp='ts10m_mse_svp', tt='ts10m_mse_tt', main='ts10m_mse_main', lr='0.005',
                      p1='ts10m_mse_tt_pilot', p2='ts10m_mse_tt_pilot2', p3='ts10m_mse_tt_pilot3'),
          'ce': dict(svp='ts10m_ce_svp', tt='ts10m_ce_tt', main='ts10m_main', lr='0.01',
                     p1='ts10m_ce_tt_pilot', p2='ts10m_ce_tt_pilot', p3='ts10m_ce_tt_pilot')}


def step_time(run):
    """Median seconds per training step, from the clock of the train records (evaluations fall out of the median)."""
    steps, t = run.series('train', 'time')
    return float(np.median(np.diff(t) / np.diff(steps))) if len(t) > 2 else np.nan


def update_error(runs, key):
    """Relative error of the update against the exact one: median and largest over the matrices and logged steps."""
    err = [e for run in runs for rec in run.records_of('two_track') for e in rec[f'err_{key}']]
    return (float(np.median(err)), float(np.max(err))) if err else (np.nan, np.nan)


def row(label, pattern, pilot, products, key, level, suites):
    runs = {d: Run(pattern.format(d=d, **suites)) for d in ORDERS}
    runs = {d: r for d, r in runs.items() if r.done}
    timed = [Run(f'{pilot.format(**suites)}_d{d}') for d in (0, 1)]
    hit = lambda r: float(at_loss(np.c_[r.series('eval', 'train_loss')[1], r.series('eval', 'train_loss')[0]], [level])[0])
    # the exact method's own error (float32 against float64 SVD): logged by the pilot's exact runs under MSE
    logged = key != 'svd32' or any(r.records_of('two_track') for r in runs.values())
    err = update_error(runs.values() if logged else timed, key) if key else (np.nan, np.nan)
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
    ap.add_argument('--loss', choices=list(SUITES), default='mse')
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
        r, runs = row(label, pattern, pilot, products, key, args.level, SUITES[args.loss])
        if not runs:
            continue
        rows.append(r)
        r['runs'] = runs
        if style:
            drawn.append((label, runs, style))
    muon = next(r['runs'] for r in rows if r['method'].startswith('Muon'))
    for r in rows:  # GPU minutes to plain Muon's final validation loss on the same data order
        target = {d: run.series('eval', 'val_loss')[1][-1] for d, run in muon.items()}
        r['gpu_min'] = {d: float(at_loss(np.c_[run.series('eval', 'val_loss')[1], run.series('eval', 'val_loss')[0]],
                                         [target[d]])[0]) * r['time'] / 60 for d, run in r.pop('runs').items() if d in target}
    mean = lambda v: f'{np.mean(list(v.values())):.4f}' if len(v) == len(ORDERS) else '-'
    cell = lambda v, d, fmt: format(v[d], fmt) if d in v else '-'
    lines = [f'{"method":<22} {"products":>8} {"ms/step":>8} {"err med":>8} {"err max":>8} '
             + ' '.join(f'{"val d" + str(d):>7}' for d in ORDERS) + f' {"mean":>7} '
             + ' '.join(f'{"step d" + str(d):>7}' for d in ORDERS) + ' '
             + ' '.join(f'{"min d" + str(d):>6}' for d in ORDERS)]
    for r in rows:
        lines.append(f'{r["method"]:<22} {r["products"]:>8} {1e3 * r["time"]:>8.0f} {r["err_median"]:>8.3f} '
                     f'{r["err_max"]:>8.3f} ' + ' '.join(f'{cell(r["val"], d, ".4f"):>7}' for d in ORDERS)
                     + f' {mean(r["val"]):>7} ' + ' '.join(f'{cell(r["step"], d, ".0f"):>7}' for d in ORDERS) + ' '
                     + ' '.join(f'{cell(r["gpu_min"], d, ".1f").replace("nan", "never"):>6}' for d in ORDERS))
    tag = '' if args.loss == 'mse' else f'_{args.loss}'
    (out / f'two_track{tag}.txt').write_text('\n'.join(lines) + '\n')
    (out / f'two_track{tag}.json').write_text(json.dumps(rows, indent=1))
    print('\n'.join(lines))
    curves(drawn, out / 'figures' / f'two_track_val{tag}.png', args.offset, args.ticks, args.zoom)


if __name__ == '__main__':
    main()
