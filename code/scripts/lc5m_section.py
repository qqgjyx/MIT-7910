"""The last section of results/dose_negative_p_ce.tex: LC5m (LC5 with every singular value of its update kept between
Muon's and the exact s^-0.3) against Muon, LC5, the exact update and AdamW, one row per data order, and the
validation loss against GPU time of LC5m, Muon and LC5 (as gpu_time_val_ce.pdf). GPU minutes as in gpu_time.py.
Overwrites results/lc5m_section.tex (which the document inputs) and results/ts10m/lc5m_gpu_time_ce.pdf. Run by
scripts/lc5m_section.sh, which first copies the logs from ORCD (--files lists them):

    python scripts/lc5m_section.py [--pilot <suite>/<name without _d0>] <LC5m run id> ...

Without --pilot, LC5m is timed as LC5.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from optsplit.analysis.dose import REFS, first_step, log_axis
from optsplit.analysis.gpu_time import PILOTS, STYLE
from optsplit.analysis.io import Run
from optsplit.analysis.plots import panel_labels
from optsplit.analysis.two_track import step_time

RESULTS = Path(__file__).resolve().parents[2] / 'results'
FIG = 'ts10m/lc5m_gpu_time_ce.pdf'
ORDERS = range(5)
REF = {'Muon': ('ts10m_main/muon_d{}', 'muon'), 'LC5': ('ts10m_ce_tt/muon_p-0.3_lc5_d{}', 'lc5'),
       'exact': ('ts10m_ce_svp/muon_p-0.3_lr0.01_d{}', 'svd'), 'AdamW': ('ts10m_main/adamw_d{}', 'adamw')}
STYLES = {'LC5m': dict(color='#eda100'), 'Muon': REFS['muon'][1], 'LC5': STYLE['LC5']}
LABELS = {'LC5m': 'LC5m', 'Muon': REFS['muon'][0], 'LC5': 'LC5'}


def final(r):
    return r.series('eval', 'val_loss')[1][-1]


def plot(runs, sec, orders, target, zoom=(10, 27), ticks=(2, 1.75, 1.5, 1.4, 1.35, 1.3)):
    """gpu_time.plot for LC5m, Muon and LC5 on `orders`; panel (b) capped at AdamW's loss at the start of `zoom`."""
    steps, val = runs['AdamW'][0].series('eval', 'val_loss')
    hi = val[steps * sec['AdamW'] / 60 >= zoom[0]].max()
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), layout='constrained')
    handles, lo = {}, np.inf
    for name, style in STYLES.items():
        for d in orders:
            if d not in runs[name]:
                continue
            steps, val = runs[name][d].series('eval', 'val_loss')
            minutes = steps * sec[name] / 60
            handles[LABELS[name]], = axes[0].plot(minutes, val, lw=1, **style)
            axes[1].plot(minutes, val, lw=1, **style)
            win = (minutes >= zoom[0]) & (minutes <= zoom[1])
            lo = min(lo, val[win].min()) if win.any() else lo
    for ax in axes:
        ax.axhline(target, color='k', lw=0.4)
        ax.set(xlabel='GPU minutes', ylabel='validation loss')
    log_axis(axes[0], 1, ticks)
    axes[0].set_ylim(min(np.nanmin(line.get_ydata()) for line in axes[0].lines) - 0.003, ticks[0])
    axes[0].set_xlim(0, None)
    axes[0].set_title('whole run')
    axes[1].set(xlim=zoom, ylim=(lo - 0.005, hi + 0.005))
    axes[1].set_title(f'{zoom[0]:g} to {zoom[1]:g} minutes')
    fig.legend(handles.values(), handles.keys(), loc='outside lower center', ncol=3, frameon=False)
    panel_labels(axes)
    fig.savefig(RESULTS / FIG)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('runs', nargs='*', help='LC5m run ids (<suite>/<name>), finished or not')
    ap.add_argument('--pilot', help='LC5m timing pilot, as in gpu_time.PILOTS (runs <pilot>_d0 and _d1)')
    ap.add_argument('--files', action='store_true', help='only print the files to copy from ORCD')
    args = ap.parse_args()
    pilots = {m: PILOTS[m] for _, m in REF.values()} | ({'lc5m': args.pilot} if args.pilot else {})
    if args.files:
        ids = [p.format(d) for p, _ in REF.values() for d in ORDERS] + args.runs
        ids += [f'{p}_d{d}' for p in pilots.values() for d in (0, 1)]
        print(' '.join(f'{i}/{f}' for i in ids for f in ('config.yaml', 'metrics.jsonl', 'done')))
        return

    ms = {m: float(np.mean([step_time(Run(f'{p}_d{d}')) for d in (0, 1)])) for m, p in pilots.items()}
    sec = {name: ms[m] for name, (_, m) in REF.items()} | {'LC5m': ms.get('lc5m', ms['lc5'])}
    runs = {name: {d: r for d in ORDERS if (r := Run(p.format(d))).done} for name, (p, _) in REF.items()}
    runs['LC5m'] = {r.data_seed: r for r in map(Run, args.runs) if r.done}
    target = final(runs['AdamW'][0])

    names = ['LC5m', 'Muon', 'LC5', 'exact', 'AdamW']
    missing = {name: 'running' if name == 'LC5m' else '--' for name in names}
    loss = {n: {d: final(r) for d, r in runs[n].items()} for n in names}
    mins = {n: {d: first_step(r, 'val_loss', target) * sec[n] / 60 for d, r in runs[n].items()} for n in names}
    done = sorted(runs['LC5m'])

    def row(label, cells):
        return ' & '.join([label] + cells) + r' \\'

    lines = [row(f'd{d}', [f'{loss[n][d]:.4f}' if d in loss[n] else missing[n] for n in names]
                 + [('never' if np.isnan(mins[n][d]) else f'{mins[n][d]:.1f}') if d in mins[n] else missing[n]
                    for n in names]) for d in ORDERS]

    def mean(v, fmt):
        x = [v[d] for d in done if d in v]
        return '--' if not x or np.isnan(np.mean(x)) else format(np.mean(x), fmt)

    lines += [r'\midrule', row('mean', [mean(loss[n], '.4f') for n in names] + [mean(mins[n], '.1f') for n in names])]

    plot(runs, sec, done or [0], target)
    timing = ', '.join(f'{n} {1e3 * sec[n]:.0f}' for n in names[1:] + names[:1])
    timing += '' if args.pilot else ' (timed as LC5 until its own timing pilot has run)'
    orders = ', '.join(f'd{d}' for d in done) or 'none yet'
    fig_orders = (f'data order{"s" * (len(done) > 1)} {", ".join(str(d) for d in done)}' if done else
                  'data order 0 (no LC5m run has finished yet)')

    tex = rf"""% Written by code/scripts/lc5m_section.py (scripts/lc5m_section.sh); edits here are overwritten.
\clearpage
\paragraph{{LC5m: LC5 with a Muon floor (in progress).}} LC5m is LC5 (\cref{{tab:lc}}) with a hard constraint: every singular value of its update lies between Muon's value and the exact $s^{{-0.3}}$.

\begin{{table}}[H]
\centering
\small
\caption{{LC5m, one row per data order. GPU minutes until the validation loss first reaches {target:.4f}, as in \cref{{tab:negp,tab:lc}}; ms per step: {timing}. Exact: exact SVD, $p=-0.3$. running: not finished; --: not run. Mean: over the orders where LC5m has finished ({orders}).}}
\label{{tab:lc5m}}
\setlength{{\tabcolsep}}{{3.5pt}}
\begin{{tabular}}{{l ccccc ccccc}}
\toprule
 & \multicolumn{{5}}{{c}}{{final val.\ loss}} & \multicolumn{{5}}{{c}}{{GPU min.\ to {target:.4f}}} \\
\cmidrule(lr){{2-6}}\cmidrule(lr){{7-11}}
order & LC5m & Muon & LC5 & exact & AdamW & LC5m & Muon & LC5 & exact & AdamW \\
\midrule
""" + '\n'.join(lines) + rf"""
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth]{{{FIG}}}
\caption{{As \cref{{fig:gputime}}, for LC5m, LC5 and Muon on {fig_orders}.}}
\label{{fig:lc5m}}
\end{{figure}}
"""
    (RESULTS / 'lc5m_section.tex').write_text(tex)
    print(tex)


if __name__ == '__main__':
    main()
