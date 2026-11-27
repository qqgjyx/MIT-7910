"""Fig. 2 and the last section (LC5m) of results/dose_negative_p_ce.tex, both overwritten:

- results/ts10m/lc_val_ce.pdf: validation loss of LC5-LC12, LC12 at equal GPU time and Muon against the step (data
  orders 0-2, as two_track.py), and against GPU minutes (step x time per step from the timing pilots, as gpu_time.py)
  on data order 0 with AdamW and LC5m (every finished data order);
- results/lc5m_section.tex (which the document inputs): one row per data order with the final validation loss and the
  GPU minutes to AdamW's final loss of LC5m, Muon, LC5, the exact update and AdamW.

Run by scripts/lc5m_section.sh, which first copies the logs from ORCD (--files lists them):

    python scripts/lc5m_section.py [--pilot <suite>/<name without _d0>] [<LC5m run id> ...]

Defaults: the runs of ts10m_ce_lcm and the pilot ts10m_ce_lcm_pilot/lc5m; LC5m is timed as LC5 until that pilot is done.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from optsplit.analysis.dose import REFS, first_step, log_axis
from optsplit.analysis.gpu_time import PILOTS
from optsplit.analysis.io import Run
from optsplit.analysis.plots import panel_labels
from optsplit.analysis.two_track import METHODS, SUITES, step_time

RESULTS = Path(__file__).resolve().parents[2] / 'results'
FIG = 'ts10m/lc_val_ce.pdf'
ORDERS = range(5)
SU = SUITES['ce']
MUON = REFS['muon'][0]
# Solid lines, one colour per method (checked with a colour-blindness validator)
COLORS = {'LC5': '#6a4ce2', 'LC6': '#00b894', 'LC8': '#ec4647', 'LC12': '#7a4a1e', 'LC12, equal GPU time': '#008a17',
          MUON: REFS['muon'][1]['color'], 'AdamW': REFS['adamw'][1]['color'], 'LC5m': '#f2a600'}
# label: (run id with {d}, timing pilot)
RUNS = {label: (run, pilot.format(**SU)) for label, run, pilot, *_ in METHODS if label in COLORS}
STEPWISE = list(RUNS)  # drawn against the step: LC and Muon
RUNS |= {'AdamW': ('ts10m_main/adamw_d{d}', PILOTS['adamw']), 'exact': ('{svp}/muon_p-0.3_lr{lr}_d{d}', PILOTS['svd'])}


def final(r):
    return r.series('eval', 'val_loss')[1][-1]


def width(d):
    return dict(lw=1.0) if d == 0 else dict(lw=0.5, alpha=0.6)


def figure(runs, sec, target, zoom=5000, minutes=(10, 27), ticks=(2, 1.75, 1.5, 1.4, 1.35, 1.3)):
    """(a) whole run and (b) from step `zoom` against the step; (c) the GPU `minutes`, linear, capped at AdamW's loss
    at their start."""
    fig, ax = plt.subplot_mosaic([['a', 'b'], ['c', 'legend']], figsize=(7.1, 4.6), layout='constrained')
    handles, lo = {}, np.inf
    for name in STEPWISE:
        for d, r in runs[name].items():
            if d < 3:
                steps, val = r.series('eval', 'val_loss')
                line, = ax['a'].plot(steps, val, color=COLORS[name], **width(d))
                ax['b'].plot(steps[steps >= zoom], val[steps >= zoom], color=COLORS[name], **width(d))
                handles[name] = handles.get(name, line) if d else line
    for name in STEPWISE + ['AdamW', 'LC5m']:
        for d, r in runs[name].items():
            if d == 0 or name == 'LC5m':
                steps, val = r.series('eval', 'val_loss')
                t = steps * sec[name] / 60
                line, = ax['c'].plot(t, val, color=COLORS[name], **width(d))
                handles.setdefault(name, line)
                win = (t >= minutes[0]) & (t <= minutes[1])
                lo = min(lo, val[win].min()) if win.any() else lo
                if name == 'AdamW':
                    hi = val[t >= minutes[0]].max()
    log_axis(ax['a'], 1, ticks)
    ax['a'].set(xlim=(0, None), ylim=(min(np.nanmin(line.get_ydata()) for line in ax['a'].lines) - 0.003, ticks[0]),
                title='whole run')
    ax['b'].set(xlim=(zoom, None), title=f'from step {zoom}')
    ax['c'].axhline(target, color='0.55', lw=0.4)
    ax['c'].set(xlim=minutes, ylim=(lo - 0.005, hi + 0.005), title=f'{minutes[0]:g} to {minutes[1]:g} GPU minutes',
                xlabel='GPU minutes')
    for k in 'abc':
        ax[k].set_ylabel('validation loss')
    for k in 'ab':
        ax[k].set_xlabel('step')
    ax['legend'].axis('off')
    ax['legend'].legend(handles.values(), handles.keys(), loc='center', ncol=2, frameon=False)
    panel_labels([ax[k] for k in 'abc'])
    fig.savefig(RESULTS / FIG)


def pilot_time(pilot):
    return float(np.mean([step_time(Run(f'{pilot}_d{d}')) for d in (0, 1)]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('runs', nargs='*', default=[f'ts10m_ce_lcm/muon_p-0.3_lc5m_d{d}' for d in ORDERS],
                    help='LC5m run ids (<suite>/<name>), finished or not')
    ap.add_argument('--pilot', default='ts10m_ce_lcm_pilot/lc5m', help='LC5m timing pilot (runs <pilot>_d0 and _d1)')
    ap.add_argument('--files', action='store_true', help='only print the files to copy from ORCD')
    args = ap.parse_args()
    if args.files:
        ids = [r.format(d=d, **SU) for r, _ in RUNS.values() for d in ORDERS] + args.runs
        ids += [f'{p}_d{d}' for p in {p for _, p in RUNS.values()} | {args.pilot} for d in (0, 1)]
        print(' '.join(f'{i}/{f}' for i in ids for f in ('config.yaml', 'metrics.jsonl', 'done')))
        return

    runs = {name: {d: r for d in ORDERS if (r := Run(p.format(d=d, **SU))).done} for name, (p, _) in RUNS.items()}
    runs['LC5m'] = {r.data_seed: r for r in map(Run, args.runs) if r.done}
    sec = {name: pilot_time(p) for name, (_, p) in RUNS.items()}
    timed = all(Run(f'{args.pilot}_d{d}').done for d in (0, 1))
    sec['LC5m'] = pilot_time(args.pilot) if timed else sec['LC5']
    target = final(runs['AdamW'][0])
    figure(runs, sec, target)

    names = ['LC5m', MUON, 'LC5', 'exact', 'AdamW']
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
    text = r' \Cref{fig:lc}c shows its validation loss against GPU time.' if done else ''
    text += '' if timed else " Until its own timing job has run, its GPU time uses LC5's time per step."
    orders = ', '.join(f'd{d}' for d in done) or 'none yet'

    tex = rf"""% Written by code/scripts/lc5m_section.py (scripts/lc5m_section.sh); edits here are overwritten.
\paragraph{{LC5m: LC5 with a Muon floor (in progress).}} LC5m is LC5 with a hard constraint: every singular value of its update lies between Muon's value and the exact $s^{{-0.3}}$.{text}

\begin{{table}}[H]
\centering
\small
\caption{{LC5m against the other methods, one row per data order; exact: exact SVD at $p=-0.3$. running: not finished; --: not run; mean: over the orders LC5m has finished ({orders}).}}
\label{{tab:lc5m}}
\setlength{{\tabcolsep}}{{3.5pt}}
\begin{{tabular}}{{l ccccc ccccc}}
\toprule
 & \multicolumn{{5}}{{c}}{{final val.\ loss}} & \multicolumn{{5}}{{c}}{{GPU min.\ to {target:.4f}}} \\
\cmidrule(lr){{2-6}}\cmidrule(lr){{7-11}}
order & LC5m & Muon & LC5 & exact & AdamW & LC5m & Muon & LC5 & exact & AdamW \\
\midrule
""" + '\n'.join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (RESULTS / 'lc5m_section.tex').write_text(tex)
    print(tex)


if __name__ == '__main__':
    main()
