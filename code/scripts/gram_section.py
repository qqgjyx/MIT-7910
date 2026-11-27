"""Fig. 2 and the last section (Gram LC) of results/dose_negative_p_ce.tex, both overwritten:

- results/ts10m/lc_loss_ce.pdf: mean over data orders 0-2 of the training and validation loss against the step, and of
  the validation loss against GPU minutes (step x time per step, as gpu_time.py), of LC5-LC12, LC12 at equal GPU time,
  Muon, AdamW (GPU minutes only) and the new methods (NEW: their finished runs among orders 0-2; against GPU minutes once
  timed); whole run and end of training;
- results/gram_section.tex (which the document inputs): one row per data order with the final validation loss and the
  GPU minutes to AdamW's final loss of Gram LC, LC5m (if any of its runs started), Muon, LC5, the exact update and AdamW.

The new methods' runs are found by name in any suite matching NEW. Their time per step is Muon's (ts10m_ce_tt_pilot)
times their ratio to Muon in a joint timing job on an L40S; until there is one, their GPU time is "pending". Run by
scripts/gram_section.sh, which first copies the logs from ORCD (--files lists them):

    python scripts/gram_section.py
"""
import argparse
from functools import reduce
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from optsplit.analysis.dose import REFS, first_step, log_axis
from optsplit.analysis.gpu_time import PILOTS
from optsplit.analysis.io import Run, load_runs
from optsplit.analysis.plots import panel_labels
from optsplit.analysis.two_track import METHODS, SUITES, step_time

RESULTS = Path(__file__).resolve().parents[2] / 'results'
FIG = 'ts10m/lc_loss_ce.pdf'
ORDERS = range(5)
MEAN_ORDERS = (0, 1, 2)  # Fig. 2: the data orders of LC5-LC12
SU = SUITES['ce']
MUON = REFS['muon'][0]
# Solid lines, one colour per method (checked with a colour-blindness validator)
COLORS = {'LC5': '#6a4ce2', 'LC6': '#00b894', 'LC8': '#e377c2', 'LC12': '#7a4a1e', 'LC12, equal GPU time': '#008a17',
          MUON: REFS['muon'][1]['color'], 'AdamW': REFS['adamw'][1]['color'], 'Gram LC': '#ec4647', 'LC5m': '#f2a600'}
# label: (run id with {d}, timing pilot)
RUNS = {label: (run, pilot.format(**SU)) for label, run, pilot, *_ in METHODS if label in COLORS}
STEPWISE = list(RUNS)  # drawn against the step: LC and Muon
RUNS |= {'AdamW': ('ts10m_main/adamw_d{d}', PILOTS['adamw']), 'exact': ('{svp}/muon_p-0.3_lr{lr}_d{d}', PILOTS['svd'])}
# new methods: run glob (data order from the config), timing pilot, Muon's pilot in the same job
NEW = {'Gram LC': ('ts10m_ce_gram*/*gram_d[0-4]', 'ts10m_ce_gram_pilot/gram_fast', 'ts10m_ce_gram_pilot/muon_fast'),
       'LC5m': ('ts10m_ce_lcm*/*lc5m_d[0-4]', 'ts10m_ce_lcm_pilot/lc5m', 'ts10m_ce_lcm_pilot/muon')}
SHORT = {'Gram LC': 'Gram', 'LC5m': 'LC5m', MUON: 'Muon', 'LC5': 'LC5', 'exact': 'exact', 'AdamW': 'AdamW'}


def final(r):
    return r.series('eval', 'val_loss')[1][-1]


def pilot_time(pilot):
    return float(np.mean([step_time(Run(f'{pilot}_d{d}')) for d in (0, 1)]))


def started(pattern):
    """{data order: run} of the runs matching `pattern` that have started (not timing pilots), finished ones first."""
    out = {}
    for r in load_runs(pattern):
        if 'pilot' not in r.id and r.records_of('eval') and (r.done or r.data_seed not in out):
            out[r.data_seed] = r
    return out


def mean_curve(rs, key):
    """The steps the runs share and the mean of `key` over the runs there."""
    series = [r.series('eval', key) for r in rs]
    common = reduce(np.intersect1d, [steps for steps, _ in series])
    return common, np.mean([v[np.isin(steps, common)] for steps, v in series], axis=0)


def figure(runs, sec, target, end_step=7000, end_minutes=15, ticks=(4, 3, 2, 1.75, 1.5, 1.4, 1.3)):
    """Mean over data orders 0-2 (the new methods: those finished). Rows: training and validation loss against the
    step, validation loss against GPU minutes (also AdamW; the new methods once timed). Left: whole run, on log(L - 1);
    right: from `end_step` or `end_minutes` on, linear, fitted to the curves there."""
    fig, axes = plt.subplots(3, 2, figsize=(7.1, 6.0), layout='constrained')
    handles = {}
    for (full, end), key, by_time in zip(axes, ('train_loss', 'val_loss', 'val_loss'), (False, False, True)):
        start = end_minutes if by_time else end_step
        lo, hi = np.inf, -np.inf
        for name in STEPWISE + list(NEW) + ['AdamW'] * by_time:
            rs = [r for d, r in runs[name].items() if d in MEAN_ORDERS and r.done]
            if not rs or (by_time and np.isnan(sec[name])):
                continue
            steps, v = mean_curve(rs, key)
            x = steps * sec[name] / 60 if by_time else steps
            label = name if len(rs) == len(MEAN_ORDERS) else f'{name} (mean of {len(rs)})'
            handles[label], = full.plot(x, v, color=COLORS[name], lw=1)
            end.plot(x[x >= start], v[x >= start], color=COLORS[name], lw=1)
            if (x >= start).any():
                lo, hi = min(lo, v[x >= start].min()), max(hi, v[x >= start].max())
        if by_time:
            for ax in (full, end):
                ax.axhline(target, color='0.55', lw=0.4)
        log_axis(full, 1, ticks)
        full.set(xlim=(0, None), ylim=(min(np.nanmin(line.get_ydata()) for line in full.lines) - 0.003, ticks[0]))
        end.set(xlim=(start, None), ylim=(lo - 0.002, hi + 0.002))
        for ax in (full, end):
            ax.set(xlabel='GPU minutes' if by_time else 'step',
                   ylabel='training loss' if key == 'train_loss' else 'validation loss')
    axes[0, 0].set_title('whole run')
    axes[0, 1].set_title('end of training')
    fig.legend(handles.values(), handles.keys(), loc='outside lower center', ncol=4, frameon=False)
    panel_labels(axes.ravel())
    fig.savefig(RESULTS / FIG)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--files', action='store_true', help='only print the files (globs) to copy from ORCD')
    args = ap.parse_args()
    if args.files:
        ids = [r.format(d=d, **SU) for r, _ in RUNS.values() for d in ORDERS] + [g for g, _, _ in NEW.values()]
        ids += [f'{p}_d{d}' for p in {p for _, p in RUNS.values()} | {p for _, *ps in NEW.values() for p in ps}
                for d in (0, 1)]
        print(' '.join(f'{i}/{f}' for i in ids for f in ('config.yaml', 'metrics.jsonl', 'done')))
        return

    runs = {name: {d: r for d in ORDERS if (r := Run(p.format(d=d, **SU))).done} for name, (p, _) in RUNS.items()}
    sec = {name: pilot_time(p) for name, (_, p) in RUNS.items()}
    for name, (pattern, pilot, muon) in NEW.items():
        runs[name] = started(pattern)
        # an L40S job: Muon there within 20% of its L40S time (ts10m_ce_gram_pilot ran elsewhere: muon_fast 47 ms)
        timed = (all(Run(f'{p}_d{d}').done for p in (pilot, muon) for d in (0, 1))
                 and abs(pilot_time(muon) / sec[MUON] - 1) < 0.2)
        sec[name] = sec[MUON] * pilot_time(pilot) / pilot_time(muon) if timed else np.nan
    target = final(runs['AdamW'][0])
    figure(runs, sec, target)

    names = ['Gram LC'] + ['LC5m'] * bool(runs['LC5m']) + [MUON, 'LC5', 'exact', 'AdamW']
    done = [d for d, r in sorted(runs['Gram LC'].items()) if r.done]

    def minutes(rs, n):
        m = np.mean([first_step(r, 'val_loss', target) * sec[n] / 60 for r in rs])
        return 'pending' if np.isnan(sec[n]) else 'never' if np.isnan(m) else f'{m:.1f}'

    def cells(n, d):
        r = runs[n].get(d)
        if r is None or not r.done:
            state = 'running' if r else 'queued' if n in NEW else '--'
            return state, state
        return f'{final(r):.4f}', minutes([r], n)

    def mean_cells(n):
        rs = [runs[n][d] for d in done if d in runs[n] and runs[n][d].done]
        return (f'{np.mean([final(r) for r in rs]):.4f}', minutes(rs, n).replace('never', '--')) if rs else ('--', '--')

    def row(label, pairs):
        return ' & '.join([label] + [a for a, _ in pairs] + [b for _, b in pairs]) + r' \\'

    lines = [row(f'd{d}', [cells(n, d) for n in names]) for d in ORDERS]
    lines += [r'\midrule', row('mean', [mean_cells(n) for n in names])]
    k = len(names)
    text = (r" Its time per step is Muon's times the ratio of the two in a joint timing job on an L40S."
            if not np.isnan(sec['Gram LC']) else ' Its GPU time is pending a timing job on an L40S.')
    text += r' \Cref{fig:lc} shows its mean curves.' if any(r.done for d, r in runs['Gram LC'].items() if d < 3) else ''
    if runs['LC5m']:
        text += (r" LC5m, an earlier variant (LC5 with every singular value of its update between Muon's and "
                 r"$s^{-0.3}$), was replaced by Gram LC; its runs that had started are shown too.")
    orders = ', '.join(f'd{d}' for d in done) or 'none yet'

    tex = rf"""% Written by code/scripts/gram_section.py (scripts/gram_section.sh); edits here are overwritten.
\paragraph{{Gram LC: an update from the Gram matrix (in progress).}} Gram LC iterates on the Gram matrix $A=XX^\top$ of the normalized momentum $X$, one matrix product per step (each doubling the polynomial degree), sums $S=w_I I+\sum_k w_k A_k$ over the iterates $A_k$ and updates by $SX$. Its weights are fitted so that every singular value in $(0,1]$ lands between Muon's and $s^{{-0.3}}$; in float32 it costs about 0.84 times Muon's FLOPs.{text}

\begin{{table}}[H]
\centering
\small
\caption{{Gram LC against the other methods, one row per data order; exact: exact SVD at $p=-0.3$. running, queued: not finished, not started; pending: no timing yet; --: not run; mean: over the orders Gram LC has finished ({orders}).}}
\label{{tab:gram}}
\setlength{{\tabcolsep}}{{3.5pt}}
\begin{{tabular}}{{l {'c' * k} {'c' * k}}}
\toprule
 & \multicolumn{{{k}}}{{c}}{{final val.\ loss}} & \multicolumn{{{k}}}{{c}}{{GPU min.\ to {target:.4f}}} \\
\cmidrule(lr){{2-{k + 1}}}\cmidrule(lr){{{k + 2}-{2 * k + 1}}}
order & {' & '.join(SHORT[n] for n in names)} & {' & '.join(SHORT[n] for n in names)} \\
\midrule
""" + '\n'.join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (RESULTS / 'gram_section.tex').write_text(tex)
    print(tex)


if __name__ == '__main__':
    main()
