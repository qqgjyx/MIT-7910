"""Validation loss against GPU time, and the GPU minutes until the validation loss first reaches a target (default:
AdamW's final one), for the cross-entropy runs of results/dose_negative_p_ce: the exact U S^p V^T sweep (ts10m_ce_svp),
LC (ts10m_ce_tt), plain Muon and AdamW (ts10m_main, ts10m_lrsweep).

GPU time = step x the method's time per step, measured in a timing pilot (scripts/two_track_pilot.sh: two runs side by
side on one L40S, without the measurements of the long runs): ts10m_ce_tt_pilot for Muon, LC and the exact SVD (timed
at p = -0.3 and used for every p: the SVD dominates the step), ts10m_ce_adamw_pilot for AdamW. Not the logged
wall-clock: the long runs shared GPUs and some of them measured the Hessian.

    python -m optsplit.analysis.gpu_time --out <dir>

Writes <dir>/gpu_time_ce.txt and .json (every run) and <dir>/figures/gpu_time_val_ce.pdf (rate 0.01, data order 0).
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from optsplit.analysis.dose import REFS, color, first_step, log_axis
from optsplit.analysis.io import Run, load_runs
from optsplit.analysis.plots import panel_labels
from optsplit.analysis.two_track import METHODS, step_time

PILOTS = {'muon': 'ts10m_ce_tt_pilot/muon', 'svd': 'ts10m_ce_tt_pilot/svd_p-0.3', 'adamw': 'ts10m_ce_adamw_pilot/adamw',
          **{f'lc{k}': f'ts10m_ce_tt_pilot/lc{k}' for k in (5, 6, 8, 12)}}
STYLE = {label: style for label, *_, style in METHODS}


def method(run):
    """The pilot that times this run."""
    m = run.cfg.optim.muon
    if run.optimizer == 'adamw':
        return 'adamw'
    if m.get('lc_steps'):
        return f'lc{m.lc_steps}'
    return 'svd' if m.get('sv_power') is not None else 'muon'


def drawn(by_id):
    """Rate 0.01, data order 0: (label, run, line style)."""
    out = [(f'$p={r.cfg.optim.muon.sv_power:g}$', r, dict(lw=0.9, color=color(r.cfg.optim.muon.sv_power)))
           for r in sorted((r for i, r in by_id.items() if i.startswith('ts10m_ce_svp/') and i.endswith('_lr0.01_d0')),
                           key=lambda r: r.cfg.optim.muon.sv_power)]
    out += [(REFS[o][0], by_id[f'ts10m_main/{o}_d0'], dict(lw=1, **REFS[o][1])) for o in ('muon', 'adamw')]
    out += [(f'LC{k}', by_id[f'ts10m_ce_tt/muon_p-0.3_lc{k}_d0'], dict(lw=1, **STYLE[f'LC{k}'])) for k in (5, 6, 8, 12)]
    out.append(('LC12, equal GPU time', by_id['ts10m_ce_tt/muon_p-0.3_lc12_eqtime_d0'],
                dict(lw=1, **STYLE['LC12, equal GPU time'])))
    return out


def plot(lines, sec, target, out, offset, ticks, zoom):
    """Validation loss against GPU minutes: the whole run (loss axis on log(L - offset)) and the minutes in `zoom`
    (linear, up to AdamW's loss at the start of the window: worse runs leave the panel)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), layout='constrained')
    handles, lo = {}, np.inf
    for label, run, style in lines:
        steps, val = run.series('eval', 'val_loss')
        minutes = steps * sec[method(run)] / 60
        handles[label], = axes[0].plot(minutes, val, **style)
        axes[1].plot(minutes, val, **style)
        win = (minutes >= zoom[0]) & (minutes <= zoom[1])
        lo = min(lo, val[win].min()) if win.any() else lo
        if label == REFS['adamw'][0]:
            hi = val[minutes >= zoom[0]].max()
    for ax in axes:
        ax.axhline(target, color='k', lw=0.4)
        ax.set(xlabel='GPU minutes', ylabel='validation loss')
    log_axis(axes[0], offset, ticks)
    axes[0].set_ylim(min(np.nanmin(line.get_ydata()) for line in axes[0].lines) - 0.003, ticks[0])
    axes[0].set_xlim(0, None)
    axes[0].set_title('whole run')
    axes[1].set(xlim=zoom, ylim=(lo - 0.005, hi + 0.005))
    axes[1].set_title(f'{zoom[0]:g} to {zoom[1]:g} minutes')
    fig.legend(handles.values(), handles.keys(), loc='outside lower center', ncol=6, frameon=False)
    panel_labels(axes)
    fig.savefig(out)
    print(f'wrote {out}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--target', type=float, help='validation loss to reach (default: AdamW\'s final one)')
    ap.add_argument('--offset', type=float, default=1)
    ap.add_argument('--ticks', type=float, nargs='+', default=[2, 1.75, 1.5, 1.4, 1.35, 1.3])
    ap.add_argument('--zoom', type=float, nargs=2, default=[10, 27])
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out)
    (out / 'figures').mkdir(parents=True, exist_ok=True)

    sec = {m: float(np.mean([step_time(Run(f'{p}_d{d}')) for d in (0, 1)])) for m, p in PILOTS.items()}
    runs = [r for r in load_runs(['ts10m_ce_svp/*', 'ts10m_ce_tt/*', 'ts10m_main/muon_d*', 'ts10m_main/adamw_d0',
                                  'ts10m_lrsweep/muon_lr0.005']) if r.done]
    by_id = {r.id: r for r in runs}
    target = args.target or by_id['ts10m_main/adamw_d0'].series('eval', 'val_loss')[1][-1]
    rows = []
    for r in runs:
        step = first_step(r, 'val_loss', target)
        rows.append(dict(run=r.id, method=method(r), ms_per_step=1e3 * sec[method(r)], step_target=step,
                         min_target=step * sec[method(r)] / 60, min_total=r.series('eval', 'val_loss')[0][-1]
                         * sec[method(r)] / 60, val=float(r.series('eval', 'val_loss')[1][-1])))
    lines = [' '.join(f'{m} {1e3 * s:.1f}' for m, s in sec.items()) + ' (ms per step)',
             f'{"run":<44} {"ms/step":>7} {"step@" + format(target, ".4f"):>11} {"min@" + format(target, ".4f"):>10} '
             f'{"min total":>9} {"val":>7}']
    for x in rows:
        lines.append(f'{x["run"]:<44} {x["ms_per_step"]:>7.1f} {x["step_target"]:>11.0f} '
                     f'{format(x["min_target"], ".1f").replace("nan", "never"):>10} {x["min_total"]:>9.1f} {x["val"]:>7.4f}')
    (out / 'gpu_time_ce.txt').write_text('\n'.join(lines) + '\n')
    (out / 'gpu_time_ce.json').write_text(json.dumps(dict(ms_per_step={m: 1e3 * s for m, s in sec.items()}, runs=rows),
                                                      indent=1))
    print('\n'.join(lines))
    plot(drawn(by_id), sec, target, out / 'figures' / 'gpu_time_val_ce.pdf', args.offset, args.ticks, args.zoom)


if __name__ == '__main__':
    main()
