"""Dose-response of Muon's update U S^p V^T (suites ts10m_mse_svp, ts10m_ce_svp): per run, the step at which the training
loss reaches a level, the stable rank and sigma_1 / sigma_median of the block matrices there, the step at which the
validation loss reaches a target (default: the AdamW run's final one) and the final validation loss; the loss curves,
one colour per p, with plain Muon and AdamW as references; with --paths, the distance of each p from plain Muon.

    python -m optsplit.analysis.dose [--suite ts10m_mse_svp] [--level 0.65] --out <dir>
    python -m optsplit.analysis.dose --suite ts10m_ce_svp --main ts10m_main --sweep ts10m_lrsweep --level 1.81 \
        --max-loss 4 --offset 1 --ticks 4 3 2.5 2 1.75 1.5 1.4 1.3 --zoom 5000 --paths --out <dir>

Needs the whole-model metrics of the runs (python -m optsplit.metrics.trajectory). Writes <dir>/dose.txt (every run),
<dir>/dose.json and <dir>/figures/dose_loss.pdf: training loss, one panel per learning rate; with --zoom, training and
validation loss at Muon's rate instead, over the whole run and from that step on (also as .png). --paths:
<dir>/figures/dose_paths.pdf, the Jensen-Shannon divergence of the predictions on the probe between each run (data order
0, Muon's rate) and plain Muon on the same data order, at equal step and at equal training loss, with the pairs of data
orders of plain Muon (and of a p run on several data orders) as the floor, as in Fig. 1 of the checkpoint; its curves
in <dir>/figures/dose_paths.json.
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from optsplit.analysis import ckpt_figs  # noqa: F401 (the figures' font sizes)
from optsplit.analysis.distance import distance, step_curve
from optsplit.analysis.io import Run, load_runs
from optsplit.analysis.plots import _loss_axis, panel_labels
from optsplit.analysis.rank_metrics import at_loss

REFS = {'muon': ('Muon (Newton–Schulz)', dict(color='k')), 'adamw': ('AdamW', dict(color='0.55'))}  # solid lines


def first_step(run, key, level):
    """First step at which the evaluated `key` (running minimum) reaches `level`, linear between evaluations."""
    steps, v = run.series('eval', key)
    return float(at_loss(np.c_[v, steps], [level])[0])


def row(run, level, target=np.nan):
    steps, loss = run.series('eval', 'train_loss')
    traj = [json.loads(f.read_text()) for f in sorted((run.path / 'traj_metrics').glob('*.json'))]
    at = lambda v: float(at_loss(np.c_[[t['train_loss'] for t in traj], v], [level])[0]) if len(traj) > 1 else np.nan
    return dict(run=run.id, p=run.cfg.optim.muon.get('sv_power'), lr=run.lr, data_seed=run.data_seed, done=run.done,
                step=first_step(run, 'train_loss', level), step_target=first_step(run, 'val_loss', target),
                stable_rank=at([t['stable_rank'] for t in traj]), sv_spread=at([t['sv_spread'] for t in traj]),
                final_step=int(steps[-1]), train=loss[-1], val=run.series('eval', 'val_loss')[1][-1])


def exact(run):
    """U S^p V^T by the SVD (not by matrix products)."""
    m = run.cfg.optim.muon
    return m.get('sv_power') is not None and not m.get('two_track_steps') and not m.get('lc_steps')


def color(p):
    """p < 0 yellow-orange to dark purple, p > 0 blue, darker further from 0 (down to -1, up to 0.75); p = 0 green.
    The negative values that were run are spread evenly over the colour map, so that neighbours can be told apart."""
    if p < 0:
        return plt.get_cmap('inferno')(np.interp(p, [-1, -0.75, -0.5, -0.3, -0.1], [0.2, 0.36, 0.52, 0.69, 0.86]))
    return '#008a17' if p == 0 else plt.get_cmap('Blues')(0.35 + 0.8 * p)


def log_axis(ax, offset, ticks):
    ax.set_yscale('function', functions=(lambda y: np.log(np.maximum(y - offset, 1e-9)), lambda v: np.exp(v) + offset))
    ax.set_yticks(ticks, labels=[f'{t:g}' for t in ticks])


def curves(by_lr, refs, level, out, max_loss, offset, ticks):
    """One panel per learning rate: training loss on the probe against the step, a thin line at `level`."""
    fig, axes = plt.subplots(1, len(by_lr), figsize=(7.1, 2.6), sharey=True, layout='constrained', squeeze=False)
    handles = {}
    for ax, (lr, runs) in zip(axes[0], by_lr.items()):
        for run in sorted(runs, key=lambda r: r.cfg.optim.muon.sv_power):
            p = run.cfg.optim.muon.sv_power
            handles[f'$p={p:g}$'], = ax.plot(*run.series('eval', 'train_loss'), lw=0.9, color=color(p))
        for name, run in refs[lr].items():
            handles[REFS[name][0]], = ax.plot(*run.series('eval', 'train_loss'), lw=1, **REFS[name][1])
        ax.axhline(level, color='0.55', lw=0.4)
        log_axis(ax, offset, ticks)
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


def curves_zoom(runs, refs, level, out, max_loss, offset, ticks, zoom):
    """Training loss (top) and validation loss (bottom) on the probe against the step, at one learning rate: the whole
    run (left, loss axis on log(L - offset)) and from step `zoom` on (right, linear, up to AdamW's loss: worse runs leave
    the panel)."""
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.6), layout='constrained')
    handles = {}
    lines = [(f'$p={r.cfg.optim.muon.sv_power:g}$', r, dict(lw=0.9, color=color(r.cfg.optim.muon.sv_power)))
             for r in sorted(runs, key=lambda r: r.cfg.optim.muon.sv_power)]
    lines += [(REFS[name][0], run, dict(lw=1, **REFS[name][1])) for name, run in refs.items()]
    for (full, late), key, name in zip(axes, ('train_loss', 'val_loss'), ('training', 'validation')):
        for label, run, style in lines:
            steps, v = run.series('eval', key)
            handles[label], = full.plot(steps, v, **style)
            late.plot(steps[steps >= zoom], v[steps >= zoom], **style)
            if label == REFS['adamw'][0]:
                hi = v[steps >= zoom].max()
        if key == 'train_loss':
            full.axhline(level, color='0.55', lw=0.4)
        log_axis(full, offset, ticks)
        full.set_ylim(min(np.nanmin(line.get_ydata()) for line in full.lines) - 0.003, max_loss)
        full.set_xlim(0, None)
        late.set_xlim(zoom, None)
        late.set_ylim(min(np.min(line.get_ydata()) for line in late.lines if len(line.get_ydata())) - 0.005, hi + 0.005)
        for ax in (full, late):
            ax.set_xlabel('step')
            ax.set_ylabel(f'{name} loss')
    axes[0, 0].set_title('whole run')
    axes[0, 1].set_title(f'from step {zoom}')
    fig.legend(handles.values(), handles.keys(), loc='outside lower center', ncol=6, frameon=False)
    panel_labels(axes.ravel())
    for path in (out, out.with_suffix('.png')):
        fig.savefig(path, dpi=300)
        print(f'wrote {path}')


def loss_fit(run, h=0.15, first=10):
    """Smooth training loss: a function of log step giving log loss, by local linear regression on the evaluations
    from step `first` on, with Gaussian weights of width `h` in log step (h = 0.15: about +-15% of the step)."""
    steps, loss = run.series('eval', 'train_loss')
    keep = steps >= first
    x, y = np.log(steps[keep]), np.log(loss[keep])

    def f(x0):
        x0 = np.atleast_1d(x0)
        w = np.exp(-0.5 * ((x[None, :] - x0[:, None]) / h) ** 2)
        sw, swx, swy = w.sum(1), w @ x, w @ y
        swxx, swxy = w @ x ** 2, w @ (x * y)
        slope = (sw * swxy - swx * swy) / (sw * swxx - swx ** 2)
        return (swy - slope * swx) / sw + slope * x0
    return f


def equal_loss(a, b, first=10, every=250):
    """JS between each prediction step s of `a` and the closer (smaller JS) of b's two prediction steps around the step
    t where b's fitted loss equals a's (loss_fit). From step `every` on, both runs use only the steps on the regular
    grid (multiples of `every`). Two runs on one data order are much closer at the same step than one step apart
    (they have just seen the same batches), so for runs about as fast as Muon the nearest step, or the geometric
    steps between grid points, make the curve jump between the two levels."""
    grid_steps = lambda r: np.array([s for s in r.pred_steps if s >= first and (s < every or s % every == 0)])
    fa, fb = loss_fit(a, first=first), loss_fit(b, first=first)
    sa, sb = grid_steps(a), grid_steps(b)
    dense = np.geomspace(first, sb[-1], 4000)
    lb = np.minimum.accumulate(fb(np.log(dense)))  # decreasing
    la = fa(np.log(sa))
    ok = (la <= lb[0]) & (la >= lb[-1])
    lev, idx = np.unique(lb, return_index=True)
    t = np.interp(la[ok], lev, dense[idx])
    hi = np.clip(np.searchsorted(sb, t), 1, len(sb) - 1)
    js = [min(distance(a, s, b, sb[i - 1])[0], distance(a, s, b, sb[i])[0]) for s, i in zip(sa[ok], hi)]
    return dict(loss=np.exp(la[ok]), js=np.array(js))


def paths(runs, muon, adamw, floors, out, max_loss, offset, ticks):
    """JS between each run and plain Muon on its data order, at equal step (a) and at equal training loss (b, see
    equal_loss); grey: two data orders of one optimizer (`floors`: name -> ({data order: run}, colour)); dotted: AdamW
    against Muon. The curves also go to `out` with suffix .json."""
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), sharey=True, layout='constrained')
    handles, data = {}, []
    for ax, (curve, x) in zip(axes, ((step_curve, 'step_a'), (equal_loss, 'loss'))):
        def draw(a, b, label, **style):
            c = curve(a, b)
            ok = c['loss'] <= max_loss if x == 'loss' else np.ones(len(c['loss']), bool)
            handles[label], = ax.plot(c[x][ok], c['js'][ok], **style)
            data.append(dict(x=x, label=label, a=a.id, b=b.id, at=np.asarray(c[x])[ok].tolist(),
                             js=np.asarray(c['js'])[ok].tolist()))

        for name, (group, grey) in floors.items():
            for s, t in combinations(sorted(group), 2):
                draw(group[s], group[t], f'{name}, two data orders', color=grey, lw=0.6)
        draw(adamw, muon[adamw.data_seed], f'{REFS["adamw"][0]} vs Muon', lw=1, **REFS['adamw'][1])
        for run in sorted(runs, key=lambda r: r.cfg.optim.muon.sv_power):
            p = run.cfg.optim.muon.sv_power
            draw(run, muon[run.data_seed], f'$p={p:g}$', color=color(p), lw=0.9)
    axes[0].set(xlabel='step', ylabel='JS to Muon (bits)', xlim=(0, None))
    _loss_axis(axes[1], max_loss, offset, ticks)
    fig.legend(handles.values(), handles.keys(), loc='outside lower center', ncol=6, frameon=False)
    panel_labels(axes)
    fig.savefig(out)
    out.with_suffix('.json').write_text(json.dumps(data))
    print(f'wrote {out}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--suite', default='ts10m_mse_svp')
    ap.add_argument('--main', default='ts10m_mse_main', help='suite with plain muon_d* and adamw_d0')
    ap.add_argument('--sweep', default='ts10m_mse_lrsweep', help='suite with plain Muon at the other learning rates')
    ap.add_argument('--level', type=float, default=0.65)
    ap.add_argument('--target', type=float, help='validation loss to reach (default: the AdamW run\'s final one)')
    ap.add_argument('--max-loss', type=float, default=0.9)
    ap.add_argument('--offset', type=float, default=0.5, help='loss axis on log(L - offset)')
    ap.add_argument('--ticks', type=float, nargs='+', default=[0.9, 0.8, 0.7, 0.65, 0.6, 0.57])
    ap.add_argument('--zoom', type=int, help='loss figure at Muon\'s rate only: training and validation loss, the whole '
                    'run and from this step on')
    ap.add_argument('--paths', action='store_true', help='also the distance of each p from plain Muon (dose_paths.pdf)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out)
    (out / 'figures').mkdir(parents=True, exist_ok=True)

    started = lambda pats: [r for r in load_runs(pats) if r.records_of('eval')]
    runs = [r for r in started(f'{args.suite}/*_d0') if exact(r)]
    main_lr = Run(f'{args.main}/muon_d0').lr
    by_lr = {lr: [r for r in runs if r.lr == lr] for lr in sorted({r.lr for r in runs}, key=lambda x: x != main_lr)}
    refs = {}
    adamw = Run(f'{args.main}/adamw_d0')
    for lr in by_lr:  # plain Muon at this rate (the main run, or the sweep's), and AdamW
        muon = [r for r in [Run(f'{args.main}/muon_d0')] + load_runs(f'{args.sweep}/muon_*') if r.lr == lr]
        refs[lr] = {'muon': muon[0], 'adamw': adamw} if muon else {'adamw': adamw}
    target = args.target or adamw.series('eval', 'val_loss')[1][-1]

    rows = [row(r, args.level, target) for lr in by_lr
            for r in sorted(by_lr[lr], key=lambda r: r.cfg.optim.muon.sv_power)]
    rows += [row(r, args.level, target) for r in {r.id: r for lr in refs for r in refs[lr].values()}.values()]
    listed = {r['run'] for r in rows}  # then every other run: other data orders, matrix-product methods, checks
    rows += [row(r, args.level, target) for r in started([f'{args.suite}/*', f'{args.main}/muon_d[1-9]'])
             if r.id not in listed]
    lines = [f'{"run":<44} {"lr":>7} {"p":>6} {"step@" + format(args.level, "g"):>9} {"st.rank":>8} {"s1/smed":>8} '
             f'{"val@" + format(target, ".4f"):>11} {"last":>6} {"train":>7} {"val":>7}']
    for r in rows:
        lines.append(f'{r["run"]:<44} {r["lr"]:>7g} {str(r["p"]):>6} {r["step"]:>9.0f} {r["stable_rank"]:>8.1f} '
                     f'{r["sv_spread"]:>8.2f} {r["step_target"]:>11.0f} {r["final_step"]:>6} {r["train"]:>7.4f} '
                     f'{r["val"]:>7.4f}' + ('' if r['done'] else '  (not done)'))
    (out / 'dose.txt').write_text('\n'.join(lines) + '\n')
    (out / 'dose.json').write_text(json.dumps(rows, indent=1))
    print('\n'.join(lines))
    if args.zoom:
        curves_zoom(by_lr[main_lr], refs[main_lr], args.level, out / 'figures' / 'dose_loss.pdf', args.max_loss,
                    args.offset, args.ticks, args.zoom)
    else:
        curves(by_lr, refs, args.level, out / 'figures' / 'dose_loss.pdf', args.max_loss, args.offset, args.ticks)
    if args.paths:
        muon = {r.data_seed: r for r in load_runs(f'{args.main}/muon_d*')}
        floors = {'Muon': (muon, '0.7')}
        for p in sorted({r.cfg.optim.muon.sv_power for r in by_lr[main_lr]}):  # p runs on several data orders
            group = {r.data_seed: r for r in load_runs(f'{args.suite}/*_d*') if exact(r) and r.lr == main_lr
                     and r.cfg.optim.muon.sv_power == p and r.records_of('eval')}
            if len(group) > 1:
                floors[f'$p={p:g}$'] = (group, '0.35')
        paths(by_lr[main_lr], muon, adamw, floors, out / 'figures' / 'dose_paths.pdf', args.max_loss, args.offset,
              args.ticks)


if __name__ == '__main__':
    main()
