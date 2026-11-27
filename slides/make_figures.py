"""Every figure of slides/deck.html (slides/img/*.svg), drawn from the small data files in slides/data:

    uv run --no-project --with matplotlib,numpy python slides/make_figures.py

Each data file names its source. These options rebuild them first, from the project's outputs:

    --runs DIR           data/runs.json from the run logs DIR/<suite>/<run>/metrics.jsonl (ORCD's runs/, or the copy
                         that code/scripts/gram_section.sh makes)
    --metrics FILE       data/metrics.json from rank_metrics.py's metrics.json (ORCD analysis/ts10m_ce_smooth/rank/)
    --level-paths DIR..  data/level_paths.json from optsplit.analysis.level_paths outputs (dirs with profile/ and walk/),
                         any number of data orders; AdamW vs Muon and the floors are means over their pairs
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
DATA, IMG = HERE / 'data', HERE / 'img'
INK, TEXT, MUTE, AXIS, HAIR = '#1c1c1c', '#262626', '#8b8b87', '#b9b9b4', '#e7e7e3'
ADAMW, MUON, SMUON = '#2c6cb0', '#d9472b', '#1d9a7a'  # one colour per optimizer on every slide
ORDERS = range(5)

for f in sorted((HERE / 'fonts').glob('*.ttf')):
    font_manager.fontManager.addfont(f)
plt.rcParams.update({
    'font.family': ['Manrope', 'Helvetica Neue', 'DejaVu Sans'], 'font.size': 15, 'text.color': TEXT,
    'axes.edgecolor': AXIS, 'axes.labelcolor': TEXT, 'axes.labelsize': 15, 'axes.linewidth': 1,
    'axes.spines.top': False, 'axes.spines.right': False, 'xtick.color': AXIS, 'ytick.color': AXIS,
    'xtick.labelcolor': TEXT, 'ytick.labelcolor': TEXT, 'xtick.labelsize': 14, 'ytick.labelsize': 14,
    'svg.fonttype': 'path', 'svg.hashsalt': 'slides', 'lines.solid_capstyle': 'round'})


def figure(w, h, **kw):
    """A figure of w x h pixels on the 1600 x 900 slide (100 px per inch, so 1 pt of text is 1.39 px)."""
    return plt.subplots(figsize=(w / 100, h / 100), **kw)


def save(fig, name):
    IMG.mkdir(exist_ok=True)
    fig.savefig(IMG / name, transparent=True, metadata={'Date': None})
    plt.close(fig)
    print('wrote', IMG / name)


def load(name):
    return json.loads((DATA / name).read_text())


# ---------------------------------------------------------------- data

def first_step(steps, v, level):
    """First step at which the running minimum of v reaches level, linear between evaluations (as dose.first_step)."""
    m = np.minimum.accumulate(v)
    keep = np.r_[True, m[1:] < m[:-1]]
    s, m = np.asarray(steps)[keep], m[keep]
    return float(np.interp(level, m[::-1], s[::-1])) if m[-1] <= level else None


def extract_runs(root):
    """data/runs.json: final validation losses, GPU time per step and to AdamW's final loss, mean curves, the p sweep."""
    runs = {'AdamW': 'ts10m_main/adamw_d{d}', 'Muon': 'ts10m_main/muon_d{d}', 'Smoothed Muon': 'ts10m_ce_smooth/smuon_d{d}',
            'LC5': 'ts10m_ce_tt/muon_p-0.3_lc5_d{d}', 'LC8': 'ts10m_ce_tt/muon_p-0.3_lc8_d{d}',
            'LC12': 'ts10m_ce_tt/muon_p-0.3_lc12_d{d}', 'Gram LC': 'ts10m_ce_gram/muon_p-0.3_lcgram_d{d}',
            'Exact SVD': 'ts10m_ce_svp/muon_p-0.3_lr0.01_d{d}'}
    # timing jobs (two copies each, data orders 0 and 1): one fast-path L40S job; the exact SVD keeps its older timing
    timing = {'AdamW': 'ts10m_ce_timing_fast/adamw', 'Muon': 'ts10m_ce_timing_fast/muon_fast',
              'LC5': 'ts10m_ce_timing_fast/lc5_fast', 'LC8': 'ts10m_ce_timing_fast/lc8_fast',
              'LC12': 'ts10m_ce_timing_fast/lc12_fast', 'Gram LC': 'ts10m_ce_timing_fast/gram_fast',
              'Exact SVD': 'ts10m_ce_tt_pilot/svd_p-0.3'}

    def records(run, kind):
        f = Path(root) / run / 'metrics.jsonl'
        return [r for r in map(json.loads, f.open()) if r['kind'] == kind] if f.exists() else []

    def val(run):
        r = records(run, 'eval')
        return np.array([x['step'] for x in r]), np.array([x['val_loss'] for x in r])

    def ms_per_step(run):  # median over the train records, so that evaluations fall out
        r = records(run, 'train')
        return 1000 * float(np.median(np.diff([x['time'] for x in r]) / np.diff([x['step'] for x in r])))

    curves = {name: [c for d in ORDERS if len((c := val(p.format(d=d)))[0]) and c[0][-1] == 10000]
              for name, p in runs.items()}
    ms = {name: float(np.mean([ms_per_step(f'{p}_d{d}') for d in (0, 1)])) for name, p in timing.items()}
    target = float(curves['AdamW'][0][1][-1])  # AdamW's final validation loss on data order 0
    out = dict(source=f'run logs under {root}', target=target, ms_per_step=ms, final={}, gpu_min={}, mean_curve={},
               steps_to_adamw_mean={}, p_sweep={})
    for name, cs in curves.items():
        out['final'][name] = [float(v[-1]) for _, v in cs]
        if name in ms:
            out['gpu_min'][name] = [None if (s := first_step(steps, v, target)) is None else s * ms[name] / 60000
                                    for steps, v in cs]
    adamw_mean = float(np.mean(out['final']['AdamW']))
    for name in ('AdamW', 'Muon', 'Smoothed Muon'):  # mean over the orders at the shared steps
        common = sorted(set.intersection(*(set(s.tolist()) for s, _ in curves[name])))
        mean = np.mean([v[np.isin(s, common)] for s, v in curves[name]], axis=0)
        out['mean_curve'][name] = dict(steps=common, val=mean.round(5).tolist())
        grid = [(s, m) for s, m in zip(common, mean) if s % 250 == 0]  # saved models are 250 steps apart
        out['steps_to_adamw_mean'][name] = next(s for s, m in grid if m <= adamw_mean)
    for p in ('-1', '-0.75', '-0.5', '-0.3', '-0.1', '0', '0.25', '0.5', '0.75'):
        out['p_sweep'][p] = float(val(f'ts10m_ce_svp/muon_p{p}_lr0.01_d0')[1][-1])
    (DATA / 'runs.json').write_text(json.dumps(out, indent=1))


def extract_metrics(path, level=1.81):
    """data/metrics.json: each metric on data orders 0-2 at training loss `level`, as metric_ratios.values."""
    m = json.loads(Path(path).read_text())
    L = min(m['levels'], key=lambda x: abs(x - level))

    def at_loss(pts):  # on the running-minimum loss, linear in the loss
        loss, val = [], []
        for l, v in pts:
            if not loss or l < loss[-1]:
                loss.append(l)
                val.append(v)
        j = next(j for j in range(len(loss) - 1) if loss[j + 1] <= L <= loss[j])
        return val[j] + (loss[j] - L) / (loss[j] - loss[j + 1]) * (val[j + 1] - val[j])

    keys = ('stable_rank', 'spectral', 'sv_spread', 'update_align', 'lambda_max', 'hess_trace', 'mlp_active', 'acc')
    out = dict(source=str(path), level=L, values={})
    for k in keys:
        for rid, pts in sorted(m['metrics'][k]['points'].items()):
            opt = rid.split('/')[1].split('_d')[0]
            if opt != 'sgd':
                out['values'].setdefault(k, {}).setdefault(opt, []).append(at_loss(pts))
    (DATA / 'metrics.json').write_text(json.dumps(out, indent=1))


def extract_level_paths(dirs):
    """data/level_paths.json: per loss level (the AdamW order-0 step that defines it) and pair group, the straight
    line's rise and the level-set walk's detour (length / distance - 1; None where it stopped short). A pair found in
    several dirs is taken from the last."""
    found = {}
    for d in map(Path, dirs):
        for f in (d / 'profile').glob('*.json'):
            found[f.stem] = f
    levels = {}
    for stem, f in sorted(found.items()):
        level, pair = stem.split('_', 1)
        prof = json.loads(f.read_text())
        w = f.parent.parent / 'walk' / f'{stem}_pc.json'
        walk = json.loads(w.read_text()) if w.exists() else None
        group = 'adamw-muon' if pair.startswith('adamw-muon') else pair.split('_')[0]
        r = levels.setdefault(level, dict(rise={}, walk={}))
        if '/adamw_d0:' in prof['a']:
            r['step'] = int(prof['a'].rsplit(':', 1)[1])
        r['rise'].setdefault(group, []).append(prof['rise'])
        r['walk'].setdefault(group, []).append(walk['ratio'] - 1 if walk and walk['reached'] else None)
    out = sorted(({'level': float(k), **v} for k, v in levels.items()), key=lambda r: r['step'])
    (DATA / 'level_paths.json').write_text(json.dumps(dict(source=[str(d) for d in dirs], levels=out), indent=1))


# ---------------------------------------------------------------- figures

def label_right(ax, x, ys, texts, gap, **kw):
    """Labels at x for points ys, pushed apart to at least `gap` (data units) without changing their order."""
    order = np.argsort(ys)
    pos = np.array(ys, dtype=float)[order]
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + gap)
    pos -= (pos.mean() - np.mean(np.array(ys)[order])) * 0.5
    for p, i in zip(pos, order):
        ax.text(x, p, texts[i], va='center', **kw)


def fig_speed():
    """Slide 1: validation loss of AdamW and Muon against the step, mean over the five data orders."""
    r = load('runs.json')
    fig, ax = figure(660, 560)
    target = np.mean(r['final']['AdamW'])
    for name, c in (('AdamW', ADAMW), ('Muon', MUON)):
        cur = r['mean_curve'][name]
        ax.plot(cur['steps'], cur['val'], color=c, lw=2.5)
        ax.text(10150, cur['val'][-1], f'{name}  {cur["val"][-1]:.3f}', va='center', fontsize=15, color=INK)
    s = r['steps_to_adamw_mean']['Muon']
    ax.axhline(target, color=MUTE, lw=1, ls=(0, (2, 3)))
    ax.plot([s, s], [1.22, target], color=MUTE, lw=1, ls=(0, (2, 3)))
    ax.plot(s, target, 'o', ms=8, color=MUON, mec='white', mew=1.5, zorder=5)
    ax.text(s - 200, target - 0.012, f'Muon reaches AdamW\'s\nfinal loss by step {s:,}', ha='right', va='top',
            fontsize=14, color=TEXT, linespacing=1.3)
    ax.set(xlim=(0, 10000), ylim=(1.22, 2.0), xlabel='training step', ylabel='validation loss')
    ax.set_xticks([0, 2500, 5000, 7500, 10000], ['0', '2,500', '5,000', '7,500', '10,000'])
    fig.subplots_adjust(left=0.14, right=0.75, bottom=0.13, top=0.97)
    save(fig, 'speed.svg')


ROWS = {  # metric: (label, format of the values)
    'update_align': ('Cosine of consecutive updates', '{:.2f}'),
    'stable_rank': ('Stable rank of weights', '{:.0f}'),
    'spectral': ('Spectral norm of weights', '{:.1f}'),
    'sv_spread': ('Largest / median singular value', '{:.1f}'),
    'mlp_active': ('Active MLP units', '{:.0%}'),
    'lambda_max': ('Top Hessian eigenvalue', '{:.1f}'),
    'hess_trace': ('Hessian trace', lambda x: f'{round(x, -1):,.0f}'),
    'acc': ('Next-token accuracy', '{:.1%}'),
}


def ratio_plot(rows, series, name, xlim=(0.3, 3.6), h=560):
    """Rows of metrics: each series' value over Muon's (log axis; dot at the midpoint of the range over data orders,
    bar over [min a / max Muon, max a / min Muon], as metric_ratios.ratio), Muon at 1, and every optimizer's mean value
    in columns on the right. rows: (label, {optimizer: [value per order]}, format), or None for a gap."""
    cols = [s for s, _ in series] + ['muon']
    colors = dict(series) | {'muon': MUON}
    names = {'adamw': 'AdamW', 'muon': 'Muon', 'smuon': 'Smoothed\nMuon'}
    fig = plt.figure(figsize=(13.92, h / 100))
    ax = fig.add_axes((0.255, 0.15, 0.43, 0.72))
    at = matplotlib.transforms.blended_transform_factory(fig.transFigure, ax.transData)
    colx = [0.75 + 0.085 * k for k in range(len(cols))]
    y, ticks, labels = 0, [], []
    for row in rows:
        y -= 1 if row else 0.5
        if not row:
            continue
        label, v, fmt = row
        ticks.append(y)
        labels.append(label)
        ax.axhline(y, color=HAIR, lw=1, zorder=0)
        m = v['muon']
        for k, (o, c) in enumerate(series):
            lo, hi = min(v[o]) / max(m), max(v[o]) / min(m)
            off = -(k - (len(series) - 1) / 2) * 0.24
            ax.plot([lo, hi], [y + off] * 2, color=c, lw=2.5, alpha=0.45)
            ax.plot((lo + hi) / 2, y + off, 'o', ms=12, color=c, mec='white', mew=1.5, zorder=4)
        for x, o in zip(colx, cols):
            fig.text(x, y, (fmt if callable(fmt) else fmt.format)(np.mean(v[o])), transform=at, ha='right', va='center', fontsize=15, color=TEXT)
    for x, o in zip(colx, cols):
        fig.text(x, 0.05, names[o], transform=at, ha='right', va='bottom', fontsize=14, color=INK, linespacing=1.15)
        ax.plot([x - 0.035, x], [-0.12] * 2, transform=at, color=colors[o], lw=3, clip_on=False)
    ax.axvline(1, color=MUON, lw=2, zorder=1)
    ax.set_xscale('log')
    ax.set_xlim(*xlim)
    ax.set_ylim(y - 0.6, -0.4)
    ax.set_yticks(ticks, labels, fontsize=16)
    ax.tick_params(axis='y', length=0, pad=14)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([0.5, 1, 2], ['0.5', '1', '2'])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel('value ÷ Muon\'s value (log scale)')
    save(fig, name)


def fig_metrics():
    """Slide 2: AdamW / Muon at training loss 1.81 (data orders 0-2)."""
    v = load('metrics.json')['values']
    rows = [(ROWS[k][0], v[k], ROWS[k][1]) if k else None for k in
            ('update_align', None, 'stable_rank', 'spectral', 'sv_spread', None, 'lambda_max', 'hess_trace', None,
             'mlp_active', None, 'acc')]
    ratio_plot(rows, [('adamw', ADAMW)], 'metrics.svg')


def fig_smoothed():
    """Slide 3: AdamW and smoothed Muon over Muon; first row: steps to AdamW's final validation loss (mean of 5)."""
    v, r = load('metrics.json')['values'], load('runs.json')
    steps = r['steps_to_adamw_mean']
    speed = {o: [steps[n]] for o, n in (('adamw', 'AdamW'), ('muon', 'Muon'), ('smuon', 'Smoothed Muon'))}
    rows = [('Steps to AdamW\'s final loss', speed, '{:,.0f}'), None]
    rows += [(ROWS[k][0], v[k], ROWS[k][1]) if k else None for k in
             ('update_align', None, 'stable_rank', 'spectral', 'mlp_active', None, 'lambda_max', 'hess_trace')]
    ratio_plot(rows, [('adamw', ADAMW), ('smuon', SMUON)], 'smoothed.svg', h=580)


def fig_power():
    """Slide 4: final validation loss against the power p (exact SVD, data order 0, learning rate 0.01)."""
    r = load('runs.json')
    p = np.array([float(k) for k in r['p_sweep']])
    v = np.array(list(r['p_sweep'].values()))
    top = 1.372
    fig, ax = figure(1000, 560)
    for name, c, y in (('AdamW', ADAMW, r['final']['AdamW'][0]), ('Muon', MUON, r['final']['Muon'][0])):
        ax.axhline(y, color=c, lw=2, ls=(0, (4, 3)))
        ax.text(1.12, y, name, va='center', fontsize=15, color=INK)
    ax.plot(p, v, color=INK, lw=1.5, zorder=3)
    ax.plot(p[v < top], v[v < top], 'o', ms=9, color=INK, mec='white', mew=1.5, zorder=4)
    for x, y in zip(p[v >= top], v[v >= top]):  # off the top: an arrow and the value
        ax.annotate('', (x, top + 0.003), (x, top - 0.006), arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.5),
                    annotation_clip=False)
        ax.text(x, top + 0.004, f'{y:.2f}', fontsize=14, color=TEXT, ha='center', va='bottom')
    best = np.argmin(v)
    ax.plot(p[best], v[best], 'o', ms=14, color=INK, mec='white', mew=1.5, zorder=5)
    ax.text(p[best] - 0.07, v[best], f'p = −0.3:  {v[best]:.4f}', ha='right', va='center', fontsize=15, color=INK)
    ax.set(xlim=(-1.08, 1.08), ylim=(1.293, top), xlabel='power p applied to the singular values',
           ylabel='final validation loss')
    ax.set_xticks([-1, -0.5, 0, 0.5, 1], ['−1', '−0.5', '0\nMuon', '0.5', '1\nmomentum\nas it is'])
    ax.set_yticks([1.30, 1.32, 1.34, 1.36])
    fig.subplots_adjust(left=0.11, right=0.89, bottom=0.25, top=0.9)
    save(fig, 'power.svg')


def fig_cost():
    """Slide 5: final validation loss (left) and GPU minutes to AdamW's final loss on order 0 (right)."""
    r = load('runs.json')
    names = ['Muon', 'LC5', 'LC8', 'LC12', 'Gram LC', 'Exact SVD']
    fig, axes = figure(1392, 520, ncols=2, gridspec_kw=dict(wspace=0.12))
    for ax, key, xlim, fmt in ((axes[0], 'final', (1.298, 1.316), '{:.4f}'), (axes[1], 'gpu_min', (13, 18.2), '{:.1f}')):
        ax.axvline(np.mean(r[key]['Muon']), color=MUON, lw=1.5, ls=(0, (4, 3)), zorder=1)
        for i, n in enumerate(names):
            c = MUON if n == 'Muon' else INK
            vals = [x for x in r[key][n] if x is not None]
            m = np.mean(vals)
            ax.axhline(-i, color=HAIR, lw=1, zorder=0)
            if m > xlim[1]:  # off the axis: an arrow and the value
                ax.annotate('', (xlim[1], -i), (xlim[1] - 0.5, -i),
                            arrowprops=dict(arrowstyle='-|>', color=c, lw=1.5), annotation_clip=False)
                ax.text(xlim[1] - 0.6, -i + 0.3, f'{m:.0f} min', ha='right', fontsize=14, color=TEXT)
                continue
            ax.plot(vals, [-i] * len(vals), 'o', ms=6, color=c, alpha=0.35, mew=0)
            ax.plot(m, -i, 'o', ms=12, color=c, mec='white', mew=1.5, zorder=4)
            ax.text(m, -i + 0.3, fmt.format(m), ha='center', fontsize=14, color=TEXT, zorder=3,
                    bbox=dict(fc='white', ec='none', pad=1))
        ax.set(xlim=xlim, ylim=(-len(names) + 0.5, 0.8))
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0, pad=14)
    n = {k: len(r['final'][k]) for k in names}
    axes[0].set_yticks([-i for i in range(len(names))],
                       [k if n[k] == 5 else f'{k}  (n = {n[k]})' for k in names], fontsize=16)
    axes[1].set_yticks([])
    axes[0].set_xlabel('final validation loss, after 10,000 steps')
    axes[1].set_xlabel(f'GPU minutes to reach AdamW\'s final loss ({r["target"]:.4f})')
    axes[0].set_xticks([1.30, 1.305, 1.31, 1.315])
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.17, top=0.97)
    save(fig, 'cost.svg')


def fig_paths():
    """Slide 6: densMAP of the prediction paths (data orders 0-2, from step 250) and JS at equal step."""
    runs, js = load('densmap.json')['runs'], load('js_steps.json')
    fig = plt.figure(figsize=(13.92, 5.6))
    ax = fig.add_axes((0.0, 0.02, 0.45, 0.96))
    for t in runs.values():
        c = ADAMW if t['optimizer'] == 'adamw' else MUON
        xy = np.array(t['xy'])
        ax.plot(*xy.T, color=c, lw=1.5, alpha=0.75)
        ax.plot(*xy[0], 'o', ms=8, mfc='white', mec=c, mew=1.8, zorder=4)
        ax.plot(*xy[t['steps'].index(1000)], 'o', ms=11, color=c, mec='white', mew=1.5, zorder=5)
        ax.plot(*xy[-1], 's', ms=7, color=c, zorder=4)
    loss = {o: np.mean([t['loss'][t['steps'].index(1000)] for t in runs.values() if t['optimizer'] == o])
            for o in ('adamw', 'muon')}
    H = matplotlib.lines.Line2D
    key = [H([], [], color=ADAMW, lw=2), H([], [], color=MUON, lw=2),
           H([], [], ls='', marker='o', ms=8, mfc='white', mec=INK, mew=1.8),
           H([], [], ls='', marker='o', ms=11, color=INK, mec='white'), H([], [], ls='', marker='s', ms=7, color=INK)]
    ax.legend(key, ['AdamW', 'Muon', 'step 250',
                    f'step 1000 (training loss:\nAdamW {loss["adamw"]:.2f}, Muon {loss["muon"]:.2f})', 'step 10,000'],
              loc='lower left', frameon=False, fontsize=14, handlelength=1.6,
              labelcolor=TEXT)
    ax.set_aspect('equal', 'datalim')
    ax.axis('off')
    bx = fig.add_axes((0.56, 0.15, 0.31, 0.8))
    lines = (('adamw_vs_muon', INK, 'AdamW vs Muon'), ('floor_adamw', ADAMW, 'two AdamW runs'),
             ('floor_muon', MUON, 'two Muon runs'))
    for k, c, _ in lines:
        bx.plot(js['steps'], js[k], color=c, lw=2.8 if k == 'adamw_vs_muon' else 2, marker='o', ms=7,
                mec='white', mew=1)
    label_right(bx, 10400, [js[k][-1] for k, _, _ in lines], [t for _, _, t in lines], 0.011, fontsize=15,
                color=INK)
    bx.set(xlim=(0, 10000), ylim=(0, 0.15), xlabel='training step', ylabel='distance between predictions (bits)')
    bx.set_xticks([0, 5000, 10000], ['0', '5,000', '10,000'])
    save(fig, 'paths.svg')


def fig_level_paths():
    """Slide 7: between two models at the same loss, the straight line's barrier and the level-set walk's detour,
    against the loss level (equally spaced); AdamW vs Muon and the two floors, means over their pairs (band: range).
    The detour is shown at the levels where every walk reached the other model."""
    lv = load('level_paths.json')['levels']
    style = {'adamw-muon': (INK, 'AdamW vs Muon'), 'adamw': (ADAMW, 'two AdamW runs'), 'muon': (MUON, 'two Muon runs')}
    npairs = {g: len(lv[0]['rise'][g]) for g in style}
    reached = [i for i, r in enumerate(lv) if all(w is not None for g in style for w in r['walk'][g])]
    fig, axes = figure(1392, 540, ncols=2, gridspec_kw=dict(wspace=0.22, width_ratios=(len(lv), len(reached) + 1)))
    for g, (c, name) in style.items():
        lw = 2.8 if g == 'adamw-muon' else 2
        for ax, idx, key, scale in ((axes[0], range(len(lv)), 'rise', 1), (axes[1], reached, 'walk', 100)):
            y = np.maximum(scale * np.array([lv[i][key][g] for i in idx], dtype=float), 1e-3)  # walk: >= 0.001%
            ax.plot(range(len(y)), y.mean(1), color=c, lw=lw, marker='o', ms=8, mec='white', mew=1,
                    label=f'{name} ({npairs[g]} pair{"s" * (npairs[g] > 1)})' if ax is axes[0] else None)
            if y.shape[1] > 1:
                ax.fill_between(range(len(y)), y.min(1), y.max(1), color=c, alpha=0.12, lw=0)
    for ax, idx in ((axes[0], range(len(lv))), (axes[1], reached)):
        ax.set_xticks(range(len(idx)), [f'{round(lv[i]["step"] / 100)}%' for i in idx])
        ax.set_xlim(-0.4, len(idx) - 0.6)
        ax.set_xlabel('loss level: AdamW\'s loss after this % of its run')
    axes[0].set(ylim=(0, None), ylabel='barrier on the straight line (nats)')
    axes[1].set(yscale='log', ylim=(5e-4, 40), ylabel='detour of a constant-loss walk (%)')
    axes[1].set_yticks([1e-3, 1e-2, 1e-1, 1, 10], ['0.001', '0.01', '0.1', '1', '10'])
    fig.legend(loc='upper center', ncol=3, frameon=False, fontsize=15, handlelength=1.8, columnspacing=2.5)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.16, top=0.86)
    save(fig, 'level_paths.svg')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs')
    ap.add_argument('--metrics')
    ap.add_argument('--level-paths', nargs='+')
    args = ap.parse_args()
    if args.runs:
        extract_runs(args.runs)
    if args.metrics:
        extract_metrics(args.metrics)
    if args.level_paths:
        extract_level_paths(args.level_paths)
    for f in (fig_speed, fig_metrics, fig_smoothed, fig_power, fig_cost, fig_paths, fig_level_paths):
        f()
