"""Panel (b) of Fig 1 (distance at equal step) under each readout of optsplit.analysis.readouts: CE as reference,
then MSE. Grey lines: the median step at which AdamW / Muon runs first reach training loss CLIFF (MSE only).
    python scripts/mse_paths_fig.py CE.json MSE.json OUT.pdf"""

import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

CLIFF = 0.9
COLORS = dict(cross='C0', floor_adamw='0.35', floor_muon='0.7')
LABELS = dict(cross='AdamW vs Muon', floor_adamw='AdamW, other data order', floor_muon='Muon, other data order')
PANELS = [('ce', 'clamp', 'CE: JS (bits), as Fig 1'), ('mse', 'clamp', r'MSE: JS of max(z,0)/$\Sigma$, as Fig 1'),
          ('mse', 'soft', 'MSE: JS of softmax(z/T)'), ('mse', 'disagree', 'MSE: top-1 disagreement'),
          ('mse', 'l2', 'MSE: normalised L2 of z')]


def cliff(run):
    s, l = map(np.array, run['train'])
    return s[np.argmax(l <= CLIFF)]


def main(ce, mse, out):
    data = dict(ce=json.load(open(ce)), mse=json.load(open(mse)))
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.2))
    axes = axes.ravel()
    for ax, (loss, key, title) in zip(axes, PANELS):
        d = data[loss]
        for kind in ('floor_muon', 'floor_adamw', 'cross'):
            for i, (k, rows) in enumerate((k, v) for k, v in d['step'].items() if k.split('|')[0] == kind):
                ax.plot(d['steps'], [r[key] for r in rows], color=COLORS[kind], lw=1 if kind == 'cross' else 0.8,
                        zorder=3 if kind == 'cross' else 2, label=LABELS[kind] if i == 0 else None)
        if loss == 'mse':
            for opt, ls in (('adamw', ':'), ('muon', '--')):
                c = np.median([cliff(r) for r in d['runs'].values() if r['optimizer'] == opt])
                ax.axvline(c, color='0.5', ls=ls, lw=0.8, label=f'{"AdamW" if opt == "adamw" else "Muon"} '
                           f'reaches loss {CLIFF}')
        ax.set_xscale('symlog', linthresh=100, linscale=0.5)
        ax.set(xlabel='step', title=title, xlim=(0, 10000), ylim=(0, None), xticks=[0, 100, 1000, 10000])
        ax.title.set_fontsize(8)
        ax.tick_params(labelsize=7)
        ax.xaxis.label.set_fontsize(8)
    axes[-1].axis('off')
    h, l = axes[1].get_legend_handles_labels()
    order = [2, 1, 0, 3, 4]  # AdamW vs Muon first, then the floors and the loss marks
    axes[-1].legend([h[i] for i in order], [l[i] for i in order], fontsize=7, loc='center')
    for ax, l in zip(axes, 'abcde'):
        ax.annotate(f'({l})', (0, 1), xytext=(-4, 3), xycoords='axes fraction', textcoords='offset points',
                    ha='right', va='bottom', fontweight='bold', fontsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight')
    print('wrote', out)


if __name__ == '__main__':
    main(*sys.argv[1:4])
