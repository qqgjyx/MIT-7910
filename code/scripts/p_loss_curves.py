import json, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(sys.argv[1])
ps = [-1, -0.75, -0.5, -0.3, -0.1, 0, 0.25, 0.5, 0.75]


def load(path):
    rec = {}
    for line in open(path):
        r = json.loads(line)
        if r.get('kind') == 'eval':
            rec[r['step']] = r  # resumed runs repeat steps: keep the last
    steps = sorted(rec)
    return steps, [rec[s]['train_loss'] for s in steps], [rec[s]['val_loss'] for s in steps]


cmap = plt.get_cmap('turbo')
runs = [(f'p = {p:g}', cmap(0.05 + 0.9 * ps.index(p) / (len(ps) - 1)), '-', load(root / f'ts10m_mse_svp/muon_p{p:g}_lr0.005_d0/metrics.jsonl')) for p in ps]
runs.append(('Muon (Newton–Schulz)', 'k', '--', load(root / 'ts10m_mse_main/muon_d0/metrics.jsonl')))
runs.append(('AdamW', '0.5', ':', load(root / 'ts10m_mse_main/adamw_d0/metrics.jsonl')))

fig, ax = plt.subplots(2, 2, figsize=(11, 7.5), sharex='row')
for name, color, ls, (s, tr, va) in runs:
    for col, y in enumerate((tr, va)):
        for row in (0, 1):
            ax[row, col].plot(s, y, color=color, ls=ls, lw=1.6, label=name)
for col, title in enumerate(('Training loss', 'Validation loss')):
    ax[0, col].set_title(title)
    ax[0, col].set_ylim(0.55, 1.0)
    ax[1, col].set_xlim(5000, 10000)
    ax[1, col].set_ylim(0.55, 0.62)
    ax[1, col].set_xlabel('step')
ax[0, 0].set_ylabel('MSE loss')
ax[1, 0].set_ylabel('MSE loss (zoom, steps 5000–10000)')
ax[0, 1].legend(fontsize=8, ncol=2)
fig.suptitle('Singular-value power p: Muon with update $U S^p V^T$ (lr 0.005, data order 0)')
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=160)
