"""Redraw negative_p_method_map.pdf from its own vector paths (its script is lost), renaming the LC legend entry.

usage: map_figure.py original.pdf out.pdf
"""
import sys
import numpy as np
import pymupdf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42  # TrueType, as in the original

W, H = 508.07977294921875, 125.72808837890625  # page size of the original, in points
X0 = (33.439998626708984, 209.98976135253906)  # left edge of panels (a), (b)
AW = 190.98927307128906 - 33.439998626708984   # axes width
YT, YB = 4.955680847167969, 96.84620666503906   # axes top, bottom
XLIM, YLIM = (1e-7, 1.0), (0.03, 150.0)

# curves: index of the drawing in the original, panel, zorder (exact on top of the spines)
CURVES = [(26, 0, 2), (27, 0, 2), (28, 0, 2), (29, 0, 2), (30, 0, 2), (31, 0, 2), (34, 0, 3),
          (60, 1, 2), (61, 1, 2), (62, 1, 2), (63, 1, 2), (66, 1, 3)]
LEGEND = [(34, r'exact $\sigma^{-0.3}$, constant below $10^{-6}\sigma_1$'), (26, 'two-track, 5 iterations'),
          (27, 'two-track, 8 iterations'), (28, 'two-track, 12 iterations'), (29, 'Muon, 5 iterations'),
          (30, r'DynMuon, $p=-0.3$'), (31, 'Contra-Muon'), (62, 'LC5, LC8, LC12')]
# texts: (string, size, color, baseline origin in the original)
TEXTS = [('5', 6, '0.2', 136.681, 37.798), ('8', 6, '0.2', 100.414, 26.199), ('12', 6, '0.2', 50.151, 10.874),
         ('(a)', 7, 'k', 178.073, 91.489),
         ('5', 6, '0.2', 303.184, 31.782), ('8', 6, '0.2', 275.285, 23.028), ('12', 6, '0.2', 229.304, 8.8),
         ('(b)', 7, 'k', 354.482, 91.489),
         (r'singular value of the momentum, as a fraction of the matrix norm $\|N\|_F$', 7, 'k', 129.54, 122.288)]  # its baseline lands 0.656 pt higher

drawings = pymupdf.open(sys.argv[1])[0].get_drawings()


def to_data(panel, X, Y):
    lx = np.log10(XLIM[0]) + np.log10(XLIM[1] / XLIM[0]) * (X - X0[panel]) / AW
    ly = np.log10(YLIM[0]) + np.log10(YLIM[1] / YLIM[0]) * (YB - Y) / (YB - YT)
    return 10 ** lx, 10 ** ly


def style(d):
    s = dict(color='#%02x%02x%02x' % tuple(round(255 * c) for c in d['color']), lw=round(d['width'], 2))
    if d['dashes'].startswith('[ '):
        s['ls'] = '--'
    return s


fig = plt.figure(figsize=(W / 72, H / 72))
axes = [fig.add_axes((x / W, 1 - YB / H, AW / W, (YB - YT) / H)) for x in X0]
for ax in axes:
    ax.set(xscale='log', yscale='log', xlim=XLIM, ylim=YLIM, xticks=10.0 ** np.arange(-7, 1),
           yticks=10.0 ** np.arange(-1, 3))
    ax.minorticks_off()
    ax.grid(color='0.9', lw=0.4)
    ax.tick_params(labelsize=6, length=2.5, width=0.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_linewidth(0.5)
axes[1].tick_params(labelleft=False)
axes[0].set_ylabel('singular value of the update', fontsize=7)

for i, panel, z in CURVES:
    pts = [drawings[i]['items'][0][1]] + [it[2] for it in drawings[i]['items']]
    x, y = to_data(panel, np.array([p.x for p in pts]), np.array([p.y for p in pts]))
    axes[panel].plot(x, y, zorder=z, **style(drawings[i]))
for s, size, color, x, y in TEXTS:
    fig.text(x / W, 1 - y / H, s, fontsize=size, color=color, ha='left', va='baseline')

handles = [plt.Line2D([], [], **style(drawings[i])) for i, _ in LEGEND]
fig.legend(handles, [l for _, l in LEGEND], fontsize=6, frameon=False, loc='upper left',
           bbox_to_anchor=(374.48 / W, 1 - 21.732 / H))
fig.savefig(sys.argv[2])
