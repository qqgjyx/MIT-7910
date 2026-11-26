"""negative_p_method_lc.pdf: for LC5 at p = -0.3, the singular-value curves phi_k of its iterates, their weighted
sum g and the target sigma^p, from the coefficients in final_coeffs.json (set p-0.3_lc5_eps3.0e-03, as in muon.py).

usage: lc_figure.py final_coeffs.json out.pdf
"""
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42

e = json.load(open(sys.argv[1]))['p-0.3_lc5_eps3.0e-03']
p, eps = e['p'], e['eps']
s = np.logspace(-5, 0, 4001)
phi = [s]
for a, b, c in e['steps']:  # one Newton-Schulz step on a singular value
    x = phi[-1]
    phi.append(a * x + b * x ** 3 + c * x ** 5)
g = np.array(e['weights']) @ np.array(phi)
band = np.abs(g / s ** p - 1)[s >= eps].max()
print(f'max |g / s^p - 1| on [{eps}, 1]: {band:.4f}; json half_width {e["half_width"]:.4f}')
print('weights', e['weights'])

fig = plt.figure(figsize=(4.3, 1.75))
ax = fig.add_axes((0.1, 0.2, 0.6, 0.76))
ax.set(xscale='log', yscale='log', xlim=(1e-4, 1), ylim=(3e-3, 20), xticks=10.0 ** np.arange(-4, 1),
       yticks=10.0 ** np.arange(-2, 2))
ax.minorticks_off()
ax.grid(color='0.9', lw=0.4)
ax.tick_params(labelsize=6, length=2.5, width=0.5)
ax.spines[['top', 'right']].set_visible(False)
ax.spines[['left', 'bottom']].set_linewidth(0.5)
ax.set_xlabel(r'singular value $\sigma$ of $X_0$', fontsize=7, labelpad=2)

ramp = plt.get_cmap('Greys')(np.linspace(0.3, 0.6, len(phi)))
for k, (f, col) in enumerate(zip(phi, ramp)):
    ax.plot(s, f, color=col, lw=0.7, zorder=2)
    x = 0.04 / np.prod([st[0] for st in e['steps'][:k]])  # where the straight part of phi_k is at 0.04
    ax.text(x, 0.04, rf'$\varphi_{k}$', fontsize=6, color='0.25', ha='center', va='center', zorder=2.5,
            bbox=dict(fc='white', ec='none', pad=0.4))
ax.plot(s, g, color='#c39bd8', lw=2.4, zorder=3, solid_capstyle='butt')
ax.plot(s, s ** p, color='k', lw=0.8, zorder=4)
ax.axvline(eps, color='0.4', lw=0.6, ls=':', zorder=1)
ax.text(eps * 1.12, 4e-3, r'$\varepsilon$', fontsize=6, color='0.25', va='bottom')

handles = [plt.Line2D([], [], color=ramp[3], lw=0.7), plt.Line2D([], [], color='#c39bd8', lw=2.4),
           plt.Line2D([], [], color='k', lw=0.8)]
labels = [r'$\varphi_0,\dots,\varphi_5$ (iterates)', r'$g=w_0\varphi_0+\dots+w_5\varphi_5$', r'target $\sigma^{-0.3}$']
fig.legend(handles, labels, fontsize=6, frameon=False, loc='center left', bbox_to_anchor=(0.71, 0.58))
fig.savefig(sys.argv[2])
