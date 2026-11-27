"""Figure of LC5m's scalar map against Muon's, its envelope, s^p and LC5.
    uv run --no-project --with numpy --with scipy --with matplotlib python lc5m_plot.py <out.png>"""
import json
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from lib import MUON, iterates
from lc5m import HERE, P, SC

s = np.logspace(-8, 0, 20001)
phi = iterates(s, [MUON] * 5)[-1]
lc5 = json.load(open(os.path.join(HERE, "final_coeffs.json")))["p-0.3_lc5_eps3.0e-03"]
lc5m = json.load(open(os.path.join(HERE, "lc5m_coeffs.json")))["lc5m"]
g = lambda e: sum(w * y for w, y in zip(e["weights"], iterates(s, e["steps"])))
fig, ax = plt.subplots(figsize=(6, 3.6), layout="constrained")
ax.plot(s, phi, color="0.6", lw=0.8, label=r"Muon $\varphi(\sigma)$")
ax.plot(s, np.maximum.accumulate(phi), color="k", lw=1, ls="--", label=r"Muon envelope $\max_{t\leq\sigma}\varphi(t)$")
ax.plot(s, s ** P, color="C2", lw=1.2, label=r"$\sigma^{p}$, $p=-0.3$")
ax.plot(s, g(lc5), color="C0", lw=1, label="LC5")
ax.plot(s, g(lc5m), color="C3", lw=1.2, label="LC5m")
ax.axvline(SC, color="0.8", lw=0.5)
ax.set(xscale="log", yscale="log", ylim=(1e-4, 300), xlabel=r"$\sigma = s / \|N\|_{S_8}$", ylabel="output singular value")
ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.savefig(sys.argv[1], dpi=200)
print("wrote", sys.argv[1])
