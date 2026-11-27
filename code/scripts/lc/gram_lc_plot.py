"""Gram LC figure: (a) scalar maps of Muon (envelope), s^p, LC5m and Gram LC; (b) on real momentum matrices, the
median relative error of each singular value against the method's float64 map, by quarter decade of s
(gram_lc_check.py output).
    uv run --no-project --with numpy --with scipy --with matplotlib python gram_lc_plot.py <check.npz> <out.png>"""
import json
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from lib import MUON, iterates
from lc5m import HERE, P, SC
from gram_lc import gmap

r = np.load(sys.argv[1])
gram = json.load(open(os.path.join(HERE, "gram_lc_coeffs.json")))["gram"]
lc5m = json.load(open(os.path.join(HERE, "lc5m_coeffs.json")))["lc5m"]
s = np.logspace(-8, 0, 20001)
phi = iterates(s, [MUON] * 5)[-1]
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
a.plot(s, np.maximum.accumulate(phi), color="k", lw=1, ls="--", label=r"Muon envelope $\max_{t\leq\sigma}\varphi(t)$")
a.plot(s, s ** P, color="C2", lw=1.2, label=r"$\sigma^{p}$, $p=-0.3$")
a.plot(s, sum(w * y for w, y in zip(lc5m["weights"], iterates(s, lc5m["steps"]))), color="C3", lw=1.2, label="LC5m")
a.plot(s, gmap(gram, s), color="C1", lw=1.2, label=f"Gram LC ({gram['P']} Gram products)")
a.axvline(SC, color="0.8", lw=0.5)
a.set(xscale="log", yscale="log", ylim=(1e-4, 300), xlabel=r"$\sigma = s / \|N\|_{S_8}$",
      ylabel="output singular value", title="(a) scalar maps, float64")
a.legend(frameon=False, fontsize=8, loc="lower right")
sv = r["s"]
q = np.floor(np.log10(sv) * 4) / 4
bins = np.unique(q)
for key, label, color in (("gram_float32", "Gram LC, float32", "C1"), ("gram_tf32", "Gram LC, TF32", "C4"),
                          ("gram_bfloat16", "Gram LC, bfloat16", "C0"), ("lc5m_bfloat16", "LC5m, bfloat16", "C3")):
    rel = np.abs(r[f"{key}_z"] - r[f"{key}_g"]) / np.abs(r[f"{key}_g"])
    med = [np.median(rel[q == x]) for x in bins]
    b.plot(10 ** (bins + 0.125), np.maximum(med, 1e-8), color=color, lw=1.2, marker=".", ms=3,
           label=f"{label} (floor {float(r[f'floor_{key}']):.0e})")
b.axhline(0.1, color="0.6", lw=0.6, ls=":")
b.set(xscale="log", yscale="log", xlabel=r"$\sigma = s / \|N\|_{S_8}$ of the momentum's singular value",
      ylabel="median |computed - float64 map| / map",
      title=f"(b) rounding on {len(r['whole_gram_float32'])} momentum matrices")
b.legend(frameon=False, fontsize=8)
fig.savefig(sys.argv[2], dpi=200)
print("wrote", sys.argv[2])
