"""Final coefficient sets: rescale every iterate to maximum 1 (a change of variables that leaves the map unchanged),
scale the output so that f(s) ~ s^p, round to DIG significant digits, and re-verify the band with the rounded values.
Writes final_coeffs.json (the values to report) and prints the selected sets.
Run after s3_joint.py (which writes the joint_p*.json it reads): python verify_final.py 5 [set,set,...].

Update rule (X_0 = G / ||G||_S8, M_k = X_k X_k^T):
   X_{k+1} = (a_k I + b_k M_k + c_k M_k^2) X_k,  k = 0..K-1        (3 products per step)
   update  = sum_k w_k X_k                                          (no product)
   [+1 product]  update += (sum_k u_k M_k + v_k M_k^2) X_K
"""
import glob
import json
import os
import sys
import numpy as np
from lib import iterates

HERE = os.path.dirname(os.path.abspath(__file__))
DIG = int(sys.argv[1]) if len(sys.argv) > 1 else 5
SIG = np.logspace(-9, 0, 900001)


def rnd(x, d=DIG):
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    nz = x != 0
    e = np.floor(np.log10(np.abs(x[nz])))
    out[nz] = np.round(x[nz] / 10 ** e, d - 1) * 10 ** e
    return out


def fmap(sig, H, w, uv=None):
    ys = iterates(sig, H)
    f = sum(wk * y for wk, y in zip(w, ys))
    terms = [wk * y for wk, y in zip(w, ys)]
    if uv is not None:
        for k in range(len(H)):
            t = ys[-1] * (uv[2 * k] * ys[k] ** 2 + uv[2 * k + 1] * ys[k] ** 4)
            f = f + t
            terms.append(t)
    return f, ys, terms


def gauge(H, w, uv, kappa):
    ys = iterates(SIG, H)
    a = [1.0] + [np.sign(y[0]) * np.abs(y).max() for y in ys[1:]]
    K = len(H)
    H2 = [np.array([h[0], h[1] * a[k] ** 2, h[2] * a[k] ** 4]) * a[k] / a[k + 1] for k, h in enumerate(H)]
    w2 = np.array([w[k] * a[k] for k in range(K + 1)]) / kappa
    uv2 = None
    if uv is not None:
        uv2 = np.array([[uv[2 * k] * a[k] ** 2, uv[2 * k + 1] * a[k] ** 4] for k in range(K)]).ravel() * a[K] / kappa
    return H2, w2, uv2


def describe(H, w, uv, p, eps):
    f, ys, terms = fmap(SIG, H, w, uv)
    sel = SIG >= eps
    r = f[sel] / SIG[sel] ** p
    kap = np.sqrt(r.min() * r.max())
    ratio = f / (kap * SIG ** p)
    below = np.where(ratio < 0.5)[0]
    canc = (np.sum(np.abs(terms), 0)[sel] / np.abs(f[sel])).max()
    return dict(kappa=kap, half_width=r.max() / kap - 1, half_point=SIG[below.max()] if len(below) else np.nan,
                slope0=f[0] / SIG[0], fmin=f.min(), ymax=max(np.abs(y).max() for y in ys[1:]), cancel=canc,
                peak=f.max(), peak_at=SIG[np.argmax(f)])


out = {}
for p in [-0.3, -0.5]:
    J = {}
    for fn in glob.glob(os.path.join(HERE, f"joint_p{p}_*.json")):
        J.update(json.load(open(fn)))
    for key, e in J.items():
        arch = e["arch"]
        if not arch.startswith("lc") or "c" in arch[2:] or arch in ("lc5x0", "lc4x"):
            continue
        K = len(e["H"])
        w, uv = e["out"][:K + 1], (e["out"][K + 1:] or None)
        H2, w2, uv2 = gauge(e["H"], w, uv, e["kappa"])
        d0 = describe(e["H"], w, uv, p, e["eps"])
        Hr, wr, uvr = [rnd(h) for h in H2], rnd(w2), (rnd(uv2) if uv2 is not None else None)
        d = describe(Hr, wr, uvr, p, e["eps"])
        out[f"p{p}_{key}"] = dict(p=p, K=K, products=e["products"], eps=e["eps"], steps=[list(map(float, h)) for h in Hr],
                                  weights=list(map(float, wr)), extra=(list(map(float, uvr)) if uvr is not None else None),
                                  half_width=d["half_width"], half_width_unrounded=d0["half_width"], half_point=d["half_point"],
                                  kappa=d["kappa"], max_iterate=d["ymax"], cancellation=d["cancel"], slope_at_0=d["slope0"],
                                  peak=d["peak"], peak_at=d["peak_at"], fmin=d["fmin"])
json.dump(out, open(os.path.join(HERE, "final_coeffs.json"), "w"), indent=1)

print(f"rounded to {DIG} significant digits; band = max |f/(kappa s^p) - 1| on [eps, 1]")
print(f"{'set':28s} prod  band(rounded)  band(full)  half-point  kappa   max|X_k|  cancel  slope at 0   peak (at)")
for k, e in out.items():
    print(f"{k:28s} {e['products']:3d}   {e['half_width'] * 100:7.2f}%     {e['half_width_unrounded'] * 100:7.2f}%   {e['half_point']:.1e}   "
          f"{e['kappa']:.4f}  {e['max_iterate']:.3f}    {e['cancellation']:6.1f}  {e['slope_at_0']:10.3g}  {e['peak']:.1f} ({e['peak_at']:.1e})")

SHOW = sys.argv[2].split(",") if len(sys.argv) > 2 else []
for k in SHOW:
    e = out[k]
    print(f"\n{k}: {e['products']} products, +-{e['half_width'] * 100:.1f}% on [{e['eps']:g}, 1]")
    print("  steps (a, b, c):")
    for h in e["steps"]:
        print("    (" + ", ".join(f"{v:.{DIG}g}" for v in h) + "),")
    print("  weights w_0..w_K: " + ", ".join(f"{v:.{DIG}g}" for v in e["weights"]))
    if e["extra"]:
        print("  extra (u_k, v_k): " + ", ".join(f"{v:.{DIG}g}" for v in e["extra"]))
