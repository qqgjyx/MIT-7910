"""Phase 1: joint optimization of all coefficients, several architectures, continuation in the cutoff eps.
For each architecture and eps: best band of f(s)/(kappa s^p) on [eps, 1] over several starts. Saves the coefficients.

This is the fit behind final_coeffs.json (scalar map on a sigma grid, float64; opt.py has the objective). The runs that
produced it, each writing joint_p{p}_{archs}.json next to this file (unrounded coefficients):
    uv run --no-project --with numpy --with scipy python s3_joint.py -0.3 lc5,lc6          (about 5 min)
    uv run --no-project --with numpy --with scipy python s3_joint.py -0.3 lc5x,lc4x,lc5x0
    uv run --no-project --with numpy --with scipy python s3_joint.py -0.3 lc8 1e-2,3e-3,1e-3,5e-4,3e-4,2e-4,1e-4,5e-5
    uv run --no-project --with numpy --with scipy python s3_joint.py -0.3 lc12 1e-2,1e-3,1e-4,3e-5,1e-5,5e-6,3e-6,2e-6,1e-6
    uv run --no-project --with numpy --with scipy python s3_joint.py -0.5 lc5,lc5x,lc6
then verify_final.py rounds them and writes final_coeffs.json (it keeps only the sets whose joint files are present)."""
import json
import sys
import time
import numpy as np
from lib import MUON, TT_COEF
from opt import Problem, products, describe, fmt

np.seterr(all="ignore")
p = float(sys.argv[1]) if len(sys.argv) > 1 else -0.3
only = sys.argv[2].split(",") if len(sys.argv) > 2 else None
rng = np.random.default_rng(1)
ZONLY5 = [[2.7273, -3.4572, 1.7574], [2.5675, -2.8014, 1.3162], [2.5334, -2.6088, 1.1442], [2.2596, -2.6503, 1.2804]]


def phi(a, rho):
    return (a, (5 - rho) / 2 - 2 * a, a - (3 - rho) / 2)


def starts_quintic(K):
    S = {"muon": [MUON] * K}
    for a in [2.4, 3.0, 3.6, 4.2]:
        S[f"phi{a}"] = [phi(a, a ** p)] * K
        S[f"phi{a}t"] = [phi(a, 0.2)] * K
    S["grad"] = [phi(a, 0.3) for a in np.linspace(4.5, 2.2, K)]
    if K >= 5:
        S["zonly"] = ZONLY5 + [MUON] * (K - 4)
    return S


def starts_cubic(K):
    S = {}
    for a in [1.6, 1.9, 2.2, 2.6]:
        S[f"c{a}"] = [(a, 1 - a)] * K
    S["cgrad"] = [(a, 1 - a) for a in np.linspace(2.6, 1.5, K)]
    return S


ARCH = {
    "lc5":   dict(degs=[2] * 5, family="lc"),
    "lc5x":  dict(degs=[2] * 5, family="lc", extras=5),
    "lc5x0": dict(degs=[2] * 5, family="lc", extras=0),
    "lc4x":  dict(degs=[2] * 4, family="lc", extras=4),
    "lc6":   dict(degs=[2] * 6, family="lc"),
    "lc7":   dict(degs=[2] * 7, family="lc"),
    "lc8":   dict(degs=[2] * 8, family="lc"),
    "lc10":  dict(degs=[2] * 10, family="lc"),
    "lc12":  dict(degs=[2] * 12, family="lc"),
    "lc7c":  dict(degs=[1] * 7, family="lc"),
    "lc7cx": dict(degs=[1] * 7, family="lc", extras=7),
    "tt4":   dict(degs=[2] * 4, family="tt"),
    "tt5":   dict(degs=[2] * 5, family="tt"),
}
EPS = [1e-2, 6e-3, 4e-3, 3e-3, 2e-3, 1.5e-3, 1e-3, 7e-4, 5e-4]
if len(sys.argv) > 3:
    EPS = [float(x) for x in sys.argv[3].split(",")]
out = {}
for name, a in ARCH.items():
    if only and name not in only:
        continue
    K = len(a["degs"])
    npro = products(a["degs"], a["family"], a.get("extras"))
    print(f"\n=== {name}: {npro} products, p = {p}", flush=True)
    prev = None
    for eps in EPS:
        t0 = time.time()
        pb = Problem(p, eps, a["degs"], family=a["family"], extras=a.get("extras"), cancel=60.0)
        cands = []
        base = starts_quintic(K) if a["degs"][0] == 2 else starts_cubic(K)
        if a["family"] == "tt":
            f, m, l = TT_COEF.get(p, TT_COEF[-0.3])
            cands.append(("prev-agent", pb.init([MUON] * K, [f] + [m] * (K - 2) + [l])))
            for sname, H in base.items():
                cands.append((sname, pb.init(H)))
        else:
            for sname, H in base.items():
                cands.append((sname, pb.init(H)))
        if prev is not None:
            cands.append(("cont", prev))
            for j in range(2):
                cands.append((f"cont~{j}", prev * (1 + 0.03 * rng.standard_normal(len(prev)))))
        best = None
        for sname, th0 in cands:
            try:
                th = pb.solve(th0)
                if a["family"] == "lc":
                    th, _ = pb.polish_lp(th)
                d = describe(pb, th, f"{name} eps={eps:.1e} [{sname}]")
            except Exception as ex:
                continue
            if not np.isfinite(d["hi"]) or d["fmin"] < -1e-3 or d["ymax"] > 3.0 or d["peak"] > 1.4 * eps ** p * d["hi"]:
                continue
            w = d["hi"] / d["lo"]
            if best is None or w < best[0]:
                best = (w, th, d)
        if best is None:
            print(f"   eps={eps:.1e}: nothing admissible", flush=True)
            continue
        w, th, d = best
        prev = th
        print("  ", fmt(d), f"({time.time() - t0:.0f}s)", flush=True)
        H, rest = pb.split(th)
        out[f"{name}_eps{eps:.1e}"] = dict(arch=name, products=npro, eps=eps, p=p, H=[list(map(float, h)) for h in H],
                                           out=list(map(float, rest)), lo=d["lo"], hi=d["hi"], half=d["half"], kappa=d["kappa"])
    tag = "" if not only else "_" + "-".join(only)
    json.dump(out, open(__file__.replace("s3_joint.py", f"joint_p{p}{tag}.json"), "w"), indent=1)
