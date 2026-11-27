"""Gram LC (optim/muon.py sv_power_gram, lc_steps='gram') on real momentum matrices in bfloat16, TF32 and float32, on a
GPU:

    python scripts/lc/gram_lc_check.py <out.npz> <momentum dump .pt> ...   (from the code dir; analysis/negp_fast/*.pt)

Values read off along the singular vectors of N (u_i^T Z v_i) at s = sigma / |N|_S8, as lc5m_check.py. Per precision:
the fraction of singular values between Muon's output (bfloat16, as trained) and s^p; the relative error against the
method's float64 scalar map at s, by decade of s; the noise floor (the largest s at which the median relative error in a
quarter decade still reaches 10%; below it rounding dominates); |Z - Z64|_F / |Z64|_F over the whole matrix (Z64 =
U g(S) V^T). LC5m in bfloat16 (as trained) for comparison. Then ms per call for the 14M GPT's shapes, one matrix and the
stacks of optim.muon.fast_update, for Muon, LC5m and Gram LC (CUDA events, median of 50), and the training step with
the fast path (fast_update_check.profile_step)."""
import json
import os
import sys
import numpy as np
import torch
from optsplit.optim import muon as M

P = -0.3
MODES = ("bfloat16", "tf32", "float32")
dev = "cuda"


def gram_map(s, coeffs=None):
    steps, w, d = coeffs or M.GRAM[P]
    c = [s * s + d]
    for a, b in steps:
        c.append(a * c[-1] + b * c[-1] ** 2)
    return s * (w[0] + sum(wk * ck for wk, ck in zip(w[1:], c)))


def lc5m_map(s):
    steps, w = M.LC[P, "5m"]
    ys = [s]
    for a, b, c in steps:
        ys.append(ys[-1] * (a + b * ys[-1] ** 2 + c * ys[-1] ** 4))
    return sum(wk * y for wk, y in zip(w, ys))


def gram(N, mode, coeffs=None):
    M.GRAM_DTYPE, saved = mode, M.GRAM[P]
    M.GRAM[P] = coeffs or saved
    try:
        return M.sv_power_gram(N, P)
    finally:
        M.GRAM[P] = saved


# other sets of gram_lc_coeffs.json, checked in float32: GRAM_SETS=key,key,...
_sets = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gram_lc_coeffs.json")))
SETS = {k: (_sets[k]["steps"], _sets[k]["weights"], _sets[k]["delta"])
        for k in os.environ.get("GRAM_SETS", "").split(",") if k}


def methods():
    out = {f"gram {m}": (lambda N, m=m: gram(N, m), gram_map) for m in MODES}
    for k, c in SETS.items():
        out[f"{k} float32"] = (lambda N, c=c: gram(N, "float32", c), lambda s, c=c: gram_map(s, c))
    out["lc5m bfloat16"] = (lambda N: M.sv_power_lc(N, P, "5m"), lc5m_map)
    return out


def collect(paths):
    rows = {k: [] for k in ("s", "muon", *(f"{k} {q}" for k in methods() for q in ("z", "g")))}
    whole = {k: [] for k in methods()}
    for fn in paths:
        for N in torch.load(fn, map_location=dev, weights_only=False)["final"].values():
            N = N.float()
            U, S, Vh = torch.linalg.svd(N.double(), full_matrices=False)
            s = S / (S ** 8).sum() ** 0.125
            read = lambda Z: ((U.mT @ Z.double()) * Vh).sum(1)
            rows["s"].append(s)
            rows["muon"].append(read(M.zeropower_via_newtonschulz5(N, 5)))
            for k, (f, g) in methods().items():
                Z, g64 = f(N).double(), g(s)
                rows[f"{k} z"].append(read(Z))
                rows[f"{k} g"].append(g64)
                Z64 = (U * g64) @ Vh
                whole[k].append(((Z - Z64).norm() / Z64.norm()).item())
    return {k: torch.cat(v).cpu().numpy() for k, v in rows.items()}, whole


def report(r, whole):
    s, mu, ex = r["s"], r["muon"], r["s"] ** P
    lo, hi = np.minimum(mu, ex), np.maximum(mu, ex)
    print(f"{len(whole['gram float32'])} matrices, {len(s)} singular values, s in [{s.min():.1e}, {s.max():.3f}]; "
          f"fraction of s below 1e-2 / 1e-3 / 1e-4: {(s < 1e-2).mean():.3f} / {(s < 1e-3).mean():.3f} / "
          f"{(s < 1e-4).mean():.4f}")
    dec = [(a, 10 * a) for a in 10.0 ** np.arange(np.floor(np.log10(s.min())), 0)]
    print(f"{'method':<32} {'between':>8} {'|Z-Z64|/|Z64| med':>18} {'max':>6} {'floor':>8}   median relative error "
          f"against the float64 map by decade of s: " + " ".join(f"{a:.0e}" for a, _ in dec))
    floors = {}
    for k in whole:
        z, g = r[f"{k} z"], r[f"{k} g"]
        between = np.mean((z >= lo * (1 - 1e-3)) & (z <= hi * (1 + 1e-3)))
        rel = np.abs(z - g) / np.abs(g)
        q = np.floor(np.log10(s) * 4) / 4   # quarter decades
        bad = [b for b in np.unique(q) if np.median(rel[q == b]) >= 0.1]
        floors[k] = 10 ** (max(bad) + 0.25) if bad else s.min()
        per = " ".join(f"{np.median(rel[(s >= a) & (s < b)]):.0e}" if ((s >= a) & (s < b)).any() else "  -  "
                       for a, b in dec)
        print(f"{k:<32} {between:>8.4f} {np.median(whole[k]):>18.2e} {max(whole[k]):>6.2f} {floors[k]:>8.1e}   {per}")
    return floors


def timing(n=50):
    shapes = {"384x384": (384, 384), "1536x384": (1536, 384), "384x1536": (384, 1536)}
    stack = {"384x384": 24, "1536x384": 6, "384x1536": 6}
    fns = {"muon": lambda N: M.zeropower_via_newtonschulz5(N, 5), "lc5m": lambda N: M.sv_power_lc(N, P, "5m"),
           **{f"gram {m}": (lambda N, m=m: gram(N, m)) for m in MODES}}
    for k, c in SETS.items():
        fns[f"{k} float32"] = lambda N, c=c: gram(N, "float32", c)
    print(f"\nms per call ({torch.cuda.get_device_name()}): one matrix | the stack of fast_update (24, 6, 6 matrices); "
          f"per step: 36 single calls | 3 stacked calls")
    print(f"{'method':<32} " + " ".join(f"{k:>9}" for k in shapes) + " | " + " ".join(f"{k:>9}" for k in shapes)
          + f" | {'36 calls':>9} {'3 stacks':>9}")
    res = {}
    for name, f in fns.items():
        one, many = [], []
        for k, sh in shapes.items():
            for batch, acc in ((None, one), (stack[k], many)):
                N = torch.randn(*((batch,) if batch else ()), *sh, device=dev)
                for _ in range(5):
                    f(N)
                t = []
                for _ in range(n):
                    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    a.record(); f(N); b.record()
                    torch.cuda.synchronize()
                    t.append(a.elapsed_time(b))
                acc.append(float(np.median(t)))
        per_step = 24 * one[0] + 6 * one[1] + 6 * one[2]
        res[name] = (one, many, per_step, sum(many))
        print(f"{name:<32} " + " ".join(f"{x:>9.3f}" for x in one) + " | " + " ".join(f"{x:>9.3f}" for x in many)
              + f" | {per_step:>9.2f} {sum(many):>9.2f}")
    return res


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    r, whole = collect(sys.argv[2:])
    floors = report(r, whole)
    t = timing()
    np.savez(sys.argv[1], **{k.replace(" ", "_"): v for k, v in r.items()},
             **{f"whole_{k.replace(' ', '_')}": np.array(v) for k, v in whole.items()},
             **{f"floor_{k.replace(' ', '_')}": v for k, v in floors.items()},
             **{f"ms_{k.replace(' ', '_')}": np.array([*v[0], *v[1], v[2], v[3]]) for k, v in t.items()})
    M.GRAM_DTYPE = "float32"
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from fast_update_check import profile_step
    print(f"\ntraining step, fast path, one process (ms): " + ", ".join(
        f"{k} {profile_step(o + ['optim.muon.fast_update=true'])['train']:.1f}" for k, o in (
            ("muon", []), ("lc5m", ["optim.muon.sv_power=-0.3", "optim.muon.lc_steps=5m"]),
            ("gram float32", ["optim.muon.sv_power=-0.3", "optim.muon.lc_steps=gram"]))))
