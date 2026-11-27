"""LC5m on real momentum matrices, as in training (Muon and LC iterates in bfloat16, the LC sum in float32): is each
singular value of the LC5m output between Muon's output and the exact s^p? Values are read off along the singular
vectors of N (u_i^T Z v_i, i.e. the SVD of Z matched to N's), before the final rescale, with s = sigma / |N|_S8 and the
exact value s^p. Also LC5 for comparison, and rho = |N|_S8 / |N|_F.
    python scripts/lc/lc5m_check.py /orcd/pool/014/conchell/optsplit/analysis/negp_fast/*.pt   (from the code dir, GPU)"""
import sys
import numpy as np
import torch
from optsplit.optim.muon import sv_power_lc, zeropower_via_newtonschulz5

P = -0.3
SC = 1.2024 ** (1 / P)  # Muon's largest value equals s^p here
dev = "cuda" if torch.cuda.is_available() else "cpu"
rows = {k: [] for k in ("s", "muon", "lc5m", "lc5", "rho")}
for fn in sys.argv[1:]:
    for G in torch.load(fn, weights_only=False)["final"].values():
        N = G.float().to(dev)
        U, S, Vh = torch.linalg.svd(N.double(), full_matrices=False)
        s8 = (S ** 8).sum() ** 0.125
        read = lambda Z: ((U.mT @ Z.double()) * Vh).sum(1).cpu().numpy()
        rows["s"].append((S / s8).cpu().numpy())
        rows["muon"].append(read(zeropower_via_newtonschulz5(N, 5)))
        rows["lc5m"].append(read(sv_power_lc(N, P, "5m")))
        rows["lc5"].append(read(sv_power_lc(N, P, 5)))
        rows["rho"].append(np.full(len(S), (s8 / S.norm()).item()))
r = {k: np.concatenate(v) for k, v in rows.items()}
s, mu, ex = r["s"], r["muon"], r["s"] ** P
lo, hi = np.minimum(mu, ex), np.maximum(mu, ex)
print(f"{len(rows['s'])} matrices from {len(sys.argv) - 1} files, {len(s)} singular values, on {dev}; "
      f"rho in [{r['rho'].min():.3f}, {r['rho'].max():.3f}], s in [{s.min():.1e}, {s.max():.3f}]")
for name in ("lc5m", "lc5"):
    g = r[name]
    viol = np.maximum(np.maximum((lo - g) / np.abs(lo), (g - hi) / np.abs(hi)), 0)
    i = np.argmax(viol)
    closed = np.log(np.maximum(g, 1e-30) / mu) / np.log(ex / mu)
    print(f"{name}: between Muon and s^p for {np.mean(viol == 0):.4f} of the singular values; worst violation "
          f"{viol.max():.3g} at s = {s[i]:.3g} (Muon {mu[i]:.4g}, {name} {g[i]:.4g}, s^p {ex[i]:.4g})")
    for a, b in [(1e-12, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 0.1), (0.1, SC), (SC, 1.01)]:
        sel = (s >= a) & (s < b)
        if sel.any():
            print(f"   s in [{a:.0e}, {b:.3g}): {sel.sum():6d} values, between {np.mean(viol[sel] == 0):.4f}, worst "
                  f"{viol[sel].max():.3g}, median gap closed {np.median(closed[sel]):.3f}")
