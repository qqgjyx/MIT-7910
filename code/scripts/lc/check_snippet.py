"""Check of the code exactly as printed in the report (coefficients typed by hand, 5 digits), on real late matrices:
scalar map read off by SVD, band on [eps, 1] of the normalized singular values, and errors against the exact update."""
import glob
import os
import sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))

LC5 = dict(eps=3e-3,
           steps=[(3.5969, -7.7746, 4.8280), (3.5004, -7.1445, 4.1116), (3.4189, -6.6980, 3.8465),
                  (3.0402, -4.9130, 2.8709), (1.8013, -1.6495, 0.84815)],
           w=[-0.88949, -1.2850, -1.6349, -0.90775, -11.758, 16.278])
LC6 = dict(eps=1e-3,
           steps=[(3.5791, -7.6099, 4.4535), (3.1263, -5.3070, 3.1688), (3.3447, -6.2419, 3.3154),
                  (3.4107, -6.5906, 3.5226), (3.0637, -5.0036, 2.8968), (2.7706, -3.7884, 2.0046)],
           w=[-1.0761, -0.40821, -1.7443, -2.3533, -1.2355, -3.8680, 10.337])


def negp(G, steps, w):
    """sum_k w_k X_k over quintic Newton-Schulz iterates; 3 matrix products per step, none for the combination."""
    tall = G.size(-2) > G.size(-1)
    X = G.mT if tall else G
    X = X / (X.norm() + 1e-7)
    out = 0
    for k, (a, b, c) in enumerate(steps):
        A = X @ X.mT
        A2 = A @ A
        if k == 0:                      # Schatten-8 norm, from the first squared Gram matrix: no extra product
            s = A2.norm() ** 0.25
            X, A, A2 = X / s, A / s ** 2, A2 / s ** 4
        out = out + w[k] * X
        X = a * X + (b * A + c * A2) @ X
    out = out + w[-1] * X
    return out.mT if tall else out


p = -0.3
r = torch.load(sorted(glob.glob(os.path.join(HERE, "real", "*main__muon_d0__s07000.pt")))[0], weights_only=False)
for name, cfg in [("LC5", LC5), ("LC6", LC6)]:
    lo, hi, eh, ee = [], [], [], []
    for mname, G in r["final"].items():
        G = G.float()
        U, S, Vh = torch.linalg.svd(G.double(), full_matrices=False)
        Z = negp(G, cfg["steps"], cfg["w"]).double()
        C = U.mT @ Z @ Vh.mT
        f = torch.diagonal(C)
        s8 = (S ** 8).sum() ** 0.125
        sn = S / s8
        sel = sn >= cfg["eps"]
        ratio = (f[sel] / sn[sel] ** p)
        lo.append(ratio.min().item()); hi.append(ratio.max().item())
        srel = (S / S[0]).clamp_min(1e-6)
        E = (U * srel ** p) @ Vh
        k = min(G.shape)
        un = lambda M: M * (k ** 0.5 / M.norm())
        ee.append(((un(Z) - un(E)).norm() / un(E).norm()).item())
        h = (S / S[0]) >= 1e-3
        Ch, T = C[h][:, h], torch.diag(srel[h] ** p)
        kap = (Ch * T).sum() / (Ch * Ch).sum()
        eh.append(((kap * Ch - T).norm() / T.norm()).item())
    print(f"{name}: f(s)/s^p on [eps={cfg['eps']:g}, 1] over 12 real matrices (float32): min {min(lo):.3f}, max {max(hi):.3f};"
          f" e_exact median {np.median(ee):.2f}, e_head median {np.median(eh):.2f}")
