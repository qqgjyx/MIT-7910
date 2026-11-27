"""Shared pieces for the negative-power Muon study: scalar maps (float64) and fits.

Scalar picture. X_0 = N / c has singular values s (c = normalization). One Newton-Schulz step with polynomial
h(m) = h0 + h1 m + h2 m^2 (+ h3 m^3) is y <- y h(y^2). Every method below is an odd matrix function of X_0, so it is
described by a scalar map s -> f(s) on its singular values.
Products are counted as matrix-matrix products of any shape (rectangular ones are flagged separately in the matrix code).
"""
import numpy as np
from scipy.optimize import linprog

MUON = (3.4445, -4.7750, 2.0315)
# previous agent's two-track output coefficients (first, middle, last) for the Muon X-track
TT_COEF = {
    -0.3: ((4.9125, -8.0324, 3.7790), (4.9915, -7.8279, 3.4157), (2.1740, -2.4723, 0.9797)),
    -0.5: ((6.1049, -10.8787, 5.4076), (6.3927, -10.5965, 4.7000), (2.1908, -2.9743, 1.2786)),
}
# Polar Express (Amsel et al. 2025), l = 1e-3, as published (no safety factor)
PE = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
]


def poly(c, m):
    out = np.zeros_like(m) + c[-1]
    for ci in c[-2::-1]:
        out = out * m + ci
    return out


def iterates(sig, H):
    """[y_0, ..., y_K] with y_0 = sig, y_{k+1} = y_k h_k(y_k^2)."""
    ys = [np.asarray(sig, dtype=np.float64)]
    for h in H:
        y = ys[-1]
        ys.append(y * poly(h, y * y))
    return ys


def two_track(sig, H, S):
    """z_K of the two-track scheme: z_{k+1} = z_k s_k(y_k^2); the X update after the last step is not needed."""
    ys = iterates(sig, H[:-1])
    z = np.asarray(sig, dtype=np.float64).copy()
    for y, s in zip(ys, S):
        z = z * poly(s, y * y)
    return z


def step_products(h):
    """Products of one X step y <- y h(y^2): Gram, powers of the Gram up to deg(h), and the application to X."""
    return 1 + (len(h) - 2) + 1


def lp_minimax(B, target, lb=None):
    """Coefficients c minimizing max_i |(B c)_i / target_i - 1|. B: (npts, nbasis). Returns c, error."""
    V = B / target[:, None]
    n = B.shape[1]
    cost = np.zeros(n + 1); cost[-1] = 1.0
    A = np.block([[V, -np.ones((len(V), 1))], [-V, -np.ones((len(V), 1))]])
    b = np.concatenate([np.ones(len(V)), -np.ones(len(V))])
    bounds = [(None, None)] * n + [(0, None)] if lb is None else list(lb) + [(0, None)]
    r = linprog(cost, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    if not r.success:
        return None, np.inf
    return r.x[:n], r.x[-1]


def band(f, sig, p, lo, hi=1.0):
    """f / (kappa sig^p) on [lo, hi], kappa = log-centre: returns kappa, min, max of the ratio."""
    sel = (sig >= lo) & (sig <= hi)
    r = f[sel] / sig[sel] ** p
    kappa = np.sqrt(r.min() * r.max())
    return kappa, r.min() / kappa, r.max() / kappa


def exact_weights(srel, p, floor=1e-6):
    return np.maximum(srel, floor) ** p


def unit(w):
    return w / np.sqrt(np.mean(w ** 2))


def err_vs_exact(w, srel, p, floor=1e-6):
    """Relative Frobenius error of the update with weights w against the exact clamped update, both rescaled to the
    same Frobenius norm (the repo's convention). Equal to the matrix error when the method is an exact spectral map."""
    t = unit(exact_weights(srel, p, floor))
    return np.sqrt(np.mean((unit(w) - t) ** 2))


def err_head(w, srel, p, cut=1e-3):
    """Same, restricted to the singular values above `cut` (relative to the largest), best common scale."""
    sel = srel >= cut
    t = srel[sel] ** p
    k = np.sum(w[sel] * t) / np.sum(w[sel] ** 2)
    return np.sqrt(np.sum((k * w[sel] - t) ** 2) / np.sum(t ** 2))


def synth_spectrum(kind, n=384, seed=0):
    """Previous agent's synthetic spectra (relative singular values, descending)."""
    rng = np.random.default_rng(seed)
    u = (np.arange(n) + 0.5) / n
    if kind == "late":
        lg = np.interp(u, [0, 0.02, 0.5, 0.84, 0.92, 1.0], [0, -0.6, -1.4, -3.0, -4.0, -6.0])
    elif kind == "early":
        lg = np.interp(u, [0, 0.02, 0.2, 0.5, 0.85, 1.0], [0, -0.7, -2.0, -3.0, -4.0, -6.0])
    elif kind == "flat":
        g = np.linalg.svd(rng.standard_normal((n, n)), compute_uv=False)
        return g / g[0]
    else:
        raise ValueError(kind)
    lg = lg + 0.03 * rng.standard_normal(n)
    s = np.sort(10.0 ** lg)[::-1]
    return s / s[0]


def make_matrix(s, shape, seed=0, dtype=np.float32):
    rng = np.random.default_rng(seed)
    n, m = shape
    k = min(n, m)
    U, _ = np.linalg.qr(rng.standard_normal((n, k)))
    V, _ = np.linalg.qr(rng.standard_normal((m, k)))
    return ((U * s) @ V.T).astype(dtype)
