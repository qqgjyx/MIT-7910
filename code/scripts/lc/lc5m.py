"""LC5m: LC (sum_k w_k X_k over K quintic steps, as in opt.py 'lc') with a Muon floor.

Hard constraint on all of (0, 1]: g(s) lies between Muon's value and s^p,
    min(E(s), s^p) <= g(s) <= max(E(s), s^p),   E(s) = max_{t <= s} phi_Muon(t),
where phi_Muon is Muon's scalar map (5 steps of its coefficients, input normalized by |N|_F) and s = sigma / |N|_S8.
Muon sees rho s with rho = |N|_S8 / |N|_F <= 1, so E is Muon's value in the worst case over rho. Near 0 every map is
linear and the constraint is slope_g >= 3.4445^5 (Muon's slope at rho = 1), imposed exactly. E crosses s^p once, at
s_c = max(phi_Muon)^(1/p) = 0.541, where the band closes: g(s_c) = s_c^p.

Objective, under the constraint: g as close to s^p as possible on the whole range, measured by the fraction of the
log gap between Muon's value and s^p that g leaves open, |log(g / s^p)| / |log(E / s^p)|, averaged over a log-uniform
grid of [1e-6, 1] (a weighted integral of |log(g / s^p)|; below the cutoff g is linear, so it also asks for a large
slope at 0).
Least squares on all coefficients (objective and constraint as residuals; iterates bounded, cancellation capped as in
opt.py), then, for fixed steps, the exact convex problem in the weights (linear programs with tangent cuts). The steps are then rescaled so every
iterate peaks at 1, rounded to 5 digits, and the weights solved again for the rounded steps.
    uv run --no-project --with numpy --with scipy python lc5m.py [K]   -> lc5m_coeffs.json"""
import json
import os
import sys
import time
import numpy as np
from scipy.optimize import least_squares, linprog
from lib import MUON, iterates, poly

np.seterr(all="ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
P = -0.3
SLOPE_MUON = MUON[0] ** 5
_FINE = np.logspace(-12, 0, 2_000_001)
_ENV = np.maximum.accumulate(iterates(_FINE, [MUON] * 5)[-1])
EMAX = _ENV[-1]
SC = EMAX ** (1 / P)
LOW = 1e-6            # objective on [LOW, 1]
YMAX, CANCEL, MARGIN = 2.5, 60.0, 1e-3


def envelope(s):
    """Muon's value for s = sigma / |N|_S8, worst case over rho (linear below 1e-12)."""
    return np.where(s < 1e-12, SLOPE_MUON * s, np.interp(np.log(np.maximum(s, 1e-12)), np.log(_FINE), _ENV))


def bounds(s):
    E, T = envelope(s), s ** P
    return np.minimum(E, T), np.maximum(E, T)


def basis(H, s):
    ys = [s]
    for h in H:
        ys.append(np.clip(ys[-1] * poly(h, ys[-1] ** 2), -1e4, 1e4))
    return np.stack(ys, 1)


def slope_coeffs(H):
    """Slope at 0 of each iterate: prod_{j<k} a_j."""
    return np.concatenate([[1.0], np.cumprod([h[0] for h in H])])


def gap_closed(g, s):
    """Fraction of the log gap between Muon's value E and s^p that g closes (1: on s^p, 0: on Muon's value)."""
    E = envelope(s)
    return np.log(g / E) / np.log(s ** P / E)


class Fit:
    def __init__(self, K=5, smin=SLOPE_MUON):
        self.K, self.smin = K, max(smin, SLOPE_MUON)                                   # smin: a floor on the slope
        s = np.logspace(np.log10(LOW), 0, 600)                                        # objective grid
        D = np.abs(np.log(s ** P / envelope(s)))
        keep = D > 0.01                                                               # not where the band closes
        self.s, self.v = s[keep], 1 / D[keep]
        self.cg = np.unique(np.concatenate([np.logspace(-10, 0, 4000), np.linspace(0.3, 1, 1500), [SC]]))
        self.lo, self.hi = bounds(self.cg)
        self.m = np.minimum(MARGIN, (self.hi / self.lo - 1) / 4)                      # margin, 0 where the band closes
        self.T, self.Tc = self.s ** P, self.cg ** P
        self.top = self.s >= SC

    def split(self, th):
        return [th[3 * k:3 * k + 3] for k in range(self.K)], th[3 * self.K:]

    def objective(self, H, w):
        return np.mean(self.v * np.abs(np.log(np.maximum(basis(H, self.s) @ w, 1e-300) / self.T)))

    def resid(self, th, lam=10.0):
        H, w = self.split(th)
        B, Bc = basis(H, self.s), basis(H, self.cg)
        g, gc = B @ w, Bc @ w
        lg = np.log(np.maximum(g, 1e-30) / self.T)
        out = [(np.sqrt(self.v * np.abs(lg) + 1e-4) - 1e-2) / len(g) ** 0.5,
               lam * np.maximum(self.lo * (1 + 2 * self.m) - gc, 0) / self.lo / len(gc) ** 0.5,
               lam * np.maximum(gc - self.hi * (1 - 2 * self.m), 0) / self.hi / len(gc) ** 0.5,
               [lam * max(self.smin * (1 + 2 * MARGIN) - slope_coeffs(H) @ w, 0) / SLOPE_MUON],
               3.0 * np.maximum(np.abs(Bc[:, 1:]) - YMAX, 0).ravel() / len(gc) ** 0.5,
               0.3 * np.maximum(np.abs(B * w).sum(1) / np.abs(g) - CANCEL, 0) / len(g) ** 0.5]
        return np.concatenate(out)

    def weights(self, H, iters=30):
        """Exact weights for fixed steps. The problem is convex in w: the constraint and the cancellation cap are linear,
        -log(g/s^p) is convex (on [s_c, 1], where g >= s^p, |log(g/s^p)| is replaced by its upper bound g/s^p - 1).
        Solved as linear programs (HiGHS) with tangent cuts of -log added at each solution until the gap is < 1e-7."""
        n, K1 = len(self.s), self.K + 1
        low = np.flatnonzero(~self.top)
        nl = len(low)
        X = basis(H, self.s) / self.T[:, None]                  # x = g / s^p = X w
        Bc = basis(H, self.cg) / self.Tc[:, None]
        pin = self.cg == SC                                     # zero-width band: an equality
        # variables: w (K1), a >= |w| (K1), t >= -log(x) at the low points (nl)
        nv = 2 * K1 + nl
        cost = np.zeros(nv)
        cost[2 * K1:] = self.v[low] / n
        cost[:K1] = (self.v[self.top][:, None] * X[self.top]).sum(0) / n
        Z = lambda r, c: np.zeros((r, c))
        rows = [np.hstack([-Bc[~pin], Z((~pin).sum(), K1 + nl)]), np.hstack([Bc[~pin], Z((~pin).sum(), K1 + nl)]),
                np.hstack([-slope_coeffs(H)[None] / SLOPE_MUON, Z(1, K1 + nl)]),
                np.hstack([np.eye(K1), -np.eye(K1), Z(K1, nl)]), np.hstack([-np.eye(K1), -np.eye(K1), Z(K1, nl)]),
                np.hstack([-CANCEL * X, np.abs(X), Z(n, nl)])]
        rhs = [-(self.lo * (1 + self.m) / self.Tc)[~pin], (self.hi * (1 - self.m) / self.Tc)[~pin],
               [-(1 + MARGIN) * self.smin / SLOPE_MUON],
               np.zeros(K1), np.zeros(K1), np.zeros(n)]
        A_eq = np.hstack([Bc[pin], Z(1, K1 + nl)])
        b_eq = [EMAX / SC ** P]
        xmin = (envelope(self.s) / self.T)[low]
        cuts = [xmin * (1 / xmin) ** q for q in np.linspace(0, 1, 12)]   # tangent points between Muon and s^p
        w = None
        for it in range(iters):
            cr, cb = [], []
            for x0 in cuts:   # t_i >= -log(x0) - (x_i - x0) / x0
                M = Z(nl, nv)
                M[:, :K1] = -X[low] / x0[:, None]
                M[np.arange(nl), 2 * K1 + np.arange(nl)] = -1
                cr.append(M); cb.append(np.log(x0) - 1)
            r = linprog(cost, A_ub=np.vstack(rows + cr), b_ub=np.concatenate(rhs + cb), A_eq=A_eq, b_eq=b_eq,
                        bounds=[(None, None)] * K1 + [(0, None)] * K1 + [(None, None)] * nl, method="highs")
            if not r.success:
                return w
            w = r.x[:K1]
            x = X[low] @ w
            gap = (self.v[low] * (-np.log(x) - r.x[2 * K1:])).sum() / n
            if gap < 1e-7:
                break
            cuts.append(x)
        return w

    def solve(self, th0, lams=(10, 100, 1e3, 1e4, 1e5), nfev=600):
        """Least squares with the constraint as a penalty of growing weight (and twice the margin)."""
        th = np.asarray(th0, float)
        for lam in lams:
            sol = least_squares(self.resid, th, args=(lam,), method="trf", x_scale="jac", max_nfev=nfev)
            if np.all(np.isfinite(sol.x)):
                th = sol.x
        return th


def check(H, w, lo_s=1e-12, n=3_000_001):
    """Dense float64 check of the constraint on [lo_s, 1] (E as the running max on the same grid) and of the slope."""
    s = np.unique(np.concatenate([np.logspace(np.log10(lo_s), 0, n), [SC]]))
    ys = iterates(s, H)
    g = sum(wk * y for wk, y in zip(w, ys))
    E = np.maximum.accumulate(iterates(s, [MUON] * 5)[-1])
    lo, hi = np.minimum(E, s ** P), np.maximum(E, s ** P)
    viol = np.maximum(np.maximum((lo - g) / lo, (g - hi) / hi), 0)
    terms = np.abs(np.stack([wk * y for wk, y in zip(w, ys)]))
    return dict(worst=viol.max(), at=s[np.argmax(viol)], frac_bad=(viol > 0).mean(), slope=slope_coeffs(H) @ w,
                ymax=max(np.abs(y).max() for y in ys[1:]), cancel=(terms.sum(0) / np.abs(g))[s >= LOW].max())


def band(H, w, eps):
    s = np.logspace(np.log10(eps), 0, 200001)
    r = sum(wk * y for wk, y in zip(w, iterates(s, H))) / s ** P
    return np.sqrt(r.max() / r.min()) - 1


def rnd(x, d=5):
    return np.array([float(f"{v:.{d}g}") for v in x])


def gauge(H):
    """Steps rescaled so that every iterate peaks at 1 on (0, 1] (the map is unchanged up to the weights)."""
    ys = iterates(np.logspace(-9, 0, 900001), H)
    a = [1.0] + [np.sign(y[0]) * np.abs(y).max() for y in ys[1:]]
    return [np.array([h[0], h[1] * a[k] ** 2, h[2] * a[k] ** 4]) * a[k] / a[k + 1] for k, h in enumerate(H)]


def starts(K):
    phi = lambda a, r: (a, (5 - r) / 2 - 2 * a, a - (3 - r) / 2)
    S = {"muon": [MUON] * K, "grad": [phi(a, 0.3) for a in np.linspace(4.5, 2.2, K)]}
    for a in [2.4, 3.0, 3.6, 4.2]:
        S[f"phi{a}"] = [phi(a, a ** P)] * K
    for key, e in json.load(open(os.path.join(HERE, "final_coeffs.json"))).items():
        if key.startswith(f"p{P}_lc{K}_eps"):
            S[key] = e["steps"]
    return S


def summary(H, w):
    s = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-6])
    g = basis(H, s) @ w
    return dict(zip([f"{x:g}" for x in s], map(float, gap_closed(g, s))))


if __name__ == "__main__":
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    fit = Fit(K)
    rng = np.random.default_rng(0)
    results = []
    for name, H0 in starts(K).items():
        t0 = time.time()
        H0 = [np.asarray(h, float) for h in H0]
        w0 = fit.weights(H0)
        if w0 is None:  # infeasible start: weights of a plain fit
            w0 = np.linalg.lstsq(basis(H0, fit.s) / fit.T[:, None], np.ones(len(fit.s)), rcond=None)[0]
        th = np.concatenate([np.concatenate(H0), w0])
        for rep in range(2):
            th = fit.solve(th)
            H, _ = fit.split(th)
            w = fit.weights(H)
            if w is None:
                break
            th = np.concatenate([np.concatenate(H), w])
        if w is None:
            print(f"{name}: infeasible after the fit ({time.time() - t0:.0f}s)", flush=True)
            continue
        J = fit.objective(H, w)
        results.append((J, name, H, w))
        print(f"{name}: gap left open {J:.4f}, slope {slope_coeffs(H) @ w:.0f}, gap closed {summary(H, w)} "
              f"({time.time() - t0:.0f}s)", flush=True)
    results.sort(key=lambda r: r[0])
    out = {}
    for J, name, H, w in results[:3]:
        Hr = [rnd(h) for h in gauge(H)]
        wr = fit.weights(Hr)
        if wr is None:
            print(f"{name}: rounded steps infeasible")
            continue
        wr = rnd(wr, 10)
        c = check(Hr, wr)
        J = fit.objective(Hr, wr)
        print(f"{name} rounded: gap left open {J:.4f}; constraint worst violation {c['worst']:.1e} at {c['at']:.4g} "
              f"(fraction of grid {c['frac_bad']:.1e}); slope {c['slope']:.0f} vs Muon {SLOPE_MUON:.0f}; max|X_k| "
              f"{c['ymax']:.3f}; cancellation {c['cancel']:.1f}; band +-{band(Hr, wr, 3e-3) * 100:.1f}% on [3e-3, 1], "
              f"+-{band(Hr, wr, 1e-2) * 100:.1f}% on [1e-2, 1]; gap closed {summary(Hr, wr)}", flush=True)
        out[f"lc{K}m_{name}"] = dict(p=P, K=K, products=3 * K, start=name, objective=J, low=LOW,
                                     steps=[list(map(float, h)) for h in Hr], weights=list(map(float, wr)),
                                     gap_closed=summary(Hr, wr), **{k: float(v) for k, v in c.items()})
    out = {f"lc{K}m": min(out.values(), key=lambda e: e["objective"]), **out}     # the set used (muon.py)
    json.dump(out, open(os.path.join(HERE, f"lc{K}m_coeffs.json"), "w"), indent=1)
