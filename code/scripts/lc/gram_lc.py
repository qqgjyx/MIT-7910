"""Gram LC: the LC5m problem solved with iterates of the Gram matrix instead of X.

X_0 = N / |N|_S8 (m x n, m <= n) has singular values s in (0, 1]; A_0 = X_0 X_0^T has eigenvalues s^2. Iterates
    C_0 = A_0 + d I,   C_{k+1} = a_k C_k + b_k C_k^2    (k = 0 .. P-1; C_1 uses A_0^2, already needed for |N|_S8),
    S = w_I I + sum_k w_k C_k,   output S X_0,       singular values f(s) = s (w_I + sum_k w_k c_k(s^2 + d)).
Products: X X^T and S X_0 (m x n x m), and P Gram products (m x m x m). With a_k in (2, 3) and b_k = -a_k^2 / 4 the
scalar step is the logistic map with r = a_k (in a x / 4): c_k grows like (prod a) l from 0 and settles at a fixed
point, so the iterates are bounded and smooth (no chaos below 3). For l < 0 it diverges, and rounding makes the
smallest eigenvalues of A_0 negative (about -1e-7 in float32): the shift d keeps them >= 0. Slope at 0: h(d).

Same constraint and objective as LC5m (lc5m.py): min(E, s^p) <= f <= max(E, s^p) on all of (0, 1] with E Muon's
envelope; objective the mean fraction of the log gap between E and s^p left open on [1e-6, 1]. Also h >= Muon's slope
and |c_k| <= YMAX for l in [-d, 0] (eigenvalues that rounding made negative). Least squares on all coefficients from
several starts, then exact weights for fixed steps (LP with tangent cuts, rows scaled by the bounds), steps rounded to
5 digits and weights solved again. No cancellation cap: S cancels w_I against the iterates at large s by design.
Weights capped by sum |w_k| <= wmax: rounding errors of the iterates (~eps of their norm, 1) reach the output as
eps sum |w_k|, against an output of ~1 for the largest singular values.
    uv run --no-project --with numpy --with scipy python gram_lc.py P:d:wmax ...
-> gram_lc_coeffs.json, keys gram<P>_d<d>_w<wmax>; the set in muon.py is copied to the key "gram"."""
import json
import os
import sys
import time
import numpy as np
from multiprocessing import Pool
from scipy.optimize import least_squares, linprog
from lc5m import EMAX, LOW, MARGIN, P, SC, SLOPE_MUON, bounds, envelope, gap_closed, rnd

np.seterr(all="ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
YMAX = 2.5


def chis(H, lam):
    """[c_0, ..., c_P] at c_0 = lam."""
    c = [np.asarray(lam, float)]
    for a, b in H:
        c.append(a * c[-1] + b * c[-1] ** 2)
    return c


def hbasis(H, lam):
    """Columns of h = f / s at c_0 = lam (= s^2 + d)."""
    c = chis(H, np.atleast_1d(np.asarray(lam, float)))
    return np.stack([np.ones_like(c[0])] + c, 1)


def basis(H, s, d):
    s = np.asarray(s, float)
    return s[:, None] * hbasis(H, s * s + d)


def gmap(e, s):
    return basis(e["steps"], s, e["delta"]) @ np.asarray(e["weights"])


class Fit:
    def __init__(self, nP, d, wmax):
        self.nP, self.d, self.wmax = nP, d, wmax
        s = np.logspace(np.log10(LOW), 0, 600)
        D = np.abs(np.log(s ** P / envelope(s)))
        keep = D > 0.01
        self.s, self.v = s[keep], 1 / D[keep]
        self.cg = np.unique(np.concatenate([np.logspace(-10, 0, 4000), np.linspace(0.3, 1, 1500),
                                            np.linspace(SC - 0.02, SC + 0.02, 801)]))
        self.lo, self.hi = bounds(self.cg)
        self.m = np.minimum(MARGIN, (self.hi / self.lo - 1) / 4)
        self.neg = np.concatenate([[0], d * np.logspace(-12, 0, 241)])   # c_0 for eigenvalues in [-d, 0]
        self.T = self.s ** P
        self.top = self.s >= SC

    def split(self, th):
        return th[:2 * self.nP].reshape(-1, 2), th[2 * self.nP:]

    def objective(self, H, w):
        return np.mean(self.v * np.abs(np.log(np.maximum(basis(H, self.s, self.d) @ w, 1e-300) / self.T)))

    def resid(self, th, lam):
        H, w = self.split(th)
        g, gc = basis(H, self.s, self.d) @ w, basis(H, self.cg, self.d) @ w
        lg = np.log(np.maximum(g, 1e-30) / self.T)
        hn = hbasis(H, self.neg) @ w
        C = np.stack(chis(H, np.concatenate([self.cg ** 2 + self.d, self.neg]))[1:], 1)
        return np.concatenate([
            (np.sqrt(self.v * np.abs(lg) + 1e-4) - 1e-2) / len(g) ** 0.5,
            lam * np.maximum(self.lo * (1 + 2 * self.m) - gc, 0) / self.lo / len(gc) ** 0.5,
            lam * np.maximum(gc - self.hi * (1 - 2 * self.m), 0) / self.hi / len(gc) ** 0.5,
            lam * np.maximum(SLOPE_MUON * (1 + 2 * MARGIN) - hn, 0) / SLOPE_MUON,
            [lam * max(np.abs(w).sum() / self.wmax - 1, 0)],
            3.0 * np.maximum(np.abs(C) - YMAX, 0).ravel() / len(C) ** 0.5])

    def solve(self, th, lams=(10, 1e3, 1e5), nfev=200):
        for lam in lams:
            sol = least_squares(self.resid, th, args=(lam,), method="trf", x_scale="jac", max_nfev=nfev)
            if np.all(np.isfinite(sol.x)):
                th = sol.x
        return th

    def weights(self, H, iters=60):
        """Exact weights for fixed steps (convex): as lc5m.Fit.weights, with each constraint row divided by its bound
        (the columns s c_k(s) span ~30 decades; unscaled rows fall under the LP's coefficient tolerance)."""
        n = len(self.s)
        X = basis(H, self.s, self.d) / self.T[:, None]
        K1 = X.shape[1]
        low = np.flatnonzero(~self.top)
        nl = len(low)
        nv, T0 = 2 * K1 + nl, 2 * K1                 # variables: w, a >= |w|, t (-log of g / s^p, cuts)
        Bc = basis(H, self.cg, self.d)
        Hn = hbasis(H, self.neg)
        pin = self.cg == SC
        cost = np.zeros(nv)
        cost[T0:] = self.v[low] / n
        cost[:K1] = (self.v[self.top][:, None] * X[self.top]).sum(0) / n
        Z = lambda r, c: np.zeros((r, c))
        I, one = np.eye(K1), np.ones((1, K1))
        rows = [np.hstack([-Bc[~pin] / self.lo[~pin, None], Z((~pin).sum(), K1 + nl)]),
                np.hstack([Bc[~pin] / self.hi[~pin, None], Z((~pin).sum(), K1 + nl)]),
                np.hstack([-Hn / SLOPE_MUON, Z(len(Hn), K1 + nl)]),
                np.hstack([I, -I, Z(K1, nl)]), np.hstack([-I, -I, Z(K1, nl)]), np.hstack([Z(1, K1), one, Z(1, nl)])]
        rhs = [-(1 + self.m)[~pin], (1 - self.m)[~pin], -np.full(len(Hn), 1 + MARGIN), np.zeros(2 * K1), [self.wmax]]
        A_eq, b_eq = np.hstack([Bc[pin] / EMAX, Z(1, K1 + nl)]), [1.0]
        xmin = (envelope(self.s) / self.T)[low]
        cuts = [xmin * (1 / xmin) ** q for q in np.linspace(0, 1, 12)]
        w = None
        for it in range(iters):
            cr, cb = [], []
            for x0 in cuts:   # t_i >= -log(x0) - (x_i - x0) / x0
                M = Z(nl, nv)
                M[:, :K1] = -X[low] / x0[:, None]
                M[np.arange(nl), T0 + np.arange(nl)] = -1
                cr.append(M)
                cb.append(np.log(x0) - 1)
            r = linprog(cost, A_ub=np.vstack(rows + cr), b_ub=np.concatenate(rhs + cb), A_eq=A_eq, b_eq=b_eq,
                        bounds=[(None, None)] * nv, method="highs")
            if not r.success:
                return w
            w = r.x[:K1]
            x = X[low] @ w
            if (self.v[low] * (-np.log(np.maximum(x, 1e-300)) - r.x[T0:])).sum() / n < 1e-7:
                break
            cuts.append(np.maximum(x, 1e-12))
        return w


def check(H, w, d, lo_s=1e-12, n=3_000_001):
    """Dense float64 check of the constraint on [lo_s, 1] (Muon's envelope on the same grid), slope at 0 (h(d)),
    iterate bound on eigenvalues in [-d, 1], and the smallest h on [-d, 0]."""
    from lib import MUON, iterates
    s = np.unique(np.concatenate([np.logspace(np.log10(lo_s), 0, n), np.linspace(0.3, 1, n // 3), [SC]]))
    g = basis(H, s, d) @ w
    E = np.maximum.accumulate(iterates(s, [MUON] * 5)[-1])
    lo, hi = np.minimum(E, s ** P), np.maximum(E, s ** P)
    viol = np.maximum(np.maximum((lo - g) / lo, (g - hi) / hi), 0)
    neg = np.concatenate([[0], d * np.logspace(-14, 0, 20001)])
    return dict(worst=viol.max(), at=s[np.argmax(viol)], frac_bad=(viol > 0).mean(), slope=(hbasis(H, d) @ w)[0],
                ymax=max(np.abs(c).max() for c in chis(H, np.concatenate([s * s + d, neg]))[1:]),
                hmin_neg=(hbasis(H, neg) @ w).min())


def summary(H, w, d):
    s = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-6])
    return dict(zip([f"{x:g}" for x in s], map(float, gap_closed(basis(H, s, d) @ w, s))))


def starts(nP, d):
    """Logistic steps whose product of slopes, prod a_k, is kappa / d: the last iterates turn over near c_0 = d."""
    S = {"laplace": [(2.0, -1.0)] * nP}
    for kappa in (1, 4, 16):
        a = min(max((kappa / d) ** (1 / nP), 1.9), 2.9)
        S[f"logistic{a:.3f}"] = [(a, -a * a / 4)] * nP
        r = np.linspace(1.15, 0.85, nP)
        S[f"graded{a:.3f}"] = [(x, -x * x / 4) for x in np.clip(a * r, 1.9, 2.95)]
    only = os.environ.get("GRAM_STARTS")   # comma-separated names: run only these starts
    return {k: v for k, v in S.items() if not only or k in only.split(",")}


def run_start(args):
    nP, d, wmax, name, H0 = args
    fit, t0, H0 = Fit(nP, d, wmax), time.time(), np.asarray(H0, float)
    w0 = fit.weights(H0)
    if w0 is None:
        print(f"P={nP} d={d:g} wmax={wmax:g} {name}: start infeasible", flush=True)
        return None
    J0 = fit.objective(H0, w0)
    H, _ = fit.split(fit.solve(np.concatenate([H0.ravel(), w0])))
    w = fit.weights(H)
    J = fit.objective(H, w) if w is not None else np.inf
    if J > J0:
        H, w, J = H0, w0, J0
    print(f"P={nP} d={d:g} wmax={wmax:g} {name}: gap left open {J0:.4f} -> {J:.4f}, slope {(hbasis(H, d) @ w)[0]:.0f}, "
          f"sum|w| {np.abs(w).sum():.3g}, gap closed "
          f"{ {k: round(v, 3) for k, v in summary(H, w, d).items()} } ({time.time() - t0:.0f}s)", flush=True)
    return nP, d, wmax, J, name, H, w


def flops(nP, m, n, steps=5):
    """FLOPs (2 m k n per m x k by k x n product): Gram LC and Muon / LC5m (both 3 products per step)."""
    return 2 * (2 * m * m * n + nP * m ** 3), steps * 2 * (2 * m * m * n + m ** 3)


if __name__ == "__main__":
    out_path = os.path.join(HERE, "gram_lc_coeffs.json")
    out = json.load(open(out_path)) if os.path.exists(out_path) else {}
    cfgs = [(int(a), float(b), float(c)) for a, b, c in (x.split(":") for x in sys.argv[1:])]
    with Pool(10) as pool:
        jobs = [(nP, d, wm, k, v) for nP, d, wm in cfgs for k, v in starts(nP, d).items()]
        res = [r for r in pool.map(run_start, jobs) if r]
    for nP, d, wm in cfgs:
        fit, key = Fit(nP, d, wm), f"gram{nP}_d{d:g}_w{wm:g}"
        results = sorted((r[3:] for r in res if r[:3] == (nP, d, wm)), key=lambda r: r[0])
        for J, name, H, w in results[:2]:
            Hr = np.array([rnd(h) for h in H])
            wr = fit.weights(Hr)
            if wr is None:
                print(f"P={nP} d={d:g} wmax={wm:g} {name}: rounded steps infeasible")
                continue
            wr = rnd(wr, 12)
            c = check(Hr, wr, d)
            J = fit.objective(Hr, wr)
            fm, fs = flops(nP, 384, 1536), flops(nP, 384, 384)
            print(f"P={nP} d={d:g} wmax={wm:g} {name} rounded: gap left open {J:.4f}; worst violation "
                  f"{c['worst']:.1e} at {c['at']:.4g}; slope {c['slope']:.0f}; max|C_k| {c['ymax']:.3f}; "
                  f"min h on [-d, 0] "
                  f"{c['hmin_neg']:.0f}; gap closed {summary(Hr, wr, d)}", flush=True)
            out[f"{key}_{name}"] = dict(
                p=P, P=nP, delta=d, wmax=wm, sum_abs_w=float(np.abs(wr).sum()), products=nP + 2, start=name,
                objective=J, low=LOW, steps=Hr.tolist(),
                weights=list(map(float, wr)), gap_closed=summary(Hr, wr, d), flops_384x1536=fm[0],
                flops_384x384=fs[0], muon_flops_384x1536=fm[1], muon_flops_384x384=fs[1],
                **{k: float(v) for k, v in c.items()})
        if any(k.startswith(f"{key}_") for k in out):
            out[key] = min((e for k, e in out.items() if k.startswith(f"{key}_")),
                                          key=lambda e: e["objective"])
        json.dump(out, open(out_path, "w"), indent=1)
