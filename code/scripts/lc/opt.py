"""Joint (end-to-end) optimization of all coefficients for the final scalar map.

Families (all odd matrix polynomials of X_0 = N / c):
  'lc'  : X track with free per-step coefficients; output = sum_k c_k X_k (no extra product).
          extras=j adds one product: output += [sum_k (b_k M_k + g_k M_k^2)] X_j, M_k = X_k X_k^T (already computed).
  'tt'  : two-track; output Z_K, Z_{k+1} = s_k(M_k) Z_k; optional free additive combination of the X_k and Z_k.
Objective: minimax of f(s) / (kappa s^p) - 1 on a log grid of [eps, 1], reached by least squares on |r|^(q/2) with
growing q, then (for 'lc') an exact linear program in the output coefficients.
Penalties keep every intermediate value bounded on [0, top] (top slightly above 1) and the map non-negative below eps."""
import numpy as np
from scipy.optimize import least_squares
from lib import poly, lp_minimax


class Problem:
    def __init__(self, p, eps, degs, family="lc", extras=None, ngrid=300, ymax=2.5, top=1.0, zdeg=2, cancel=None):
        self.p, self.eps, self.degs, self.family, self.extras = p, eps, list(degs), family, extras
        self.K = len(degs)
        self.sig = np.logspace(np.log10(eps), np.log10(top), ngrid)
        self.low = np.logspace(-9, np.log10(eps), 120)          # below the cutoff: only sign and bound checks
        self.allsig = np.concatenate([self.low, self.sig])
        self.nfit = ngrid
        self.ymax = ymax
        self.zdeg = zdeg
        self.cancel = cancel
        self.target = self.sig ** p

    # ---- parameter packing
    def nH(self):
        return sum(d + 1 for d in self.degs)

    def split(self, th):
        H, i = [], 0
        for d in self.degs:
            H.append(th[i:i + d + 1]); i += d + 1
        return H, th[i:]

    def tracks(self, H, sig):
        ys = [sig]
        for h in H:
            y = ys[-1]
            ys.append(np.clip(y * poly(h, y * y), -1e4, 1e4))
        return ys

    def basis(self, H, sig):
        ys = self.tracks(H, sig)
        cols = list(ys)
        if self.extras is not None:
            yj = ys[self.extras]
            for k in range(self.K):          # M_k = X_k X_k^T is computed for k = 0..K-1
                cols += [yj * ys[k] ** 2, yj * ys[k] ** 4]
        return np.stack(cols, 1), ys

    def nout(self):
        if self.family == "lc":
            return self.K + 1 + (2 * self.K if self.extras is not None else 0)
        return self.K * (self.zdeg + 1)

    def output(self, th, sig):
        H, rest = self.split(th)
        if self.family == "z":       # single track: the output is the last iterate (the "Z-only" iteration)
            ys = self.tracks(H, sig)
            return ys[-1], ys[:-1], None
        if self.family == "lc":
            B, ys = self.basis(H, sig)
            return B @ rest, ys, B * rest
        ys = self.tracks(H[:-1] if len(H) == self.K else H, sig)
        z = sig.copy()
        zs = [z]
        S = rest.reshape(self.K, self.zdeg + 1)
        for y, s in zip(ys, S):
            z = np.clip(z * poly(s, y * y), -1e6, 1e6)
            zs.append(z)
        return z, ys, None          # the Z track is allowed to grow (its target reaches eps^p); only X is bounded

    def resid(self, th, q):
        f, ys, terms = self.output(th, self.allsig)
        ff = f[-self.nfit:]
        r = ff / self.target - 1.0
        out = [np.sign(r) * np.abs(r) ** (q / 2) / self.nfit ** 0.5]
        for y in ys[1:]:
            out.append(3.0 * np.maximum(np.abs(y) - self.ymax, 0.0) / len(y) ** 0.5)
        fl = f[:len(self.low)]
        out.append(3.0 * np.maximum(-fl, 0.0) / self.eps ** self.p)
        out.append(3.0 * np.maximum(fl / self.eps ** self.p - 1.3, 0.0))      # no spike below the cutoff
        if self.cancel is not None and terms is not None:
            c = np.abs(terms[-self.nfit:]).sum(1) / self.target
            out.append(0.3 * np.maximum(c - self.cancel, 0.0) / self.nfit ** 0.5)
        return np.concatenate(out)

    def maxerr(self, th):
        f, _, _ = self.output(th, self.sig)
        return np.max(np.abs(f / self.target - 1.0))

    def solve(self, th0, qs=(2, 4, 8, 16, 32), verbose=False, nfev=400):
        th = np.array(th0, dtype=float)
        for q in qs:
            sol = least_squares(self.resid, th, args=(q,), method="trf", x_scale="jac", max_nfev=nfev)
            if np.all(np.isfinite(sol.x)):
                th = sol.x
            if verbose:
                print(f"   q={q}: max rel err {self.maxerr(th):.4f}")
        return th

    def polish_lp(self, th):
        """Exact minimax in the output coefficients for the fixed X track ('lc' only)."""
        H, rest = self.split(th)
        B, _ = self.basis(H, self.sig)
        c, e = lp_minimax(B, self.target)
        if c is None:
            return th, self.maxerr(th)
        return np.concatenate([np.concatenate(H), c]), e

    def init(self, H0, S0=None):
        H0 = [np.asarray(h, dtype=float)[:d + 1] if len(h) >= d + 1 else np.concatenate([h, np.zeros(d + 1 - len(h))])
              for h, d in zip(H0, self.degs)]
        if self.family == "z":
            return np.concatenate(H0)
        if self.family == "lc":
            B, _ = self.basis(H0, self.sig)
            c, e = lp_minimax(B, self.target)
            if c is None:
                c = np.zeros(B.shape[1]); c[self.K] = 1.0
            return np.concatenate([np.concatenate(H0), c])
        if S0 is None:
            S0 = [np.abs(h[0]) ** (1 - self.p) * np.array([1.0, h[1] / h[0], 0.0][:self.zdeg + 1]) for h in H0]
        return np.concatenate([np.concatenate(H0), np.concatenate([np.asarray(s, dtype=float) for s in S0])])


def products(degs, family="lc", extras=None):
    """Matrix products per step of the optimizer. X step of degree d in M: Gram + (d-1) powers + application."""
    n = sum(1 + (d - 1) + 1 for d in degs)
    if family == "z":
        return n
    if family == "lc":
        return n + (1 if extras is not None else 0)
    # two-track: Z update each step (+1), and the last X update is not needed (-1)
    return n + len(degs) - 1


def describe(pb, th, name=""):
    """Band, cutoff and tail of the final map on a fine grid."""
    sig = np.logspace(-8, 0, 160001)
    f, ys, terms = pb.output(th, sig)
    sel = sig >= pb.eps
    r = f[sel] / sig[sel] ** pb.p
    kappa = np.sqrt(r.min() * r.max())
    ratio = f / (kappa * sig ** pb.p)
    below = np.where(ratio < 0.5)[0]
    half = sig[below.max()] if len(below) else np.nan
    ymax = max(np.abs(y).max() for y in ys[1:])
    canc = np.abs(terms[sel]).sum(1).max() / kappa if terms is not None else np.nan
    canc_rel = (np.abs(terms[sel]).sum(1) / np.abs(f[sel])).max() if terms is not None else np.nan
    return dict(name=name, eps=pb.eps, lo=r.min() / kappa, hi=r.max() / kappa, half=half, fmin=f.min() / kappa,
                peak=f.max() / kappa, peak_at=sig[np.argmax(f)], ymax=ymax, cancel=canc_rel, kappa=kappa)


def fmt(d):
    return (f"{d['name']}: on [{d['eps']:.1e}, 1] f/(k s^p) in [{d['lo']:.3f}, {d['hi']:.3f}]; half-target point {d['half']:.1e};"
            f" peak {d['peak']:.2f} at {d['peak_at']:.1e}; min f {d['fmin']:.2g}; max |iterate| {d['ymax']:.2f};"
            f" cancellation {d['cancel']:.1f}")
