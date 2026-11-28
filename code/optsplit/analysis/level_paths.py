"""Constant-loss paths between two checkpoints at (nearly) equal training loss c (results/level_paths.tex).

- `losses`: loss of checkpoints on the fixed path set (to pick, for each run, the checkpoint closest to a level).
- `profile`: loss along the straight line (1-t) theta_A + t theta_B; its highest rise above c and deepest dip below.
- `band`: elastic band. N points theta_1..theta_N between the fixed endpoints theta_0 = theta_A, theta_{N+1} = theta_B
  minimise E_w = w sum_i ((L_i - c_i) / tol)^2 + sum_k |theta_{k+1} - theta_k|^2 / (D^2 / (N+1)), D = |theta_A - theta_B|,
  c_i interpolated linearly between the endpoint losses. First a stochastic string method brings the points to the
  level set and shortens the path (Adam steps on (L_i - c_i)^2 plus curve shortening within the level set; reaching
  the level set from the straight line takes training-like steps), then Levenberg-Marquardt steps on E_w (exact for
  the springs, linearised loss) converge it, w growing over stages. Every step uses a fresh group of training
  sequences, so that the path cannot fit one fixed set. Then the loss on a held-out set is checked densely along
  every segment; segments that leave c +- tol get a midpoint and the band continues. Output: path length / D and the
  loss along the path.

python -m optsplit.analysis.level_paths losses 'ts10m_main/adamw_d0' --steps 9000 10000 --out f.json
python -m optsplit.analysis.level_paths profile ts10m_main/adamw_d0:10000 ts10m_main/muon_d0:8000 --out p.json
python -m optsplit.analysis.level_paths band ts10m_main/adamw_d0:10000 ts10m_main/muon_d0:8000 --out b.json
python -m optsplit.analysis.level_paths walk ts10m_main/adamw_d0:10000 ts10m_main/muon_d0:8000 --out w.json"""

import argparse
import contextlib
import json
import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from optsplit.analysis.distance import js_divergence
from optsplit.analysis.io import Run, load_runs
from optsplit.loss import log_probs, mean_loss
from optsplit.models import build_model, layer

TAU = 0.05  # the reported tolerance of the paths (nats); the bands optimize and first check against 0.02
PATH_SEED, HELDOUT_SEED = 101, 102  # fixed training sequences (the probe uses seed 0)
STARTED, RESTART = time.time(), None  # RESTART: a worker's control file (see restart_if_asked)


def restart_if_asked():
    """In a worker (cmd_work): once its control file is newer than the process, re-execute the worker in place, with
    fresh code and defaults but the same job and GPU (`touch <dir>/restart_workers` after a git pull: every worker;
    `<dir>/restart_workers.<job id>`: the workers of one job). The task in progress starts again; its lock names
    this pid, which the new process takes back."""
    if RESTART is None:
        return
    job = os.environ.get('SLURM_JOB_ID', 'local')
    for f in (RESTART, RESTART.with_name(f'{RESTART.name}.{job}'), RESTART.with_name(f'{RESTART.name}.{job}.{os.getpid()}')):
        if f.exists() and f.stat().st_mtime > STARTED:
            print(f'restart: {f} is newer than this worker', flush=True)
            os.execv(sys.executable, [sys.executable, '-m', 'optsplit.analysis.level_paths', *sys.argv[1:]])


def say(s):
    print(s, flush=True)
    restart_if_asked()


class Stream:
    """Groups of fresh training sequences without end, for Objective(groups=...): group j is n random training
    sequences drawn with seed `seed + j`, in chunks (so that a long iteration never sees a sequence twice, like
    training)."""

    def __init__(self, data, n, chunk, seed):
        self.data, self.n, self.chunk, self.seed = data, n, chunk, seed
        f = getattr(getattr(data, 'splits', {}).get('train'), 'filename', None)
        if f:  # read the training tokens once, so that the random windows come from the page cache, not the network
            with open(f, 'rb') as fh:
                while fh.read(1 << 26):
                    pass

    def __len__(self):
        return 2 ** 31

    def __getitem__(self, j):
        x, y = self.data.random_batches(1, self.n, self.seed + j)[0]
        return [(x[i:i + self.chunk], y[i:i + self.chunk]) for i in range(0, self.n, self.chunk)]


class Objective:
    """Mean loss on fixed batches as a function of the flat parameter vector of `model`. With `groups` (several lists
    of batches), rotate() moves to the next group: the band and the walk take a fresh group at every iteration, so
    that the path cannot fit one fixed set of sequences."""

    def __init__(self, model, batches, fp32=True, groups=None, amp=False):
        self.model, self.batches, self.fp32, self.amp = model, batches, fp32, amp
        self.params = list(model.parameters())
        self.groups, self.g = groups, 0

    def rotate(self):
        self.g = (self.g + 1) % len(self.groups)
        self.batches = self.groups[self.g]

    @contextlib.contextmanager
    def _precision(self):
        """fp32=True: exact float32 matmuls (checks); False: TF32 (band iterations, faster); amp: bfloat16 autocast
        on GPU, as in training (string-method steps)."""
        old = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = not self.fp32
        amp = self.amp and self.params[0].is_cuda
        try:
            with torch.autocast('cuda', dtype=torch.bfloat16) if amp else contextlib.nullcontext():
                yield
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old

    def _load(self, theta):
        i = 0
        for p in self.params:
            p.data.copy_(theta[i:i + p.numel()].view_as(p))
            i += p.numel()

    @torch.no_grad()
    def loss(self, theta):
        self._load(theta)
        with self._precision():
            return float(sum(self.model.loss(x, y).item() for x, y in self.batches) / len(self.batches))

    def loss_grad(self, theta):
        self._load(theta)
        for p in self.params:
            p.grad = None
        total = 0.0
        with self._precision():
            for x, y in self.batches:
                l = self.model.loss(x, y) / len(self.batches)
                l.backward()
                total += l.item()
        return total, torch.cat([p.grad.reshape(-1) for p in self.params])

    def hvp(self, theta, v, n_batches=2):
        """Hessian-vector product of the loss on the first n_batches batches (a cheaper estimate of the curvature)."""
        from optsplit.models import double_backward_ok
        self._load(theta)
        vs = list(v.split([p.numel() for p in self.params]))
        out = torch.zeros_like(v)
        with self._precision(), double_backward_ok():
            for x, y in self.batches[:n_batches]:
                g = torch.autograd.grad(self.model.loss(x, y) / n_batches, self.params, create_graph=True)
                hv = torch.autograd.grad(sum((gi * vi.view_as(gi)).sum() for gi, vi in zip(g, vs)), self.params)
                out += torch.cat([h.reshape(-1) for h in hv])
        return out

    @torch.no_grad()
    def walk(self, thetas):
        """Loss at each parameter vector of the sequence `thetas`, the Jensen-Shannon divergence (bits, mean over
        tokens, as in Fig. 1) between consecutive ones, and between the first and the last. Batch by batch, so no
        predictions are stored."""
        n, kind = len(thetas), self.model.loss_kind
        L, J, ends = np.zeros(n), np.zeros(max(n - 1, 0)), 0.0
        with self._precision():
            for x, y in self.batches:
                first = prev = None
                for i in range(n):
                    self._load(thetas[i])
                    z = self.model(x).float().reshape(-1, self.model.config.vocab_size)
                    L[i] += mean_loss(z, y.reshape(-1), kind).item()
                    lp = log_probs(z, kind)
                    if prev is not None:
                        J[i - 1] += js_divergence(prev, lp)
                    first = lp if first is None else first
                    prev = lp
                ends += js_divergence(first, prev)
                del first, prev
        k = len(self.batches)
        return L / k, J / k, ends / k

    @torch.no_grad()
    def per_seq(self, theta):
        """Mean loss of each sequence (for the noise of the set's mean)."""
        self._load(theta)
        out = []
        with self._precision():
            for x, y in self.batches:
                z = self.model(x).float()
                out.append(torch.nn.functional.cross_entropy(z.transpose(1, 2), y, reduction='none').mean(1))
        return torch.cat(out).cpu()


def setup(cfg, n_seqs, device, chunk=64, pool=1):
    """Model; the path set and the held-out set (n_seqs training sequences each, in chunks); the probe's training
    sequences, on which the training loss of Fig. 1 is measured; the pool: the path set and pool - 1 more groups of
    n_seqs sequences, which the band and walk iterations cycle through; the data (for streams of fresh sequences)."""
    from optsplit.data import build_data
    data = build_data(cfg, device)
    model = build_model(cfg, vocab_size=data.vocab_size).to(device).eval()
    sets = []
    seeds = [(n_seqs, PATH_SEED), (n_seqs, HELDOUT_SEED), (cfg.probe.train_seqs, 0)]
    for n, seed in seeds + [(n_seqs, 1000 + j) for j in range(1, pool)]:
        x, y = data.fixed_batch('train', n, seed)
        sets.append([(x[i:i + chunk], y[i:i + chunk]) for i in range(0, n, chunk)])
    return model, sets[:3] + [sets[:1] + sets[3:], data]


def load_theta(spec, device):
    """'suite/run:step' -> flat float32 parameter vector."""
    run, step = spec.rsplit(':', 1)
    sd = torch.load(Run(run).ckpt_path(int(step)), map_location=device)['model']
    return torch.cat([v.float().reshape(-1) for v in sd.values()])


def check_order(model, spec, device):
    """The state dict and model.parameters() list the same tensors in the same order (no buffers, no ties)."""
    run, step = spec.rsplit(':', 1)
    sd = torch.load(Run(run).ckpt_path(int(step)), map_location='cpu')['model']
    names = [n for n, _ in model.named_parameters()]
    assert names == list(sd), 'parameter order differs from the state dict'


# ---------------------------------------------------------------- straight line

class Polyline:
    """The sequence of points P[k] + t (P[k+1] - P[k]) for consecutive (k, t) pairs, built lazily."""

    def __init__(self, P, kt):
        self.P, self.kt = P, kt

    def __len__(self):
        return len(self.kt)

    def __getitem__(self, i):
        k, t = self.kt[i]
        return self.P[k] if t == 0 else torch.lerp(self.P[k], self.P[k + 1], float(t))


def walk(obj, seq):
    """(losses, JS between consecutive points or None, JS between the ends or None)."""
    if hasattr(obj, 'walk'):
        return obj.walk(seq)
    return np.array([obj.loss(seq[i]) for i in range(len(seq))]), None, None


def pred_length(J, ends):
    """Length in prediction space (sum of JS distances, the square roots of the JS divergences) and its ratio to the
    JS distance between the ends."""
    if J is None:
        return None, None
    length = float(np.sqrt(np.maximum(J, 0)).sum())
    return length, length / math.sqrt(ends)


def profile(obj, a, b, n=51, ca=None, cb=None):
    """Loss at (1-t) a + t b for t in linspace(0, 1, n); c(t) interpolates the endpoint losses. Also the length of
    the line in prediction space over these n points."""
    ts = np.linspace(0, 1, n)
    L, J, ends = walk(obj, Polyline(torch.stack([a, b]), [(0, t) for t in ts]))
    ca = L[0] if ca is None else ca
    cb = L[-1] if cb is None else cb
    dev = L - (ca + ts * (cb - ca))
    plen, pratio = pred_length(J, ends)
    return dict(t=ts.tolist(), loss=L.tolist(), c_a=float(ca), c_b=float(cb),
                rise=float(max(dev.max(), 0)), dip=float(max(-dev.min(), 0)), distance=float((a - b).norm()),
                js_steps=None if J is None else J.tolist(), js_ends=ends, pred_length=plen, pred_ratio=pratio)


# ---------------------------------------------------------------- elastic band

def straight(a, b, n):
    t = torch.arange(1, n + 1, dtype=a.dtype, device=a.device)[:, None] / (n + 1)
    return a + t * (b - a)


def targets(ca, cb, n):
    return ca + (cb - ca) * np.arange(1, n + 1) / (n + 1)


def lengths(X, a, b):
    P = [a, *X, b]
    return np.array([float((P[k + 1] - P[k]).double().norm()) for k in range(len(P) - 1)])


def dots(U, V, chunk=2 ** 20):
    """U @ V.T in float64, a slice of parameters at a time (U, V: (n, P) float32)."""
    out = 0
    for i in range(0, U.shape[1], chunk):
        out = out + U[:, i:i + chunk].double() @ V[:, i:i + chunk].double().T
    return out.cpu().numpy()


def merit(X, a, b, L, c, w, tol):
    """E_w: the constraint term and the spring term (1 on the evenly spaced straight line)."""
    D2 = float((a - b).double().norm() ** 2)
    return w * float(np.sum(((L - c) / tol) ** 2)) + (len(X) + 1) * float(np.sum(lengths(X, a, b) ** 2)) / D2


def lm_target(X, a, b, L, G, c, w, tol, lam):
    """Minimiser of E_w with each loss linearised at the current points, plus the damping (lam / kappa) |X' - X|^2
    (a Levenberg-Marquardt step; exact for the springs). With Z = X - X_line, M = tridiag(-1, 2, -1) and
    M_lam = M + lam I: X' = X_line + M_lam^{-1} (lam Z + (gamma_i g_i)_i), where gamma solves
    (tol^2 / (kappa w) I + M_lam^{-1} o Gram) gamma = e - lam rowsum(M_lam^{-1} o H), H_ij = g_i . Z_j,
    e_i = c_i - L_i + g_i . Z_i, kappa = D^2 / (N+1). lam = 0 is the Gauss-Newton step. At a fixed point the spring
    force on each point is parallel to its gradient (a discrete geodesic of the level set when w is large)."""
    n = len(X)
    Xl = straight(a, b, n)
    Z = X - Xl
    Minv = np.linalg.inv(2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1) + lam * np.eye(n))
    gram, H = dots(G, G), dots(G, Z)
    e = np.asarray(c) - np.asarray(L) + np.diag(H)
    kappa = float((a - b).double().norm() ** 2) / (n + 1)
    gamma = np.linalg.solve(tol ** 2 / (kappa * w) * np.eye(n) + Minv * gram, e - lam * (Minv * H).sum(1))
    t = lambda A: torch.tensor(A, dtype=X.dtype, device=X.device)
    return Xl + t(lam * Minv) @ Z + t(Minv * gamma[None, :]) @ G


def evaluate(obj, X):
    L, G = zip(*(obj.loss_grad(x) for x in X))
    return np.array(L), torch.stack(G)


def dense_check(obj, X, a, b, c_pts, tol, m=8):
    """Loss at the points P_0..P_{N+1} (endpoints included) and at m interior points of every segment; c_pts (N+2
    values) interpolated linearly along each segment. -> per segment: max |L - c|; arrays (arclength, loss, c)."""
    P = [a, *X, b]
    seg = lengths(X, a, b)
    s0 = np.concatenate([[0], np.cumsum(seg)])
    ts = np.linspace(0, 1, m + 2)
    kt = [(k, t) for k in range(len(P) - 1) for t in ts[:-1]] + [(len(P) - 1, 0)]
    L, J, ends = walk(obj, Polyline(P, kt))
    c = np.array([c_pts[k] + t * (c_pts[k + 1] - c_pts[k]) if t else c_pts[k] for k, t in kt])
    s = np.array([s0[k] + t * seg[k] if t else s0[k] for k, t in kt])
    dev = np.abs(L - c)
    worst = np.array([dev[k * (m + 1):(k + 1) * (m + 1) + 1].max() for k in range(len(P) - 1)])
    plen, pratio = pred_length(J, ends)
    return worst, dict(s=s.tolist(), loss=L.tolist(), c=c.tolist(), points=L[::m + 1].tolist(),
                       js_steps=None if J is None else J.tolist(), js_ends=ends, pred_length=plen, pred_ratio=pratio)


def band(obj, a, b, ca, cb, obj_check=None, ca_check=None, cb_check=None, n=10, obj_string=None, steps=1000, lr=1e-3,
         beta=0.25, ws=(1, 10), iters=10, tol=0.02, rounds=4, round_steps=300, round_iters=10, m=8, max_points=40,
         init=None, obj_shorten=None, shorten_steps=0, shorten_beta=0.4, stash=None, resume_steps=2500, accept=None,
         log=print):
    """Elastic band between a and b at the losses ca -> cb (see the module docstring): `steps` steps of the string
    method on `obj_string` (default obj), then Levenberg-Marquardt stages at the weights `ws` on `obj`, then at most
    `shorten_steps` steps of curve shortening within the level set on `obj_shorten` (default obj); `obj_check`
    (default obj, with ca_check/cb_check) does the dense check. The points after the string and LM phases are saved
    to `stash` (if given), and a later call starts from there. If they are off the level (max |L - c| above 2 tol),
    the string method continues from them once without the springs (beta = 0), which cut corners off the level, with
    a quarter of the step size, warming up over 10% of `resume_steps` steps and annealing to 0 over the rest, and the
    LM stages run again; the points closer to the level are kept. Points still off the level get no insertion rounds
    (the result is invalid); after such a continuation, the insertion rounds' strings run without springs too.
    accept (default tol) is the tolerance of the dense check, the insertions and validity; tol, that of the
    optimization (the LM penalty, the string's stopping rule, the shortening's trust region).
    Returns (points, result)."""
    obj_check, accept = obj_check or obj, accept or tol
    obj_string = obj_string or obj
    obj_shorten = obj_shorten or obj
    ca_check = ca if ca_check is None else ca_check
    cb_check = cb if cb_check is None else cb_check
    D = float((a - b).norm())
    X = straight(a, b, n) if init is None else init.clone()
    hist, t0 = [], time.time()
    rotating = len(getattr(obj, 'groups', None) or []) > 1

    def ends(o=obj):  # endpoint losses on the next group of sequences of o (if it has several)
        if len(getattr(o, 'groups', None) or []) <= 1:
            return ca, cb
        o.rotate()
        return o.loss(a), o.loss(b)

    def string(X, k, scale=1.0, hold=0.7, springs=True, ckpt=None):
        """At most k steps of a stochastic string method, with Adam step size scale * lr: on a fresh group of
        sequences, each point takes an Adam step on (L_i - c_i)^2 / 2 (descent to the level set, like training; from
        below it climbs), then moves by beta times the discrete Laplacian of the path minus its component along the
        point's loss gradient (curve shortening within the level set; it also evens out the spacing). Both step sizes
        warm up over the first 10% of the steps and then decay linearly to 0: over the last 30% of the k steps or
        earlier, once the max |L - c| at the points on the check set (the better of two checks 50 steps apart, against
        occasional spikes) is below 2 tol (then over as many steps as taken so far) or has improved by less than 2%
        over 200 steps (a noise plateau above the level, as in training at constant lr: then over all the remaining
        steps, at least as many as taken so far). Adam's betas are training's (0.9, 0.95). With ckpt, the points and
        the schedule are saved every 250 steps, and a later call with the same arguments continues from there (with
        fresh Adam moments), e.g. after a preemption."""
        mom, sq = torch.zeros_like(X), torch.zeros_like(X)
        warm, stop, decay, chk = 0.1 * k, int(hold * k), k - int(hold * k), []
        t = ta = 0  # ta: Adam steps since the (re)start
        key = dict(k=k, scale=scale, hold=hold, springs=springs, n=len(X))
        if ckpt is not None and Path(ckpt).exists():
            sv = torch.load(ckpt, map_location=X.device)
            if sv['key'] == key:
                X, t, stop, decay, chk = sv['X'], sv['t'], sv['stop'], sv['decay'], sv['chk']
                hist[:] = sv['hist']
                log(f'string: continuing at t={t} from {ckpt}')
        while t < stop + decay:
            t, ta = t + 1, ta + 1
            c = targets(*ends(obj_string), len(X))
            L, G = evaluate(obj_string, X)
            g = torch.tensor(L - c, dtype=X.dtype, device=X.device)[:, None] * G
            mom.lerp_(g, 0.1)
            sq.lerp_(g * g, 0.05)
            f = min(1.0, t / warm) if t <= stop else max(0.0, 1 - (t - stop) / decay)
            P = torch.cat([a[None], X, b[None]])
            lap = P[:-2] + P[2:] - 2 * X
            u = G / G.norm(dim=1, keepdim=True).clamp_min(1e-12)
            lap -= (lap * u).sum(1, keepdim=True) * u
            X = X - f * scale * lr * (mom / (1 - 0.9 ** ta)) / ((sq / (1 - 0.95 ** ta)).sqrt() + 1e-8) + f * springs * beta * lap
            del P, lap, u, G
            if t <= stop and t % 50 == 0 and t > warm:  # early stop: at the level, or flat over the last 200 steps
                chk.append(check_dev(X))
                log(f"string n={len(X)} t={t}: max|L-c| at the points {chk[-1]:.4f} (check set), "
                    f"ratio {float(lengths(X, a, b).sum() / D):.5f}")
                d1 = min(chk[-2:])
                flat = len(chk) >= 6 and min(chk[-6:-4]) - d1 < max(0.003, 0.02 * min(chk[-6:-4]))
                if flat or d1 <= 2 * tol:
                    stop, decay = t, t if d1 <= 2 * tol else max(t, k - t)
                    log(f"string: {'flat' if flat else 'at the level'}, decaying over {decay} steps")
            if ckpt is not None and t % 250 == 0:
                torch.save(dict(key=key, X=X, t=t, stop=stop, decay=decay, chk=chk, hist=hist), f'{ckpt}.tmp')
                os.replace(f'{ckpt}.tmp', ckpt)
            if t % 50 == 0 or t == stop + decay:
                hist.append(dict(phase='string', n=len(X), t=t, max_r=float(np.abs(L - c).max()),
                                 ratio=float(lengths(X, a, b).sum() / D), time=time.time() - t0))
                log(f"string n={len(X)} t={t}: max|L-c| {hist[-1]['max_r']:.4f} (batch, before the step), "
                    f"ratio {hist[-1]['ratio']:.5f} ({hist[-1]['time']:.0f}s)")
        return X

    def check_dev(X):  # max |L - c| at the points, on the check set
        cc = targets(ca_check, cb_check, len(X))
        return float(np.abs(np.array([obj_check.loss(x) for x in X]) - cc).max())

    def shorten(X, k):
        """At most k steps of curve shortening within the level set, to remove the wander that the noise of the string
        phase leaves in the path: on a fresh group of sequences of obj_shorten (a larger batch than the string
        phase's), each point moves by beta times the discrete Laplacian of the path minus its component along its
        loss gradient, plus a fraction nu of a Newton step back to its target loss. beta and nu adapt like a trust
        region: if the max |L - c| after a step exceeds lim = max(tol, its value at the start), the step is
        undone and both are halved; after a step that keeps it below lim / 2, both grow by 20% (up to shorten_beta and
        1; they start at 1/16 of shorten_beta and 1/4). Stops once the length has fallen by less than 1% of its excess over the last 100 steps. Skipped for a
        path off the level (max |L - c| on the check set above 2 accept), where the Newton steps fail."""
        bs, nu, ratios, prev, lim, undone = shorten_beta / 16, 0.25, [float(lengths(X, a, b).sum() / D)], None, None, 0
        hist.append(dict(phase='shorten', n=len(X), t=0, beta=bs, check_max_r=check_dev(X), ratio=ratios[0],
                         time=time.time() - t0))
        if hist[-1]['check_max_r'] > 2 * accept:
            log(f"shorten: skipped, max|L-c| {hist[-1]['check_max_r']:.4f} on the check set")
            return X
        for t in range(1, k + 1):
            c = targets(*ends(obj_shorten), len(X))
            L, G = evaluate(obj_shorten, X)
            dev = float(np.abs(L - c).max())
            lim = lim or max(tol, dev)
            if prev is not None and dev > lim:  # the last step left the level set: undo it, smaller steps
                X, bs, nu, undone = prev, bs / 2, nu / 2, undone + 1
                L, G = evaluate(obj_shorten, X)
            elif dev < lim / 2:
                bs, nu = min(bs * 1.2, shorten_beta), min(nu * 1.2, 1.0)
            prev = X
            P = torch.cat([a[None], X, b[None]])
            lap = P[:-2] + P[2:] - 2 * X
            gg = (G * G).sum(1, keepdim=True).clamp_min(1e-24)
            lap -= (lap * G).sum(1, keepdim=True) / gg * G
            X = X + bs * lap - nu * torch.tensor(L - c, dtype=X.dtype, device=X.device)[:, None] * G / gg
            del P, lap, G, gg
            if t % 25 == 0:
                ratios.append(float(lengths(prev, a, b).sum() / D))
                h = dict(phase='shorten', n=len(X), t=t, beta=bs, max_r=float(np.abs(L - c).max()), ratio=ratios[-1],
                         undone=undone, time=time.time() - t0)
                if t % 100 == 0:
                    h['check_max_r'] = check_dev(prev)
                hist.append(h)
                log(f"shorten n={len(X)} t={t}: ratio {h['ratio']:.5f}, max|L-c| {h['max_r']:.4f} (batch)" +
                    (f", {h['check_max_r']:.4f} (check set)" if 'check_max_r' in h else '') +
                    f", beta {bs:.3g}, {undone} steps undone ({h['time']:.0f}s)")
                if len(ratios) >= 5 and ratios[-5] - ratios[-1] < max(1e-6, 0.01 * (ratios[-5] - 1)):
                    break
        return prev if prev is not None else X

    lam = [1.0]  # Levenberg-Marquardt damping, adapted: /3 after an accepted step, x4 after a rejected one

    def run_stage(X, w, k, ramp=0):
        """k iterations at weight w. With ramp > 0 the targets start at the current losses and reach c_i linearly
        over the first `ramp` iterations (a continuation, so that each step's linearisation stays accurate)."""
        n, dev0, stuck = len(X), None, 0
        for j in range(k):
            c_end = targets(*ends(), n)
            if rotating or j == 0:
                L, G = evaluate(obj, X)
            if dev0 is None:
                dev0 = L - c_end
                ramp = ramp if np.abs(dev0).max() > tol / 10 else 0  # already at c: no continuation
            s = min(1.0, (j + 1) / ramp) if ramp else 1.0
            c = c_end + (1 - s) * dev0
            E = merit(X, a, b, L, c, w, tol)
            for _ in range(12):
                Xn = lm_target(X, a, b, L, G, c, w, tol, lam[0])
                Ln = np.array([obj.loss(x) for x in Xn])
                En = merit(Xn, a, b, Ln, c, w, tol)
                if En < E:
                    lam[0] = max(lam[0] / 3, 1e-6)
                    break
                lam[0] = min(lam[0] * 4, 1e9)
            else:
                if s == 1:
                    break  # no step lowers E_w: converged at this w
                continue
            X = Xn
            if not rotating:
                L, G = evaluate(obj, X)
            seg = lengths(X, a, b)
            hist.append(dict(n=n, w=w, s=s, lam=lam[0], E=En, max_r=float(np.abs(Ln - c_end).max()),
                             ratio=float(seg.sum() / D), time=time.time() - t0))
            log(f"n={n} w={w:g} s={s:.2f} lam={lam[0]:.3g} E={En:.4g} max|L-c|={hist[-1]['max_r']:.4f} "
                f"ratio={hist[-1]['ratio']:.5f} ({hist[-1]['time']:.0f}s)")
            stuck = stuck + 1 if s == 1 and lam[0] >= 1e6 else 0
            if s == 1 and (E - En < 1e-4 * (E - 1) or stuck >= 3):
                break  # converged at this w (E_w >= 1), or stuck (for 3 iterations only tiny steps lowered E_w)
        return X

    stages, first = [], dict(n=n, steps=steps, lr=lr, beta=beta, ws=list(ws), iters=iters, version=3)
    st = torch.load(stash, map_location=a.device) if stash is not None and Path(stash).exists() else None
    st = st if st is not None and st['settings']['n'] == n else None
    if st is not None:
        X, stages = st['X'], list(st['stages'])
        hist.extend(st['hist'])
        log(f"resumed after the string and LM phases from {stash}, max|L-c| {stages[-1]['max_r']:.4f}")
    sck = None if stash is None else Path(stash).with_suffix('.string.pt')  # the string phase's own checkpoint

    def string_lm(X, cont):
        if cont:  # from a path off the level, without the springs, which cut corners off it
            X = string(X, resume_steps, scale=0.25, hold=0.1, springs=False, ckpt=sck)
        elif steps:
            X = string(X, steps, ckpt=sck)
        if cont or steps:
            stages.append(dict(w='string', ratio=hist[-1]['ratio'], max_r=hist[-1]['max_r']))
        for w in ws:
            X = run_stage(X, w, iters, ramp=iters // 2)
            if hist:
                stages.append(dict(w=w, ratio=hist[-1]['ratio'], max_r=hist[-1]['max_r']))
        return X

    if st is None:
        X = string_lm(X, False)
        if stash is not None and stages:
            torch.save(dict(X=X, hist=hist, stages=stages, settings=first), stash)
    strings = sum(s['w'] == 'string' for s in stages)
    if resume_steps and stages and stages[-1]['max_r'] > 2 * tol and strings < 2:  # off the level: one continuation
        best = X, list(stages)
        X = string_lm(X, True)
        if stages[-1]['max_r'] >= best[1][-1]['max_r']:
            log(f"no closer to the level ({stages[-1]['max_r']:.4f}): back to the saved points")
            X, stages = best[0], list(best[1])
        elif stash is not None:
            torch.save(dict(X=X, hist=hist, stages=stages, settings=first), stash)
    if sck is not None:
        sck.unlink(missing_ok=True)
    if shorten_steps:
        X = shorten(X, shorten_steps)
        stages.append(dict(w='shorten', ratio=hist[-1]['ratio'], max_r=hist[-1].get('max_r')))
    def summary(X, worst, dense):
        seg = lengths(X, a, b)
        _, Jv, ends = walk(obj_check, [a, *X, b])
        return dict(n=len(X), ratio=float(seg.sum() / D), max_dev=float(np.max(worst)),
                    ok=bool(np.max(worst) <= accept), pred_ratio=dense['pred_ratio'],
                    pred_ratio_points=pred_length(Jv, ends)[1],
                    offline_max=float((X - straight(a, b, len(X))).norm(dim=1).max() / D))

    checks = []
    for r in range(rounds + 1):
        c_chk = np.concatenate([[ca_check], targets(ca_check, cb_check, len(X)), [cb_check]])
        worst, dense = dense_check(obj_check, X, a, b, c_chk, accept, m)
        checks.append(dict(n=len(X), worst=worst.tolist()))
        if r == 0:
            fixed = summary(X, worst, dense)  # fixed N, before any insertion
        bad = np.flatnonzero(worst > accept)
        log(f'check {r}: {len(X)} points, worst segment {worst.max():.4f} (tol {accept}), {len(bad)} segments out')
        if not len(bad) or r == rounds or len(X) + len(bad) > max_points:
            break
        lm = [s for s in stages if s['w'] not in ('string', 'shorten')]
        if lm and lm[-1]['max_r'] > 2 * accept:
            log('no insertion rounds: the points are off the level')
            break
        P = [a, *X, b]
        new = []
        for k in range(len(P) - 1):
            if k > 0:
                new.append(P[k])
            if k in bad:
                new.append((P[k] + P[k + 1]) / 2)
        X = torch.stack(new)
        if round_steps:
            X = string(X, round_steps, scale=0.25, springs=sum(s['w'] == 'string' for s in stages) < 2)  # a gentle continuation
        X = run_stage(X, ws[-1], round_iters, ramp=round_iters // 2)
        if shorten_steps:
            X = shorten(X, shorten_steps)
    valid = summary(X, worst, dense)  # after the insertions
    res = dict(**valid, fixed=fixed, valid=valid, distance=D, seg=lengths(X, a, b).tolist(), tol=accept, opt_tol=tol,
               c_a=ca_check, c_b=cb_check, dense=dense, history=hist, checks=checks, stages=stages,
               settings=dict(n=n, steps=steps, lr=lr, beta=beta, ws=list(ws), iters=iters, rounds=rounds,
                             round_steps=round_steps, round_iters=round_iters, m=m, shorten_steps=shorten_steps,
                             shorten_beta=shorten_beta))
    return X, res


# ---------------------------------------------------------------- level-set walk (second estimate)

def level_walk(obj, a, b, obj_check, ca_check, cb_check, obj_verify=None, tol=0.01, step=0.01, max_step=0.02,
               min_step=5e-4, corrections=30, max_iters=12000, m=4, taus=(0.02, 0.1), stash=None, log=print):
    """Predictor-corrector walk from a to b that stays on the level set {L = c} (after Huang et al.'s continuation);
    c is interpolated between the endpoint losses by the progress s = 1 - |x - b| / D. Predictor: a step of length
    eta D from the current point x along b - x minus its component along the loss gradient g (the loss is unchanged
    to first order). Corrector: Gauss-Newton steps y -= nu r P g / (g . P g) back to the level, with r = L - c on a
    fresh group of `obj` (paired: c comes from the endpoint losses on the same sequences), g and P as in Adam: g the
    running mean of the gradients on these groups (beta 0.9, which averages out their noise), P the inverse square
    root of the running mean of their squares (beta 0.95); nu (at most 1) is halved after an overshoot and grows by
    50% otherwise. The step is accepted once |r| <= 0.4 tol on the group, then |r| <= tol / 2 at y and |r| <= tol at
    the midpoint of the new segment, each on a fresh group of `obj_verify`. A failed correction (`corrections`
    gradient steps, or a midpoint off the level) returns to x and halves eta, a quick one grows it
    (eta between min_step and max_step). Stops at b, or stuck: b - x along the gradient (sin < 0.05: every step
    towards b leaves the level), eta below min_step, under 1% of D of progress in 100 steps, or `max_iters` gradient
    steps. Every accepted segment is then checked on obj_check (the held-out set) at m interior points; the check
    does not feed back. At tau in taus the walk counts up to its first segment off by more than tau. With `stash`,
    the state is saved every 10 minutes and a later call continues from it. -> result dict."""
    obj_verify = obj_verify or obj
    D = float((a - b).norm())
    prog = lambda y: min(1.0, max(0.0, 1 - float((y - b).norm()) / D))
    settings = dict(tol=tol, step=step, max_step=max_step, min_step=min_step, corrections=corrections,
                    max_iters=max_iters, m=m)

    def fresh(o):  # the next group of sequences of o (if it has several) and the endpoint losses on it
        if len(getattr(o, 'groups', None) or []) > 1:
            o.rotate()
        return o.loss(a), o.loss(b)

    def resid_grad(y):
        la, lb = fresh(obj)
        L, g = obj.loss_grad(y)
        return L - (la + (lb - la) * prog(y)), g

    def resid(*ys):
        la, lb = fresh(obj_verify)
        return [obj_verify.loss(y) - (la + (lb - la) * prog(y)) for y in ys]

    st = torch.load(stash, map_location=a.device) if stash is not None and Path(stash).exists() else None
    if st is not None and st['settings'] == settings:
        x, v, g, eta, nu, iters, rec = st['x'], st['v'], st['g'], st['eta'], st['nu'], st['iters'], st['rec']
        for o, j in zip((obj, obj_verify), st['groups']):
            if j is not None:
                o.g = j - 1
                o.rotate()
        log(f"walk: continuing at step {len(rec['seg'])} from {stash}")
    else:
        x, eta, nu, iters = a.clone(), step, 1.0, 1
        _, g = resid_grad(x)
        v = g * g
        rec = dict(seg=[], dev=[], dist=[], s=[0.0], loss=[], c=[], hist=[])
    t0, saved, reason = time.time(), time.time(), None
    while True:
        dist = float((b - x).norm())
        if dist == 0:
            break
        if iters >= max_iters:
            reason = 'budget'
            break
        if len(rec['dist']) > 100 and rec['dist'][-101] - rec['dist'][-1] < 0.01 * D:
            reason = 'stall'
            break
        u = b - x
        direct = dist <= eta * D
        if direct:  # the last step: straight to b, if the segment stays on the level
            y = b
            ok, j = abs(resid((x + b) / 2)[0]) <= tol, 0
            if not ok:
                eta = 0.5 * dist / D
                continue
        else:
            d = u - float(u @ g) / float(g @ g) * g
            sin = float(d.norm()) / dist
            if sin < 0.05:
                reason = 'gradient'
                break
            y, ok, prev, rs = x + eta * D * d / d.norm(), False, None, []
            for j in range(1, corrections + 1):
                r, gy = resid_grad(y)
                iters += 1
                rs.append(r)
                g.lerp_(gy, 0.1)
                v.lerp_(gy * gy, 0.05)
                if abs(r) <= 0.4 * tol and abs(resid(y)[0]) <= tol / 2:
                    ok = abs(resid((x + y) / 2)[0]) <= tol
                    break  # if not ok, the chord leaves the level: a shorter step
                if prev is not None and r * prev < 0 and abs(r) > max(0.5 * abs(prev), tol / 2):
                    nu = nu / 2  # overshoot
                else:
                    nu = min(1.0, 1.5 * nu)
                pg = g / (v.sqrt() + 1e-8)
                y = y - nu * r * pg / float(g @ pg)
                prev = r
            if not ok:
                eta /= 2
                log(f"walk: correction failed after {j} steps, eta {eta:.2e}; L - c: {np.round(rs, 4).tolist()}")
                if eta < min_step:
                    reason = 'step'
                    break
                continue
            if j <= 2:
                eta = min(max_step, 1.25 * eta)
        ts = np.linspace(0, 1, m + 2)  # the held-out check of the new segment
        L = walk(obj_check, Polyline(torch.stack([x, y]), [(0, t) for t in ts]))[0]
        cx, cy = (ca_check + (cb_check - ca_check) * prog(p) for p in (x, y))
        c = cx + ts * (cy - cx)
        seg = float((y - x).norm())
        first = 0 if not rec['loss'] else 1
        rec['loss'] += L[first:].tolist()
        rec['c'] += c[first:].tolist()
        rec['s'] += (rec['s'][-1] + seg * ts[1:]).tolist()
        rec['seg'].append(seg)
        rec['dev'].append(float(np.abs(L - c).max()))
        x = y.clone()
        rec['dist'].append(float((b - x).norm()))
        rec['hist'].append(dict(step=len(rec['seg']), dist=rec['dist'][-1] / D, eta=eta, j=j, iters=iters, nu=nu,
                                dev=rec['dev'][-1], time=time.time() - t0))
        log(f"walk {len(rec['seg'])}: |x - b|/D {rec['dist'][-1] / D:.4f}, eta {eta:.2e}, {j} corrections "
            f"({iters} in all), held-out max|L-c| {rec['dev'][-1]:.4f}, length/D {sum(rec['seg']) / D:.5f} "
            f"({time.time() - t0:.0f}s)")
        if stash is not None and time.time() - saved > 600:
            groups = [getattr(o, 'g', None) if getattr(o, 'groups', None) else None for o in (obj, obj_verify)]
            torch.save(dict(settings=settings, x=x, v=v, g=g, eta=eta, nu=nu, iters=iters, rec=rec, groups=groups),
                       f'{stash}.tmp')
            os.replace(f'{stash}.tmp', stash)
            saved = time.time()
    reached = reason is None
    left = float((b - x).norm()) / D
    at = {}
    for tau in taus:  # the walk up to its first segment off the level by more than tau
        k = next((i for i, d in enumerate(rec['dev']) if d > tau), len(rec['dev']))
        at[f'{tau:g}'] = dict(reached=reached and k == len(rec['dev']), segments=k,
                              left=rec['dist'][k - 1] / D if k else 1.0, ratio=sum(rec['seg'][:k]) / D,
                              max_dev=max(rec['dev'][:k], default=0.0))
    max_dev = max(rec['dev'], default=0.0)
    log(f"walk: {'reached b' if reached else 'stuck (' + reason + ')'}, {left:.4f} D left, length/D "
        f"{sum(rec['seg']) / D:.5f}, held-out max|L-c| {max_dev:.4f}, {iters} gradient steps")
    if stash is not None:
        Path(stash).unlink(missing_ok=True)
    return dict(distance=D, n=len(rec['seg']), ratio=sum(rec['seg']) / D, reached=reached, stop=reason, left=left,
                max_dev=max_dev, ok=reached and max_dev <= taus[0], iters=iters, at=at, tol=tol,
                dense=dict(s=rec['s'], loss=rec['loss'], c=rec['c']), seg=rec['seg'], dev=rec['dev'],
                history=rec['hist'], settings=settings)


# ---------------------------------------------------------------- CLI

def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


_RUNNING = [0.0, None]


def running_jobs():
    """The user's running Slurm jobs (asked at most once a minute); None while Slurm cannot be reached."""
    if time.time() - _RUNNING[0] > 60:
        try:
            q = subprocess.run(['squeue', '-h', '-t', 'R', '-u', os.environ.get('USER', ''), '-o', '%i'],
                               capture_output=True, text=True, timeout=30)
            _RUNNING[1] = None if q.returncode else set(q.stdout.split())
        except (subprocess.TimeoutExpired, OSError):  # also: no Slurm here
            _RUNNING[1] = None
        _RUNNING[0] = time.time()
    return _RUNNING[1]


def claim(out):
    """True if this process may compute `out`: it does not exist and nobody else holds its lock. A lock names the
    Slurm job, host and process: it is held while that process runs (same job: several workers share a job) or, for
    another job, while that job runs or Slurm cannot tell (a pending, e.g. preempted, job holds none; a requeued job
    takes its locks back).
    Claims in one directory take turns (an atomic mkdir), so two workers cannot both take over a stale lock."""
    out = Path(out)
    if out.exists():
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    mutex, lock, running = out.parent / 'claims.mutex', out.with_suffix('.lock'), running_jobs()
    job, host = os.environ.get('SLURM_JOB_ID', 'local'), socket.gethostname()
    while True:
        try:
            mutex.mkdir()
            break
        except FileExistsError:
            try:
                if time.time() - mutex.stat().st_mtime > 120:  # left by a killed process
                    mutex.rmdir()
            except FileNotFoundError:
                pass
            time.sleep(0.05)
    try:
        if out.exists():
            return False
        if lock.exists():
            o_job, o_host, o_pid = (lock.read_text().strip().split(':') + ['', ''])[:3]
            if o_job == job:
                if o_host == host and o_pid.isdigit() and int(o_pid) != os.getpid() and _alive(int(o_pid)):
                    return False
            elif o_job not in ('', 'local') and (running is None or o_job in running):
                return False  # it runs, or Slurm cannot tell (then it is taken to run)
        lock.write_text(f'{job}:{host}:{os.getpid()}')
        return True
    finally:
        mutex.rmdir()


def release(out):
    Path(out).with_suffix('.lock').unlink(missing_ok=True)


def _setup_from(spec, args):
    run = Run(spec.rsplit(':', 1)[0])
    model, sets = setup(run.cfg, args.n_seqs, args.device, args.chunk, args.pool)
    check_order(model, spec, args.device)
    return model, sets


def cmd_losses(args):
    runs = load_runs(args.runs)
    model, (path_set, *_) = setup(runs[0].cfg, args.n_seqs, args.device, args.chunk)
    obj = Objective(model, path_set)
    out = {}
    for r in runs:
        steps = args.steps or r.ckpt_steps
        out[r.id] = {}
        for s in steps:
            if s in r.ckpt_steps:
                out[r.id][s] = obj.loss(load_theta(f'{r.id}:{s}', args.device))
                print(f'{r.id} {s}: {out[r.id][s]:.4f}', flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))


def scale_free(model, a, b, init):
    """|a - b| / mean(|a - init|, |b - init|), over all parameters and per layer (emb, 1..n, head)."""
    out, i = {}, 0
    parts = {}
    for name, p in model.named_parameters():
        parts.setdefault(layer(name), []).append(slice(i, i + p.numel()))
        i += p.numel()
    parts['all'] = [slice(0, i)]
    for k, sl in parts.items():
        sq = lambda v: float(sum(v[s].double().pow(2).sum() for s in sl)) ** 0.5
        out[k] = sq(a - b) / (0.5 * (sq(a - init) + sq(b - init)))
    return out


def do_profile(model, sets, sa_, sb_, args, out):
    path_set, held, probe = sets[:3]
    a, b = load_theta(sa_, args.device), load_theta(sb_, args.device)
    init = load_theta(sa_.rsplit(':', 1)[0] + ':0', args.device)
    obj = Objective(model, path_set)
    res = profile(obj, a, b, args.points)
    la, lb = obj.per_seq(a), obj.per_seq(b)
    res.update(a=sa_, b=sb_, n_seqs=args.n_seqs, se_mean=float(la.std() / math.sqrt(len(la))),
               se_pair=float((la - lb).std() / math.sqrt(len(la))),
               heldout=[Objective(model, held).loss(x) for x in (a, b)],
               probe=[Objective(model, probe).loss(x) for x in (a, b)],
               from_init=[float((a - init).norm()), float((b - init).norm())],
               rel_distance=scale_free(model, a, b, init))
    print(f"{sa_} vs {sb_}: c {res['c_a']:.4f} -> {res['c_b']:.4f}, rise {res['rise']:.4f}, "
          f"dip {res['dip']:.4f}, D {res['distance']:.2f}, D/init {res['rel_distance']['all']:.3f}, "
          f"pred ratio {res['pred_ratio']:.4f}, se_pair {res['se_pair']:.4f}", flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1))


def do_band(model, sets, sa_, sb_, args, out, n=None, perturb=0.0, perturb_seed=1, points_out=None):
    path_set, held, _, pool, data = sets
    n = n or args.n
    a, b = load_theta(sa_, args.device), load_theta(sb_, args.device)
    # iterations on the pool (a fresh group of sequences each), the dense check on the held-out set
    fast, held = Objective(model, path_set, fp32=False, groups=pool), Objective(model, held)
    # the string and shortening phases: fresh sequences at every step (batch_seqs and shorten_seqs of them)
    small = Objective(model, path_set[:max(1, args.batch_seqs // args.chunk)], fp32=False, amp=True,
                      groups=Stream(data, args.batch_seqs, args.chunk, 10 ** 6))
    mid = Objective(model, path_set[:max(1, args.shorten_seqs // args.chunk)], fp32=False, amp=True,
                    groups=Stream(data, args.shorten_seqs, args.chunk, 2 * 10 ** 6))
    ca, cb = fast.loss(a), fast.loss(b)
    init = None
    if perturb:  # a bent initial guess: straight line + perturb * D * sin(pi t) * (random unit direction)
        u = torch.randn(a.shape, generator=torch.Generator().manual_seed(perturb_seed)).to(a.device)
        t = torch.arange(1, n + 1, device=a.device)[:, None] / (n + 1)
        init = straight(a, b, n) + perturb * float((a - b).norm()) * torch.sin(math.pi * t) * u / u.norm()
    print(f'band {sa_} -> {sb_}, N={n}, perturb={perturb}', flush=True)
    X, res = band(fast, a, b, ca, cb, obj_check=held, ca_check=held.loss(a), cb_check=held.loss(b), n=n,
                  obj_string=small, steps=args.steps, lr=args.lr, beta=args.beta, ws=args.ws, iters=args.iters,
                  tol=args.tol, rounds=args.rounds, round_steps=args.round_steps, round_iters=args.round_iters,
                  m=args.m, init=init, obj_shorten=mid, shorten_steps=args.shorten_steps,
                  shorten_beta=args.shorten_beta, stash=Path(out).with_suffix('.stage.pt'),
                  resume_steps=args.resume_steps, accept=getattr(args, 'accept', None), log=say)
    pl = np.array([Objective(model, path_set).loss(x) for x in [a, *X, b]])  # the path set, at the points
    pc = pl[0] + (pl[-1] - pl[0]) * np.linspace(0, 1, len(pl))
    res.update(a=sa_, b=sb_, n_seqs=args.n_seqs, pool=args.pool, pathset=pl.tolist(), perturb=perturb,
               perturb_seed=perturb_seed, pathset_max_dev=float(np.abs(pl - pc).max()))
    f, v = res['fixed'], res['valid']
    print(f"fixed N={f['n']}: ratio {f['ratio']:.5f}, pred {f['pred_ratio']:.4f}, max dev {f['max_dev']:.4f}; "
          f"valid N={v['n']}: ratio {v['ratio']:.5f}, pred {v['pred_ratio']:.4f}, max dev {v['max_dev']:.4f} "
          f"(held-out set); path set max dev at the points {res['pathset_max_dev']:.4f}", flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1))
    if res['ok']:  # a path off the level keeps its saved points, to continue from
        Path(out).with_suffix('.stage.pt').unlink(missing_ok=True)
    if points_out:
        torch.save(dict(a=sa_, b=sb_, offsets=(X - straight(a, b, len(X))).half().cpu()), points_out)


def do_walk(model, sets, sa_, sb_, args, out):
    path_set, held, *_, data = sets
    a, b = load_theta(sa_, args.device), load_theta(sb_, args.device)
    # corrections on fresh groups of walk_seqs sequences, acceptance on fresh groups of n_seqs, the check held out
    k = lambda n: path_set[:max(1, n // args.chunk)]
    grad = Objective(model, k(args.walk_seqs), fp32=False, groups=Stream(data, args.walk_seqs, args.chunk, 3 * 10 ** 6))
    verify = Objective(model, k(args.n_seqs), fp32=False, groups=Stream(data, args.n_seqs, args.chunk, 4 * 10 ** 6))
    held = Objective(model, held)
    print(f'walk {sa_} -> {sb_}', flush=True)
    res = level_walk(grad, a, b, held, held.loss(a), held.loss(b), obj_verify=verify, tol=args.walk_tol,
                     step=args.walk_step, max_iters=args.walk_iters, m=args.walk_m,
                     stash=Path(out).with_suffix('.state.pt'), log=say)
    res.update(a=sa_, b=sb_, n_seqs=args.n_seqs, walk_seqs=args.walk_seqs)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1))


PAIRS = {'adamw-muon_d0': ('adamw_d0', 'muon_d0'), 'adamw-muon_d1': ('adamw_d1', 'muon_d1'),
         'adamw_d0-d1': ('adamw_d0', 'adamw_d1'), 'muon_d0-d1': ('muon_d0', 'muon_d1')}
FIRST = list(PAIRS)  # data orders 0 and 1; tier 6 adds orders 2-4: AdamW vs Muon per order, and all order pairs
PAIRS.update({f'adamw-muon_d{i}': (f'adamw_d{i}', f'muon_d{i}') for i in range(2, 5)})
PAIRS.update({f'{o}_d{i}-d{j}': (f'{o}_d{i}', f'{o}_d{j}') for o in ('adamw', 'muon') for i in range(5)
              for j in range(i + 1, 5) if (i, j) != (0, 1)})


def cmd_select(args):
    """For each level, the checkpoint of each run whose loss on the path set is closest to c. A level is a loss
    ('2.4': checkpoints of --suite) or 'RUN:STEP' (c is that checkpoint's loss; checkpoints of its suite)."""
    losses = {}
    for f in args.losses:
        losses.update(json.loads(Path(f).read_text()))
    out = {}
    for level in args.levels:
        if ':' in level:
            ref, step = level.split(':')
            suite, c = ref.split('/')[0], losses[ref][step]
        else:
            suite, c = args.suite, float(level)
        pick = {}
        for run in {r for pair in PAIRS.values() for r in pair}:
            steps = {int(s): l for s, l in losses[f'{suite}/{run}'].items() if int(s) > 0}
            s = min(steps, key=lambda s: abs(steps[s] - c))
            pick[run] = (f'{suite}/{run}:{s}', steps[s])
        out[f'{c:.3f}'] = dict(ref=level, c=c, pick=pick, pairs={k: [pick[x][0], pick[y][0], abs(pick[x][1] - pick[y][1])]
                                                       for k, (x, y) in PAIRS.items()})
        print(f'c={c:.4f}: ' + ', '.join(f'{r} {s.split(":")[1]} ({l - c:+.4f})' for r, (s, l) in sorted(pick.items())))
    Path(args.out).write_text(json.dumps(out, indent=1))


def cmd_prune(args):
    """Delete the checkpoints of the rerun suite that no level selected (keeps step 0, the last step and the picks)."""
    sel = json.loads(Path(args.select).read_text())
    keep = {}
    for level in sel.values():
        for spec, _ in level['pick'].values():
            run, step = spec.rsplit(':', 1)
            keep.setdefault(run, set()).add(int(step))
    for run, steps in keep.items():
        r = Run(run)
        assert r.id.startswith('ts10m_level_paths/'), f'prune only touches ts10m_level_paths, not {r.id}'
        last = r.cfg.train.max_steps
        for s in r.ckpt_steps:
            if s not in steps | {0, last}:
                r.ckpt_path(s).unlink()
        print(f'{run}: kept {sorted(steps | {0, last})}')


def cmd_run(args):
    """Profiles and bands for the selected pairs (skips outputs that exist). A level '#i' is the i-th level."""
    sel = json.loads(Path(args.select).read_text())
    out = Path(args.out_dir)
    model = sets = None
    levels = [list(sel)[int(l[1:])] if l.startswith('#') else l for l in args.levels or sel]
    for level in levels:
        for pair in args.pairs or PAIRS:
            sa_, sb_, _ = sel[level]['pairs'][pair]
            if model is None:
                model, sets = _setup_from(sa_, args)
            name = f'{level}_{pair}'
            f = out / 'profile' / f'{name}.json'
            if claim(f):
                do_profile(model, sets, sa_, sb_, args, f)
                release(f)
            if args.profiles_only:
                continue
            for n in args.band_n:
                f = out / 'band' / f'{name}_n{n}.json'
                if claim(f):
                    do_band(model, sets, sa_, sb_, args, f, n=n,
                            points_out=out / 'points' / f'{name}_n{n}.pt' if args.save_points else None)
                    release(f)
            for seed in args.perturb_seeds:
                f = out / 'band' / f'{name}_n{args.n}_perturb{args.perturb}_s{seed}.json'
                if claim(f):
                    do_band(model, sets, sa_, sb_, args, f, perturb=args.perturb, perturb_seed=seed)
                    release(f)
            f = out / 'walk' / f'{name}_pc.json'
            if args.walk and claim(f):
                do_walk(model, sets, sa_, sb_, args, f)
                release(f)


# the rerun levels, by the AdamW d0 step that defines them -> tier of their profiles and N=10 bands (the late level,
# from the ts10m_main checkpoints, is tier 1); 1 and 2: the six levels spread over training; 4, 5: extra levels
TIERS = {397: 1, 5000: 1, 1440: 2, 3300: 2, 7500: 2, 272: 4, 680: 4, 10000: 5}


def task_list(A, args):
    """The work of results/level_paths.tex in priority order, as (tier, kind, select file, output dir, level, pair, n,
    perturbation seed). The late level uses the ts10m_main checkpoints (select_main.json), the others the reruns
    (select.json; a level's 'ref' is the AdamW d0 checkpoint that defines it). Tiers 1-5 cover the FIRST pairs (data
    orders 0 and 1), tier 6 the other PAIRS: profiles, walks and N=20 bands (not at the late level) at the levels of
    tiers 1-4, once the select files pair them. Tier 1: profiles and N=10 bands at the
    late level and at AdamW steps 5000 and 397; tier 2: the same at steps 1440, 3300 and 7500; tier 3: N=20 bands,
    bent starts (step 1440, AdamW vs Muon, order 0) and walks at these six levels; tier 4: everything at steps 272
    and 680; tier 5: everything at the reruns' late level (step 10000). After tier 2: 'band01', the N=10 bands that
    missed tol, finished at tau = 0.1 from their saved points (no N=20 bands at the late level)."""
    main, rr = (A / 'select_main.json', A / 'main'), (A / 'select.json', A / 'rerun')
    sel = json.loads(rr[0].read_text()) if rr[0].exists() else {}
    step = {level: int(v['ref'].rsplit(':', 1)[1]) for level, v in sel.items()}
    by_tier = lambda k: sorted((l for l in sel if TIERS.get(step[l]) == k), key=lambda l: list(TIERS).index(step[l]))
    levels = {k: [(rr, l) for l in by_tier(k)] for k in (1, 2, 4, 5)}
    levels[1] = [(main, l) for l in json.loads(main[0].read_text())] + levels[1]
    six = levels[1] + levels[2]
    T = []
    for tier in (1, 2):
        for (f, d), level in levels[tier]:
            T += [(tier, 'profile', f, d, level, pair, None, None) for pair in FIRST]
            T += [(tier, 'band', f, d, level, pair, 10, None) for pair in FIRST]
    for (f, d), level in six:  # tau = 0.1 from the saved points of the N=10 bands that missed tol
        for pair in FIRST:
            g = d / 'band' / f'{level}_{pair}_n10.json'
            if g.exists() and g.with_suffix('.stage.pt').exists() and not json.loads(g.read_text())['ok']:
                T.append((2, 'band01', f, d, level, pair, 10, None))
    T += [(3, 'band', f, d, level, pair, 20, None) for (f, d), level in six if d != main[1] for pair in FIRST]
    T += [(3, 'band', *rr, l, 'adamw-muon_d0', args.n, seed) for l in sel if step[l] == 1440 for seed in (1, 2)]
    T += [(3, 'walk', f, d, level, pair, None, None) for (f, d), level in six for pair in FIRST]
    for tier in (4, 5):
        T += [(tier, kind, f, d, level, pair, n, None) for (f, d), level in levels[tier]
              for kind, n in (('profile', None), ('band', 10), ('band', 20), ('walk', None)) for pair in FIRST]
    new = [((f, d), level, pair) for (f, d), level in six + levels[4] for pair in PAIRS
           if pair not in FIRST and pair in json.loads(f.read_text())[level]['pairs']]
    T += [(6, 'profile', f, d, level, pair, None, None) for (f, d), level, pair in new]  # the cheap ones first
    T += [(6, 'walk', f, d, level, pair, None, None) for (f, d), level, pair in new]
    T += [(6, 'band', f, d, level, pair, 20, None) for (f, d), level, pair in new if d != main[1]]
    return T


def task_out(t, args):
    _, kind, _, d, level, pair, n, seed = t
    name = f'{level}_{pair}'
    if kind == 'profile':
        return d / 'profile' / f'{name}.json'
    if kind == 'walk':
        return d / 'walk' / f'{name}_pc.json'
    if kind == 'band01':
        return d / 'band_t0.1' / f'{name}_n{n}.json'
    return d / 'band' / (f'{name}_n{n}.json' if seed is None else f'{name}_n{n}_perturb{args.perturb}_s{seed}.json')


def reruns_alive(A):
    """Something is still making the reruns or the selection: a file of the reruns or the selection lock changed in
    the last 20 minutes."""
    recent = lambda f: time.time() - f.stat().st_mtime < 1200
    R = Path(os.environ['OPTSPLIT_ROOT']) / 'runs' / 'ts10m_level_paths'
    lock = A / 'select.lock'
    return (lock.exists() and recent(lock)) or any(recent(f) for f in R.rglob('*') if f.exists())


def _holder(out):
    """The Slurm job that holds the lock of output `out`, or None."""
    with contextlib.suppress(FileNotFoundError):
        return Path(out).with_suffix('.lock').read_text().split(':')[0]


def cmd_work(args):
    """One worker: repeatedly takes the first task of task_list (tiers <= --max-tier) that nobody holds and does it.
    While select.json is missing, waits (checking every minute that the reruns or the selection are progressing).
    Writes tier1_done when every tier-1 output exists. Exit 0: nothing left to take; 3: the reruns are not ready
    and nothing is making them. Restarts itself in place when asked (restart_if_asked), dropping its locks. With YIELD=1 in the
    environment, stops (exit 0) instead of taking a new task while a job of the user that is not a level-paths one
    waits, unheld, for this job's partition; with YIELD=limit, only while it waits for a per-user limit (QOS...,
    Assoc...: it would start if this job ended). A task that ran out of GPU memory (its .oom file holds the largest
    GPU it failed on) is left to larger GPUs. A GPU runs at most memory // --band-gb bands at once (two already keep
    an A100 busy), and its other workers take the cheaper tasks or wait."""
    global RESTART
    A = Path(args.dir)
    RESTART, model, sets = A / 'restart_workers', None, None
    me = f"{os.environ.get('SLURM_JOB_ID', 'local')}:{socket.gethostname()}:{os.getpid()}"
    if os.environ.get('SLURM_JOB_ID') and not torch.cuda.is_available():  # a broken node: do not work on the CPU
        sys.exit(f'FAILED: no CUDA on {socket.gethostname()}')
    gpu = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
    for t in task_list(A, args):  # locks this process took before a restart in place
        lock = Path(task_out(t, args)).with_suffix('.lock')
        with contextlib.suppress(FileNotFoundError):  # another worker may release it meanwhile
            if lock.read_text().strip() == me:
                lock.unlink()
    while True:
        restart_if_asked()
        if os.environ.get('YIELD') in ('1', 'limit'):
            q = subprocess.run(['squeue', '-u', os.environ.get('USER', ''), '-h', '-t', 'PD', '-p',
                                os.environ.get('SLURM_JOB_PARTITION', ''), '-o', '%j|%r'], capture_output=True, text=True)
            others = [l for l in q.stdout.splitlines() if not l.startswith(('lp_', 'level_paths'))
                      and not any(r in l for r in ('Held', 'Dependency', 'BeginTime'))
                      and (os.environ['YIELD'] == '1' or any(r in l for r in ('QOS', 'Assoc')))]
            if others:
                print(f'yielding the GPU: {others[0]} waits', flush=True)
                return 0
        T = [t for t in task_list(A, args)
             if args.min_tier <= t[0] <= args.max_tier and (not args.kinds or t[1] in args.kinds)]
        # bands of this job (one GPU) against the room for them: a band of up to 40 points takes about --band-gb
        job, cap = me.split(':')[0], max(1, int(gpu // (args.band_gb * 2 ** 30)))
        held = lambda: sum(_holder(task_out(u, args)) == job for u in T if u[1] == 'band')
        room = cap - held()
        short = False  # a band left for want of room: wait for it instead of exiting
        for t in T:
            f = task_out(t, args)
            oom = Path(f).with_suffix('.oom')
            if t[1] == 'band' and room <= 0 and not Path(f).exists():
                short = True
                continue
            if oom.exists() and float(oom.read_text()) >= gpu or not claim(f):
                continue
            if t[1] == 'band' and held() > cap:  # other workers of this job claimed bands meanwhile
                release(f)
                short = True
                continue
            _, kind, sel, _, level, pair, n, seed = t
            sa_, sb_, _ = json.loads(sel.read_text())[level]['pairs'][pair]
            if model is None:
                model, sets = _setup_from(sa_, args)
            print(f'task: tier {t[0]} {kind} {level} {pair} n={n} seed={seed} -> {f}', flush=True)
            try:
                if kind == 'profile':
                    do_profile(model, sets, sa_, sb_, args, f)
                elif kind == 'walk':
                    do_walk(model, sets, sa_, sb_, args, f)
                elif kind == 'band01':  # from the saved points of the tau = 0.02 band (a link, so it stays)
                    link = Path(f).with_suffix('.stage.pt')
                    if not link.is_symlink():
                        link.symlink_to(task_out((*t[:1], 'band', *t[2:]), args).with_suffix('.stage.pt'))
                    do_band(model, sets, sa_, sb_, argparse.Namespace(**{**vars(args), 'accept': 0.1, 'resume_steps': 0}),
                           f, n=n)
                else:
                    do_band(model, sets, sa_, sb_, args, f, n=n, perturb=args.perturb if seed else 0.0,
                            perturb_seed=seed or 1)
            except torch.OutOfMemoryError:  # the GPU is shared by the job's workers: leave the task to another
                print('out of GPU memory: this worker stops and leaves the task to larger GPUs', flush=True)
                oom.write_text(str(max(gpu, float(oom.read_text()) if oom.exists() else 0)))
                release(f)
                return 0
            release(f)
            if gpu:
                torch.cuda.empty_cache()  # an idle worker leaves the memory to the bands of the others
            if (A / 'select.json').exists() and not (A / 'tier1_done').exists() and \
                    all(task_out(u, args).exists() for u in task_list(A, args) if u[0] == 1):
                (A / 'tier1_done').write_text(time.strftime('%Y-%m-%d %H:%M:%S\n'))
                print('TIER 1 DONE', flush=True)
            break
        else:  # nothing to take now
            if short:
                time.sleep(60 + os.getpid() % 60)
                continue
            if (A / 'select.json').exists():
                return 0
            if not reruns_alive(A):
                print('reruns not ready and nothing is making them', flush=True)
                sys.exit(3)
            time.sleep(60)


STYLE = {'adamw-muon': dict(color='C0', ls='-', label='AdamW vs Muon'),  # the groups of pairs (see group)
         'adamw': dict(color='0.25', ls='-', label='AdamW vs AdamW'),
         'muon': dict(color='0.6', ls='-', label='Muon vs Muon')}


def group(pair):
    return 'adamw-muon' if pair.startswith('adamw-muon') else pair.split('_')[0]


def collect(out_dirs):
    """{pair: [record per level]} from the profiles and bands of out_dirs; the level's x is the mean training loss of
    the endpoints on the probe (the training loss of Fig. 1); 'step' is the AdamW d0 step that defines it."""
    recs = {}
    for f in sorted(f for d in out_dirs for f in (Path(d) / 'profile').glob('*.json')):
        out = f.parent.parent
        level, pair = f.stem.split('_', 1)
        prof = json.loads(f.read_text())
        r = dict(level=float(level), probe=float(np.mean(prof['probe'])), prof=prof)
        for n in (10, 20):
            g = out / 'band' / f'{f.stem}_n{n}.json'
            r[f'n{n}'] = json.loads(g.read_text()) if g.exists() else None
        g = out / 'band_t0.1' / f'{f.stem}_n10.json'
        r['n10_t01'] = json.loads(g.read_text()) if g.exists() else None
        g = out / 'walk' / f'{f.stem}_pc.json'
        r['walk'] = json.loads(g.read_text()) if g.exists() else None
        recs.setdefault(pair, []).append(r)
    step = {r['level']: int(r['prof']['a'].rsplit(':', 1)[1]) for rs in recs.values() for r in rs
            if '/adamw_d0:' in r['prof']['a']}
    for pair in recs:
        for r in recs[pair]:
            r['step'] = step.get(r['level'])
        recs[pair].sort(key=lambda r: -r['level'])
    return recs


def effort(res):
    """(string-method steps, best max |L - c| on the held-out set over the dense checks) of a band result."""
    steps, last = 0, 0
    for h in res['history']:
        if h.get('phase') == 'string':
            steps, last = (steps + h['t'] - last, h['t']) if h['t'] > last else (steps + h['t'], h['t'])
    return steps, min(max(c['worst']) for c in res['checks'])


def at_tol(r, tol):
    """(the shortest N=10 path of record r within tol, or None; the path whose held-out max |L - c| is reported: that
    one, else the pair's path with the smallest max |L - c|, or None): at 0.02 the band; at 0.1 also its continuation
    in band_t0.1/."""
    cands = [b for b in ((r['n10'], r.get('n10_t01')) if tol > 0.05 else (r['n10'],)) if b]
    b = min((b for b in cands if b['max_dev'] <= tol), key=lambda b: b['ratio'], default=None)
    return b, b or min(cands, key=lambda b: b['max_dev'], default=None)


def _series(ax, x, y, ok, st, marker='o'):
    ax.plot(x, y, ls=st['ls'], lw=0.8, color=st['color'])
    ax.plot(x[ok], y[ok], marker, ms=3, color=st['color'])
    ax.plot(x[~ok], y[~ok], marker, ms=3, mfc='w', color=st['color'])


def _lines(recs, f):
    """{group of STYLE: (x, y per panel, ok per panel, legend label)} against the level, from f(record) = (value per
    panel, valid?): per level, the mean over the group's pairs that have a value (NaN: none), valid where all of them
    are; the label gives the number n of pairs averaged (a range if it varies)."""
    out = {}
    for g in STYLE:
        rs = [r for pair, rr in recs.items() if group(pair) == g for r in rr]
        x, Y, OK, ns = [], [], [], set()
        for lv in sorted({r['level'] for r in rs}, reverse=True):
            at = [r for r in rs if r['level'] == lv]
            v = [f(r) for r in at]
            y, ok = np.array([a for a, _ in v], dtype=float), np.array([k for _, k in v])  # pairs x panels, pairs
            has = ~np.isnan(y)
            x.append(np.mean([r['probe'] for r in at]))
            Y.append([y[h, i].mean() if h.any() else np.nan for i, h in enumerate(has.T)])
            OK.append([ok[h].all() for h in has.T])
            ns |= {int(h.sum()) for h in has.T if h.any()}
        n = f'{min(ns)}\u2013{max(ns)}' if len(ns) > 1 else str(max(ns, default=0))
        out[g] = (np.array(x), np.array(Y).T, np.array(OK).T, f"{STYLE[g]['label']} (n={n})")
    return out


def _finish(fig, axes, recs, h, labels, out):
    """Fig. 1's x axis on axes (the level's training loss; on top, the level as a fraction of AdamW training), panel
    labels and the legend below; saves to out."""
    from optsplit.analysis.plots import panel_labels
    from optsplit.analysis.rank_metrics import loss_axis
    at = {}
    for rs in recs.values():
        for r in rs:
            if r['step']:
                at.setdefault(r['step'], []).append(r['probe'])
    xs = [r['probe'] for rs in recs.values() for r in rs]
    for ax in axes:
        loss_axis(ax)
        ax.set_xlim(max(xs) + 0.2, min(xs) - 0.02)
        ax.set_xlabel('training loss at the level')
        for i, s in enumerate(sorted(at)):  # the levels, as a fraction of AdamW training (10k steps), staggered
            ax.text(np.mean(at[s]), 1.01 + 0.06 * (i % 2), f'{round(s / 100)}%', transform=ax.get_xaxis_transform(),
                    ha='center', va='bottom', fontsize=5)
    panel_labels(axes)
    pad = 0.03 * 4.4 / fig.get_figheight()  # Fig. 1's space for the legend, whatever the figure's height
    fig.legend(h, labels, loc='lower center', ncol=len(h), frameon=False, bbox_to_anchor=(0.5, -pad))
    fig.tight_layout(rect=(0, pad, 1, 1))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches='tight')
    print(f'wrote {out}')


def cmd_plot(args):
    """Against the level, per group of pairs the mean over its pairs (_lines); open markers: max |L - c| > TAU.
    Fig. 1 (--out): the N=20 paths' length ratio in weights, held-out max |L - c| and length in weights;
    <out>_line.pdf (Fig. 2): the straight line's rise and distance / distance from init;
    <out>_robust.pdf (Fig. 3): the level-set walks' length ratio in weights (only those that reached b; triangles on a
    right axis: % of the distance left where a walk stopped short) and max |L - c|."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7, 'axes.titlepad': 14, 'axes.labelsize': 7,
                         'legend.fontsize': 6, 'xtick.labelsize': 6, 'ytick.labelsize': 6})
    recs = collect(args.out_dirs)

    def gap_axis(ax):  # log scale, the dotted TAU inside and labelled powers of 10, as in Fig. 1
        ax.set(yscale='log', title='max $|L - c|$ (held-out, nats)')
        ax.axhline(TAU, color='k', lw=0.5, ls=':')
        lo, hi = ax.get_ylim()
        ax.set_ylim(min(lo, 1e-3), max(hi, 2 * TAU))

    def fig1(r):  # the N=20 path's detour, gap and length; within TAU?
        b = r['n20']
        if b is None:
            return (np.nan, np.nan, np.nan), False
        return (b['ratio'] - 1, b['max_dev'], sum(b['seg'])), b['max_dev'] <= TAU

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.4))
    lines = _lines(recs, fig1)
    for k, (x, ys, oks, _) in lines.items():
        for ax, y, ok in zip(axes, ys, oks):
            _series(ax, x, y, ok, STYLE[k])
    axes[0].set(yscale='log', ylabel=f'$\\tau$ = {TAU:g}', title='length / distance $-$ 1 (weights)')
    gap_axis(axes[1])
    axes[2].set(title='path length (weights, $\\ell_2$)')
    names = lambda: [lines[k][3] for k in STYLE]
    marks = [plt.Line2D([], [], color='k', marker='o', ms=3, ls=''),
             plt.Line2D([], [], color='k', marker='o', ms=3, mfc='w', ls='')]
    h = [plt.Line2D([], [], color=STYLE[k]['color'], ls=STYLE[k]['ls'], lw=0.9) for k in STYLE]
    _finish(fig, axes.ravel(), recs, h + marks, names() + ['within $\\tau$', 'missed $\\tau$'], args.out)

    out = Path(args.out)
    h = [plt.Line2D([], [], color=STYLE[k]['color'], marker='o', ms=3, lw=0.9) for k in STYLE]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.4))
    lines = _lines(recs, lambda r: ((r['prof']['rise'], r['prof']['rel_distance']['all']), True))
    for k, (x, ys, oks, _) in lines.items():
        for ax, y, ok in zip(axes, ys, oks):
            _series(ax, x, y, ok, STYLE[k])
    axes[0].set(title='rise: max $L - c$ (nats)')
    axes[1].set(title='distance / mean distance from init')
    _finish(fig, axes, recs, h, names(), out.with_name(out.stem + '_line.pdf'))

    def fig3(r):  # the walk's detour (if it reached b), gap and % of the distance left (if it did not); within TAU?
        b = r['walk']
        if b is None:
            return (np.nan, np.nan, np.nan), False
        done = b['reached']
        return (b['ratio'] - 1 if done else np.nan, b['max_dev'], np.nan if done else 100 * b['left']), \
            b['max_dev'] <= TAU

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.4))
    left = axes[0].twinx()
    lines = _lines(recs, fig3)
    for k, (x, ys, oks, _) in lines.items():
        for ax, y, ok in zip(axes, ys, oks):
            _series(ax, x, y, ok, STYLE[k])
        _series(left, x, ys[2], oks[2], STYLE[k], '^')
    left.set(ylim=(0, 100), ylabel='distance left to $\\theta_B$ (%)')
    axes[0].set(yscale='log', ylabel='level-set walk', title='length / distance $-$ 1 (weights)')
    gap_axis(axes[1])
    _finish(fig, axes, recs, h + marks + [plt.Line2D([], [], color='k', marker='^', ms=3, ls='')],
            names() + [f'within $\\tau$ = {TAU:g}', 'missed $\\tau$', 'distance left (right axis)'],
            out.with_name(out.stem + '_robust.pdf'))


NAMES = {'adamw-muon_d0': 'AdamW vs Muon, order 0', 'adamw-muon_d1': 'AdamW vs Muon, order 1',
         'adamw_d0-d1': 'AdamW, orders 0 vs 1', 'muon_d0-d1': 'Muon, orders 0 vs 1'}


def cmd_table(args):
    """LaTeX rows, per level (decreasing loss) and pair: the straight line (rise, distance / distance from init), then
    the shortest N=10 path within TAU: length / distance - 1 (%), held-out max |L - c| and the string-method steps
    behind it (no path within TAU: the max |L - c| and steps of the run with the smallest max |L - c|). Then a
    second table: the N=20 band's length / distance - 1 (%) and max |L - c|, and the level-set walk's distance left to
    b (% of D; 0: reached), length / distance - 1 (%, if it reached b) and max |L - c|; values in parentheses missed
    TAU."""
    recs = collect(args.out_dirs)
    f = lambda v, d=2, ok=True: '--' if v is None else (f'{v:.{d}f}' if ok else f'({v:.{d}f})')
    short = {'adamw-muon_d0': 'AdamW--Muon, 0', 'adamw-muon_d1': 'AdamW--Muon, 1', 'adamw_d0-d1': 'AdamW, 0--1',
             'muon_d0-d1': 'Muon, 0--1'}
    rows, extra = [], []
    for level in sorted({r['level'] for rs in recs.values() for r in rs}, reverse=True):
        x = np.mean([r['probe'] for rs in recs.values() for r in rs if r['level'] == level])
        s = next((r['step'] for rs in recs.values() for r in rs if r['level'] == level and r['step']), None)
        first = True
        for pair in short:
            for r in (r for r in recs.get(pair, []) if r['level'] == level):
                p, (b, run) = r['prof'], at_tol(r, TAU)
                head = f'{x:.2f} ({round(s / 100)}\\%)' if first and s else ''
                first = False
                rows.append(' & '.join([head, short[pair], f(p['rise']), f(p['rel_distance']['all']),
                                        f(b and 100 * (b['ratio'] - 1)), f(run and run['max_dev'], 3),
                                        '--' if run is None else str(effort(run)[0])]) + r' \\')
                n20, w = r['n20'], r['walk']
                cells = [f(n20 and 100 * (n20['ratio'] - 1), 2, n20 is None or n20['max_dev'] <= TAU),
                         f(n20 and n20['max_dev'], 3)]
                done, ok = w is not None and w['reached'], w is None or w['max_dev'] <= TAU
                cells += [f(w and 100 * w['left'], 0), f(100 * (w['ratio'] - 1) if done else None, 2, ok),
                          f(w and w['max_dev'], 3)]
                extra.append(' & '.join([head, short[pair], *cells]) + r' \\')
        rows.append(r'\midrule')
        extra.append(r'\midrule')
    print('\n'.join(rows + ['% N=20 bands and level-set walks'] + extra))


def cmd_profile(args):
    model, sets = _setup_from(args.a, args)
    do_profile(model, sets, args.a, args.b, args, args.out)


def cmd_band(args):
    model, sets = _setup_from(args.a, args)
    do_band(model, sets, args.a, args.b, args, args.out, perturb=args.perturb, perturb_seed=args.perturb_seed,
            points_out=args.save_points)


def cmd_walk(args):
    model, sets = _setup_from(args.a, args)
    do_walk(model, sets, args.a, args.b, args, args.out)


def band_args(s):
    s.add_argument('--n', type=int, default=10)
    s.add_argument('--steps', type=int, default=6000, help='string-method steps')
    s.add_argument('--resume-steps', type=int, default=2500, help='string-method steps from saved points off the level')
    s.add_argument('--lr', type=float, default=2e-3, help='Adam step of the string method')
    s.add_argument('--beta', type=float, default=0.05, help='curve-shortening step of the string method')
    s.add_argument('--batch-seqs', type=int, default=128, help='sequences per string-method step')
    s.add_argument('--ws', type=float, nargs='+', default=[10], help='weights of the Levenberg-Marquardt stages')
    s.add_argument('--iters', type=int, default=10, help='Levenberg-Marquardt iterations per stage')
    s.add_argument('--tol', type=float, default=0.02)
    s.add_argument('--rounds', type=int, default=4)
    s.add_argument('--round-steps', type=int, default=300, help='string-method steps after an insertion')
    s.add_argument('--round-iters', type=int, default=10)
    s.add_argument('--m', type=int, default=8)
    s.add_argument('--shorten-steps', type=int, default=400, help='at most this many curve-shortening steps')
    s.add_argument('--shorten-beta', type=float, default=0.4, help='curve-shortening step (stable below 0.5)')
    s.add_argument('--shorten-seqs', type=int, default=256, help='sequences per curve-shortening step')
    s.add_argument('--points', type=int, default=51)
    s.add_argument('--walk-tol', type=float, default=0.01, help='the walk accepts a step within this of the level')
    s.add_argument('--walk-step', type=float, default=0.01, help="the walk's first step, as a fraction of D")
    s.add_argument('--walk-seqs', type=int, default=256, help='sequences per correction step of the walk')
    s.add_argument('--walk-iters', type=int, default=12000, help='at most this many correction steps per walk')
    s.add_argument('--walk-m', type=int, default=4, help='held-out check points inside each segment of the walk')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--n-seqs', type=int, default=512, help='training sequences in the path set and the held-out set')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--chunk', type=int, default=64, help='sequences per forward pass (memory)')
    p.add_argument('--pool', type=int, default=16, help='groups of n_seqs sequences that band and walk iterations '
                   'cycle through (1: the path set only); paths are checked on the held-out set')
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('losses')
    s.add_argument('runs', nargs='+')
    s.add_argument('--steps', type=int, nargs='*')
    s.add_argument('--out', required=True)
    s = sub.add_parser('select')
    s.add_argument('--losses', nargs='+', required=True)
    s.add_argument('--levels', nargs='+', required=True)
    s.add_argument('--suite', default='ts10m_level_paths')
    s.add_argument('--out', required=True)
    s = sub.add_parser('plot')
    s.add_argument('out_dirs', nargs='+')
    s.add_argument('--out', required=True)
    s = sub.add_parser('table')
    s.add_argument('out_dirs', nargs='+')
    s.add_argument('--levels', type=int, nargs='*')
    s = sub.add_parser('prune')
    s.add_argument('select')
    s = sub.add_parser('profile')
    s.add_argument('a')
    s.add_argument('b')
    band_args(s)
    s.add_argument('--out', required=True)
    s = sub.add_parser('band')
    s.add_argument('a')
    s.add_argument('b')
    band_args(s)
    s.add_argument('--perturb', type=float, default=0.0, help='initial bend, as a fraction of D (0: straight line)')
    s.add_argument('--perturb-seed', type=int, default=1)
    s.add_argument('--out', required=True)
    s.add_argument('--save-points')
    s = sub.add_parser('walk')
    s.add_argument('a')
    s.add_argument('b')
    band_args(s)
    s.add_argument('--out', required=True)
    s = sub.add_parser('work')
    s.add_argument('dir', help='the analysis directory (select_main.json, select.json, outputs)')
    s.add_argument('--max-tier', type=int, default=4)
    s.add_argument('--min-tier', type=int, default=1)
    s.add_argument('--band-gb', type=float, default=36, help='GPU memory per band: a GPU runs total // this at once')
    s.add_argument('--kinds', nargs='*', help='only these kinds of tasks (profile, band, band01, walk)')
    s.add_argument('--perturb', type=float, default=0.1)
    band_args(s)
    s = sub.add_parser('run')
    s.add_argument('select')
    s.add_argument('--out-dir', required=True)
    s.add_argument('--levels', nargs='*')
    s.add_argument('--pairs', nargs='*')
    s.add_argument('--profiles-only', action='store_true')
    s.add_argument('--band-n', type=int, nargs='*', default=[10, 20])
    s.add_argument('--perturb', type=float, default=0.1)
    s.add_argument('--perturb-seeds', type=int, nargs='*', default=[])
    s.add_argument('--save-points', action='store_true')
    s.add_argument('--walk', action='store_true', help='also the level-set walk')
    band_args(s)
    args = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    {'losses': cmd_losses, 'select': cmd_select, 'prune': cmd_prune, 'profile': cmd_profile, 'band': cmd_band,
     'run': cmd_run, 'walk': cmd_walk, 'work': cmd_work, 'plot': cmd_plot, 'table': cmd_table}[args.cmd](args)


if __name__ == '__main__':
    main()
