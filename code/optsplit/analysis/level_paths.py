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
python -m optsplit.analysis.level_paths band ts10m_main/adamw_d0:10000 ts10m_main/muon_d0:8000 --out b.json"""

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
         init=None, obj_shorten=None, shorten_steps=0, shorten_beta=0.4, stash=None, resume_steps=2500, log=print):
    """Elastic band between a and b at the losses ca -> cb (see the module docstring): `steps` steps of the string
    method on `obj_string` (default obj), then Levenberg-Marquardt stages at the weights `ws` on `obj`, then at most
    `shorten_steps` steps of curve shortening within the level set on `obj_shorten` (default obj); `obj_check`
    (default obj, with ca_check/cb_check) does the dense check. The points after the string and LM phases are saved
    to `stash` (if given), and a later call starts from there. If they are off the level (max |L - c| above 2 tol),
    the string method continues from them once without the springs (beta = 0), which cut corners off the level, with
    a quarter of the step size, warming up over 10% of `resume_steps` steps and annealing to 0 over the rest, and the
    LM stages run again; the points closer to the level are kept. Points still off the level get no insertion rounds
    (the result is invalid). Returns (points, result)."""
    obj_check = obj_check or obj
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

    def string(X, k, scale=1.0, hold=0.7, springs=True):
        """At most k steps of a stochastic string method, with Adam step size scale * lr: on a fresh group of
        sequences, each point takes an Adam step on (L_i - c_i)^2 / 2 (descent to the level set, like training; from
        below it climbs), then moves by beta times the discrete Laplacian of the path minus its component along the
        point's loss gradient (curve shortening within the level set; it also evens out the spacing). Both step sizes
        warm up over the first 10% of the steps and then decay linearly to 0: over the last 30% of the k steps or
        earlier, once the max |L - c| at the points on the check set (the better of two checks 50 steps apart, against
        occasional spikes) is below 2 tol (then over as many steps as taken so far) or has improved by less than 2%
        over 200 steps (a noise plateau above the level, as in training at constant lr: then over all the remaining
        steps, at least as many as taken so far). Adam's betas are training's (0.9, 0.95)."""
        mom, sq = torch.zeros_like(X), torch.zeros_like(X)
        warm, stop, decay, chk = 0.1 * k, int(hold * k), k - int(hold * k), []
        t = 0
        while t < stop + decay:
            t += 1
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
            X = X - f * scale * lr * (mom / (1 - 0.9 ** t)) / ((sq / (1 - 0.95 ** t)).sqrt() + 1e-8) + f * springs * beta * lap
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
        loss gradient, plus a Newton step back to its target loss. beta adapts like a trust region: if the max
        |L - c| after a step exceeds lim = max(tol, 1.5 x its value at the start), the step is undone and beta
        halved; after a step that keeps it below lim / 2, beta grows by 20% (up to shorten_beta; it starts at 1/16 of
        that). Stops once the length has fallen by less than 1% of its excess over the last 100 steps. Skipped for a
        path off the level (max |L - c| on the check set above 2 tol), where the Newton steps fail."""
        bs, ratios, prev, lim, undone = shorten_beta / 16, [float(lengths(X, a, b).sum() / D)], None, None, 0
        hist.append(dict(phase='shorten', n=len(X), t=0, beta=bs, check_max_r=check_dev(X), ratio=ratios[0],
                         time=time.time() - t0))
        if hist[-1]['check_max_r'] > 2 * tol:
            log(f"shorten: skipped, max|L-c| {hist[-1]['check_max_r']:.4f} on the check set")
            return X
        for t in range(1, k + 1):
            c = targets(*ends(obj_shorten), len(X))
            L, G = evaluate(obj_shorten, X)
            dev = float(np.abs(L - c).max())
            lim = lim or max(tol, 1.5 * dev)
            if prev is not None and dev > lim:  # the last step left the level set: undo it, smaller steps
                X, bs, undone = prev, bs / 2, undone + 1
                L, G = evaluate(obj_shorten, X)
            elif dev < lim / 2:
                bs = min(bs * 1.2, shorten_beta)
            prev = X
            P = torch.cat([a[None], X, b[None]])
            lap = P[:-2] + P[2:] - 2 * X
            gg = (G * G).sum(1, keepdim=True).clamp_min(1e-24)
            lap -= (lap * G).sum(1, keepdim=True) / gg * G
            X = X + bs * lap - torch.tensor(L - c, dtype=X.dtype, device=X.device)[:, None] * G / gg
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
    def string_lm(X, cont):
        if cont:  # from a path off the level, without the springs, which cut corners off it
            X = string(X, resume_steps, scale=0.25, hold=0.1, springs=False)
        elif steps:
            X = string(X, steps)
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
    if shorten_steps:
        X = shorten(X, shorten_steps)
        stages.append(dict(w='shorten', ratio=hist[-1]['ratio'], max_r=hist[-1].get('max_r')))
    def summary(X, worst, dense):
        seg = lengths(X, a, b)
        _, Jv, ends = walk(obj_check, [a, *X, b])
        return dict(n=len(X), ratio=float(seg.sum() / D), max_dev=float(np.max(worst)),
                    ok=bool(np.max(worst) <= tol), pred_ratio=dense['pred_ratio'],
                    pred_ratio_points=pred_length(Jv, ends)[1],
                    offline_max=float((X - straight(a, b, len(X))).norm(dim=1).max() / D))

    checks = []
    for r in range(rounds + 1):
        c_chk = np.concatenate([[ca_check], targets(ca_check, cb_check, len(X)), [cb_check]])
        worst, dense = dense_check(obj_check, X, a, b, c_chk, tol, m)
        checks.append(dict(n=len(X), worst=worst.tolist()))
        if r == 0:
            fixed = summary(X, worst, dense)  # fixed N, before any insertion
        bad = np.flatnonzero(worst > tol)
        log(f'check {r}: {len(X)} points, worst segment {worst.max():.4f} (tol {tol}), {len(bad)} segments out')
        if not len(bad) or r == rounds or len(X) + len(bad) > max_points:
            break
        if stages and stages[-1]['max_r'] > 2 * tol:
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
            X = string(X, round_steps, scale=0.25)  # a gentle continuation from the path so far
        X = run_stage(X, ws[-1], round_iters, ramp=round_iters // 2)
        if shorten_steps:
            X = shorten(X, shorten_steps)
    valid = summary(X, worst, dense)  # after the insertions
    res = dict(**valid, fixed=fixed, valid=valid, distance=D, seg=lengths(X, a, b).tolist(), tol=tol, c_a=ca_check,
               c_b=cb_check, dense=dense, history=hist, checks=checks, stages=stages,
               settings=dict(n=n, steps=steps, lr=lr, beta=beta, ws=list(ws), iters=iters, rounds=rounds,
                             round_steps=round_steps, round_iters=round_iters, m=m, shorten_steps=shorten_steps,
                             shorten_beta=shorten_beta))
    return X, res


# ---------------------------------------------------------------- level-set walk (second estimate)

def cg(A, v, iters):
    """Conjugate gradients for A x = v from x = 0; stops at negative curvature (returns the iterate so far)."""
    x, r = torch.zeros_like(v), v.clone()
    p, rr = r.clone(), float(r @ r)
    for _ in range(iters):
        Ap = A(p)
        pAp = float(p @ Ap)
        if pAp <= 0:
            break
        alpha = rr / pAp
        x += alpha * p
        r -= alpha * Ap
        rr_new = float(r @ r)
        if rr_new < 1e-12 * float(v @ v):
            break
        p, rr = r + (rr_new / rr) * p, rr_new
    return x if float(x @ x) > 0 else v


def level_walk(obj, a, b, ca, cb, obj_check=None, ca_check=None, cb_check=None, steps=80, mu=10.0, cg_iters=10,
               tol=0.02, corrections=5, m=8, log=print):
    """Walk from a to b on the level set (inspired by Hessian null-space continuation, Huang et al.): each step has
    length D/steps along d = (I + H/mu)^{-1} grad phi, phi = -|theta - b|^2 (by CG on Hessian-vector products, so
    directions of curvature >> mu are avoided), made perpendicular to the loss gradient; then up to `corrections`
    Newton steps theta -= (L - c) g / |g|^2 bring the loss back to c (interpolated by the progress towards b).
    The path is the polyline of the corrected points; it is an upper bound on the shortest constant-loss path where
    its dense check stays within tol."""
    obj_check = obj_check or obj
    ca_check = ca if ca_check is None else ca_check
    cb_check = cb if cb_check is None else cb_check
    D = float((a - b).norm())
    eta = D / steps
    pts, x, t0, hist = [], a.clone(), time.time(), []
    target = lambda x: ca + (cb - ca) * min(1.0, max(0.0, 1 - float((x - b).norm()) / D))
    for k in range(3 * steps):
        if float((x - b).norm()) <= eta:
            break
        if len(getattr(obj, 'groups', None) or []) > 1:  # a fresh group of sequences for each step
            obj.rotate()
            ca, cb = obj.loss(a), obj.loss(b)
        L, g = obj.loss_grad(x)
        v = (b - x) / (b - x).norm()
        d = cg(lambda u: u + obj.hvp(x, u) / mu, v, cg_iters)
        d = d - (d @ g) / (g @ g) * g
        x = x + eta * d / d.norm()
        for _ in range(corrections):
            L, g = obj.loss_grad(x)
            r = L - target(x)
            if abs(r) < tol / 4:
                break
            x = x - r * g / (g @ g)
        pts.append(x.clone())
        hist.append(dict(step=k, dist=float((x - b).norm()) / D, r=float(r), cos=float(d @ v / d.norm()),
                         time=time.time() - t0))
        log(f"walk {k}: |x - b|/D {hist[-1]['dist']:.4f}, L - c {r:+.4f}, cos(d, b - x) {hist[-1]['cos']:.3f} "
            f"({hist[-1]['time']:.0f}s)")
    X = torch.stack(pts)
    reached = float((X[-1] - b).norm()) <= 1.5 * eta
    s_pts = np.concatenate([[0], np.cumsum(lengths(X, a, b))])
    c_chk = ca_check + (cb_check - ca_check) * s_pts / s_pts[-1]  # c by arclength along the walk
    worst, dense = dense_check(obj_check, X, a, b, c_chk, tol, m)
    _, Jv, ends = walk(obj_check, [a, *X, b])
    res = dict(distance=D, n=len(X), ratio=float(s_pts[-1] / D), reached=reached, max_dev=float(worst.max()),
               ok=bool(worst.max() <= tol), pred_ratio=dense['pred_ratio'], pred_ratio_points=pred_length(Jv, ends)[1],
               tol=tol, dense=dense, history=hist, settings=dict(steps=steps, mu=mu, cg_iters=cg_iters,
                                                                 corrections=corrections, m=m))
    return X, res


# ---------------------------------------------------------------- CLI

def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


def claim(out):
    """True if this process may compute `out`: it does not exist and nobody else holds its lock. A lock names the
    Slurm job, host and process: it is held while that process runs (same job: several workers share a job) or, for
    another job, while that job runs (a pending, e.g. preempted, job holds none; a requeued job takes its locks back).
    Claims in one directory take turns (an atomic mkdir), so two workers cannot both take over a stale lock."""
    out = Path(out)
    if out.exists():
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    mutex, lock = out.parent / 'claim.mutex', out.with_suffix('.lock')
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
            elif o_job not in ('', 'local') and subprocess.run(['squeue', '-h', '-t', 'R', '-j', o_job],
                                                               capture_output=True, text=True).stdout.strip():
                return False
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
                  resume_steps=args.resume_steps, log=say)
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
    path_set, held, _, pool = sets[:4]
    a, b = load_theta(sa_, args.device), load_theta(sb_, args.device)
    fast, held = Objective(model, path_set, fp32=False, groups=pool), Objective(model, held)
    print(f'walk {sa_} -> {sb_}, mu={args.mu}', flush=True)
    X, res = level_walk(fast, a, b, fast.loss(a), fast.loss(b), obj_check=held, ca_check=held.loss(a),
                        cb_check=held.loss(b), steps=args.walk_steps, mu=args.mu, cg_iters=args.cg_iters, tol=args.tol,
                        m=args.m, log=say)
    res.update(a=sa_, b=sb_, n_seqs=args.n_seqs)
    print(f"walk: ratio {res['ratio']:.5f}, reached {res['reached']}, pred {res['pred_ratio']:.4f}, "
          f"max dev {res['max_dev']:.4f}", flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1))


PAIRS = {'adamw-muon_d0': ('adamw_d0', 'muon_d0'), 'adamw-muon_d1': ('adamw_d1', 'muon_d1'),
         'adamw_d0-d1': ('adamw_d0', 'adamw_d1'), 'muon_d0-d1': ('muon_d0', 'muon_d1')}


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
            f = out / 'walk' / f'{name}_mu{args.mu:g}.json'
            if args.walk and claim(f):
                do_walk(model, sets, sa_, sb_, args, f)
                release(f)


# the rerun levels, by the AdamW d0 step that defines them -> tier of their profiles and N=10 bands (the late level,
# from the ts10m_main checkpoints, is tier 1); 1 and 2: the six levels spread over training; 4, 5: extra levels
TIERS = {397: 1, 5000: 1, 1440: 2, 3300: 2, 7500: 2, 272: 4, 680: 4, 10000: 5}


def task_list(A, args):
    """The work of results/level_paths.tex in priority order, as (tier, kind, select file, output dir, level, pair, n,
    perturbation seed). The late level uses the ts10m_main checkpoints (select_main.json), the others the reruns
    (select.json; a level's 'ref' is the AdamW d0 checkpoint that defines it). Tier 1: profiles and N=10 bands at the
    late level and at AdamW steps 5000 and 397; tier 2: the same at steps 1440, 3300 and 7500; tier 3: N=20 bands,
    bent starts (step 1440, AdamW vs Muon, order 0) and walks at these six levels; tier 4: everything at steps 272
    and 680; tier 5: everything at the reruns' late level (step 10000)."""
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
            T += [(tier, 'profile', f, d, level, pair, None, None) for pair in PAIRS]
            T += [(tier, 'band', f, d, level, pair, 10, None) for pair in PAIRS]
    T += [(3, 'band', f, d, level, pair, 20, None) for (f, d), level in six for pair in PAIRS]
    T += [(3, 'band', *rr, l, 'adamw-muon_d0', args.n, seed) for l in sel if step[l] == 1440 for seed in (1, 2)]
    T += [(3, 'walk', f, d, level, pair, None, None) for (f, d), level in six for pair in PAIRS]
    for tier in (4, 5):
        T += [(tier, kind, f, d, level, pair, n, None) for (f, d), level in levels[tier]
              for kind, n in (('profile', None), ('band', 10), ('band', 20), ('walk', None)) for pair in PAIRS]
    return T


def task_out(t, args):
    _, kind, _, d, level, pair, n, seed = t
    name = f'{level}_{pair}'
    if kind == 'profile':
        return d / 'profile' / f'{name}.json'
    if kind == 'walk':
        return d / 'walk' / f'{name}_mu{args.mu:g}.json'
    return d / 'band' / (f'{name}_n{n}.json' if seed is None else f'{name}_n{n}_perturb{args.perturb}_s{seed}.json')


def reruns_alive(A):
    """Something is still making the reruns or the selection: a file of the reruns or the selection lock changed in
    the last 20 minutes."""
    recent = lambda f: time.time() - f.stat().st_mtime < 1200
    R = Path(os.environ['OPTSPLIT_ROOT']) / 'runs' / 'ts10m_level_paths'
    lock = A / 'select.lock'
    return (lock.exists() and recent(lock)) or any(recent(f) for f in R.rglob('*') if f.exists())


def cmd_work(args):
    """One worker: repeatedly takes the first task of task_list (tiers <= --max-tier) that nobody holds and does it.
    While select.json is missing, waits (checking every minute that the reruns or the selection are progressing).
    Writes tier1_done when every tier-1 output exists. Exit 0: nothing left to take; 3: the reruns are not ready
    and nothing is making them. Restarts itself in place when asked (restart_if_asked), dropping its locks. With YIELD=1 in the
    environment, stops (exit 0) instead of taking a new task while a job of the user that is not a level-paths one
    waits, unheld, for this job's partition; with YIELD=limit, only while it waits for a per-user limit (QOS...,
    Assoc...: it would start if this job ended)."""
    global RESTART
    A = Path(args.dir)
    RESTART, model, sets = A / 'restart_workers', None, None
    me = f"{os.environ.get('SLURM_JOB_ID', 'local')}:{socket.gethostname()}:{os.getpid()}"
    for t in task_list(A, args):  # locks this process took before a restart in place
        lock = Path(task_out(t, args)).with_suffix('.lock')
        if lock.exists() and lock.read_text().strip() == me:
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
        T = [t for t in task_list(A, args) if t[0] <= args.max_tier]
        for t in T:
            f = task_out(t, args)
            if not claim(f):
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
                else:
                    do_band(model, sets, sa_, sb_, args, f, n=n, perturb=args.perturb if seed else 0.0,
                            perturb_seed=seed or 1)
            except torch.OutOfMemoryError:  # the GPU is shared by the job's workers: leave the task to another
                print('out of GPU memory: this worker stops and leaves the task to others', flush=True)
                release(f)
                return 0
            release(f)
            if (A / 'select.json').exists() and not (A / 'tier1_done').exists() and \
                    all(task_out(u, args).exists() for u in task_list(A, args) if u[0] == 1):
                (A / 'tier1_done').write_text(time.strftime('%Y-%m-%d %H:%M:%S\n'))
                print('TIER 1 DONE', flush=True)
            break
        else:  # nothing to take now
            if (A / 'select.json').exists():
                return 0
            if not reruns_alive(A):
                print('reruns not ready and nothing is making them', flush=True)
                sys.exit(3)
            time.sleep(60)


STYLE = {'adamw-muon_d0': dict(color='C0', label='AdamW vs Muon, same data order'),
         'adamw-muon_d1': dict(color='C0'),
         'adamw_d0-d1': dict(color='0.25', label='AdamW, two data orders'),
         'muon_d0-d1': dict(color='0.6', label='Muon, two data orders')}


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
        g = sorted((out / 'walk').glob(f'{f.stem}_mu*.json'))
        r['walk'] = json.loads(g[0].read_text()) if g else None
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


def cmd_plot(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from optsplit.analysis.rank_metrics import loss_axis
    plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7, 'axes.titlepad': 9, 'axes.labelsize': 7,
                         'legend.fontsize': 6, 'xtick.labelsize': 6, 'ytick.labelsize': 6})
    recs = collect(args.out_dirs)
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 3.9))
    axes = axes.ravel()
    for pair, rs in recs.items():
        st = STYLE[pair]
        x = np.array([r['probe'] for r in rs])
        val = lambda f: np.array([np.nan if v is None else v for v in map(f, rs)], dtype=float)
        up = val(lambda r: r['n10'] and r['n10']['valid']['ratio'])
        lo = val(lambda r: r['n20'] and r['n20']['fixed']['ratio'])
        axes[0].plot(x, up - 1, 'o-', ms=2.5, lw=0.9, **st)
        axes[0].vlines(x, lo - 1, up - 1, color=st['color'], lw=0.6)
        axes[0].plot(x, lo - 1, '_', ms=4, color=st['color'])
        axes[1].plot(x, [r['prof']['rise'] for r in rs], 'o-', ms=2.5, lw=0.9, color=st['color'])
        axes[1].plot(x, [-r['prof']['dip'] for r in rs], 'o--', ms=2.5, lw=0.9, color=st['color'], mfc='w')
        axes[2].plot(x, val(lambda r: r['n10'] and r['n10']['valid']['pred_ratio']), 'o-', ms=2.5, lw=0.9,
                     color=st['color'])
        axes[2].plot(x, [r['prof']['pred_ratio'] for r in rs], 'o--', ms=2.5, lw=0.9, color=st['color'], mfc='w')
        axes[3].plot(x, [r['prof']['rel_distance']['all'] for r in rs], 'o-', ms=2.5, lw=0.9, color=st['color'])
    axes[0].set(yscale='log', title='constant-loss path: length / distance $-$ 1')
    axes[1].axhline(0, color='k', lw=0.5)
    axes[1].set(title='straight line: rise and dip (nats)')
    axes[2].set(yscale='log', title='prediction space: length / distance')
    axes[3].set(title='distance / distance from init')
    at = {}
    for rs in recs.values():
        for r in rs:
            if r['step']:
                at.setdefault(r['step'], []).append(r['probe'])
    xs = [r['probe'] for rs in recs.values() for r in rs]
    for ax in axes:
        loss_axis(ax)
        ax.set_xlim(max(xs) + 0.2, min(xs) - 0.02)
        ax.set_xlabel('training loss')
        for s, v in at.items():  # the levels, as a fraction of AdamW training (10k steps)
            ax.text(np.mean(v), 1.01, f'{round(s / 100)}%', transform=ax.get_xaxis_transform(), ha='center',
                    va='bottom', fontsize=5)
    from optsplit.analysis.plots import panel_labels
    panel_labels(axes)
    h = [plt.Line2D([], [], color=STYLE[k]['color'], lw=0.9) for k in ('adamw-muon_d0', 'adamw_d0-d1', 'muon_d0-d1')]
    fig.legend(h, [STYLE[k]['label'] for k in ('adamw-muon_d0', 'adamw_d0-d1', 'muon_d0-d1')], loc='lower center',
               ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches='tight')
    print(f'wrote {args.out}')


NAMES = {'adamw-muon_d0': 'AdamW vs Muon, order 0', 'adamw-muon_d1': 'AdamW vs Muon, order 1',
         'adamw_d0-d1': 'AdamW, orders 0 vs 1', 'muon_d0-d1': 'Muon, orders 0 vs 1'}


def cmd_table(args):
    """LaTeX rows: per level (the given ones, by index in the levels sorted by decreasing loss, or all) and pair. The
    first column is the level's training loss (mean over the pairs' endpoints on the probe) and AdamW step."""
    recs = collect(args.out_dirs)
    f = lambda v, d=3: '--' if v is None else f'{v:.{d}f}'
    levels = sorted({r['level'] for rs in recs.values() for r in rs}, reverse=True)
    keep = [levels[i] for i in args.levels] if args.levels else levels
    for level in keep:
        x = np.mean([r['probe'] for rs in recs.values() for r in rs if r['level'] == level])
        s = next((r['step'] for rs in recs.values() for r in rs if r['level'] == level and r['step']), None)
        for pair in NAMES:
            for r in recs.get(pair, []):
                if r['level'] != level:
                    continue
                n10, n20, p = r['n10'], r['n20'], r['prof']
                print(' & '.join([
                    f'{x:.2f}', '--' if s is None else f'{s} ({round(s / 100)}\\%)', NAMES[pair],
                    f(abs(p['c_a'] - p['c_b'])), f(p['rise'], 2), f(abs(p['dip']), 2),
                    f(p['rel_distance']['all'], 2), f(n10 and n10['fixed']['ratio'], 4),
                    f(n20 and n20['fixed']['ratio'], 4),
                    f(n10 and n10['valid']['ratio'], 4) + (f" ({n10['valid']['n']})" if n10 else ''),
                    f(n10 and n10['valid']['max_dev']), f(p['pred_ratio'], 2),
                    f(n10 and n10['valid']['pred_ratio'], 2),
                    f(n10 and effort(n10)[1]), '--' if n10 is None else str(effort(n10)[0])]) + r' \\')


def cmd_profile(args):
    model, sets = _setup_from(args.a, args)
    do_profile(model, sets, args.a, args.b, args, args.out)


def cmd_band(args):
    model, sets = _setup_from(args.a, args)
    do_band(model, sets, args.a, args.b, args, args.out, perturb=args.perturb, perturb_seed=args.perturb_seed,
            points_out=args.save_points)


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
    s.add_argument('--walk-steps', type=int, default=80)
    s.add_argument('--mu', type=float, default=10.0)
    s.add_argument('--cg-iters', type=int, default=10)


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
    s = sub.add_parser('work')
    s.add_argument('dir', help='the analysis directory (select_main.json, select.json, outputs)')
    s.add_argument('--max-tier', type=int, default=4)
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
     'run': cmd_run, 'work': cmd_work, 'plot': cmd_plot, 'table': cmd_table}[args.cmd](args)


if __name__ == '__main__':
    main()
