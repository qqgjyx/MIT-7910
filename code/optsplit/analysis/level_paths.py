"""Constant-loss paths between two checkpoints at (nearly) equal training loss c (results/level_paths.tex).

- `losses`: loss of checkpoints on the fixed path set (to pick, for each run, the checkpoint closest to a level).
- `profile`: loss along the straight line (1-t) theta_A + t theta_B; its highest rise above c and deepest dip below.
- `band`: elastic band. N points theta_1..theta_N between the fixed endpoints theta_0 = theta_A, theta_{N+1} = theta_B
  minimise E_w = w sum_i ((L_i - c_i) / tol)^2 + sum_k |theta_{k+1} - theta_k|^2 / (D^2 / (N+1)), D = |theta_A - theta_B|,
  c_i interpolated linearly between the endpoint losses. Each iteration is a Levenberg-Marquardt step on E_w (exact
  for the springs, linearised loss) on a fresh group of training sequences (so that the path cannot fit one fixed
  set); w grows over stages. Then the loss on a held-out set is checked densely along every segment; segments that
  leave c +- tol get a midpoint and the band continues. Output: path length / D and the loss along the path.

python -m optsplit.analysis.level_paths losses 'ts10m_main/adamw_d0' --steps 9000 10000 --out f.json
python -m optsplit.analysis.level_paths profile ts10m_main/adamw_d0:10000 ts10m_main/muon_d0:8000 --out p.json
python -m optsplit.analysis.level_paths band ts10m_main/adamw_d0:10000 ts10m_main/muon_d0:8000 --out b.json"""

import argparse
import contextlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from optsplit.analysis.distance import js_divergence
from optsplit.analysis.io import Run, load_runs
from optsplit.loss import log_probs, mean_loss
from optsplit.models import build_model, layer

PATH_SEED, HELDOUT_SEED = 101, 102  # fixed training sequences (the probe uses seed 0)


class Objective:
    """Mean loss on fixed batches as a function of the flat parameter vector of `model`. With `groups` (several lists
    of batches), rotate() moves to the next group: the band and the walk take a fresh group at every iteration, so
    that the path cannot fit one fixed set of sequences."""

    def __init__(self, model, batches, fp32=True, groups=None):
        self.model, self.batches, self.fp32 = model, batches, fp32
        self.params = list(model.parameters())
        self.groups, self.g = groups, 0

    def rotate(self):
        self.g = (self.g + 1) % len(self.groups)
        self.batches = self.groups[self.g]

    @contextlib.contextmanager
    def _precision(self):
        """fp32=True: exact float32 matmuls (checks); False: TF32 (band iterations, faster)."""
        old = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = not self.fp32
        try:
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
    n_seqs sequences, which the band and walk iterations cycle through."""
    from optsplit.data import build_data
    data = build_data(cfg, device)
    model = build_model(cfg, vocab_size=data.vocab_size).to(device).eval()
    sets = []
    seeds = [(n_seqs, PATH_SEED), (n_seqs, HELDOUT_SEED), (cfg.probe.train_seqs, 0)]
    for n, seed in seeds + [(n_seqs, 1000 + j) for j in range(1, pool)]:
        x, y = data.fixed_batch('train', n, seed)
        sets.append([(x[i:i + chunk], y[i:i + chunk]) for i in range(0, n, chunk)])
    return model, sets[:3] + [sets[:1] + sets[3:]]


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


def band(obj, a, b, ca, cb, obj_check=None, ca_check=None, cb_check=None, n=10, ws=(1e-4, 1e-3, 1e-2, 1e-1, 1, 10),
         iters=20, tol=0.02, rounds=4, round_iters=10, m=8, max_points=60, init=None, log=print):
    """Elastic band between a and b at the losses ca -> cb (see the module docstring). `obj` drives the iterations,
    `obj_check` (default obj, with ca_check/cb_check) the dense check. Returns (points, result dict)."""
    obj_check = obj_check or obj
    ca_check = ca if ca_check is None else ca_check
    cb_check = cb if cb_check is None else cb_check
    D = float((a - b).norm())
    X = straight(a, b, n) if init is None else init.clone()
    hist, t0 = [], time.time()
    rotating = len(getattr(obj, 'groups', None) or []) > 1

    def ends():  # endpoint losses on the current group of sequences (after moving to the next one)
        if not rotating:
            return ca, cb
        obj.rotate()
        return obj.loss(a), obj.loss(b)

    lam = [1.0]  # Levenberg-Marquardt damping, adapted: /3 after an accepted step, x4 after a rejected one

    def run_stage(X, w, k, ramp=0):
        """k iterations at weight w. With ramp > 0 the targets start at the current losses and reach c_i linearly
        over the first `ramp` iterations (a continuation, so that each step's linearisation stays accurate)."""
        n, dev0 = len(X), None
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
            if s == 1 and E - En < 1e-4 * E:
                break  # converged at this w
        return X

    stages = []
    for w in ws:
        X = run_stage(X, w, iters, ramp=iters // 2)
        if hist:
            stages.append(dict(w=w, ratio=hist[-1]['ratio'], max_r=hist[-1]['max_r']))
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
        P = [a, *X, b]
        new = []
        for k in range(len(P) - 1):
            if k > 0:
                new.append(P[k])
            if k in bad:
                new.append((P[k] + P[k + 1]) / 2)
        X = run_stage(torch.stack(new), ws[-1], round_iters, ramp=round_iters // 2)
    valid = summary(X, worst, dense)  # after the insertions
    res = dict(**valid, fixed=fixed, valid=valid, distance=D, seg=lengths(X, a, b).tolist(), tol=tol, c_a=ca_check,
               c_b=cb_check, dense=dense, history=hist, checks=checks, stages=stages,
               settings=dict(n=n, ws=list(ws), iters=iters, rounds=rounds, round_iters=round_iters, m=m))
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

def claim(out):
    """True if this job may compute `out`: it does not exist and no other running Slurm job holds its lock (so that
    several jobs can share the list of outputs; a requeued job keeps its id and takes its own lock back)."""
    out = Path(out)
    if out.exists():
        return False
    lock, me = out.with_suffix('.lock'), os.environ.get('SLURM_JOB_ID', 'local')
    if lock.exists():
        other = lock.read_text().strip()
        if other != me and other != 'local' and subprocess.run(['squeue', '-h', '-j', other], capture_output=True,
                                                                text=True).stdout.strip():
            return False
    out.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(me)
    return True


def release(out):
    Path(out).with_suffix('.lock').unlink(missing_ok=True)


def _setup_from(spec, args):
    run = Run(spec.rsplit(':', 1)[0])
    model, sets = setup(run.cfg, args.n_seqs, args.device, args.chunk, args.pool)
    check_order(model, spec, args.device)
    return model, sets


def cmd_losses(args):
    runs = load_runs(args.runs)
    model, (path_set, _, _, _) = setup(runs[0].cfg, args.n_seqs, args.device, args.chunk)
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
    path_set, held, probe, _ = sets
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
    path_set, held, _, pool = sets
    n = n or args.n
    a, b = load_theta(sa_, args.device), load_theta(sb_, args.device)
    # iterations on the pool (a fresh group of sequences each), the dense check on the held-out set
    fast, held = Objective(model, path_set, fp32=False, groups=pool), Objective(model, held)
    ca, cb = fast.loss(a), fast.loss(b)
    init = None
    if perturb:  # a bent initial guess: straight line + perturb * D * sin(pi t) * (random unit direction)
        u = torch.randn(a.shape, generator=torch.Generator().manual_seed(perturb_seed)).to(a.device)
        t = torch.arange(1, n + 1, device=a.device)[:, None] / (n + 1)
        init = straight(a, b, n) + perturb * float((a - b).norm()) * torch.sin(math.pi * t) * u / u.norm()
    print(f'band {sa_} -> {sb_}, N={n}, perturb={perturb}', flush=True)
    X, res = band(fast, a, b, ca, cb, obj_check=held, ca_check=held.loss(a), cb_check=held.loss(b), n=n,
                  ws=args.ws, iters=args.iters, tol=args.tol, rounds=args.rounds, round_iters=args.round_iters,
                  m=args.m, init=init, log=lambda s: print(s, flush=True))
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
    if points_out:
        torch.save(dict(a=sa_, b=sb_, offsets=(X - straight(a, b, len(X))).half().cpu()), points_out)


def do_walk(model, sets, sa_, sb_, args, out):
    path_set, held, _, pool = sets
    a, b = load_theta(sa_, args.device), load_theta(sb_, args.device)
    fast, held = Objective(model, path_set, fp32=False, groups=pool), Objective(model, held)
    print(f'walk {sa_} -> {sb_}, mu={args.mu}', flush=True)
    X, res = level_walk(fast, a, b, fast.loss(a), fast.loss(b), obj_check=held, ca_check=held.loss(a),
                        cb_check=held.loss(b), steps=args.walk_steps, mu=args.mu, cg_iters=args.cg_iters, tol=args.tol,
                        m=args.m, log=lambda s: print(s, flush=True))
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
        out[f'{c:.3f}'] = dict(c=c, pick=pick, pairs={k: [pick[x][0], pick[y][0], abs(pick[x][1] - pick[y][1])]
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


STYLE = {'adamw-muon_d0': dict(color='C0', label='AdamW vs Muon, same data order'),
         'adamw-muon_d1': dict(color='C0'),
         'adamw_d0-d1': dict(color='0.25', label='AdamW, two data orders'),
         'muon_d0-d1': dict(color='0.6', label='Muon, two data orders')}


def collect(out_dir):
    """{pair: [record per level]} from the profiles and bands of out_dir; the level's x is the mean training loss of
    the endpoints on the probe (the training loss of Fig. 1)."""
    out = Path(out_dir)
    recs = {}
    for f in sorted((out / 'profile').glob('*.json')):
        level, pair = f.stem.split('_', 1)
        prof = json.loads(f.read_text())
        r = dict(level=float(level), probe=float(np.mean(prof['probe'])), prof=prof)
        for n in (10, 20):
            g = out / 'band' / f'{f.stem}_n{n}.json'
            r[f'n{n}'] = json.loads(g.read_text()) if g.exists() else None
        g = sorted((out / 'walk').glob(f'{f.stem}_mu*.json'))
        r['walk'] = json.loads(g[0].read_text()) if g else None
        recs.setdefault(pair, []).append(r)
    for pair in recs:
        recs[pair].sort(key=lambda r: -r['level'])
    return recs


def cmd_plot(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from optsplit.analysis.rank_metrics import loss_axis
    plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7, 'axes.labelsize': 7, 'legend.fontsize': 6,
                         'xtick.labelsize': 6, 'ytick.labelsize': 6})
    recs = collect(args.out_dir)
    fig, axes = plt.subplots(1, 4, figsize=(7.1, 1.75))
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
    for ax in axes:
        loss_axis(ax)
        ax.set_xlabel('training loss')
    from optsplit.analysis.plots import panel_labels
    panel_labels(axes)
    h = [plt.Line2D([], [], color=STYLE[k]['color'], lw=0.9) for k in ('adamw-muon_d0', 'adamw_d0-d1', 'muon_d0-d1')]
    fig.legend(h, [STYLE[k]['label'] for k in ('adamw-muon_d0', 'adamw_d0-d1', 'muon_d0-d1')], loc='lower center',
               ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches='tight')
    print(f'wrote {args.out}')


NAMES = {'adamw-muon_d0': 'AdamW vs Muon, order 0', 'adamw-muon_d1': 'AdamW vs Muon, order 1',
         'adamw_d0-d1': 'AdamW, orders 0 vs 1', 'muon_d0-d1': 'Muon, orders 0 vs 1'}


def cmd_table(args):
    """LaTeX rows: per level (the given ones, by index in the sorted levels, or all) and pair."""
    recs = collect(args.out_dir)
    f = lambda v, d=3: '--' if v is None else f'{v:.{d}f}'
    rows = []
    for pair, rs in recs.items():
        for r in rs:
            n10, n20 = r['n10'], r['n20']
            rows.append((r['probe'], pair, ' & '.join([
                f"{r['probe']:.2f}", NAMES[pair], f(abs(r['prof']['c_a'] - r['prof']['c_b'])),
                f(r['prof']['rise'], 2), f(r['prof']['dip'], 2), f(r['prof']['rel_distance']['all'], 2),
                f(n10 and n10['fixed']['ratio'], 4), f(n20 and n20['fixed']['ratio'], 4),
                f(n10 and n10['valid']['ratio'], 4) + (f" ({n10['valid']['n']})" if n10 else ''),
                f(n10 and n10['valid']['max_dev']), f(r['prof']['pred_ratio'], 2),
                f(n10 and n10['valid']['pred_ratio'], 2)]) + r' \\'))
    keep = sorted({x for x, _, _ in rows}, reverse=True)
    keep = [keep[i] for i in args.levels] if args.levels else keep
    order = list(NAMES)
    for x, _, row in sorted(rows, key=lambda t: (-t[0], order.index(t[1]))):
        if x in keep:
            print(row)


def cmd_profile(args):
    model, sets = _setup_from(args.a, args)
    do_profile(model, sets, args.a, args.b, args, args.out)


def cmd_band(args):
    model, sets = _setup_from(args.a, args)
    do_band(model, sets, args.a, args.b, args, args.out, perturb=args.perturb, perturb_seed=args.perturb_seed,
            points_out=args.save_points)


def band_args(s):
    s.add_argument('--n', type=int, default=10)
    s.add_argument('--ws', type=float, nargs='+', default=[1e-4, 1e-3, 1e-2, 1e-1, 1, 10])
    s.add_argument('--iters', type=int, default=20)
    s.add_argument('--tol', type=float, default=0.02)
    s.add_argument('--rounds', type=int, default=4)
    s.add_argument('--round-iters', type=int, default=10)
    s.add_argument('--m', type=int, default=8)
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
    s.add_argument('out_dir')
    s.add_argument('--out', required=True)
    s = sub.add_parser('table')
    s.add_argument('out_dir')
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
     'run': cmd_run, 'plot': cmd_plot, 'table': cmd_table}[args.cmd](args)


if __name__ == '__main__':
    main()
