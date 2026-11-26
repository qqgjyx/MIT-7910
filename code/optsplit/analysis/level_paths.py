"""Constant-loss paths between two checkpoints at (nearly) equal training loss c (results/level_paths.tex).

- `losses`: loss of checkpoints on the fixed path set (to pick, for each run, the checkpoint closest to a level).
- `profile`: loss along the straight line (1-t) theta_A + t theta_B; its highest rise above c and deepest dip below.
- `band`: elastic band. N points theta_1..theta_N between the fixed endpoints theta_0 = theta_A, theta_{N+1} = theta_B
  minimise E_w = w sum_i ((L_i - c_i) / tol)^2 + sum_k |theta_{k+1} - theta_k|^2 / (D^2 / (N+1)), D = |theta_A - theta_B|,
  c_i interpolated linearly between the endpoint losses. Each iteration is a Gauss-Newton step on E_w (exact for the
  springs, linearised loss), with backtracking; w grows over stages. Then the loss is checked densely along every
  segment; segments that leave c +- tol get a midpoint and the band continues. Output: path length / D and the loss
  along the path, on the path set and on a held-out set.

python -m optsplit.analysis.level_paths losses 'ts10m_main/adamw_d0' --steps 9000 10000 --out f.json
python -m optsplit.analysis.level_paths profile ts10m_main/adamw_d0:10000 ts10m_main/muon_d0:8000 --out p.json
python -m optsplit.analysis.level_paths band ts10m_main/adamw_d0:10000 ts10m_main/muon_d0:8000 --out b.json"""

import argparse
import contextlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from optsplit.analysis.io import Run, load_runs
from optsplit.models import build_model, layer

PATH_SEED, HELDOUT_SEED = 101, 102  # fixed training sequences (the probe uses seed 0)


class Objective:
    """Mean loss on fixed batches as a function of the flat parameter vector of `model`."""

    def __init__(self, model, batches, fp32=True):
        self.model, self.batches, self.fp32 = model, batches, fp32
        self.params = list(model.parameters())

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


def setup(cfg, n_seqs, device, chunk=64):
    """Model and the path set and held-out set (n_seqs training sequences each, in chunks)."""
    from optsplit.data import build_data
    data = build_data(cfg, device)
    model = build_model(cfg, vocab_size=data.vocab_size).to(device).eval()
    sets = []
    for seed in (PATH_SEED, HELDOUT_SEED):
        x, y = data.fixed_batch('train', n_seqs, seed)
        sets.append([(x[i:i + chunk], y[i:i + chunk]) for i in range(0, n_seqs, chunk)])
    return model, sets


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

def profile(obj, a, b, n=51, ca=None, cb=None):
    """Loss at (1-t) a + t b for t in linspace(0, 1, n); c(t) interpolates the endpoint losses."""
    ts = np.linspace(0, 1, n)
    L = np.array([obj.loss(torch.lerp(a, b, float(t))) for t in ts])
    ca = L[0] if ca is None else ca
    cb = L[-1] if cb is None else cb
    dev = L - (ca + ts * (cb - ca))
    return dict(t=ts.tolist(), loss=L.tolist(), c_a=float(ca), c_b=float(cb),
                rise=float(max(dev.max(), 0)), dip=float(max(-dev.min(), 0)), distance=float((a - b).norm()))


# ---------------------------------------------------------------- elastic band

def straight(a, b, n):
    t = torch.arange(1, n + 1, dtype=a.dtype, device=a.device)[:, None] / (n + 1)
    return a + t * (b - a)


def targets(ca, cb, n):
    return ca + (cb - ca) * np.arange(1, n + 1) / (n + 1)


def lengths(X, a, b):
    P = torch.cat([a[None], X, b[None]])
    return (P[1:] - P[:-1]).norm(dim=1).double().cpu().numpy()


def merit(X, a, b, L, c, w, tol):
    """E_w: the constraint term and the spring term (1 on the evenly spaced straight line)."""
    D2 = float((a - b).double().norm() ** 2)
    return w * float(np.sum(((L - c) / tol) ** 2)) + (len(X) + 1) * float(np.sum(lengths(X, a, b) ** 2)) / D2


def gn_target(X, a, b, L, G, c, w, tol):
    """Minimiser of E_w with each loss linearised at the current points (Gauss-Newton step, exact for the springs):
    X* = X_line + M^{-1} (gamma_i g_i)_i, where M = tridiag(-1, 2, -1) and gamma solves
    (tol^2 / (kappa w) I + M^{-1} o Gram) gamma = e, e_i = c_i - L_i + g_i . (X_i - X_line_i), kappa = D^2 / (N+1).
    At a fixed point the spring force on each point is parallel to its gradient (a discrete geodesic of the level
    set when w is large)."""
    n = len(X)
    Xl = straight(a, b, n)
    M = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    Minv = np.linalg.inv(M)
    gram = (G.double() @ G.double().T).cpu().numpy()
    e = np.asarray(c) - np.asarray(L) + ((X - Xl).double() * G.double()).sum(1).cpu().numpy()
    kappa = float((a - b).double().norm() ** 2) / (n + 1)
    gamma = np.linalg.solve(tol ** 2 / (kappa * w) * np.eye(n) + Minv * gram, e)
    coef = torch.tensor(Minv * gamma[None, :], dtype=X.dtype, device=X.device)  # coef[i, j] = Minv_ij gamma_j
    return Xl + coef @ G


def evaluate(obj, X):
    L, G = zip(*(obj.loss_grad(x) for x in X))
    return np.array(L), torch.stack(G)


def dense_check(obj, X, a, b, c_pts, tol, m=8):
    """Loss at the points P_0..P_{N+1} (endpoints included) and at m interior points of every segment; c_pts (N+2
    values) interpolated linearly along each segment. -> per segment: max |L - c|; arrays (arclength, loss, c)."""
    P = torch.cat([a[None], X, b[None]])
    seg = lengths(X, a, b)
    s0 = np.concatenate([[0], np.cumsum(seg)])
    ts = np.linspace(0, 1, m + 2)
    Lp = np.array([obj.loss(p) for p in P])
    out_s, out_l, out_c, worst = [], [], [], []
    for k in range(len(P) - 1):
        Lk = np.array([Lp[k]] + [obj.loss(torch.lerp(P[k], P[k + 1], float(t))) for t in ts[1:-1]] + [Lp[k + 1]])
        ck = c_pts[k] + ts * (c_pts[k + 1] - c_pts[k])
        worst.append(float(np.abs(Lk - ck).max()))
        keep = slice(None) if k == len(P) - 2 else slice(None, -1)
        out_s += (s0[k] + ts * seg[k])[keep].tolist()
        out_l += Lk[keep].tolist()
        out_c += ck[keep].tolist()
    return np.array(worst), dict(s=out_s, loss=out_l, c=out_c, points=Lp.tolist())


def band(obj, a, b, ca, cb, obj_check=None, ca_check=None, cb_check=None, n=10, ws=(1e-2, 1e-1, 1, 10, 100),
         iters=10, tol=0.02, rounds=4, round_iters=10, m=8, max_points=80, trust=1.0, init=None, log=print):
    """Elastic band between a and b at the losses ca -> cb (see the module docstring). `obj` drives the iterations,
    `obj_check` (default obj, with ca_check/cb_check) the dense check. Returns (points, result dict)."""
    obj_check = obj_check or obj
    ca_check = ca if ca_check is None else ca_check
    cb_check = cb if cb_check is None else cb_check
    D = float((a - b).norm())
    X = straight(a, b, n) if init is None else init.clone()
    hist, t0 = [], time.time()

    def run_stage(X, w, k):
        n = len(X)
        c = targets(ca, cb, n)
        L, G = evaluate(obj, X)
        for _ in range(k):
            E = merit(X, a, b, L, c, w, tol)
            step = gn_target(X, a, b, L, G, c, w, tol) - X
            alpha = min(1.0, trust * D / (n + 1) / float(step.norm(dim=1).max().clamp_min(1e-30)))
            for _ in range(6):
                Xn = X + alpha * step
                Ln, Gn = evaluate(obj, Xn)
                En = merit(Xn, a, b, Ln, c, w, tol)
                if En <= E or alpha < 1e-3:
                    break
                alpha /= 2
            X, L, G = Xn, Ln, Gn
            seg = lengths(X, a, b)
            hist.append(dict(n=n, w=w, alpha=alpha, E=En, max_r=float(np.abs(L - c).max()),
                             ratio=float(seg.sum() / D), time=time.time() - t0))
            log(f"n={n} w={w:g} alpha={alpha:.3g} E={En:.4g} max|L-c|={hist[-1]['max_r']:.4f} "
                f"ratio={hist[-1]['ratio']:.5f} ({hist[-1]['time']:.0f}s)")
        return X

    stages = []
    for w in ws:
        X = run_stage(X, w, iters)
        stages.append(dict(w=w, ratio=hist[-1]['ratio'], max_r=hist[-1]['max_r']))
    checks = []
    for r in range(rounds + 1):
        c_chk = np.concatenate([[ca_check], targets(ca_check, cb_check, len(X)), [cb_check]])
        worst, dense = dense_check(obj_check, X, a, b, c_chk, tol, m)
        checks.append(dict(n=len(X), worst=worst.tolist()))
        bad = np.flatnonzero(worst > tol)
        log(f'check {r}: {len(X)} points, worst segment {worst.max():.4f} (tol {tol}), {len(bad)} segments out')
        if not len(bad) or r == rounds or len(X) + len(bad) > max_points:
            break
        P = torch.cat([a[None], X, b[None]])
        new = []
        for k in range(len(P) - 1):
            if k > 0:
                new.append(P[k])
            if k in bad:
                new.append((P[k] + P[k + 1]) / 2)
        X = run_stage(torch.stack(new), ws[-1], round_iters)
    seg = lengths(X, a, b)
    Xl = straight(a, b, len(X))
    res = dict(distance=D, n=len(X), ratio=float(seg.sum() / D), seg=seg.tolist(), tol=tol,
               offline_max=float((X - Xl).norm(dim=1).max() / D), c_a=ca_check, c_b=cb_check,
               max_dev=float(np.max(worst)), ok=bool(np.max(worst) <= tol), dense=dense, history=hist, checks=checks,
               stages=stages,
               settings=dict(n=n, ws=list(ws), iters=iters, rounds=rounds, round_iters=round_iters, m=m, trust=trust))
    return X, res


# ---------------------------------------------------------------- CLI

def _setup_from(spec, args):
    run = Run(spec.rsplit(':', 1)[0])
    model, sets = setup(run.cfg, args.n_seqs, args.device)
    check_order(model, spec, args.device)
    return model, sets


def cmd_losses(args):
    runs = load_runs(args.runs)
    model, (path_set, _) = setup(runs[0].cfg, args.n_seqs, args.device)
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


def cmd_profile(args):
    model, (path_set, held) = _setup_from(args.a, args)
    a, b = load_theta(args.a, args.device), load_theta(args.b, args.device)
    init = load_theta(args.a.rsplit(':', 1)[0] + ':0', args.device)
    obj = Objective(model, path_set)
    res = profile(obj, a, b, args.points)
    sa, sb = obj.per_seq(a), obj.per_seq(b)
    res.update(a=args.a, b=args.b, n_seqs=args.n_seqs, se_mean=float(sa.std() / math.sqrt(len(sa))),
               se_pair=float((sa - sb).std() / math.sqrt(len(sa))),
               heldout=[Objective(model, held).loss(x) for x in (a, b)],
               from_init=[float((a - init).norm()), float((b - init).norm())],
               rel_distance=scale_free(model, a, b, init))
    print(f"{args.a} vs {args.b}: c {res['c_a']:.4f} -> {res['c_b']:.4f}, rise {res['rise']:.4f}, "
          f"dip {res['dip']:.4f}, D {res['distance']:.2f}, D/init {res['rel_distance']['all']:.3f}, "
          f"se_pair {res['se_pair']:.4f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))


def cmd_band(args):
    model, (path_set, held) = _setup_from(args.a, args)
    a, b = load_theta(args.a, args.device), load_theta(args.b, args.device)
    fast, exact, hold = Objective(model, path_set, fp32=False), Objective(model, path_set), Objective(model, held)
    ca, cb = fast.loss(a), fast.loss(b)
    init = None
    if args.perturb:  # a bent initial guess: straight line + perturb * D * sin(pi t) * (random unit direction)
        u = torch.randn(a.shape, generator=torch.Generator().manual_seed(args.perturb_seed)).to(a.device)
        t = torch.arange(1, args.n + 1, device=a.device)[:, None] / (args.n + 1)
        init = straight(a, b, args.n) + args.perturb * float((a - b).norm()) * torch.sin(math.pi * t) * u / u.norm()
    X, res = band(fast, a, b, ca, cb, obj_check=exact, ca_check=exact.loss(a), cb_check=exact.loss(b), n=args.n,
                  ws=args.ws, iters=args.iters, tol=args.tol, rounds=args.rounds, round_iters=args.round_iters,
                  m=args.m, trust=args.trust, init=init, log=lambda s: print(s, flush=True))
    hl = np.array([hold.loss(x) for x in torch.cat([a[None], X, b[None]])])
    hc = hl[0] + (hl[-1] - hl[0]) * np.linspace(0, 1, len(hl))
    res.update(a=args.a, b=args.b, n_seqs=args.n_seqs, heldout=hl.tolist(), perturb=args.perturb,
               perturb_seed=args.perturb_seed,
               heldout_max_dev=float(np.abs(hl - hc).max()))
    print(f"ratio {res['ratio']:.5f}, {res['n']} points, max dev {res['max_dev']:.4f}, "
          f"held-out max dev {res['heldout_max_dev']:.4f}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    if args.save_points:
        torch.save(dict(a=args.a, b=args.b, offsets=(X - straight(a, b, len(X))).half().cpu()), args.save_points)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--n-seqs', type=int, default=512, help='training sequences in the path set and the held-out set')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('losses')
    s.add_argument('runs', nargs='+')
    s.add_argument('--steps', type=int, nargs='*')
    s.add_argument('--out', required=True)
    s = sub.add_parser('profile')
    s.add_argument('a')
    s.add_argument('b')
    s.add_argument('--points', type=int, default=51)
    s.add_argument('--out', required=True)
    s = sub.add_parser('band')
    s.add_argument('a')
    s.add_argument('b')
    s.add_argument('--n', type=int, default=10)
    s.add_argument('--ws', type=float, nargs='+', default=[1e-2, 1e-1, 1, 10, 100])
    s.add_argument('--iters', type=int, default=10)
    s.add_argument('--tol', type=float, default=0.02)
    s.add_argument('--rounds', type=int, default=4)
    s.add_argument('--round-iters', type=int, default=10)
    s.add_argument('--m', type=int, default=8)
    s.add_argument('--trust', type=float, default=1.0)
    s.add_argument('--perturb', type=float, default=0.0, help='initial bend, as a fraction of D (0: straight line)')
    s.add_argument('--perturb-seed', type=int, default=1)
    s.add_argument('--out', required=True)
    s.add_argument('--save-points')
    args = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    {'losses': cmd_losses, 'profile': cmd_profile, 'band': cmd_band}[args.cmd](args)


if __name__ == '__main__':
    main()
