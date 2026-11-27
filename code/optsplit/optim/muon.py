"""Muon, vendored from https://github.com/KellerJordan/Muon (muon.py at f98f1ca, MIT license).

Only the single-device optimizer with auxiliary AdamW is kept. One change: `muon_update` no longer
overwrites `grad` in place (upstream does `grad.lerp_`), so a shadow optimizer or ghost stepping after Muon on
the same gradients sees them unchanged. Added, as ablations (DESIGN.md): `orthogonalize=False`, `sv_power=p` and
`two_track_steps=K` and `lc_steps=K` (sv_power by matrix products instead of the SVD); and `fast_update=True`
(optim.muon.fast_update: the same update with the matrices of one shape in one batch, see fast_muon_step).
"""

import torch

SV_FLOOR = 1e-6  # sv_power < 0: singular values below this fraction of the largest are raised to it (float32 SVD noise)


def zeropower_via_newtonschulz5(G, steps: int):
    """Approximate orthogonalization U S' V^T of G by a quintic Newton-Schulz iteration (see upstream)."""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


# Output-track polynomials (alpha, beta, gamma) of sv_power_two_track for each p: first step, middle steps, last step.
# One set per p serves every number of steps (fitted jointly on 5, 8 and 12 steps).
TWO_TRACK = {
    -0.3: ((4.9125, -8.0324, 3.7790), (4.9915, -7.8279, 3.4157), (2.1740, -2.4723, 0.9797)),
    -0.5: ((6.1049, -10.8787, 5.4076), (6.3927, -10.5965, 4.7000), (2.1908, -2.9743, 1.2786)),
}


def sv_power_two_track(G, p, steps):
    """Approximate U S^p V^T of G = U S V^T (up to a scale) with 4 * steps - 1 matrix products and no SVD.

    Two tracks start at G / |G|_F. X follows Muon's Newton-Schulz iteration, so its singular values grow towards 1,
    the small ones by 3.4445 per step. Z is multiplied in each step by a polynomial of X X^T chosen so that, once
    a singular value of X has reached 1, the matching one of Z is its starting value to the power p. Relative singular
    values above about 4 / 3.4445^steps come out as S^p within +-25% (8e-3, 2e-4, 1e-6 for 5, 8, 12 steps); smaller
    ones get less than S^p, falling to zero with S. Always float32 without TF32: bfloat16 breaks from 8 steps on."""
    first, middle, last = TWO_TRACK[p]
    a, b, c = (3.4445, -4.7750, 2.0315)
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.autocast(G.device.type, enabled=False):
            X = G.float()
            if G.size(-2) > G.size(-1):
                X = X.mT
            X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
            Z = X
            for k in range(steps):
                A = X @ X.mT
                A2 = A @ A
                s = first if k == 0 else last if k == steps - 1 else middle
                Z = s[0] * Z + (s[1] * A + s[2] * A2) @ Z
                if k < steps - 1:
                    X = a * X + (b * A + c * A2) @ X
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    return Z.mT if G.size(-2) > G.size(-1) else Z


# LC (linear combination of Newton-Schulz iterates): for (p, steps), the quintic coefficients (a, b, c) of each step and
# the weights w_0..w_steps of the iterates X_0..X_steps. Fitted as a scalar float64 minimax fit of g(sigma) on [eps, 1]
# by scripts/lc/s3_joint.py (scripts/lc/final_coeffs.json, sets p-0.3_lc5_eps3.0e-03, p-0.3_lc6_eps1.0e-03,
# p-0.3_lc8_eps2.0e-04 and p-0.3_lc12_eps2.0e-06); real momentum matrices were used only to evaluate the sets and
# pick eps (scripts/lc/check_snippet.py checked the first two on momentum matrices that are not in the repository).
# (-0.3, '5m'), LC5m: 5 steps whose output lies between Muon's and S^p for every singular value in (0, 1] of the
# Schatten-8 norm (scripts/lc/lc5m.py, lc5m_coeffs.json): below S^p, above Muon's value for any |G|_S8 / |G|_F.
LC = {
    (-0.3, 5): (((3.5969, -7.7746, 4.8280), (3.5004, -7.1445, 4.1116), (3.4189, -6.6980, 3.8465),
                 (3.0402, -4.9130, 2.8709), (1.8013, -1.6495, 0.84815)),
                (-0.88949, -1.2850, -1.6349, -0.90775, -11.758, 16.278)),
    (-0.3, 6): (((3.5791, -7.6099, 4.4535), (3.1263, -5.3070, 3.1688), (3.3447, -6.2419, 3.3154),
                 (3.4107, -6.5906, 3.5226), (3.0637, -5.0036, 2.8968), (2.7706, -3.7884, 2.0046)),
                (-1.0761, -0.40821, -1.7443, -2.3533, -1.2355, -3.8680, 10.337)),
    (-0.3, 8): (((3.2348, -5.7683, 3.3109), (3.2507, -5.7459, 2.9408), (3.2121, -5.5611, 2.843),
                 (2.6895, -3.4919, 1.7981), (3.2704, -5.917, 3.3123), (3.0618, -4.8785, 2.4672),
                 (2.9111, -4.2724, 2.1788), (2.7287, -3.5843, 1.7617)),
                (-0.70074, -1.1829, -1.2574, -0.67631, -3.0728, -3.277, -3.8518, -0.16758, 13.782)),
    (-0.3, 12): (((2.9697, -4.4367, 2.0673), (3.4655, -6.8891, 3.6988), (3.1281, -5.2348, 2.8713),
                  (3.2114, -5.5393, 2.7639), (3.3399, -6.2026, 3.2349), (2.9449, -4.4931, 2.539),
                  (3.1715, -5.3421, 2.624), (3.2469, -5.8017, 3.2428), (3.2104, -5.5553, 2.848),
                  (2.7248, -3.7049, 1.9801), (2.1957, -3.0101, 1.8144), (3.1917, -5.5021, 2.9403)),
                 (-0.64225, -1.5195, -0.66817, -2.3435, -4.0227, -2.3636, -6.2484, -7.3231, -11.162, -9.2759,
                  13.077, -10.476, 54.037)),
    (-0.3, '5m'): (((3.213, -5.628, 3.1012), (3.2007, -5.4618, 2.6274), (3.2174, -5.8979, 3.6806),
                    (2.8932, -4.25, 2.2902), (1.8606, -2.0992, 1.2386)),
                   (-0.660203352, -1.061696401, -0.4829314639, -0.01733564992, -11.23855637, 14.71319634)),
}
LC_FLOAT32 = 8  # from this many steps on, float32 without TF32 (12 steps break in bfloat16, 8 are borderline)


def sv_power_lc(G, p, steps):
    """Approximate U S^p V^T of G (up to a scale) with 3 * steps matrix products, Muon's cost for 5 steps: the weighted
    sum of the iterates of a Newton-Schulz iteration with its own coefficients, started at G / |G|_S8 (Schatten-8 norm,
    from the first squared Gram matrix). Singular values above 3e-3 (5 steps), 1e-3 (6), 2e-4 (8) or 2e-6 (12) of |G|_S8
    come out as S^p within about +-13% (+-11%, +-5%, +-16%); smaller ones get less. steps = '5m' (LC5m): between
    Muon's output and S^p for every singular value, S^p within -32%/+7% above 3e-3 (+-26% about a common scale).
    Iterates in bfloat16 as in Muon for 5 and 6 steps, in float32 without TF32 for 8 and 12; the sum in float32."""
    coeffs, w = LC[p, steps]
    dt = torch.float32 if len(coeffs) >= LC_FLOAT32 else torch.bfloat16
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.autocast(G.device.type, enabled=False):
            X = G.to(dt)
            if G.size(-2) > G.size(-1):
                X = X.mT
            X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
            out = 0
            for k, (a, b, c) in enumerate(coeffs):
                A = X @ X.mT
                A2 = A @ A
                if k == 0:
                    s = A2.float().norm(dim=(-2, -1), keepdim=True) ** 0.25
                    X, A, A2 = (X / s).to(dt), (A / s ** 2).to(dt), (A2 / s ** 4).to(dt)
                out = out + w[k] * X.float()
                X = a * X + (b * A + c * A2) @ X
            out = out + w[-1] * X.float()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    return out.mT if G.size(-2) > G.size(-1) else out


def sv_power_exact(G, p):
    """U S^p V^T of G (p = 0: exact orthogonalization); for p < 0 with relative singular values, floored at SV_FLOOR."""
    U, S, Vh = torch.linalg.svd(G, full_matrices=False)
    if p < 0:  # S^p is unbounded as S -> 0 (the scale cancels in the rescaling of muon_update)
        S = (S / S[0].clamp_min(1e-30)).clamp_min(SV_FLOOR)
    return (U * S.pow(p)) @ Vh


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True, orthogonalize=True, sv_power=None,
                two_track_steps=None, lc_steps=None):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp(momentum, beta) if nesterov else momentum
    if update.ndim == 4:  # conv filters
        update = update.view(len(update), -1)
    return muon_direction(update, ns_steps, orthogonalize, sv_power, two_track_steps, lc_steps)


def muon_direction(update, ns_steps=5, orthogonalize=True, sv_power=None, two_track_steps=None, lc_steps=None,
                   batched=False):
    """The end of muon_update: the update from the matrix it orthogonalizes, or from a stack of such matrices
    (batched=True: each one rescaled by its own norm)."""
    if sv_power is None and orthogonalize:
        update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    else:  # ablations, rescaled to the Frobenius norm of an orthogonal update, sqrt(min(n_out, n_in))
        if two_track_steps:  # U S^p V^T of the momentum by matrix products
            update = sv_power_two_track(update, sv_power, two_track_steps)
        elif lc_steps:
            update = sv_power_lc(update, sv_power, lc_steps)
        elif sv_power is not None:  # the same by an exact SVD
            update = sv_power_exact(update.float(), sv_power)
        norm = update.norm(dim=(-2, -1), keepdim=True) if batched else update.norm()
        update = update * (min(update.shape[-2:]) ** 0.5 / (norm + 1e-7))
    update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
    return update


def fast_muon_step(group, momenta):
    """muon_update and the parameter update of SingleDeviceMuonWithAuxAdam.step for all matrices of a Muon group at once
    (optim.muon.fast_update): momentum and parameter updates as foreach ops, and the matrices of each shape stacked into
    one batch, so the iteration runs once per shape (3 times for the GPT instead of 36). The same arithmetic, launched as
    about 20 times fewer GPU kernels; batched products and norms round differently, so the update equals the per-matrix
    one up to bfloat16 rounding."""
    params, lr = group["params"], group["lr"]
    grads = [p.grad for p in params]
    torch._foreach_lerp_(momenta, grads, 1 - group["momentum"])
    updates = torch._foreach_lerp(grads, momenta, group["momentum"])  # Nesterov
    torch._foreach_mul_(params, 1 - lr * group["weight_decay"])
    for shape in dict.fromkeys(p.shape for p in params):
        idx = [i for i, p in enumerate(params) if p.shape == shape]
        update = muon_direction(torch.stack([updates[i] for i in idx]).flatten(2), orthogonalize=group["orthogonalize"],
                                sv_power=group["sv_power"], two_track_steps=group["two_track_steps"],
                                lc_steps=group["lc_steps"], batched=True)
        update = update.reshape(len(idx), *shape).to(params[idx[0]].dtype)
        torch._foreach_add_([params[i] for i in idx], list(update), alpha=-lr)


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """Muon for the param groups with `use_muon=True`, AdamW for the others."""

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["orthogonalize"] = group.get("orthogonalize", True)
                group["sv_power"] = group.get("sv_power", None)
                group["two_track_steps"] = group.get("two_track_steps", None)
                group["lc_steps"] = group.get("lc_steps", None)
                group["fast_update"] = group.get("fast_update", False)
                assert not group["two_track_steps"] or group["sv_power"] in TWO_TRACK, "two-track: no coefficients"
                assert not group["lc_steps"] or (group["sv_power"], group["lc_steps"]) in LC, "LC: no coefficients"
                assert not group["fast_update"] or group["sv_power"] is None or group["two_track_steps"] or \
                    group["lc_steps"], "fast_update: not with the exact SVD"
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon",
                                                 "orthogonalize", "sv_power", "two_track_steps", "lc_steps",
                                                 "fast_update"])
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    if group["fast_update"]:
                        continue  # all matrices at once, below
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"],
                                         orthogonalize=group["orthogonalize"], sv_power=group["sv_power"],
                                         two_track_steps=group["two_track_steps"], lc_steps=group["lc_steps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
                if group["fast_update"]:
                    fast_muon_step(group, [self.state[p]["momentum_buffer"] for p in group["params"]])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


@torch.no_grad()
def two_track_errors(opt, steps=(5, 8, 12)):
    """Two-track update against the exact one (float64 SVD) on the matrices the last `opt.step()` used (call it before
    zero_grad): per matrix and number of steps, the cosine with the exact update and the relative error
    |Z - E|_F / |E|_F after rescaling both to the same norm (so error^2 = 2 - 2 cosine). `lc5` ... `lc12`: the LC update
    (p = -0.3 only); `svd32`: the float32 SVD of the exact method, for scale."""
    lc = [k for (q, k) in LC if q == opt.param_groups[0]["sv_power"]]
    out = {f'{kind}_{k}': [] for k in (*steps, *(f'lc{k}' for k in lc), 'svd32') for kind in ('err', 'cos')}
    for group in opt.param_groups:
        if not group["use_muon"]:
            continue
        p = group["sv_power"]
        for param in group["params"]:
            N = param.grad.lerp(opt.state[param]["momentum_buffer"], group["momentum"]).float()
            exact = sv_power_exact(N.double(), p)
            approx = {k: sv_power_two_track(N, p, k) for k in steps}
            approx.update({f'lc{k}': sv_power_lc(N, p, k) for k in lc})
            approx['svd32'] = sv_power_exact(N, p)
            for k, Z in approx.items():
                cos = ((Z.double() * exact).sum() / (Z.double().norm() * exact.norm())).item()
                out[f'cos_{k}'].append(cos)
                out[f'err_{k}'].append(max(0.0, 2 - 2 * cos) ** 0.5)
    return out
