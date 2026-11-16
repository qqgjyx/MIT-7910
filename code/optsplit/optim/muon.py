"""Muon, vendored from https://github.com/KellerJordan/Muon (muon.py at f98f1ca, MIT license).

Only the single-device optimizer with auxiliary AdamW is kept. One change: `muon_update` no longer
overwrites `grad` in place (upstream does `grad.lerp_`), so a shadow optimizer or ghost stepping after Muon on
the same gradients sees them unchanged. Added, as ablations (DESIGN.md): `orthogonalize=False`, `sv_power=p` and
`two_track_steps=K` (sv_power by matrix products instead of the SVD).
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


def sv_power_exact(G, p):
    """U S^p V^T of G (p = 0: exact orthogonalization); for p < 0 with relative singular values, floored at SV_FLOOR."""
    U, S, Vh = torch.linalg.svd(G, full_matrices=False)
    if p < 0:  # S^p is unbounded as S -> 0 (the scale cancels in the rescaling of muon_update)
        S = (S / S[0].clamp_min(1e-30)).clamp_min(SV_FLOOR)
    return (U * S.pow(p)) @ Vh


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True, orthogonalize=True, sv_power=None,
                two_track_steps=None):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp(momentum, beta) if nesterov else momentum
    if update.ndim == 4:  # conv filters
        update = update.view(len(update), -1)
    if sv_power is None and orthogonalize:
        update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    else:  # ablations, rescaled to the Frobenius norm of an orthogonal update, sqrt(min(n_out, n_in))
        if two_track_steps:  # U S^p V^T of the momentum by matrix products
            update = sv_power_two_track(update, sv_power, two_track_steps)
        elif sv_power is not None:  # the same by an exact SVD
            update = sv_power_exact(update.float(), sv_power)
        update = update * (min(update.shape[-2:]) ** 0.5 / (update.norm() + 1e-7))
    update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
    return update


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
                assert not group["two_track_steps"] or group["sv_power"] in TWO_TRACK, "two-track: no coefficients"
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon",
                                                 "orthogonalize", "sv_power", "two_track_steps"])
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
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"],
                                         orthogonalize=group["orthogonalize"], sv_power=group["sv_power"],
                                         two_track_steps=group["two_track_steps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
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
    |Z - E|_F / |E|_F after rescaling both to the same norm (so error^2 = 2 - 2 cosine). `svd32` is the float32 SVD
    of the exact method, for scale."""
    out = {f'{kind}_{k}': [] for k in (*steps, 'svd32') for kind in ('err', 'cos')}
    for group in opt.param_groups:
        if not group["use_muon"]:
            continue
        p = group["sv_power"]
        for param in group["params"]:
            N = param.grad.lerp(opt.state[param]["momentum_buffer"], group["momentum"]).float()
            exact = sv_power_exact(N.double(), p)
            approx = {k: sv_power_two_track(N, p, k) for k in steps}
            approx['svd32'] = sv_power_exact(N, p)
            for k, Z in approx.items():
                cos = ((Z.double() * exact).sum() / (Z.double().norm() * exact.norm())).item()
                out[f'cos_{k}'].append(cos)
                out[f'err_{k}'].append(max(0.0, 2 - 2 * cos) ** 0.5)
    return out
