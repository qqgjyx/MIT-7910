"""Equivalence and profile of optim.muon.fast_update (optim/muon.py, fast_muon_step), on a GPU:

    python scripts/fast_update_check.py [momentum dump .pt ...]   (dumps: analysis/negp_fast/*.pt, their `final` matrices)

1. Equivalence: one optimizer step with the flag off and on from the same matrices, for Muon and every LC set: random
   matrices with the shapes of the 14M GPT's 36 Muon matrices, and the real Nesterov inputs of the dumps. Per matrix
   |U_on - U_off|_F / |U_off|_F; for scale, the same distance from U_off to the method run in float32 without TF32.
2. Profile: one optimizer step of Muon and LC5 on the GPT (tinystories_10m, random tokens), flag off and on: GPU kernels,
   their summed time (matrix products / the rest), the CPU time to issue the step, and the training step.
"""
import sys
import time

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

from optsplit.config import CODE_DIR, load_config
from optsplit.models import build_model
from optsplit.optim import build_optimizer
from optsplit.optim import muon as M

SHAPES = [(384, 384)] * 4 + [(1536, 384), (384, 1536)]  # q, k, v, attn.c_proj, mlp.c_fc, mlp.c_proj of a layer


def update(mats, fast, **kw):
    """The update of one Muon step whose input (the Nesterov average) is each of `mats`."""
    params = [torch.zeros_like(m) for m in mats]
    opt = M.SingleDeviceMuonWithAuxAdam([dict(params=params, lr=1.0, use_muon=True, fast_update=fast, **kw)])
    for p, m in zip(params, mats):
        p.grad = m.clone()
        opt.state[p]['momentum_buffer'] = m.clone()  # m.lerp(m, w) is m: the input is m exactly
    opt.step()
    return [-p for p in params]


def ns_float32(G, steps):
    """zeropower_via_newtonschulz5 in float32."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float().mT if G.size(-2) > G.size(-1) else G.float()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        X = a * X + (b * A + c * A @ A) @ X
    return X.mT if G.size(-2) > G.size(-1) else X


def reference(mats, **kw):
    """The flag-off update with the iteration in float32 (no TF32)."""
    ns, lc32 = M.zeropower_via_newtonschulz5, M.LC_FLOAT32
    M.zeropower_via_newtonschulz5, M.LC_FLOAT32 = ns_float32, 0
    try:
        return update(mats, False, **kw)
    finally:
        M.zeropower_via_newtonschulz5, M.LC_FLOAT32 = ns, lc32


def rel(a, b):
    return [((x - y).norm() / y.norm()).item() for x, y in zip(a, b)]


def equivalence(sets, methods):
    print(f'{"matrices":<34} {"method":<10} {"max |on-off|/|off|":>18} {"median":>9} {"max |f32-off|/|off|":>19}')
    worst = 0.0
    for name, mats in sets.items():
        for label, kw in methods.items():
            off = update(mats, False, **kw)
            d, r = rel(update(mats, True, **kw), off), rel(reference(mats, **kw), off)
            worst = max(worst, max(d))
            print(f'{name:<34} {label:<10} {max(d):>18.2e} {sorted(d)[len(d) // 2]:>9.2e} {max(r):>19.2e}')
    print(f'largest relative difference, flag on vs off: {worst:.2e}')


def profile_step(overrides, n=20):
    """Kernels and times of one optimizer step and of the training step (ms)."""
    cfg = load_config(CODE_DIR / 'configs' / 'base' / 'tinystories_10m.yaml', ['optimizer=muon'] + overrides)
    torch.manual_seed(0)
    model = build_model(cfg, vocab_size=4096).cuda()
    opt = build_optimizer('muon', model, cfg)
    x, y = torch.randint(4096, (2, 128, 256), device='cuda')

    def backward():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            model.loss(x, y).backward()

    def train_step():
        backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    for _ in range(3):
        train_step()
    issue, wall = [], []
    for _ in range(n):
        backward()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.step()
        t1 = time.perf_counter()
        torch.cuda.synchronize()
        issue.append(t1 - t0)
        wall.append(time.perf_counter() - t0)
        opt.zero_grad(set_to_none=True)
    backward()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        opt.step()
        torch.cuda.synchronize()
    opt.zero_grad(set_to_none=True)
    kernels = [e for e in prof.events() if e.device_type == DeviceType.CUDA]
    is_mm = [any(s in e.name.lower() for s in ('gemm', 'xmma', 'cutlass', 'cublas')) for e in kernels]
    us = [e.time_range.elapsed_us() for e in kernels]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        train_step()
    torch.cuda.synchronize()
    med = lambda v: 1e3 * sorted(v)[len(v) // 2]
    return dict(kernels=len(kernels), mm=sum(is_mm), gpu=sum(us) / 1e3,
                gpu_mm=sum(u for u, m in zip(us, is_mm) if m) / 1e3, issue=med(issue), wall=med(wall),
                train=1e3 * (time.perf_counter() - t0) / n)


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    print(torch.cuda.get_device_name(), torch.__version__)
    g = torch.Generator(device='cuda').manual_seed(0)
    sets = {'random, GPT shapes (36)': [torch.randn(s, device='cuda', generator=g) for s in SHAPES * 6]}
    for path in sys.argv[1:]:
        final = torch.load(path, map_location='cuda', weights_only=False)['final']
        sets[path.rsplit('/', 1)[-1][:-3][-34:]] = list(final.values())
    methods = {'muon': {}, **{f'lc{k}_p{p:g}': dict(sv_power=p, lc_steps=k) for p, k in M.LC}}
    equivalence(sets, methods)

    print(f'\n{"optimizer step":<12} {"kernels":>8} {"(mm)":>5} {"GPU ms":>7} {"(mm)":>6} {"issue ms":>9} {"wall ms":>8}'
          f' {"train step ms":>14}')
    lc5 = ['optim.muon.sv_power=-0.3', 'optim.muon.lc_steps=5']
    for label, o in (('muon', []), ('muon fast', ['optim.muon.fast_update=true']), ('lc5', lc5),
                     ('lc5 fast', lc5 + ['optim.muon.fast_update=true'])):
        r = profile_step(o)
        print(f'{label:<12} {r["kernels"]:>8} {r["mm"]:>5} {r["gpu"]:>7.2f} {r["gpu_mm"]:>6.2f} {r["issue"]:>9.2f}'
              f' {r["wall"]:>8.2f} {r["train"]:>14.1f}')


if __name__ == '__main__':
    main()
