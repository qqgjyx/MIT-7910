"""How much rounding moves the Muon and LC updates on real momentum matrices, on a GPU:

    python scripts/rounding_check.py <momentum dump .pt> ...   (analysis/negp_fast/*.pt, their `final` matrices)

Per matrix N: dU = U(N (1 + 2^-9)) - U(N), zero in exact arithmetic (U: muon_update's update, flag off; the fast path
rounds the same way). Relative to |U|_F: the whole change, and its part in the singular directions of N with
s >= 1e-4 s_1 on both sides (bulk); per matrix also the fraction of singular values of N below 1e-3 s_1. LC5 and LC5m
also with float32 iterates (LC_FLOAT32 = 0), and the training step with them (fast path, one process).
"""
import sys
from statistics import median

import torch

from fast_update_check import profile_step
from optsplit.optim import muon as M

LC5, LC5M = dict(sv_power=-0.3, lc_steps=5), dict(sv_power=-0.3, lc_steps='5m')
METHODS = {'muon': ({}, False), 'lc5': (LC5, False), 'lc5 f32': (LC5, True), 'lc5m': (LC5M, False),
           'lc5m f32': (LC5M, True)}


def direction(N, kw, f32):
    lc32 = M.LC_FLOAT32
    M.LC_FLOAT32 = 0 if f32 else lc32
    try:
        return M.muon_direction(N, **kw).double()
    finally:
        M.LC_FLOAT32 = lc32


def sensitivity(mats):
    """{method: [(change, bulk change, fraction of s < 1e-3 s_1, label)]}"""
    svd = {}
    for label, N in mats.items():
        P, S, Qh = torch.linalg.svd(N.double(), full_matrices=False)
        svd[label] = P, S / S[0], Qh
    out = {}
    for m, (kw, f32) in METHODS.items():
        out[m] = []
        for label, N in mats.items():
            U = direction(N, kw, f32)
            dU = direction(N * (1 + 2 ** -9), kw, f32) - U
            P, s, Qh = svd[label]
            big = s >= 1e-4
            bulk = P[:, big].T @ dU @ Qh[big].T
            out[m].append(((dU.norm() / U.norm()).item(), (bulk.norm() / U.norm()).item(), (s < 1e-3).double().mean().item(),
                           label))
    return out


def report(out):
    print(f'{"method":<9} {"change: median":>14} {"max":>7} {"bulk: median":>12} {"max":>7}   max change by start step; by matrix')
    for m, rows in out.items():
        ch, bu = [r[0] for r in rows], [r[1] for r in rows]
        by = lambda key: ' '.join(f'{k} {max(r[0] for r in rows if key(r[3]) == k):.1e}'
                                  for k in dict.fromkeys(key(r[3]) for r in rows))
        print(f'{m:<9} {median(ch):>14.1e} {max(ch):>7.1e} {median(bu):>12.1e} {max(bu):>7.1e}   '
              f'{by(lambda lab: lab.split()[1])}; {by(lambda lab: lab.split()[2].split(".", 1)[1])}')
    for m in ('muon', 'lc5m'):
        print(f'worst {m}: ' + '; '.join(f'{r[3]} {r[0]:.2f} (bulk {r[1]:.2f}, {100 * r[2]:.0f}% of s < 1e-3 s1)'
                                          for r in sorted(out[m], reverse=True)[:4]))


def main():
    mats = {}
    for path in sys.argv[1:]:
        d = torch.load(path, map_location='cuda', weights_only=False)
        tag = f'{"svp" if "svp" in d["run"] else "main"} s{d["start"]}'
        for name, N in d['final'].items():
            mats[f'{tag} {name.removeprefix("transformer.h.").removesuffix(".weight")}'] = N
    print(torch.cuda.get_device_name(), len(mats), 'matrices')
    report(sensitivity(mats))
    print(f'\n{"train step, one process":<24} ms')
    fast, lc5m = ['optim.muon.fast_update=true'], ['optim.muon.sv_power=-0.3', 'optim.muon.lc_steps=5m']
    lc32 = M.LC_FLOAT32
    for label, o, f32 in (('muon fast', fast, False), ('lc5m fast', lc5m + fast, False),
                          ('lc5m fast, f32 iterates', lc5m + fast, True), ('lc5m', lc5m, False)):
        M.LC_FLOAT32 = 0 if f32 else lc32
        print(f'{label:<24} {profile_step(o)["train"]:.1f}')
    M.LC_FLOAT32 = lc32


if __name__ == '__main__':
    main()
