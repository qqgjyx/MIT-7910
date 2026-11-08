"""No-signal baseline for next step (ii): each update rule on gradient streams with no learning in them, on the
three block matrix shapes of TinyStories-10M, from the model's initialization, with the main runs' learning rates and
schedule. Streams: `noise`, i.i.d. Gaussian gradients; `lowrank`, a fixed rank-4 direction plus noise of the same
Frobenius norm. Both rules see the same noise draws. Reports stable rank, spectral norm, sigma_1/sigma_median of W and
the cosine of consecutive updates, as in metrics/trajectory.py.

    PYTHONPATH=code python code/scripts/rule_noise.py --out DIR [--steps 10000] [--device cuda]
    PYTHONPATH=code python code/scripts/rule_noise.py --plot DIR/rule_noise.json --out FIG.pdf
"""

import argparse
import json
from pathlib import Path

import torch

from optsplit.config import CODE_DIR, load_config, step_grid
from optsplit.models import build_model
from optsplit.optim import lr_mult
from optsplit.optim.muon import adam_update, muon_update

# Update cosines of the real runs at training loss 1.81, drawn dashed in the figure.
REAL = {'adamw': 0.886, 'muon': 0.738}
SHAPES = {'attn': 'transformer.h.0.attn.q.weight', 'mlp_in': 'transformer.h.0.mlp.c_fc.weight',
          'mlp_out': 'transformer.h.0.mlp.c_proj.weight'}


def stats(W):
    s = torch.linalg.svdvals(W.double())
    return dict(stable_rank=(s.pow(2).sum() / s[0] ** 2).item(), spectral=s[0].item(),
                sv_spread=(s[0] / s.median()).item())


def simulate(rule, W0, signal, cfg, steps, log, device):
    """`steps` updates of one rule on noise (+ `signal`); every call draws the same noise."""
    gen = torch.Generator(device).manual_seed(2)
    W, prev, o = W0.clone(), None, cfg.optim[rule]
    m, v = torch.zeros_like(W), torch.zeros_like(W)
    rec = []
    for t in range(steps + 1):
        if t in log:
            rec.append(dict(step=t, **stats(W), cos=None))
        if t == steps:
            return rec
        N = torch.randn(W.shape, generator=gen, device=device)
        G = N / N.norm() + signal
        lr = o.lr * lr_mult(t, cfg)
        u = (muon_update(G, m, beta=o.momentum).reshape(W.shape) if rule == 'muon'
             else adam_update(G, m, v, t + 1, tuple(o.betas), o.eps))
        step = -lr * u - lr * o.weight_decay * W
        if prev is not None and t in log:
            rec[-1]['cos'] = torch.nn.functional.cosine_similarity(prev.flatten(), step.flatten(), dim=0).item()
        W += step
        prev = step


def plot(path, out, kind='mlp_in'):
    """Stable rank (both streams) and update cosine (noise) for one shape, with the real runs' cosines dashed."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    runs = json.loads(Path(path).read_text())['runs']
    color = {'adamw': 'C0', 'muon': 'C1'}
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.6))
    axes[0].sharey(axes[1])  # same scale for both stable-rank panels, so a few units of ripple stay flat
    panels = [('noise', 'stable_rank', '(a) stable rank of W, noise'),
              ('lowrank', 'stable_rank', '(b) stable rank of W, rank-4 signal + noise'),
              ('noise', 'cos', '(c) cosine of consecutive updates, noise')]
    for ax, (stream, key, title) in zip(axes, panels):
        for rule in ('adamw', 'muon'):
            pts = [(e['step'], e[key]) for e in runs[f'{kind}/{stream}/{rule}'] if e[key] is not None and e['step'] > 0]
            ax.plot(*zip(*pts), color=color[rule], lw=1.3, label={'adamw': 'AdamW rule', 'muon': 'Muon rule'}[rule])
            if key == 'cos':
                ax.axhline(REAL[rule], color=color[rule], ls='--', lw=0.9)
        ax.set_xscale('log')
        ax.set_xlabel('step')
        ax.set_title(title, fontsize=9)
        if key == 'stable_rank':
            ax.set_yscale('log')
    axes[0].legend(fontsize=8, frameon=False)
    axes[2].text(0.98, 0.04, 'dashed: real runs at loss 1.81', transform=axes[2].transAxes, ha='right', fontsize=7)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    ap.add_argument('--plot', help='rule_noise.json to plot (to --out) instead of running')
    ap.add_argument('--steps', type=int, default=10_000)
    ap.add_argument('--rank', type=int, default=4)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    if args.plot:
        plot(args.plot, args.out)
        return
    cfg = load_config(CODE_DIR / 'configs/base/tinystories_10m.yaml',
                      ['optim.adamw.lr=2.0e-3', 'optim.muon.lr=0.01', f'train.max_steps={args.steps}'])
    torch.manual_seed(cfg.train.init_seed)
    named = dict(build_model(cfg, vocab_size=4096).named_parameters())
    log_steps = step_grid({'every': 1000, 'geometric': 30}, args.steps)
    res = {}
    for kind, pname in SHAPES.items():
        W0 = named[pname].detach().to(args.device, torch.float32)
        gen = torch.Generator(args.device).manual_seed(1)
        S = sum(torch.randn(W0.shape[0], 1, generator=gen, device=args.device)
                @ torch.randn(1, W0.shape[1], generator=gen, device=args.device) for _ in range(args.rank))
        S = S / S.norm()
        for stream in ('noise', 'lowrank'):
            for rule in ('muon', 'adamw'):
                rec = res[f'{kind}/{stream}/{rule}'] = simulate(rule, W0, S if stream == 'lowrank' else 0, cfg,
                                                                 args.steps, log_steps, args.device)
                print(f"{kind:8s} {stream:8s} {rule:6s} stable rank {rec[-1]['stable_rank']:7.1f} "
                      f"spectral {rec[-1]['spectral']:6.2f} s1/smed {rec[-1]['sv_spread']:6.2f}", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'rule_noise.json').write_text(json.dumps(dict(args=vars(args), runs=res)))
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
