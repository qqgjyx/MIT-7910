"""Open-loop ghosts (optim.Ghost), the Muon ablations (orthogonalize=false, sv_power) and smoothed Muon (smooth)."""

import copy
import json

import numpy as np
import pytest
import torch

from optsplit import train
from optsplit.config import load_config, save_config
from optsplit.models import build_model
from optsplit.optim import Ghost, build_optimizer, load_optimizer_state, set_lr
from optsplit.optim.muon import SingleDeviceMuonWithAuxAdam, muon_update

TINY = ['model.n_layer=2', 'model.n_embd=16', 'model.n_head=2', 'model.block_size=8']


def run(rule, ghost, steps=6, extra=()):
    cfg = load_config(None, [*TINY, f'optimizer={rule}', f'ghost={ghost}', *extra])
    torch.manual_seed(0)
    model = build_model(cfg, vocab_size=11)
    opt, g = build_optimizer(rule, model, cfg), Ghost(model, cfg)
    gen = torch.Generator().manual_seed(1)
    for _ in range(steps):
        x = torch.randint(0, 11, (4, 8), generator=gen)
        model.loss(x, x.roll(-1, 1)).backward()
        set_lr(opt, 1.0)
        opt.step()
        g.step(model, 1.0)
        opt.zero_grad(set_to_none=True)
    return model, g


def differ(a, b):
    return any(not torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))


@pytest.mark.parametrize('rule', ['adamw', 'muon', 'sgd'])
def test_ghost_of_own_rule_is_the_run(rule):
    model, g = run(rule, f'[{rule}]')
    assert not differ(model, g.models[rule])


def test_ghost_is_open_loop():
    alone, _ = run('adamw', '[]')
    model, g = run('adamw', '[muon]')
    assert not differ(alone, model)  # the ghost does not change the run
    assert differ(model, g.models['muon'])


def test_ghost_load_refuses_missing_state():
    _, g = run('adamw', '[muon]')
    with pytest.raises(AssertionError):
        g.load(None)


def test_muon_without_orthogonalization():
    torch.manual_seed(0)
    for shape in [(16, 48), (48, 16)]:
        grad = torch.randn(shape)
        u = muon_update(grad, torch.zeros(shape), orthogonalize=False)
        assert u.norm().item() == pytest.approx(min(shape) ** 0.5 * max(1, shape[0] / shape[1]) ** 0.5, rel=1e-5)
        assert torch.allclose(u / u.norm(), grad / grad.norm(), atol=1e-6)  # direction unchanged


def test_muon_sv_power():
    torch.manual_seed(0)
    grad, zeros = torch.randn(16, 48), torch.zeros(16, 48)
    S = torch.linalg.svdvals(grad)  # the momentum after one step from zero is a multiple of grad
    for p in (0.0, 0.5, 1.0):
        s = torch.linalg.svdvals(muon_update(grad, zeros, sv_power=p))
        assert torch.allclose(s, S.pow(p) * (16 ** 0.5 / S.pow(p).norm()), atol=1e-4)
    assert torch.allclose(muon_update(grad, zeros, sv_power=1.0), muon_update(grad, zeros, orthogonalize=False),
                          atol=1e-5)


@pytest.mark.parametrize('option', ['orthogonalize=false', 'sv_power=0.5'])
def test_muon_options_reach_the_optimizer(option):
    assert differ(run('muon', '[]')[0], run('muon', '[]', extra=[f'optim.muon.{option}'])[0])


def test_smoothed_muon_with_a_1_is_muon():
    assert not differ(run('muon', '[]')[0], run('smuon', '[]', extra=['optim.smuon.smooth=1.0'])[0])
    assert differ(run('muon', '[]')[0], run('smuon', '[]')[0])


def muon_steps(grads, smooth=None, reload_at=None, state=None):
    """The steps -(W_t - W_{t-1}) / lr of one matrix on fixed gradients (Muon's update does not depend on W)."""
    W = torch.zeros_like(grads[0])
    make = lambda: SingleDeviceMuonWithAuxAdam([dict(params=[W], lr=0.5, use_muon=True, smooth=smooth)])
    opt, steps = make(), []
    if state is not None:
        load_optimizer_state(opt, state)
    for t, g in enumerate(grads):
        if t == reload_at:  # as a resumed run: a new optimizer with the saved state
            saved, opt = opt.state_dict(), make()
            load_optimizer_state(opt, saved)
        W.grad, before = g.clone(), W.clone()
        opt.step()
        assert torch.equal(W.grad, g)  # shadow optimizers and ghosts step on the same gradients afterwards
        steps.append((before - W) / 0.5)
    return steps, opt


def test_smoothed_muon_recursion():
    torch.manual_seed(0)
    grads, a = [torch.randn(16, 48) for _ in range(6)], 0.3
    O, muon = muon_steps(grads)
    steps, _ = muon_steps(grads, smooth=a)
    U = torch.zeros(16, 48)
    for t, (o, s) in enumerate(zip(O, steps), 1):
        U = (1 - a) * U + a * o
        assert torch.allclose(s, U / (1 - (1 - a) ** t), atol=1e-5)
    assert torch.allclose(steps[0], O[0], atol=1e-6)  # bias correction: the first step is Muon's, at full size
    resumed, _ = muon_steps(grads, smooth=a, reload_at=3)
    assert all(torch.equal(s, r) for s, r in zip(steps, resumed))
    # started from a Muon state (a swap from a parent without smuon in shadow): the average restarts
    state = lambda: copy.deepcopy(muon.state_dict())  # loading shares the tensors, stepping changes them
    swapped, _ = muon_steps(grads[:1], smooth=a, state=state())
    assert torch.allclose(swapped[0], muon_steps(grads[:1], state=state())[0][0], atol=1e-6)


def train_tiny(tmp_path, monkeypatch, name, *over):
    monkeypatch.setenv('OPTSPLIT_ROOT', str(tmp_path))
    d = tmp_path / 'data' / 'shakespeare_char'
    if not d.exists():
        d.mkdir(parents=True)
        for split in ('train', 'val'):
            np.random.default_rng(0).integers(0, 11, 5000).astype(np.uint16).tofile(d / f'{split}.bin')
        (d / 'meta.json').write_text(json.dumps({'vocab_size': 11, 'itos': list('abcdefghijk')}))
    cfg = load_config(None, [*TINY, 'suite=t', f'name={name}', *over,
                             'train.max_steps=6', 'train.batch_size=4', 'probe.n_seqs=4', 'probe.train_seqs=4',
                             'log.ckpt_steps={every: 3}', 'measure.hessian=false', 'measure.delta_loss=false'])
    save_config(cfg, tmp_path / 'cfg.yaml')
    assert train.main([str(tmp_path / 'cfg.yaml')]) == 0
    return tmp_path / 'runs' / 't'


def test_swaps_with_smoothed_muon(tmp_path, monkeypatch):
    """Out of smuon: the shadow states. Into smuon from a parent without its shadow: Muon's state."""
    runs = train_tiny(tmp_path, monkeypatch, 'smuon_d0', 'optimizer=smuon', 'shadow=[adamw, muon]', 'log.fork_steps=[3]')
    state = torch.load(runs / 'smuon_d0' / 'fork' / '0000003.pt')['optimizer']
    assert set(state) == {'smuon', 'adamw', 'muon'}
    own, shadow = (state[o]['state'] for o in ('smuon', 'muon'))
    assert all(v['smooth_step'] == 3 and 'smooth_buffer' in v for v in own.values() if 'momentum_buffer' in v)
    assert all(torch.equal(v['momentum_buffer'], shadow[k]['momentum_buffer'])  # the momentum does not see the smoothing
               for k, v in own.items() if 'momentum_buffer' in v)
    train_tiny(tmp_path, monkeypatch, 'muon_d0', 'optimizer=muon', 'shadow=[adamw]', 'log.fork_steps=[3]')
    for parent, new in (('smuon_d0', 'muon'), ('smuon_d0', 'adamw'), ('muon_d0', 'smuon')):
        name = f'{parent}_to_{new}'
        train_tiny(tmp_path, monkeypatch, name, f'optimizer={new}', f'fork.parent=t/{parent}', 'fork.step=3')
        assert (runs / name / 'done').exists()


def test_train_writes_ghost_runs(tmp_path, monkeypatch):
    runs = train_tiny(tmp_path, monkeypatch, 'adamw_d0', 'optimizer=adamw', 'ghost=[adamw, muon]')
    for rule in ('adamw', 'muon'):
        g = runs / f'adamw_d0__ghost_{rule}'
        assert (g / 'done').exists() and load_config(g / 'config.yaml').ghost_of == 't/adamw_d0'
        assert (g / 'ckpt' / '0000006.pt').exists()
    ckpt = lambda name: torch.load(runs / name / 'ckpt' / '0000006.pt')['model']
    own, real = ckpt('adamw_d0__ghost_adamw'), ckpt('adamw_d0')
    assert all(torch.equal(v, own[k]) for k, v in real.items())
