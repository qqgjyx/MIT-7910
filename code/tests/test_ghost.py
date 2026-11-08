"""Open-loop ghosts (optim.Ghost) and the Muon ablations (orthogonalize=false, sv_power)."""

import json

import numpy as np
import pytest
import torch

from optsplit import train
from optsplit.config import load_config, save_config
from optsplit.models import build_model
from optsplit.optim import Ghost, build_optimizer, set_lr
from optsplit.optim.muon import muon_update

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


def test_train_writes_ghost_runs(tmp_path, monkeypatch):
    monkeypatch.setenv('OPTSPLIT_ROOT', str(tmp_path))
    d = tmp_path / 'data' / 'shakespeare_char'
    d.mkdir(parents=True)
    for split in ('train', 'val'):
        np.random.default_rng(0).integers(0, 11, 5000).astype(np.uint16).tofile(d / f'{split}.bin')
    (d / 'meta.json').write_text(json.dumps({'vocab_size': 11, 'itos': list('abcdefghijk')}))
    cfg = load_config(None, [*TINY, 'suite=t', 'name=adamw_d0', 'optimizer=adamw', 'ghost=[adamw, muon]',
                             'train.max_steps=6', 'train.batch_size=4', 'probe.n_seqs=4', 'probe.train_seqs=4',
                             'log.ckpt_steps={every: 3}', 'measure.hessian=false', 'measure.delta_loss=false'])
    save_config(cfg, tmp_path / 'cfg.yaml')
    assert train.main([str(tmp_path / 'cfg.yaml')]) == 0
    runs = tmp_path / 'runs' / 't'
    for rule in ('adamw', 'muon'):
        g = runs / f'adamw_d0__ghost_{rule}'
        assert (g / 'done').exists() and load_config(g / 'config.yaml').ghost_of == 't/adamw_d0'
        assert (g / 'ckpt' / '0000006.pt').exists()
    ckpt = lambda name: torch.load(runs / name / 'ckpt' / '0000006.pt')['model']
    own, real = ckpt('adamw_d0__ghost_adamw'), ckpt('adamw_d0')
    assert all(torch.equal(v, own[k]) for k, v in real.items())
