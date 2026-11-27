"""Open-loop ghosts (optim.Ghost) and the Muon ablations (orthogonalize=false, sv_power, two_track_steps, lc_steps)."""

import json

import numpy as np
import pytest
import torch

from optsplit import train
from optsplit.config import load_config, save_config
from optsplit.models import build_model
from optsplit.optim import Ghost, build_optimizer, set_lr
from optsplit.optim.muon import (GRAM, LC, SV_FLOOR, muon_update, sv_power_gram, sv_power_lc, sv_power_two_track,
                                 two_track_errors)

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


def test_muon_negative_sv_power():
    torch.manual_seed(0)
    U, _ = torch.linalg.qr(torch.randn(16, 16))
    V, _ = torch.linalg.qr(torch.randn(48, 16))
    S = torch.logspace(0, -4, 16)
    update = lambda grad, p: muon_update(grad, torch.zeros(16, 48), sv_power=p)
    for p in (-0.1, -0.5):  # above the floor: exactly S^p, so the smallest singular direction becomes the largest
        u = update((U * S) @ V.T, p)
        s = S.pow(p) * (16 ** 0.5 / S.pow(p).norm())
        assert torch.allclose(torch.linalg.svdvals(u), s.flip(0), rtol=1e-3)
        assert torch.dot(U[:, -1], u @ V[:, -1]).item() == pytest.approx(s[-1].item(), rel=1e-3)
    S[8:] = 0  # rank-deficient (and zero) momentum: finite, the null directions get the floor's weight
    for grad in ((U * S) @ V.T, torch.zeros(16, 48)):
        u = update(grad, -0.5)
        assert torch.isfinite(u).all() and u.norm().item() == pytest.approx(4, rel=1e-4)
    s = torch.linalg.svdvals(update((U * S) @ V.T, -0.5))
    assert s[0] / s[-1] <= SV_FLOOR ** -0.5 * 1.01
    g = torch.randn(16, 48)  # p >= 0 does not go through the floor: the plain formula, bit for bit
    for p in (0.0, 0.5):
        A, s, Bh = torch.linalg.svd(g.lerp(g * 0.05, 0.95), full_matrices=False)
        plain = (A * s.pow(p)) @ Bh
        assert torch.equal(update(g, p), plain * (4 / (plain.norm() + 1e-7)))


def wide_spectrum(shape, low=-6.0, seed=0):
    """Matrix with singular values from 1 down to 10^low, evenly spaced in the logarithm, and its factors."""
    gen = torch.Generator().manual_seed(seed)
    n = min(shape)
    U, _ = torch.linalg.qr(torch.randn(shape[0], n, generator=gen, dtype=torch.float64))
    V, _ = torch.linalg.qr(torch.randn(shape[1], n, generator=gen, dtype=torch.float64))
    S = torch.logspace(0, low, n, dtype=torch.float64)
    return ((U * S) @ V.T).float(), U, S, V


@pytest.mark.parametrize('p,tol', [(-0.3, 0.15), (-0.5, 0.2)])
@pytest.mark.parametrize('shape', [(96, 96), (64, 256), (256, 64)])
def test_two_track_12_steps_is_the_exact_negative_power(p, tol, shape):
    grad = wide_spectrum(shape)[0]
    update = lambda **kw: muon_update(grad, torch.zeros(shape), sv_power=p, **kw)
    exact, approx = update(), update(two_track_steps=12)
    assert approx.dtype == torch.float32 and approx.norm().item() == pytest.approx(exact.norm().item(), rel=1e-4)
    assert ((approx - exact).norm() / exact.norm()).item() < tol  # about 0.1
    with torch.autocast('cpu', dtype=torch.bfloat16):  # float32 whatever the autocast
        assert torch.equal(update(two_track_steps=12), approx)
    assert ((update(two_track_steps=5) - exact).norm() / exact.norm()).item() > 0.5  # 5 steps miss the small values


@pytest.mark.parametrize('p,steps,band', [(-0.3, 5, 1.3), (-0.3, 8, 1.4), (-0.3, 12, 1.6),
                                          (-0.5, 5, 1.65), (-0.5, 8, 2.0), (-0.5, 12, 2.7)])
def test_two_track_power_law_above_the_cutoff(p, steps, band):
    """The map on singular values is c S^p for S / |G|_F above 4 / 3.4445^steps; far below, much less than that."""
    cutoff = 4 / 3.4445 ** steps
    G, U, S, V = wide_spectrum((128, 128), low=-8.0 if steps < 12 else -6.0)  # float32 holds nothing under 1e-7
    Z = sv_power_two_track(G, p, steps).double()
    z = ((U.T @ Z) * V.T).sum(1)  # u_i^T Z v_i: G's singular vectors are Z's
    rel = S / S.norm()
    ratio = z / rel.pow(p)
    above = ratio[rel >= cutoff]
    assert above.min() > 0 and above.max() / above.min() < band
    if steps < 12:
        assert ratio[rel < cutoff / 100].abs().max() < 0.1 * above.min()


def test_two_track_options_reach_the_optimizer():
    power = ['optim.muon.sv_power=-0.3']
    assert differ(run('muon', '[]', extra=power)[0], run('muon', '[]', extra=power + ['optim.muon.two_track_steps=5'])[0])
    with pytest.raises(AssertionError):  # coefficients exist for p = -0.3 and -0.5 only
        run('muon', '[]', extra=['optim.muon.sv_power=-0.1', 'optim.muon.two_track_steps=5'])


def test_two_track_errors():
    cfg = load_config(None, [*TINY, 'optimizer=muon', 'optim.muon.sv_power=-0.3'])
    torch.manual_seed(0)
    model = build_model(cfg, vocab_size=11)
    opt = build_optimizer('muon', model, cfg)
    x = torch.randint(0, 11, (4, 8))
    model.loss(x, x.roll(-1, 1)).backward()
    opt.step()
    rec = two_track_errors(opt)
    n = len(opt.param_groups[0]['params'])
    assert set(rec) == {f'{kind}_{k}' for kind in ('err', 'cos')
                        for k in (5, 8, 12, 'lc5', 'lc6', 'lc8', 'lc12', 'lc5m', 'lcgram', 'svd32')}
    assert all(len(v) == n for v in rec.values())
    assert all(e == pytest.approx(max(0, 2 - 2 * c) ** 0.5)
               for k in (5, 8, 12) for e, c in zip(rec[f'err_{k}'], rec[f'cos_{k}']))
    # no bound on the values: these gradients have a null direction, which only the exact update steps along
    assert all(abs(c) <= 1 + 1e-9 for k, v in rec.items() if k.startswith('cos') for c in v)
    json.dumps(rec)


@pytest.mark.parametrize('steps,eps,band,low', [(5, 3e-3, 1.3, -2.0), (6, 1e-3, 1.4, -2.0), (8, 2e-4, 1.3, -3.0),
                                               (12, 2e-6, 1.5, -5.0)])
@pytest.mark.parametrize('shape', [(128, 128), (64, 256), (256, 64)])
def test_lc_power_law_above_the_cutoff(steps, eps, band, low, shape):
    """LC: c S^p for S above eps of the Schatten-8 norm, much less far below; with no singular value below eps (spectrum
    down to 10^low) the update is the exact one up to about 0.1."""
    G, U, S, V = wide_spectrum(shape, low=-6.0)
    Z = sv_power_lc(G, -0.3, steps).double()
    z = ((U.T @ Z) * V.T).sum(1)
    rel = S / S.pow(8).sum() ** 0.125
    ratio = z / rel.pow(-0.3)
    above = ratio[rel >= eps]
    assert above.min() > 0 and above.max() / above.min() < band  # about +-13%, +-15%, +-5%, +-16%
    if steps < 12:  # 12 steps: float32 holds nothing under eps / 100
        assert ratio[rel < eps / 100].abs().max() < 0.05 * above.min()
    grad = wide_spectrum(shape, low=low)[0]
    exact = muon_update(grad, torch.zeros(shape), sv_power=-0.3)
    approx = muon_update(grad, torch.zeros(shape), sv_power=-0.3, lc_steps=steps)
    assert approx.dtype == torch.float32 and ((approx - exact).norm() / exact.norm()).item() < 0.15
    if steps >= 8:
        with torch.autocast('cpu', dtype=torch.bfloat16):  # float32 whatever the autocast
            assert torch.equal(muon_update(grad, torch.zeros(shape), sv_power=-0.3, lc_steps=steps), approx)


@pytest.mark.parametrize('shape', [(128, 128), (64, 256), (256, 64)])
def test_lc5m_is_its_scalar_map_between_muon_and_the_power(shape):
    """LC5m (lc_steps='5m'): the matrix function is the scalar map g of its coefficients at s / |G|_S8 (bfloat16
    iterates, cancelling in the sum: within 6% of the largest value), and g lies between Muon's map (its running maximum) and s^p."""
    steps, w = LC[-0.3, '5m']
    scalar = lambda s, coeffs: [s := s * (a + b * s ** 2 + c * s ** 4) for a, b, c in coeffs]
    G, U, S, V = wide_spectrum(shape, low=-6.0)
    z = ((U.T @ sv_power_lc(G, -0.3, '5m').double()) * V.T).sum(1)
    s = S / S.pow(8).sum() ** 0.125
    g = w[0] * s + sum(wk * y for wk, y in zip(w[1:], scalar(s, steps)))
    assert ((z - g).abs() / g.abs().max()).max() < 0.06
    s = torch.logspace(-12, 0, 200001, dtype=torch.float64)
    g = w[0] * s + sum(wk * y for wk, y in zip(w[1:], scalar(s, steps)))
    muon = torch.cummax(scalar(s, [(3.4445, -4.7750, 2.0315)] * 5)[-1], 0).values
    lo, hi = torch.minimum(muon, s ** -0.3), torch.maximum(muon, s ** -0.3)
    assert (g >= lo * (1 - 1e-6)).all() and (g <= hi * (1 + 1e-6)).all()


@pytest.mark.parametrize('shape', [(128, 128), (64, 256), (256, 64)])
def test_gram_lc_is_its_scalar_map_between_muon_and_the_power(shape):
    """Gram LC (lc_steps='gram'): the matrix function is the scalar map f(s) = s (w_I + sum_k w_k c_k(s^2 + d)) at
    s / |G|_S8 (float32: within 1% wherever s >= 1e-3), and f lies between Muon's map (its running maximum) and s^p."""
    steps, w, d = GRAM[-0.3]

    def f(s):
        c = [s * s + d]
        for a, b in steps:
            c.append(a * c[-1] + b * c[-1] ** 2)
        return s * (w[0] + sum(wk * ck for wk, ck in zip(w[1:], c)))

    G, U, S, V = wide_spectrum(shape, low=-6.0)
    z = ((U.T @ sv_power_gram(G, -0.3).double()) * V.T).sum(1)
    s = S / S.pow(8).sum() ** 0.125
    assert ((z - f(s)).abs() / f(s))[s >= 1e-3].max() < 0.01
    update = muon_update(G, torch.zeros(shape), sv_power=-0.3, lc_steps='gram')
    assert update.dtype == torch.float32 and update.shape == shape
    s = torch.logspace(-12, 0, 200001, dtype=torch.float64)
    scalar = lambda s, coeffs: [s := s * (a + b * s ** 2 + c * s ** 4) for a, b, c in coeffs]
    muon = torch.cummax(scalar(s, [(3.4445, -4.7750, 2.0315)] * 5)[-1], 0).values
    lo, hi = torch.minimum(muon, s ** -0.3), torch.maximum(muon, s ** -0.3)
    assert (f(s) >= lo * (1 - 1e-6)).all() and (f(s) <= hi * (1 + 1e-6)).all()


def test_lc_options_reach_the_optimizer():
    power = ['optim.muon.sv_power=-0.3']
    assert differ(run('muon', '[]', extra=power)[0], run('muon', '[]', extra=power + ['optim.muon.lc_steps=5'])[0])
    assert differ(run('muon', '[]', extra=power + ['optim.muon.lc_steps=5'])[0],
                  run('muon', '[]', extra=power + ['optim.muon.lc_steps=gram'])[0])
    with pytest.raises(AssertionError):  # coefficients for p = -0.3 with 5, 6, 8 or 12 steps only
        run('muon', '[]', extra=['optim.muon.sv_power=-0.3', 'optim.muon.lc_steps=4'])


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


def test_train_two_track_run(tmp_path, monkeypatch):
    """A two-track run logs its error against the exact update at log.two_track_check; log.preds=false stores no outputs."""
    monkeypatch.setenv('OPTSPLIT_ROOT', str(tmp_path))
    d = tmp_path / 'data' / 'shakespeare_char'
    d.mkdir(parents=True)
    for split in ('train', 'val'):
        np.random.default_rng(0).integers(0, 11, 5000).astype(np.uint16).tofile(d / f'{split}.bin')
    (d / 'meta.json').write_text(json.dumps({'vocab_size': 11, 'itos': list('abcdefghijk')}))
    cfg = load_config(None, [*TINY, 'suite=t', 'name=tt', 'optimizer=muon', 'optim.muon.sv_power=-0.3',
                             'optim.muon.two_track_steps=5', 'log.two_track_check=[2]', 'log.preds=false',
                             'train.max_steps=4', 'train.batch_size=4', 'probe.n_seqs=4', 'probe.train_seqs=4',
                             'measure.hessian=false', 'measure.delta_loss=false'])
    save_config(cfg, tmp_path / 'cfg.yaml')
    assert train.main([str(tmp_path / 'cfg.yaml')]) == 0
    out = tmp_path / 'runs' / 't' / 'tt'
    recs = [json.loads(line) for line in (out / 'metrics.jsonl').read_text().splitlines()]
    assert [r['step'] for r in recs if r['kind'] == 'two_track'] == [2]
    assert not list((out / 'preds').iterdir()) and (out / 'ckpt' / '0000004.pt').exists()
