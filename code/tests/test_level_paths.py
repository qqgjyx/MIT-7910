"""Elastic band of constant loss (optsplit.analysis.level_paths) on toy losses and a tiny GPT."""

import math

import numpy as np
import pytest
import torch

from optsplit.analysis.level_paths import Objective, band, level_walk, profile, straight
from optsplit.config import load_config
from optsplit.models import build_model


class Toy:
    """Loss f(x) of a flat vector x, with the Objective interface."""

    def __init__(self, f):
        self.f = f

    def loss(self, x):
        return float(self.f(x))

    def loss_grad(self, x):
        x = x.detach().clone().requires_grad_(True)
        l = self.f(x)
        l.backward()
        return float(l.detach()), x.grad

    def hvp(self, x, v):
        return torch.autograd.functional.hvp(self.f, x, v)[1]


def quiet(_):
    pass


def test_flat_loss_gives_the_evenly_spaced_straight_line():
    a, b = torch.zeros(5, dtype=torch.float64), torch.arange(5, dtype=torch.float64)
    obj = Toy(lambda x: 0 * x.sum() + 1.0)
    bent = straight(a, b, 10) + torch.randn(10, 5, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    X, res = band(obj, a, b, 1.0, 1.0, n=10, init=bent, log=quiet)
    assert torch.allclose(X, straight(a, b, 10), atol=1e-3)
    assert res['ratio'] == pytest.approx(1, abs=1e-6)
    assert np.ptp(res['seg']) < 1e-3
    assert res['ok']


@pytest.mark.parametrize('sign', [1, -1])
def test_band_follows_a_circular_level_set(sign):
    """L = |x|^2 (the chord dips below c = 1) or 2 - |x|^2 (it rises above): the band follows the unit circle."""
    obj = Toy(lambda x: 1 + sign * (x.pow(2).sum() - 1))
    angle = 2 * math.pi / 3
    a = torch.tensor([1.0, 0.0], dtype=torch.float64)
    b = torch.tensor([math.cos(angle), math.sin(angle)], dtype=torch.float64)
    prof = profile(obj, a, b)
    assert (prof['dip'] > 0.2) if sign == 1 else (prof['rise'] > 0.2)
    X, res = band(obj, a, b, 1.0, 1.0, n=10, tol=0.01, log=quiet)
    assert res['ok'] and res['max_dev'] <= 0.01
    assert np.abs(X.norm(dim=1).numpy() - 1).max() < 0.01
    assert res['ratio'] == pytest.approx(angle / (2 * math.sin(angle / 2)), rel=0.01)


def test_shortening_removes_wander_on_a_sphere():
    """L = |x|^2 in 3-d, c = 1: a path on the unit sphere with random wander between two points of the equator. The
    shortening phase alone (no string phase, no LM) brings it to the great-circle arc, still on the sphere."""
    obj = Toy(lambda x: x.pow(2).sum())
    angle = 2 * math.pi / 3
    a = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    b = torch.tensor([math.cos(angle), math.sin(angle), 0.0], dtype=torch.float64)
    t = torch.linspace(0, angle, 12, dtype=torch.float64)[1:-1]
    z = 0.3 * torch.randn(10, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    wander = torch.stack([torch.cos(t), torch.sin(t), z], 1)
    wander = wander / wander.norm(dim=1, keepdim=True)
    X, res = band(obj, a, b, 1.0, 1.0, n=10, steps=0, ws=(), rounds=0, init=wander, shorten_steps=1000, log=quiet)
    arc = angle / (2 * math.sin(angle / 2))
    shorten = [h for h in res['history'] if h.get('phase') == 'shorten']
    assert shorten[0]['ratio'] > arc + 0.03
    assert res['ratio'] == pytest.approx(arc, rel=2e-3)
    assert np.abs(X.norm(dim=1).numpy() - 1).max() < 1e-3


@pytest.mark.parametrize('rotate', [False, True])
def test_band_respects_the_constraint_on_a_tiny_gpt(rotate):
    """Two tiny GPTs trained from one initialization on two data orders (the straight line between them rises by
    about 0.3): every point and every segment of the band stays within tol of c_i, which is interpolated between the
    two endpoint losses, and the band is longer than the straight line. rotate: the iterations cycle through two
    groups of batches (here the same sequences in two orders)."""
    cfg = load_config(None, {'task': 'gpt', 'model': {'block_size': 8, 'n_layer': 1, 'n_head': 2, 'n_embd': 8}})
    seqs = torch.randint(0, 11, (64, 9), generator=torch.Generator().manual_seed(1))
    batches = [(seqs[i:i + 16, :-1], seqs[i:i + 16, 1:]) for i in range(0, 64, 16)]

    def trained(order):
        torch.manual_seed(0)
        m = build_model(cfg, vocab_size=11)
        opt = torch.optim.Adam(m.parameters(), lr=0.01)
        gen = torch.Generator().manual_seed(order)
        for _ in range(300):
            i = torch.randint(0, 64, (8,), generator=gen)
            m.loss(seqs[i, :-1], seqs[i, 1:]).backward()
            opt.step()
            opt.zero_grad()
        return torch.cat([p.detach().reshape(-1) for p in m.parameters()])

    a, b = trained(1), trained(2)
    obj = Objective(build_model(cfg, vocab_size=11), batches, groups=[batches, batches[::-1]] if rotate else None)
    ca, cb = obj.loss(a), obj.loss(b)
    assert profile(obj, a, b)['rise'] > 0.2
    X, res = band(obj, a, b, ca, cb, n=10, tol=0.01, log=quiet)
    assert res['ok'] and res['max_dev'] <= 0.01
    c = ca + (cb - ca) * np.arange(1, len(X) + 1) / (len(X) + 1)
    assert np.abs(np.array([obj.loss(x) for x in X]) - c).max() <= 0.01
    assert res['ratio'] > 1.01


def test_level_walk_follows_a_circular_level_set():
    obj = Toy(lambda x: x.pow(2).sum())
    angle = 2 * math.pi / 3
    a = torch.tensor([1.0, 0.0], dtype=torch.float64)
    b = torch.tensor([math.cos(angle), math.sin(angle)], dtype=torch.float64)
    X, res = level_walk(obj, a, b, 1.0, 1.0, steps=40, tol=0.01, log=quiet)
    assert res['reached'] and res['ok']
    assert res['ratio'] == pytest.approx(angle / (2 * math.sin(angle / 2)), rel=0.01)
