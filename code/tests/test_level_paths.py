"""Elastic band of constant loss (optsplit.analysis.level_paths) on toy losses and a tiny GPT."""

import math

import numpy as np
import pytest
import torch

from optsplit.analysis.level_paths import Objective, band, profile, straight
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
        return float(l), x.grad


def quiet(_):
    pass


def test_flat_loss_gives_the_evenly_spaced_straight_line():
    a, b = torch.zeros(5, dtype=torch.float64), torch.arange(5, dtype=torch.float64)
    obj = Toy(lambda x: 0 * x.sum() + 1.0)
    bent = straight(a, b, 10) + torch.randn(10, 5, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    X, res = band(obj, a, b, 1.0, 1.0, n=10, init=bent, log=quiet)
    assert torch.allclose(X, straight(a, b, 10), atol=1e-6)
    assert res['ratio'] == pytest.approx(1, abs=1e-9)
    assert np.ptp(res['seg']) < 1e-6
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


def test_band_respects_the_constraint_on_a_tiny_gpt():
    """Two tiny GPTs trained from different initializations (so the straight line between them rises): every point
    and every segment stays within tol of c_i, which is interpolated between the two endpoint losses."""
    cfg = load_config(None, {'task': 'gpt', 'model': {'block_size': 8, 'n_layer': 1, 'n_head': 2, 'n_embd': 8}})
    seqs = torch.randint(0, 11, (32, 9), generator=torch.Generator().manual_seed(1))
    batches = [(seqs[:16, :-1], seqs[:16, 1:]), (seqs[16:, :-1], seqs[16:, 1:])]

    def trained(seed, steps):
        torch.manual_seed(seed)
        m = build_model(cfg, vocab_size=11)
        opt = torch.optim.Adam(m.parameters(), lr=0.02)
        for i in range(steps):
            m.loss(*batches[i % 2]).backward()
            opt.step()
            opt.zero_grad()
        return torch.cat([p.detach().reshape(-1) for p in m.parameters()])

    a, b = trained(0, 100), trained(1, 110)
    obj = Objective(build_model(cfg, vocab_size=11), batches)
    ca, cb = obj.loss(a), obj.loss(b)
    assert profile(obj, a, b)['rise'] > 0.1
    X, res = band(obj, a, b, ca, cb, n=10, tol=0.01, log=quiet)
    assert res['ok'] and res['max_dev'] <= 0.01
    c = ca + (cb - ca) * np.arange(1, len(X) + 1) / (len(X) + 1)
    assert np.abs(np.array([obj.loss(x) for x in X]) - c).max() <= 0.01
    assert res['ratio'] >= 1
