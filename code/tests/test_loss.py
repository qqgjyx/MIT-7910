import torch
import torch.nn.functional as F
from torch.func import functional_call

from optsplit.analysis.io import _load_logp
from optsplit.config import load_config
from optsplit.loss import log_probs, mean_loss, probs, token_loss
from optsplit.measure.hessian import HVP, flat, lanczos, split
from optsplit.models import build_model, double_backward_ok
from optsplit.probe import Probe
from test_measure import SETUPS, FakeData

V = 11


def model_and_batch(loss=None):
    torch.manual_seed(0)
    cfg = load_config(None, SETUPS['gpt'] + ([f'train.loss={loss}'] if loss else []))
    data = FakeData('lm')
    return cfg, build_model(cfg, vocab_size=V), data, data.fixed_batch('train', 6, seed=1)


def test_mse_is_squared_error_to_one_hot():
    z, y = torch.randn(5, V), torch.randint(0, V, (5,))
    direct = (z - F.one_hot(y, V)).pow(2).sum(-1)
    torch.testing.assert_close(token_loss(z, y, 'mse'), direct)
    torch.testing.assert_close(mean_loss(z, y, 'mse'), direct.mean())
    torch.testing.assert_close(token_loss(z, y).mean(), mean_loss(z, y))


def test_model_loss_follows_config():
    for loss in (None, 'ce', 'mse'):
        _, model, _, (x, y) = model_and_batch(loss)
        z = model(x).reshape(-1, V)
        torch.testing.assert_close(model.loss(x, y), mean_loss(z, y.reshape(-1), loss or 'ce'))
    assert model.loss(x, y) != F.cross_entropy(z, y.reshape(-1))


def test_mse_predictions():
    z = torch.tensor([[2., -1., 2., 0.], [-1., -2., 0., -3.]])
    torch.testing.assert_close(probs(z, 'mse'), torch.tensor([[.5, 0., .5, 0.], [.25, .25, .25, .25]]))
    torch.testing.assert_close(log_probs(z, 'ce').exp(), probs(z, 'ce'))


def test_probe_stores_raw_outputs(tmp_path):
    cfg, model, data, _ = model_and_batch('mse')
    cfg.probe.update(n_seqs=4, train_seqs=4, positions_per_seq=0)
    probe = Probe(cfg, data)
    preds, rec = probe.evaluate(model)
    x, y = probe.val
    z = model(x).reshape(-1, V)
    assert set(preds) == {'z'} and rec['val_loss'] == mean_loss(z, y.reshape(-1), 'mse').item()
    torch.testing.assert_close(preds['z'], z.half())
    torch.save(preds, tmp_path / '0.pt')  # the analysis reads the predictions off the stored outputs
    torch.testing.assert_close(_load_logp(str(tmp_path / '0.pt')).exp().float(), probs(preds['z'].float(), 'mse'),
                               atol=1e-3, rtol=0)


def test_hessian_is_of_the_mse_loss():
    _, model, _, (x, y) = model_and_batch('mse')
    named = list(model.named_parameters())
    params = [p for _, p in named]

    def loss(theta):
        d = {n: t for (n, _), t in zip(named, split(theta, params))}
        return mean_loss(functional_call(model, d, (x,)).reshape(-1, V), y.reshape(-1), 'mse')

    with double_backward_ok():
        H = torch.autograd.functional.hessian(loss, flat([p.detach() for p in params]))
        evals, _ = lanczos(HVP(model, x, y, params), len(H), 'cpu', 200, 1)
    torch.testing.assert_close(torch.tensor(evals), torch.linalg.eigvalsh(H)[-1:], rtol=1e-3, atol=1e-5)
