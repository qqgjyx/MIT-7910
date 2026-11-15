"""The loss selected by `train.loss` (ce | mse) and the predicted distribution that goes with it. `z` are the model's
outputs (N, V), `y` the targets (N,)."""

import torch
import torch.nn.functional as F


def token_loss(z, y, kind='ce'):
    """Loss of each row. mse: squared error to the one-hot target, summed over the classes (Cohen et al. 2021),
    as |z|^2 - 2 z_y + 1."""
    if kind == 'mse':
        return z.pow(2).sum(-1) - 2 * z.gather(-1, y[:, None])[:, 0] + 1
    return F.cross_entropy(z, y, reduction='none')


def mean_loss(z, y, kind='ce'):
    return F.cross_entropy(z, y) if kind == 'ce' else token_loss(z, y, kind).mean()


def probs(z, kind='ce'):
    """Predicted distribution. mse: the minimizer of the loss is the conditional probability vector itself, so
    p = max(z, 0) normalized to sum 1 (uniform where no entry is positive)."""
    if kind == 'ce':
        return F.softmax(z, -1)
    p = z.clamp(min=0)
    s = p.sum(-1, keepdim=True)
    return torch.where(s > 0, p / s, 1 / z.size(-1))


def log_probs(z, kind='ce'):
    return F.log_softmax(z, -1) if kind == 'ce' else probs(z, kind).log()
