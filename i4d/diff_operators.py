# coding: utf-8

import torch
from torch.autograd import grad


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    """Gradient of `y` with respect to `x`
    """
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(
        y,
        [x],
        grad_outputs=grad_outputs,
        create_graph=True
    )[0]
    return grad


def vector_dot(u, v):
    return torch.sum(u * v, dim=-1, keepdim=True)


def mean_curvature(grad, x):
    grad = grad[..., 0:3]
    grad_norm = torch.norm(grad, dim=-1)
    unit_grad = grad/grad_norm.unsqueeze(-1)

    Km = divergence(unit_grad, x)
    return Km
