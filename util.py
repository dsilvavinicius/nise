# coding: utf-8

import json
import os
import shutil
from warnings import warn
import torch
from torch.autograd import grad

def create_output_paths(checkpoint_path, experiment_name, overwrite=True):
    """Helper function to create the output folders. Returns the resulting path.
    """
    full_path = os.path.join(".", checkpoint_path, experiment_name)
    if os.path.exists(full_path) and overwrite:
        shutil.rmtree(full_path)
    elif os.path.exists(full_path):
        warn("Output path exists. Not overwritting.")
        return full_path

    os.makedirs(os.path.join(full_path, "models"))
    os.makedirs(os.path.join(full_path, "reconstructions"))
    return full_path


def load_experiment_parameters(parameters_path):
    try:
        with open(parameters_path, "r") as fin:
            parameter_dict = json.load(fin)
    except FileNotFoundError:
        warn("File '{parameters_path}' not found.")
        return {}
    return parameter_dict

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

def jacobian(y, x):
    ''' jacobian of y wrt x '''
    meta_batch_size, num_observations = y.shape[:2]
    jac = torch.zeros(meta_batch_size, num_observations, y.shape[-1], x.shape[-1]).to(y.device) # (meta_batch_size*num_points, 2, 2)
    for i in range(y.shape[-1]):
        # calculate dydx over batches for each feature value of y
        y_flat = y[...,i].view(-1, 1)
        jac[:, :, i, :] = grad(y_flat, x, torch.ones_like(y_flat), create_graph=True)[0]

    status = 0
    if torch.any(torch.isnan(jac)):
        status = -1

    return jac, status


def mean_curvature(y, x):
    grad = gradient(y, x)
    grad_norm = torch.norm(grad, dim=-1)
    unit_grad = grad.squeeze(-1)/grad_norm.unsqueeze(-1)

    Km = 0.5*divergence(unit_grad, x)
    return Km


def vector_dot(u, v):
    return torch.sum(u * v, dim=-1, keepdim=True)