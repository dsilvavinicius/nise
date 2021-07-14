# coding: utf-8

import os
import shutil
from warnings import warn
import numpy as np
import torch
from mesh_to_sdf import get_surface_point_cloud


def create_output_paths(checkpoint_path, experiment_name, overwrite=True):
    """Helper function to create the output folders. Returns the resulting path.
    """
    full_path = os.path.join(".", checkpoint_path, experiment_name)
    if os.path.exists(full_path) and overwrite:
        shutil.rmtree(full_path)
    elif os.path.exists(full_path):
        warn("Output path exists. Not overwritting.")
        return full_path
    os.makedirs(full_path)
    return full_path


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
