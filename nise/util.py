# coding: utf-8

import json
import math
import os
import os.path as osp
import shutil
import numpy as np
import torch
from warnings import warn
from nise.diff_operators import gradient, mean_curvature
from nise.meshing import create_mesh, save_ply


def create_output_paths(checkpoint_path, experiment_name, overwrite=True):
    """Helper function to create the output folders. Returns the resulting
    path.
    """
    full_path = os.path.join(".", checkpoint_path, experiment_name)
    if os.path.exists(full_path) and overwrite:
        shutil.rmtree(full_path)
    elif os.path.exists(full_path):
        warn("Output path exists. Not overwritting.")
        return full_path

    os.makedirs(os.path.join(full_path, "models"))
    os.makedirs(os.path.join(full_path, "reconstructions"))
    os.makedirs(os.path.join(full_path, "kaolin"))
    os.makedirs(os.path.join(full_path, "summaries"))

    return full_path


def load_experiment_parameters(parameters_path):
    try:
        with open(parameters_path, "r") as fin:
            parameter_dict = json.load(fin)
    except FileNotFoundError:
        warn("File '{parameters_path}' not found.")
        return {}
    return parameter_dict


def reconstruct_at_times(model, times, meshpath, resolution=256, device="cpu"):
    """Runs marching cubes on `model` at times `times`.

    Parameters
    ----------
    model: torch.nn.Module
        The model to run the inference. Must accept $\mathbb{R}^4$ inputs.

    times: collection of numbers
        The timesteps to use as input for `model`. The number of meshes
        generated will be `len(times)`.

    meshpath: str, PathLike
        Base folder to save all meshes.

    resolution: int, optional
        Marching cubes resolution. The input volume will have
        `resolution` ** 3 voxels. Default value is 256.

    device: str or torch.Device, optional
        The device where we will run the inference on `model`.
        Default value is "cpu".

    See Also
    --------
    nise.meshing.create_mesh
    """
    model = model.eval()
    with torch.no_grad():
        for t in times:
            create_mesh(
                model,
                filename=osp.join(meshpath, f"time_{t}.ply"),
                t=t,  # time instant for 4d SIREN function
                N=resolution,
                device=device
            )


def estimate_differential_properties(
        model: torch.nn.Module, coords: torch.Tensor, with_curvs: bool = True,
        device: str = "cpu", batchsize: int = 10000
    ) -> np.ndarray:
    """Estimates gradient and curvature (optional) at `coords` using `model`.

    Parameters
    ----------
    model: torch.nn.Module
        The model to run the inference. Must accept $\mathbb{R}^4$ inputs.

    coords: numpy.ndarray
        The space-time coordinates to estimate the gradient and curvature on.

    with_curvs: bool, optional
        Whether to estimate the curvatures (True, default) or not (False).

    device: str or torch.Device, optional
        The device where we will run the inference on `model`.
        Default value is "cpu".

    batchsize: int, optional
        Number of points to perform the inference on at each step. We will
        iterate sequentially on the rows of `coords` running the inference on
        `batchsize` points. Default value is 10000. Tweak this to fit your
        specs.

    Returns
    -------
    verts: np.ndarray
        The output coords appended with normals and, optionally, mean
        curvature.

    See Also
    --------
    nise.diff_operators.gradient, nise.diff_operators.mean_curvature
    """
    model = model.eval()

    grads = torch.zeros_like(coords, device=device, requires_grad=False)
    if with_curvs:
        curvs = torch.zeros((coords.shape[0], 1), device=device, requires_grad=False)
    
    #computing the gradient in batches
    steps = int(math.ceil(coords.shape[0] / batchsize))
    for s in range(steps):
        a = s * batchsize
        b = (s+1) * batchsize
        out = model(coords[a:b, ...].unsqueeze(0).float())
        X = out['model_in']
        y = out['model_out']
        g = gradient(y, X)
        grads[a:b, ...] = g.detach().squeeze(0)
        if with_curvs:
            curvs[a:b, ...] = mean_curvature(g, X).detach().squeeze(0)

    verts = np.hstack((
        coords[..., :3].detach().cpu().numpy(),
        grads[..., :3].detach().cpu().numpy()
    ))
    if with_curvs:
        verts = np.hstack((
            verts,
            curvs.detach().cpu().numpy()
        ))

    return verts


def reconstruct_with_curvatures(model, times, meshpath, resolution=256,
                                device="cpu", batch_size=10000):
    attrs = [("nx", "f4"), ("ny", "f4"), ("nz", "f4"), ("quality", "f4")]
    model = model.eval()
    for t in times:
        verts, faces, _, _ = create_mesh(
            model,
            t=t,  # time instant for 4d SIREN function
            N=resolution,
            device=device
        )

        verts = torch.from_numpy(verts)
        coords = torch.cat((verts, t*torch.ones_like(verts[..., :1])), dim=1).squeeze(0).to(device)
        nsteps = int(math.ceil(verts.shape[0] / batch_size))
        grads = torch.zeros_like(coords)
        curvs = torch.zeros((grads.shape[0], 1))
        for s in range(nsteps):
            a = s * batch_size
            b = (s+1) * batch_size
            out = model(coords[a:b, ...].unsqueeze(0).float())
            X = out['model_in']
            y = out['model_out']
            g = gradient(y, X)
            c = mean_curvature(g, X)
            grads[a:b, ...] = g.detach().squeeze(0)
            curvs[a:b, ...] = c.detach().squeeze(0)

        verts = np.hstack((
            verts.detach().cpu().numpy(),
            grads[..., :3].detach().cpu().numpy(),
            curvs.detach().cpu().numpy()
        ))
        save_ply(
            verts=verts, faces=faces,
            filename=osp.join(meshpath, f"time_{t}.ply"),
            vertex_attributes=attrs
        )
