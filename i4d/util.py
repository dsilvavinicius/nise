# coding: utf-8

from collections import OrderedDict
import json
import math
import os
import os.path as osp
import shutil
import numpy as np
import torch
from warnings import warn
from i4d.diff_operators import gradient, mean_curvature
from i4d.meshing import create_mesh, save_ply
from i4d.model import SIREN


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


def siren_v1_to_v2(model_in, check_equals=False):
    """Converts the models trained using the old class to the new format.

    Parameters
    ----------
    model_in: OrderedDict
        Model trained by our old SIREN version (Sitzmann code).

    check_equals: boolean, optional
        Whether to check if the converted models weight match. By default this
        is False.

    Returns
    -------
    model_out: OrderedDict
        The input model converted to a format recognizable by our version of
        SIREN.

    divergences: list[tuple[str, str]]
        If `check_equals` is True, then this list contains the keys where the
        original and converted model dictionaries are not equal. Else, this is
        an empty list.

    See Also
    --------
    `model.SIREN`
    """
    model_out = OrderedDict()
    for k, v in model_in.items():
        model_out[k[4:]] = v

    divergences = []
    if check_equals:
        for k in model_in.keys():
            test = model_in[k] == model_out[k[4:]]
            if test.sum().item() != test.numel():
                divergences.append((k, k[4:]))

    return model_out, divergences


def from_pth(path, device="cpu", w0=1, ww=None):
    """Builds a SIREN given a weights file.

    Parameters
    ----------
    path: str
        Path to the pth file.

    device: str, optional
        Device to load the weights. Default value is cpu.

    w0: number, optional
        Frequency parameter for the first layer. Default value is 1.

    ww: number, optional
        Frequency parameter for the intermediate layers. Default value is None,
        we will assume that ww = w0 in this case.

    Returns
    -------
    model: torch.nn.Module
        The resulting model.

    Raises
    ------
    FileNotFoundError if `path` points to a non-existing file.
    """
    if not osp.exists(path):
        raise FileNotFoundError(f"Weights file not found at \"{path}\"")

    weights = torch.load(path, map_location=torch.device(device))
    # Each layer has two tensors, one for weights other for biases.
    n_layers = len(weights) // 2
    hidden_layer_config = [None] * (n_layers - 1)
    keys = list(weights.keys())

    bias_keys = [k for k in keys if "bias" in k]
    i = 0
    while i < (n_layers - 1):
        k = bias_keys[i]
        hidden_layer_config[i] = weights[k].shape[0]
        i += 1

    n_in_features = weights[keys[0]].shape[1]
    n_out_features = weights[keys[-1]].shape[0]
    model = SIREN(
        n_in_features=n_in_features,
        n_out_features=n_out_features,
        hidden_layer_config=hidden_layer_config,
        w0=w0, ww=ww, delay_init=True
    )

    # Loads the weights. Converts to version 2 if they are from the old version
    # of SIREN.
    try:
        model.load_state_dict(weights)
    except RuntimeError:
        print("Found weights from old version of SIREN. Converting to v2.")
        new_weights, diff = siren_v1_to_v2(weights, True)
        new_weights_file = path.split(".")[0] + "_v2.pth"
        torch.save(new_weights, new_weights_file)
        model.load_state_dict(new_weights)

    return model


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
    i4d.meshing.create_mesh
    """
    model = model.eval()
    with torch.no_grad():
        for t in times:
            verts, faces, normals, _ = create_mesh(
                model,
                filename=osp.join(meshpath, f"time_{t}.ply"),
                t=t,  # time instant for 4d SIREN function
                N=resolution,
                device=device
            )


def reconstruct_with_curvatures(model, times, meshpath, resolution=256, device="cpu"):
    attrs = [("nx", "f4"), ("ny", "f4"), ("nz", "f4"), ("quality", "f4")]
    BATCH_SIZE = 10000
    # attrs = [("quality", "f4")]
    model = model.eval()
    for t in times:
        verts, faces, normals, _ = create_mesh(
            model,
            # filename=osp.join(meshpath, f"time_{t}.ply"),
            t=t,  # time instant for 4d SIREN function
            N=resolution,
            device=device
        )

        verts = torch.from_numpy(verts)
        coords = torch.cat((verts, t*torch.ones_like(verts[...,-1:])), dim=1).squeeze(0).to(device)
        nsteps = int(math.ceil(verts.shape[0] / BATCH_SIZE))
        grads = torch.zeros_like(coords)
        curvs = torch.zeros((grads.shape[0], 1))
        for s in range(nsteps):
            a = s*BATCH_SIZE
            b = (s+1)*BATCH_SIZE
            out = model(coords[a:b, ...].unsqueeze(0).float())
            X = out['model_in']
            y = out['model_out']
            g = gradient(y, X)
            c = mean_curvature(g, X)
            grads[a:b, ...] = g.detach().squeeze(0)
            curvs[a:b, ...] = c.detach().squeeze(0)

        verts = np.hstack((verts.cpu().numpy(), grads[...,0:3].cpu().numpy(), curvs.cpu().numpy()))
        save_ply(
            verts=verts, faces=faces,
            filename=osp.join(meshpath, f"time_{t}_meancurv.ply"),
            vertex_attributes=attrs
        )
