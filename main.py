# coding: utf-8

import argparse
import json
import os
from warnings import warn
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import BatchSampler, DataLoader
from dataset import PointCloud
from model import SIREN, SDFDecoder
from samplers import SitzmannSampler
from util import create_output_paths, gradient
from meshing import create_mesh


def sdf_loss(X, gt):
    """Loss function employed in Sitzmann et al. for SDF experiments [1].

    Parameters
    ----------
    X: dict[str=>torch.Tensor]
        Model output with the following keys: 'model_in' and 'model_out'
        with the model input and SDF values respectively.

    gt: dict[str=>torch.Tensor]
        Ground-truth data with the following keys: 'sdf' and 'normals', with
        the actual SDF values and the input data normals, respectively.

    Returns
    -------
    loss: torch.Tensor
        The calculated loss value.

    References
    ----------
    [1] Sitzmann, V., Martel, J. N. P., Bergman, A. W., Lindell, D. B.,
    & Wetzstein, G. (2020). Implicit Neural Representations with Periodic
    Activation Functions. ArXiv. Retrieved from http://arxiv.org/abs/2006.09661
    """
    gt_sdf = gt["sdf"]
    gt_normals = gt["normals"]

    coords = X["model_in"]
    pred_sdf = X["model_out"]

    grad = gradient(pred_sdf, coords)
    sdf_constraint = torch.where(
        gt_sdf != -1,
        pred_sdf,
        torch.zeros_like(pred_sdf)
    )
    inter_constraint = torch.where(
        gt_sdf != -1,
        torch.zeros_like(pred_sdf),
        torch.exp(-1e2 * torch.abs(pred_sdf))
    )
    normal_constraint = torch.where(
        gt_sdf != -1,
        1 - F.cosine_similarity(grad, gt_normals, dim=-1)[..., None],
        torch.zeros_like(grad[..., :1])
    )
    grad_constraint = torch.abs(grad.norm(dim=-1) - 1)
    return torch.abs(sdf_constraint).mean() * 3e3 + \
        inter_constraint.mean() * 1e2 + \
        normal_constraint.mean() * 1e2 + \
        grad_constraint.mean() * 5e1


def train_model(dataset, model, device, train_config, silent=False):
    BATCH_SIZE = train_config["batch_size"]
    EPOCHS = train_config["epochs"]
    EPOCHS_TIL_CHECKPOINT = train_config["epochs_to_checkpoint"]

    loss_fn = train_config["loss_fn"]
    optim = train_config["optimizer"]
    sampler = train_config["sampler"]
    train_loader = DataLoader(
        dataset,
        batch_sampler=BatchSampler(sampler, batch_size=BATCH_SIZE, drop_last=False)
    )
    model.to(device)

    losses = [0.] * EPOCHS
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, data in enumerate(train_loader, start=0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data["coords"].to(device)
            gt = {
                "sdf": data["sdf"].to(device),
                "normals": data["normals"].to(device)
            }

            # zero the parameter gradients
            optim.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_fn(outputs, gt)
            loss.backward()
            optim.step()

            # accumulate statistics
            running_loss += loss.item()

        # saving the model at checkpoints
        if not epoch % EPOCHS_TIL_CHECKPOINT and epoch:
            torch.save(
                model.state_dict(),
                os.path.join(full_path, f"model_{epoch}.pth")
            )

        losses[epoch] = running_loss
        if not silent:
            print(f"Epoch: {epoch} - Loss: {running_loss}")

    # saving the final model
    torch.save(
        model.state_dict(),
        os.path.join(full_path, "model_final.pth")
    )

    return losses


def load_experiment_parameters(parameters_path):
    try:
        with open(parameters_path, "r") as fin:
            parameter_dict = json.load(fin)
    except FileNotFoundError:
        warn("File '{parameters_path}' not found.")
        return {}
    return parameter_dict


if __name__ == "__main__":
    p = argparse.ArgumentParser(usage="python main.py path_to_experiments")
    p.add_argument(
        "experiment_path",
        help="Path to the JSON experiment description file"
    )
    p.add_argument(
        "-s", "--silent", action="store_true",
        help="Suppresses informational output messages"
    )
    args = p.parse_args()
    parameter_dict = load_experiment_parameters(args.experiment_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    EPOCHS = parameter_dict["num_epochs"]
    SAMPLES_ON_SURFACE = parameter_dict["sampling_opts"]["samples_on_surface"]
    SAMPLES_OFF_SURFACE = parameter_dict["sampling_opts"]["samples_off_surface"]
    BATCH_SIZE = parameter_dict["batch_size"]
    EPOCHS_TIL_CHECKPOINT = parameter_dict["epochs_to_checkpoint"]

    full_path = create_output_paths(
        parameter_dict["checkpoint_path"],
        parameter_dict["experiment_name"],
        overwrite=False
    )

    dataset = PointCloud(
        os.path.join("data", parameter_dict["dataset"]),
        SAMPLES_ON_SURFACE,
        scaling="sphere",
        off_surface_sdf=-1
    )
    sampler = SitzmannSampler(dataset, SAMPLES_OFF_SURFACE)
    model = SIREN(
        n_in_features=3,
        n_out_features=1,
        hidden_layer_config=parameter_dict["network"]["hidden_layer_nodes"],
        w0=parameter_dict["network"]["w0"]
    )

    opt_params = parameter_dict["optimizer"]
    if opt_params["type"] == "adam":
        optimizer = torch.optim.Adam(
            lr=opt_params["lr"],
            params=model.parameters()
        )

    config_dict = {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "epochs_to_checkpoint": EPOCHS_TIL_CHECKPOINT,
        "sampler": sampler,
        "log_path": full_path,
        "optimizer": optimizer,
        "loss_fn": sdf_loss
    }

    losses = train_model(dataset, model, device, config_dict, silent=args.silent)
    np.savetxt(os.path.join(full_path, "losses.txt"), np.array(losses))

    # full_path = "logs/double_torus"
    mesh_file = parameter_dict["reconstruction"]["output_file"]
    mesh_resolution = parameter_dict["reconstruction"]["resolution"]
    decoder = SDFDecoder(
        os.path.join(full_path, "model_final.pth"),
        n_in_features=3,
        n_out_features=1,
        hidden_layer_config=parameter_dict["network"]["hidden_layer_nodes"],
        w0=parameter_dict["network"]["w0"]
    )
    create_mesh(decoder, os.path.join(full_path, mesh_file), N=mesh_resolution)
