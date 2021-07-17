# coding: utf-8

import argparse
import json
import os
from warnings import warn
import pandas as pd
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
    loss: dict[str=>torch.Tensor]
        The calculated loss values for each constraint.

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
    return {
        "sdf_constraint": torch.abs(sdf_constraint).mean() * 3e3,
        "inter_constraint": inter_constraint.mean() * 1e2,
        "normal_constraint": normal_constraint.mean() * 1e2,
        "grad_constraint": grad_constraint.mean() * 5e1,
    }


def train_model(dataset, model, device, train_config, silent=False):
    BATCH_SIZE = train_config["batch_size"]
    EPOCHS = train_config["epochs"]
    EPOCHS_TIL_CHECKPOINT = 0
    if "epochs_to_checkpoint" in train_config and train_config["epochs_to_checkpoint"] > 0:
        EPOCHS_TIL_CHECKPOINT = train_config["epochs_to_checkpoint"]

    EPOCHS_TIL_RECONSTRUCTION = 0
    if "epochs_to_reconstruct" in train_config and train_config["epochs_to_reconstruct"] > 0:
        EPOCHS_TIL_RECONSTRUCTION = train_config["epochs_to_reconstruct"]

    loss_fn = train_config["loss_fn"]
    optim = train_config["optimizer"]
    sampler = train_config["sampler"] if "sampler" in train_config else None
    if sampler is not None:
        train_loader = DataLoader(
            dataset,
            batch_sampler=BatchSampler(sampler, batch_size=BATCH_SIZE, drop_last=False)
        )
    else:
        train_loader = DataLoader(
            dataset,
            shuffle=True,
            batch_size=1,
            pin_memory=True,
            num_workers=0
        )
    model.to(device)

    losses = dict()
    for epoch in range(EPOCHS):
        running_loss = dict()
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
            train_loss = torch.zeros((1, 1), device=device)
            for it, l in loss.items():
                train_loss += l
                # accumulating statistics per loss term
                if it not in running_loss:
                    running_loss[it] = l.item()
                else:
                    running_loss[it] += l.item()
            train_loss.backward()
            optim.step()

            # accumulate statistics
            # running_loss += train_loss.item()

        for it, l in running_loss.items():
            if it in losses:
                losses[it][epoch] = l
            else:
                losses[it] = [0.] * EPOCHS
                losses[it][epoch] = l

        if not silent:
            epoch_loss = 0
            for k, v in running_loss.items():
                epoch_loss += v
            print(f"Epoch: {epoch} - Loss: {epoch_loss}")

        # saving the model at checkpoints
        if epoch and EPOCHS_TIL_CHECKPOINT and not epoch % EPOCHS_TIL_CHECKPOINT:
            if not silent:
                print(f"Saving model for epoch {epoch}")
            torch.save(
                model.state_dict(),
                os.path.join(full_path, f"model_{epoch}.pth")
            )

        # reconstructing a mesh at checkpoints
        if epoch and EPOCHS_TIL_RECONSTRUCTION and not epoch % EPOCHS_TIL_RECONSTRUCTION:
            if not silent:
                print(f"Reconstructing mesh for epoch {epoch}")

            mesh_file = f"{epoch}"
            mesh_resolution = train_config["mc_resolution"]
            decoder = SDFDecoder(
                model.state_dict(),
                n_in_features=3,
                n_out_features=1,
                hidden_layer_config=[x[0].out_features for x in model.net[:-1]],
                w0=model.w0
            )
            create_mesh(
                decoder,
                os.path.join(full_path, mesh_file),
                N=mesh_resolution
            )

    # saving the final model
    torch.save(
        model.state_dict(),
        os.path.join(full_path, "model_final.pth")
    )

    return losses


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

    sampling_config = parameter_dict["sampling_opts"]

    full_path = create_output_paths(
        parameter_dict["checkpoint_path"],
        parameter_dict["experiment_name"],
        overwrite=False
    )

    # Saving the parameters to the output path
    with open(os.path.join(full_path, "params.json"), "w+") as fout:
        json.dump(parameter_dict, fout, indent=4)

    dataset = PointCloud(
        os.path.join("data", parameter_dict["dataset"]),
        sampling_config["samples_on_surface"],
        scaling="bbox",
        off_surface_sdf=-1,
        random_surf_samples=sampling_config["random_surf_samples"],
        silent=False
    )
    sampler = SitzmannSampler(dataset, sampling_config["samples_off_surface"])
    model = SIREN(
        n_in_features=3,
        n_out_features=1,
        hidden_layer_config=parameter_dict["network"]["hidden_layer_nodes"],
        w0=parameter_dict["network"]["w0"]
    )
    if not args.silent:
        print(model.net)

    opt_params = parameter_dict["optimizer"]
    if opt_params["type"] == "adam":
        optimizer = torch.optim.Adam(
            lr=opt_params["lr"],
            params=model.parameters()
        )

    config_dict = {
        "epochs": parameter_dict["num_epochs"],
        "batch_size": parameter_dict["batch_size"],
        "epochs_to_checkpoint": parameter_dict["epochs_to_checkpoint"],
        "sampler": sampler,
        "log_path": full_path,
        "optimizer": optimizer,
        "loss_fn": sdf_loss,
        "epochs_to_reconstruct": 10,
        "mc_resolution": parameter_dict["reconstruction"]["resolution"]
    }

    losses = train_model(
        dataset,
        model,
        device,
        config_dict,
        silent=args.silent
    )
    loss_df = pd.DataFrame.from_dict(losses)
    loss_df.to_csv(os.path.join(full_path, "losses.csv"), sep=";", index=None)

    # full_path = "logs/double_torus"
    mesh_file = parameter_dict["reconstruction"]["output_file"]
    mesh_resolution = parameter_dict["reconstruction"]["resolution"]
    decoder = SDFDecoder(
        torch.load(os.path.join(full_path, "model_final.pth")),
        n_in_features=3,
        n_out_features=1,
        hidden_layer_config=parameter_dict["network"]["hidden_layer_nodes"],
        w0=parameter_dict["network"]["w0"]
    )
    create_mesh(decoder, os.path.join(full_path, mesh_file), N=mesh_resolution)
