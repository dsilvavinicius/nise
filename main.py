# coding: utf-8

import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import BatchSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import PointCloud
from model import SIREN
from samplers import SitzmannSampler
from loss import sdf_sitzmann
from meshing import create_mesh
from util import create_output_paths, load_experiment_parameters


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
            batch_sampler=BatchSampler(sampler, batch_size=BATCH_SIZE, drop_last=False),
            pin_memory=True,
            num_workers=0
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

    # Creating the summary storage folder
    summary_path = os.path.join(full_path, 'summaries')
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)
    writer = SummaryWriter(summary_path)

    # Adding the input point cloud to the summary
    # As it turns out, this increases the summary size to a point that the
    # browser hangs.
    # writer.add_mesh(
    #     tag="input_point_cloud",
    #     vertices=torch.from_numpy(dataset.point_cloud.points).unsqueeze(0),
    #     colors=torch.from_numpy(dataset.point_cloud.normals).unsqueeze(0)
    # )

    losses = dict()
    for epoch in range(EPOCHS):
        running_loss = dict()
        for i, data in enumerate(train_loader, start=0):
            # If we have a custom sampler, we must reshape the Tensors from
            # [B, N, D] to [1, B*N, D]
            if sampler is not None:
                for k, v in data.items():
                    b, n, d = v.size()
                    data[k] = v.reshape(1, -1, d)

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

            # Adding an iteration of the training data to tensorboard
            colors = torch.zeros_like(data["coords"], device="cpu", requires_grad=False)
            colors[data["sdf"].squeeze(-1) < 0, :] = torch.Tensor([255, 0, 0])
            colors[data["sdf"].squeeze(-1) == 0, :] = torch.Tensor([0, 255, 0])
            colors[data["sdf"].squeeze(-1) > 0, :] = torch.Tensor([0, 0, 255])
            writer.add_mesh(
                "input", inputs, colors=colors, global_step=epoch
            )
            writer.add_scalar("train_loss", train_loss.item(), epoch)

            train_loss.backward()
            optim.step()

        # accumulate statistics
        for it, l in running_loss.items():
            if it in losses:
                losses[it][epoch] = l
            else:
                losses[it] = [0.] * EPOCHS
                losses[it][epoch] = l
            writer.add_scalar(it, l, epoch)

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
                os.path.join(full_path, "models", f"model_{epoch}.pth")
            )

        # reconstructing a mesh at checkpoints
        if epoch and EPOCHS_TIL_RECONSTRUCTION and not epoch % EPOCHS_TIL_RECONSTRUCTION:
            if not silent:
                print(f"Reconstructing mesh for epoch {epoch}")

            mesh_file = f"{epoch}.ply"
            mesh_resolution = train_config["mc_resolution"]
            verts, _, normals, _ = create_mesh(
                model,
                filename=os.path.join(full_path, "reconstructions", mesh_file),
                N=mesh_resolution,
                device=device
            )
            if normals.strides[1] < 0:
                normals = normals[:, ::-1]
            verts = torch.from_numpy(verts).unsqueeze(0)
            normals = torch.from_numpy(np.abs(normals)*255).unsqueeze(0)

            writer.add_mesh(
                "reconstructed_point_cloud",
                vertices=verts,
                colors=normals,
                global_step=epoch
            )

            model.train()

    writer.flush()
    writer.close()
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

    no_sampler = True
    if "sampler" in sampling_config and sampling_config["sampler"]:
        no_sampler = False

    dataset = PointCloud(
        os.path.join("data", parameter_dict["dataset"]),
        sampling_config["samples_on_surface"],
        scaling=None,
        off_surface_sdf=-1,
        off_surface_normals=np.array([-1, -1, -1]),
        random_surf_samples=sampling_config["random_surf_samples"],
        no_sampler=no_sampler,
        batch_size=parameter_dict["batch_size"],
        silent=False
    )

    sampler = None
    if "sampler" in sampling_config and sampling_config["sampler"] == "sitzmann":
        sampler = SitzmannSampler(
            dataset,
            sampling_config["samples_off_surface"]
        )

    hidden_layers = parameter_dict["network"]["hidden_layer_nodes"]
    model = SIREN(
        n_in_features=3,
        n_out_features=1,
        hidden_layer_config=parameter_dict["network"]["hidden_layer_nodes"],
        w0=parameter_dict["network"]["w0"]
    )
    if not args.silent:
        print(model)

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
        "epochs_to_reconstruct": parameter_dict["epochs_to_reconstruction"],
        "sampler": sampler,
        "log_path": full_path,
        "optimizer": optimizer,
        "loss_fn": sdf_sitzmann,
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

    # saving the final model
    torch.save(
        model.state_dict(),
        os.path.join(full_path, "models", "model_final.pth")
    )

    # reconstructing the final mesh
    mesh_file = parameter_dict["reconstruction"]["output_file"] + ".ply"
    mesh_resolution = parameter_dict["reconstruction"]["resolution"]
    create_mesh(
        model,
        os.path.join(full_path, "reconstructions", mesh_file),
        N=mesh_resolution,
        device=device
    )
