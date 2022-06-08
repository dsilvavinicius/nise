# coding: utf-8

import argparse
import json
import os
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import BatchSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import PointCloud, SpaceTimePointCloud, SpaceTimePointCloudNI
from model import SIREN
from samplers import SitzmannSampler
from loss import loss_mean_curv, sdf_sitzmann, true_sdf_off_surface, sdf_sitzmann_time, sdf_time, sdf_boundary_problem, loss_eikonal, loss_eikonal_mean_curv, loss_constant, loss_transport, loss_vector_field_morph
from meshing import create_mesh
from util import create_output_paths, load_experiment_parameters


def train_model(dataset, model, device, train_config, space_time=False, silent=False):
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
            # if space_time:
                # coords_time0 = data["coords"][data["coords"][..., 3] == 0]
                # colors = torch.zeros_like(coords_time0[..., 0:3], device="cpu", requires_grad=False)
                # sdf_time0 = data["sdf"][data["coords"][..., 3] == 0]
                # inputs = coords_time0[..., 0:3].to(device)

                # #at time zero
                # colors[sdf_time0.squeeze(-1) < 0, :] = torch.Tensor([255, 0, 0])
                # colors[sdf_time0.squeeze(-1) == 0, :] = torch.Tensor([0, 255, 0])
                # colors[sdf_time0.squeeze(-1)  > 0, :] = torch.Tensor([0, 0, 255])

                # writer.add_mesh(
                #     "input", inputs.unsqueeze(-1), colors=colors.unsqueeze(-1), global_step=epoch
                # )
            if not space_time:
                colors = torch.zeros_like(data["coords"], device="cpu", requires_grad=False)
            
                #at time zero
                colors[data["sdf"].squeeze(-1) < 0, :] = torch.Tensor([255, 0, 0])
                colors[data["sdf"].squeeze(-1) == 0, :] = torch.Tensor([0, 255, 0])
                colors[data["sdf"].squeeze(-1)  > 0, :] = torch.Tensor([0, 0, 255])

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
            
            if space_time:
                N = 5    # number of samples of the interval time
                for i in range(N):
                    T = -1+2*(i/(N-1))
                    mesh_file = f"epoch_{epoch}_time_{T}.ply"
                    verts, _, normals, _ = create_mesh(
                        model,
                        filename=os.path.join(full_path, "reconstructions", mesh_file), 
                        t=T,  # time instant for 4d SIREN function
                        N=mesh_resolution,
                        device=device
                    )
            else:
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
    
    #experiment_path = "experiments/double_torus_toy.json"
    #experiment_path = "experiments/armadillo.json"
    #experiment_path = "experiments/cube_time_0.json"
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

    space_time = parameter_dict.get("space_time")  # consider the spacetime (x,y,z,t) as domain
    n_in_features = 3  # implicit 3D models
    if space_time:
        n_in_features = 4  # used to animate implicit 3D models

    # Saving the parameters to the output path
    with open(os.path.join(full_path, "params.json"), "w+") as fout:
        json.dump(parameter_dict, fout, indent=4)

    no_sampler = True
    if sampling_config.get("sampler"):
        no_sampler = False

    off_surface_sdf = parameter_dict.get("off_surface_sdf")
    off_surface_normals = parameter_dict.get("off_surface_normals")
    if off_surface_normals is not None:
        off_surface_normals = np.array(off_surface_normals)

    scaling = parameter_dict.get("scaling")

    dataset = None
    datasets = parameter_dict["dataset"]
    for d in datasets:
        d[0] = os.path.join("data", d[0])

    if len(datasets[0]) == 3:
        pretrained_ni = SIREN(3, 1, [128, 128, 128], w0=30)
        pretrained_ni.load_state_dict(torch.load(datasets[0][1]))
        pretrained_ni.eval()
        pretrained_ni.to(device) 
        datasets[0] = [datasets[0][0], datasets[0][2]]

    dataset = SpaceTimePointCloudNI(
        datasets,
        sampling_config["samples_on_surface"],
        pretrained_ni=[pretrained_ni],
        batch_size=parameter_dict["batch_size"],
        silent=False,
        device=device
    )

    sampler = None
    sampler_opt = sampling_config.get("sampler")
    if sampler_opt is not None and sampler_opt == "sitzmann":
        sampler = SitzmannSampler(
            dataset,
            sampling_config["samples_off_surface"]
        )

    hidden_layers = parameter_dict["network"]["hidden_layer_nodes"]
    model = SIREN(
        n_in_features,
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

    loss = parameter_dict.get("loss")
    if loss is not None and loss:
        if loss == "sitzmann":
            if space_time:
                loss_fn = sdf_sitzmann_time   
            else:    
                loss_fn = sdf_sitzmann
        elif loss == "true_sdf":  
            if space_time:
                loss_fn = sdf_time
            else:    
                loss_fn = true_sdf_off_surface
        elif loss == "sdf_boundary_problem":
            loss_fn = sdf_boundary_problem
        elif loss == "loss_mean_curv":
            loss_fn = loss_mean_curv
        elif loss == "loss_eikonal":
            loss_fn = loss_eikonal
        elif loss == "loss_eikonal_mean_curv":
            loss_fn = loss_eikonal_mean_curv
        elif loss == "loss_constant":
            loss_fn = loss_constant
        elif loss == "loss_transport":
            loss_fn = loss_transport
        elif loss == "loss_vector_field_morph":
            loss_fn = loss_vector_field_morph
        else:
            warnings.warn(f"Invalid loss function option {loss}. Using default.")

    config_dict = {
        "epochs": parameter_dict["num_epochs"],
        "batch_size": parameter_dict["batch_size"],
        "epochs_to_checkpoint": parameter_dict["epochs_to_checkpoint"],
        "epochs_to_reconstruct": parameter_dict["epochs_to_reconstruction"],
        "sampler": sampler,
        "log_path": full_path,
        "optimizer": optimizer,
        "loss_fn": loss_fn,
        "mc_resolution": parameter_dict["reconstruction"]["resolution"]
    }

    losses = train_model(
        dataset,
        model,
        device,
        config_dict,
        space_time,
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
    
    if space_time:
        create_mesh(
            model,
            os.path.join(full_path, "reconstructions", mesh_file),
            0,  # time instant for 4d SIREN function
            N=mesh_resolution,
            device=device
        )
    else: 
        create_mesh(
            model,
            os.path.join(full_path, "reconstructions", mesh_file),
            N=mesh_resolution,
            device=device
        )
