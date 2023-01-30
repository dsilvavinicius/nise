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
from dataset import SpaceTimePointCloud, SpaceTimePointCloudNI
from model import SIREN
from samplers import SitzmannSampler
from loss import loss_level_set, loss_mean_curv_with_restrictions, loss_morphing_two_sirens, loss_GPNF, loss_mean_curv, sdf_sitzmann, true_sdf_off_surface, sdf_sitzmann_time, sdf_time, sdf_boundary_problem, loss_eikonal, loss_eikonal_mean_curv, loss_constant, loss_transport, loss_vector_field_morph
from meshing import create_mesh
from util import create_output_paths, load_experiment_parameters

import kaolin

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
            
            N = 6    # number of samples of the interval time
            #T= -0.1#-0.75
            for i in range(N):
                T = (-1 + 2*(i/(N-1)))*0.2
                #T = (-1 + 2*(i/(N-1)))
                mesh_file = f"epoch_{epoch}_time_{T}.ply"
                verts, faces, normals, _ = create_mesh(
                    model,
                    filename=os.path.join(full_path, "reconstructions", mesh_file), 
                    t=T,  # time instant for 4d SIREN function
                    N=mesh_resolution,
                    device=device
                )
                #T += 0.1

                #adding checkpoint to kaolin
                tensor_faces = torch.from_numpy(faces.copy())
                tensor_verts = torch.from_numpy(verts.copy())
                timelapse.add_mesh_batch(category=f"output_{i}", iteration=epoch/EPOCHS_TIL_RECONSTRUCTION, faces_list=[tensor_faces], vertices_list=[tensor_verts])

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


    timelapse = kaolin.visualize.Timelapse(os.path.join(full_path, "kaolin"))


    dataset = None
    datasets = parameter_dict["dataset"]
    for d in datasets:
        d[0] = os.path.join("data", d[0])

    # if len(datasets[0]) == 3:
    #     #pretrained_ni = SIREN(3, 1, [64, 64], w0=16)#for neural spot
    #     # pretrained_ni = SIREN(3, 1, [128,128,128], w0=30)#for neural spot
    #     # pretrained_ni = SIREN(3, 1, [128,128], w0=24)
    #     # pretrained_ni = SIREN(3, 1, [64,64], w0=16)
    #     pretrained_ni = SIREN(3, 1, [256, 256, 256], w0=30)
    #     # pretrained_ni = SIREN(3, 1, [64, 64], w0=16)
    #     pretrained_ni.load_state_dict(torch.load(datasets[0][1]))
    #     pretrained_ni.eval()
    #     pretrained_ni.to(device) 
    #     datasets[0] = [datasets[0][0], datasets[0][2]]

    # TODO: think in how to consider multiples trained sirens
    # pretrained_ni1 = SIREN(3, 1, [64, 64], w0=16)
    pretrained_ni1 = SIREN(3, 1, [128,128,128], w0=20)
    #pretrained_ni1 = SIREN(3, 1, [64,64], w0=16)
    #pretrained_ni1.load_state_dict(torch.load('shapeNets/spot_1x64_w0-16.pth'))
    #pretrained_ni1.load_state_dict(torch.load('shapeNets/torus_1x64_w0-16.pth'))
    # pretrained_ni1.load_state_dict(torch.load('shapeNets/fantasma_1x64_w0-16.pth'))
    pretrained_ni1.load_state_dict(torch.load('shapeNets/falcon_smooth_2x128_w0-20.pth'))
    pretrained_ni1.eval()
    pretrained_ni1.to(device) 

    # # pretrained_ni2 = SIREN(3, 1, [128,128], w0=20)
    pretrained_ni2 = SIREN(3, 1, [128,128,128], w0=30)
    #pretrained_ni2 = SIREN(3, 1, [128,128], w0=20)
    
    # pretrained_ni2 = SIREN(3, 1, [64,64], w0=16)
    #pretrained_ni2.load_state_dict(torch.load('shapeNets/bob_1x64_w0-16.pth'))
    # pretrained_ni2.load_state_dict(torch.load('shapeNets/bitorus_1x64_w0-16.pth'))
    # pretrained_ni2.load_state_dict(torch.load('shapeNets/blub_1x64_w0-16.pth'))
    # pretrained_ni2.load_state_dict(torch.load('shapeNets/pig_1x128_w0-20.pth'))
    # pretrained_ni2.load_state_dict(torch.load('shapeNets/skull_1x128_w0-20.pth'))
    pretrained_ni2.load_state_dict(torch.load('shapeNets/witch_2x128_w0-30.pth'))
    pretrained_ni2.eval()
    pretrained_ni2.to(device)

    dataset = SpaceTimePointCloudNI(
        datasets,
        sampling_config["samples_on_surface"],
        pretrained_ni=[pretrained_ni1, pretrained_ni2],
        # pretrained_ni=[pretrained_ni],
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
    use_trained_i4d_weights = False
    if use_trained_i4d_weights:
        # trained_i4d_weights = torch.load("logs/armadillo_2x256_w-30_twist_t=0_0.5/models/model_1000.pth")
        trained_i4d_weights = torch.load("logs/falcon_witch_96x1_w0_20_t_-0.2_0.2/models/model_1000.pth")
        model.load_state_dict(trained_i4d_weights)
        model.to(device=device)

    #use the weights of a trained i3d net
    use_trained_i3d_weights = False
    if use_trained_i3d_weights:
        #layer_0 = model.net[0][0].weight[...,3].unsqueeze(-1)
        # i3d_weights = torch.load("shapeNets/dragon_2x256_w-60.pth")
        #i3d_weights = torch.load("shapeNets/armadillo_2x256_w-60.pth")
        i3d_weights = torch.load("shapeNets/bunny_2x256_w-30.pth")
        #i3d_weights = torch.load("shapeNets/witch_2x128_w0-30.pth")
        # i3d_weights = torch.load("shapeNets/falcon_2x128_w0-30.pth")
        #i3d_weights = torch.load("shapeNets/torus_1x64_w0-16.pth")
        #i3d_weights = torch.load("shapeNets/spot_1x64_w0-16.pth")
        first_layer = i3d_weights['net.0.0.weight']
        new_first_layer = torch.cat((first_layer,torch.zeros_like(first_layer[...,0].unsqueeze(-1))), dim=-1) #initialize with zeros
        #new_first_layer = torch.cat((first_layer, layer_0), dim=-1) #initialize using siren scheme
        i3d_weights['net.0.0.weight'] = new_first_layer
        model.load_state_dict(i3d_weights)
        model.to(device=device)

    if not args.silent:
        print(model)

    #zero checkpoint
    # N = 7    # number of samples of the interval time
    # for i in range(N):
    #     T = (-1 + 2*(i/(N-1)))
    #     mesh_file = f"epoch_{0}_time_{T}.ply"
    #     verts, faces, normals, _ = create_mesh(
    #         model,
    #         filename=os.path.join(full_path, "reconstructions", mesh_file), 
    #         t=T,  # time instant for 4d SIREN function
    #         N= parameter_dict["reconstruction"]["resolution"],
    #         device=device
    #     )

    #     #adding checkpoint to kaolin
    #     tensor_faces = torch.from_numpy(faces.copy())
    #     tensor_verts = torch.from_numpy(verts.copy())
    #     timelapse.add_mesh_batch(category=f"output_{i}", iteration=0, faces_list=[tensor_faces], vertices_list=[tensor_verts])


    opt_params = parameter_dict["optimizer"]
    if opt_params["type"] == "adam":
        optimizer = torch.optim.Adam(
            lr=opt_params["lr"],
            params=model.parameters()
        )

    loss = parameter_dict.get("loss")
    if loss is not None and loss:
        if loss == "sitzmann":
            loss_fn = sdf_sitzmann_time   
        elif loss == "true_sdf":  
            loss_fn = sdf_time
        elif loss == "sdf_boundary_problem":
            loss_fn = sdf_boundary_problem
        elif loss == "loss_mean_curv":
            loss_fn = loss_mean_curv
        elif loss == "loss_mean_curv_with_restrictions":
            loss_fn = loss_mean_curv_with_restrictions
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
        elif loss == "loss_GPNF":
            loss_fn = loss_GPNF(pretrained_ni)
        elif loss == "loss_level_set":
            loss_fn = loss_level_set(pretrained_ni)
        elif loss == "loss_morphing_two_sirens":
            loss_fn = loss_morphing_two_sirens(pretrained_ni1, pretrained_ni2)
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
        silent=args.silent
    )
