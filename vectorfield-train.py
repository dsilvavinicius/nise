#!/usr/bin/env python
# coding: utf-8

import argparse
import copy
import math
import os
import os.path as osp
import time
import sys
import kaolin
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import yaml
from i4d.dataset import SpaceTimePointCloudNI
from i4d.loss import loss_level_set
from i4d.meshing import create_mesh, save_ply
from i4d.model import SIREN, from_pth
from i4d.util import create_output_paths


if __name__ == '__main__':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser(
        description="Default training script when using Neural Implicits for"
        " SDF querying and mean curvature experiments. Note that command line"
        " arguments have precedence over configuration file values."
    )
    parser.add_argument(
        "experiment_config", type=str, help="Path to the YAML experiment"
        " configuration file."
    )
    parser.add_argument(
        "--seed", default=668123, type=int,
        help="Seed for the random-number generator."
    )
    parser.add_argument(
        "--device", "-d", default="cuda:0", help="Device to run the training."
    )
    parser.add_argument(
        "--batchsize", "-b", default=0, type=int,
        help="Number of points to use per step of training. If set to 0,"
        " fetches it from the configuration file."
    )
    parser.add_argument(
        "--epochs", "-e", default=0, type=int,
        help="Number of epochs of training to perform. If set to 0, fetches it"
        " from the configuration file."
    )
    parser.add_argument(
        "--time_benchmark", "-t", action="store_true", help="Indicates that we"
        " are running a training time measurement. Disables writing to"
        " tensorboard, model checkpoints, best model serialization and mesh"
        " generation during training."
    )
    parser.add_argument(
        "--kaolin", action="store_true", default=False, help="When saving"
        " mesh checkpoints, use kaolin format, or simply save the PLY files"
        " (default). Note that this respects the checkpoint configuration in"
        " the experiment files, if no checkpoints are enabled, then nothing"
        " will be saved."
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    with open(args.experiment_config, 'r') as f:
        config = yaml.safe_load(f)

    devstr = args.device
    if "cuda" in args.device and not torch.cuda.is_available():
        print(f"[WARNING] Selected device {args.device}, but CUDA is not"
              " available. Using CPU", file=sys.stderr)
        devstr = "cpu"
    device = torch.device(devstr)

    training_config = config["training"]
    training_data_config = config["training_data"]
    training_mesh_config = training_data_config["mesh"]

    # Just one mesh for mean-curvature-equation cases
    mesh = list(training_mesh_config.keys())[0]
    ni = training_mesh_config[mesh]["ni"]
    w0 = training_mesh_config[mesh].get("omega_0", 1)

    epochs = training_config.get("n_epochs", 100)
    if args.epochs:
        epochs = args.epochs

    batchsize = training_data_config.get("batchsize", 20000)
    if args.batchsize:
        batchsize = args.batchsize

    dataset = SpaceTimePointCloudNI(
        [(mesh, ni, training_mesh_config[mesh]['t'], w0)],
        batchsize
    )

    nsteps = round(epochs * (4 * len(dataset) / batchsize))
    WARMUP_STEPS = nsteps // 10
    checkpoint_at = training_config.get("checkpoints_at_every_epoch", 0)
    if checkpoint_at:
        checkpoint_at = round(checkpoint_at * (4 * len(dataset) / batchsize))
        print(f"Checkpoints at every {checkpoint_at} training steps")
    else:
        print("Checkpoints disabled")

    print(f"Total # of training steps = {nsteps}")

    network_config = config["network"]
    model = SIREN(4, 1, network_config["hidden_layer_nodes"],
                  w0=network_config["omega_0"], delay_init=False).to(device)
    print(model)

    experiment = osp.split(args.experiment_config)[-1].split('.')[0]
    experimentpath = create_output_paths(
        "results",
        experiment,
        overwrite=False
    )

    writer = SummaryWriter(osp.join(experimentpath, 'summaries'))

    model.zero_grad(set_to_none=True)
    # w0 = model.w0
    # model.update_omegas(1)
    #model.from_pretrained_initial_condition(torch.load(ni))
    # model.update_omegas(w0)

    if "timesampler" in training_data_config:
        timerange = training_data_config["timesampler"].get("range", [-1.0, 1.0])
        dataset.time_sampler = torch.distributions.uniform.Uniform(
            timerange[0], timerange[1]
        )

    optim = torch.optim.Adam(
        lr=1e-4,
        params=model.parameters()
    )

    trainingpts = torch.zeros((batchsize, 4), device=device)
    trainingnormals = torch.zeros((batchsize, 3), device=device)
    trainingsdf = torch.zeros((batchsize), device=device)

    n_on_surface = training_data_config.get(
        "n_on_surface", math.floor(batchsize * 0.25)
    )
    n_off_surface = training_data_config.get(
        "n_off_surface", math.ceil(batchsize * 0.25)
    )
    n_int_times = training_data_config.get(
        "n_int_times", batchsize - (n_on_surface + n_off_surface)
    )

    scale = float(config["loss"].get("scale", 1e-3))
    lossvf = loss_level_set(from_pth(ni, w0=training_mesh_config[mesh].get("omega_0", 1)))

    checkpoint_times = config["training"].get(
        "checkpoint_times", [-1.0, 0.0, 1.0]
    )

    updated_config = copy.deepcopy(config)
    updated_config["training"]["n_epochs"] = epochs
    updated_config["training_data"]["batchsize"] = batchsize
    updated_config["training_data"]["n_on_surface"] = n_on_surface
    updated_config["training_data"]["n_off_surface"] = n_off_surface
    updated_config["training_data"]["n_int_times"] = n_int_times

    with open(osp.join(experimentpath, "config.yaml"), 'w') as f:
        yaml.dump(updated_config, f)

    best_loss = torch.inf
    best_weigths = None
    omegas = dict()  # {3: 10}  # Setting the omega_0 value of t (coord. 3) to 10
    training_loss = {}

    # Reconstruct without training
    # meshpath = osp.join(
    #     experimentpath, "reconstructions", "check_0"
    # )
    # os.makedirs(meshpath, exist_ok=True)
    # reconstruct_with_curvatures(
    #     model, checkpoint_times, meshpath, device=device,
    #     resolution=256
    # )
    # model = model.train()

    if args.kaolin:
        timelapse = kaolin.visualize.Timelapse(
            osp.join(experimentpath, "kaolin")
        )

    start_training_time = time.time()
    for e in range(nsteps):
        data = dataset[e]
        # ===============================================================
        trainingpts[:n_on_surface, ...] = data["on_surf"][0]
        trainingnormals[:n_on_surface, ...] = data["on_surf"][1]
        trainingsdf[:n_on_surface] = data["on_surf"][2]

        trainingpts[n_on_surface:(n_on_surface + n_off_surface), ...] = data["off_surf"][0]
        trainingnormals[n_on_surface:(n_on_surface + n_off_surface), ...] = data["off_surf"][1]
        trainingsdf[n_on_surface:(n_on_surface + n_off_surface), ...] = data["off_surf"][2].squeeze()

        trainingpts[(n_on_surface + n_off_surface):, ...] = data["int_times"][0]
        trainingnormals[(n_on_surface + n_off_surface):, ...] = data["int_times"][1]
        trainingsdf[(n_on_surface + n_off_surface):, ...] = data["int_times"][2]

        gt = {
            "sdf": trainingsdf.float().unsqueeze(1),
            "normals": trainingnormals.float(),
        }

        optim.zero_grad(set_to_none=True)
        y = model(trainingpts, omegas=omegas)
        loss = lossvf(y, gt)

        running_loss = torch.zeros((1, 1), device=device)
        for k, v in loss.items():
            running_loss += v
            if not args.time_benchmark:
                writer.add_scalar(f"train/{k}_term", v.detach().item(), e)
            if k not in training_loss:
                training_loss[k] = [v.detach().item()]
            else:
                training_loss[k].append(v.detach().item())

        running_loss.backward()
        optim.step()
        if not args.time_benchmark:
            writer.add_scalar("train/loss", running_loss.detach().item(), e)

            if e > WARMUP_STEPS and best_loss > running_loss.item():
                best_weights = copy.deepcopy(model.state_dict())
                best_loss = running_loss.item()

            if checkpoint_at and e and not e % checkpoint_at:
                for i, t in enumerate(checkpoint_times):
                    verts, faces, normals, _ = create_mesh(
                        model,
                        t=t,
                        N=256,
                        device=device
                    )
                    if args.kaolin:
                        timelapse.add_mesh_batch(
                            category=f"check_{i}",
                            iteration=e // checkpoint_at,
                            faces_list=[torch.from_numpy(faces.copy())],
                            vertices_list=[torch.from_numpy(verts.copy())]
                        )
                    else:
                        meshpath = osp.join(
                            experimentpath, "reconstructions", f"check_{e}"
                        )
                        os.makedirs(meshpath, exist_ok=True)
                        save_ply(
                            verts, faces, osp.join(meshpath, f"time_{t}.ply")
                        )

                model = model.train()

            if not e % 100 and e > 0:
                print(f"Step {e} --- Loss {running_loss.item()}")

    training_time = time.time() - start_training_time
    print(f"training took {training_time} s")
    writer.flush()
    writer.close()

    torch.save(
        model.state_dict(), osp.join(experimentpath, "models", "weights.pth")
    )
    model.load_state_dict(best_weights)
    model.update_omegas(w0=1)
    torch.save(
        model.state_dict(), osp.join(experimentpath, "models", "best.pth")
    )
