# coding: utf-8

import copy
import os
import os.path as osp
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from dataset import SpaceTimePointCloud, SpaceTimePointCloudNI
from loss import loss_mean_curv
from meshing import create_mesh
from model import SIREN
from util import create_output_paths, load_experiment_parameters


def reconstruct_at_times(model, times, meshpath, resolution=256, device="cpu"):
    """Runs marching cubes on `model` at times `times`.

    Parameters
    ----------
    model: torch.nn.Module
        The model to run the inference. Must accept \mathbb{R}^4 inputs.

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXPERIMENT = "smooth_noisy_bunny"
EPOCHS = 10000
NSAMPLES = 15000
WARMUP = 100
WEIGHTSPATH = osp.join("logs", EXPERIMENT, "models", "weights.pth")

experimentpath = create_output_paths(
    "logs",
    EXPERIMENT,
    overwrite=False
)

summarypath = osp.join(experimentpath, 'summaries')
if not osp.exists(summarypath):
    os.makedirs(summarypath)
writer = SummaryWriter(summarypath)

dataset = SpaceTimePointCloud(
    [("data/bunny_noisy.ply", 0.0)],
    NSAMPLES,
    timerange=[-0.2, 0.2]
)

# bunny_ni = SIREN(3, 1, [256] * 3, w0=30).eval().to(device)
# bunny_ni.load_state_dict(torch.load("ni/bunny_2x256_w-30.pth"))

# dataset = SpaceTimePointCloudNI(
#     [("data/bunny.ply", 0.0)],
#     samples_on_surface=NSAMPLES,
#     pretrained_ni=[bunny_ni],
#     timerange=[-0.2, 0.2],
#     device=device
# )

model = SIREN(4, 1, [256] * 3, w0=30).to(device)
print(model)

optimizer = torch.optim.Adam(
     lr=1e-4,
    params=model.parameters()
)

best_loss = np.inf
best_epoch = 0
best_weights = None
for e in range(EPOCHS):
    data = dataset[0]
    inputs = data["coords"].to(device)
    gt = {
        "sdf": data["sdf"].to(device),
        "normals": data["normals"].to(device)
    }

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = loss_mean_curv(outputs, gt)

    train_loss = torch.zeros((1, 1), device=device)
    for it, l in loss.items():
        writer.add_scalar(f"train/loss_{it}", l.detach().item(), e)
        train_loss += l

    print(f"Epoch {e} -- Loss {train_loss.item()}")
    t = train_loss.detach().item()
    writer.add_scalar("train/total_loss", t, e)
    if e > WARMUP and best_loss > t:
        best_loss = t
        best_epoch = e
        best_weights = copy.deepcopy(model.state_dict())

    train_loss.backward()
    optimizer.step()

torch.save(best_weights, osp.join(experimentpath, "models", "weights.pth"))

model.load_state_dict(best_weights)
times = [-0.2, -0.1, 0.0, 0.01, 0.05, 0.1, 0.2]
meshpath = osp.join(experimentpath, "reconstructions")
reconstruct_at_times(model, times, meshpath, device=device)
