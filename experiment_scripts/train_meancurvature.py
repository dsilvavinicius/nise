#!/usr/bin/env python
# coding: utf-8

import argparse
import copy
import math
import os
import os.path as osp
import time
import sys
import numpy as np
import open3d as o3d
import open3d.core as o3c
from plyfile import PlyData
import torch
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import yaml
from i4d.diff_operators import gradient
from i4d.loss import LossMeanCurvature
from i4d.model import SIREN
from i4d.util import (create_output_paths, from_pth,
                      reconstruct_with_curvatures)


def sample_on_surface(vertices: torch.Tensor, n_points: int, device: str):
    """Samples row of a torch tensor containing vertices of a surface.

    Parameters
    ----------
    vertices: torch.Tensor
        A mode-2 tensor where each row is a vertex.

    n_points: int
        The number of points to sample. If `n_points` >= `vertices.shape[0]`,
        we simply return `vertices` without any change.

    device: str or torch.device
        The device where we should generate the indices of sampled points.
        Ideally, this is the same device where `vertices` is stored.

    Returns
    -------
    sampled: torch.tensor
        The points sampled from `vertices`. If
        `n_points` == `vertices.shape[0]`, then we simply return `vertices`.

    idx: torch.tensor
        The indices of points sampled from `vertices`. Naturally, these are
        row indices in `vertices`.

    See Also
    --------
    torch.randperm, torch.arange
    """
    if n_points >= vertices.shape[0]:
        return vertices, torch.arange(end=n_points, step=1, device=device)
    idx = torch.randperm(vertices.shape[0], device=device)[:n_points]
    sampled = vertices[idx, ...]
    return sampled, idx


def sample_initial_condition(
    vertices: torch.tensor,   # pass a list of tensors (vertices), one for each mesh
    n_on_surf: int,
    n_off_surf: int,
    ni: torch.nn.Module,      # Same here
    device: torch.device = torch.device("cpu"),
    no_sdf: bool = False,
):
    """Creates a set of training data with coordinates, normals and SDF
    values.

    Parameters
    ----------
    vertices: torch.tensor
        A mode-2 tensor with the mesh vertices.

    n_on_surf: int
        # of points to sample from the mesh.

    n_off_surf: int
        # of points to sample from the domain. Note that we sample points
        uniformely at random from the domain.

    ni: torch.nn.Module
        Neural Implicit Open3D raycasting scene to use when querying SDF
        for domain points.

    device: str or torch.device, optional
        The compute device where `vertices` is stored. By default its
        torch.device("cpu")

    no_sdf: boolean, optional
        Don't query SDF for domain points, instead we mark them with SDF = -1.

    Returns
    -------
    coords: dict[str => list[torch.Tensor]]
        A dictionary with points sampled from the surface (key = "on_surf")
        and the domain (key = "off_surf"). Each dictionary element is a list
        of tensors with the vertex coordinates as the first element of said
        list, the normals as the second element, finally, the SDF is the last
        element.

    See Also
    --------
    sample_on_surface, curvature_segmentation
    """
    surf_pts, _ = sample_on_surface(
        vertices,
        n_on_surf,
        device=device
    )

    coord_dict = {
        "on_surf": [surf_pts[..., :4],   # x, y, z, t
                    surf_pts[..., 4:7],  # nx, ny, nz
                    surf_pts[..., -1]]   # sdf
    }

    if n_off_surf != 0:
        domain_pts = torch.rand((n_off_surf, 3), device=device) * 2 - 1
        # domain_pts = off_surf_sampler.sample((n_off_surf, 3)).to(device)
        t = surf_pts[0, 3]  # We assume that all points in surf_pts have the same value of t.
        if no_sdf is False:
            out = ni(domain_pts)
            domain_sdf = out["model_out"]
            domain_normals = gradient(domain_sdf, out["model_in"]).detach()
            domain_sdf = domain_sdf.detach()
        else:
            domain_sdf = torch.full(
                (n_off_surf, 1),
                fill_value=-1,
                device=device
            )
            domain_normals = -torch.ones_like(domain_pts, device=device)

        coord_dict["off_surf"] = [
            torch.column_stack((
                domain_pts,
                torch.full_like(domain_pts, fill_value=t, device=device)[..., 0]
            )),  # x, y, z, t
            domain_normals,
            domain_sdf.squeeze()
        ]

    return coord_dict


def create_training_data(
    initial_conditions: list,
    n_samples: int,
    off_surface_sampler: torch.distributions.distribution.Distribution,
    device: torch.device,
    time_sampler: torch.distributions.distribution.Distribution=None,
    fraction_on_surface: float = 0.25,
    fraction_off_surface: float = 0.25
):
    """Samples a batch of training points.

    Parameters
    ----------
    initial_conditions: list[torch.Tensor, torch.nn.Module]
        A list with all initial conditions. Each initial condition contains
        a tensor with the mesh vertices and a neural implicit representation to
        estimate the SDF values.

    n_samples: int
        The total number of samples to draw.

    off_surface_sampler: torch.distributions.distribution.Distribution
        The distribution to draw samples for off-surface point coordinates at
        intermediate-times

    device: torch.device
        The device to store any tensors created.

    time_sampler: torch.distributions.distribution.Distribution
        The distribution to draw samples for off-surface parameter values at
        intermediate times. Default value is None, meaning that
        `off_surface_sample` will be used for this as well.

    fraction_on_surface: float
        Fraction of points to be drawn from the initial condition vertices.

    fraction_off_surface: float
        Fraction of points to be drawn off-surface from the initial conditions.
        intermediate time points as well.

    Returns
    -------
    full_samples: dict[str => list[torch.Tensor]]
        A dictionary with three keys: "on_surf", "off_surf", "int_times". Each
        element is an ordered list of tensors, where the first element is the
        point coordinates, followed by the normals and, the SDF values.
    """
    n_on_surface = math.ceil(n_samples * fraction_on_surface)
    n_off_surface = math.floor(n_samples * fraction_off_surface)
    n_int_times = n_samples - (n_on_surface + n_off_surface)

    if len(initial_conditions) > 1:
        n_on_surface = n_on_surface // len(initial_conditions)
        n_off_surface = n_off_surface // len(initial_conditions)

    full_samples = []

    for vertices, ni in initial_conditions:
        samples = sample_initial_condition(
            vertices,
            n_on_surf=n_on_surface,
            n_off_surf=n_off_surface,
            ni=ni,
            device=device
        )
        if not full_samples:
            full_samples = samples
        else:
            for k in full_samples:
                for i in len(full_samples[k]):
                    full_samples[k][i] = torch.cat((
                        full_samples[k][i], samples[k][i]
                    ), dim=-1)

    if n_int_times:
        int_pts = None
        if time_sampler is None:
            int_pts = off_surface_sampler.sample((n_int_times, 4)).to(device)
        else:
            int_pts = off_surface_sampler.sample((n_int_times, 3)).to(device)
            times = time_sampler.sample((n_int_times,)).to(device)
            int_pts = torch.column_stack((int_pts, times))

        full_samples["int_times"] = [
            int_pts,
            -torch.ones((n_int_times, 3), dtype=torch.float32, device=device),
            -torch.ones((n_int_times,), dtype=torch.float32, device=device)
        ]
    else:
        full_samples["int_times"] = []

    return full_samples


def read_ply(path: str, t: float):
    """Reads a PLY file with position and normal data.

    Note that we expect the input ply to contain x,y,z vertex data, as well
    as nx,ny,nz normal data.

    Parameters
    ----------
    path: str, PathLike
        Path to the ply file. We except the file to be in binary format.

    t: number
        The parameter value for this mesh.

    Returns
    -------
    mesh: o3d.t.geometry.TriangleMesh
        The fully constructed Open3D Triangle Mesh. By default, the mesh is
        allocated on the CPU:0 device, since Open3D still doesn't support GPU
        nearest-neighbor operations.

    vertices: torch.tensor
        The same vertex information as stored in `mesh`, augmented by the SDF
        values as the last column (a column of zeroes). Returned for easier,
        structured access.

    See Also
    --------
    PlyData.read, o3d.t.geometry.TriangleMesh
    """
    # Reading the PLY file and adding the time info
    n_columns = 8  # x, y, z, t, nx, ny, nz, sdf
    with open(path, "rb") as f:
        plydata = PlyData.read(f)
        num_verts = plydata["vertex"].count
        vertices = np.zeros(shape=(num_verts, n_columns), dtype=np.float32)
        vertices[:, 0] = plydata["vertex"].data["x"]
        vertices[:, 1] = plydata["vertex"].data["y"]
        vertices[:, 2] = plydata["vertex"].data["z"]
        # column 3 is time
        if t != 0:
            vertices[:, 3] = t
        vertices[:, 4] = plydata["vertex"].data["nx"]
        vertices[:, 5] = plydata["vertex"].data["ny"]
        vertices[:, 6] = plydata["vertex"].data["nz"]

        faces = np.stack(plydata["face"].data["vertex_indices"])

    # Converting the PLY data to open3d format
    device = o3c.Device("CPU:0")
    mesh = o3d.t.geometry.TriangleMesh(device)
    mesh.vertex["positions"] = o3c.Tensor(vertices[:, :3], dtype=o3c.float32)
    mesh.vertex["normals"] = o3c.Tensor(vertices[:, 3:6], dtype=o3c.float32)
    mesh.triangle["indices"] = o3c.Tensor(faces, dtype=o3c.int32)

    return mesh, torch.from_numpy(vertices).requires_grad_(False)


class STPointCloudNI(Dataset):
    """Space-time varying point clouds with NI for SDF querying.

    Parameters
    ----------
    inputpaths: list[(str, str, number, number)]
        List of tuples with paths to the base meshes (PLY format only), their
        neural implicit (NI) representations, the parameter value
        (-1 <= t <= 1) for each mesh, and omega_0 value for the NI.

    batchsize: int
        # of points to sample at each call to `__getitem__`.

    device: torch.device
        Device to store the NIs and vertex data read. By default, we store
        them on "cuda:0"

    fraction_on_surface: number, optional
        Fraction of points to sample from the initial conditions' surface per
        each batch. By default we sample 1/4 of points from the meshes at each
        call to `__getitem__`

    fraction_off_surface: number, optional
        Fraction of points to sample from the initial conditions' domain, i.e.
        off-surface points per batch. By default we sample 1/4 of points from
        the meshes' domains at each call to `__getitem__`
    """
    def __init__(self, inputpaths, batchsize, device=torch.device("cuda:0"),
                 fraction_on_surface=0.25, fraction_off_surface=0.25):
        super().__init__()
        self.batchsize = batchsize
        self.fraction_on_surface = fraction_on_surface
        self.fraction_off_surface = fraction_off_surface
        self.device = device

        vertices = [None] * len(inputpaths)
        ni = [None] * len(inputpaths)
        for i, (meshpath, nipath, paramval, w0) in enumerate(inputpaths):
            _, verts = read_ply(meshpath, paramval)
            vertices[i] = verts.to(device)
            ni[i] = from_pth(nipath, device=device, w0=w0).to(device)

        self.vertices_ni = list(zip(vertices, ni))

        self.off_surf_sampler = torch.distributions.uniform.Uniform(-1, 1)
        self.time_sampler = torch.distributions.uniform.Uniform(-0.2, 1.0)

    def __len__(self):
        return sum([m.shape[0] for m, _ in self.vertices_ni])

    def __getitem__(self, n):
        return create_training_data(
            self.vertices_ni,
            n_samples=self.batchsize,
            fraction_on_surface=self.fraction_on_surface,
            fraction_off_surface=self.fraction_off_surface,
            off_surface_sampler=self.off_surf_sampler,
            time_sampler=self.time_sampler,
            device=self.device
        )


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
        "--init_method", "-i", default="",
        help="Initialization method. Either standard (\"sitz\") or ours"
        " (\"i3d\"). If left empty, fetches it from the configuration file."
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

    MESH = list(training_mesh_config.keys())[0]
    NI = training_mesh_config[MESH]["ni"]
    W0 = training_mesh_config[MESH].get("omega_0", 1)

    EPOCHS = training_config.get("n_epochs", 100)
    if args.epochs:
        EPOCHS = args.epochs

    BATCHSIZE = training_data_config["batchsize"]
    dataset = STPointCloudNI(
        [(MESH, NI, training_mesh_config[MESH]['t'], W0)],
        BATCHSIZE
    )

    nsteps = round(EPOCHS * (2 * len(dataset) / BATCHSIZE))
    WARMUP_STEPS = nsteps // 10
    CHECKPOINT_AT = 0
    print(f"Total # of training steps = {nsteps}")

    network_config = config["network"]
    model = SIREN(4, 1, network_config["hidden_layer_nodes"],
                  w0=network_config["omega_0"], delay_init=True).to(device)
    print(model)

    experiment = osp.split(args.experiment_config)[-1].split('.')[0]
    experimentpath = create_output_paths(
        "results",
        experiment,
        overwrite=False
    )

    writer = SummaryWriter(osp.join(experimentpath, 'summaries'))

    model.zero_grad(set_to_none=True)
    model.reset_weights()

    init_method = network_config.get("init_method", args.init_method)
    if init_method == "i3d":
        model.from_pretrained_initial_condition(torch.load(NI))

    if "timesampler" in training_mesh_config:
        timerange = training_mesh_config["timesampler"].get("range", [-1.0, 1.0])
        dataset.time_sampler = torch.distributions.uniform.Uniform(
            timerange[0], timerange[1]
        )

    optim = torch.optim.Adam(
        lr=1e-4,
        params=model.parameters()
    )

    trainingpts = torch.zeros((BATCHSIZE, 4), device=device)
    trainingnormals = torch.zeros((BATCHSIZE, 3), device=device)
    trainingsdf = torch.zeros((BATCHSIZE), device=device)

    n_on_surface = config["training_data"].get("n_on_surface", math.ceil(BATCHSIZE * 0.25))
    n_off_surface = config["training_data"].get("n_off_surface", math.ceil(BATCHSIZE * 0.25))
    n_int_times = config["training_data"].get("n_int_times", BATCHSIZE - (n_on_surface + n_off_surface))
    training_loss = {}
    scale = float(config["loss"].get("scale", 1e-3))
    lossmeancurv = LossMeanCurvature(scale=scale)

    best_loss = torch.inf
    best_weigths = None
    omegas = {3: 10}  # Setting the omega_0 value of t (coord. 3) to 10
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
        loss = lossmeancurv(y, gt)

        running_loss = torch.zeros((1, 1), device=device)
        for k, v in loss.items():
            running_loss += v
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

            if CHECKPOINT_AT and e and not e % CHECKPOINT_AT:
                times = [-1., -0.5, 0.0, 0.9]
                meshpath = osp.join(experimentpath, "reconstructions", f"check_{e}")
                os.makedirs(meshpath, exist_ok=True)
                reconstruct_with_curvatures(
                    model, times, meshpath, device=device, resolution=256
                )
                model = model.train()

            if not e % 100 and e > 0:
                print(f"Step {e} --- Loss {running_loss.item()}")

    training_time = time.time() - start_training_time
    print(f"training took {training_time} s")
    writer.flush()
    writer.close()

    torch.save(model.state_dict(), osp.join(experimentpath, "models", "weights.pth"))
    model.load_state_dict(best_weights)
    model.update_omegas(w0=1)
    torch.save(
        model.state_dict(), osp.join(experimentpath, "models", "best.pth")
    )
