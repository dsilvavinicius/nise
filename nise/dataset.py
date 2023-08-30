# coding: utf-8

import math
import numpy as np
import open3d as o3d
import open3d.core as o3c
from plyfile import PlyData
import torch
from torch.utils.data import Dataset
from nise.model import SIREN, from_pth
from nise.diff_operators import gradient


def _sample_on_surface(vertices: torch.Tensor, n_points: int, device: str):
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


def _sample_initial_condition(
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
    surf_pts, _ = _sample_on_surface(
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


def _create_training_data(
    initial_conditions: list,
    n_samples: int,
    off_surface_sampler: torch.distributions.distribution.Distribution,
    device: torch.device,
    time_sampler: torch.distributions.distribution.Distribution = None,
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
        samples = _sample_initial_condition(
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
                for i in range(len(full_samples[k])):
                    full_samples[k][i] = torch.cat((
                        full_samples[k][i], samples[k][i]
                    ), dim=0)

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


def _read_ply(path: str, t: float):
    """Reads a PLY file with position and normal data.

    Note that we expect the input ply to contain x,y,z vertex data, as well
    as nx,ny,nz normal data. The time coordinate is added as a column in the
    returned `vertices` tensor.

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

    vertices: torch.Tensor
        The same vertex information as stored in `mesh`, augmented by the SDF
        values as the last column (a column of zeroes) and time data in column
        3. Returned for easier, structured access.

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


class SpaceTimePointCloudNI(Dataset):
    """Space-time varying point clouds with Neural Implicits for SDF querying.

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

    See Also
    --------
    SpaceTimePointCloud

    References
    ----------
    Tiago Novello, Guilherme Schardong, Luiz Schirmer, Vinícius da Silva,
    Hélio Lopes, and Luiz Velho. Exploring differential geometry in neural
    implicits. Computers & Graphics, 108, 2022
    """
    def __init__(self, inputpaths, batchsize, device=torch.device("cuda:0"),
                 fraction_on_surface=0.25, fraction_off_surface=0.25):
        super(SpaceTimePointCloudNI, self).__init__()
        self.batchsize = batchsize
        self.fraction_on_surface = fraction_on_surface
        self.fraction_off_surface = fraction_off_surface
        self.device = device

        vertices = [None] * len(inputpaths)
        ni = [None] * len(inputpaths)
        for i, (meshpath, nipath, paramval, w0) in enumerate(inputpaths):
            _, verts = _read_ply(meshpath, paramval)
            vertices[i] = verts.to(device)
            ni[i] = from_pth(nipath, device=device, w0=w0).to(device)

        self.vertices_ni = list(zip(vertices, ni))

        self.off_surf_sampler = torch.distributions.uniform.Uniform(-1, 1)
        self.time_sampler = torch.distributions.uniform.Uniform(-0.2, 1.0)

    def __len__(self):
        return sum([m.shape[0] for m, _ in self.vertices_ni])

    def __getitem__(self, n):
        return _create_training_data(
            self.vertices_ni,
            n_samples=self.batchsize,
            fraction_on_surface=self.fraction_on_surface,
            fraction_off_surface=self.fraction_off_surface,
            off_surface_sampler=self.off_surf_sampler,
            time_sampler=self.time_sampler,
            device=self.device
        )


class SpaceTimePointCloud(Dataset):
    """Point Cloud dataset with time-varying data. SDF querying is done
     directly on the meshes.

    Parameters
    ----------
    mesh_paths: list of tuples[str, number]
        Paths to the base meshes. Each item in this list is a tuple with the
        mesh path and its time.

    samples_on_surface: int
        Number of surface samples to fetch (i.e. {X | f(X) = 0}).

    timerange: list of two numbers, optional
        Range of time to sample points. Overrides the timerange set by
        `mesh_paths`. Default value is `None`, meaning we will infer the
        timerange from `mesh_paths`.

    batch_size: integer, optional
        Only used when `no_sampler` is `True`. Used for fetching `batch_size`
        at every call of `__getitem__`. If set to 0 (default), fetches all
        on-surface points at every call.

    silent: boolean, optional
        Whether to report the progress of loading and processing the mesh (if
        set to False, default behavior), or not (if True).

    See Also
    --------
    trimesh.load, mesh_to_sdf.get_surface_point_cloud,
    _sample_on_surface
    """
    def __init__(self, mesh_paths, samples_on_surface, timerange=None,
                 batch_size=0, silent=False):
        super().__init__()
        self.samples_on_surface = samples_on_surface
        self.batch_size = batch_size

        # This is a mode-2 tensor that will hold our surface samples for all
        # given meshes. This tensor's shape is [NxT, 8], where N is the number
        # of surface samples, 8 for the features (x, y, z, t, nx, ny, nz,
        # sdf) and, T is the number of timesteps.
        self.surface_samples = torch.zeros(samples_on_surface * len(mesh_paths), 8)

        # SDF query structure for each initial condition.
        self.scenes = [None] * len(mesh_paths)

        self.min_time, self.max_time = np.inf, -np.inf
        if timerange is not None:
            self.min_time, self.max_time = timerange
        else:
            if len(mesh_paths) == 1:
                self.min_time, self.max_time = 0.0, 1.0

        for i, mesh_path in enumerate(mesh_paths):
            path, t = mesh_path

            if not silent:
                print(f"Loading mesh \"{path}\" at time {t}.")

            mesh = o3d.io.read_triangle_mesh(path)
            mesh.compute_vertex_normals()
            mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

            if not silent:
                print(f"Creating point-cloud and acceleration structures for time {t}.")

            self.scenes[i] = o3d.t.geometry.RaycastingScene()
            self.scenes[i].add_triangles(mesh)

            if not silent:
                print(f"Sampling surface at time {t}.")

            surface_samples, _ = _sample_on_surface(
                mesh,
                samples_on_surface
            )
            rows = range(i * samples_on_surface, (i+1) * samples_on_surface)
            self.surface_samples[rows, :3] = surface_samples[..., :3]
            self.surface_samples[rows, 3] = t
            self.surface_samples[rows, 4:] = surface_samples[..., 3:]

            if not silent:
                print(f"Done for time {t}.")

        if not silent:
            print("Done preparing the dataset.")

    def __len__(self):
        return len(self.scenes) * self.samples_on_surface // self.batch_size

    def __getitem__(self, idx):
        return self._random_sampling(self.batch_size)

    def _random_sampling(self, n_points):
        """Randomly samples points on the surface and function domain."""
        if n_points <= 0:
            n_points = self.samples_on_surface

        # on_surface_count = n_points // 3
        on_surface_count = n_points // 4
        off_surface_count = on_surface_count
        intermediate_count = n_points - (on_surface_count + off_surface_count)

        on_surface_samples = self._sample_on_surface_init_conditions(on_surface_count)
        off_surface_samples = self._sample_off_surface_init_conditions(off_surface_count)
        intermediate_samples = self._sample_intermediate_times(intermediate_count)

        samples = torch.cat(
            (on_surface_samples, off_surface_samples, intermediate_samples),
            dim=0
        )

        return {
            "coords": samples[:, :4].float(),
            "normals": samples[:, 4:7].float(),
            "sdf": samples[:, -1].unsqueeze(-1).float(),
        }

    def _sample_on_surface_init_conditions(self, n_points):
        # Selecting the points on surface. Each mesh has `samples_on_surface`
        # points sampled from it, thus, we must select
        # `num_meshes * samples_on_surface` points here.
        idx = np.random.choice(
            range(self.surface_samples.shape[0]),
            size=n_points,
            replace=False
        )

        return self.surface_samples[idx, :]

    def _sample_off_surface_init_conditions(self, n_points):
        # Same principle here. We select the points off-surface and then
        # distribute them along time.
        off_surface_points = np.random.uniform(-1, 1, size=(n_points, 3))
        unique_times = torch.unique(self.surface_samples[:, 3])
        times = np.random.choice(
            unique_times,
            size=n_points,
            replace=True
        )

        # Concatenating the time as a new coordinate => (x, y, z, t).
        off_surface_points = torch.cat((
            torch.from_numpy(off_surface_points),
            torch.from_numpy(times).unsqueeze(-1)
        ), dim=1)

        # Estimating the SDF and normals for each initial condition.
        num_times = len(unique_times)
        off_surface_coords, off_surface_sdf, off_surface_normals = None, None, None
        for i in range(num_times):
            points_idx = off_surface_points[:, -1] == unique_times[i]
            domain_pts = o3c.Tensor(
                off_surface_points[points_idx, :-1].numpy(), dtype=o3c.Dtype.Float32
            )
            domain_sdf = self.scenes[i].compute_signed_distance(domain_pts)

            # sdf_i, normals_i = self.scenes[i].get_sdf(
            #     off_surface_points[points_idx, :-1],
            #     use_depth_buffer=False,
            #     return_gradients=True
            # )

            if off_surface_sdf is None:
                off_surface_coords = off_surface_points[points_idx, :]
                off_surface_sdf = domain_sdf.numpy()[:, np.newaxis]
                # off_surface_normals = normals_i
                continue

            off_surface_coords = np.vstack((off_surface_coords, off_surface_points[points_idx, :]))
            off_surface_sdf = np.vstack((off_surface_sdf, domain_sdf.numpy()[:, np.newaxis]))
            # off_surface_normals = np.vstack((off_surface_normals, normals_i))

        off_surface_samples = torch.from_numpy(np.hstack((
            off_surface_coords,
            -1 * np.ones((n_points, 3)),
            # off_surface_normals,
            off_surface_sdf
        )).astype(np.float32))

        return off_surface_samples

    def _sample_intermediate_times(self, n_points):
        # Samples for intermediate times.
        off_spacetime_points = np.random.uniform(
            self.min_time, self.max_time, size=(n_points, 4)
        )
        # Warning: time goes from -1 to 1
        # Also note that these points have no normals, or F values.
        # We mark them with -1 here.
        samples = torch.cat((
            torch.from_numpy(off_spacetime_points.astype(np.float32)),
            torch.full(size=(n_points, 3), fill_value=-1, dtype=torch.float32),
            torch.full(size=(n_points, 1), fill_value=-1, dtype=torch.float32),
        ), dim=1)
        return samples


class SpaceTimePointCloudNIDeprecated(Dataset):
    """SDF Point Cloud dataset with time-varying data and NI SDF querying.

    Parameters
    ----------
    mesh_paths: list of tuples[str, number]
        Paths to the base meshes. Each item in this list is a tuple with the
        mesh path and its time.

    samples_on_surface: int
        Number of surface samples to fetch (i.e. {X | f(X) = 0}).

    pretrained_ni: list of SIREN
        A pre-trained neural network to be used for points
        where SDF!=0. This may help reduce running times since we avoid a
        costly closest point calculation. As for `mesh_paths`, we pass the
        model and time associated to it.

    timerange: list of two numbers, optional
        Range of time to sample points. Overrides the timerange set by
        `mesh_paths`. Default value is `None`, meaning we will infer the
        timerange from `mesh_paths`.

    batch_size: integer, optional
        Only used when `no_sampler` is `True`. Used for fetching `batch_size`
        at every call of `__getitem__`. If set to 0 (default), fetches all
        on-surface points at every call.

    silent: boolean, optional
        Whether to report the progress of loading and processing the mesh (if
        set to False, default behavior), or not (if True).

    See Also
    --------
    open3d.io.read_triangle_mesh, open3d.t.geometry.TriangleMesh.from_legacy,
    _sample_on_surface
    """
    def __init__(self, mesh_paths, samples_on_surface, pretrained_ni,
                 timerange=None, batch_size=0, silent=False, device='cpu'):
        super().__init__()

        self.device = device
        self.samples_on_surface = samples_on_surface
        self.batch_size = batch_size

        # This is a mode-2 tensor that will hold our surface samples for all
        # given meshes. This tensor's shape is [NxT, 8], where N is the number
        # of points of each mesh, 8 for the features (x, y, z, t, nx, ny, nz,
        # sdf) and, T is the number of timesteps.
        self.surface_samples = torch.zeros(samples_on_surface * len(mesh_paths), 8)

        # SDF query structure for each initial condition.
        if isinstance(pretrained_ni, (SIREN, torch.nn.Module)):
            self.pretrained_ni = [pretrained_ni]
        else:
            self.pretrained_ni = pretrained_ni

        self.min_time, self.max_time = np.inf, -np.inf
        if timerange is not None:
            self.min_time, self.max_time = timerange
        else:
            if len(mesh_paths) == 1:
                self.min_time, self.max_time = -1, 1

        for i, mesh_path in enumerate(mesh_paths):
            path, t = mesh_path
            if timerange is None and len(mesh_paths) > 1:
                if self.min_time > t:
                    self.min_time = t
                if self.max_time < t:
                    self.max_time = t

            if not silent:
                print(f"Loading mesh \"{path}\" at time {t}.")

            mesh = o3d.io.read_triangle_mesh(path)
            mesh.compute_vertex_normals()
            mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
            # mesh = trimesh.load(path)

            if not silent:
                print(f"Creating point-cloud and acceleration structures for time {t}.")

            # We will fetch random samples at every access.
            if not silent:
                print(f"Sampling surface at time {t}.")

            surface_samples, _ = _sample_on_surface(
                mesh,
                samples_on_surface
            )
            rows = range(i * samples_on_surface, (i+1) * samples_on_surface)
            self.surface_samples[rows, :3] = surface_samples[..., :3]
            self.surface_samples[rows, 3] = t
            self.surface_samples[rows, 4:] = surface_samples[..., 3:]

            if not silent:
                print(f"Done for time {t}.")

        if not silent:
            print("Done preparing the dataset.")

    def __len__(self):
        return 4 * self.samples_on_surface // self.batch_size

    def __getitem__(self, idx):
        return self._random_sampling(self.batch_size)

    def _random_sampling(self, n_points):
        """Randomly samples points on the surface and function domain."""
        if n_points <= 0:
            n_points = self.samples_on_surface

        on_surface_count = n_points // 4
        off_surface_count = on_surface_count
        intermediate_count = n_points - (on_surface_count + off_surface_count)

        surface_samples = self._sample_surface_init_conditions(off_surface_count).cpu()
        intermediate_samples = self._sample_intermediate_times(intermediate_count)

        samples = torch.cat(
            #(on_surface_samples, off_surface_samples, intermediate_samples),
            (surface_samples, intermediate_samples),
            dim=0
        )

        return {
            # "coords": samples[:, :4].float(),
            "coords": samples[:, :4].float(),
            "normals": samples[:, 4:7].float(),
            "sdf": samples[:, -1].unsqueeze(-1).float(),
        }

    def _sample_surface_init_conditions_no_net(self, n_points):
        # Same principle here. We select the points off-surface and then
        # distribute them along time.
        off_surface_points = np.random.uniform(-1, 1, size=(n_points, 3))
        unique_times = torch.unique(self.surface_samples[:, 3])
        times = np.random.choice(
            unique_times,
            size=2*n_points,
            replace=True
        )

        idx = np.random.choice(
            range(self.surface_samples.shape[0]),
            size=n_points,
            replace=False
        )

        on_surface_coords = self.surface_samples[idx, 0:3]
        surface_coords = torch.cat((on_surface_coords, torch.from_numpy(off_surface_points)))

        # Concatenating the time as a new coordinate => (x, y, z, t).
        off_surface_points = torch.cat((
            surface_coords,
            torch.from_numpy(times).unsqueeze(-1)
        ), dim=1).float().to(self.device)

        # Estimating the SDF and normals for each initial condition.
        num_times = len(unique_times)
        off_surface_coords, off_surface_sdf, off_surface_normals = None, None, None

        for i in range(num_times):
            points_idx = off_surface_points[:, -1] == unique_times[i]

            if off_surface_sdf is None:
                off_surface_coords = off_surface_points[points_idx, :]
                off_surface_sdf = torch.full(size=(n_points, 3), fill_value=-1, dtype=torch.float32).cuda()#[:, np.newaxis]
                off_surface_normals = torch.full(size=(n_points, 1), fill_value=-1, dtype=torch.float32).cuda()
                continue

            off_surface_coords = torch.cat((off_surface_coords, off_surface_points[points_idx, :]), dim=0)
            off_surface_sdf = torch.cat((off_surface_sdf, torch.full(size=(n_points, 3), fill_value=-1, dtype=torch.float32).cuda()), dim=0)
            off_surface_normals = torch.cat((off_surface_normals, torch.full(size=(n_points, 1), fill_value=-1, dtype=torch.float32).cuda()), dim=0)

        off_surface_samples = torch.cat((
            off_surface_coords,
            off_surface_normals,
            off_surface_sdf
        ), dim=1).float()

        return off_surface_samples.clone().detach()

    def _sample_surface_init_conditions(self, n_points):
        # Same principle here. We select the points off-surface and then
        # distribute them along time.
        off_surface_points = np.random.uniform(-1, 1, size=(n_points, 3))
        unique_times = torch.unique(self.surface_samples[:, 3])
        times = np.random.choice(
            unique_times,
            size=2*n_points,
            replace=True
        )

        idx = np.random.choice(
            range(self.surface_samples.shape[0]),
            size=n_points,
            replace=False
        )

        on_surface_coords = self.surface_samples[idx, 0:3]
        surface_coords = torch.cat((on_surface_coords, torch.from_numpy(off_surface_points)))

        # Concatenating the time as a new coordinate => (x, y, z, t).
        off_surface_points = torch.cat((
            surface_coords,
            torch.from_numpy(times).unsqueeze(-1)
        ), dim=1).float().to(self.device)

        # Estimating the SDF and normals for each initial condition.
        num_times = len(unique_times)
        off_surface_coords, off_surface_sdf, off_surface_normals = None, None, None

        for i in range(num_times):
            points_idx = off_surface_points[:, -1] == unique_times[i]
            model_sdf_i = self.pretrained_ni[i](
                off_surface_points[points_idx, :-1].to(self.device)
            )

            sdf_i = model_sdf_i['model_out']
            normals_i = gradient(sdf_i, model_sdf_i['model_in'])

            if off_surface_sdf is None:
                off_surface_coords = off_surface_points[points_idx, :]
                off_surface_sdf = sdf_i#[:, np.newaxis]
                off_surface_normals = normals_i
                continue

            off_surface_coords = torch.cat((off_surface_coords, off_surface_points[points_idx, :]), dim=0)
            off_surface_sdf = torch.cat((off_surface_sdf, sdf_i), dim=0)
            off_surface_normals = torch.cat((off_surface_normals, normals_i), dim=0)

        off_surface_samples = torch.cat((
            off_surface_coords,
            off_surface_normals,
            off_surface_sdf
        ), dim=1).float()

        return off_surface_samples.clone().detach()

    def _sample_on_surface_init_conditions(self, n_points):
        # Selecting the points on surface. Each mesh has `samples_on_surface`
        # points sampled from it, thus, we must select
        # `num_meshes * samples_on_surface` points here.
        idx = np.random.choice(
            range(self.surface_samples.shape[0]),
            size=n_points,
            replace=False
        )

        return self.surface_samples[idx, :]

    def _sample_off_surface_init_conditions(self, n_points):
        # Same principle here. We select the points off-surface and then
        # distribute them along time.
        off_surface_points = np.random.uniform(-1, 1, size=(n_points, 3))
        unique_times = torch.unique(self.surface_samples[:, 3])
        times = np.random.choice(
            unique_times,
            size=n_points,
            replace=True
        )

        # Concatenating the time as a new coordinate => (x, y, z, t).
        off_surface_points = torch.cat((
            torch.from_numpy(off_surface_points),
            torch.from_numpy(times).unsqueeze(-1)
        ), dim=1).float().to(self.device)

        # Estimating the SDF and normals for each initial condition.
        num_times = len(unique_times)
        off_surface_coords, off_surface_sdf, off_surface_normals = None, None, None

        for i in range(num_times):
            points_idx = off_surface_points[:, -1] == unique_times[i]
            model_sdf_i = self.pretrained_ni[i](
                off_surface_points[points_idx, :-1].to(self.device)
            )

            sdf_i = model_sdf_i['model_out']
            normals_i = gradient(sdf_i, model_sdf_i['model_in'])

            if off_surface_sdf is None:
                off_surface_coords = off_surface_points[points_idx, :]
                off_surface_sdf = sdf_i#[:, np.newaxis]
                off_surface_normals = normals_i
                continue

            off_surface_coords = torch.cat((off_surface_coords, off_surface_points[points_idx, :]), dim=0)
            off_surface_sdf = torch.cat((off_surface_sdf, sdf_i), dim=0)
            off_surface_normals = torch.cat((off_surface_normals, normals_i), dim=0)

        off_surface_samples = torch.cat((
            off_surface_coords,
            off_surface_normals,
            off_surface_sdf
        ), dim=1).float()

        return off_surface_samples

    def _sample_intermediate_times(self, n_points):
        # Samples for intermediate times.
        #off_spacetime_points = np.random.uniform(-0.6, 0.6, size=(n_points, 4))

        off_spacetime_coords = np.random.uniform(-1, 1, size=(n_points, 3))
        # off_spacetime_time = np.random.uniform(self.min_time, self.max_time, size=(n_points, 1))
        # off_spacetime_time = np.random.uniform(-1, 1, size=(n_points, 1))
        off_spacetime_time = np.random.uniform(-0.2, 0.2, size=(n_points, 1))

        # warning: time goes from -1 to 1
        samples = torch.cat((
            torch.from_numpy(off_spacetime_coords.astype(np.float32)),
            torch.from_numpy(off_spacetime_time.astype(np.float32)),
            torch.full(size=(n_points, 3), fill_value=-1, dtype=torch.float32),
            torch.full(size=(n_points, 1), fill_value=-1, dtype=torch.float32),
        ), dim=1)
        return samples


class SpaceTimePointCloudNILipschitzDeprecated(Dataset):
    def __init__(self, mesh_paths, samples_on_surface, pretrained_ni, batch_size=0,
                 silent=False, device='cpu'):
        super().__init__()

        self.device = device
        self.samples_on_surface = samples_on_surface
        self.no_sampler = True
        self.batch_size = batch_size

        # This is a mode-2 tensor that will hold our surface samples for all
        # given meshes. This tensor's shape is [NxT, 8], where N is the number
        # of points of each mesh, 8 for the features (x, y, z, t, nx, ny, nz,
        # sdf) and, T is the number of timesteps.
        self.surface_samples = torch.zeros(samples_on_surface * len(mesh_paths), 8)

        # SDF query structure for each initial condition.
        self.pretrained_ni = pretrained_ni

        self.min_time, self.max_time = np.inf, -np.inf

        if len(mesh_paths) == 1:
            self.min_time, self.max_time = -1, 1


        for i, mesh_path in enumerate(mesh_paths):
            path, t = mesh_path
            if self.min_time > t:
                self.min_time = t
            if self.max_time < t:
                self.max_time = t

            if not silent:
                print(f"Loading mesh \"{path}\" at time {t}.")
            mesh = trimesh.load(path)

            if not silent:
                print(f"Creating point-cloud and acceleration structures for time {t}.")

            # We will fetch random samples at every access.
            if not silent:
                print(f"Sampling surface at time {t}.")

            surface_samples = _sample_on_surface(
                mesh,
                samples_on_surface,
                sample_vertices=True
            )
            rows = range(i * samples_on_surface, (i+1) * samples_on_surface)
            self.surface_samples[rows, :3] = surface_samples[..., :3]
            self.surface_samples[rows, 3] = t
            self.surface_samples[rows, 4:] = surface_samples[..., 3:]

            if not silent:
                print(f"Done for time {t}.")

        if not silent:
            print("Done preparing the dataset.")

    def __len__(self):
        if self.no_sampler:
            return 4 * self.samples_on_surface // self.batch_size
        return self.samples_on_surface

    def __getitem__(self, idx):
        if self.no_sampler:
            return self._random_sampling(self.batch_size)
        raise NotImplementedError

    def _random_sampling(self, n_points):
        """Randomly samples points on the surface and function domain."""
        if n_points <= 0:
            n_points = self.samples_on_surface

        samples = self._sample_surface_init_conditions(n_points).cpu()
        return {
            "coords": samples[:, :4].float(),
            #"normals": samples[:, 4:7].float(),
            #"sdf": samples[:, -1].unsqueeze(-1).float(),
        }


    def _sample_surface_init_conditions(self, n_points):
        # Same principle here. We select the points off-surface and then
        # distribute them along time.

        n_on_surface = math.floor(n_points*0.4)
        n_near_surface = n_on_surface#math.floor(n_points*0.4)
        n_off_surface = n_points - n_on_surface - n_near_surface


        off_surface_points = np.random.uniform(-1.0, 1.0, size=(n_off_surface, 3))
        unique_times = torch.unique(self.surface_samples[:, 3])
        times = np.random.choice(
            unique_times,
            size=n_points,
            replace=True
        )

        idx = np.random.choice(
            range(self.surface_samples.shape[0]),
            size=n_on_surface,
            replace=False
        )

        on_surface_coords = self.surface_samples[idx, 0:3]
        near_surface_coords = self.surface_samples[idx, 0:3] + (torch.rand(n_on_surface, 3)*2e-4 - 1e-4)
        surface_coords = torch.cat((on_surface_coords, near_surface_coords, torch.from_numpy(off_surface_points)))

        # Concatenating the time as a new coordinate => (x, y, z, t).
        off_surface_points = torch.cat((
            surface_coords,
            torch.from_numpy(times).unsqueeze(-1)
        ), dim=1).float().to(self.device)


        return off_surface_points.clone().detach()



    def _sample_on_surface_init_conditions(self, n_points):
        # Selecting the points on surface. Each mesh has `samples_on_surface`
        # points sampled from it, thus, we must select
        # `num_meshes * samples_on_surface` points here.
        idx = np.random.choice(
            range(self.surface_samples.shape[0]),
            size=n_points,
            replace=False
        )

        return self.surface_samples[idx, :]

    def _sample_off_surface_init_conditions(self, n_points):
        # Same principle here. We select the points off-surface and then
        # distribute them along time.
        off_surface_points = np.random.uniform(-1, 1, size=(n_points, 3))
        unique_times = torch.unique(self.surface_samples[:, 3])
        times = np.random.choice(
            unique_times,
            size=n_points,
            replace=True
        )

        # Concatenating the time as a new coordinate => (x, y, z, t).
        off_surface_points = torch.cat((
            torch.from_numpy(off_surface_points),
            torch.from_numpy(times).unsqueeze(-1)
        ), dim=1).float().to(self.device)

        # Estimating the SDF and normals for each initial condition.
        num_times = len(unique_times)
        off_surface_coords, off_surface_sdf, off_surface_normals = None, None, None

        for i in range(num_times):
            points_idx = off_surface_points[:, -1] == unique_times[i]
            model_sdf_i = self.pretrained_ni[i](
                off_surface_points[points_idx, :-1].to(self.device)
            )

            sdf_i = model_sdf_i['model_out']
            normals_i = gradient(sdf_i, model_sdf_i['model_in'])

            if off_surface_sdf is None:
                off_surface_coords = off_surface_points[points_idx, :]
                off_surface_sdf = sdf_i#[:, np.newaxis]
                off_surface_normals = normals_i
                continue

            off_surface_coords = torch.cat((off_surface_coords, off_surface_points[points_idx, :]), dim=0)
            off_surface_sdf = torch.cat((off_surface_sdf, sdf_i), dim=0)
            off_surface_normals = torch.cat((off_surface_normals, normals_i), dim=0)

        off_surface_samples = torch.cat((
            off_surface_coords,
            off_surface_normals,
            off_surface_sdf
        ), dim=1).float()

        return off_surface_samples

    def _sample_intermediate_times(self, n_points):
        # Samples for intermediate times.
        #off_spacetime_points = np.random.uniform(-0.6, 0.6, size=(n_points, 4))
        off_spacetime_points = np.random.uniform(self.min_time, self.max_time, size=(n_points, 4))
        # warning: time goes from -1 to 1
        samples = torch.cat((
            torch.from_numpy(off_spacetime_points.astype(np.float32)),
            torch.full(size=(n_points, 3), fill_value=-1, dtype=torch.float32),
            torch.full(size=(n_points, 1), fill_value=-1, dtype=torch.float32),
        ), dim=1)
        return samples


if __name__ == "__main__":
    meshes = [
        ("data/bunny_noisy.ply", 0),
        #("data/happy.ply", 0.1),
    ]
    spc = SpaceTimePointCloud(meshes, 20, timerange=[0, 0.4])
    data = spc[0]
    X = data["coords"]
    gt = {k: v for k, v in data.items() if k != "coords"}

    device = torch.device("cuda:0")
    bunny_ni = SIREN(3, 1, [256] * 3, w0=30).eval().to(device)
    bunny_ni.load_state_dict(torch.load("ni/bunny_2x256_w-30.pth"))

    spcni = SpaceTimePointCloudNI(meshes, 20, bunny_ni, device=device)
    datani = spcni[0]
    Xni = datani["coords"]
    gtni = {k: v for k, v in datani.items() if k != "coords"}

    print(X.shape, Xni.shape)
    print(X)
    print(Xni)

    for k in gt:
        print(f"---------------------------{k}-----------------------------")
        print(gt[k].shape, gtni[k].shape)
        print(gt[k])
        print(gtni[k])
