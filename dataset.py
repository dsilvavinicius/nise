# coding: utf-8

from itertools import repeat
from mesh_to_sdf.surface_point_cloud import SurfacePointCloud
from mesh_to_sdf import (get_surface_point_cloud, scale_to_unit_cube,
                         scale_to_unit_sphere)
import numpy as np
import trimesh
import torch
from torch.utils.data import Dataset

from util import gradient
# from util import sample_mesh


def _sample_on_surface(mesh: trimesh.Trimesh,
                       n_points: int,
                       sample_vertices=True) -> torch.Tensor:
    if sample_vertices:
        idx = np.random.choice(
            np.arange(start=0, stop=len(mesh.vertices)),
            size=n_points,
            replace=False
        )
        on_points = mesh.vertices[idx]
        on_normals = mesh.vertex_normals[idx]
    else:
        on_points, face_idx = mesh.sample(
            count=n_points,
            return_index=True
        )
        on_normals = mesh.face_normals[face_idx]

    return torch.from_numpy(np.hstack((
        on_points,
        on_normals,
        np.zeros((n_points, 1))
    )).astype(np.float32))


def _sample_domain(point_cloud: SurfacePointCloud,
                   n_points: int,
                   balance_in_out_points=False) -> torch.Tensor:
    domain_points = np.random.uniform(-1, 1, size=(n_points, 3))
    domain_sdf, domain_normals = point_cloud.get_sdf(
        domain_points,
        use_depth_buffer=False,
        return_gradients=True
    )
    if balance_in_out_points and n_points > 1:
        in_surf = domain_sdf < 0
        domain_sdf = np.concatenate((
            domain_sdf[in_surf],
            domain_sdf[~in_surf][:sum(in_surf)]), axis=0
        )
        domain_points = np.vstack((
            domain_points[in_surf, :],
            domain_points[~in_surf, :][:sum(in_surf)]
        ))
        domain_normals = np.vstack((
            domain_normals[in_surf, :],
            domain_normals[~in_surf, :][:sum(in_surf)]
        ))

    return torch.from_numpy(np.hstack((
        domain_points,
        domain_normals,
        domain_sdf[:, np.newaxis]
    )).astype(np.float32))


class PointCloud(Dataset):
    """SDF Point Cloud dataset.

    Parameters
    ----------
    mesh_path: str
        Path to the base mesh.

    samples_on_surface: int
        Number of surface samples to fetch (i.e. {X | f(X) = 0}).

    scaling: str or None, optional
        The scaling to apply to the mesh. Possible values are: None
        (no scaling), "bbox" (-1, 1 in all axes), "sphere" (to fit the mesh in
        an unit sphere). Default is None.

    off_surface_sdf: number, optional
        Value to replace the SDF calculated by the sampling function for points
        with SDF != 0. May be used to replicate the behavior of Sitzmann et al.
        If set to `None` (default) uses the SDF estimated by the sampling
        function.

    off_surface_normals: np.array(size=(1, 3)), optional
        Value to replace the normals calculated by the sampling algorithm for
        points with SDF != 0. May be used to replicate the behavior of Sitzmann
        et al. If set to `None` (default) uses the SDF estimated by the
        sampling function.

    random_surf_samples: boolean, optional
        Wheter to randomly return points on surface (SDF=0) on `__getitem__`.
        If set to False (default), will return the specified point. Otherwise,
        `__getitem__` will return a randomly selected point, even if the same
        `idx` is provided.

    no_sampler: boolean, optional
        When this option is True, we assume that no sampler will be provided to
        the DataLoader, meaning that our `__getitem__` will return a batch of
        points instead of a single point, mimicking the behavior of Sitzmann
        et al. [1]. Default is False, meaning that an external sampler will be
        used.

    batch_size: integer, optional
        Only used when `no_sampler` is `True`. Used for fetching `batch_size`
        at every call of `__getitem__`. If set to 0 (default), fetches all
        on-surface points at every call.

    silent: boolean, optional
        Whether to report the progress of loading and processing the mesh (if
        set to False, default behavior), or not (if True).

    See Also
    --------
    trimesh.load, mesh_to_sdf.get_surface_point_cloud, _sample_domain,
    _sample_on_surface

    References
    ----------
    [1] Sitzmann, V., Martel, J. N. P., Bergman, A. W., Lindell, D. B.,
    & Wetzstein, G. (2020). Implicit Neural Representations with Periodic
    Activation Functions. ArXiv. Retrieved from http://arxiv.org/abs/2006.09661
    """
    def __init__(self, mesh_path, samples_on_surface, scaling=None,
                 off_surface_sdf=None, off_surface_normals=None,
                 random_surf_samples=False, no_sampler=False, batch_size=0,
                 silent=False):
        super().__init__()

        self.samples_on_surface = samples_on_surface
        self.off_surface_sdf = off_surface_sdf
        self.no_sampler = no_sampler
        self.batch_size = batch_size

        if off_surface_normals is None:
            self.off_surface_normals = None
        else:
            self.off_surface_normals = torch.from_numpy(
                off_surface_normals.astype(np.float32)
            )

        if not silent:
            print(f"Loading mesh \"{mesh_path}\".")

        mesh = trimesh.load(mesh_path)
        if scaling is not None and scaling:
            if scaling == "bbox":
                mesh = scale_to_unit_cube(mesh)
            elif scaling == "sphere":
                mesh = scale_to_unit_sphere(mesh)
            else:
                raise ValueError("Invalid scaling option.")

        self.mesh = mesh
        if not silent:
            print("Creating point-cloud and acceleration structures.")

        self.point_cloud = get_surface_point_cloud(
            mesh,
            surface_point_method="scan",
            # bounding_radius=1,
            calculate_normals=True
        )

        # We will fetch random samples at every access.
        self.random_surf_samples = random_surf_samples
        if not silent:
            print("Sampling surface.")

        self.surface_samples = _sample_on_surface(
            mesh,
            samples_on_surface,
            sample_vertices=True
        )

        if not silent:
            print("Done preparing the dataset.")

    def __len__(self):
        if self.no_sampler:
            return 2 * self.samples_on_surface // self.batch_size
        return self.samples_on_surface

    def __getitem__(self, idx):
        if self.no_sampler:
            return self._random_sampling(self.batch_size)

        # For indices of points-on-surface, we select and return a point with
        # sdf=0.
        if idx in range(self.samples_on_surface):
            if self.random_surf_samples:
                i = np.random.choice(
                    range(self.samples_on_surface),
                    size=1,
                    replace=False
                )
                sample = self.surface_samples[i, :]
            else:
                sample = self.surface_samples[idx, :]
                # Reshaping from [D] to [1, D] because to keep it uniform with
                # other sampling approaches
                sample = sample.unsqueeze(0)
            return {
                "coords": sample[:, :3].float(),
                "normals": sample[:, 3:6].float(),
                "sdf": sample[:, -1].unsqueeze(-1).float(),
            }

        # If the provided idx is out-of-range, we simply sample from the
        # domain.
        sample = _sample_domain(self.point_cloud, 1)
        if self.off_surface_sdf is not None:
            sample[0, -1] = self.off_surface_sdf
        if self.off_surface_normals is not None:
            sample[0, 3:6] = self.off_surface_normals
        return {
            "coords": sample[:, :3].float(),
            "normals": sample[:, 3:6].float(),
            "sdf": sample[:, -1].unsqueeze(-1).float(),
        }

    def _random_sampling(self, n_points):
        """Randomly samples points on the surface and function domain."""
        if n_points <= 0:
            n_points = self.samples_on_surface

        on_surface_count = n_points // 2
        off_surface_count = n_points - on_surface_count

        on_surface_idx = np.random.choice(
            range(self.samples_on_surface),
            size=on_surface_count,
            replace=False
        )
        on_surface_samples = self.surface_samples[on_surface_idx, :]

        off_surface_points = np.random.uniform(-1, 1, size=(off_surface_count, 3))
        off_surface_sdf, off_surface_normals = self.point_cloud.get_sdf(
            off_surface_points,
            use_depth_buffer=False,
            return_gradients=True
        )
        off_surface_samples = torch.from_numpy(np.hstack((
            off_surface_points,
            off_surface_normals,
            off_surface_sdf[:, np.newaxis]
        )).astype(np.float32))

        if self.off_surface_sdf is not None:
            off_surface_samples[:, -1] = self.off_surface_sdf
        if self.off_surface_normals is not None:
            off_surface_samples[:, 3:6] = self.off_surface_normals

        samples = torch.cat((on_surface_samples, off_surface_samples), dim=0)

        # Unsqueezing the SDF since it returns a shape [1] tensor and we need a
        # [1, 1] shaped tensor.
        return {
            "coords": samples[:, :3].float(),
            "normals": samples[:, 3:6].float(),
            "sdf": samples[:, -1].unsqueeze(-1).float()
        }


class SpaceTimePointCloud(Dataset):
    """SDF Point Cloud dataset with time-varying data.

    Parameters
    ----------
    mesh_paths: list of tuples[str, number]
        Paths to the base meshes. Each item in this list is a tuple with the
        mesh path and its time.

    samples_on_surface: int
        Number of surface samples to fetch (i.e. {X | f(X) = 0}).

    off_surface_sdf: number, optional
        Value to replace the SDF calculated by the sampling function for points
        with SDF != 0. May be used to replicate the behavior of Sitzmann et al.
        If set to `None` (default) uses the SDF estimated by the sampling
        function.

    off_surface_normals: np.array(size=(1, 3)), optional
        Value to replace the normals calculated by the sampling algorithm for
        points with SDF != 0. May be used to replicate the behavior of Sitzmann
        et al. If set to `None` (default) uses the SDF estimated by the
        sampling function.

    batch_size: integer, optional
        Only used when `no_sampler` is `True`. Used for fetching `batch_size`
        at every call of `__getitem__`. If set to 0 (default), fetches all
        on-surface points at every call.

    silent: boolean, optional
        Whether to report the progress of loading and processing the mesh (if
        set to False, default behavior), or not (if True).

    pretrained_ni: list of tuples[SIREN, number], optional
        You may provide a pre-trained neural network to be used for points
        where SDF!=0. This may help reduce running times since we avoid a
        costly closest point calculation. As for `mesh_paths`, we pass the
        model and time associated to it.

    See Also
    --------
    trimesh.load, mesh_to_sdf.get_surface_point_cloud, _sample_domain,
    _sample_on_surface
    """
    def __init__(self, mesh_paths, samples_on_surface, scaling=None,
                 off_surface_sdf=None, off_surface_normals=None, batch_size=0,
                 silent=False, pretrained_ni=None):
        super().__init__()

        self.samples_on_surface = samples_on_surface
        self.off_surface_sdf = off_surface_sdf
        self.no_sampler = True
        self.batch_size = batch_size

        if off_surface_normals is None:
            self.off_surface_normals = None
        else:
            self.off_surface_normals = torch.from_numpy(
                off_surface_normals.astype(np.float32)
            )

        # This is a mode-2 tensor that will hold our surface samples for all
        # given meshes. This tensor's shape is [NxT, 8], where N is the number
        # of points of each mesh, 8 for the features (x, y, z, t, nx, ny, nz,
        # sdf) and, T is the number of timesteps.
        self.surface_samples = torch.zeros(samples_on_surface * len(mesh_paths), 8)

        # SDF query structure for each initial condition.
        self.point_clouds = [None] * len(mesh_paths)
        self.pretrained_ni = [None] * len(pretrained_ni)

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

            self.point_clouds[i] = get_surface_point_cloud(
                mesh,
                surface_point_method="scan",
                #bounding_radius=1,
                calculate_normals=True
            )

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
            return 3 * self.samples_on_surface // self.batch_size
        return self.samples_on_surface

    def __getitem__(self, idx):
        if self.no_sampler:
            return self._random_sampling(self.batch_size)
        raise NotImplementedError

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
            sdf_i, normals_i = self.point_clouds[i].get_sdf(
                off_surface_points[points_idx, :-1],
                use_depth_buffer=False,
                return_gradients=True
            )
            
            if off_surface_sdf is None:
                off_surface_coords = off_surface_points[points_idx, :]
                off_surface_sdf = sdf_i[:, np.newaxis]
                off_surface_normals = normals_i
                continue

            off_surface_coords = np.vstack((off_surface_coords, off_surface_points[points_idx, :]))
            off_surface_sdf = np.vstack((off_surface_sdf, sdf_i[:, np.newaxis]))
            off_surface_normals = np.vstack((off_surface_normals, normals_i))

        off_surface_samples = torch.from_numpy(np.hstack((
            off_surface_coords,
            off_surface_normals,
            off_surface_sdf
        )).astype(np.float32))

        if self.off_surface_sdf is not None:
            off_surface_samples[:, -1] = self.off_surface_sdf
        if self.off_surface_normals is not None:
            off_surface_samples[:, 4:7] = self.off_surface_normals

        return off_surface_samples

    def _sample_intermediate_times(self, n_points):
        # Samples for intermediate times.
        #off_spacetime_points = np.random.uniform(-1, 1, size=(n_points, 4))
        off_spacetime_points = np.random.uniform(self.min_time, self.max_time, size=(n_points, 4))
        # warning: time goes from -1 to 1
        samples = torch.cat((
            torch.from_numpy(off_spacetime_points.astype(np.float32)),
            torch.full(size=(n_points, 3), fill_value=-1, dtype=torch.float32),
            torch.full(size=(n_points, 1), fill_value=-1, dtype=torch.float32),
        ), dim=1)
        return samples


class SpaceTimePointCloudNI(Dataset):
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

    off_surface_sdf: number, optional
        Value to replace the SDF calculated by the sampling function for points
        with SDF != 0. May be used to replicate the behavior of Sitzmann et al.
        If set to `None` (default) uses the SDF estimated by the sampling
        function.

    off_surface_normals: np.array(size=(1, 3)), optional
        Value to replace the normals calculated by the sampling algorithm for
        points with SDF != 0. May be used to replicate the behavior of Sitzmann
        et al. If set to `None` (default) uses the SDF estimated by the
        sampling function.

    batch_size: integer, optional
        Only used when `no_sampler` is `True`. Used for fetching `batch_size`
        at every call of `__getitem__`. If set to 0 (default), fetches all
        on-surface points at every call.

    silent: boolean, optional
        Whether to report the progress of loading and processing the mesh (if
        set to False, default behavior), or not (if True).

    pretrained_ni: list of tuples[SIREN, number], optional
        You may provide a pre-trained neural network to be used for points
        where SDF!=0. This may help reduce running times since we avoid a
        costly closest point calculation. As for `mesh_paths`, we pass the
        model and time associated to it.

    See Also
    --------
    trimesh.load, mesh_to_sdf.get_surface_point_cloud, _sample_domain,
    _sample_on_surface
    """
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

        # on_surface_count = n_points // 3
        on_surface_count = n_points // 4
        off_surface_count = on_surface_count
        intermediate_count = n_points - (on_surface_count + off_surface_count)

        #on_surface_samples = self._sample_on_surface_init_conditions(on_surface_count)
        #off_surface_samples = self._sample_off_surface_init_conditions(off_surface_count).cpu()
        surface_samples = self._sample_surface_init_conditions(off_surface_count).cpu()
        #surface_samples = self._sample_surface_init_conditions_no_net(off_surface_count).cpu()
        intermediate_samples = self._sample_intermediate_times(intermediate_count)

        samples = torch.cat(
            #(on_surface_samples, off_surface_samples, intermediate_samples),
            (surface_samples, intermediate_samples),
            dim=0
        )

        return {
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
        ("data/armadillo.ply", 0),
        ("data/double_torus_low.ply", 0.1),
        ("data/cube.ply", 0.6)
    ]
    spc = SpaceTimePointCloud(meshes, 30)
