# coding: utf-8

from mesh_to_sdf.surface_point_cloud import SurfacePointCloud
from mesh_to_sdf import (get_surface_point_cloud, scale_to_unit_cube,
                         scale_to_unit_sphere)
import numpy as np
import trimesh
import torch
from torch.utils.data import Dataset
# from util import sample_mesh


def _sample_on_surface(mesh: trimesh.Trimesh,
                       n_points: int,
                       sample_vertices=True):
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
                   balance_in_out_points=False):
    domain_points = np.random.uniform(-1, 1, size=(n_points, 3))
    domain_sdf, domain_normals = point_cloud.get_sdf(
        domain_points,
        use_depth_buffer=False,
        return_gradients=True
    )
    if balance_in_out_points:
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
        Value to replace the SDF calculated by `sample_mesh` for points with
        SDF != 0. May be used to replicate the behavior of Sitzmann et al. If
        set to `None` uses the SDF estimated by `sample_mesh`.

    See Also
    --------
    sample_mesh

    References
    ----------
    [1] Sitzmann, V., Martel, J. N. P., Bergman, A. W., Lindell, D. B.,
    & Wetzstein, G. (2020). Implicit Neural Representations with Periodic
    Activation Functions. ArXiv. Retrieved from http://arxiv.org/abs/2006.09661
    """
    def __init__(self, mesh_path, samples_on_surface, scaling=None,
                 off_surface_sdf=None):
        super().__init__()

        self.samples_on_surface = samples_on_surface
        self.off_surface_sdf = off_surface_sdf

        mesh = trimesh.load(mesh_path)
        if scaling is not None:
            if scaling == "bbox":
                mesh = scale_to_unit_cube(mesh)
            elif scaling == "sphere":
                mesh = scale_to_unit_sphere(mesh)
            else:
                raise ValueError("Invalid scaling option.")

        self.mesh = mesh
        self.point_cloud = get_surface_point_cloud(
            mesh,
            surface_point_method="scan",
            bounding_radius=1,
            calculate_normals=True
        )
        self.surface_samples = _sample_on_surface(
            mesh,
            samples_on_surface,
            sample_vertices=True
        )

    def __len__(self):
        return self.samples_on_surface

    def __getitem__(self, idx):
        if idx in range(self.surface_samples.size(0)):
            return {
                "coords": self.surface_samples[idx, :3].float(),
                "normals": self.surface_samples[idx, 3:6].float(),
                "sdf": self.surface_samples[idx, -1].float(),
            }

        sample = _sample_domain(self.point_cloud, 1)
        if self.off_surface_sdf is not None:
            sample[0, -1] = self.off_surface_sdf
        return {
            "coords": sample[0, :3].float(),
            "normals": sample[0, 3:6].float(),
            "sdf": sample[0, -1].float(),
        }
