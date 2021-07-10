# coding: utf-8

from mesh_to_sdf import scale_to_unit_sphere
import trimesh
import torch
from torch.utils.data import Dataset
from util import sample_mesh


class PointCloud(Dataset):
    """SDF Point Cloud dataset.

    Parameters
    ----------
    mesh_path: str
        Path to the base mesh.

    samples_on_surface: int
        Number of surface samples to fetch (i.e. {X | f(X) = 0}).

    samples_off_surface: int
        Number of samples with f(X) != 0 to fetch.

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
    def __init__(self, mesh_path, samples_on_surface, samples_off_surface,
                 off_surface_sdf=None):
        super().__init__()

        self.samples_on_surface = samples_on_surface

        mesh = trimesh.load(mesh_path)
        mesh = scale_to_unit_sphere(mesh)
        self.point_cloud = torch.from_numpy(
            sample_mesh(mesh, samples_on_surface, samples_off_surface)
        )
        if off_surface_sdf is not None:
            self.point_cloud[(self.samples_on_surface + 1):, -1] = off_surface_sdf

        # Since we can't guarantee that the number of off-surface samples
        # requested will be respected, we set their number to the actual number
        # of samples obtained. We will fix the sampling algorithm in the future
        # to fix this.
        self.samples_off_surface = self.point_cloud.shape[0] - samples_on_surface

    def __len__(self):
        return self.samples_on_surface + self.samples_off_surface

    def __getitem__(self, idx):
        return {
            "coords": self.point_cloud[idx, :3].float(),
            "normals": self.point_cloud[idx, 3:6].float(),
            "sdf": self.point_cloud[idx, -1].float(),
        }
