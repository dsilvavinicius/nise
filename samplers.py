# coding: utf-8

import torch
from torch.utils.data import Sampler


class SitzmannSampler(Sampler):
    """An implementation of the sampling approach used by Sitzmann et al. for
    their SDF experiments on [1].

    Parameters
    ----------
    data_source: PointCloud

    off_surface_points: integer

    References
    ----------
    [1] Sitzmann, V., Martel, J. N. P., Bergman, A. W., Lindell, D. B., &
    Wetzstein, G. (2020). Implicit Neural Representations with Periodic
    Activation Functions. ArXiv. Retrieved from http://arxiv.org/abs/2006.09661
    """
    def __init__(self, data_source, off_surface_points):
        self.data_source = data_source
        self.on_surface_points = data_source.samples_on_surface
        self.off_surface_points = off_surface_points

    def __iter__(self):
        total_points = self.on_surface_points + self.off_surface_points
        yield from torch.randperm(total_points).tolist()

    def __len__(self):
        return len(self.data_source) + self.off_surface_points
