# coding: utf-8

import os
import shutil
from warnings import warn
import numpy as np
import torch
from mesh_to_sdf import get_surface_point_cloud


def create_output_paths(checkpoint_path, experiment_name, overwrite=True):
    """Helper function to create the output folders. Returns the resulting path.
    """
    full_path = os.path.join(".", checkpoint_path, experiment_name)
    if os.path.exists(full_path) and overwrite:
        shutil.rmtree(full_path)
    elif os.path.exists(full_path):
        warn("Output path exists. Not overwritting.")
        return full_path
    os.makedirs(full_path)
    return full_path


def sample_mesh(mesh, points_on_surface: int, points_off_surface: int,
                sample_vertices=True, perc_in_out_surface=(50, 50)):
    """Samples points from a mesh.

    Parameters
    ----------
    mesh: trimesh.Mesh

    points_on_surface: int
        Number of points to sample on the surface

    points_off_surface: int
        Number of points to sample off-surface (inside and out)

    sample_vertices: boolean, optional
        Whether the samples on surface will be drawn from the mesh vertices
        (True) or from its faces (False). Default is True

    perc_in_out_surface: tuple(int, int), optional, unused
        The percentage of samples to be drawn from inside and outside the mesh.
        Only affects the off-surface samples, i.e. The `points_on_surface`
        parameter is not affected by this.
    """
    full_samples = None
    if points_on_surface is not None and points_on_surface:
        if sample_vertices:
            idx = np.random.choice(
                np.arange(start=0, stop=len(mesh.vertices)),
                size=points_on_surface,
                replace=False
            )
            on_points = mesh.vertices[idx]
            on_normals = mesh.vertex_normals[idx]
        else:
            on_points, face_idx = mesh.sample(
                count=points_on_surface,
                return_index=True
            )
            on_normals = mesh.face_normals[face_idx]

        full_samples = np.hstack((
            on_points,
            on_normals,
            np.zeros((points_on_surface, 1))
        )).astype(np.float32)

    if points_off_surface:
        point_cloud = get_surface_point_cloud(
            mesh,
            surface_point_method="scan",
            bounding_radius=1,
            calculate_normals=True
        )
        domain_points = np.random.uniform(-1, 1, size=(points_off_surface, 3))
        domain_sdf, domain_normals = point_cloud.get_sdf(
            domain_points,
            use_depth_buffer=False,
            return_gradients=True
        )
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

        domain_samples = np.hstack((
            domain_points,
            domain_normals,
            domain_sdf[:, np.newaxis]
        ))
        if full_samples is None:
            full_samples = domain_samples
        else:
            full_samples = np.vstack((full_samples, domain_samples))

    return full_samples


def gradient(y, x, grad_outputs=None):
    """Gradient of `y` with respect to `x`
    """
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(
        y,
        [x],
        grad_outputs=grad_outputs,
        create_graph=True
    )[0]
    return grad
