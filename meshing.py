'''From the DeepSDF repository https://github.com/facebookresearch/DeepSDF
'''

from ast import IsNot
import logging
import numpy as np
import plyfile
from skimage.measure import marching_cubes
import time
import torch

from util import gradient, mean_curvature


def create_mesh(
    shapeNet,
    flowNet = None,
    filename="",
    t=-1000, #time=-1000 means we are only in the space
    N=256,
    max_batch=64 ** 3,
    offset=None,
    scale=None,
    device="cpu",
    silent=False
):
    shapeNet.eval()
    if flowNet is not None:
        flowNet.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    
    sdf_coord = 3
    if (t!=-1000):
        sdf_coord = 4

    # (x,y,z,sdf) if we are not considering time
    # (x,y,z,t,sdf) otherwise
    samples = torch.zeros(N ** 3, sdf_coord + 1, device=device)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]
    samples.requires_grad = False

    #adding the time
    if(t!=-1000):
        samples[:, sdf_coord-1] = t


    num_samples = N ** 3
    head = 0

    start = time.time()
    while head < num_samples:
        #print(head)
        sample_subset = samples[head:min(head + max_batch, num_samples), 0: sdf_coord]
        
        if flowNet is None:
            sdfs = shapeNet(sample_subset)["model_out"]
        else:
            sdfs = eval_composedNet(shapeNet, flowNet, sample_subset)
        
        samples[head:min(head + max_batch, num_samples), sdf_coord] = (
            sdfs
            .squeeze()
            .detach()
            .cpu()
        )
        head += max_batch

    sdf_values = samples[:, sdf_coord]
    sdf_values = sdf_values.reshape(N, N, N)

    end = time.time()
    if not silent:
        print(f"Sampling took: {end-start} s")

    verts, faces, normals, values = convert_sdf_samples_to_ply(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        offset,
        scale,
    )

    if filename:
        if not silent:
            print(f"Saving mesh to {filename}")

        save_ply(verts, faces, filename, shapeNet, flowNet, t)
        #save_ply_with_grad(verts, faces, filename, shapeNet, flowNet, t)

        if not silent:
            print("Done")

    return verts, faces, normals, values


def eval_composedNet(shapeNet, flowNet, coords_4d):
    coords_3d = flowNet(coords_4d)['model_out']
    sdfs = shapeNet(coords_3d)['model_out']
    return sdfs



def compute_textures(verts, flowNet, t):
    coords = torch.from_numpy(verts).float().cuda()
    times = t*torch.ones_like(coords[...,0].unsqueeze(-1))
    coords_4d = torch.cat((coords, times), dim = -1)

    flowNet_model_i = flowNet(coords_4d.unsqueeze(0))
    coords_3d = flowNet_model_i['model_out']

    textures = 100*coords_3d[...,0].unsqueeze(-1)%2

    return textures

def compute_curvatures(verts, shapeNet, flowNet, t):
    num_verts = verts.shape[0]
    coords = torch.from_numpy(verts).float().cuda()
    times = t*torch.ones_like(coords[...,0].unsqueeze(-1))
    coords_4d = torch.cat((coords, times), dim = -1)

    pred_curvature = []
    N = 200
    for i in range(N):
        coords_i = coords_4d[int(num_verts*i/N): int(num_verts*(i+1)/N),:]
        
        # model_output_i = decoder(coords_i.unsqueeze(0))
        # for the curvature of the initial surface
        #coords_3d_i = coords_i[...,0:3]

        # for the curvature of the deformed surfaces
        flowNet_model_i = flowNet(coords_i.unsqueeze(0))
        coords_3d_i = flowNet_model_i['model_out']
        #coords_4d_i = flowNet_model_i['model_in']

        #model_shapeNet = shapeNet(coords_3d_i, preserve_grad=True)
        model_shapeNet = shapeNet(coords_3d_i)
        model_output_i = model_shapeNet['model_out']
        model_input_i = model_shapeNet['model_in']
        
        pred_curvature_i = mean_curvature(model_output_i, model_input_i).squeeze(0).cpu().detach().numpy()
        #pred_curvature_i = mean_curvature(model_output_i, coords_4d_i).squeeze(0).cpu().detach().numpy()
        #pred_curvature_i = gradient(model_output_i, coords_4d_i)[...,0:3].squeeze(0).cpu().detach().numpy()
        if len(pred_curvature)==0:
            pred_curvature = pred_curvature_i
        else:
            pred_curvature = np.concatenate((pred_curvature, pred_curvature_i), axis=0)
    
    return pred_curvature


def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)

    # Check if the cubes contains the zero-level set
    level = 0.0
    if level < numpy_3d_sdf_tensor.min() or level > numpy_3d_sdf_tensor.max():
        print(f"Surface level must be within volume data range.")
    else:
        verts, faces, normals, values = marching_cubes(
            numpy_3d_sdf_tensor, level, spacing=[voxel_size] * 3
        )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    return mesh_points, faces, normals, values


def save_ply(verts, faces, filename, shapeNet, flowNet, t):
    # try writing to the ply file
    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros(
        (num_verts,),
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("quality", "f4")]
    )

    #curvatures = compute_curvatures(verts, shapeNet, flowNet, t)
    curvatures = compute_textures(verts, flowNet, t).squeeze(0).cpu().detach().numpy()
    
    for i in range(0, num_verts):
        verts_tuple[i] = tuple(verts[i, :]) + tuple(curvatures[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(
        faces_building,
        dtype=[("vertex_indices", "i4", (3,))]
    )

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    ply_data.write(filename)


def save_ply_with_grad(verts, faces, filename, shapeNet, flowNet, t):
    # try writing to the ply file
    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros(
        (num_verts,),
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4"), ("ny", "f4"), ("nz", "f4")]
    )

    grad = compute_curvatures(verts, shapeNet, flowNet, t)

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(verts[i, :]) + tuple(grad[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(
        faces_building,
        dtype=[("vertex_indices", "i4", (3,))]
    )

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    ply_data.write(filename)
