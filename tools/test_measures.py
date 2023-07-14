
import os
import torch
from model import SIREN
from model_lipschitz_mlp import lipmlp
import trimesh
import numpy as np

from util import divergence, gradient, vector_dot

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


def source_vector_field(x, center = [0,0,0], spreads = [5,5,5] ):
        X = x[...,0].unsqueeze(-1)
        Y = x[...,1].unsqueeze(-1)
        Z = x[...,2].unsqueeze(-1)
        
        vx = X-center[0]
        vy = Y-center[1]
        vz = Z-center[2]

        gaussian = torch.exp(-(vx**2/(2*spreads[0]**2)+vy**2/(2*spreads[1]**2)+vz**2/(2*spreads[2]**2)))

        return gaussian*torch.cat((vx,vy,vz),dim=-1)

def vector_field(x):
    # center1 = [-0.4, 0.2, 0.0]
    # spreads1 = [0.2,0.2,0.2]
    center1 = [-0.5, 0.4, 0.0]
    center2 = [0.2, -0.2, 0.0]
    spreads1 = [0.3,0.3,0.3]

    # V = source_vector_field(x, center1, spreads1)
    V = source_vector_field(x, center1, spreads1) - source_vector_field(x, center2, spreads1)

    return V

def twist_vector_field(x):
    X = x[...,0].unsqueeze(-1)
    Y = x[...,1].unsqueeze(-1)+1
    Z = x[...,2].unsqueeze(-1)
    
    vx = - 2. * Y * Z
    vy = 0*Y
    vz =   2. * Y * X
    return torch.cat((vx,vy,vz),dim=-1)

def level_set_equation(grad, x):
    ft = grad[...,3].unsqueeze(-1)
    V = twist_vector_field(x)
    #V = vector_field(x)
    dot = vector_dot(grad[...,0:3], V)
    
    return (ft + dot)**2

def mean_curvature_equation(grad, x, scale = 0.00000001):
    ft = grad[...,3].unsqueeze(-1) # Partial derivative of the SIREN function f with respect to the time t
   
    grad = grad[...,0:3] # Gradient of the SIREN function f with respect to the space (x,y,z)
    grad_norm = torch.norm(grad, dim=-1).unsqueeze(-1)
    unit_grad = grad/grad_norm
    div = divergence(unit_grad, x)

    return torch.abs(ft - scale*grad_norm*div)


# model_i4d = SIREN(4, 1, [128,128,128],w0=20)
#model_i4d = lipmlp(4, 1, [256,256,256,256,256],w0=20)
model_i4d = lipmlp(4, 1, [512,512,512,512,512],w0=20)
# model_i4d = lipmlp(4, 1, [128,128],w0=16)
#model_i4d = SIREN(4, 1, [128,128,128],w0=20)
# model_i4d = SIREN(4, 1, [256,256,256], w0=36)
#model_i4d = lipmlp(4, 1, [256,256,256,256,256],w0=20)
#model_i4d.load_state_dict(torch.load('./logs/falcon_witch_2x128_w-20_good/models/falcon_witch_2x128_w-20_epoch_23000.pth'))
# model_i4d.load_state_dict(torch.load('./logs/dumbell_1x128_w0-16_mean_curv/models/model_50000.pth'))
#model_i4d.load_state_dict(torch.load('./logs/cube_1x180_w0-24_mean_curv/models/model_50000.pth'))
# model_i4d.load_state_dict(torch.load('./logs/spot_bob_1x128_w-20_lipchitz/models/model_100000.pth'))
#model_i4d.load_state_dict(torch.load('./logs/spot_bob_tanh_4x256_lipchitz_t_-0.1_0.1/models/model_100000.pth'))
# model_i4d.load_state_dict(torch.load('./logs/armadillo_2x256_w-36_mean_curv/models/model_10000.pth'))
# model_i4d.load_state_dict(torch.load('./logs/armadillo_2x256_w-30_twist_t=0_0.5/models/model_13000.pth'))
# model_i4d.load_state_dict(torch.load('./logs/spot_1x128_w0-20_vector_field/models/model_46000.pth'))
#model_i4d.load_state_dict(torch.load('./logs/falcon_smooth_witch_2x128_w-20_lipschitz/models/model_100000.pth'))
model_i4d.load_state_dict(torch.load('./logs/falcon_smooth_witch_tanh_4x512_t_-0.1_0.1/models/model_100000.pth'))
#model_i4d.load_state_dict(torch.load('./logs/spot_bob_1x128_w-30/models/model_final.pth'))

model_i4d.cuda()
model_i4d.eval()


# initial conditions
#model_i3d = SIREN(3, 1, [64, 64], w0=16)
# model_i3d = SIREN(3, 1, [256,256,256], w0=60)
model_i3d = SIREN(3, 1, [128,128,128], w0=30)
# model_i3d = SIREN(3, 1, [256,256,256], w0=60)
#model_i3d.load_state_dict(torch.load('shapeNets/spot_1x64_w0-16.pth'))
#model_i3d.load_state_dict(torch.load('shapeNets/bob_1x64_w0-16.pth'))
# model_i3d.load_state_dict(torch.load('shapeNets/armadillo_2x256_w-60.pth'))
model_i3d.load_state_dict(torch.load('shapeNets/falcon_2x128_w0-30.pth'))
# model_i3d.load_state_dict(torch.load('shapeNets/dumbbell_1x64_w0-16.pth'))
#model_i3d.load_state_dict(torch.load('shapeNets/cube_1x128_w0-24.pth'))
#model_i3d.load_state_dict(torch.load('shapeNets/witch_2x128_w0-30.pth'))

model_i3d.cuda()
model_i3d.eval()


samples_on_surface = 1000

# tmin, tmax = -1, 1
tmin=-0.1
path = "./data/falcon_smooth.ply"
#path = "./data/witch.ply"
print(f"Loading mesh \"{path}\"")

#sample of points on surface
on_surf_coords_4d_tmin = torch.zeros(samples_on_surface, 4).cuda()
# on_surf_coords_4d_tmax = torch.zeros(samples_on_surface, 4).cuda()

mesh = trimesh.load(path)

coords_on_surf = _sample_on_surface(
    mesh,
    samples_on_surface,
    sample_vertices=True
)
on_surf_coords_4d_tmin[..., :3] = coords_on_surf[..., :3]
# on_surf_coords_4d_tmax[..., :3] = coords_on_surf[..., :3]
on_surf_coords_4d_tmin[..., 3] = tmin
# on_surf_coords_4d_tmax[..., 3] = tmax

pred_values_on_surf = model_i4d(on_surf_coords_4d_tmin)['model_out']
gt_values_on_surf = model_i3d(on_surf_coords_4d_tmin[...,:3])['model_out']

diff_values_on_surf = torch.abs(pred_values_on_surf - gt_values_on_surf)
max_values_on_surf = torch.max(diff_values_on_surf)
mean_values_on_surf = torch.mean(diff_values_on_surf)
print(f"max_values_on_surf {max_values_on_surf}.")
print(f"mean_values_on_surf {mean_values_on_surf}.")

samples_off_surface = samples_on_surface
samples_on_space = samples_on_surface


#sample of points off surface
off_surface_points = torch.from_numpy(np.random.uniform(-1, 1, size=(samples_off_surface, 3)))

off_surf_coords_4d_tmin = torch.zeros(samples_on_surface, 4)
# off_surf_coords_4d_tmax = torch.zeros(samples_on_surface, 4)

off_surf_coords_4d_tmin[..., :3] = off_surface_points[..., :3]
# off_surf_coords_4d_tmax[..., :3] = off_surface_points[..., :3]
off_surf_coords_4d_tmin[..., 3] = tmin
# off_surf_coords_4d_tmax[..., 3] = tmax

pred_values_off_surf = model_i4d(off_surf_coords_4d_tmin.cuda())['model_out']
gt_values_off_surf = model_i3d(off_surf_coords_4d_tmin[...,:3].cuda())['model_out']

diff_values_off_surf = torch.abs(pred_values_off_surf - gt_values_off_surf)
max_values_off_surf = torch.max(diff_values_off_surf)
mean_values_off_surf = torch.mean(diff_values_off_surf)
print(f"max_values_off_surf {max_values_off_surf}.")
print(f"mean_values_off_surf {mean_values_off_surf}.")

# #sample of points on the 4d space
# # space_points = np.random.uniform(tmin, tmax, size=(samples_on_space, 4))
# space_coords = torch.from_numpy(np.random.uniform(-1.0, 1.0, size=(samples_on_space, 3))).float()
# space_time = torch.from_numpy(np.random.uniform(-0.8, 0.0, size=(samples_on_space))).float()
# space_points = torch.zeros(samples_on_surface, 4)
# space_points[...,0:3] = space_coords
# space_points[...,3] = space_time

# model_values_space = model_i4d(space_points.cuda())
# pred_coords_space = model_values_space['model_in']
# pred_values_space = model_values_space['model_out']

# grad = gradient(pred_values_space, pred_coords_space)
# # level_set_value = level_set_equation(grad, pred_coords_space)
# level_set_value = mean_curvature_equation(grad, pred_coords_space, scale=0.025)


# max_level_set_value = torch.max(level_set_value)
# mean_level_set_value = torch.mean(level_set_value)
# print(f"max_level_set_value {max_level_set_value}.")
# print(f"mean_level_set_value {mean_level_set_value}.")