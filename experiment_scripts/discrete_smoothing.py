
import os
import torch
from meshing import save_ply
from model import SIREN
import trimesh
import numpy as np
import igl
import time

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def cot_laplacian(mesh):
    vertices = mesh.vertices
    faces = mesh.faces
    L = igl.cotmatrix(vertices, faces)
    return L
    # vel = torch.FloatTensor(L.dot(vertices)).cuda()
    
    # vertices = vertices + vel.cpu().numpy()

    # mesh = trimesh.Trimesh(vertices, faces)
    # if not i%10 and i>0 :
    #     mesh.export(f'./discrete_smooth/mesh_smoth_{i}.ply')

path = "D:/Users/tiago/projects/i4d/i4d/logs/good_2_armadillo_meancurvature_10000epochs_i3dinit_smooth/reconstructions_check_17000/time_-1.0_meancurv_rep.ply"
# path ="./data/dumbbell.ply"
# path ="./data/armadillo.ply"
print(f"Loading mesh \"{path}\"")

mesh = trimesh.load(path)
NITERS = 100

times = [0] * NITERS
for i in range(NITERS):
    start_time = time.time()
    L = cot_laplacian(mesh)

    # normals = trimesh.smoothing.get_vertices_normals(mesh)
    # vertices = mesh.vertices.copy().view(np.ndarray)
    # kn = L.dot(vertices)

    # dots = kn[:,0]*normals[:,0]+kn[:,1]*normals[:,1]+kn[:,2]*normals[:,2]

    # curvs_abs = np.sqrt(kn[:,0]**2+kn[:,1]**2+kn[:,2]**2)

    # curvs = np.where(dots<0, -curvs_abs, curvs_abs )

    # verts = np.hstack((vertices, normals, curvs[:, None]))
    # faces = mesh.faces.copy().view(np.ndarray)

    # save_ply(
    #     verts, faces,
    #     f"./discrete_smothing/armadillo_curv{i}.ply",
    #     vertex_attributes=[("nx", "f4"), ("ny", "f4"), ("nz", "f4"), ("quality", "f4")]
    # )


    #mesh = trimesh.smoothing.filter_laplacian(mesh=mesh, iterations=100, lamb=100, laplacian_operator=L, volume_constraint=True, implicit_time_integration=True)
    mesh = trimesh.smoothing.filter_laplacian(mesh=mesh, iterations=1, lamb=0.5, laplacian_operator=L, volume_constraint=True, implicit_time_integration=True)#armadillo
    #mesh = trimesh.smoothing.filter_laplacian(mesh=mesh, iterations=10, lamb=0.8, volume_constraint=False, implicit_time_integration=True)
    times[i] = time.time() - start_time
    
times_np = np.array(times)
print(times_np.mean())
np.savetxt("armadillo_iteration_times.csv", times_np, delimiter=";")
