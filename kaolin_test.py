
import os
import torch
from model import SIREN
from meshing import create_mesh
from util import create_output_paths
import kaolin

model_i4d = SIREN(4, 1, [256,256,256],w0=30)
model_i4d.load_state_dict(torch.load('./logs/bunny_2x256_w-30_mean_curv/models/model_17000.pth'))
model_i4d.cuda()
model_i4d.eval()

full_path = create_output_paths(
    'logs',
    'bunny_2x256_w-30_mean_curv',
    overwrite=False
)

timelapse = kaolin.visualize.Timelapse(os.path.join(full_path, "kaolin"))

# number of samples of the interval time
T = [-1,-0.75,-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1]
for i in range(len(T)):
    mesh_file = f"time_{T[i]}.ply"
    verts, faces, normals, _ = create_mesh(
        model_i4d,
        filename=os.path.join(full_path, "reconstructions", mesh_file), 
        t=T[i],  # time instant for 4d SIREN function
        N=128,
        device='cuda'
    )

    tensor_faces = torch.from_numpy(faces.copy())
    tensor_verts = torch.from_numpy(verts.copy())
    timelapse.add_mesh_batch(category=f"output_{i}", iteration=1, faces_list=[tensor_faces], vertices_list=[tensor_verts])