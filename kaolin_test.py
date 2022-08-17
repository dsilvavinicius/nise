
import os
import torch
from model import SIREN
from model_lipschitz_mlp import lipmlp
from meshing import create_mesh
from util import create_output_paths
import kaolin

#model_i4d = SIREN(4, 1, [128,128,128],w0=20)
# model_i4d = lipmlp(4, 1, [128,128,128],w0=20)
model_i4d = lipmlp(4, 1, [128,128],w0=20)
# model_i4d.load_state_dict(torch.load('./logs/falcon_witch_2x128_w-20/models/model_20000.pth'))
model_i4d.load_state_dict(torch.load('./logs/spot_bob_1x128_w-20_lipchitz/models/model_100000.pth'))
#model_i4d.load_state_dict(torch.load('./logs/falcon_smooth_witch_2x128_w-20_lipschitz/models/model_100000.pth'))
model_i4d.cuda()
model_i4d.eval()

full_path = create_output_paths(
    'logs',
    'spot_bob_1x128_w-20_lipchitz',
    overwrite=False
)

timelapse = kaolin.visualize.Timelapse(os.path.join(full_path, "kaolin"))

# number of samples of the interval time
T = [-1,-0.75,-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1]
#T = [-0.2,-0.1,0.0,0.1,0.2]
for i in range(len(T)):
    mesh_file = f"time_{T[i]}.ply"
    verts, faces, normals, _ = create_mesh(
        model_i4d,
        filename=os.path.join(full_path, "reconstructions", mesh_file), 
        t=T[i],  # time instant for 4d SIREN function
        N=400,
        device='cuda'
    )

    tensor_faces = torch.from_numpy(faces.copy())
    tensor_verts = torch.from_numpy(verts.copy())
    timelapse.add_mesh_batch(category=f"time_{i}", iteration=1, faces_list=[tensor_faces], vertices_list=[tensor_verts],)