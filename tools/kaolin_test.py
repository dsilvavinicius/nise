
import os
import torch
from model import SIREN
from model_lipschitz_mlp import lipmlp
from meshing import create_mesh
from util import create_output_paths
import kaolin

#model_i4d = SIREN(4, 1, [128,128,128],w0=20)
# model_i4d = lipmlp(4, 1, [128,128,128],w0=20)
#model_i4d = lipmlp(4, 1, [128,128],w0=20)
#model_i4d = SIREN(4, 1, [128,128],w0=20)
model_i4d = SIREN(4, 1, [256,256,256],w0=30)
#model_i4d = lipmlp(4, 1, [256,256,256,256,256],w0=20)
# model_i4d.load_state_dict(torch.load('./logs/falcon_witch_2x128_w-20/models/model_20000.pth'))
# model_i4d.load_state_dict(torch.load('./logs/spot_bob_1x128_w-20_lipchitz/models/model_100000.pth'))
#model_i4d.load_state_dict(torch.load('./logs/spot_bob_tanh_4x256_lipchitz_t_-0.1_0.1/models/model_100000.pth'))
#model_i4d.load_state_dict(torch.load('./logs/spot_1x128_w0-20_vector_field/models/model_46000.pth'))
# model_i4d.load_state_dict(torch.load('./logs/armadillo_2x256_w-36_mean_curv/models/model_10000.pth'))
model_i4d.load_state_dict(torch.load('./logs/bunny_2x256_w-30_mean_curv_init_i3d/models/model_500.pth'))
#model_i4d.load_state_dict(torch.load('./logs/falcon_smooth_witch_2x128_w-20_lipschitz/models/model_100000.pth'))
model_i4d.cuda()
model_i4d.eval()

full_path = create_output_paths(
    'logs',
    'bunny_2x256_w-30_mean_curv_init_i3d',
    overwrite=False
)

timelapse = kaolin.visualize.Timelapse(os.path.join(full_path, "kaolin"))

# number of samples of the interval time
#T = [-1,-0.75,-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1]
#T = [-1,-0.75,-0.5, -0.25, 0.0, 0.25, 0.5, 0.75,0.95, 1]
#T = [-0.2,-0.1,0.0,0.1,0.2]
#T = [-0.8,-0.75,-0.7,-0.65,-0.6]
T = [-0.2,0.0,0.2,0.4,0.6]
#T = [-0.1,-0.05,0.0,0.05,0.09,0.1]
for i in range(len(T)):
    mesh_file = f"time_{T[i]}.ply"
    verts, faces, normals, _ = create_mesh(
        model_i4d,
        filename=os.path.join(full_path, "reconstructions", mesh_file), 
        t=T[i],  # time instant for 4d SIREN function
        N=512,
        device='cuda'
    )

    tensor_faces = torch.from_numpy(faces.copy())
    tensor_verts = torch.from_numpy(verts.copy())
    timelapse.add_mesh_batch(category=f"sharpenin_time_{i}", iteration=1, faces_list=[tensor_faces], vertices_list=[tensor_verts],)