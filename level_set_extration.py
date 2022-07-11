import os
import torch
from model import SIREN
from meshing import create_mesh

model = SIREN(4, 1, [256, 256, 256], w0=30)
model.load_state_dict(torch.load('D:\\Users\\tiago\\projects\\i4d\\i4d\\logs\\armadillo_2x256_w-30_twist\\models\\model_50000.pth'))
model.eval()
model.to('cuda')

# w0=model.w0
# l1=w0*model.net[0][0].weight
# print(l1)
# print(torch.max(torch.abs(l1)))
# l2=w0*model.net[1][0].weight
# print(l2)
# print(torch.max(torch.abs(l2)))
# l3=w0*model.net[2][0].weight
# print(l3)
# print(torch.max(torch.abs(l3)))

T=[-1, -0.8, -0.6, -0.3, 0.0, -0.3, 0.6, 0.8, 1]
for i in range(len(T)):
    mesh_file = f"twist_time_{T[i]}.ply"
    verts, _, normals, _ = create_mesh(
        model,
        filename=os.path.join("temp_reconstructions", mesh_file), 
        t=T[i],  # time instant for 4d SIREN function
        N=512,
        device='cuda'
    )
    