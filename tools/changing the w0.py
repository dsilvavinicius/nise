
import os
import torch
from model import SIREN
from model_lipschitz_mlp import lipmlp
from meshing import create_mesh
from util import create_output_paths

#model_i4d = SIREN(4, 1, [64,64],w0=30)
#model_i4d.load_state_dict()
#model_i4d.cuda()
#model_i4d.eval()


state_dict = torch.load('./logs/falcon_witch_64x1_w0_30_t_-0.2_0.2/models/falcon_witch_64x1_w0_30_t_-0.2_0.2.pth')

print(state_dict['net.0.0.weight'])
state_dict['net.0.0.weight'] *= 3/2
state_dict['net.0.0.bias'] *= 3/2
print(state_dict['net.1.0.weight'])
state_dict['net.1.0.weight'] *= 3/2
state_dict['net.1.0.bias'] *= 3/2
# print(state_dict['net.2.0.weight'])
# state_dict['net.2.0.weight'] *= 3/2

model_i4d = SIREN(4, 1, [64,64],w0=20)
model_i4d.load_state_dict(state_dict)

torch.save(model_i4d.state_dict(), './logs/falcon_witch_64x1_w0_30_t_-0.2_0.2/models/falcon_witch_64x1_w0_20_t_-0.2_0.2.pth')