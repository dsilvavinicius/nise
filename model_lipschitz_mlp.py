import torch
from torch import nn

class lipmlp(nn.Module):

    def __init__(self, n_in_features, n_out_features, hidden_layer_config=[],
                 w0=30):
        super().__init__()
        
        def init_W(size_out, size_in): 
            W = torch.randn(size_out, size_in) * torch.sqrt(torch.Tensor([2 / size_in]))
            return W
        
        self.w0 = w0
        sizes = hidden_layer_config
        sizes.insert(0, n_in_features)
        sizes.append(n_out_features)
        self.num_layers = len(sizes)
        self.params_W = []
        self.params_b = []
        self.params_c = []
        #net = []
        for ii in range(len(sizes)-1):
            W = torch.nn.Parameter(init_W(sizes[ii+1], sizes[ii]))
            b = torch.nn.Parameter(torch.zeros(sizes[ii+1]))
            c = torch.nn.Parameter(torch.max(torch.sum(torch.abs(W), axis=1)))
            self.params_W.append(W)
            self.params_b.append(b)
            self.params_c.append(c)
        
        self.params_W = nn.ParameterList(self.params_W)
        self.params_b = nn.ParameterList(self.params_b)
        self.params_c = nn.ParameterList(self.params_c)
    
    def weight_normalization(self, W, softplus_c):
        """
        Lipschitz weight normalization based on the L-infinity norm
        """
        absrowsum = torch.sum(torch.abs(W), axis=1)
        scale = torch.minimum(torch.Tensor([1.0]).cuda(), softplus_c/absrowsum)
        return W * scale[:,None]

    def get_lipschitz_loss(self):
        loss_lip = 1.0
        for ii in range(len(self.params_c)):
            c = self.params_c[ii]
            # loss_lip = loss_lip * nn.Softplus()(c)
            loss_lip = loss_lip * nn.Softplus()(c)
        return loss_lip

    def forward(self, x):
     # forward pass
        
        coords_org = x.clone().detach().requires_grad_(True)
        coords = coords_org
        for ii in range(len(self.params_W) - 1):
            #W, b, c = self.params_net[ii]
            W = self.params_W[ii]
            b = self.params_b[ii]
            c = self.params_c[ii]
            W = self.weight_normalization(W, nn.Softplus()(c))
            coords = nn.Tanh()(torch.matmul(coords,W.T) + b)
            # coords = nn.ReLU()(torch.matmul(coords,W.T) + b)
            #coords = nn.ELU()(torch.matmul(coords,W.T) + b)

        # final layer
        # W, b, c = self.params_net[-1]
        W = self.params_W[-1]
        b = self.params_b[-1]
        c = self.params_c[-1]
        W = self.weight_normalization(W, nn.Softplus()(c)) 
        out = torch.matmul(coords, W.T) + b
        return {"model_in": coords_org, "model_out": out}
    