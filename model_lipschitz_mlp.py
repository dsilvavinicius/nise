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
        self.params_net_c = []    
        net = []
        for ii in range(len(sizes)-1):
            W = init_W(sizes[ii+1], sizes[ii])
            b = torch.zeros(sizes[ii+1])
            c = torch.max(torch.sum(torch.abs(W), axis=1))
            self.params_net_c.append(c)
            l = nn.Linear(sizes[ii], sizes[ii+1])
            l.weight.data = W
            l.bias.data = b
            net.append(l)
            if ii < len(sizes)-2:
                 net.append(nn.Softplus())
        #return params_net
        
        self.net = nn.Sequential(*net)
    
    def weight_normalization(self, W, softplus_c):
        """
        Lipschitz weight normalization based on the L-infinity norm
        """
        absrowsum = torch.sum(torch.abs(W), axis=1)
        scale = torch.minimum(torch.Tensor([1.0]).cuda(), softplus_c/absrowsum)
        return W * scale[:,None]

    def forward(self, x):
        
        for ii in range(self.num_layers-1):
            if isinstance(self.net[ii], nn.Linear):
                W = self.net[ii].weight.data
                c = self.params_net_c[ii]
                W = self.weight_normalization(W, nn.Softplus()(c))
                self.net[ii].weight.data = W
        
        W = self.net[-1].weight.data
        c = self.params_net_c[-1]
        W = self.weight_normalization(W, nn.Softplus()(c))
        self.net[-1].weight.data = W

        coords_org = x.clone().detach().requires_grad_(True)
        coords = coords_org
        y = self.net(coords)
        return {"model_in": coords_org, "model_out": y}

  