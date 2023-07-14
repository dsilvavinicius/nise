# coding: utf-8

from collections import OrderedDict
import torch
from torch import nn
import numpy as np


@torch.no_grad()
def sine_init(m, w0):
    if hasattr(m, 'weight'):
        num_input = m.weight.size(-1)
        m.weight.uniform_(-np.sqrt(6 / num_input) / w0,
                          np.sqrt(6 / num_input) / w0)


@torch.no_grad()
def first_layer_sine_init(m):
    if hasattr(m, 'weight'):
        num_input = m.weight.size(-1)
        m.weight.uniform_(-1 / num_input, 1 / num_input)


class SineLayer(nn.Module):
    """A Sine non-linearity layer.
    """
    def __init__(self, w0=30):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)

    def __repr__(self):
        return f"SineLayer(w0={self.w0})"


class SIREN(nn.Module):
    """SIREN Module

    Parameters
    ----------
    n_in_features: int
        Number of input features.

    n_out_features: int
        Number of output features.

    hidden_layer_config: list[int], optional
        Number of neurons at each hidden layer of the network. The model will
        have `len(hidden_layer_config)` hidden layers. Only used in during
        model training. Default value is None.

    w0: number, optional
        Frequency multiplier for the Sine layers. Only useful for training the
        model. Default value is 30, as per [1].

    ww: number, optional
        Frequency multiplier for the hidden Sine layers. Only useful for
        training the model. Default value is None.

    delay_init: boolean, optional
        Indicates if we should perform the weight initialization or not.
        Default value is False, meaning that we perform the weight
        initialization as usual. This is useful if we will load the weights of
        a pre-trained network, in this case, initializing the weights does not
        make sense, since they will be overwritten.

    References
    ----------
    [1] Sitzmann, V., Martel, J. N. P., Bergman, A. W., Lindell, D. B.,
    & Wetzstein, G. (2020). Implicit Neural Representations with Periodic
    Activation Functions. ArXiv. http://arxiv.org/abs/2006.09661
    """
    def __init__(self, n_in_features, n_out_features, hidden_layer_config=[],
                 w0=30, ww=None, delay_init=False):
        super().__init__()
        self.w0 = w0
        if ww is None:
            self.ww = w0
        else:
            self.ww = ww

        net = []
        net.append(nn.Sequential(
            nn.Linear(n_in_features, hidden_layer_config[0]),
            SineLayer(self.w0)
        ))

        for i in range(1, len(hidden_layer_config)):
            net.append(nn.Sequential(
                nn.Linear(hidden_layer_config[i-1], hidden_layer_config[i]),
                SineLayer(self.ww)
            ))

        net.append(nn.Sequential(
            nn.Linear(hidden_layer_config[-1], n_out_features),
        ))

        self.net = nn.Sequential(*net)
        if not delay_init:
            self.reset_weights()

    def forward(self, x):
        """Forward pass of the model.

        Parameters
        ----------
        x: torch.Tensor
            The model input containing of size Nx3

        Returns
        -------
        dict
            Dictionary of tensors with the input coordinates under 'model_in'
            and the model output under 'model_out'.
        """
        # Enables us to compute gradients w.r.t. coordinates
        coords_org = x.clone().detach().requires_grad_(True)
        coords = coords_org
        y = self.net(coords)
        return {"model_in": coords_org, "model_out": y}

    def reset_weights(self):
        self.net[0].apply(first_layer_sine_init)
        self.net[1:].apply(lambda module: sine_init(module, self.ww))

    def from_pretrained_initial_condition(self, other: OrderedDict):
        """Neural network initialization given a pretrained network.

        This method assumes that the network defined by `self` is as deep as
        `other`, and each layer is at least as wide as `other` as well. If the
        depth and `self, or the width of `other` is larger than `self`, the
        method will abort.

        Let us consider `other`'s weights to be `B`, while `self`'s weights are
        `A`. Our initialization assigns:
        $A_1 = ( B_1 0; f1 f2 )$ and
        $A_i = ( B_i 0; 0  0  ), i > 1$

        where $f1$ and $f2$ are weight values initilized as proposed by [1].

        The biases are defined as the column
        vector:
        $ a_i = ( b_i 0 )^T $

        Parameters
        ----------
        other: OrderedDict
            The state_dict to use as reference. Note that it must have the same
            depth as `model.state_dict()`, and the first layer weights of this
            state_dict must have shape [N, D-1], where `D` is the number of
            columns of model.state_dict()["net.0.0.weight"].

        Returns
        -------
        model: niif.model.SIREN
            The initialized model

        References
        ----------
        [1] Sitzmann, V., Martel, J. N. P., Bergman, A. W., Lindell, D. B.,
        & Wetzstein, G. (2020). Implicit Neural Representations with Periodic
        Activation Functions. ArXiv. http://arxiv.org/abs/2006.09661
        """
        depth_other = len(other) // 2
        depth_us = len(self.state_dict()) // 2
        if depth_other != depth_us:
            raise ValueError("Number of layers does not match.")

        try:
            first_layer = other["net.0.0.weight"]
        except KeyError:
            raise ValueError("Invalid state_dict provided."
                             " Convert it to v2 first")

        if first_layer.shape[1] != self.net[0][0].weight.shape[1] - 1:
            raise ValueError(
                "Invalid first-layer size on the reference weights."
                f" Is {first_layer.shape[1]}, should be"
                f" {self.net[0][0].weights.shape[1] - 1}."
            )

        my_sd = self.state_dict()
        for k, v in my_sd.items():
            if v.shape[0] < other[k].shape[0]:
                raise AttributeError(
                    "The input layer has more rows than ours. Ensure that they"
                    " either match, or ours is taller than the input."
                )
            if v.ndim > 1:
                if v.shape[1] < other[k].shape[1]:
                    raise AttributeError(
                        "The input layer has more columns than ours. Ensure"
                        " that they either match, or ours is wider than the"
                        " input."
                    )

        # Appending new column to input weights (all zeroes)
        new_first_layer = torch.cat((
            first_layer, torch.zeros_like(first_layer[..., 0].unsqueeze(-1))
        ), dim=-1)

        flh, flw = new_first_layer.shape
        keys = list(my_sd.keys())
        # Ensuring that the input layer weights are all zeroes.
        # my_sd[keys[0]] = torch.zeros_like(my_sd[keys[0]])
        my_sd[keys[0]].uniform_(-1/4, 1/4)
        my_sd[keys[0]][:flh, :flw] = new_first_layer

        # Solving the last layer weights.
        # State dict keys are interleaved: weights, biases, weights, biases,...
        # The second-to-last key is the last weights tensor.
        llw = other[keys[-2]].shape[1]
        # my_sd[keys[-2]] = torch.zeros_like(my_sd[keys[-2]])
        outlayer_w = my_sd[keys[-2]].size(-1)
        m1 = np.sqrt(6.0 / outlayer_w) / self.ww
        my_sd[keys[-2]].uniform_(-m1, m1)
        my_sd[keys[-2]][:, :llw] = other[keys[-2]]

        # Handling the intermediate layer weights.
        for k in keys[2:-2:2]:
            hh, hw = other[k].shape
            z = torch.zeros_like(my_sd[k])
            z[:hh, :hw] = other[k]
            my_sd[k] = z

        # Handling the layers biases.
        for k in keys[1::2]:
            ll = other[k].shape[0]
            z = torch.zeros_like(my_sd[k])
            z[:ll] = other[k]
            my_sd[k] = z

        self.load_state_dict(my_sd)
        return self


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
            coords = nn.Tanh()(torch.matmul(coords, W.T) + b)
            # coords = nn.ReLU()(torch.matmul(coords,W.T) + b)
            # coords = nn.ELU()(torch.matmul(coords,W.T) + b)

        # final layer
        # W, b, c = self.params_net[-1]
        W = self.params_W[-1]
        b = self.params_b[-1]
        c = self.params_c[-1]
        W = self.weight_normalization(W, nn.Softplus()(c))
        out = torch.matmul(coords, W.T) + b
        return {"model_in": coords_org, "model_out": out}
