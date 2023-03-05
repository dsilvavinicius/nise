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
            self.net[0].apply(first_layer_sine_init)
            self.net[1:].apply(lambda module: sine_init(module, self.ww))

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

    def from_pretrained_initial_condition(self, other: OrderedDict):
        """Application of the neural network initialization 

        This method assumes that the network defined by `self` is as deep as
        `other`, and each layer is at least as wide as `other` as well. If the
        depth and `self, or the width of `other` is larger than `self`, the
        method will abort.

        Let us consider `other`'s weights to be `B`, while `self`'s weights are
        `A`. Our initialization assigns:
        $A_1 = ( B_1 0; f1 f2 )$ and
        $A_i = ( B_i 0; 0  0  ), i > 1$
        
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
            The model with the initialization described in [1] applied, or no
            transformation at all if `other` is not a valid state_dict.
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
        my_sd[keys[-2]].uniform_(-np.sqrt(6 / 4) / self.ww, np.sqrt(6 / 4) / self.ww)
        my_sd[keys[-2]][:, :llw] = other[keys[-2]]

        # Handling the intermediate layer weights.
        for k in keys[2:-2:2]:
            hh, hw = other[k].shape
            z = torch.zeros_like(my_sd[k])
            z[:hh, :hw] = other[k]
            my_sd[k] = z

        # Handling the layers biases.
        for k in keys[1:-2:2]:
            ll = other[k].shape[0]
            z = torch.zeros_like(my_sd[k])
            z[:ll] = other[k]
            my_sd[k] = z

        self.load_state_dict(my_sd)
        return self

