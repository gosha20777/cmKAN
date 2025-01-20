import torch
import torch.nn as nn
from cm_kan.ml.layers import CmKANLayer, LightCmKANLayer
from cm_kan.core import Logger


class CmKAN(torch.nn.Module):
    """ Input features BxCxN """

    def __init__(self, in_dims, out_dims, grid_size, spline_order, residual_std, grid_range):
        super(CmKAN, self).__init__()

        Logger.info(f"CmKAN: in_dims={in_dims}, out_dims={out_dims}")

        cm_kan_size = [s for s in zip(in_dims, out_dims)]

        self.layers = []
        for in_dim, out_dim in cm_kan_size:
            self.layers.append(
                CmKANLayer(in_channels=in_dim,
                         out_channels=out_dim,
                         grid_size=grid_size,
                         spline_order=spline_order,
                         residual_std=residual_std,
                         grid_range=grid_range))

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class LightCmKAN(torch.nn.Module):
    """ Input features BxCxN """

    def __init__(self, in_dims, out_dims, grid_size, spline_order, residual_std, grid_range):
        super(LightCmKAN, self).__init__()

        Logger.info(f"LightCmKAN: in_dims={in_dims}, out_dims={out_dims}")

        cm_kan_size = [s for s in zip(in_dims, out_dims)]

        self.layers = []
        for in_dim, out_dim in cm_kan_size:
            self.layers.append(
                LightCmKANLayer(in_channels=in_dim,
                         out_channels=out_dim,
                         grid_size=grid_size,
                         spline_order=spline_order,
                         residual_std=residual_std,
                         grid_range=grid_range))

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
