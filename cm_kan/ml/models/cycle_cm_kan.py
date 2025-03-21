import torch
import torch.nn as nn
from cm_kan.core import Logger
from .cm_kan import CmKAN, LightCmKAN
from ..layers import PatchDiscriminator


class CycleCmKAN(torch.nn.Module):
    def __init__(self, in_dims, out_dims, grid_size, spline_order, residual_std, grid_range):
        super(CycleCmKAN, self).__init__()

        Logger.info(f"CycleCmKAN: in_dims={in_dims}, out_dims={out_dims}")

        self.gen_ab = CmKAN(
            in_dims=in_dims,
            out_dims=out_dims,
            grid_size=grid_size,
            spline_order=spline_order,
            residual_std=residual_std,
            grid_range=grid_range,
        )
        self.gen_ba = CmKAN(
            in_dims=out_dims,
            out_dims=in_dims,
            grid_size=grid_size,
            spline_order=spline_order,
            residual_std=residual_std,
            grid_range=grid_range,
        )
        self.dis_a = PatchDiscriminator(in_dim=in_dims[0])
        self.dis_b = PatchDiscriminator(in_dim=out_dims[0])
