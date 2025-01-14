import torch
from torch import nn
from typing import List
from ..layers.kan_conv.kan_convs import KANConv2DLayer


class ConvKanModel(nn.Module):
    """
    Standard architecture for Kolmogorov-Arnold Networks described in the original paper.
    Layers are defined via a list of layer widths.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        grid_size: int,
        spline_order: int,
        residual_std: float = 0.1,
        grid_range: List[float] = [-1, 1],
    ) -> None:
        super(ConvKanModel, self).__init__()
        self.conv1 = KANConv2DLayer(
            input_dim=in_channels,
            output_dim=out_channels,
            spline_order=spline_order,
            groups=1,
            padding=1,
            stride=1,
            grid_size=grid_size,
            grid_range=grid_range
        )
        self.conv2 = KANConv2DLayer(
            input_dim=in_channels,
            output_dim=out_channels,
            spline_order=spline_order,
            groups=1,
            padding=1,
            stride=1,
            grid_size=grid_size,
            grid_range=grid_range
        )
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass sequentially across each layer.
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x
