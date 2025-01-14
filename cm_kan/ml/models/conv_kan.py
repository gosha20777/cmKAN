import torch
from torch import nn
from typing import List
from ..layers import KANLayer
from ..utils.process import (
    feature_to_colors,
    colors_to_feature
)
from ..layers.kan_conv.kan_convs import KANConv2DLayer
from ..layers.mw_isp import DWTForward, RCAGroup, DWTInverse, seq


'''
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
        self.down = nn.Sequential(
            nn.Conv2d(
                in_channels*1,
                in_channels*2,
                3,
                2,
                padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels*2,
                in_channels*4,
                3,
                2,
                padding=1
            ),
            nn.ReLU(inplace=True)
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                out_channels*4,
                out_channels*2,
                2,
                2
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                out_channels*2,
                out_channels*1,
                2,
                2
            ),
            nn.ReLU(inplace=True)
        )
        self.norm = torch.nn.BatchNorm2d(in_channels*4)
        self.kan = KANLayer(
            in_dim=in_channels*4,
            out_dim=out_channels*4,
            grid_size=grid_size,
            spline_order=spline_order,
            residual_std=residual_std,
            grid_range=grid_range,
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass sequentially across each layer.
        """
        x = self.down(x)
        x, shape = feature_to_colors(x)
        x = self.kan(x)
        #x = self.sigmoid(x)
        x = colors_to_feature(x, shape)
        x = self.up(x)
        return x

'''


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
        
        c1 = 16
        n_b = 4
        
        self.head = DWTForward()

        self.down1 = seq(
            nn.Conv2d(in_channels * 4, c1, 3, 1, 1),
            nn.PReLU(),
            RCAGroup(in_channels=c1, out_channels=c1, nb=n_b)
        )

        self.up1 = seq(
            RCAGroup(in_channels=c1, out_channels=c1, nb=n_b),
            nn.Conv2d(c1, out_channels*4, 3, 1, 1)
        )

        self.tail = seq(
            DWTInverse(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.PReLU(),
        )

        self.kan = KANLayer(
            in_dim=c1,
            out_dim=c1,
            grid_size=grid_size,
            spline_order=spline_order,
            residual_std=residual_std,
            grid_range=grid_range,
        )
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass sequentially across each layer.
        """
        c1 = self.head(x)
        m = self.down1(c1)

        m, shape = feature_to_colors(m)
        m = self.kan(m)
        m = colors_to_feature(m, shape)

        u1 = self.up1(m) + c1
        out = self.tail(u1)

        return out
    

'''
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
            output_dim=out_channels*2,
            kernel_size=3,
            spline_order=spline_order,
            groups=1,
            padding=1,
            stride=1,
            grid_size=grid_size,
            grid_range=grid_range
        )
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = KANConv2DLayer(
            input_dim=out_channels*2,
            output_dim=out_channels,
            kernel_size=3,
            spline_order=spline_order,
            groups=1,
            padding=1,
            stride=1,
            grid_size=grid_size,
            grid_range=grid_range
        )
        self.relu2 = nn.ReLU(inplace=True)
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass sequentially across each layer.
        """
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x

'''