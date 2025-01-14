import torch
from torch import nn
from ..utils.colors.rgb import rgb_to_linear_rgb, linear_rgb_to_rgb


class GammaCorrection(nn.Module):

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return rgb_to_linear_rgb(image)


class DegammaCorrection(nn.Module):

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return linear_rgb_to_rgb(image)