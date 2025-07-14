import torch
from typing import Tuple, List
from .io import read_numpy_feature


def feature_to_colors(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Size]:
    """
    Transform 3D feature or 4D image to 2D colors.

    For feature [b,feat,c] -> [b*feat,c].
    For feature [b,c,h,w] -> [b*h*w,c].
    """
    shape = x.size()

    if len(shape) == 4:
        b, c, h, w = shape
        x = x.permute(0, 2, 3, 1)  # b,c,h,w -> b,h,w,c
        x = x.reshape(b * h * w, c)
    elif len(shape) == 3:
        b, f, c = shape
        x = x.reshape(b * f, c)
    return x, shape


def colors_to_feature(x: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    Transform 2D colors to 3D feature or 4D image

    For feature [b*feat,c] -> [b,feat,c].
    For feature [b*h*w,c] -> [b,c,h,w].
    """
    if len(shape) == 4:
        b, c, h, w = shape
        x = x.reshape(b, h, w, c)
        x = x.permute(0, 3, 1, 2)  # b,h,w,c -> b,c,h,w
    if len(shape) == 3:
        x = x.reshape(shape)
    return x


def find_minimal_feature_size(pathes: List[str]) -> int:
    feature_dims = [read_numpy_feature(p).shape[0] for p in pathes]
    return min(feature_dims)
