import torch
from torch import nn
import random
from torchvision.transforms.v2 import functional as F
from torchvision.transforms import RandomCrop
from torchvision.transforms import Compose
from typing import Tuple


class PairTransform(nn.Module):
    def __init__(self, crop_size: int = 255, p: float = 0.5, seed: int = 42) -> None:
        super().__init__()
        self.p = p
        self.crop_size = crop_size
        random.seed(seed)

    def forward(self, image1: torch.Tensor, image2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        i, j, h, w = RandomCrop.get_params(image1, output_size=(self.crop_size, self.crop_size))
        image1 = F.crop(image1, i, j, h, w)
        image2 = F.crop(image2, i, j, h, w)
        
        if random.random() > self.p:
            image1 = F.hflip(image1)
            image2 = F.hflip(image2)
        
        if random.random() > self.p:
            image1 = F.vflip(image1)
            image2 = F.vflip(image2)

        return image1, image2
