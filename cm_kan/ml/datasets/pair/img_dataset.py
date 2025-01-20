import torch
from torch.utils.data import Dataset
from cm_kan.ml.utils.io import read_rgb_image
from typing import List
from torchvision.transforms.v2 import Compose
from cm_kan.ml.transforms.pair_trransform import PairTransform


class PairDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, lengh: int) -> None:
        self.x = x
        self.y = y
        self.lengh = lengh

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.x
        y = self.y
        return x, y

    def __len__(self) -> int:
        return self.lengh