import torch
from torch.utils.data import Dataset
from color_transfer.ml.utils.io import read_rgb_image
from typing import List
from torchvision.transforms.v2 import Compose
from color_transfer.ml.transforms.pair_trransform import PairTransform


class Image2ImageDataset(Dataset):
    def __init__(self, paths_a: List[str], paths_b: List[str], transform: Compose, p_transform: PairTransform = None) -> None:
        assert len(paths_a) == len(paths_b), "paths_a and paths_b must have same length"
        self.paths_a = paths_a
        self.paths_b = paths_b
        self.transform = transform
        self.p_transform = p_transform

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.paths_a[idx]
        x = read_rgb_image(path)
        if self.transform is not None:
            x = self.transform(x)

        path = self.paths_b[idx]
        y = read_rgb_image(path)
        if self.transform is not None:
            y = self.transform(y)

        if self.p_transform is not None:
            x, y = self.p_transform(x, y)

        return x, y

    def __len__(self) -> int:
        return len(self.paths_a)