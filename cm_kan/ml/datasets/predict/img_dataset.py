import torch
import os
from torch.utils.data import Dataset
from cm_kan.ml.utils.io import read_rgb_image
from typing import List
from torchvision.transforms.v2 import Compose


class ImagePredictDataset(Dataset):
    def __init__(self, paths: List[str], transform: Compose) -> None:
        self.paths = paths
        self.transform = transform

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.paths[idx]
        x = read_rgb_image(path)
        path = os.path.basename(path)
        if self.transform is not None:
            x = self.transform(x)
        return path, x

    def __len__(self) -> int:
        return len(self.paths)


class ImagePairedPredictDataset(Dataset):
    def __init__(
        self, paths: List[str], ref_paths: List[str], transform: Compose
    ) -> None:
        assert len(paths) == len(ref_paths), "paths and ref_paths must have same length"
        self.paths = paths
        self.transform = transform
        self.ref_paths = ref_paths

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.paths[idx]
        x = read_rgb_image(path)

        ref_path = self.ref_paths[idx]
        y = read_rgb_image(ref_path)

        path = os.path.basename(path)

        if self.transform is not None:
            x = self.transform(x)
            y = self.transform(y)
        return path, x, y

    def __len__(self) -> int:
        return len(self.paths)
