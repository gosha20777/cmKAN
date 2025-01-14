import torch
from torch.utils.data import Dataset
from color_transfer.ml.utils.io import read_numpy_feature
from typing import List
from torchvision.transforms.v2 import Compose
import numpy as np


class Feature2FeatureDataset(Dataset):
    def __init__(self, paths_a: List[str], paths_b: List[str], n_samples: int, transform: Compose) -> None:
        assert len(paths_a) == len(paths_b), "paths_a and paths_b must have same length"
        self.paths_a = paths_a
        self.paths_b = paths_b
        self.n_samples = n_samples
        self.transform = transform

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.paths_a[idx]
        x = read_numpy_feature(path)
        if self.n_samples > 0:
            choices = np.random.choice(x.shape[0], self.n_samples, replace=False)
            x = x[choices]
        if self.transform is not None:
            x = self.transform(x)

        path = self.paths_b[idx]
        y = read_numpy_feature(path)
        if self.n_samples > 0:
            y = y[choices]
        if self.transform is not None:
            y = self.transform(y)

        return x, y

    def __len__(self) -> int:
        return len(self.paths_a)