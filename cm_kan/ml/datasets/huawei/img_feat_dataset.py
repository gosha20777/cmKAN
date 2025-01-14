from pathlib import Path
import torch
from torch.utils.data import Dataset
from color_transfer.ml.utils.io import read_rgb_image
from typing import List
from torchvision.transforms.v2 import Compose
from .feat_dataset import Feature2FeatureDataset
from .img_dataset import Image2ImageDataset


class ImageFeature2ImageFeatureDataset(Dataset):
    def __init__(self, img_dataset: Image2ImageDataset, feat_dataset: Feature2FeatureDataset) -> None:
        assert len(img_dataset) == len(feat_dataset), "image and feature dataset must be the same length"
        self.img_dataset = img_dataset
        self.feat_dataset = feat_dataset

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = self.img_dataset[idx]
        feat = self.feat_dataset[idx]
        return (img[0], feat[0], feat[1]), img[1]

    def __len__(self) -> int:
        return len(self.img_dataset)