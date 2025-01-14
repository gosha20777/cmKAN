import os
import random
import torch
import lightning as L
import colour
import numpy as np
from torchvision.transforms.v2 import (
    Compose,
    ToImage,
    ToDtype,
    CenterCrop,
)
from torch.utils.data import DataLoader
from typing import Tuple
from .img_feat_dataset import ImageFeature2ImageFeatureDataset
from .feat_dataset import Feature2FeatureDataset
from .img_dataset import Image2ImageDataset
from ...transforms import GammaCorrection
from ...utils.process import find_minimal_feature_size
from color_transfer.core import Logger
from color_transfer.ml.transforms.pair_trransform import PairTransform
from ...utils import images

CROP = 512

class HuaweiSMSDataModule(L.LightningDataModule):
    def __init__(
            self,
            path_a: str,
            path_b: str,
            batch_size: int = 32,
            num_workers: int = min(12, os.cpu_count() - 1),
            feat_exts: Tuple[str] = (".npy"),
            img_exts: Tuple[str] = (".png", ".jpg"),
            seed: int = 42,
    ) -> None:
        super().__init__()
        self.test_dataset = None

        random.seed(seed)

        # train
        paths_a_img = [
            os.path.join(path_a, fname)
            for fname in os.listdir(path_a)
            if fname.endswith(img_exts)
        ]
        paths_a_feat = [
            os.path.join(path_a, fname)
            for fname in os.listdir(path_a)
            if fname.endswith(feat_exts)
        ]
        paths_b_img = [
            os.path.join(path_b, fname)
            for fname in os.listdir(path_b)
            if fname.endswith(img_exts)
        ]
        paths_b_feat = [
            os.path.join(path_b, fname)
            for fname in os.listdir(path_b)
            if fname.endswith(feat_exts)
        ]

        self.test_paths_a_img = sorted(paths_a_img)
        self.test_paths_a_feat = sorted(paths_a_feat)
        self.test_paths_b_img = sorted(paths_b_img)
        self.test_paths_b_feat = sorted(paths_b_feat)

        self.batch_size = 1 #batch_size
        self.image_pair_transform = None
        self.image_transform = images.rgb
        self.feature_transform = None
        self.num_workers = num_workers

    def setup(self, stage: str) -> None:
        if stage == 'test' or stage is None:
            test_img_dataset = Image2ImageDataset(
                self.test_paths_a_img, self.test_paths_b_img, self.image_transform, self.image_pair_transform,
            )
            test_feat_dataset = Feature2FeatureDataset(
                self.test_paths_a_feat, self.test_paths_b_feat, -1, self.feature_transform,
            )
            self.test_dataset = ImageFeature2ImageFeatureDataset(
                test_img_dataset, test_feat_dataset
            )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
        )
