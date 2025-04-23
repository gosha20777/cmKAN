import os
import random
import torch
import lightning as L
from torchvision.transforms.v2 import (
    Compose,
    ToImage,
    ToDtype,
    CenterCrop,
)
from torch.utils.data import DataLoader
from typing import Tuple
from .img_dataset import Image2ImageDataset
from .img_dataset_unpaired import Image2ImageUnpairedDataset
from cm_kan.core import Logger
from cm_kan.ml.transforms.pair_trransform import PairTransform


class Valga2kImgUnpairedDataModule(L.LightningDataModule):
    def __init__(
            self,
            train_a: str,
            train_b: str,
            val_a: str,
            val_b: str,
            test_a: str,
            test_b: str,
            batch_size: int = 32,
            val_batch_size: int = 32,
            test_batch_size: int = 32,
            num_workers: int = min(12, os.cpu_count() - 1),
            img_exts: Tuple[str] = (".png", ".jpg"),
            seed: int = 42,
    ) -> None:
        super().__init__()
        self.test_dataset = None
        self.train_dataset = None
        self.val_dataset = None

        random.seed(seed)

        train_a_recolor = os.path.join(train_a, "recolor")
        train_b_recolor = os.path.join(train_b, "recolor")
        train_a = os.path.join(train_a, "real")
        train_b = os.path.join(train_b, "real")

        paths_ab_recolor = [
            os.path.join(train_a_recolor, fname)
            for fname in os.listdir(train_a_recolor)
            if fname.endswith(img_exts)
        ]
        paths_a = [
            os.path.join(train_a, fname)
            for fname in os.listdir(train_a)
            if fname.endswith(img_exts)
        ]
        paths_ba_recolor = [
            os.path.join(train_b_recolor, fname)
            for fname in os.listdir(train_b_recolor)
            if fname.endswith(img_exts)
        ]
        paths_b = [
            os.path.join(train_b, fname)
            for fname in os.listdir(train_b)
            if fname.endswith(img_exts)
        ]

        self.train_paths_ab_recolor = sorted(paths_ab_recolor)
        self.train_paths_a = sorted(paths_a)
        self.train_paths_ba_recolor = sorted(paths_ba_recolor)
        self.train_paths_b = sorted(paths_b)

        paths_a = [
            os.path.join(val_a, fname)
            for fname in os.listdir(val_a)
            if fname.endswith(img_exts)
        ]
        paths_b = [
            os.path.join(val_b, fname)
            for fname in os.listdir(val_b)
            if fname.endswith(img_exts)
        ]

        self.val_paths_a = sorted(paths_a)
        self.val_paths_b = sorted(paths_b)

        paths_a = [
            os.path.join(test_a, fname)
            for fname in os.listdir(test_a)
            if fname.endswith(img_exts)
        ]
        paths_b = [
            os.path.join(test_b, fname)
            for fname in os.listdir(test_b)
            if fname.endswith(img_exts)
        ]

        self.test_paths_a = sorted(paths_a)
        self.test_paths_b = sorted(paths_b)

        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.image_p_transform = PairTransform(
            crop_size=512, p=0.5, seed=seed
        )

        self.image_train_transform = Compose([
            ToImage(),
            CenterCrop(1024),
            ToDtype(dtype=torch.float32, scale=True),
        ])
        self.image_val_transform = Compose([
            ToImage(),
            CenterCrop(1024),
            ToDtype(dtype=torch.float32, scale=True),
        ])
        self.image_test_transform = Compose([
            ToImage(),
            CenterCrop(1024),
            ToDtype(dtype=torch.float32, scale=True),
        ])
        self.num_workers = num_workers

    def setup(self, stage: str) -> None:
        if stage == 'fit' or stage is None:
            self.train_dataset = Image2ImageUnpairedDataset(
                self.train_paths_ab_recolor,
                self.train_paths_a, 
                self.train_paths_ba_recolor,
                self.train_paths_b, 
                self.image_train_transform, 
                self.image_p_transform,
            )
            self.val_dataset = Image2ImageDataset(
                self.val_paths_a, self.val_paths_b, self.image_val_transform,
            )
        if stage == 'test' or stage is None:
            self.test_dataset = Image2ImageDataset(
                self.test_paths_a, self.test_paths_b, self.image_test_transform,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
        )
