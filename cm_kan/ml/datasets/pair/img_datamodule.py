import torch
import lightning as L
from torch.utils.data import DataLoader
from .img_dataset import PairDataset


class PairDataModule(L.LightningDataModule):
    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        num_iters: int = 10,
    ) -> None:
        super().__init__()
        self.test_dataset = None
        self.train_dataset = None
        self.val_dataset = None

        self.num_iters = num_iters
        self.x = x
        self.y = y

    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = PairDataset(
                self.x,
                self.y,
                self.num_iters,
            )
            self.val_dataset = PairDataset(
                self.x,
                self.y,
                self.num_iters,
            )
        if stage == "test" or stage is None:
            self.test_dataset = PairDataset(
                self.x,
                self.y,
                self.num_iters,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=1,
            pin_memory=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            pin_memory=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            pin_memory=False,
        )
