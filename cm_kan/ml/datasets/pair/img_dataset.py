import torch
from torch.utils.data import Dataset


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
