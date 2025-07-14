from lightning.pytorch.callbacks import BasePredictionWriter
from lightning import LightningModule, Trainer
from typing import List, Tuple
import torch
import torchvision
import os


class ImagePredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir: str, write_interval: str = "batch") -> None:
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.write_interval = write_interval

        os.makedirs(output_dir, exist_ok=True)

    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: torch.Tensor,
        batch_indices: List[int],
        batch: Tuple[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        pathes = batch[0]
        for i, path in enumerate(pathes):
            image = prediction[i].cpu()
            torchvision.utils.save_image(image, os.path.join(self.output_dir, path))
