from lightning.pytorch.callbacks import Callback
from lightning import LightningModule, Trainer
import torch
import torchvision
import os


class GenerateCallback(Callback):
    def __init__(self, every_n_epochs=1) -> None:
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.input_imgs = None
        self.save_dir = None
        self.target_imgs = None

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        dataloader = trainer.val_dataloaders
        self.input_imgs, self.target_imgs = next(iter(dataloader))
        self.input_imgs = self.input_imgs.to(pl_module.device)
        self.target_imgs = self.target_imgs.to(pl_module.device)
        self.save_dir = os.path.join(trainer.log_dir, "figures")

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            with torch.no_grad():
                pl_module.eval()
                reconst_imgs = pl_module(self.input_imgs)
                pl_module.train()
            # Plot and add to tensorboard
            imgs = torch.stack(
                [self.input_imgs, self.target_imgs, reconst_imgs], dim=1
            ).flatten(0, 1)
            grid = torchvision.utils.make_grid(imgs, nrow=3)
            # Save image
            save_path = os.path.join(
                self.save_dir, f"reconst_{trainer.current_epoch}.png"
            )
            os.makedirs(self.save_dir, exist_ok=True)
            torchvision.utils.save_image(grid, save_path)

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        dataloader = trainer.test_dataloaders
        self.input_imgs, self.target_imgs = next(iter(dataloader))
        self.input_imgs = self.input_imgs.to(pl_module.device)
        self.target_imgs = self.target_imgs.to(pl_module.device)
        self.save_dir = os.path.join(trainer.log_dir, "figures")

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            with torch.no_grad():
                pl_module.eval()
                reconst_imgs = pl_module(self.input_imgs)
                pl_module.train()
            # Plot and add to tensorboard
            imgs = torch.stack(
                [self.input_imgs, self.target_imgs, reconst_imgs], dim=1
            ).flatten(0, 1)
            grid = torchvision.utils.make_grid(imgs, nrow=3)
            # Save image
            save_path = os.path.join(self.save_dir, f"test_{trainer.current_epoch}.png")
            os.makedirs(self.save_dir, exist_ok=True)
            torchvision.utils.save_image(grid, save_path)
