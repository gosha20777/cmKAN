import torch
from torch import nn
import lightning as L
from lightning.pytorch.callbacks import (
    RichProgressBar,
)
from rich.progress import Progress
from torch import optim
from ..models import LightCmKAN
from ..datasets import PairDataModule
from cm_kan.core import Logger
from ..metrics import (
    PSNR,
    SSIM,
    DeltaE,
)
import copy


class PairBasedPipeline(L.LightningModule):
    def __init__(self,
        model: LightCmKAN,
        optimiser: str = 'adam',
        lr: float = 1e-3,
        weight_decay: float = 0,
        finetune_iters: int = 10,
    ) -> None:
        super(PairBasedPipeline, self).__init__()

        self.model = model
        self.optimizer_type = optimiser
        self.lr = lr
        self.weight_decay = weight_decay
        self.mae_loss = nn.L1Loss(reduction='mean')
        self.ssim_loss = SSIM(data_range=(0, 1))
        self.de_metric = DeltaE()
        self.ssim_metric = SSIM(data_range=(0, 1))
        self.psnr_metric = PSNR(data_range=(0, 1))

        self.finetune_iters = finetune_iters
        self.save_hyperparameters(ignore=['model', 'internal_trainer'])
        self.progress = None
        self.finetune_task = None

    
    def setup(self, stage: str) -> None:
        '''
        Initialize model weights before training
        '''
        if stage == 'fit' or stage is None:
            for m in self.model.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)
        
        Logger.info('Initialized model weights with [bold green]Pair Based[/bold green] pipeline.')

    def configure_optimizers(self):
        if self.optimizer_type == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_type == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f'unsupported optimizer_type: {self.optimizer_type}')
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=500, T_mult=1, eta_min=1e-5
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pred = self.model(x)
        return pred
    
    def on_validation_end(self):
        if self.progress:
            self.progress.update(
                self.finetune_task, 
                advance=0, 
                completed=True,
                description=f'Finetune', 
                visible=False,
            )
        return super().on_validation_end()
    
    def _finetune_predict(self, inputs, targets) -> torch.Tensor:
        assert len(inputs) == 1, f"Expected batch size = 1, got {len(inputs)}"

        if self.progress is None:
            for c in self._trainer.callbacks:
                if isinstance(c, RichProgressBar):
                    self.progress = c.progress
                    self.finetune_task = self.progress.add_task(f"Finetune", total=self.finetune_iters)
                    break

        if self.progress:
            self.progress.update(
                self.finetune_task, 
                advance=0, 
                description=f'Finetune', 
                visible=True,
                refresh=True
            )

        dm = PairDataModule(inputs[0], targets[0], num_iters=self.finetune_iters)
        dm.setup('fit')
        dl = dm.train_dataloader()

        finetune_model = copy.deepcopy(self.model)
        optimizer = optim.Adam(finetune_model.parameters())
        loss_fn = nn.L1Loss(reduction='mean')
        finetune_model.train(True)
                
        with torch.set_grad_enabled(True):
            for i, batch in enumerate(dl):
                inputs, targets = batch
                optimizer.zero_grad()
                predictions = finetune_model(inputs)
                loss = loss_fn(predictions, targets)
                loss.backward()
                optimizer.step()
                if self.progress:
                    self.progress.update(
                        self.finetune_task, 
                        advance=1, 
                        description=f'Finetune', 
                        visible=True
                    )
        
        finetune_model.eval()
        with torch.no_grad():
            predictions = finetune_model(inputs)

        del finetune_model

        if self.progress:
            self.progress.update(
                self.finetune_task, 
                advance=0, 
                completed=True,
                description=f'Finetune', 
                visible=True,
            )

        return predictions

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        predictions = self(inputs)
        mae_loss = self.mae_loss(predictions, targets)
        ssim_loss = self.ssim_loss(predictions, targets)
        loss = mae_loss + (1 - ssim_loss) * 0.15

        self.log('train_loss', loss, prog_bar=True, logger=True)
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        predictions = self._finetune_predict(inputs, targets)
        mae_loss = self.mae_loss(predictions, targets)
        psnr_metric = self.psnr_metric(predictions, targets)
        ssim_metric = self.ssim_metric(predictions, targets)
        de_metric = self.de_metric(predictions, targets)
        
        self.log('val_psnr', psnr_metric, prog_bar=True, logger=True)
        self.log('val_ssim', ssim_metric, prog_bar=True, logger=True)
        self.log('val_de', de_metric, prog_bar=True, logger=True)
        self.log('val_loss', mae_loss, prog_bar=True, logger=True)
        return {'loss': mae_loss}
    
    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        predictions = self._finetune_predict(inputs, targets)
        mae_loss = self.mae_loss(predictions, targets)
        panr_metric = self.psnr_metric(predictions, targets)
        ssim_metric = self.ssim_metric(predictions, targets)
        de_metric = self.de_metric(predictions, targets)
        
        self.log('test_panr', panr_metric, prog_bar=True, logger=True)
        self.log('test_ssim', ssim_metric, prog_bar=True, logger=True)
        self.log('test_de', de_metric, prog_bar=True, logger=True)
        self.log('test_loss', mae_loss, prog_bar=True, logger=True)
        return {'loss': mae_loss}
    
    def predict_step(self, batch, batch_idx, dataloader_idx):
        inputs, targets = batch
        predictions = self._finetune_predict(inputs, targets)
        return predictions
