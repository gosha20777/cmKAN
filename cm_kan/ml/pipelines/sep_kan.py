import torch
from torch import nn
import lightning as L
from torch import optim
from ..models import ConvKanModel
from color_transfer.core import Logger
from ..transforms.gamma import DegammaCorrection
from ..metrics import (
    PSNR,
    SSIM,
    DeltaE
)


class SepKanPipeline(L.LightningModule):
    def __init__(self,
        model: ConvKanModel,
        optimiser: str = 'adam',
        lr: float = 1e-3,
        weight_decay: float = 0,
    ) -> None:
        super(SepKanPipeline, self).__init__()

        self.model = model
        self.optimizer_type = optimiser
        self.lr = lr
        self.weight_decay = weight_decay
        self.mae_loss = nn.L1Loss(reduction='mean')
        self.de_metric = DeltaE()
        self.ssim_metric = SSIM(data_range=(0, 1))
        self.psnr_metric = PSNR(data_range=(0, 1))
        self.degamma = DegammaCorrection()
        
        self.save_hyperparameters(ignore=['model'])
    
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
        
        Logger.info('Initialized model weights with isp pipeline.')

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

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        predictions = self(inputs)
        mae_loss = self.mae_loss(predictions, targets)

        self.log('train_loss', mae_loss, prog_bar=True, logger=True)
        return {'loss': mae_loss}
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        predictions = self(inputs)
        mae_loss = self.mae_loss(predictions, targets)

        prediction = self.degamma(predictions)
        target = self.degamma(targets)
        psnr_metric = self.psnr_metric(prediction, target)
        ssim_metric = self.ssim_metric(prediction, target)
        de_metric = self.de_metric(prediction, target)
        
        self.log('val_psnr', psnr_metric, prog_bar=True, logger=True)
        self.log('val_ssim', ssim_metric, prog_bar=True, logger=True)
        self.log('val_de', de_metric, prog_bar=True, logger=True)
        self.log('val_loss', mae_loss, prog_bar=True, logger=True)
        return {'loss': mae_loss}
    
    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        predictions = self(inputs)
        mae_loss = self.mae_loss(predictions, targets)

        prediction = self.degamma(predictions)
        target = self.degamma(targets)
        psnr_metric = self.psnr_metric(prediction, target)
        ssim_metric = self.ssim_metric(prediction, target)
        de_metric = self.de_metric(prediction, target)
        
        self.log('test_panr', psnr_metric, prog_bar=True, logger=True)
        self.log('test_ssim', ssim_metric, prog_bar=True, logger=True)
        self.log('test_de', de_metric, prog_bar=True, logger=True)
        self.log('test_loss', mae_loss, prog_bar=True, logger=True)
        return {'loss': mae_loss}
    
    def predict_step(self, batch, batch_idx, dataloader_idx):
        inputs, _ = batch
        output = self(inputs)
        return output
