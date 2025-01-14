import torch
from torch import nn
import lightning as L
from torch import optim
from ..models import MWRCAN, ConvKanModel
from color_transfer.core import Logger
from ..metrics import (
    PSNR,
    SSIM,
    DeltaE
)
from ..losses import VggLoss
from typing import Union


class MWImageSignalPipeline(L.LightningModule):
    def __init__(self,
        model: Union[MWRCAN, ConvKanModel],
        optimiser: str = 'adam',
        lr: float = 1e-3,
        weight_decay: float = 0,
    ) -> None:
        super(MWImageSignalPipeline, self).__init__()

        self.model = model
        self.optimizer_type = optimiser
        self.lr = lr
        self.weight_decay = weight_decay
        self.mae_loss = nn.L1Loss(reduction='mean')
        self.vgg_loss = VggLoss()
        self.ssim_loss = SSIM()


        self.de_metric = DeltaE()
        self.ssim_metric = SSIM(data_range=(0, 1))
        self.psnr_metric = PSNR(data_range=(0, 1))
        
        self.save_hyperparameters(ignore=['model'])
    
    def setup(self, stage: str) -> None:
        # if stage == 'fit' or stage is None:
        #     for m in self.model.kan.modules():
        #         if isinstance(m, nn.Conv1d):
        #             nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        #             if m.bias is not None:
        #                 nn.init.constant_(m.bias, 0)
        #         if isinstance(m, nn.Conv2d):
        #             nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        #             if m.bias is not None:
        #                 nn.init.constant_(m.bias, 0)
        #         elif isinstance(m, nn.BatchNorm2d):
        #             nn.init.constant_(m.weight, 1)
        #             nn.init.constant_(m.bias, 0)
        #         elif isinstance(m, nn.Linear):
        #             nn.init.normal_(m.weight, 0, 0.01)
        #             nn.init.constant_(m.bias, 0)
        Logger.info('Initialized model weights with mw_isp pipeline.')

    def configure_optimizers(self):
        if self.optimizer_type == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=0.0001)
        elif self.optimizer_type == 'sgd':
            optimizer = optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f'unsupported optimizer_type: {self.optimizer_type}')
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [50,100,150,200], gamma=0.5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pred = self.model(x)
        return pred

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        predictions = self(inputs)
        
        mae_loss = self.mae_loss(predictions, targets)
        vgg_loss = self.vgg_loss(predictions, targets)
        ssim_loss = self.ssim_loss(predictions, targets)
        total_loss = mae_loss + vgg_loss + (1 - ssim_loss) * 0.15

        self.log('train_loss', total_loss, prog_bar=True, logger=True)
        return {'loss': total_loss}
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        predictions = self(inputs)
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
        predictions = self(inputs)
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
        inputs, _ = batch
        output = self(inputs)
        return output
