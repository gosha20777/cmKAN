import torch
from torch import nn
import lightning as L
from torch import optim
from ..models import SMS
from color_transfer.core import Logger
from ..metrics import (
    PSNR,
    SSIM,
    DeltaE
)


class SMSPipeline(L.LightningModule):

    def __init__(self, model: SMS) -> None:
        super(SMSPipeline, self).__init__()

        self.model = model

        self.mae_loss = nn.L1Loss(reduction='mean')
        self.de_metric = DeltaE(gamma=False)
        self.ssim_metric = SSIM(data_range=(0, 1))
        self.psnr_metric = PSNR(data_range=(0, 1))
        
        self.save_hyperparameters(ignore=['model'])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pred = self.model(x)
        return pred

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        predictions = self(inputs)
        prediction = predictions.permute(0,3,1,2)
        target = targets.permute(0,3,1,2)
        mae_loss = self.mae_loss(prediction, target)
        psnr_metric = self.psnr_metric(prediction, target)
        ssim_metric = self.ssim_metric(prediction, target)
        de_metric = self.de_metric(prediction, target)
        
        print(f'PSNR: {psnr_metric}, dE: {de_metric}')
        self.log('test_panr', psnr_metric, prog_bar=True, logger=True)
        self.log('test_ssim', ssim_metric, prog_bar=True, logger=True)
        self.log('test_de', de_metric, prog_bar=True, logger=True)
        self.log('test_loss', mae_loss, prog_bar=True, logger=True)
        return {'loss': mae_loss}
    
    def predict_step(self, batch, batch_idx, dataloader_idx):
        inputs, _ = batch
        output = self(inputs)
        return output
