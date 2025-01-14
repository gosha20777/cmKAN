import torch
from torch import nn
import numpy as np
from ..transforms.hist import RGBuvHistBlock


class HistLoss(nn.Module):
    def __init__(self, h: int = 128, insz: int = 256):
        super(HistLoss, self).__init__()
        self.hist_block = RGBuvHistBlock(h=h, insz=insz)

    def _histogram_loss(self, input_hist: torch.Tensor, target_hist: torch.Tensor) -> torch.Tensor:
        return (1/np.sqrt(2.0) * (torch.sqrt(torch.sum(
            torch.pow(torch.sqrt(target_hist) - torch.sqrt(input_hist), 2)))) / 
        input_hist.shape[0])

    @torch.no_grad()
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_hist = self.hist_block(pred)
        target_hist = self.hist_block(target)
        return self._histogram_loss(pred_hist, target_hist)
