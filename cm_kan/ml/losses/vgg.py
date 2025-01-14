import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def _normalize_batch(batch):
    # Normalize batch using ImageNet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (batch - mean) / std


class VggLoss(nn.Module):
    """VGG/Perceptual Loss
    
    Parameters
    ----------
    conv_index : str
        Convolutional layer in VGG model to use as perceptual output

    """
    def __init__(self, conv_index: str = '22'):
        super(VggLoss, self).__init__()
        vgg_features = torchvision.models.vgg19(pretrained=True).features
        modules = [m for m in vgg_features]
        
        if conv_index == '22':
            self.vgg = nn.Sequential(*modules[:8]).eval()
        elif conv_index == '54':
            self.vgg = nn.Sequential(*modules[:35]).eval()
        
        self.vgg = self.vgg.requires_grad_(False)
        self.vgg = self.vgg.train(False)

    @torch.no_grad()
    def forward(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute VGG/Perceptual loss between Super-Resolved and High-Resolution

        Parameters
        ----------
        sr : torch.Tensor
            Super-Resolved model output tensor
        hr : torch.Tensor
            High-Resolution image tensor

        Returns
        -------
        loss : torch.Tensor
            Perceptual VGG loss between sr and hr

        """ 
        vgg_sr = self.vgg(_normalize_batch(pred))
        vgg_hr = self.vgg(_normalize_batch(y))
        return F.mse_loss(vgg_sr, vgg_hr)
