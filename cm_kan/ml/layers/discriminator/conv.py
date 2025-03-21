import torch
from torch import nn


class DiscConvBlock(nn.Module):
    def __init__(self, channels_in, channels_out, stride=2, is_first=False):
        super(DiscConvBlock, self).__init__()
        block = (
            nn.Conv2d(channels_in, channels_out, kernel_size=4, stride=stride, padding=1),
            nn.InstanceNorm2d(channels_out),
            nn.LeakyReLU(0.2, True),
        )
        if is_first: # remove the second element
            block = block[0], block[2]
        self.block = nn.Sequential(*block)
        
    def forward(self, x):
        return self.block(x)