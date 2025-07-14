from torch import nn
from .conv import DiscConvBlock


class PatchDiscriminator(nn.Module):
    def __init__(self, in_dim=3):
        super(PatchDiscriminator, self).__init__()
        self.model = nn.Sequential(
            DiscConvBlock(in_dim, 64, is_first=True),
            DiscConvBlock(64, 128),
            DiscConvBlock(128, 256),
            DiscConvBlock(256, 512, stride=1),
            # last block uses 1 channel conv
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x):
        return self.model(x)
