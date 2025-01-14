import torch
from torch import nn


# -------------------------------------------------------
# Channel Attention (CA) Layer
# -------------------------------------------------------
class CALayer(nn.Module):
    def __init__(self, channel=64, reduction=16):
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel//reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel//reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)


class spatial_attn_layer(nn.Module):
    def __init__(self, kernel_size=3):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = nn.Conv2d(2, 1, 3, stride=1, padding=1, bias=True)

    def forward(self, x):
        # import pdb;pdb.set_trace()
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale


# -------------------------------------------------------
# Content Unrelated Channel Attention (CUCA) Layer
# -------------------------------------------------------
class CUCALayer(nn.Module):
    def __init__(self, channel=64, min=0, max=None):
        super(CUCALayer, self).__init__()

        self.attention = nn.Conv2d(channel, channel, 1, padding=0,
                                   groups=channel, bias=False)
        self.min, self.max = min, max
        nn.init.uniform_(self.attention.weight, 0, 1)

    def forward(self, x):
        self.attention.weight.data.clamp_(self.min, self.max)
        return self.attention(x)
