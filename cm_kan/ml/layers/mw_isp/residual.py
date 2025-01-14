from torch import nn
from .hepler import conv
from .attention import CALayer


# -------------------------------------------------------
# Res Block: x + conv(relu(conv(x)))
# -------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1,
                 padding=1, bias=True, mode='CRC'):
        super(ResBlock, self).__init__()

        assert in_channels == out_channels
        if mode[0] in ['R','L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size,
                        stride, padding, bias=bias, mode=mode)

    def forward(self, x):
        res = self.res(x)
        return x + res


# -------------------------------------------------------
# Residual Channel Attention Block (RCAB)
# -------------------------------------------------------
class RCABlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1,
                 padding=1, bias=True, mode='CRC', reduction=16):
        super(RCABlock, self).__init__()
        assert in_channels == out_channels
        if mode[0] in ['R','L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size,
                        stride, padding, bias=bias, mode=mode)
        self.CA = CALayer(out_channels, reduction)
        #self.SA = spatial_attn_layer()            ## Spatial Attention
        #self.conv1x1 = nn.Conv2d(in_channels*2, in_channels, kernel_size=1)

    def forward(self, x):  
        res = self.res(x)
        #sa_branch = self.SA(res)
        ca_branch = self.CA(res)
        #res = torch.cat([sa_branch, ca_branch], dim=1)
        #res = self.conv1x1(res)
        return ca_branch + x


# -------------------------------------------------------
# Residual Channel Attention Group (RG)
# -------------------------------------------------------
class RCAGroup(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1,
                 padding=1, bias=True, mode='CRC', reduction=16, nb=12):
        super(RCAGroup, self).__init__()
        assert in_channels == out_channels
        if mode[0] in ['R','L']:
            mode = mode[0].lower() + mode[1:]

        RG = [RCABlock(in_channels, out_channels, kernel_size, stride, padding,
                       bias, mode, reduction) for _ in range(nb)]
        RG.append(conv(out_channels, out_channels, mode='C'))

        # self.rg = ShortcutBlock(nn.Sequential(*RG))
        self.rg = nn.Sequential(*RG)

    def forward(self, x):
        res = self.rg(x)
        return res + x
