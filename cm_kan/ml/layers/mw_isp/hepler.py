import torch.nn as nn
from collections import OrderedDict

def seq(*args):
    if len(args) == 1:
        args = args[0]
    if isinstance(args, nn.Module):
        return args
    modules = OrderedDict()
    if isinstance(args, OrderedDict):
        for k, v in args.items():
            modules[k] = seq(v)
        return nn.Sequential(modules)
    assert isinstance(args, (list, tuple))
    return nn.Sequential(*[seq(i) for i in args])

# -------------------------------------------------------
# return nn.Sequantial of (Conv + BN + ReLU)
# -------------------------------------------------------
def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1,
         output_padding=0, dilation=1, groups=1, bias=True,
         padding_mode='reflect', mode='CBR'):
    L = []
    for t in mode:
        if t == 'C':
            L.append(nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               dilation=dilation,
                               groups=groups,
                               bias=bias,
                               padding_mode=padding_mode))
        elif t == 'X':
            assert in_channels == out_channels
            L.append(nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               dilation=dilation,
                               groups=in_channels,
                               bias=bias,
                               padding_mode=padding_mode))
        elif t == 'T':
            L.append(nn.ConvTranspose2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        output_padding=output_padding,
                                        groups=groups,
                                        bias=bias,
                                        dilation=dilation,
                                        padding_mode=padding_mode))
        elif t == 'B':
            L.append(nn.BatchNorm2d(out_channels))
        elif t == 'I':
            L.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif t == 'i':
            L.append(nn.InstanceNorm2d(out_channels))
        elif t == 'R':
            L.append(nn.ReLU(inplace=True))
        elif t == 'r':
            L.append(nn.ReLU(inplace=False))
        elif t == 'P':
            L.append(nn.PReLU())
        elif t == 'L':
            L.append(nn.LeakyReLU(negative_slope=1e-1, inplace=True))
        elif t == 'l':
            L.append(nn.LeakyReLU(negative_slope=1e-1, inplace=False))
        elif t == '2':
            L.append(nn.PixelShuffle(upscale_factor=2))
        elif t == '3':
            L.append(nn.PixelShuffle(upscale_factor=3))
        elif t == '4':
            L.append(nn.PixelShuffle(upscale_factor=4))
        elif t == 'U':
            L.append(nn.Upsample(scale_factor=2, mode='nearest'))
        elif t == 'u':
            L.append(nn.Upsample(scale_factor=3, mode='nearest'))
        elif t == 'M':
            L.append(nn.MaxPool2d(kernel_size=kernel_size,
                                  stride=stride,
                                  padding=0))
        elif t == 'A':
            L.append(nn.AvgPool2d(kernel_size=kernel_size,
                                  stride=stride,
                                  padding=0))
        else:
            raise NotImplementedError('Undefined type: '.format(t))
    return seq(*L)
