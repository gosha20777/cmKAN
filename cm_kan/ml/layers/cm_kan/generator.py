import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from cm_kan.ml.layers.mw_isp import DWTForward


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class IlluminationEstimator(nn.Module):
    def __init__(
            self, n_fea_middle, n_fea_in=4, n_fea_out=3):
        super(IlluminationEstimator, self).__init__()

        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)

        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)

        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):
        # img:        b,c=3,h,w
        # mean_c:     b,c=1,h,w
        
        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w
        
        mean_c = img.mean(dim=1).unsqueeze(1)
        input = torch.cat([img,mean_c], dim=1)

        x_1 = self.conv1(input)
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map
    

class LayerNorm(nn.Module):

    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = nn.LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature_a = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature_v = nn.Parameter(torch.ones(num_heads, 1, 1))

        # q: what we want to attend to (spatial information)
        self.q_proj = nn.Conv2d(
            dim, dim, kernel_size=3,
            padding=1, stride=2, padding_mode='reflect', 
            groups=dim, bias=bias
        )
        # k: what we use to calculate attention (channel information)
        self.k_proj = nn.Conv2d(
            dim, dim, kernel_size=3,
            padding=1, stride=2, padding_mode='reflect', 
            bias=bias
        )
        # v: what we use to calculate the output (channel information)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        # a: anchor information (reduced spatial information and channel information)
        self.a_proj = nn.Sequential(
            nn.Conv2d(
                dim, dim, kernel_size=3,
                padding=1, stride=2, padding_mode='reflect', 
                groups=dim, bias=bias
            ),
            nn.Conv2d(dim, dim//2, kernel_size=1)
        )
        # output projection
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, illu_feat):
        b, c, h, w = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x) * illu_feat
        a = self.a_proj(x)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        a = rearrange(a, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        a = torch.nn.functional.normalize(a, dim=-1)

        # Q - C×(H/s×W/s), K - C×(H/s×W/s), V - C×(H×W), A - C/r×(H/s×W/s) 

        # transposed self-attention with attention map of shape (C×C)
        attn_a = (q @ a.transpose(-2, -1)) * self.temperature_a
        attn_a = attn_a.softmax(dim=-1)

        attn_k = (a @ k.transpose(-2, -1)) * self.temperature_v
        attn_k = attn_k.softmax(dim=-1)
        
        out_v = (attn_k @ v)

        out = (attn_a @ out_v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class FFN(nn.Module):
    """
    Feed-forward Network with Depth-wise Convolution
    """
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.pointwise1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.depthwise = nn.Conv2d(hidden_features,hidden_features, kernel_size=3,stride=1,padding=1,dilation=1,groups=hidden_features)
        self.pointwise2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.act_layer = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.pointwise1(x)
        x = self.depthwise(x)
        x = self.act_layer(x)
        x = self.pointwise2(x)
        return x


class TransformerBlock(nn.Module):
    """
    from restormer
    input size: (B,C,H,W)
    output size: (B,C,H,W)
    H, W could be different
    """

    def __init__(self, in_channel, mid_channel, out_channel, num_heads, bias):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(in_channel)
        self.attn = Attention(in_channel, num_heads, bias)
        self.norm2 = LayerNorm(in_channel)
        self.ffn = FFN(in_channel, mid_channel, out_channel)

    def forward(self, x, illu_feat):
        x = x + self.attn(self.norm1(x), illu_feat)
        x = x + self.ffn(self.norm2(x))

        return x


class Encoder2D(torch.nn.Module):
    """ Input features BxCxN """

    def __init__(self, in_dim, out_dim, kernel_size):
        super(Encoder2D, self).__init__()
        self.estimator = IlluminationEstimator(12, in_dim+1, in_dim)
        
        self.down1 = DWTForward() # 12 h/2
        self.trans1 = TransformerBlock(
            12, 12, 12, 3, True
        )
        self.illu_down1 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(12, 12, 1),
        )

        self.down2 = DWTForward() # 48 h/4
        self.trans2 = TransformerBlock(
            48, 48, 48, 3, True
        )
        self.illu_down2 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(12, 48, 1),
        )
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up2 = nn.Upsample(scale_factor=4, mode='bilinear')
        
        self.conv_out = nn.Sequential(
            LayerNorm(3+12+48),
            FFN(3+12+48, out_dim)
        )
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        illu_fea, illu_map = self.estimator(x)
        x = x * illu_map + x # 3 h w

        x1 = self.down1(x)
        illu_fea = self.illu_down1(illu_fea)
        x1 = self.trans1(x1, illu_fea) # 12 h/2 w/2

        x2 = self.down2(x1)
        illu_fea = self.illu_down2(illu_fea)
        x2 = self.trans2(x2, illu_fea) # 48 h/4 w/4

        x1 = self.up1(x1)
        x2 = self.up2(x2)
        x = torch.cat([x, x1, x2], dim=1)
        x = self.conv_out(x)
        return x


class GeneratorLayer(torch.nn.Module):
    """
    sepconv replace conv_out to reduce GFLOPS
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        MID_CHANNELS = 21 * in_channels
        self.encoder = Encoder2D(in_channels, MID_CHANNELS, 3)

        self.norm1 = LayerNorm(MID_CHANNELS)

        N = 128
        self.basis = nn.Parameter(torch.rand(1, MID_CHANNELS, N))

        self.q = nn.Conv2d(MID_CHANNELS, MID_CHANNELS, kernel_size=1)
        self.k = nn.Conv1d(MID_CHANNELS, MID_CHANNELS, kernel_size=1)
        self.v = nn.Conv1d(MID_CHANNELS, MID_CHANNELS, kernel_size=1)

        self.norm2 = LayerNorm(MID_CHANNELS)

        self.conv_reproj = FFN(in_features=MID_CHANNELS, out_features=out_channels)

    def forward(self, x:torch.Tensor):

        B, C, H, W = x.shape

        # forward projection
        x = self.encoder(x)

        x = self.norm1(x)

        # basis coeff
        q = self.q(x)
        k = self.k(self.basis)
        v = self.v(self.basis)

        q = rearrange(q, 'b c h w -> b c (h w)')

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        a = (q.transpose(-2, -1) @ k).transpose(-2, -1)
        a = F.relu(a)

        y = v @ a

        y = rearrange(y, 'b c (h w) -> b c h w', h=H, w=W)

        # back projection
        x = self.norm2(y)
        x = self.conv_reproj(x)

        return x


class LightEncoder2D(torch.nn.Module):
    """ Input features BxCxN """

    def __init__(self, in_dim, out_dim, kernel_size):
        super(LightEncoder2D, self).__init__()
        self.estimator = IlluminationEstimator(12, in_dim+1, in_dim)
        
        self.down1 = DWTForward() # 12 h/2
        self.trans1 = TransformerBlock(
            12, 12, 12, 3, True
        )
        self.illu_down1 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(12, 12, 1),
        )

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear')
        
        self.conv_out = nn.Sequential(
            LayerNorm(3+12),
            FFN(3+12, out_dim)
        )
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        illu_fea, illu_map = self.estimator(x)
        x = x * illu_map + x # 3 h w

        x1 = self.down1(x)
        illu_fea = self.illu_down1(illu_fea)
        x1 = self.trans1(x1, illu_fea) # 12 h/2 w/2

        x1 = self.up1(x1)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_out(x)
        return x


class LightGeneratorLayer(torch.nn.Module):
    """
    sepconv replace conv_out to reduce GFLOPS
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        MID_CHANNELS = 3+12
        self.encoder = LightEncoder2D(in_channels, MID_CHANNELS, 3)

        self.norm1 = LayerNorm(MID_CHANNELS)

        N = 30
        self.basis = nn.Parameter(torch.rand(1, MID_CHANNELS, N))

        self.q = nn.Conv2d(MID_CHANNELS, MID_CHANNELS, kernel_size=1)
        self.k = nn.Conv1d(MID_CHANNELS, MID_CHANNELS, kernel_size=1)
        self.v = nn.Conv1d(MID_CHANNELS, MID_CHANNELS, kernel_size=1)

        self.norm2 = LayerNorm(MID_CHANNELS)

        self.conv_reproj = FFN(in_features=MID_CHANNELS, out_features=out_channels)

    def forward(self, x:torch.Tensor):

        B, C, H, W = x.shape

        # forward projection
        x = self.encoder(x)

        x = self.norm1(x)

        # basis coeff
        q = self.q(x)
        k = self.k(self.basis)
        v = self.v(self.basis)

        q = rearrange(q, 'b c h w -> b c (h w)')

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        a = (q.transpose(-2, -1) @ k).transpose(-2, -1)
        a = F.relu(a)

        y = v @ a

        y = rearrange(y, 'b c (h w) -> b c h w', h=H, w=W)

        # back projection
        x = self.norm2(y)
        x = self.conv_reproj(x)

        return x