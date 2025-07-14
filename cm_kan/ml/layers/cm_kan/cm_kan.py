import numpy as np
import torch
from .kan import KANLayer
from .generator import GeneratorLayer, LightGeneratorLayer


class CmKANLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        grid_size,
        spline_order,
        residual_std,
        grid_range,
    ):
        super(CmKANLayer, self).__init__()

        self.kan_layer = KANLayer(
            in_dim=in_channels,
            out_dim=out_channels,
            grid_size=grid_size,
            spline_order=spline_order,
            residual_std=residual_std,
            grid_range=grid_range,
        )

        # Arbitrary layers configuration fc
        self.kan_params_num = 0
        self.kan_params_indices = [0]

        coef_len = np.prod(self.kan_layer.activation_fn.coef_shape)
        univariate_weight_len = np.prod(
            self.kan_layer.residual_layer.univariate_weight_shape
        )
        residual_weight_len = np.prod(
            self.kan_layer.residual_layer.residual_weight_shape
        )
        self.kan_params_indices.extend(
            [coef_len, univariate_weight_len, residual_weight_len]
        )

        self.kan_params_num = np.sum(self.kan_params_indices)
        self.kan_params_indices = np.cumsum(self.kan_params_indices)

        self.generator = GeneratorLayer(in_channels, self.kan_params_num)

    def kan(self, x, w):
        i, j = self.kan_params_indices[0], self.kan_params_indices[1]
        coef = w[:, i:j].view(-1, *self.kan_layer.activation_fn.coef_shape)
        i, j = self.kan_params_indices[1], self.kan_params_indices[2]
        univariate_weight = w[:, i:j].view(
            -1, *self.kan_layer.residual_layer.univariate_weight_shape
        )
        i, j = self.kan_params_indices[2], self.kan_params_indices[3]
        residual_weight = w[:, i:j].view(
            -1, *self.kan_layer.residual_layer.residual_weight_shape
        )
        x = self.kan_layer(x, coef, univariate_weight, residual_weight)

        return x.squeeze(0)

    def forward(self, x):
        B, C, H, W = x.shape

        # kan weights (b, kan_params_num, h, w)
        weights = self.generator(x)
        # kan weights (b, h * w, kan_params_num)
        weights = weights.permute(0, 2, 3, 1)
        weights = weights.reshape(B * H * W, self.kan_params_num)

        x = x.permute(0, 2, 3, 1).reshape(B * H * W, C)

        # img (b * h * w, 3), weights (b * h * w, kan_params_num)
        x = self.kan(x, weights)

        x = x.view(B, H, W, self.kan_layer.out_dim).permute(0, 3, 1, 2)

        return x


class LightCmKANLayer(CmKANLayer):
    def __init__(
        self,
        in_channels,
        out_channels,
        grid_size,
        spline_order,
        residual_std,
        grid_range,
    ):
        super(LightCmKANLayer, self).__init__(
            in_channels, out_channels, grid_size, spline_order, residual_std, grid_range
        )
        self.generator = LightGeneratorLayer(in_channels, self.kan_params_num)
