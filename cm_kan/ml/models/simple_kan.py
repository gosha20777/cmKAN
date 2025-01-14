import torch
from torch import nn
from typing import List
from ..layers import KANLayer


class SimpleKanModel(nn.Module):
    """
    Standard architecture for Kolmogorov-Arnold Networks described in the original paper.
    Layers are defined via a list of layer widths.

    old kan.Model class
    """

    def __init__(
        self,
        in_dims: List[int],
        out_dims: List[int],
        grid_size: int,
        spline_order: int,
        residual_std: float = 0.1,
        grid_range: List[float] = [-1, 1],
    ) -> None:
        super(SimpleKanModel, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.sigmoid = torch.nn.Sigmoid()

        for in_dim, out_dim in zip(in_dims, out_dims):
            self.layers.append(
                KANLayer(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    residual_std=residual_std,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass sequentially across each layer.
        """
        for layer in self.layers:
            x = layer(x)

        x = self.sigmoid(x)
        return x

    @torch.no_grad
    def set_symbolic(
        self,
        layer: int,
        in_index: int,
        out_index: int,
        fix: bool,
        fn,
    ) -> None:
        """
        For layer {layer}, activation {in_index, out_index}, fix (or unfix if {fix=False})
        the output to the function {fn}. This is grossly inefficient, but works.
        """
        self.layers[layer].set_symbolic(in_index, out_index, fix, fn)

    @torch.no_grad
    def prune(self, x: torch.Tensor, mag_threshold: float = 0.01) -> None:
        """
        Prune (mask) a node in a KAN layer if the normalized activation
        incoming or outgoing are lower than mag_threshold.
        """
        # Collect activations and cache
        self.forward(x)

        # Can't prune at last layer
        for l_idx in range(len(self.layers) - 1):
            # Average over the batch and take the abs of all edges
            in_mags = torch.abs(torch.mean(self.layers[l_idx].activations, dim=0))

            # (in_dim, out_dim), average over out_dim
            in_score = torch.max(in_mags, dim=-1)[0]

            # Average over the batch and take the abs of all edges
            out_mags = torch.abs(torch.mean(self.layers[l_idx + 1].activations, dim=0))

            # (in_dim, out_dim), average over out_dim
            out_score = torch.max(out_mags, dim=0)[0]

            # Check for input, output (normalized) activations > mag_threshold
            active_neurons = (in_score > mag_threshold) * (out_score > mag_threshold)
            inactive_neurons_indices = (active_neurons == 0).nonzero()

            # Mask all relevant activations
            self.layers[l_idx + 1].activation_mask[:, inactive_neurons_indices] = 0
            self.layers[l_idx].activation_mask[inactive_neurons_indices, :] = 0

    @torch.no_grad
    def grid_extension(self, x: torch.Tensor, new_grid_size: int) -> None:
        """
        Increase granularity of B-spline by changing the grid size
        in the B-spline computation to be new_grid_size.
        """
        self.forward(x)
        for l_idx in range(len(self.layers)):
            self.layers[l_idx].grid_extension(self.layers[l_idx].inp, new_grid_size)
        self.config.grid_size = new_grid_size
