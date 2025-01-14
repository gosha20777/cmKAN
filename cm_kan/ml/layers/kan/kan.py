import torch
from torch import nn
import torch.nn.functional as F
from typing import List
from .activation import KANActivation
from .symbolic import KANSymbolic
from .weighted_residual import WeightedResidualLayer


class KANLayer(nn.Module):
    "Defines a KAN layer from in_dim variables to out_dim variables."

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        grid_size: int,
        spline_order: int,
        residual_std: float = 0.1,
        grid_range: List[float] = [-1, 1],
    ):
        super(KANLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Define univariate function (splines in original KAN)
        self.activation_fn = KANActivation(
            in_dim,
            out_dim,
            spline_order,
            grid_size,
            grid_range,
        )
        
        self.symbolic_fn = KANSymbolic(
            in_dim,
            out_dim,
        )

        self.activation_mask = nn.Parameter(
            torch.ones((out_dim, in_dim))
        ).requires_grad_(False)
        self.symbolic_mask = torch.nn.Parameter(torch.zeros(out_dim, in_dim)).requires_grad_(False)

        # Define the residual connection layer used to compute \phi
        self.residual_layer = WeightedResidualLayer(in_dim, out_dim, residual_std)

        # Cache for regularization
        self.inp = torch.empty(0)
        self.activations = torch.empty(0)

    def cache(self, inp: torch.Tensor, acts: torch.Tensor):
        self.inp = inp
        self.activations = acts

    def set_symbolic(self, in_index: int, out_index: int, fix:bool, fn):
        """
        Set the symbolic mask to be fixed (fix=1) or unfixed. 
        """
        if fix:
            self.symbolic_mask[out_index, in_index] = 1
            self.symbolic_fn.set_symbolic(in_index, out_index, fn)
        else:
            self.symbolic_mask[out_index, in_index] = 0


    def forward(self, x: torch.Tensor):
        """
        Forward pass of KAN. x is expected to be of shape (batch_size, input_size)
        where input_size is the number of input scalars.

        Stores the activations needed for computing the L1 regularization and
        entropy regularization terms.

        Returns the output of the KAN operation.
        """
        #print(f"{self.in_dim = }")
        #print(f"{x.size() = }")

        spline = self.activation_fn(x)

        # Form the batch of matrices phi(x) of shape [batch_size x out_dim x in_dim]
        phi = self.residual_layer(x, spline)

        # Perform symbolic computations
        sym_phi = self.symbolic_fn(x)
        phi = phi * (self.symbolic_mask == 0) + sym_phi * self.symbolic_mask

        # Mask out pruned edges
        phi = phi * self.activation_mask[None, ...]

        # Cache activations for regularization during training.
        # Also useful for visualizing. Can remove for inference.
        self.cache(x, phi)

        # Really inefficient matmul
        out = torch.sum(phi, dim=-1)

        return out

    def grid_extension(self, x: torch.Tensor, new_grid_size: int):
        """
        Increase granularity of B-spline by increasing the
        number of grid points while maintaining the spline shape.
        """

        self.grid_size = new_grid_size
        self.activation_fn.grid_extension(x, new_grid_size)
