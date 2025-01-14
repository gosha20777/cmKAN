import torch
from torch import nn
from typing import List
import torch.nn.functional as F
from .helper import (
    generate_control_points,
    compute_bspline
)


class KANActivation(nn.Module):
    """
    Defines a KAN Activation layer that computes the spline(x) logic
    described in the original paper.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        spline_order: int,
        grid_size: int,
        grid_range: List[float],
    ):
        super(KANActivation, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.spline_order = spline_order
        self.grid_size = grid_size
        self.grid_range = grid_range
        # Generate (out, in) copies of equally spaced control points on [a, b]
        grid = generate_control_points(
            grid_range[0],
            grid_range[1],
            in_dim,
            out_dim,
            spline_order,
            grid_size,
        )
        self.register_buffer("grid", grid)

        # Define the univariate B-spline function
        self.univarate_fn = compute_bspline

        # Spline parameters
        self.coef = torch.nn.Parameter(
            torch.Tensor(out_dim, in_dim, grid_size + spline_order)
        )

        self._initialization()

    def _initialization(self):
        """
        Initialize each parameter according to the original paper.
        """
        nn.init.xavier_normal_(self.coef)

    def forward(self, x: torch.Tensor):
        """
        Compute and evaluate the learnable activation functions
        applied to a batch of inputs of size in_dim each.
        """
        # [bsz x in_dim] to [bsz x out_dim x in_dim x (grid_size + spline_order)]
        bases = self.univarate_fn(x, self.grid, self.spline_order)

        # [bsz x out_dim x in_dim x (grid_size + spline_order)]
        postacts = bases * self.coef[None, ...]

        # [bsz x out_dim x in_dim] to [bsz x out_dim]
        spline = torch.sum(postacts, dim=-1)

        return spline

    def grid_extension(self, x: torch.Tensor, new_grid_size: int):
        """
        Increase granularity of B-spline activation by increasing the
        number of grid points while maintaining the spline shape.
        """

        # Re-generate grid points with extended size (uniform)
        new_grid = generate_control_points(
            self.grid_range[0],
            self.grid_range[1],
            self.in_dim,
            self.out_dim,
            self.spline_order,
            new_grid_size,
        )

        # bsz x out_dim x in_dim x (old_grid_size + spline_order)
        old_bases = self.univarate_fn(x, self.grid, self.spline_order)

        # bsz x out_dim x in_dim x (new_grid_size + spline_order)
        bases = self.univarate_fn(x, new_grid, self.spline_order)
        # out_dim x in_dim x bsz x (new_grid_size + spline_order)
        bases = bases.permute(1, 2, 0, 3)

        # bsz x out_dim x in_dim
        postacts = torch.sum(old_bases * self.coef[None, ...], dim=-1)
        # out_dim x in_dim x bsz
        postacts = postacts.permute(1, 2, 0)

        # solve for X in AX = B, A is bases and B is postacts
        new_coefs = torch.linalg.lstsq(
            bases.to(x.device),
            postacts.to(x.device),
            driver="gelsy" if x.device == "cpu" else "gelsd",
        ).solution

        # Set new parameters
        self.grid_size = new_grid_size
        self.grid = new_grid
        self.coef = torch.nn.Parameter(new_coefs, requires_grad=True)
