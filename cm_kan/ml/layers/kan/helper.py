import torch
# Helper functions for computing B splines over a grid


def compute_bspline(x: torch.Tensor, grid: torch.Tensor, k: int):
    """
    For a given grid with G_1 intervals and spline order k, we *recursively* compute
    and evaluate each B_n(x_{ij}). x is a (batch_size, in_dim) and grid is a
    (out_dim, in_dim, # grid points + 2k + 1)

    Returns a (batch_size, out_dim, in_dim, grid_size + k) intermediate tensor to 
    compute sum_i {c_i B_i(x)} with.

    """
    
    grid = grid[None, :, :, :].to(x.device)
    x = x[:, None, :, None].to(x.device)
    
    # Base case: B_{i,0}(x) = 1 if (grid_i <= x <= grid_{i+k}) 0 otherwise
    bases = (x >= grid[:, :, :, :-1]) * (x < grid[:, :, :, 1:])

    # Recurse over spline order j, vectorize over basis function i
    for j in range (1, k + 1):
        n = grid.size(-1) - (j + 1)
        b1 = ((x[:, :, :, :] - grid[:, :, :, :n]) / (grid[:, :, :, j:-1] - grid[:, :, :, :n])) * bases[:, :, :, :-1]
        b2 = ((grid[:, :, :, j+1:] - x[:, :, :, :])  / (grid[:, :, :, j+1:] - grid[:, :, :, 1:n+1])) * bases[:, :, :, 1:]
        bases = b1 + b2

    return bases


def coef2curve (x : torch.Tensor, grid: torch.Tensor, coefs: torch.Tensor, k: int):
    """
    For a given (batch of) x, control points (grid), and B-spline coefficients,
    evaluate and return x on the B-spline function.
    """
    bases = compute_bspline(x, grid, k)
    spline = torch.sum(bases * coefs[None, ...], dim=-1)
    return spline


def generate_control_points(
    low_bound: float,
    up_bound: float,
    in_dim: int,
    out_dim: int,
    spline_order: int,
    grid_size: int,
):
    """
    Generate a vector of {grid_size} equally spaced points in the interval [low_bound, up_bound] and broadcast (out_dim, in_dim) copies.
    To account for B-splines of order k, using the same spacing, generate an additional
    k points on each side of the interval. See 2.4 in original paper for details.
    """

    # vector of size [grid_size + 2 * spline_order + 1]
    spacing = (up_bound - low_bound) / grid_size
    grid = torch.arange(-spline_order, grid_size + spline_order + 1)
    grid = grid * spacing + low_bound

    # [out_dim, in_dim, G + 2k + 1]
    grid = grid[None, None, ...].expand(out_dim, in_dim, -1).contiguous()
    return grid
