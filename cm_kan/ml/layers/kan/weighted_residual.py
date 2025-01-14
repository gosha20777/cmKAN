import torch
from torch import nn
import torch.nn.functional as F


class WeightedResidualLayer(nn.Module):
    """
    Defines the activation function used in the paper,
    phi(x) = w_b SiLU(x) + w_s B_spline(x)
    as a layer.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        residual_std: float = 0.1,
    ):
        super(WeightedResidualLayer, self).__init__()
        self.univariate_weight = torch.nn.Parameter(
            torch.Tensor(out_dim, in_dim)
        )  # w_s in paper

        # Residual activation functions
        self.residual_fn = F.silu
        self.residual_weight = torch.nn.Parameter(
            torch.Tensor(out_dim, in_dim)
        )  # w_b in paper

        self._initialization(residual_std)

    def _initialization(self, residual_std):
        """
        Initialize each parameter according to the original paper.
        """
        nn.init.normal_(self.residual_weight, mean=0.0, std=residual_std)
        nn.init.ones_(self.univariate_weight)

    def forward(self, x: torch.Tensor, post_acts: torch.Tensor):
        """
        Given the input to a KAN layer and the activation (e.g. spline(x)),
        compute a weighted residual.

        x has shape (bsz, in_dim) and act has shape (bsz, out_dim, in_dim)
        """

        # Broadcast the input along out_dim of post_acts
        res = self.residual_weight * self.residual_fn(x[:, None, :])
        act = self.univariate_weight * post_acts
        return res + act
