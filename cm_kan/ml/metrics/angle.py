import torch
from torchmetrics import Metric
from torchmetrics import CosineSimilarity
from einops import rearrange


class Angle(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state(
            "correct",
            default=torch.tensor(0, dtype=torch.float64),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum"
        )
        self.cosine_similarity = CosineSimilarity(reduction="mean")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")

        preds = rearrange(preds, "b c h w -> (b h w) c")
        target = rearrange(target, "b c h w -> (b h w) c")

        cos = self.cosine_similarity(preds, target)

        self.correct += cos
        self.total += 1

    def compute(self) -> torch.Tensor:
        return self.correct.float() / self.total
