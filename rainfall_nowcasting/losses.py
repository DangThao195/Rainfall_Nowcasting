from __future__ import annotations

from typing import cast

from torch import Tensor, nn
from torch.nn import functional as F


class WeightedRainfallLoss(nn.Module):
    def __init__(
        self,
        spatial_weight: Tensor,
        rain_threshold_normalized: float,
        rain_boost: float = 2.0,
        beta: float = 0.5,
    ) -> None:
        super().__init__()
        self.register_buffer("spatial_weight", spatial_weight[None, None, None])
        self.rain_threshold_normalized = rain_threshold_normalized
        self.rain_boost = rain_boost
        self.beta = beta

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        error = F.smooth_l1_loss(prediction, target, reduction="none", beta=self.beta)
        rain_weight = 1.0 + self.rain_boost * (target >= self.rain_threshold_normalized)
        weight = cast(Tensor, self.spatial_weight) * rain_weight
        return (error * weight).sum() / weight.sum().clamp_min(1.0)
