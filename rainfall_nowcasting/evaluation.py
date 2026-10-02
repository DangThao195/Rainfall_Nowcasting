from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .data import DataMetadata, inverse_normalize
from .metrics import NowcastMetrics


@torch.inference_mode()
def evaluate_nowcasts(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor]],
    metadata: DataMetadata,
    land_mask: Tensor,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, object]:
    model.eval()
    model_metrics = NowcastMetrics(land_mask, metadata.forecast_steps)
    persistence_metrics = NowcastMetrics(land_mask, metadata.forecast_steps)

    for batch_index, (inputs, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        prediction = model(inputs, forecast_steps=metadata.forecast_steps)
        persistence = inputs[:, -1:].expand(-1, metadata.forecast_steps, -1, -1, -1)
        target_mm = inverse_normalize(targets, metadata)
        model_metrics.update(inverse_normalize(prediction, metadata), target_mm)
        persistence_metrics.update(inverse_normalize(persistence, metadata), target_mm)

    return {
        "convlstm": model_metrics.compute(),
        "persistence": persistence_metrics.compute(),
    }
