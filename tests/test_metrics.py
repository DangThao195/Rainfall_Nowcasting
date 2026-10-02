from __future__ import annotations

import pytest
import torch

from rainfall_nowcasting.metrics import NowcastMetrics


def test_perfect_forecast_metrics() -> None:
    mask = torch.ones(5, 4)
    target = torch.full((2, 4, 1, 5, 4), 3.0)
    metrics = NowcastMetrics(mask, forecast_steps=4)

    metrics.update(target, target)
    result = metrics.compute()["aggregate"]

    assert result["mae_mm"] == pytest.approx(0.0)
    assert result["rmse_mm"] == pytest.approx(0.0)
    assert result["csi_0.1mm"] == pytest.approx(1.0)
    assert result["csi_2.5mm"] == pytest.approx(1.0)
    assert result["fss_2.5mm_3x3"] == pytest.approx(1.0)


def test_metrics_ignore_pixels_outside_land_mask() -> None:
    mask = torch.tensor([[1, 0], [0, 0]])
    target = torch.zeros(1, 4, 1, 2, 2)
    prediction = target.clone()
    prediction[..., 0, 1] = 100.0
    metrics = NowcastMetrics(mask, forecast_steps=4)

    metrics.update(prediction, target)
    result = metrics.compute()["aggregate"]

    assert result["mae_mm"] == pytest.approx(0.0)
    assert result["rmse_mm"] == pytest.approx(0.0)
