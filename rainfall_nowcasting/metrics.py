from __future__ import annotations

from collections.abc import Iterable
from typing import TypedDict

import numpy as np
from torch import Tensor
from torch.nn import functional as F


class MetricsResult(TypedDict):
    aggregate: dict[str, float]
    per_lead: list[dict[str, float]]


class NowcastMetrics:
    def __init__(
        self,
        land_mask: Tensor,
        forecast_steps: int,
        thresholds: Iterable[float] = (0.1, 2.5, 10.0),
        fss_windows: Iterable[int] = (3, 5),
    ) -> None:
        self.mask = land_mask.bool()[None, None, None]
        self.forecast_steps = forecast_steps
        self.thresholds = tuple(thresholds)
        self.fss_windows = tuple(fss_windows)
        self.absolute_error = np.zeros(forecast_steps, dtype=np.float64)
        self.squared_error = np.zeros(forecast_steps, dtype=np.float64)
        self.count = np.zeros(forecast_steps, dtype=np.float64)
        self.hits = {threshold: np.zeros(forecast_steps) for threshold in self.thresholds}
        self.misses = {threshold: np.zeros(forecast_steps) for threshold in self.thresholds}
        self.false_alarms = {threshold: np.zeros(forecast_steps) for threshold in self.thresholds}
        self.fss_numerator = {
            (threshold, window): np.zeros(forecast_steps)
            for threshold in self.thresholds
            for window in self.fss_windows
        }
        self.fss_denominator = {key: np.zeros(forecast_steps) for key in self.fss_numerator}

    def update(self, prediction: Tensor, target: Tensor) -> None:
        prediction = prediction.detach().cpu()
        target = target.detach().cpu()
        mask = self.mask.expand_as(target)
        difference = prediction - target
        reduce_dims = (0, 2, 3, 4)
        self.absolute_error += (difference.abs() * mask).sum(dim=reduce_dims).numpy()
        self.squared_error += (difference.square() * mask).sum(dim=reduce_dims).numpy()
        self.count += mask.sum(dim=reduce_dims).numpy()

        for threshold in self.thresholds:
            predicted_event = prediction >= threshold
            observed_event = target >= threshold
            hits = (predicted_event & observed_event & mask).sum(dim=reduce_dims)
            misses = (~predicted_event & observed_event & mask).sum(dim=reduce_dims)
            self.hits[threshold] += hits.numpy()
            self.misses[threshold] += misses.numpy()
            self.false_alarms[threshold] += (
                (predicted_event & ~observed_event & mask).sum(dim=reduce_dims).numpy()
            )
            self._update_fss(predicted_event, observed_event, mask, threshold)

    def _update_fss(
        self,
        predicted_event: Tensor,
        observed_event: Tensor,
        mask: Tensor,
        threshold: float,
    ) -> None:
        batch, steps, _, height, width = predicted_event.shape
        predicted = predicted_event.float().reshape(batch * steps, 1, height, width)
        observed = observed_event.float().reshape(batch * steps, 1, height, width)
        land = mask.reshape(batch * steps, 1, height, width)
        for window in self.fss_windows:
            predicted_fraction = F.avg_pool2d(
                predicted, window, stride=1, padding=window // 2, count_include_pad=False
            )
            observed_fraction = F.avg_pool2d(
                observed, window, stride=1, padding=window // 2, count_include_pad=False
            )
            numerator = (
                ((predicted_fraction - observed_fraction).square() * land)
                .reshape(batch, steps, -1)
                .sum(dim=(0, 2))
            )
            denominator = (
                ((predicted_fraction.square() + observed_fraction.square()) * land)
                .reshape(batch, steps, -1)
                .sum(dim=(0, 2))
            )
            key = (threshold, window)
            self.fss_numerator[key] += numerator.numpy()
            self.fss_denominator[key] += denominator.numpy()

    def compute(self) -> MetricsResult:
        aggregate = self._compute_slice(slice(None))
        per_lead = [self._compute_slice(step) for step in range(self.forecast_steps)]
        return {"aggregate": aggregate, "per_lead": per_lead}

    def _compute_slice(self, index: int | slice) -> dict[str, float]:
        count = float(np.sum(self.count[index]))
        metrics = {
            "mae_mm": float(np.sum(self.absolute_error[index]) / max(count, 1.0)),
            "rmse_mm": float(np.sqrt(np.sum(self.squared_error[index]) / max(count, 1.0))),
        }
        for threshold in self.thresholds:
            hits = float(np.sum(self.hits[threshold][index]))
            misses = float(np.sum(self.misses[threshold][index]))
            false_alarms = float(np.sum(self.false_alarms[threshold][index]))
            metrics[f"csi_{threshold:g}mm"] = _safe_ratio(hits, hits + misses + false_alarms)
            for window in self.fss_windows:
                key = (threshold, window)
                numerator = float(np.sum(self.fss_numerator[key][index]))
                denominator = float(np.sum(self.fss_denominator[key][index]))
                metrics[f"fss_{threshold:g}mm_{window}x{window}"] = (
                    1.0 - _safe_ratio(numerator, denominator) if denominator else 1.0
                )
        return metrics


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
