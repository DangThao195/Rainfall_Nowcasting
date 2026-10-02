from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import netCDF4
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DataMetadata:
    input_steps: int
    forecast_steps: int
    time_step_minutes: int
    train_log_mean: float
    train_log_std: float

    @classmethod
    def from_file(cls, path: str | Path) -> DataMetadata:
        with Path(path).open(encoding="utf-8") as file:
            raw = json.load(file)
        normalization = raw["normalization"]
        return cls(
            input_steps=int(raw["input_steps"]),
            forecast_steps=int(raw["forecast_steps"]),
            time_step_minutes=int(raw["time_step_minutes"]),
            train_log_mean=float(normalization["train_log_mean"]),
            train_log_std=float(normalization["train_log_std"]),
        )

    def to_dict(self) -> dict[str, int | float]:
        return {
            "input_steps": self.input_steps,
            "forecast_steps": self.forecast_steps,
            "time_step_minutes": self.time_step_minutes,
            "train_log_mean": self.train_log_mean,
            "train_log_std": self.train_log_std,
        }


class ImergWindowDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(
        self,
        data_path: str | Path,
        starts_path: str | Path,
        input_steps: int,
        forecast_steps: int,
        variable: str = "rainfall_normalized",
    ) -> None:
        self.data_path = Path(data_path)
        self.starts = np.load(starts_path).astype(np.int64, copy=False)
        self.input_steps = input_steps
        self.forecast_steps = forecast_steps
        self.variable = variable
        self._file: netCDF4.Dataset | None = None
        self._rainfall: Any | None = None

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        rainfall = self._get_rainfall()
        start = int(self.starts[index])
        stop = start + self.input_steps + self.forecast_steps
        window = np.asarray(rainfall[start:stop], dtype=np.float32)
        expected = self.input_steps + self.forecast_steps
        if window.shape[0] != expected:
            message = f"Window {index} contains {window.shape[0]} frames; expected {expected}"
            raise IndexError(message)
        frames = torch.from_numpy(window).unsqueeze(1)
        return frames[: self.input_steps], frames[self.input_steps :]

    def _get_rainfall(self) -> Any:
        if self._file is None:
            self._file = netCDF4.Dataset(self.data_path, mode="r")
            self._rainfall = self._file.variables[self.variable]
        return self._rainfall

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._rainfall = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_file"] = None
        state["_rainfall"] = None
        return state

    def __del__(self) -> None:
        self.close()


def load_spatial_field(path: str | Path, variable: str) -> Tensor:
    with netCDF4.Dataset(path, mode="r") as dataset:
        values = np.asarray(dataset.variables[variable][:], dtype=np.float32)
    return torch.from_numpy(values)


def inverse_normalize(values: Tensor, metadata: DataMetadata) -> Tensor:
    rainfall = torch.expm1(values * metadata.train_log_std + metadata.train_log_mean)
    return rainfall.clamp_min_(0.0)


def normalized_rain_threshold(rainfall_mm: float, metadata: DataMetadata) -> float:
    return (float(np.log1p(rainfall_mm)) - metadata.train_log_mean) / metadata.train_log_std
