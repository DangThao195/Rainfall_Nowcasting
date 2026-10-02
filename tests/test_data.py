from __future__ import annotations

import json
from pathlib import Path

import netCDF4
import numpy as np
import torch

from rainfall_nowcasting.data import DataMetadata, ImergWindowDataset, inverse_normalize


def test_window_dataset_reads_expected_frames(tmp_path: Path) -> None:
    data_path = tmp_path / "data.nc"
    with netCDF4.Dataset(data_path, mode="w") as netcdf:
        netcdf.createDimension("time", 12)
        netcdf.createDimension("lat", 3)
        netcdf.createDimension("lon", 4)
        rainfall = netcdf.createVariable("rainfall_normalized", "f4", ("time", "lat", "lon"))
        values = np.arange(12 * 3 * 4, dtype=np.float32).reshape(12, 3, 4)
        rainfall[:, :, :] = values
    starts_path = tmp_path / "starts.npy"
    np.save(starts_path, np.array([0, 2], dtype=np.int64))

    window_dataset = ImergWindowDataset(data_path, starts_path, input_steps=6, forecast_steps=4)
    inputs, targets = window_dataset[1]

    assert inputs.shape == (6, 1, 3, 4)
    assert targets.shape == (4, 1, 3, 4)
    assert inputs[0, 0, 0, 0].item() == 24.0
    assert targets[-1, 0, -1, -1].item() == 143.0


def test_metadata_and_inverse_transform(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "input_steps": 6,
                "forecast_steps": 4,
                "time_step_minutes": 30,
                "normalization": {"train_log_mean": 0.0, "train_log_std": 1.0},
            }
        ),
        encoding="utf-8",
    )
    metadata = DataMetadata.from_file(metadata_path)

    normalized = torch.tensor([0.0, np.log(2.0)], dtype=torch.float32)
    restored = inverse_normalize(normalized, metadata)

    assert metadata.forecast_steps == 4
    assert torch.allclose(restored, torch.tensor([0.0, 1.0]), atol=1e-6)
