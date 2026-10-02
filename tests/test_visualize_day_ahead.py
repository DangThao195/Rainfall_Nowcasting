from __future__ import annotations

from pathlib import Path

import netCDF4
import numpy as np
import torch

from rainfall_nowcasting.data import DataMetadata
from rainfall_nowcasting.model import ConvLSTMNowcaster, ModelConfig
from rainfall_nowcasting.visualize_day_ahead import load_day_sequence, rollout_forecast


def test_load_day_sequence_reads_two_complete_days(tmp_path: Path) -> None:
    data_path = tmp_path / "two_days.nc"
    with netCDF4.Dataset(data_path, mode="w") as netcdf:
        netcdf.createDimension("time", 96)
        netcdf.createDimension("lat", 2)
        netcdf.createDimension("lon", 3)
        time = netcdf.createVariable("time", "f8", ("time",))
        time.units = "minutes since 2025-01-01 00:00:00"
        time.calendar = "standard"
        time[:] = np.arange(96, dtype=np.float64) * 30
        netcdf.createVariable("lat", "f4", ("lat",))[:] = [10.0, 11.0]
        netcdf.createVariable("lon", "f4", ("lon",))[:] = [105.0, 106.0, 107.0]
        rainfall = netcdf.createVariable("rainfall_normalized", "f4", ("time", "lat", "lon"))
        rainfall[:, :, :] = np.arange(96, dtype=np.float32)[:, None, None]
    metadata = DataMetadata(6, 4, 30, 0.0, 1.0)

    sequence = load_day_sequence(data_path, "2025-01-01", metadata)

    assert sequence.observed_normalized.shape == (48, 1, 2, 3)
    assert sequence.target_normalized.shape == (48, 1, 2, 3)
    assert sequence.target_normalized[0, 0, 0, 0].item() == 48.0
    assert sequence.context.target_times[0] == "2025-01-02 00:00"
    assert sequence.context.target_times[-1] == "2025-01-02 23:30"


def test_rollout_forecast_generates_requested_number_of_frames() -> None:
    torch.manual_seed(7)
    metadata = DataMetadata(6, 4, 30, 0.0, 1.0)
    model = ConvLSTMNowcaster(ModelConfig(hidden_channels=(2,), kernel_size=3))
    observed = torch.rand((48, 1, 5, 4), dtype=torch.float32)

    prediction = rollout_forecast(
        model,
        observed,
        metadata,
        torch.device("cpu"),
        total_steps=10,
    )

    assert prediction.shape == (10, 1, 5, 4)
    assert torch.isfinite(prediction).all()
