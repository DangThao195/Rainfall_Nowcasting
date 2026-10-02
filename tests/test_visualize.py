from __future__ import annotations

from pathlib import Path

import netCDF4
import numpy as np

from rainfall_nowcasting.data import DataMetadata
from rainfall_nowcasting.visualize import (
    GridContext,
    calculate_sample_metrics,
    render_comparison,
    select_rainiest_window,
)


def test_select_rainiest_window_uses_forecast_frames(tmp_path: Path) -> None:
    path = tmp_path / "test.nc"
    rainfall = np.zeros((8, 2, 2), dtype=np.float32)
    rainfall[4:6] = np.log1p(5.0)
    with netCDF4.Dataset(path, mode="w") as netcdf:
        netcdf.createDimension("time", 8)
        netcdf.createDimension("lat", 2)
        netcdf.createDimension("lon", 2)
        variable = netcdf.createVariable("rainfall_normalized", "f4", ("time", "lat", "lon"))
        variable[:, :, :] = rainfall
    metadata = DataMetadata(2, 2, 30, 0.0, 1.0)

    selected = select_rainiest_window(
        path,
        np.array([0, 2], dtype=np.int64),
        metadata,
        np.ones((2, 2), dtype=np.bool_),
    )

    assert selected == 1


def test_render_comparison_writes_png_and_metrics(tmp_path: Path) -> None:
    target = np.ones((4, 3, 2), dtype=np.float32)
    prediction = target + 0.5
    mask = np.ones((3, 2), dtype=np.bool_)
    metadata = DataMetadata(6, 4, 30, 0.0, 1.0)
    context = GridContext(
        latitude=np.array([8.0, 9.0, 10.0], dtype=np.float32),
        longitude=np.array([102.0, 103.0], dtype=np.float32),
        target_times=("t1", "t2", "t3", "t4"),
    )
    metrics = calculate_sample_metrics(prediction, target, mask)
    output = tmp_path / "comparison.png"

    render_comparison(target, prediction, mask, context, metadata, metrics, output)

    assert output.is_file()
    assert output.stat().st_size > 0
    assert metrics[0]["mae_mm"] == 0.5
    assert metrics[0]["rmse_mm"] == 0.5
