from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import netCDF4
import numpy as np
import torch
from matplotlib import colormaps
from matplotlib.colors import BoundaryNorm, TwoSlopeNorm
from matplotlib.figure import Figure
from numpy.typing import NDArray

from .checkpoints import load_trained_model
from .data import DataMetadata, ImergWindowDataset, inverse_normalize, load_spatial_field
from .training import select_device

RAINFALL_LEVELS = np.array([0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 50.0])


@dataclass(frozen=True)
class GridContext:
    latitude: NDArray[np.float32]
    longitude: NDArray[np.float32]
    target_times: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render IMERG truth, ConvLSTM prediction, and their difference"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("IMERG_data/prepared"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--show-full-grid", action="store_true")
    return parser.parse_args()


def select_rainiest_window(
    data_path: Path,
    starts: NDArray[np.int64],
    metadata: DataMetadata,
    land_mask: NDArray[np.bool_],
    chunk_size: int = 256,
) -> int:
    with netCDF4.Dataset(data_path, mode="r") as netcdf:
        variable = netcdf.variables["rainfall_normalized"]
        frame_mean = np.empty(variable.shape[0], dtype=np.float64)
        for chunk_start in range(0, variable.shape[0], chunk_size):
            chunk_stop = min(chunk_start + chunk_size, variable.shape[0])
            normalized = np.asarray(variable[chunk_start:chunk_stop], dtype=np.float32)
            rainfall = np.expm1(normalized * metadata.train_log_std + metadata.train_log_mean).clip(
                min=0.0
            )
            frame_mean[chunk_start:chunk_stop] = rainfall[:, land_mask].mean(axis=1)
    offsets = metadata.input_steps + np.arange(metadata.forecast_steps)
    target_indices = starts[:, None] + offsets[None, :]
    return int(np.argmax(frame_mean[target_indices].mean(axis=1)))


def load_grid_context(data_path: Path, target_indices: NDArray[np.int64]) -> GridContext:
    with netCDF4.Dataset(data_path, mode="r") as netcdf:
        latitude = np.asarray(netcdf.variables["lat"][:], dtype=np.float32)
        longitude = np.asarray(netcdf.variables["lon"][:], dtype=np.float32)
        time_variable = netcdf.variables["time"]
        calendar = cast(str, getattr(time_variable, "calendar", "standard"))
        dates = cast(
            NDArray[np.object_],
            netCDF4.num2date(
                time_variable[target_indices],
                units=time_variable.units,
                calendar=calendar,
            ),
        )
    target_times = tuple(str(date)[:16] for date in dates)
    return GridContext(latitude, longitude, target_times)


def calculate_sample_metrics(
    prediction: NDArray[np.float32],
    target: NDArray[np.float32],
    land_mask: NDArray[np.bool_],
) -> list[dict[str, float]]:
    metrics: list[dict[str, float]] = []
    for step in range(target.shape[0]):
        difference = prediction[step][land_mask] - target[step][land_mask]
        metrics.append(
            {
                "mae_mm": float(np.mean(np.abs(difference))),
                "rmse_mm": float(np.sqrt(np.mean(np.square(difference)))),
                "observed_max_mm": float(np.max(target[step][land_mask])),
                "predicted_max_mm": float(np.max(prediction[step][land_mask])),
            }
        )
    return metrics


def render_comparison(
    target: NDArray[np.float32],
    prediction: NDArray[np.float32],
    land_mask: NDArray[np.bool_],
    context: GridContext,
    metadata: DataMetadata,
    sample_metrics: list[dict[str, float]],
    output_path: Path,
) -> None:
    figure = Figure(figsize=(16, 17), constrained_layout=True)
    axes = figure.subplots(metadata.forecast_steps, 3, squeeze=False)
    color_map = colormaps["turbo"].with_extremes(under="white", bad="#d9d9d9")
    color_norm = BoundaryNorm(RAINFALL_LEVELS, color_map.N, extend="max")
    difference = prediction - target
    difference_limit = max(float(np.max(np.abs(difference[:, land_mask]))), 0.1)
    difference_map = colormaps["RdBu_r"].with_extremes(bad="#d9d9d9")
    difference_norm = TwoSlopeNorm(
        vmin=-difference_limit,
        vcenter=0.0,
        vmax=difference_limit,
    )
    extent = [
        float(context.longitude.min()),
        float(context.longitude.max()),
        float(context.latitude.min()),
        float(context.latitude.max()),
    ]
    rainfall_image = None
    difference_image = None
    for step in range(metadata.forecast_steps):
        observed = np.ma.masked_where(~land_mask, target[step])
        predicted = np.ma.masked_where(~land_mask, prediction[step])
        for column, frame in enumerate((observed, predicted)):
            rainfall_image = axes[step, column].imshow(
                frame,
                origin="lower",
                extent=extent,
                cmap=color_map,
                norm=color_norm,
                aspect="auto",
            )
            axes[step, column].set_xlabel("Longitude")
            axes[step, column].set_ylabel("Latitude")
        difference_image = axes[step, 2].imshow(
            np.ma.masked_where(~land_mask, difference[step]),
            origin="lower",
            extent=extent,
            cmap=difference_map,
            norm=difference_norm,
            aspect="auto",
        )
        axes[step, 2].set_xlabel("Longitude")
        axes[step, 2].set_ylabel("Latitude")
        lead_minutes = (step + 1) * metadata.time_step_minutes
        axes[step, 0].set_title(f"Ground truth +{lead_minutes} min\n{context.target_times[step]}")
        metric = sample_metrics[step]
        axes[step, 1].set_title(
            f"ConvLSTM +{lead_minutes} min\n"
            f"MAE={metric['mae_mm']:.3f}, RMSE={metric['rmse_mm']:.3f} mm"
        )
        axes[step, 2].set_title(f"Difference +{lead_minutes} min\nPredicted - ground truth")
    if rainfall_image is not None:
        figure.colorbar(
            rainfall_image,
            ax=axes[:, :2],
            ticks=RAINFALL_LEVELS,
            label="Rainfall (mm/30 min)",
            shrink=0.8,
        )
    if difference_image is not None:
        figure.colorbar(
            difference_image,
            ax=axes[:, 2],
            label="Difference (mm/30 min)",
            shrink=0.8,
        )
    figure.suptitle("IMERG rainfall nowcast: truth, prediction, and error", fontsize=16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    metadata = DataMetadata.from_file(args.data_dir / "dataset_metadata.json")
    dataset = ImergWindowDataset(
        args.data_dir / "imerg_test_normalized.nc",
        args.data_dir / "test_window_starts.npy",
        metadata.input_steps,
        metadata.forecast_steps,
    )
    metric_mask = (
        load_spatial_field(args.data_dir / "vietnam_land_mask.nc", "land_mask").bool().numpy()
    )
    sample_index = args.sample_index
    if sample_index is None:
        sample_index = select_rainiest_window(
            dataset.data_path, dataset.starts, metadata, metric_mask
        )
    if not 0 <= sample_index < len(dataset):
        raise ValueError(f"sample-index must be between 0 and {len(dataset) - 1}")

    inputs, targets = dataset[sample_index]
    model = load_trained_model(args.checkpoint, device)
    model.eval()
    with torch.inference_mode():
        prediction = model(inputs.unsqueeze(0).to(device), forecast_steps=metadata.forecast_steps)[
            0
        ].cpu()
    target_mm = inverse_normalize(targets, metadata).squeeze(1).numpy()
    prediction_mm = inverse_normalize(prediction, metadata).squeeze(1).numpy()
    sample_metrics = calculate_sample_metrics(prediction_mm, target_mm, metric_mask)

    start = int(dataset.starts[sample_index])
    target_indices = start + metadata.input_steps + np.arange(metadata.forecast_steps)
    context = load_grid_context(dataset.data_path, target_indices)
    display_mask = np.ones_like(metric_mask) if args.show_full_grid else metric_mask
    output_dir = args.output_dir or args.checkpoint.parent / "visualizations"
    image_path = output_dir / f"test_window_{sample_index:04d}.png"
    metrics_path = output_dir / f"test_window_{sample_index:04d}.json"
    render_comparison(
        target_mm,
        prediction_mm,
        display_mask,
        context,
        metadata,
        sample_metrics,
        image_path,
    )
    result = {
        "sample_index": sample_index,
        "window_start": start,
        "target_times": list(context.target_times),
        "per_lead": sample_metrics,
        "image": str(image_path),
        "device": str(device),
    }
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
