from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import cast

import netCDF4
import numpy as np
import torch
from matplotlib import colormaps
from matplotlib.colors import BoundaryNorm, Normalize, TwoSlopeNorm
from matplotlib.figure import Figure
from numpy.typing import NDArray
from torch import Tensor

from .checkpoints import load_trained_model
from .data import DataMetadata, inverse_normalize, load_spatial_field
from .model import ConvLSTMNowcaster
from .training import select_device
from .visualize import RAINFALL_LEVELS, GridContext, calculate_sample_metrics

MINUTES_PER_DAY = 24 * 60
MAX_SNAPSHOTS = 8


@dataclass(frozen=True)
class DaySequence:
    observed_normalized: Tensor
    target_normalized: Tensor
    context: GridContext
    observed_date: str
    forecast_date: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict and visualize all 48 half-hour frames of the next IMERG day"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("IMERG_data/prepared"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--date",
        required=True,
        help="Complete observation date in YYYY-MM-DD format; the next day is predicted",
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="test",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--show-full-grid", action="store_true")
    return parser.parse_args()


def _frames_per_day(metadata: DataMetadata) -> int:
    if MINUTES_PER_DAY % metadata.time_step_minutes != 0:
        raise ValueError("time_step_minutes must divide evenly into one day")
    return MINUTES_PER_DAY // metadata.time_step_minutes


def _format_netcdf_times(time_variable: netCDF4.Variable) -> tuple[str, ...]:
    calendar = cast(str, getattr(time_variable, "calendar", "standard"))
    dates = cast(
        NDArray[np.object_],
        netCDF4.num2date(
            time_variable[:],
            units=time_variable.units,
            calendar=calendar,
        ),
    )
    return tuple(date.strftime("%Y-%m-%d %H:%M") for date in dates)


def load_day_sequence(
    data_path: Path,
    observed_date: str,
    metadata: DataMetadata,
) -> DaySequence:
    try:
        observed_start = datetime.strptime(observed_date, "%Y-%m-%d")
    except ValueError as error:
        raise ValueError("date must use YYYY-MM-DD format") from error

    frames_per_day = _frames_per_day(metadata)
    required_frames = frames_per_day * 2
    with netCDF4.Dataset(data_path, mode="r") as netcdf:
        times = _format_netcdf_times(netcdf.variables["time"])
        start_label = observed_start.strftime("%Y-%m-%d %H:%M")
        try:
            start = times.index(start_label)
        except ValueError as error:
            raise ValueError(f"Date {observed_date} is not present in {data_path.name}") from error
        stop = start + required_frames
        if stop > len(times):
            raise ValueError(
                f"{observed_date} and its following day are not both complete in {data_path.name}"
            )

        expected_times = tuple(
            (observed_start + timedelta(minutes=metadata.time_step_minutes * step)).strftime(
                "%Y-%m-%d %H:%M"
            )
            for step in range(required_frames)
        )
        selected_times = times[start:stop]
        if selected_times != expected_times:
            raise ValueError("The selected two-day period contains missing or irregular time steps")

        rainfall = np.asarray(netcdf.variables["rainfall_normalized"][start:stop], dtype=np.float32)
        latitude = np.asarray(netcdf.variables["lat"][:], dtype=np.float32)
        longitude = np.asarray(netcdf.variables["lon"][:], dtype=np.float32)

    observed = torch.from_numpy(rainfall[:frames_per_day]).unsqueeze(1)
    target = torch.from_numpy(rainfall[frames_per_day:]).unsqueeze(1)
    forecast_date = (observed_start + timedelta(days=1)).strftime("%Y-%m-%d")
    context = GridContext(
        latitude=latitude,
        longitude=longitude,
        target_times=selected_times[frames_per_day:],
    )
    return DaySequence(observed, target, context, observed_date, forecast_date)


def rollout_forecast(
    model: ConvLSTMNowcaster,
    observed: Tensor,
    metadata: DataMetadata,
    device: torch.device,
    total_steps: int,
) -> Tensor:
    if observed.ndim != 4:
        raise ValueError("observed must have shape [time, channel, height, width]")
    if observed.shape[0] < metadata.input_steps:
        raise ValueError(f"At least {metadata.input_steps} observed frames are required")
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")

    history = observed[-metadata.input_steps :].unsqueeze(0).to(device)
    blocks: list[Tensor] = []
    generated_steps = 0
    model.eval()
    with torch.inference_mode():
        while generated_steps < total_steps:
            block_steps = min(metadata.forecast_steps, total_steps - generated_steps)
            block = model(history, forecast_steps=block_steps)
            blocks.append(block[0].cpu())
            history = torch.cat((history, block), dim=1)[:, -metadata.input_steps :]
            generated_steps += block_steps
    return torch.cat(blocks, dim=0)


def _map_extent(context: GridContext) -> list[float]:
    return [
        float(context.longitude.min()),
        float(context.longitude.max()),
        float(context.latitude.min()),
        float(context.latitude.max()),
    ]


def render_day_snapshots(
    target: NDArray[np.float32],
    prediction: NDArray[np.float32],
    display_mask: NDArray[np.bool_],
    context: GridContext,
    per_step_metrics: list[dict[str, float]],
    time_step_minutes: int,
    output_path: Path,
) -> None:
    snapshot_indices = np.linspace(
        0, target.shape[0] - 1, min(MAX_SNAPSHOTS, target.shape[0]), dtype=np.int64
    )
    figure = Figure(figsize=(11, 3.7 * len(snapshot_indices)), constrained_layout=True)
    axes = figure.subplots(len(snapshot_indices), 2, squeeze=False)
    color_map = colormaps["turbo"].with_extremes(under="white", bad="#d9d9d9")
    color_norm = BoundaryNorm(RAINFALL_LEVELS, color_map.N, extend="max")
    extent = _map_extent(context)
    image = None
    for row, step_value in enumerate(snapshot_indices):
        step = int(step_value)
        observed_frame = np.ma.masked_where(~display_mask, target[step])
        predicted_frame = np.ma.masked_where(~display_mask, prediction[step])
        for column, frame in enumerate((observed_frame, predicted_frame)):
            image = axes[row, column].imshow(
                frame,
                origin="lower",
                extent=extent,
                cmap=color_map,
                norm=color_norm,
                aspect="auto",
            )
            axes[row, column].set_xlabel("Longitude")
            axes[row, column].set_ylabel("Latitude")
        lead_hours = (step + 1) * time_step_minutes / 60
        axes[row, 0].set_title(f"Ground truth +{lead_hours:g} h\n{context.target_times[step]}")
        metric = per_step_metrics[step]
        axes[row, 1].set_title(
            f"Rolling ConvLSTM +{lead_hours:g} h\n"
            f"MAE={metric['mae_mm']:.3f}, RMSE={metric['rmse_mm']:.3f} mm"
        )
    if image is not None:
        figure.colorbar(
            image,
            ax=axes,
            ticks=RAINFALL_LEVELS,
            label=f"Rainfall (mm/{time_step_minutes} min)",
            shrink=0.8,
        )
    figure.suptitle("Next-day IMERG rainfall: ground truth vs rolling prediction", fontsize=16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)


def render_daily_accumulation(
    target: NDArray[np.float32],
    prediction: NDArray[np.float32],
    display_mask: NDArray[np.bool_],
    context: GridContext,
    output_path: Path,
) -> None:
    observed_total = target.sum(axis=0)
    predicted_total = prediction.sum(axis=0)
    difference = predicted_total - observed_total
    valid_totals = np.concatenate((observed_total[display_mask], predicted_total[display_mask]))
    total_max = max(float(np.percentile(valid_totals, 99.5)), 1.0)
    difference_max = max(float(np.percentile(np.abs(difference[display_mask]), 99.5)), 1.0)

    figure = Figure(figsize=(15, 5.2), constrained_layout=True)
    axes = figure.subplots(1, 3, squeeze=False)[0]
    extent = _map_extent(context)
    total_map = colormaps["turbo"].with_extremes(under="white", bad="#d9d9d9")
    total_norm = Normalize(vmin=0.0, vmax=total_max)
    total_image = None
    for axis, frame, title in zip(
        axes[:2],
        (observed_total, predicted_total),
        ("Ground-truth daily total", "Predicted daily total"),
        strict=True,
    ):
        total_image = axis.imshow(
            np.ma.masked_where(~display_mask, frame),
            origin="lower",
            extent=extent,
            cmap=total_map,
            norm=total_norm,
            aspect="auto",
        )
        axis.set_title(title)
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Latitude")
    error_image = axes[2].imshow(
        np.ma.masked_where(~display_mask, difference),
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-difference_max, vcenter=0.0, vmax=difference_max),
        aspect="auto",
    )
    axes[2].set_title("Prediction error (predicted - truth)")
    axes[2].set_xlabel("Longitude")
    axes[2].set_ylabel("Latitude")
    if total_image is not None:
        figure.colorbar(total_image, ax=axes[:2], label="Daily rainfall (mm)", shrink=0.8)
    figure.colorbar(error_image, ax=axes[2], label="Daily rainfall error (mm)", shrink=0.8)
    figure.suptitle("Next-day accumulated rainfall", fontsize=16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)


def render_mean_time_series(
    target: NDArray[np.float32],
    prediction: NDArray[np.float32],
    metric_mask: NDArray[np.bool_],
    time_step_minutes: int,
    output_path: Path,
) -> None:
    observed_mean = target[:, metric_mask].mean(axis=1)
    predicted_mean = prediction[:, metric_mask].mean(axis=1)
    lead_hours = (np.arange(target.shape[0]) + 1) * time_step_minutes / 60
    figure = Figure(figsize=(11, 4.8), constrained_layout=True)
    axis = figure.subplots()
    axis.plot(lead_hours, observed_mean, label="Ground truth", linewidth=2)
    axis.plot(lead_hours, predicted_mean, label="Rolling prediction", linewidth=2)
    axis.set_xlabel("Forecast lead time (hours)")
    axis.set_ylabel(f"Vietnam mean rainfall (mm/{time_step_minutes} min)")
    axis.set_title("Spatial-mean rainfall through the forecast day")
    axis.grid(alpha=0.3)
    axis.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)


def calculate_day_metrics(
    prediction: NDArray[np.float32],
    target: NDArray[np.float32],
    metric_mask: NDArray[np.bool_],
) -> dict[str, float]:
    difference = prediction[:, metric_mask] - target[:, metric_mask]
    observed_total = target[:, metric_mask].sum(axis=0)
    predicted_total = prediction[:, metric_mask].sum(axis=0)
    return {
        "mae_mm_per_frame": float(np.mean(np.abs(difference))),
        "rmse_mm_per_frame": float(np.sqrt(np.mean(np.square(difference)))),
        "bias_mm_per_frame": float(np.mean(difference)),
        "observed_mean_daily_total_mm": float(np.mean(observed_total)),
        "predicted_mean_daily_total_mm": float(np.mean(predicted_total)),
        "daily_total_mae_mm": float(np.mean(np.abs(predicted_total - observed_total))),
    }


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    metadata = DataMetadata.from_file(args.data_dir / "dataset_metadata.json")
    data_path = args.data_dir / f"imerg_{args.split}_normalized.nc"
    sequence = load_day_sequence(data_path, args.date, metadata)
    metric_mask = (
        load_spatial_field(args.data_dir / "vietnam_land_mask.nc", "land_mask").bool().numpy()
    )
    display_mask = np.ones_like(metric_mask) if args.show_full_grid else metric_mask

    model = load_trained_model(args.checkpoint, device)
    forecast_steps = _frames_per_day(metadata)
    prediction = rollout_forecast(
        model,
        sequence.observed_normalized,
        metadata,
        device,
        total_steps=forecast_steps,
    )
    target_mm = inverse_normalize(sequence.target_normalized, metadata).squeeze(1).numpy()
    prediction_mm = inverse_normalize(prediction, metadata).squeeze(1).numpy()
    per_step_metrics = calculate_sample_metrics(prediction_mm, target_mm, metric_mask)
    aggregate_metrics = calculate_day_metrics(prediction_mm, target_mm, metric_mask)

    output_dir = args.output_dir or args.checkpoint.parent / "day_ahead_visualizations"
    file_stem = f"{sequence.observed_date}_to_{sequence.forecast_date}"
    snapshots_path = output_dir / f"{file_stem}_snapshots.png"
    accumulation_path = output_dir / f"{file_stem}_accumulation.png"
    time_series_path = output_dir / f"{file_stem}_time_series.png"
    metrics_path = output_dir / f"{file_stem}_metrics.json"
    render_day_snapshots(
        target_mm,
        prediction_mm,
        display_mask,
        sequence.context,
        per_step_metrics,
        metadata.time_step_minutes,
        snapshots_path,
    )
    render_daily_accumulation(
        target_mm,
        prediction_mm,
        display_mask,
        sequence.context,
        accumulation_path,
    )
    render_mean_time_series(
        target_mm,
        prediction_mm,
        metric_mask,
        metadata.time_step_minutes,
        time_series_path,
    )

    result = {
        "observed_date": sequence.observed_date,
        "forecast_date": sequence.forecast_date,
        "split": args.split,
        "observed_context_frames_available": int(sequence.observed_normalized.shape[0]),
        "model_input_frames_used": metadata.input_steps,
        "forecast_frames": forecast_steps,
        "rolling_blocks": int(np.ceil(forecast_steps / metadata.forecast_steps)),
        "warning": (
            "The model was trained for a short 4-frame horizon. This 48-frame forecast "
            "feeds predictions back as inputs, so errors can accumulate strongly."
        ),
        "aggregate": aggregate_metrics,
        "per_step": [
            {"time": time, **metrics}
            for time, metrics in zip(sequence.context.target_times, per_step_metrics, strict=True)
        ],
        "outputs": {
            "snapshots": str(snapshots_path),
            "accumulation": str(accumulation_path),
            "time_series": str(time_series_path),
            "metrics": str(metrics_path),
        },
        "device": str(device),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
