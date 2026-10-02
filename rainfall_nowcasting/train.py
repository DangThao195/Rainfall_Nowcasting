from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sized
from pathlib import Path
from typing import cast

import torch
from matplotlib.figure import Figure
from torch.utils.data import DataLoader

from .data import (
    DataMetadata,
    ImergWindowDataset,
    load_spatial_field,
    normalized_rain_threshold,
)
from .evaluation import evaluate_nowcasts
from .losses import WeightedRainfallLoss
from .model import ConvLSTMNowcaster, ModelConfig
from .training import (
    count_parameters,
    run_epoch,
    seed_everything,
    select_device,
    teacher_forcing_ratio,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an IMERG ConvLSTM nowcasting model")
    parser.add_argument("--data-dir", type=Path, default=Path("IMERG_data/prepared"))
    parser.add_argument("--output-dir", type=Path, default=Path("IMERG_data/outputs/convlstm"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden-channels", nargs="+", type=int, default=[16, 32])
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--rain-threshold", type=float, default=0.1)
    parser.add_argument("--rain-boost", type=float, default=2.0)
    parser.add_argument("--huber-beta", type=float, default=0.5)
    parser.add_argument("--teacher-forcing-start", type=float, default=0.5)
    parser.add_argument("--teacher-forcing-end", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    return parser.parse_args()


def make_loader(
    data_dir: Path,
    split: str,
    metadata: DataMetadata,
    batch_size: int,
    workers: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    dataset = ImergWindowDataset(
        data_dir / f"imerg_{split}_normalized.nc",
        data_dir / f"{split}_window_starts.npy",
        metadata.input_steps,
        metadata.forecast_steps,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


def save_checkpoint(
    path: Path,
    model: ConvLSTMNowcaster,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_loss: float,
    metadata: DataMetadata,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "validation_loss": validation_loss,
            "model_config": model.config.to_dict(),
            "metadata": metadata.to_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def save_history(history: list[dict[str, float]], output_dir: Path) -> None:
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    epochs = [entry["epoch"] for entry in history]
    figure = Figure(figsize=(7, 4))
    axis = figure.subplots()
    axis.plot(epochs, [entry["train_loss"] for entry in history], label="train")
    axis.plot(epochs, [entry["validation_loss"] for entry in history], label="validation")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Weighted Huber loss")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "loss_curve.png", dpi=160)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = select_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = DataMetadata.from_file(args.data_dir / "dataset_metadata.json")
    train_loader = make_loader(
        args.data_dir, "train", metadata, args.batch_size, args.num_workers, True, device
    )
    validation_loader = make_loader(
        args.data_dir, "validation", metadata, args.batch_size, args.num_workers, False, device
    )
    test_loader = make_loader(
        args.data_dir, "test", metadata, args.batch_size, args.num_workers, False, device
    )
    model = ConvLSTMNowcaster(
        ModelConfig(hidden_channels=tuple(args.hidden_channels), kernel_size=args.kernel_size)
    ).to(device)
    spatial_weight = load_spatial_field(args.data_dir / "vietnam_land_weight.nc", "W_land")
    loss_function = WeightedRainfallLoss(
        spatial_weight,
        normalized_rain_threshold(args.rain_threshold, metadata),
        rain_boost=args.rain_boost,
        beta=args.huber_beta,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    amp_enabled = device.type == "cuda" and not args.disable_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    best_path = args.output_dir / "best.pt"
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    stale_epochs = 0

    print(
        json.dumps(
            {
                "device": str(device),
                "amp": amp_enabled,
                "parameters": count_parameters(model.parameters()),
                "train_windows": len(cast(Sized, train_loader.dataset)),
                "validation_windows": len(cast(Sized, validation_loader.dataset)),
                "test_windows": len(cast(Sized, test_loader.dataset)),
            }
        )
    )
    for epoch in range(args.epochs):
        started = time.perf_counter()
        forcing = teacher_forcing_ratio(
            epoch, args.epochs, args.teacher_forcing_start, args.teacher_forcing_end
        )
        train_loss = run_epoch(
            model,
            train_loader,
            loss_function,
            device,
            metadata.forecast_steps,
            optimizer=optimizer,
            scaler=scaler,
            teacher_forcing_ratio=forcing,
            max_batches=args.max_train_batches,
            gradient_clip=args.gradient_clip,
        )
        validation_loss = run_epoch(
            model,
            validation_loader,
            loss_function,
            device,
            metadata.forecast_steps,
            max_batches=args.max_val_batches,
        )
        record = {
            "epoch": float(epoch + 1),
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "teacher_forcing_ratio": forcing,
            "seconds": time.perf_counter() - started,
        }
        history.append(record)
        print(json.dumps(record))
        save_checkpoint(
            args.output_dir / "last.pt", model, optimizer, epoch + 1, validation_loss, metadata
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            stale_epochs = 0
            save_checkpoint(best_path, model, optimizer, epoch + 1, validation_loss, metadata)
        else:
            stale_epochs += 1
        save_history(history, args.output_dir)
        if stale_epochs >= args.patience:
            print(f"Early stopping after {epoch + 1} epochs")
            break

    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    land_mask = load_spatial_field(args.data_dir / "vietnam_land_mask.nc", "land_mask")
    metrics = evaluate_nowcasts(
        model, test_loader, metadata, land_mask, device, args.max_test_batches
    )
    metrics["best_validation_loss"] = best_loss
    metrics["best_checkpoint"] = str(best_path)
    metrics["device"] = str(device)
    (args.output_dir / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
