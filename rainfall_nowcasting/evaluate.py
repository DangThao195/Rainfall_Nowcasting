from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.data import DataLoader

from .checkpoints import load_trained_model
from .data import DataMetadata, ImergWindowDataset, load_spatial_field
from .evaluation import evaluate_nowcasts
from .training import seed_everything, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ConvLSTM and persistence on IMERG test data"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("IMERG_data/prepared"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = select_device(args.device)
    metadata = DataMetadata.from_file(args.data_dir / "dataset_metadata.json")
    dataset = ImergWindowDataset(
        args.data_dir / "imerg_test_normalized.nc",
        args.data_dir / "test_window_starts.npy",
        metadata.input_steps,
        metadata.forecast_steps,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    model = load_trained_model(args.checkpoint, device)
    land_mask = load_spatial_field(args.data_dir / "vietnam_land_mask.nc", "land_mask")
    metrics = evaluate_nowcasts(model, loader, metadata, land_mask, device, args.max_batches)
    metrics["checkpoint"] = str(args.checkpoint)
    metrics["device"] = str(device)
    output = args.output or args.checkpoint.with_name("test_metrics.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
