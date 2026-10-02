from __future__ import annotations

import random
from collections.abc import Iterable

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def run_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor]],
    loss_function: nn.Module,
    device: torch.device,
    forecast_steps: int,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    teacher_forcing_ratio: float = 0.0,
    max_batches: int | None = None,
    gradient_clip: float = 1.0,
) -> float:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    sample_count = 0
    amp_enabled = scaler is not None and scaler.is_enabled()

    for batch_index, (inputs, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = inputs.shape[0]

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with (
            torch.set_grad_enabled(is_training),
            torch.autocast(device_type=device.type, enabled=amp_enabled),
        ):
            prediction = model(
                inputs,
                forecast_steps=forecast_steps,
                targets=targets if is_training else None,
                teacher_forcing_ratio=teacher_forcing_ratio if is_training else 0.0,
            )
            loss = loss_function(prediction, targets)

        if optimizer is not None:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()

        total_loss += float(loss.detach()) * batch_size
        sample_count += batch_size
    if sample_count == 0:
        raise RuntimeError("No batches were processed")
    return total_loss / sample_count


def teacher_forcing_ratio(epoch: int, epochs: int, start: float, end: float) -> float:
    if epochs <= 1:
        return end
    progress = epoch / (epochs - 1)
    return start + progress * (end - start)


def count_parameters(parameters: Iterable[nn.Parameter]) -> int:
    return sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
