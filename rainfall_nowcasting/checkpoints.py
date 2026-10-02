from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import torch
from torch import Tensor

from .model import ConvLSTMNowcaster, ModelConfig


def load_trained_model(path: str | Path, device: torch.device) -> ConvLSTMNowcaster:
    checkpoint = cast(
        dict[str, object],
        torch.load(path, map_location=device, weights_only=True),
    )
    raw_config = checkpoint.get("model_config")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(raw_config, dict) or not isinstance(state_dict, Mapping):
        raise ValueError("Checkpoint is missing model_config or model_state_dict")
    model = ConvLSTMNowcaster(ModelConfig.from_dict(raw_config)).to(device)
    model.load_state_dict(cast(Mapping[str, Tensor], state_dict))
    return model
