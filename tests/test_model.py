from __future__ import annotations

from pathlib import Path

import torch

from rainfall_nowcasting.checkpoints import load_trained_model
from rainfall_nowcasting.losses import WeightedRainfallLoss
from rainfall_nowcasting.model import ConvLSTMNowcaster, ModelConfig


def test_convlstm_forward_and_backward() -> None:
    model = ConvLSTMNowcaster(ModelConfig(hidden_channels=(2, 4), kernel_size=3))
    inputs = torch.randn(2, 6, 1, 17, 8)
    targets = torch.randn(2, 4, 1, 17, 8)
    loss_function = WeightedRainfallLoss(torch.ones(17, 8), rain_threshold_normalized=0.0)

    prediction = model(inputs, forecast_steps=4, targets=targets, teacher_forcing_ratio=0.5)
    loss = loss_function(prediction, targets)
    loss.backward()

    assert prediction.shape == targets.shape
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_load_trained_model_preserves_architecture(tmp_path: Path) -> None:
    config = ModelConfig(hidden_channels=(2, 4), kernel_size=3)
    model = ConvLSTMNowcaster(config)
    checkpoint_path = tmp_path / "model.pt"
    torch.save(
        {
            "model_config": config.to_dict(),
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )

    restored = load_trained_model(checkpoint_path, torch.device("cpu"))

    assert restored.config == config
    assert all(
        torch.equal(original, loaded)
        for original, loaded in zip(model.parameters(), restored.parameters(), strict=True)
    )
