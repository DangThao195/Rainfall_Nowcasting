from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ModelConfig:
    input_channels: int = 1
    hidden_channels: tuple[int, ...] = (16, 32)
    kernel_size: int = 3

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> ModelConfig:
        hidden_channels = raw.get("hidden_channels")
        if not isinstance(hidden_channels, (list, tuple)) or not all(
            isinstance(value, int) for value in hidden_channels
        ):
            raise ValueError("hidden_channels must be a sequence of integers")
        return cls(
            input_channels=int(cast(int, raw.get("input_channels", 1))),
            hidden_channels=tuple(hidden_channels),
            kernel_size=int(cast(int, raw.get("kernel_size", 3))),
        )

    def to_dict(self) -> dict[str, int | list[int]]:
        return {
            "input_channels": self.input_channels,
            "hidden_channels": list(self.hidden_channels),
            "kernel_size": self.kernel_size,
        }


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, inputs: Tensor, state: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        hidden, cell = state
        input_gate, forget_gate, output_gate, candidate = self.gates(
            torch.cat((inputs, hidden), dim=1)
        ).chunk(4, dim=1)
        cell = torch.sigmoid(forget_gate) * cell + torch.sigmoid(input_gate) * torch.tanh(candidate)
        hidden = torch.sigmoid(output_gate) * torch.tanh(cell)
        return hidden, cell

    def initial_state(self, reference: Tensor) -> tuple[Tensor, Tensor]:
        shape = (reference.shape[0], self.hidden_channels, reference.shape[-2], reference.shape[-1])
        zeros = reference.new_zeros(shape)
        return zeros, zeros.clone()


class ConvLSTMNowcaster(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        layer_inputs = (self.config.input_channels, *self.config.hidden_channels[:-1])
        self.cells = nn.ModuleList(
            ConvLSTMCell(input_size, hidden_size, self.config.kernel_size)
            for input_size, hidden_size in zip(
                layer_inputs, self.config.hidden_channels, strict=True
            )
        )
        self.output = nn.Conv2d(
            self.config.hidden_channels[-1], self.config.input_channels, kernel_size=1
        )

    def forward(
        self,
        inputs: Tensor,
        forecast_steps: int,
        targets: Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> Tensor:
        states = [cast(ConvLSTMCell, cell).initial_state(inputs[:, 0]) for cell in self.cells]
        for step in range(inputs.shape[1]):
            states = self._advance(inputs[:, step], states)

        previous = inputs[:, -1]
        predictions: list[Tensor] = []
        for step in range(forecast_steps):
            states = self._advance(previous, states)
            previous = self.output(states[-1][0])
            predictions.append(previous)
            if targets is not None and teacher_forcing_ratio > 0:
                use_target = torch.rand((), device=inputs.device) < teacher_forcing_ratio
                previous = torch.where(use_target, targets[:, step], previous)
        return torch.stack(predictions, dim=1)

    def _advance(
        self,
        frame: Tensor,
        states: list[tuple[Tensor, Tensor]],
    ) -> list[tuple[Tensor, Tensor]]:
        layer_input = frame
        next_states: list[tuple[Tensor, Tensor]] = []
        for cell, state in zip(self.cells, states, strict=True):
            state = cast(ConvLSTMCell, cell)(layer_input, state)
            next_states.append(state)
            layer_input = state[0]
        return next_states
