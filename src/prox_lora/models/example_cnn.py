from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from torch import Tensor, nn

from prox_lora.infrastructure.configs import yaml


@yaml.register_class
@dataclass(frozen=True)
class ExampleCNNConfig:
    input_shape: tuple[int, int, int] = (1, 28, 28)
    """Shape of the input images as (C, H, W)."""

    hidden_channels: Sequence[int] = (16, 32)
    """Defines intermediate conv layers by the number of output channels in each hidden layer."""

    num_classes: int = 10
    """Number of output classes (logits)."""

    def instantiate(self) -> ExampleCNN:
        return ExampleCNN(self)


class ExampleCNN(nn.Module):
    """A very basic CNN for demonstration/testing purposes."""

    def __init__(self, config: ExampleCNNConfig) -> None:
        super().__init__()
        self.config = config

        C, H, W = self.config.input_shape
        hidden_channels = self.config.hidden_channels

        self.layers = nn.Sequential()

        for i in range(len(hidden_channels)):
            in_channels = hidden_channels[i - 1] if i > 0 else C
            out_channels = hidden_channels[i]
            self.layers.add_module(f"conv{i}", nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            self.layers.add_module(f"bn{i}", nn.BatchNorm2d(out_channels))
            self.layers.add_module(f"relu{i}", nn.ReLU())

        self.layers.add_module("flatten", nn.Flatten(start_dim=-3))
        self.layers.add_module("fc", nn.Linear(hidden_channels[-1] * H * W, self.config.num_classes))

    def forward(self, x: Tensor) -> Tensor:
        """
        Input: shape (B, C, H, W), normalized image.
        Output: shape (B, num_classes), logits for each class (without softmax).
        """
        return cast(Tensor, self.layers(x))
