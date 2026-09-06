from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from torch import Tensor, nn

from prox_lora.infrastructure.configs import yaml


@yaml.register_class
@dataclass(frozen=True)
class KaggleConvNetConfig:
    input_shape: tuple[int, int, int] = (3, 224, 224)
    """Shape of the input images as (C, H, W)."""

    channels: Sequence[int] = (32, 64, 128, 256)
    """Number of output channels for each VGG-style block."""

    num_classes: int = 5
    """Number of output classes (5 for Diabetic Retinopathy)."""

    dropout_rate: float = 0.4
    """Dropout rate before the final classifier."""

    def instantiate(self) -> KaggleConvNet:
        return KaggleConvNet(self)


class KaggleConvNet(nn.Module):
    """
    VGG-style ConvNet, similar to that from two Kaggle solutions.

    - https://www.kaggle.com/competitions/diabetic-retinopathy-detection/writeups/o-o-team-o-o-solution-summary
    - https://deepsense.ai/blog/diagnosing-diabetic-retinopathy-with-deep-learning/
    """

    def __init__(self, config: KaggleConvNetConfig) -> None:
        super().__init__()
        self.config = config

        C, _H, _W = self.config.input_shape
        channels = self.config.channels

        features = []
        in_channels = C

        # Build VGG-style blocks: [Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> MaxPool]
        for out_channels in channels:
            features.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                ]
            )
            in_channels = out_channels

        self.features = nn.Sequential(*features)

        # output is 1x1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=self.config.dropout_rate),
            nn.Linear(in_channels, self.config.num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Input: shape (B, C, H, W), normalized image.
        Output: shape (B, num_classes), logits.
        """
        x = self.features(x)
        x = self.pool(x)
        return cast(Tensor, self.classifier(x))
