from dataclasses import dataclass

import timm
from torch import nn

from prox_lora.infrastructure.configs import yaml


@yaml.register_class
@dataclass(frozen=True)
class TimmConfig:
    model_name: str = "mobilenetv3_small_100"
    """Name of the timm model to use. See e.g.: https://huggingface.co/docs/timm/results"""

    pretrained: bool = False
    """Whether to use pretrained weights."""

    num_classes: int = 10
    """Number of output classes (logits)."""

    def instantiate(self) -> nn.Module:
        return timm.create_model(self.model_name, pretrained=self.pretrained, num_classes=self.num_classes)
