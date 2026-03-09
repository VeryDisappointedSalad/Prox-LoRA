from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
from open_clip import create_model_from_pretrained
from open_clip.model import CustomTextCLIP, TimmModel
from timm.models import VisionTransformer
from torch import Tensor, nn

from prox_lora.infrastructure.configs import yaml


@yaml.register_class
@dataclass(frozen=True)
class BiomedCLIPConfig:
    model_name: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

    input_shape: tuple[int, int, int] = (3, 224, 224)
    """Shape of the input images as (C, H, W)."""

    num_classes: int = 10
    """Number of output classes (logits)."""

    unfrozen_groups: int = 0
    """Number of unfrozen tail groups in the backbone."""

    freeze_bn_stats: bool = False
    """Whether to freeze batch norm stats (running mean and var) in the backbone."""

    def instantiate(self) -> BiomedCLIP:
        return BiomedCLIP(self)


class BiomedCLIP(nn.Module):
    def __init__(self, config: BiomedCLIPConfig) -> None:
        super().__init__()
        self.config = config

        model = create_model_from_pretrained(self.config.model_name, return_transform=False)
        assert isinstance(model, CustomTextCLIP)
        assert isinstance(model.visual, TimmModel)
        assert isinstance(model.visual.trunk, VisionTransformer)
        assert isinstance(model.visual.head, nn.Sequential)

        model.visual.set_input_size(self.config.input_shape[1:])  # Possibly resize positional embedding.
        model.visual.lock(unlocked_groups=self.config.unfrozen_groups, freeze_bn_stats=self.config.freeze_bn_stats)
        # model.visual.set_grad_checkpointing(True)

        self.backbone = model.visual.trunk
        self.head = model.visual.head  # Re-use original head (linear backbone.num_features=768 → embedding_dim=512).

        with torch.device("meta"):
            dummy = torch.empty(self.config.input_shape, device="meta").unsqueeze(0)
            dummy = deepcopy(self.backbone).to(device="meta")(dummy)
            dummy = deepcopy(self.head).to(device="meta")(dummy)
            _, embedding_dim = dummy.shape

        self.head.append(nn.ReLU())
        self.head.append(nn.Dropout(0.3))
        self.head.append(nn.Linear(embedding_dim, self.config.num_classes))

    def forward(self, x: Tensor) -> Tensor:
        """
        Input: shape (B, C, H, W), normalized image.
        Output: shape (B, num_classes), logits for each class (without softmax).
        """
        x = self.backbone(x)
        x = self.head(x)
        return x

    @torch.jit.ignore()  # type: ignore
    def no_weight_decay(self) -> set[str]:
        """Set of parameters that should not use weight decay."""
        no_wd = set[str]()
        if hasattr(self.backbone, "no_weight_decay"):
            for n in self.backbone.no_weight_decay():
                no_wd.add("backbone." + n)
        return no_wd