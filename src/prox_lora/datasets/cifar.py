from dataclasses import dataclass
from typing import cast

from torch import Tensor
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10

from prox_lora.datasets.common import SizedDataset
from prox_lora.infrastructure.configs import yaml
from prox_lora.utils.io import PROJECT_ROOT

from .base_data_module import BaseDataModule, DataLoaderConfig
from .transforms import default_transform

CIFARDataset = SizedDataset[tuple[Tensor, int]]


class CIFAR10DataModule(BaseDataModule[tuple[Tensor, int]]):
    """
    CIFAR10 dataset. See https://www.cs.toronto.edu/~kriz/cifar.html

    Yields (image, label) pairs, where image is a 3x32x32 normalized tensor and label is an int in 0..9.
    """

    def __init__(self, dataloader: DataLoaderConfig | None = None) -> None:
        super().__init__(num_classes=10, dataloader=dataloader)
        self.image_shape = (3, 32, 32)
        self.data_dir = PROJECT_ROOT / "data" / "cifar10"

        self.transform = default_transform(target_image_size=32)

    def prepare_data(self) -> None:
        # Download.
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            full = cast(CIFARDataset, CIFAR10(self.data_dir, train=True, transform=self.transform))
            self.train_dataset, self.val_dataset = cast(list[CIFARDataset], random_split(full, [45000, 5000]))

        if stage == "test" or stage is None:
            self.test_dataset = cast(CIFARDataset, CIFAR10(self.data_dir, train=False, transform=self.transform))


@yaml.register_class
@dataclass(frozen=True)
class CIFAR10Config:
    name: str = "CIFAR10"

    def instantiate(self, dataloader: DataLoaderConfig | None = None) -> CIFAR10DataModule:
        return CIFAR10DataModule(dataloader=dataloader)
