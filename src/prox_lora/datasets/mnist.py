from dataclasses import dataclass
from typing import cast

from torch import Tensor
from torch.utils.data import random_split
from torchvision.datasets import MNIST

from prox_lora.infrastructure.configs import yaml
from prox_lora.utils.io import PROJECT_ROOT

from .base_data_module import BaseDataModule, DataLoaderConfig
from .common import SizedDataset
from .transforms import default_transform

MNISTDataset = SizedDataset[tuple[Tensor, int]]


class MNISTDataModule(BaseDataModule[tuple[Tensor, int]]):
    """
    MNIST handwritten digits dataset.

    Yields (image, label) pairs, where image is a 1x28x28 normalized tensor and label is an int in 0..9.
    """

    def __init__(self, dataloader: DataLoaderConfig | None = None) -> None:
        super().__init__(num_classes=10, dataloader=dataloader)
        self.image_shape = (1, 28, 28)
        self.data_dir = PROJECT_ROOT / "data" / "mnist"

        self.transform = default_transform(target_image_size=28)

    def prepare_data(self) -> None:
        # Download.
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            full = cast(MNISTDataset, MNIST(self.data_dir, train=True, transform=self.transform))
            self.train_dataset, self.val_dataset = cast(list[MNISTDataset], random_split(full, [55000, 5000]))

        if stage == "test" or stage is None:
            self.test_dataset = cast(MNISTDataset, MNIST(self.data_dir, train=False, transform=self.transform))


@yaml.register_class
@dataclass(frozen=True)
class MNISTConfig:
    name: str = "MNIST"

    def instantiate(self, dataloader: DataLoaderConfig | None = None) -> MNISTDataModule:
        return MNISTDataModule(dataloader=dataloader)
