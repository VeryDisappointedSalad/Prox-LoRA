from dataclasses import dataclass
from typing import cast

from torch import Tensor
from torch.utils.data import Subset, random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2

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

    def __init__(self, dataloader: DataLoaderConfig | None = None, *, augmentations: bool = False) -> None:
        super().__init__(num_classes=10, dataloader=dataloader)
        self.image_shape = (3, 32, 32)
        self.data_dir = PROJECT_ROOT / "data" / "cifar10"

        self.transform = default_transform(target_image_size=32)
        self.train_transforms = (
            self.transform
            if not augmentations
            else v2.Compose(
                [
                    v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                    v2.RandomPerspective(distortion_scale=0.5, p=0.5),
                    v2.RandomHorizontalFlip(p=0.5),
                    self.transform,
                ]
            )
        )

    def prepare_data(self) -> None:
        # Download.
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            full = cast(CIFARDataset, CIFAR10(self.data_dir, train=True, transform=self.transform))
            self.train_dataset, self.val_dataset = cast(list[CIFARDataset], random_split(full, [45000, 5000]))

            if self.train_transforms is not None:
                train_indices = cast(Subset[tuple[Tensor, int]], self.train_dataset).indices
                self.train_dataset = cast(
                    CIFARDataset,
                    Subset(CIFAR10(self.data_dir, train=True, transform=self.train_transforms), train_indices),
                )

        if stage == "test" or stage is None:
            self.test_dataset = cast(CIFARDataset, CIFAR10(self.data_dir, train=False, transform=self.transform))


@yaml.register_class
@dataclass(frozen=True)
class CIFAR10Config:
    name: str = "CIFAR10"
    augmentations: bool = False

    def instantiate(self, dataloader: DataLoaderConfig | None = None) -> CIFAR10DataModule:
        return CIFAR10DataModule(dataloader=dataloader, augmentations=self.augmentations)