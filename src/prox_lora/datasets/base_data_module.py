from dataclasses import asdict, dataclass

import lightning as L
import torch
import torch.utils.data

from prox_lora.datasets.common import SizedDataset
from prox_lora.infrastructure.configs import yaml


@yaml.register_class
@dataclass(frozen=True)
class DataLoaderConfig:
    """
    Kwargs for torch.DataLoader (all optional).

    Does not include shuffle: we always use shuffle=True for train_dataloader and shuffle=False otherwise.
    """

    batch_size: int = 1
    drop_last: bool = False
    pin_memory: bool = False
    num_workers: int = 0
    prefetch_factor: int | None = None  # Defaults to 2 if num_workers > 0.
    persistent_workers: bool = False


class BaseDataModule[T](L.LightningDataModule):
    """
    A LightningDataModule base class with the typical train/val/test dataloaders.

    The setup() method needs to be defined to initialize self.train_dataset, val_dataset and test_dataset.

    Args:
    - dataloader: kwargs passed to torch.utils.data.DataLoader().
        Common/default args are batch_size=1, num_workers=0, pin_memory=False.
    """

    num_classes: int

    def __init__(self, num_classes: int, dataloader: DataLoaderConfig | None = None) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dataloader: DataLoaderConfig = dataloader or DataLoaderConfig()

        # These should be initialized in setup().
        self.train_dataset: SizedDataset[T]
        self.val_dataset: SizedDataset[T]
        self.test_dataset: SizedDataset[T]

    def setup(self, stage: str | None = None) -> None:
        """Initialize self.train_dataset, self.val_dataset and self.test_dataset."""
        raise NotImplementedError()

    def train_dataloader(self) -> torch.utils.data.DataLoader[T]:
        return torch.utils.data.DataLoader(self.train_dataset, **asdict(self.dataloader), shuffle=True)

    def val_dataloader(self) -> torch.utils.data.DataLoader[T]:
        return torch.utils.data.DataLoader(self.val_dataset, **asdict(self.dataloader))

    def test_dataloader(self) -> torch.utils.data.DataLoader[T]:
        return torch.utils.data.DataLoader(self.test_dataset, **asdict(self.dataloader))
