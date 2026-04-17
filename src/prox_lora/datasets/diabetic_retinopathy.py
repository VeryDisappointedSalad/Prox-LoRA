import pandas as pd
import torch

from PIL import Image
from pathlib import Path
from dataclasses import dataclass
from typing import cast
from torch import Tensor
from torch.utils.data import Dataset, random_split
from torchvision.transforms import v2

from prox_lora.infrastructure.configs import yaml
from prox_lora.utils.io import PROJECT_ROOT
from .base_data_module import BaseDataModule, DataLoaderConfig


class KaggleDRDataset(Dataset):
    def __init__(self, img_dir: Path, csv_path: Path, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        raw_df = pd.read_csv(csv_path)
        self.df = raw_df[raw_df.apply(lambda x: (img_dir / f"{x.iloc[0]}.jpeg").exists(), axis=1)]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx, 0]
        label = int(self.df.iloc[idx, 1])
        img_path = self.img_dir / f"{img_id}.jpeg"
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class DRDataModule(BaseDataModule[tuple[Tensor, int]]):
    def __init__(self, dataloader: DataLoaderConfig | None = None, augmentations: bool = False) -> None:
        super().__init__(num_classes=5, dataloader=dataloader)
        self.data_dir = PROJECT_ROOT / "data" / "retinopathy"

        # hardcoded 224x224?
        standard_transform = v2.Compose(
            [
                v2.Resize((224, 224)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.4814, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2757]),
            ]
        )

        self.transform = standard_transform
        self.train_transforms = (
            standard_transform
            if not augmentations
            else v2.Compose(
                [
                    v2.RandomHorizontalFlip(p=0.5),
                    v2.RandomVerticalFlip(p=0.5),
                    v2.RandomRotation(degrees=15),
                    standard_transform,
                ]
            )
        )

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            full_dataset = KaggleDRDataset(
                img_dir=self.data_dir / "train",
                csv_path=self.data_dir / "trainLabels.csv",
                transform=self.train_transforms,
            )
            # again hardcoded 9/1 split
            self.train_dataset, self.val_dataset = random_split(full_dataset, [0.9, 0.1])

        if stage == "test" or stage is None:
            self.test_dataset = KaggleDRDataset(
                img_dir=self.data_dir / "test",
                csv_path=self.data_dir / "testLabels.csv",  # i found some testlabels, I think they will work
                transform=self.transform,
            )


@yaml.register_class
@dataclass(frozen=True)
class DRConfig:
    name: str = "DR-Kaggle"
    augmentations: bool = True

    def instantiate(self, dataloader: DataLoaderConfig | None = None) -> DRDataModule:
        return DRDataModule(dataloader=dataloader, augmentations=self.augmentations)
