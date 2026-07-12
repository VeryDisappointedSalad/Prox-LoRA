from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast
from collections import Counter

import pandas as pd
import numpy as np
import torch
from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from PIL import Image
from torch import Tensor
from torch.utils.data import random_split, WeightedRandomSampler, DataLoader
from torchvision.transforms import v2

from prox_lora.datasets.base_data_module import BaseDataModule, DataLoaderConfig
from prox_lora.datasets.common import SizedDataset
from prox_lora.infrastructure.configs import yaml
from prox_lora.utils.io import PROJECT_ROOT


class KaggleDRDataset(SizedDataset[tuple[Tensor, int]]):
    """
    Kaggle's Diabetic Retinopathy Detection dataset.

    A classification dataset with (image, label) pairs.
    Labels are 0/1/2/3/4 for No DR/Mild/Moderate/Severe/Proliferative DR.

    https://www.kaggle.com/competitions/diabetic-retinopathy-detection/overview

    See README.md for instructions on downloading and preprocessing.
    """

    def __init__(self, img_dir: Path, csv_path: Path, transform: Callable[[Image.Image], Tensor] | None = None) -> None:
        self.img_dir = img_dir
        self.transform = transform

        df = pd.read_csv(csv_path)

        # Check that images and labels match.
        img_names = {p.stem for p in img_dir.glob("*.jpeg")}
        label_names = set(df["image"])
        assert img_names == label_names, (
            f"Image names and label names differ:\nOnly in labels:{label_names - img_names}\nOnly in images: {img_names - label_names}"
        )

        # Store pairs (image path, label), where label is a int (0..5).
        self.data = [(img_dir / f"{row['image']}.jpeg", row["level"]) for _, row in df.iterrows()]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            return self.transform(image), label
        else:
            return image, label  # type: ignore[return-value]


class DRDataModule(BaseDataModule[tuple[Tensor, int]]):
    def __init__(
        self,
        dataloader: DataLoaderConfig | None = None,
        *,
        size: Literal["original", 1024, 512, 256, 224] = 224,
        augmentations: bool = False,
        weighted_sampler: bool = False,
    ) -> None:
        super().__init__(num_classes=5, dataloader=dataloader)
        self.data_dir = PROJECT_ROOT / "data" / "retinopathy" / str(size)

        self.weighted_sampler = weighted_sampler

        if size == "original":
            resize = []
        else:
            # Images in data_dir, are already rescaled, CenterCrop will actually add padding to make them square.
            resize = [v2.CenterCrop((size, size))]

        standard_transform = v2.Compose(
            [
                *resize,
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
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
                    v2.RandomRotation(degrees=15),  # type: ignore[arg-type]
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
            # Hardcoded 9:1 train:val split. Does not group left and right eye together.
            rng = torch.Generator().manual_seed(42)
            self.train_dataset, self.val_dataset = cast(
                list[SizedDataset[tuple[Tensor, int]]], random_split(full_dataset, [0.9, 0.1], generator=rng)
            )

        if stage == "test" or stage is None:
            self.test_dataset = KaggleDRDataset(
                img_dir=self.data_dir / "test", csv_path=self.data_dir / "testLabels.csv", transform=self.transform
            )

    # new method for uniform sampling, Marcin's suggestion on Slack from 30.06 2.14PM
    def train_dataloader(self) -> DataLoader:
        """
        Suggestion to make the batches more uniform. Another problem is that when switching to 512x512 images from 224x224, the batch_size had to be
        downsized from 64 to 16 to fit in the memory, so the samples were often probably just class 0
        """
        if not getattr(self, "weighted_sampler", False):
            return super().train_dataloader()

        # get original dataset and indices
        dataset = self.train_dataset.dataset  # SizedDataset[tuple[Tensor, int]], indices at [1]
        indices = self.train_dataset.indices

        # get train labels
        train_labels = [dataset.data[i][1] for i in indices]

        # get distribution per class
        class_counts = Counter(train_labels)

        # weights = \frac{1}{\sqrt{count}}
        class_weights = {cls: 1.0 / np.sqrt(count) for cls, count in class_counts.items()}

        # weight per sample
        sample_weights = [class_weights[label] for label in train_labels]

        # suggestion was to use https://docs.pytorch.org/docs/2.12/data.html#torch.utils.data.WeightedRandomSampler
        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights), num_samples=len(sample_weights), replacement=True
        )

        # TODO: shuffle left as None
        dl_config = self.dataloader
        return DataLoader(
            self.train_dataset,
            batch_size=dl_config.batch_size,
            num_workers=dl_config.num_workers,
            pin_memory=dl_config.pin_memory,
            sampler=sampler,
        )


@yaml.register_class
@dataclass(frozen=True)
class DRConfig:
    name: str = "DR-Kaggle"
    augmentations: bool = True
    size: Literal["original", 1024, 512, 256, 224] = 224
    weighed_sampler: bool = False

    def instantiate(self, dataloader: DataLoaderConfig | None = None) -> DRDataModule:
        return DRDataModule(
            dataloader=dataloader,
            augmentations=self.augmentations,
            size=self.size,
            weighted_sampler=self.weighed_sampler,
        )
