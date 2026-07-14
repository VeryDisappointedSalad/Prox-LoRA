from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd
import torch
from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
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
        weighted_sampler: bool | Literal["sqrt"] = False,
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

        self.train_labels: list[int]  # Initialized in setup('fit') for weighted sampling.

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            full_dataset = KaggleDRDataset(
                img_dir=self.data_dir / "train",
                csv_path=self.data_dir / "trainLabels.csv",
                transform=self.train_transforms,
            )
            # Hardcoded 9:1 train:val split. Does not group left and right eye together.
            rng = torch.Generator().manual_seed(42)
            train_subset, val_subset = random_split(full_dataset, [0.9, 0.1], generator=rng)
            self.train_dataset = cast(SizedDataset[tuple[Tensor, int]], train_subset)
            self.val_dataset = cast(SizedDataset[tuple[Tensor, int]], val_subset)

            # Store labels of the training set for weighted sampling.
            self.train_labels = [full_dataset.data[i][1] for i in train_subset.indices]

        if stage == "test" or stage is None:
            self.test_dataset = KaggleDRDataset(
                img_dir=self.data_dir / "test", csv_path=self.data_dir / "testLabels.csv", transform=self.transform
            )

    def train_dataloader(self) -> DataLoader[tuple[Tensor, int]]:
        """Create a DataLoader with a weighted sampler."""
        if not self.weighted_sampler:
            return super().train_dataloader()

        class_frequencies = self.get_class_frequencies()

        if self.weighted_sampler == "sqrt":
            class_weights = {cls: 1.0 / np.sqrt(freq) for cls, freq in enumerate(class_frequencies)}
        elif self.weighted_sampler is True:
            class_weights = {cls: 1.0 / freq for cls, freq in enumerate(class_frequencies)}
        else:
            raise ValueError(f"Unknown weighted_sampler value: {self.weighted_sampler}")

        sampler = WeightedRandomSampler(
            weights=[class_weights[label] for label in self.train_labels],  # Weight for each item.
            num_samples=len(self.train_labels),  # This many are drawn to make one epoch.
            replacement=True,  # Since we preserve epoch size, sampling without replacement is not possible.
        )

        return DataLoader(self.train_dataset, sampler=sampler, **asdict(self.dataloader))

    def get_class_frequencies(self) -> list[float]:
        """Return the frequency of each label in the training set (they sum to 1)."""
        if not hasattr(self, "train_labels"):
            raise RuntimeError("Call setup('fit') before get_class_frequencies")

        counter = Counter(self.train_labels)
        return [counter[i] / counter.total() for i in range(self.num_classes)]


@yaml.register_class
@dataclass(frozen=True)
class DRConfig:
    name: str = "DR-Kaggle"
    augmentations: bool = True
    size: Literal["original", 1024, 512, 256, 224] = 224
    weighted_sampler: bool | Literal["sqrt"] = False

    def instantiate(self, dataloader: DataLoaderConfig | None = None) -> DRDataModule:
        return DRDataModule(
            dataloader=dataloader,
            augmentations=self.augmentations,
            size=self.size,
            weighted_sampler=self.weighted_sampler,
        )


def test_weighted_sampler(batch_size: int, n_batches: int) -> None:
    """Test the DR dataset and dataloader."""
    datamodule = DRDataModule(DataLoaderConfig(batch_size=batch_size), size=224, weighted_sampler="sqrt")
    datamodule.setup(stage="fit")
    train_loader = datamodule.train_dataloader()
    assert len(train_loader) == np.ceil(len(datamodule.train_dataset) / batch_size)

    all_targets = list[int]()
    for i, (_inputs, targets) in enumerate(train_loader):
        assert targets.shape == (batch_size,) or i == len(train_loader) - 1
        all_targets.extend(targets.tolist())
        if i + 1 >= n_batches:
            break

    counter = Counter(all_targets)
    frequencies = [counter[i] / counter.total() for i in range(5)]
    print(f"Class distribution in {n_batches} batches: {[f'{x:.1%}' for x in frequencies]}")


if __name__ == "__main__":
    test_weighted_sampler(batch_size=16, n_batches=100)
