from collections.abc import Callable, Sequence, Sized
from typing import TypeVar

import torch
import torch.utils.data

S_co = TypeVar("S_co", covariant=True)
T_co = TypeVar("T_co", covariant=True)


class SizedDataset(torch.utils.data.Dataset[T_co], Sized):
    """A dataset that supports `len()`."""


class TransformedDataset(SizedDataset[T_co]):
    """
    Applies a transform to dataitems produced by any dataset.

    This is useful e.g. for datasets that don't allow transforming the whole dataitems, just images.
    """

    def __init__(self, dataset: SizedDataset[S_co] | Sequence[S_co], transform: Callable[[S_co], T_co]):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index: int) -> T_co:
        return self.transform(self.dataset[index])

    def __len__(self) -> int:
        return self.dataset.__len__()
