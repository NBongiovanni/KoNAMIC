from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from torch.utils.data import DataLoader


@dataclass(frozen=True)
class DataLoaderSplit:
    split: str
    name: str
    loader: DataLoader
    num_steps_pred: int | None = None


@dataclass(frozen=True)
class PreparedDataLoaders(Mapping[str, DataLoader]):
    train: DataLoaderSplit
    validations: tuple[DataLoaderSplit, ...]

    def __post_init__(self) -> None:
        if self.train.split != "train":
            raise ValueError(f"train split must be named 'train', got {self.train.split!r}.")

        split_names = [item.split for item in self.validation_items]
        duplicates = sorted({split for split in split_names if split_names.count(split) > 1})
        if duplicates:
            raise ValueError(f"Duplicate validation split names: {duplicates}.")

    @property
    def validation_items(self) -> tuple[DataLoaderSplit, ...]:
        return self.validations

    @property
    def validation_splits(self) -> tuple[str, ...]:
        return tuple(item.split for item in self.validations)

    def get_split(self, split: str) -> DataLoaderSplit:
        if split == self.train.split:
            return self.train

        for item in self.validations:
            if item.split == split:
                return item

        raise KeyError(split)

    def to_dict(self) -> dict[str, DataLoader]:
        return {split: self[split] for split in self}

    def __getitem__(self, split: str) -> DataLoader:
        return self.get_split(split).loader

    def __iter__(self) -> Iterator[str]:
        yield self.train.split
        for item in self.validations:
            yield item.split

    def __len__(self) -> int:
        return 1 + len(self.validations)
