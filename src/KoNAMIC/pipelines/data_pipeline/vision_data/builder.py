import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from KoNAMIC.core import utils
from .image_dataset import ImageDataset

MODALITY = "vision"

class Builder:
    def __init__(
            self,
            dataset_version: int,
            num_val_datasets: int,
            resolution: int,
            batch_size: int,
            processed_dataset: dict,
            num_workers: int,
            drone_dim: int,
            seed: int,
    ):
        super().__init__()
        self.dataset_version = dataset_version
        self.jean_zay = utils.is_jean_zay_env()
        self.res = resolution
        self.num_workers = num_workers
        self.drone_dim = drone_dim
        self.seed = seed
        self.datasets = processed_dataset
        self.num_val_datasets = num_val_datasets
        self.batch_size = batch_size
        self.x_dim, self.u_dim, _ = utils.get_dimensions(drone_dim)

    def pipeline(self) -> dict:
        raw_train_dataset = self._get_raw_datasets("train")
        loaders = {"train": self._get_dataloaders(
            raw_train_dataset,
            True,
            True
        )}
        for phase in range(self.num_val_datasets):
            raw_dataset = self._get_raw_datasets(f"val_{phase+1}")
            loaders[f"val_{phase+1}"] = self._get_dataloaders(
                raw_dataset,
                False,
                False
            )
        return loaders

    def _get_raw_datasets(self, phase: str) -> ImageDataset:
        print(f"Loading of {phase} dataset")
        u_data = self.datasets[phase]["u"]
        x_data = self.datasets[phase]["x"]
        version = self.dataset_version

        path = utils.build_dataset_path(
            self.jean_zay,
            self.drone_dim,
            MODALITY,
            str(version),
            phase
        )
        mmap_path = path / "dataset_memmap.dat"
        metadata = json.load(open(path / "metadata.json"))

        y_shape = tuple(metadata["y_shape"])
        dtype = np.dtype(metadata["dtype"])
        n_vis = y_shape[0]

        n_sens = len(x_data)
        assert n_sens >= n_vis, f"{phase}: sensor has {n_sens} traj, vision has {n_vis}"

        return ImageDataset(
            y_path=mmap_path,
            y_shape=y_shape,
            x_data=x_data,
            u_data=u_data,
            dtype=dtype
        )

    def _get_dataloaders(
            self,
            im_dataset: ImageDataset,
            shuffle: bool,
            drop_last: bool
    ) -> DataLoader:
        gen = torch.Generator()
        gen.manual_seed(self.seed)

        # Ensure each worker process has a different but deterministic NumPy seed
        def _worker_init_fn(worker_id):
            np.random.seed(self.seed + worker_id)

        return DataLoader(
            im_dataset,
            batch_size = self.batch_size,
            shuffle = shuffle,
            num_workers=self.num_workers,
            drop_last = drop_last,
            pin_memory = True,  # Faster transfer from CPU to GPU
            worker_init_fn=_worker_init_fn, # Controls reproducible shuffling
            generator=gen,
        )




