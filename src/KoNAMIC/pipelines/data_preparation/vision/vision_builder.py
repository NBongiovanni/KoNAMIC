import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from KoNAMIC import paths
from KoNAMIC.pipelines.data_preparation.data_loaders import (
    DataLoaderSplit,
    PreparedDataLoaders,
)

from .config import VisionPreparationConfig
from .image_dataset import ImageDataset


class Builder:
    def __init__(
        self,
        dataset_paths: paths.DatasetPaths,
        config: VisionPreparationConfig,
        processed_dataset: dict,
        system,
        seed: int,
    ):
        super().__init__()
        self.dataset_paths = dataset_paths
        self.config = config
        self.jean_zay = paths.is_jean_zay_env()
        self.res = config.resolution
        self.num_workers = config.dataloader.num_workers
        self.drone_dim = system.system_dim
        self.seed = seed
        self.datasets = processed_dataset
        self.batch_size = config.dataloader.batch_size
        self.x_dim = system.x_dim
        self.u_dim = system.u_dim

    def pipeline(self) -> PreparedDataLoaders:
        raw_train_dataset = self._get_raw_datasets("train")
        train_loader = self._get_dataloaders(
            raw_train_dataset,
            True,
            True,
            "train",
        )
        train = DataLoaderSplit(
            split="train",
            name="train",
            loader=train_loader,
            num_steps_pred=self.config.train.num_steps_pred,
        )

        validations = []
        for val_cfg in self.config.val_datasets:
            raw_dataset = self._get_raw_datasets(val_cfg.split)
            loader = self._get_dataloaders(
                raw_dataset,
                False,
                False,
                val_cfg.split,
            )
            validations.append(
                DataLoaderSplit(
                    split=val_cfg.split,
                    name=val_cfg.name,
                    loader=loader,
                    num_steps_pred=val_cfg.num_steps_pred,
                )
            )

        return PreparedDataLoaders(
            train=train,
            validations=tuple(validations),
        )

    def _get_raw_datasets(self, phase: str) -> ImageDataset:
        print(f"Loading of {phase} dataset")
        u_data = self.datasets[phase]["u"]
        x_data = self.datasets[phase]["x"]
        metadata = json.load(open(self.dataset_paths.vision_metadata(phase)))

        y_shape = tuple(metadata["y_shape"])
        dtype = np.dtype(metadata["dtype"])
        n_vis = y_shape[0]

        n_sens = len(x_data)
        assert n_sens >= n_vis, f"{phase}: sensor has {n_sens} traj, vision has {n_vis}"

        return ImageDataset(
            y_path=self.dataset_paths.vision_file(phase),
            y_shape=y_shape,
            x_data=x_data,
            u_data=u_data,
            dtype=dtype
        )

    def _get_dataloaders(
        self,
        im_dataset: ImageDataset,
        shuffle: bool,
        drop_last: bool,
        phase: str,
    ) -> DataLoader:
        gen = torch.Generator()
        gen.manual_seed(self._phase_seed(phase))

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

    def _phase_seed(self, phase: str) -> int:
        if phase == "train":
            return self.seed

        for idx, val_cfg in enumerate(self.config.val_datasets, start=1):
            if phase == val_cfg.split:
                return self.seed + idx

        raise KeyError(phase)
