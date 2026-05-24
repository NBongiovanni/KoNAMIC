from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, y_path, y_shape, x_data, u_data, dtype=np.uint8):
        # TODO: add type hints
        """
        y_path : Path(.../dataset_memmap.dat)
        y_shape : (N, n_steps_pred, 2, H, W)
        x_data, u_data : restent tels quels (numpy memmap possible)
        """

        self.y_path = Path(y_path)
        self.y_shape = y_shape
        self.dtype = dtype

        self.x_data = x_data
        self.u_data = u_data

        # Les memmaps seront ouverts paresseusement
        self._y_mm = None

    def _init_memmaps(self):
        if self._y_mm is None:
            self._y_mm = np.memmap(
                self.y_path,
                mode="r",
                dtype=self.dtype,
                shape=self.y_shape
            )

    def __len__(self):
        return self.y_shape[0]

    def __getitem__(self, idx):
        self._init_memmaps()
        y = torch.tensor(self._y_mm[idx].copy(), dtype=torch.float32) / 255.0
        x = torch.from_numpy(self.x_data[idx]).float()
        u = torch.from_numpy(self.u_data[idx]).float()
        return y, u, x
