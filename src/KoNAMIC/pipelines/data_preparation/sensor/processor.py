import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from sklearn.preprocessing import StandardScaler

class Processor:
    def __init__(
            self,
            batch_size: int,
            train_datasets_specs: dict,
            val_1_datasets_specs: dict,
            val_2_datasets_specs: dict,
            raw_datasets: dict,
            scaler_specs: dict,
            delay: int,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.train_datasets_specs = train_datasets_specs
        self.val_1_datasets_specs = val_1_datasets_specs
        self.val_2_datasets_specs = val_2_datasets_specs
        self.raw_datasets = raw_datasets
        self.scaler_specs = scaler_specs
        self.delay = delay

        self.x_scaler = None
        self.u_scaler = None
        self.processed_datasets = None
        self.data_loader = {}

    def generate_x_scaler(self) -> None:
        x_data = self.raw_datasets["train"]["x"]
        scaler = StandardScaler()
        scaler = self.fit_standardizer(x_data, scaler)
        mean_scaler_x = np.asarray(self.scaler_specs["mean_x"], dtype=scaler.mean_.dtype)

        if self.scaler_specs["center"]:
            assert mean_scaler_x.shape == scaler.mean_.shape, (mean_scaler_x.shape, scaler.mean_.shape)
            scaler.mean_ = mean_scaler_x
            # mean_ is overridden to center around a reference operating point
        self.x_scaler = scaler

    def generate_u_scaler(self) -> None:
        u_data = self.raw_datasets["train"]["u"]
        scaler = StandardScaler()
        scaler = self.fit_standardizer(u_data, scaler)
        mean_scaler_u = np.asarray(self.scaler_specs["mean_u"], dtype=scaler.mean_.dtype)

        if self.scaler_specs["center"]:
            assert mean_scaler_u.shape == scaler.mean_.shape, (mean_scaler_u.shape, scaler.mean_.shape)
            scaler.mean_ = mean_scaler_u
            # mean_ is overridden to center around a reference operating point
        self.u_scaler = scaler

    def build_data_loader(self) -> None:
        self.data_loader["train"] = self._build_single_data_loader(
            "train",
            True,
            True
        )

        self.data_loader["val_1"] = self._build_single_data_loader(
            "val_1",
            False,
            False
        )

        self.data_loader["val_2"] = self._build_single_data_loader(
            "val_2",
            False,
            False
        )

    def _build_single_data_loader(
            self,
            phase: str,
            shuffle: bool,
            drop_last: bool
    ) -> DataLoader:
        batch_size = self.batch_size
        x_data = self.processed_datasets[phase]["x"]
        u_data = self.processed_datasets[phase]["u"]

        x_data = torch.from_numpy(x_data).float()
        u_data = torch.from_numpy(u_data).float()

        tensor_dataset = TensorDataset(x_data, u_data)
        return DataLoader(
            dataset=tensor_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=True
        )

    def process_datasets(self) -> None:
        n_steps_train = self.train_datasets_specs["num_steps_pred"]
        self.processed_datasets = {"train": self._process_dataset("train", n_steps_train)}

        n_steps_val_1 = self.val_1_datasets_specs["num_steps_pred"]
        self.processed_datasets["val_1"] = self._process_dataset("val_1", n_steps_val_1)

        n_steps_val_2 = self.val_2_datasets_specs["num_steps_pred"]
        self.processed_datasets["val_2"] = self._process_dataset("val_2", n_steps_val_2)

    def _process_dataset(self, phase: str, window_size: int) -> dict:
        u = self.raw_datasets[phase]["u"]
        u_sliced = self._slice_and_scale(u, self.u_scaler, window_size)
        x = self.raw_datasets[phase]["x"]

        if self.scaler_specs["scale_x"]:
            x_sliced = self._slice_and_scale(x, self.x_scaler, window_size)
        else:
            x_sliced = self._slice_and_scale(x, None, window_size)
        return {"x": x_sliced, "u": u_sliced}

    def _slice_and_scale(
            self,
            array: np.ndarray,
            scaler: StandardScaler | None,
            window_size: int
    ) -> np.ndarray:

        arr = array.astype(np.float32)
        scaled = self.apply_scaler_dataset(arr, scaler) if scaler else arr

        assert scaled.shape[1] > self.delay
        scaled = scaled[:, self.delay:]
        n_traj, n_steps, dim = scaled.shape
        assert n_steps % window_size == 0, f"sinon il faut gérer le reste proprement {n_steps} {window_size}"

        sliced = scaled.reshape(n_traj, n_steps // window_size, window_size, dim)
        sliced_arr = sliced.reshape(-1, window_size, dim)  # concat des trajs, mais sans mélange
        return sliced_arr

    @staticmethod
    def fit_standardizer(
            data: np.ndarray,
            standardizer: StandardScaler,
            flattened=False
    ) -> StandardScaler:
        if flattened:
            data_flat = data
        else:
            n_traj, traj_length, n = data.shape
            data_flat = data.T.reshape((n, n_traj * traj_length), order='F').T
        standardizer.fit(data_flat)
        return standardizer

    @staticmethod
    def apply_scaler_dataset(
            data: np.ndarray, scaler: StandardScaler | None
    ) -> np.ndarray:
        if scaler is None:
            return data
        n_traj, T, dim = data.shape
        flat = data.reshape(-1, dim)
        flat_scaled = scaler.transform(flat)
        return flat_scaled.reshape(n_traj, T, dim)
