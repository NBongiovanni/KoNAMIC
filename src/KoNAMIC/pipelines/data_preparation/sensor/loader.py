import os
from pathlib import Path

import numpy as np

from KoNAMIC.core.utils import is_jean_zay_env, DatasetPaths


class Loader:
    def __init__(
        self,
        dataset_paths: DatasetPaths,
        drone_dim: int,
        train_dataset_specs: dict,
        val_1_dataset_specs: dict,
        val_2_dataset_specs: dict,
        downsample_factor: int,
        verbose: bool = True,
    ):
        super().__init__()

        self.drone_dim = drone_dim

        self.train_dataset_specs = train_dataset_specs
        self.val_1_dataset_specs = val_1_dataset_specs
        self.val_2_dataset_specs = val_2_dataset_specs

        self.downsample_factor = downsample_factor
        self.verbose = verbose

        self.jean_zay = is_jean_zay_env()

        self.dataset_paths = dataset_paths
        self.dataset_root = self.dataset_paths.root
        self.sensor_dir = self.dataset_paths.sensor_dir

        if self.verbose:
            print("[DEBUG] is_jean_zay_env() =", self.jean_zay)
            print("[DEBUG] SCRATCH =", os.environ.get("SCRATCH", None))
            print("[DEBUG] dataset_root =", self.dataset_root)
            print("[DEBUG] sensor_dir =", self.sensor_dir)

    def load_raw_sensor_data(self) -> dict:
        return {
            "train": self._load_split(
                split_name="train",
                num_steps_loaded=self.train_dataset_specs["num_steps_loaded"],
            ),
            "val_1": self._load_split(
                split_name="val_1",
                num_steps_loaded=self.val_1_dataset_specs["num_steps_loaded"],
            ),
            "val_2": self._load_split(
                split_name="val_2",
                num_steps_loaded=self.val_2_dataset_specs["num_steps_loaded"],
            ),
        }

    def _load_split(
        self,
        *,
        split_name: str,
        num_steps_loaded: int,
    ) -> dict:
        npz_path = self._find_split_file(split_name)

        if self.verbose:
            print(f"[INFO] Loading {split_name} from {npz_path}")

        return self._load_single_npz_dataset(
            npz_path=npz_path,
            num_steps_loaded=num_steps_loaded,
            downsample_factor=self.downsample_factor,
        )

    def _find_split_file(self, split_name: str) -> Path:
        """
        Prefer the new format:
            sensor/train.npz
            sensor/val_1.npz
            sensor/val_2.npz

        Still accepts legacy names:
            sensor/train_dataset.npz
            sensor/val_1_dataset.npz
            sensor/val_2_dataset.npz
        """
        candidates = [
            self.dataset_paths.sensor_split(split_name),
            self.sensor_dir / f"{split_name}_dataset.npz",
        ]

        for path in candidates:
            if path.exists():
                return path

        raise FileNotFoundError(
            f"Could not find dataset file for split '{split_name}'. "
            f"Tried: {[str(p) for p in candidates]}"
        )

    @staticmethod
    def _load_single_npz_dataset(
        *,
        npz_path: Path,
        num_steps_loaded: int,
        downsample_factor: int = 1,
    ) -> dict:
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing file: {npz_path}")

        data = np.load(npz_path, allow_pickle=True)

        required_keys = ["states", "inputs", "statesRef", "timeVec"]
        for key in required_keys:
            if key not in data.files:
                raise KeyError(
                    f"Missing key '{key}' in {npz_path}. "
                    f"Available keys: {data.files}"
                )

        x = data["states"]
        u = data["inputs"]
        x_ref = data["statesRef"]
        time = data["timeVec"]

        metadata = None
        if "metadata" in data.files:
            metadata_array = data["metadata"]
            metadata = metadata_array.item() if metadata_array.shape == () else metadata_array

        if x.ndim != 3:
            raise ValueError(f"Expected states as (N_traj, T, x_dim), got {x.shape}")
        if u.ndim != 3:
            raise ValueError(f"Expected inputs as (N_traj, T, u_dim), got {u.shape}")
        if x_ref.ndim != 3:
            raise ValueError(f"Expected statesRef as (N_traj, T, ref_dim), got {x_ref.shape}")
        if time.ndim != 1:
            raise ValueError(f"Expected timeVec as (T,), got {time.shape}")

        if x.shape[0] != u.shape[0]:
            raise ValueError(f"N_traj mismatch: states {x.shape}, inputs {u.shape}")
        if x.shape[0] != x_ref.shape[0]:
            raise ValueError(f"N_traj mismatch: states {x.shape}, statesRef {x_ref.shape}")

        x = x[:, :num_steps_loaded, :]
        u = u[:, :num_steps_loaded, :]
        x_ref = x_ref[:, :num_steps_loaded, :]
        time = time[:num_steps_loaded]

        if downsample_factor > 1:
            x = x[:, ::downsample_factor, :]
            u = u[:, ::downsample_factor, :]
            x_ref = x_ref[:, ::downsample_factor, :]
            time = time[::downsample_factor]

        if x.shape[1] != u.shape[1]:
            raise ValueError(f"T mismatch: states {x.shape}, inputs {u.shape}")
        if x.shape[1] != x_ref.shape[1]:
            raise ValueError(f"T mismatch: states {x.shape}, statesRef {x_ref.shape}")
        if x.shape[1] != time.shape[0]:
            raise ValueError(f"T mismatch: states {x.shape}, timeVec {time.shape}")

        return {
            "x": x,
            "u": u,
            "x_ref": x_ref,
            "time": time,
            "metadata": metadata,
        }