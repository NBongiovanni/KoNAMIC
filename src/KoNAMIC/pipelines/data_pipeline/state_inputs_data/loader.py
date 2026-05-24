import logging
import os
from pathlib import Path

import numpy as np
import scipy.io as sio

from KoNAMIC.core.utils import is_jean_zay_env, build_dataset_path

class Loader:
    def __init__(
            self,
            dataset_version: int,
            drone_dim: int,
            train_dataset_specs: dict,
            val_1_dataset_specs: dict,
            val_2_dataset_specs: dict,
            downsample_factor: int,
            verbose: bool = True,
    ):
        super().__init__()
        self.dataset_version = dataset_version
        self.drone_dim = drone_dim
        self.train_dataset_specs = train_dataset_specs
        self.val_1_dataset_specs = val_1_dataset_specs
        self.val_2_dataset_specs = val_2_dataset_specs
        self.verbose = verbose
        self.downsample_factor = downsample_factor

        self.jean_zay = is_jean_zay_env()
        self.path_dir = build_dataset_path(
            self.jean_zay,
            self.drone_dim,
            "sensor",
            str(self.dataset_version),
            "train"
        ).parent

        print("[DEBUG] is_jean_zay_env() =", self.jean_zay)
        print("[DEBUG] SCRATCH =", os.environ.get("SCRATCH", None))
        print("[DEBUG] path_dir =", self.path_dir)

    def load_raw_sensor_data(self) -> dict:
        datasets = {
            "train": self._load_group_dataset(
                "train",
                self.train_dataset_specs["num_steps_loaded"],
            ),
            "val_1": self._load_group_dataset(
                "val_1",
                self.val_1_dataset_specs["num_steps_loaded"],
            ),
            "val_2": self._load_group_dataset(
                "val_2",
                self.val_2_dataset_specs["num_steps_loaded"],
            ),
        }
        return datasets

    @staticmethod
    def _load_single_raw_dataset(
            path: Path,
            phase: str,
            num_steps_loaded: int,
            downsample_factor: int = 1,
    ) -> dict:
        logger = logging.getLogger(__name__)
        path_subdir = path / phase

        states_path = path_subdir / "states.mat"
        inputs_path = path_subdir / "inputs.mat"

        if not states_path.exists():
            raise FileNotFoundError(f"Missing file: {states_path}")
        if not inputs_path.exists():
            raise FileNotFoundError(f"Missing file: {inputs_path}")

        x = sio.loadmat(str(states_path))["states"][:, :, :]  # (N_traj, T, x_dim)
        u = sio.loadmat(str(inputs_path))["inputs"][:, :, :]   # (N_traj, T, u_dim)

        assert x.ndim == 3, f"Expected x as (N_traj, T, x_dim), got {x.shape}"
        assert u.ndim == 3, f"Expected u as (N_traj, T, u_dim), got {u.shape}"
        assert x.shape[0] == u.shape[0], f"N_traj mismatch: x {x.shape}, u {u.shape}"

        x = x[:, :num_steps_loaded, :]
        u = u[:, :num_steps_loaded, :]

        # 🔹 Downsample (sans moyennage)
        if downsample_factor > 1:
            x = x[:, ::downsample_factor, :]
            u = u[:, ::downsample_factor, :]
        assert x.shape[1] == u.shape[1], f"T mismatch: x {x.shape}, u {u.shape}"

        return {"x": x, "u": u}

    @staticmethod
    def _list_numeric_subdirs(path: Path) -> list[str]:
        if not path.exists():
            return []
        subdirs = [p.name for p in path.iterdir() if p.is_dir() and p.name.isdigit()]
        return sorted(subdirs, key=lambda s: int(s))

    def _load_group_dataset(
            self,
            group_name: str,
            num_steps_loaded: int,
    ) -> dict:
        """
        Charge uniquement le format: group_name/{i}/(sensor.mat, inputs.mat)
        Refuse group_name/(sensor.mat, inputs.mat) directement.
        """
        group_path = self.path_dir / group_name
        ids = self._list_numeric_subdirs(group_path)

        if len(ids) == 0:
            if not group_path.exists():
                raise ValueError(f"{group_path} n'existe pas")

            children = sorted([p.name + ("/" if p.is_dir() else "") for p in group_path.iterdir()])
            raise ValueError(
                f"Aucun sous-dataset numérique trouvé dans {group_path}\n"
                f"Contenu du dossier: {children}\n"
                f"Astuce: j'attends {group_name}/{{i}}/(sensor.mat, inputs.mat) avec i numérique."
            )

        if self.verbose:
            print(f"{group_name} number of sub_datasets: {ids}")

        datasets = [self._load_single_raw_dataset(
            self.path_dir,
            f"{group_name}/{i}",
            num_steps_loaded,
            self.downsample_factor,
        ) for i in ids]

        dataset = datasets[0]
        for d in datasets[1:]:
            dataset = self._concat_datasets(dataset, d)
        return dataset

    @staticmethod
    def _concat_datasets(d1: dict, d2: dict) -> dict:
        """
        Concatène deux data { "x": ..., "u": ... } le long de l'axe 0.

        On suppose que l’axe 0 indexe les trajectoires (ou les samples),
        et l’axe 1 le temps, ce qui est cohérent avec le slicing [:, :num_steps].
        Si ta structure est différente, il suffira de changer axis=... ici.
        """
        x = np.concatenate([d1["x"], d2["x"]], axis=0)
        u = np.concatenate([d1["u"], d2["u"]], axis=0)
        return {"x": x, "u": u}