from pathlib import Path

import numpy as np
import yaml

from .path_utils import find_project_root, RunPaths



def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_config_yaml(
    params: dict, path: Path | None = None, name="sensor_3d.yaml"
) -> None:
    """
    Save config dict as a YAML file.

    Parameters
    ----------
    params : dict
        Configuration dictionary.
    path : Path | None, optional
        Directory where the config file will be saved.
        If None, uses params["training_params"]["run_dir"].
    name : str, optional
        Name of the YAML file (default: "sensor_3d.yaml").
    """
    if path is None:
        save_path = Path(params["training_params"]["run_dir"])
    else:
        save_path = Path(path)
    save_path.mkdir(parents=True, exist_ok=True)

    with open(save_path / name, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            make_serializable(params),
            f,
            sort_keys=False,
            indent=2,
            allow_unicode=True
        )


def load_base_configs(
    config: str,
    task: str,
    modality: str,
    drone_dim: int,
    controller: str | None = None,
):

    if task == "training":
        base_configs_dir = Path("configs") / task / modality
    else:
        base_configs_dir = Path("configs") / task / modality / controller
    root = find_project_root()

    config_subdir = Path(f"{drone_dim}d") / f"config_{config}.yaml"
    path_config = root / base_configs_dir / config_subdir
    path = root/ "configs/training/sensor_3d.yaml"
    control_path = root / "configs/control/knmpc_sensor_3d_base.yaml"
    dataset_path = root / "configs/data/sensor_3d.yaml"
    with open(path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    with open(control_path, "r", encoding="utf-8") as f:
        control_params = yaml.safe_load(f)
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset_params = yaml.safe_load(f)
    return params, control_params, dataset_params


def load_checkpoint_config(path: RunPaths) -> dict:
    path_config = path.run_dir / "sensor_3d.yaml"
    with open(path_config, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    return params


def make_serializable(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, Path):
        return str(obj)
    else:
        return obj