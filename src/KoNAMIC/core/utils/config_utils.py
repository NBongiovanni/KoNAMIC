from pathlib import Path

import numpy as np
import yaml

from .paths.run_paths import RunPaths


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(
        params: dict,
        path: Path | None = None,
        name="sensor_3d.yaml"
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