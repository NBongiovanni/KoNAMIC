from pathlib import Path

import numpy as np
import yaml


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(
        params: dict,
        system_name: str,
        path: Path | None = None,
) -> None:
    """
    Save config dict as a YAML file.
    """
    save_path = Path(path)
    save_path.mkdir(parents=True, exist_ok=True)

    with open(save_path / system_name, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            make_serializable(params),
            f,
            sort_keys=False,
            indent=2,
            allow_unicode=True
        )


def load_checkpoint_config(run_dir) -> dict:
    path_config = run_dir / "sensor_3d.yaml"
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


def require_keys(cfg: dict, keys: list[str], context: str) -> None:
    missing = [key for key in keys if key not in cfg]

    if missing:
        raise KeyError(
            f"Missing required key(s) in {context}: {missing}. "
            f"Available keys: {list(cfg.keys())}"
        )