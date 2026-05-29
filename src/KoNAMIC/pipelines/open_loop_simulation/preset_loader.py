from pathlib import Path
import yaml

from KoNAMIC.core import utils

from .configs import OpenLoopComparisonConfig, ModelSimuConfig


def _resolve_project_path(path_value: str | Path | None, default: Path) -> Path:
    """
    Resolve a path that may be absolute or relative to the project root.
    """
    project_root = utils.find_project_root()

    if path_value is None:
        return default

    path = Path(path_value)

    if path.is_absolute():
        return path

    return project_root / path


def load_open_loop_overlay_preset(
    path: Path,
    preset_name: str,
) -> OpenLoopComparisonConfig:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Preset file not found: {path}")

    project_root = utils.find_project_root()

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "figures" not in data:
        raise ValueError(f"Invalid YAML format in {path}: missing top-level 'figures' key.")

    presets = data["figures"]
    if preset_name not in presets:
        available = ", ".join(sorted(presets.keys()))
        raise KeyError(
            f"Unknown preset '{preset_name}' in {path}. "
            f"Available figures: {available}"
        )

    preset = presets[preset_name]

    if "models" not in preset or not preset["models"]:
        raise ValueError(f"Preset '{preset_name}' must contain a non-empty 'models' list.")

    models = []
    for m in preset["models"]:
        models.append(
            ModelSimuConfig(
                case_id=int(m["case_id"]),
                label=str(m["label"]),
                color=str(m["color"]),
            )
        )

    output_dir = preset.get("output_dir", None)

    if output_dir is None:
        output_dir = project_root / "outputs"
    else:
        output_dir = Path(output_dir)
        if not output_dir.is_absolute():
            output_dir = project_root / output_dir

    return OpenLoopComparisonConfig(
        num_traj=int(preset.get("num_traj", 10)),
        modality=str(preset.get("modality", "sensor")),
        trajectory_type=str(preset.get("trajectory_type", "setpoint_tracking")),
        run_status=str(preset.get("run_status", "interim")),
        task=str(preset.get("task", "open_loop")),
        output_dir=output_dir,
        comparison_name=str(preset.get("comparison_name", preset_name)),
        models=models,
    )