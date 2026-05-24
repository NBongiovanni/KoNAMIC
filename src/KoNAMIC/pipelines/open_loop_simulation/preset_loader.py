from pathlib import Path
import yaml

from .configs import OpenLoopComparisonConfig, ModelSimuConfig


def load_open_loop_overlay_preset(path: Path, preset_name: str) -> OpenLoopComparisonConfig:
    if not path.exists():
        raise FileNotFoundError(f"Preset file not found: {path}")

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

    return OpenLoopComparisonConfig(
        num_traj=int(preset.get("num_traj", 10)),
        modality=str(preset.get("modality", "sensor")),
        dt=float(preset.get("dt", 0.02)),
        trajectory_type=str(preset.get("trajectory_type", "setpoint_tracking")),
        run_status=str(preset.get("run_status", "interim")),
        task=str(preset.get("task", "open_loop")),
        output_dir=Path(preset.get(
            "output_dir",
            "/home/nicolas/Desktop/KoNAMIC/outputs"
        )),
        comparison_name=str(preset.get("comparison_name", preset_name)),
        models=models,
    )