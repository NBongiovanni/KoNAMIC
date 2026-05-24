from pathlib import Path
import yaml

from KoNAMIC.pipelines.closed_loop_simulation.viz.visualization_config import (
    ClosedLoopComparisonConfig,
    ModelSimuConfig,
    PIDSimuConfig,
)


def load_closed_loop_overlay_preset(
    path: Path,
    preset_name: str,
) -> ClosedLoopComparisonConfig:
    if not path.exists():
        raise FileNotFoundError(f"Preset file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or "figures" not in data:
        raise ValueError(
            f"Invalid YAML format in {path}: missing top-level 'figures' key."
        )

    presets = data["figures"]
    if preset_name not in presets:
        available = ", ".join(sorted(presets.keys()))
        raise KeyError(
            f"Unknown preset '{preset_name}' in {path}. "
            f"Available figures: {available}"
        )

    preset = presets[preset_name]

    if "models" not in preset or not preset["models"]:
        raise ValueError(
            f"Preset '{preset_name}' must contain a non-empty 'models' list."
        )

    models = [
        ModelSimuConfig(
            case_id=int(m["case_id"]),
            label=str(m["label"]),
            color=str(m["color"]),
        )
        for m in preset["models"]
    ]

    pid_data = preset.get("pid", {})
    pid = PIDSimuConfig(
        enabled=bool(pid_data.get("enabled", True)),
        label=str(pid_data.get("label", "PID")),
        color=str(pid_data.get("color", "tab:green")),
        results_path=str(pid_data.get("results_path", "pid/1/")),
    )

    return ClosedLoopComparisonConfig(
        modality=str(preset.get("modality", "sensor")),
        dt=float(preset.get("dt", 0.01)),
        trajectory_type=str(preset.get("trajectory_type", "setpoint_tracking")),
        run_status=str(preset.get("run_status", "final")),
        output_dir=Path(
            preset.get(
                "output_dir",
                "/home/nicolas/Desktop/KoNAMIC/outputs",
            )
        ),
        num_columns=int(preset.get("num_columns", 2)),
        task=str(preset.get("task", "control")),
        comparison_name=str(preset.get("comparison_name", preset_name)),
        models=models,
        pid=pid,
    )