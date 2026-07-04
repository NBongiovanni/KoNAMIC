from pathlib import Path
import yaml
from dataclasses import dataclass, field

from KoNAMIC import paths


def _resolve_project_path(path_value: str | Path | None, default: Path) -> Path:
    """
    Resolve a path that may be absolute or relative to the project root.
    """
    project_root = paths.find_project_root()

    if path_value is None:
        return default

    path = Path(path_value)

    if path.is_absolute():
        return path

    return project_root / path


def load_open_loop_overlay_preset(
    path: Path,
    system_name: str,
    preset_name: str,
    modality: str,
):
    path = Path(path) / system_name / f"{modality}.yaml"

    if not path.exists():
        raise FileNotFoundError(f"Preset file not found: {path}")

    project_root = paths.find_project_root()

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML format in {path}: expected a mapping at root.")

    if "comparisons" in data:
        presets = data["comparisons"]
    elif "figures" in data:
        presets = data["figures"]
    else:
        raise ValueError(
            f"Invalid YAML format in {path}: missing top-level 'comparisons' key."
        )

    if not isinstance(presets, dict):
        raise ValueError(f"Invalid YAML format in {path}: comparison presets must be a mapping.")

    if preset_name not in presets:
        available = ", ".join(sorted(presets.keys()))
        raise KeyError(
            f"Unknown preset '{preset_name}' in {path}. "
            f"Available comparisons: {available}"
        )

    preset = presets[preset_name]
    if not isinstance(preset, dict):
        raise ValueError(f"Preset '{preset_name}' in {path} must be a mapping.")

    eval_type = preset.get("eval_type")
    if eval_type is not None and eval_type != "open_loop":
        raise ValueError(
            f"Preset '{preset_name}' in {path} has eval_type={eval_type!r}; "
            "expected 'open_loop'."
        )

    if "experiments" not in preset or not preset["experiments"]:
        raise ValueError(f"Preset '{preset_name}' must contain a non-empty 'experiments' list.")

    if "trajectory_type" not in preset:
        raise ValueError(
            f"Preset '{preset_name}' in {path} must define 'trajectory_type'."
        )

    models = []
    for idx, raw_experiment in enumerate(preset["experiments"]):
        experiment = _parse_model_config(raw_experiment, idx)
        models.append(
            ModelSimuConfig(
                case_id=experiment.case_id,
                label=experiment.label,
                color=experiment.color,
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
        system_name=system_name,
        modality=str(preset.get("modality", "sensor")),
        trajectory_type=str(preset["trajectory_type"]),
        run_status=str(preset.get("run_status", "interim")),
        task=str(preset.get("task", "open_loop")),
        output_dir=output_dir,
        comparison_name=str(preset.get("comparison_name", preset_name)),
        models=models,
    )


@dataclass(frozen=True)
class ModelSimuConfig:
    case_id: str | int
    label: str
    color: str


@dataclass(frozen=True)
class OpenLoopComparisonConfig:
    system_name: str = "quadrotor_2d"
    modality: str = "sensor"
    trajectory_type: str = "setpoint_tracking"
    run_status: str = "interim"
    task: str = "open_loop"
    output_dir: Path = paths.find_project_root() / Path("/outputs")
    comparison_name: str = "open_loop_overlay"

    models: list[ModelSimuConfig] = field(default_factory=lambda: [
        ModelSimuConfig(case_id="lin_sensor_1", label="lin_sensor_1", color="tab:blue"),
        ModelSimuConfig(case_id="bilin_sensor_1", label="bilin_sensor_1", color="tab:orange"),
    ])


_DEFAULT_COLORS = (
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
)


def _parse_model_config(raw_experiment, idx: int) -> ModelSimuConfig:
    if isinstance(raw_experiment, str):
        return ModelSimuConfig(
            case_id=raw_experiment,
            label=raw_experiment,
            color=_DEFAULT_COLORS[idx % len(_DEFAULT_COLORS)],
        )

    if isinstance(raw_experiment, dict):
        case_id = raw_experiment.get("case_id", raw_experiment.get("experiment_id"))
        if case_id is None:
            raise ValueError(
                "Experiment mappings must define either 'case_id' or 'experiment_id'."
            )

        return ModelSimuConfig(
            case_id=case_id,
            label=str(raw_experiment.get("label", case_id)),
            color=str(raw_experiment.get("color", _DEFAULT_COLORS[idx % len(_DEFAULT_COLORS)])),
        )

    raise TypeError(
        "Experiments must be strings or mappings, "
        f"got {type(raw_experiment).__name__}."
    )
