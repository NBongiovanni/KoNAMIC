from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ModelSimuConfig:
    case_id: str | int
    label: str
    color: str


@dataclass(frozen=True)
class ClosedLoopComparisonConfig:
    system_name: str = "quadrotor_2d"
    modality: str = "sensor"
    dt: float = 0.01
    trajectory_type: str = "setpoint_tracking"
    run_status: str | None = None
    output_dir: Path = Path("/home/bongiovanni/Desktop/KoNAMIC/outputs")
    num_columns: int = 2
    task: str = "closed_loop"
    comparison_name: str = "default_name"

    models: list[ModelSimuConfig] = field(default_factory=lambda: [
        ModelSimuConfig(case_id=5, label="Bilin.", color="tab:orange"),
        # ModelSimuConfig(case_id=6, label="Lin.", color="tab:blue"),
    ])
