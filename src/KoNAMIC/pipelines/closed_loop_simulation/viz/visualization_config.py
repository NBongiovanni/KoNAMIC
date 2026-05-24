from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ModelSimuConfig:
    case_id: int
    label: str
    color: str


@dataclass(frozen=True)
class PIDSimuConfig:
    enabled: bool = True
    label: str = "PID"
    color: str = "tab:green"
    results_path: str = "pid/1/"


@dataclass(frozen=True)
class ClosedLoopComparisonConfig:
    modality: str = "sensor"
    dt: float = 0.01
    trajectory_type: str = "setpoint_tracking"
    run_status: str = "final"
    output_dir: Path = Path("/home/bongiovanni/Desktop/KoNAMIC/outputs")
    num_columns: int = 2
    task: str = "control"
    comparison_name: str = "default_name"

    models: list[ModelSimuConfig] = field(default_factory=lambda: [
        ModelSimuConfig(case_id=5, label="Bilin.", color="tab:orange"),
        # ModelSimuConfig(case_id=6, label="Lin.", color="tab:blue"),
    ])

    pid: PIDSimuConfig = PIDSimuConfig()
