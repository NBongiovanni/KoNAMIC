from dataclasses import dataclass, field
from pathlib import Path

from KoNAMIC.core import utils

@dataclass(frozen=True)
class ModelSimuConfig:
    case_id: int
    label: str
    color: str


@dataclass(frozen=True)
class OpenLoopComparisonConfig:
    num_traj: int = 10
    modality: str = "sensor"
    trajectory_type: str = "setpoint_tracking"
    run_status: str = "interim"
    task: str = "open_loop"
    output_dir: Path = utils.find_project_root() / Path("/outputs")
    comparison_name: str = "open_loop_overlay"

    models: list[ModelSimuConfig] = field(default_factory=lambda: [
        ModelSimuConfig(case_id=9, label="Without state inclusion", color="tab:orange"),
        ModelSimuConfig(case_id=10, label="With state inclusion", color="tab:blue"),
    ])

