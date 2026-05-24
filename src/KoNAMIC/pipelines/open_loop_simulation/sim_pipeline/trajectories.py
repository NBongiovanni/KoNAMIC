from dataclasses import dataclass

from pathlib import Path
from typing import Tuple, Any
from KoNAMIC.core.models import SensorValForwardOutputs, VisionValForwardOutputs

@dataclass
class OpenLoopSensorResult:
    val_output: SensorValForwardOutputs
    u_scaler: Any
    x_scaler: Any
    run_dir: Path
    open_loop_eval_dir: Path


@dataclass
class OpenLoopVisionResult:
    val_output: VisionValForwardOutputs
    u_scaler: Any
    training_params: dict
    run_dir: Path
