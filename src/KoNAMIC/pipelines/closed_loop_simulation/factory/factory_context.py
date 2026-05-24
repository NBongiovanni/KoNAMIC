from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any
from sklearn.preprocessing import StandardScaler

from KoNAMIC.core.models import VisionKoopModel
from KoNAMIC.core.control.mpc_core import SolverBackend


@dataclass
class FactoryContext:
    paths: Any
    control_params: dict
    num_simulations: int
    model_params: Optional[dict] = None
    koop_model: Optional[VisionKoopModel] = None
    solver_backend: Optional[SolverBackend] = None
    x_scaler: Optional[StandardScaler] = None
    u_scaler: Optional[StandardScaler] = None