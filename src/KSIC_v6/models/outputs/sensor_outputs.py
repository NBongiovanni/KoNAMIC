from dataclasses import dataclass
from torch import Tensor
from typing import Optional

@dataclass
class Pred:
    """
    Predicted quantities over future steps (t+1..T).
    """
    state: Tensor
    z: Tensor

@dataclass
class SensorValForwardOutputs:
    """
    Extended container used for validation/visualization.
    """
    rec: Tensor
    pred: Pred
    proj: Tensor
    state_gt_scaled: Tensor
    inputs_scaled: Tensor
    state_gt_physical: Optional[Tensor] = None
    inputs_physical: Optional[Tensor] = None