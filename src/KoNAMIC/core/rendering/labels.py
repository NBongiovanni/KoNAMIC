from __future__ import annotations

from KoNAMIC.core.utils import CaseConfig

_DYNAMICS_LABELS = {
    "linear": "Linear model",
    "bilinear": "Bilinear model",
}

def label_from_case(case: CaseConfig) -> str:
    try:
        return _DYNAMICS_LABELS[case.dynamics]
    except KeyError as e:
        raise ValueError(f"Unknown dynamics '{case.dynamics}'. Known: {list(_DYNAMICS_LABELS)}") from e