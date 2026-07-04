from typing import Any

from KoNAMIC.core.systems import SystemSpec

from .compute_sensor import SensorLossComputer
from .compute_vision import VisionLossComputer


def build_loss_computer(
    *,
    modality: str,
    loss_weights,
    system_spec: SystemSpec,
    x_scaler: Any | None = None,
    scale_x: bool = True,
):
    if modality == "sensor":
        state_rmse_units_scale = _state_rmse_units_scale(
            x_scaler=x_scaler,
            scale_x=scale_x,
        )
        return SensorLossComputer(
            loss_weights,
            system_spec.get_x_names(),
            state_rmse_units_scale=state_rmse_units_scale,
        )

    if modality == "vision":
        return VisionLossComputer(loss_weights)

    raise ValueError(f"Unknown modality: {modality}")

def _state_rmse_units_scale(*, x_scaler: Any | None, scale_x: bool) -> list[float] | None:
    if not scale_x:
        return None

    if x_scaler is None:
        return None

    scale = getattr(x_scaler, "scale_", None)
    if scale is None:
        return None

    return [float(value) for value in scale]
