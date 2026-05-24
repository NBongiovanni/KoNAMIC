import numpy as np
from KoNAMIC.core.drone import DroneSpec


def build_default_operating_input(drone: DroneSpec) -> np.ndarray:
    if drone.u_dim == 2:
        return np.array([drone.hover_thrust, 0.0], dtype=float)

    if drone.u_dim == 4:
        return np.array([drone.hover_thrust, 0.0, 0.0, 0.0], dtype=float)

    raise ValueError(f"Unsupported u_dim={drone.u_dim}")


def get_operating_input_from_config_or_default(
    drone: DroneSpec,
    control_params: dict,
) -> np.ndarray:
    u_eq = control_params.get("u_eq")

    if u_eq is not None:
        u_eq = np.asarray(u_eq, dtype=float).reshape(-1)
        if u_eq.shape != (drone.u_dim,):
            raise ValueError(
                f"u_eq must have shape ({drone.u_dim},), got {u_eq.shape}"
            )
        return u_eq

    return build_default_operating_input(drone)


def maybe_set_operating_point(controller, drone: DroneSpec, control_params: dict):
    if not hasattr(controller, "set_operating_point"):
        return

    u_eq = get_operating_input_from_config_or_default(drone, control_params)
    controller.set_operating_point(u_eq)