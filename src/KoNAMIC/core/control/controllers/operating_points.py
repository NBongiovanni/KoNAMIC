import numpy as np

from KoNAMIC.core.systems import DroneSpec, CartPoleSpec, SystemSpec


def maybe_set_operating_point(
        controller,
        system_spec: SystemSpec,
        controller_config: dict
):
    if not hasattr(controller, "set_operating_point"):
        return

    u_eq = get_operating_input_from_config_or_default(
        system_spec, controller_config
    )
    controller.set_operating_point(u_eq)


def maybe_get_lqr_reference(controller_config: dict, x_dim: int):
    lqr_cfg = controller_config.get("lqr", {})
    x_ref = lqr_cfg.get("x_ref")
    if x_ref is None:
        return None

    x_ref = np.asarray(x_ref, dtype=float).reshape(-1)
    if x_ref.shape != (x_dim,):
        raise ValueError(
            f"LQR x_ref must have shape ({x_dim},), got {x_ref.shape}"
        )
    return x_ref


def maybe_get_lqr_operating_input(controller_config: dict, u_dim: int):
    lqr_cfg = controller_config.get("lqr", {})
    u_eq = lqr_cfg.get("u_eq")
    if u_eq is None:
        return None

    u_eq = np.asarray(u_eq, dtype=float).reshape(-1)
    if u_eq.shape != (u_dim,):
        raise ValueError(
            f"LQR u_eq must have shape ({u_dim},), got {u_eq.shape}"
        )
    return u_eq


def build_default_operating_input(
    system_spec: SystemSpec,
) -> np.ndarray | None:

    if isinstance(system_spec, DroneSpec):
        if system_spec.u_dim == 2:
            return np.array(
                [system_spec.hover_thrust, 0.0],
                dtype=float,
            )

        if system_spec.u_dim == 4:
            return np.array(
                [system_spec.hover_thrust, 0.0, 0.0, 0.0],
                dtype=float,
            )

    if isinstance(system_spec, CartPoleSpec):
        return np.array([0.0], dtype=float)

    raise TypeError(
        f"Unsupported system specification type: "
        f"{type(system_spec).__name__}"
    )


def get_operating_input_from_config_or_default(
    system_spec: SystemSpec,
    controller_config: dict,
) -> np.ndarray:
    u_eq = controller_config.get("u_eq")

    if u_eq is not None:
        u_eq = np.asarray(u_eq, dtype=float).reshape(-1)
        if u_eq.shape != (system_spec.u_dim,):
            raise ValueError(
                f"u_eq must have shape ({system_spec.u_dim},), got {u_eq.shape}"
            )
        return u_eq

    return build_default_operating_input(system_spec)
