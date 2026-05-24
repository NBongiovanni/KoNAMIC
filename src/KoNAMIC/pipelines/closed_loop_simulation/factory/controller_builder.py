from __future__ import annotations

from typing import Optional

import numpy as np

from KoNAMIC.core.drone import DroneSpec
from KoNAMIC.core.control.controllers import (
    KoopmanMPCController,
    LQRController,
    PIDPosAttController
)

from .factory_context import FactoryContext
from .operating_point import maybe_set_operating_point


def create_koopman_mpc_controller(
    drone: DroneSpec,
    ctx: FactoryContext,
) -> KoopmanMPCController:
    if ctx.model_params is None:
        raise RuntimeError(
            "model_params must be prepared before creating KoopmanMPCController."
        )
    if ctx.koop_model is None:
        raise RuntimeError(
            "koop_model must be prepared before creating KoopmanMPCController."
        )
    if ctx.u_scaler is None:
        raise RuntimeError(
            "u_scaler must be prepared before creating KoopmanMPCController."
        )
    if ctx.solver_backend is None:
        raise RuntimeError(
            "solver_backend must be prepared before creating KoopmanMPCController."
        )

    controller = KoopmanMPCController(
        model_params=ctx.model_params,
        control_params=ctx.control_params,
        solver_backend=ctx.solver_backend,
        koop_model=ctx.koop_model,
        u_scaler=ctx.u_scaler,
        x_scaler=ctx.x_scaler,
    )

    maybe_set_operating_point(controller, drone, ctx.control_params)
    return controller


def create_lqr_controller(drone: DroneSpec, ctx: FactoryContext) -> LQRController:
    dt = float(ctx.control_params["dt"])
    x_dim = drone.x_dim
    u_dim = drone.u_dim

    u_min, u_max = extract_physical_input_bounds(ctx.control_params, u_dim, drone)

    controller = LQRController(
        dt=dt,
        x_dim=x_dim,
        u_dim=u_dim,
        u_min=u_min,
        u_max=u_max,
    )

    x_ref = maybe_get_lqr_reference(ctx.control_params, x_dim)
    if x_ref is not None:
        controller.set_reference(x_ref)

    u_eq = maybe_get_lqr_operating_input(ctx.control_params, u_dim)
    if u_eq is not None:
        controller.set_operating_point(u_eq)
    return controller


def create_pid_controller(drone: DroneSpec, ctx: FactoryContext) -> PIDPosAttController:
    controller = PIDPosAttController(
        x_dim=drone.x_dim,
        u_dim=drone.u_dim,
        dt=ctx.control_params["dt"],
        mass=drone.mass,
        gravity=drone.gravity,
        inertia=[drone.inertia[0], drone.inertia[1], drone.inertia[2]],
        kp_pos=np.asarray(ctx.control_params["kp_pos"]),
        ki_pos=np.asarray(ctx.control_params["ki_pos"]),
        kd_pos=np.asarray(ctx.control_params["kd_pos"]),

        kp_att=np.asarray(ctx.control_params["kp_att"]),
        ki_att=np.asarray(ctx.control_params["ki_att"]),
        kd_att=np.asarray(ctx.control_params["kd_att"]),

        deriv_filter_n=ctx.control_params["deriv_filter_N"],
        phi_max=np.deg2rad(ctx.control_params["phi_ref_max"]),
        theta_max=np.deg2rad(ctx.control_params["theta_ref_max"]),
        thrust_min=ctx.control_params["thrust_min"],
        thrust_max=ctx.control_params["thrust_max"],

        moment_min=(-1)*np.array(ctx.control_params["moment_max"], dtype=float),
        moment_max=np.array(ctx.control_params["moment_max"], dtype=float),
        att_cmd_alpha=ctx.control_params["att_cmd_alpha"],
        acc_xy_max=ctx.control_params["acc_xy_max"],
        max_moment_rate=np.asarray(ctx.control_params["max_moment_rates"])*2,
    )
    maybe_set_operating_point(controller, drone, ctx.control_params)
    return controller


def extract_physical_input_bounds(
    control_params: dict,
    u_dim: int,
    drone: DroneSpec,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:

    constraints = control_params.get("constraints", {})
    force_limits = constraints.get("force_limits")
    torque_limits = constraints.get("torque_limits")

    if force_limits is None or torque_limits is None:
        return None, None

    if u_dim == 2:
        u_min = np.array([force_limits[0], torque_limits[0]], dtype=float)
        u_max = np.array([force_limits[1], torque_limits[1]], dtype=float)
        return u_min, u_max

    if u_dim == 4:
        u_min = np.array(
            [force_limits[0], torque_limits[0], torque_limits[0], torque_limits[0]],
            dtype=float,
        )
        u_max = np.array(
            [force_limits[1], torque_limits[1], torque_limits[1], torque_limits[1]],
            dtype=float,
        )
        return u_min, u_max

    raise ValueError(
        f"Unsupported u_dim={u_dim} for drone_dim={drone.drone_dim}."
    )


def maybe_get_lqr_reference(
    control_params: dict,
    x_dim: int,
) -> Optional[np.ndarray]:
    lqr_cfg = control_params.get("lqr", {})
    x_ref = lqr_cfg.get("x_ref")
    if x_ref is None:
        return None

    x_ref = np.asarray(x_ref, dtype=float).reshape(-1)
    if x_ref.shape != (x_dim,):
        raise ValueError(
            f"LQR x_ref must have shape ({x_dim},), got {x_ref.shape}"
        )
    return x_ref


def maybe_get_lqr_operating_input(
    control_params: dict,
    u_dim: int,
) -> Optional[np.ndarray]:
    lqr_cfg = control_params.get("lqr", {})
    u_eq = lqr_cfg.get("u_eq")
    if u_eq is None:
        return None

    u_eq = np.asarray(u_eq, dtype=float).reshape(-1)
    if u_eq.shape != (u_dim,):
        raise ValueError(
            f"LQR u_eq must have shape ({u_dim},), got {u_eq.shape}"
        )
    return u_eq