from __future__ import annotations

import numpy as np

from KoNAMIC.core.control.config import (
    ControllerConfigT,
    LqrControllerConfig,
    Quadrotor2DPidConfig,
    Quadrotor3DPidConfig,
)
from KoNAMIC.core.systems import CartPoleSpec, DroneSpec, SystemSpec
from KoNAMIC.utils import as_array, as_float
from ..controllers import (
    BaseController,
    CartPoleLQRController,
    Quadrotor2DLQRHoverController,
    Quadrotor2DPIDController,
    Quadrotor3DLQRHoverController,
    Quadrotor3DPIDController,
)


def build_baseline_controller(
    system_spec: SystemSpec,
    controller_config: ControllerConfigT,
) -> BaseController:
    system_name = system_spec.system_name

    if system_name == "quadrotor_2d":
        if not isinstance(system_spec, DroneSpec):
            raise TypeError(
                f"Expected DroneSpec for system_name={system_name!r}, "
                f"got {type(system_spec).__name__}."
            )
        if isinstance(controller_config, Quadrotor2DPidConfig):
            return _build_quadrotor_2d_pid_controller(system_spec, controller_config)

        if isinstance(controller_config, LqrControllerConfig):
            return _build_quadrotor_2d_lqr_controller(system_spec, controller_config)

        raise ValueError(
            f"Unsupported controller={controller_config.controller_type!r} "
            f"for system_name={system_name!r}."
        )

    if system_name == "quadrotor_3d":
        if not isinstance(system_spec, DroneSpec):
            raise TypeError(
                f"Expected DroneSpec for system_name={system_name!r}, "
                f"got {type(system_spec).__name__}."
            )
        if isinstance(controller_config, Quadrotor3DPidConfig):
            return _build_quadrotor_3d_pid_controller(system_spec, controller_config)

        if isinstance(controller_config, LqrControllerConfig):
            return _build_quadrotor_3d_lqr_controller(system_spec, controller_config)

        raise ValueError(
            f"Unsupported controller={controller_config.controller_type!r} "
            f"for system_name={system_name!r}."
        )

    if system_name == "cartpole":
        if not isinstance(system_spec, CartPoleSpec):
            raise TypeError(
                f"Expected CartPoleSpec for system_name={system_name!r}, "
                f"got {type(system_spec).__name__}."
            )
        if isinstance(controller_config, LqrControllerConfig):
            return _build_cartpole_lqr_controller(system_spec, controller_config)

        raise ValueError(
            f"Unsupported controller={controller_config.controller_type!r} "
            f"for system_name={system_name!r}."
        )

    raise ValueError(f"Unsupported system_name={system_name!r}.")


def _build_quadrotor_2d_lqr_controller(
    system_spec: DroneSpec,
    controller_config: LqrControllerConfig,
) -> Quadrotor2DLQRHoverController:
    return Quadrotor2DLQRHoverController(
        dt=controller_config.dt,
        mass=system_spec.mass,
        inertia_y=system_spec.inertia[2],
        gravity=system_spec.gravity,
        q_diag=as_array(controller_config.q_diag),
        r_diag=as_array(controller_config.r_diag),
        thrust_min=controller_config.thrust_min,
        thrust_max=controller_config.thrust_max,
        moment_min=-as_float(controller_config.moment_max),
        moment_max=as_float(controller_config.moment_max),
        max_moment_rate=as_float(controller_config.max_moment_rates),
    )


def _build_quadrotor_3d_lqr_controller(
    system_spec: DroneSpec,
    controller_config: LqrControllerConfig,
) -> Quadrotor3DLQRHoverController:
    inertia = as_array(system_spec.inertia)
    if inertia.shape != (3,):
        raise ValueError(f"system_spec.inertia must have shape (3,), got {inertia.shape}.")

    moment_max = as_float(controller_config.moment_max)
    max_moment_rate = as_float(controller_config.max_moment_rates)

    moment_max_vec = moment_max * np.ones(3, dtype=float)
    max_moment_rates_vec = max_moment_rate * np.ones(3, dtype=float)

    return Quadrotor3DLQRHoverController(
        dt=controller_config.dt,
        mass=system_spec.mass,
        inertia=inertia,
        gravity=system_spec.gravity,
        q_diag=as_array(controller_config.q_diag),
        r_diag=as_array(controller_config.r_diag),
        thrust_min=controller_config.thrust_min,
        thrust_max=controller_config.thrust_max,
        moment_min=-moment_max_vec,
        moment_max=moment_max_vec,
        max_moment_rates=max_moment_rates_vec,
    )


def _build_quadrotor_2d_pid_controller(
    system_spec: DroneSpec,
    controller_config: Quadrotor2DPidConfig,
) -> Quadrotor2DPIDController:
    return Quadrotor2DPIDController(
        dt=controller_config.dt,
        x_dim=system_spec.x_dim,
        u_dim=system_spec.u_dim,
        mass=system_spec.mass,
        inertia_y=system_spec.inertia[2],
        gravity=system_spec.gravity,
        kp_pos=as_array(controller_config.kp_pos),
        ki_pos=as_array(controller_config.ki_pos),
        kd_pos=as_array(controller_config.kd_pos),
        kp_att=as_float(controller_config.kp_att),
        ki_att=as_float(controller_config.ki_att),
        kd_att=as_float(controller_config.kd_att),
        deriv_filter_n=controller_config.deriv_filter_N,
        theta_max=np.deg2rad(as_float(controller_config.theta_ref_max)),
        thrust_min=as_float(controller_config.thrust_min),
        thrust_max=as_float(controller_config.thrust_max),
        att_cmd_alpha=as_float(controller_config.att_cmd_alpha),
        moment_max=as_float(controller_config.moment_max),
        acc_x_max=as_float(controller_config.acc_xy_max) * system_spec.gravity,
        max_moment_rate=as_float(controller_config.max_moment_rates),
    )


def _build_cartpole_lqr_controller(
    system_spec: CartPoleSpec,
    controller_config: LqrControllerConfig,
) -> CartPoleLQRController:
    return CartPoleLQRController(
        dt=controller_config.dt,
        cart_mass=system_spec.cart_mass,
        pole_mass=system_spec.pole_mass,
        pole_length=system_spec.pole_length,
        gravity=system_spec.gravity,
        q_diag=as_array(controller_config.q_diag),
        r_diag=as_array(controller_config.r_diag),
        force_min=as_float(controller_config.force_min),
        force_max=as_float(controller_config.force_max),
    )


def _build_quadrotor_3d_pid_controller(
    system_spec: DroneSpec,
    controller_config: Quadrotor3DPidConfig,
) -> Quadrotor3DPIDController:
    return Quadrotor3DPIDController(
        dt=controller_config.dt,
        x_dim=system_spec.x_dim,
        u_dim=system_spec.u_dim,
        mass=system_spec.mass,
        inertia=system_spec.inertia,
        gravity=system_spec.gravity,
        kp_pos=as_array(controller_config.kp_pos),
        ki_pos=as_array(controller_config.ki_pos),
        kd_pos=as_array(controller_config.kd_pos),
        kp_att=as_array(controller_config.kp_att),
        ki_att=as_array(controller_config.ki_att),
        kd_att=as_array(controller_config.kd_att),
        deriv_filter_n=controller_config.deriv_filter_N,
        phi_max=np.deg2rad(controller_config.phi_ref_max),
        theta_max=np.deg2rad(controller_config.theta_ref_max),
        thrust_min=as_float(controller_config.thrust_min),
        thrust_max=as_float(controller_config.thrust_max),
        att_cmd_alpha=as_float(controller_config.att_cmd_alpha),
        moment_max=as_array(controller_config.moment_max),
        acc_xy_max=as_float(controller_config.acc_xy_max) * system_spec.gravity,
        max_moment_rate=as_array(controller_config.max_moment_rates),
    )
