import numpy as np
from typing import Callable

from KoNAMIC.core.drone import DroneSpec
from KoNAMIC.core.control.controllers import (
    PIDPosAttController,
    PIDPlanarPosAttController,
)
from KoNAMIC.pipelines.data_generation.sensor import SensorDatasetParams


def as_array(value) -> np.ndarray:
    return np.asarray(value, dtype=float)


def as_float(value) -> float:
    """
    Accept scalar or one-element list/array from YAML.
    """
    return float(np.asarray(value, dtype=float).reshape(-1)[0])

def build_controller_factory(
    *,
    drone: DroneSpec,
    cfg: SensorDatasetParams,
    ctrl_cfg: dict,
) -> Callable:
    if drone.drone_dim == 2:
        return build_planar_controller_factory(
            drone=drone,
            cfg=cfg,
            ctrl_cfg=ctrl_cfg,
        )

    if drone.drone_dim == 3:
        return build_3d_controller_factory(
            drone=drone,
            cfg=cfg,
            ctrl_cfg=ctrl_cfg,
        )

    raise ValueError(f"Unsupported drone_dim: {drone.drone_dim}")


def build_3d_controller_factory(
    *,
    drone: DroneSpec,
    cfg: SensorDatasetParams,
    ctrl_cfg: dict,
) -> Callable:
    def controller_factory():
        return PIDPosAttController(
            dt=cfg.dt,
            x_dim=drone.x_dim,
            u_dim=drone.u_dim,
            mass=drone.mass,
            inertia=drone.inertia,
            gravity=drone.gravity,
            kp_pos=as_array(ctrl_cfg["kp_pos"]),
            ki_pos=as_array(ctrl_cfg["ki_pos"]),
            kd_pos=as_array(ctrl_cfg["kd_pos"]),
            kp_att=as_array(ctrl_cfg["kp_att"]),
            ki_att=as_array(ctrl_cfg["ki_att"]),
            kd_att=as_array(ctrl_cfg["kd_att"]),
            deriv_filter_n=ctrl_cfg["deriv_filter_N"],
            phi_max=np.deg2rad(ctrl_cfg["phi_ref_max"]),
            theta_max=np.deg2rad(ctrl_cfg["theta_ref_max"]),
            thrust_min=ctrl_cfg["thrust_min"] * drone.hover_thrust,
            thrust_max=ctrl_cfg["thrust_max"] * drone.hover_thrust,
            att_cmd_alpha=ctrl_cfg["att_cmd_alpha"],
            moment_max=as_array(ctrl_cfg["moment_max"]),
            acc_xy_max=ctrl_cfg["acc_xy_max"] * drone.gravity,
            max_moment_rate=as_array(ctrl_cfg["max_moment_rates"]),
        )
    return controller_factory


def build_planar_controller_factory(
    *,
    drone: DroneSpec,
    cfg: SensorDatasetParams,
    ctrl_cfg: dict,
) -> Callable:
    def controller_factory():
        return PIDPlanarPosAttController(
            dt=cfg.dt,
            x_dim=drone.x_dim,
            u_dim=drone.u_dim,
            mass=drone.mass,
            inertia_y=drone.inertia[2],
            gravity=drone.gravity,
            kp_pos=as_array(ctrl_cfg["kp_pos"]),
            ki_pos=as_array(ctrl_cfg["ki_pos"]),
            kd_pos=as_array(ctrl_cfg["kd_pos"]),
            kp_att=as_float(ctrl_cfg["kp_att"]),
            ki_att=as_float(ctrl_cfg["ki_att"]),
            kd_att=as_float(ctrl_cfg["kd_att"]),
            deriv_filter_n=ctrl_cfg["deriv_filter_N"],
            theta_max=np.deg2rad(ctrl_cfg["theta_ref_max"]),
            thrust_min=ctrl_cfg["thrust_min"] * drone.hover_thrust,
            thrust_max=ctrl_cfg["thrust_max"] * drone.hover_thrust,
            att_cmd_alpha=ctrl_cfg["att_cmd_alpha"],
            moment_max=as_float(ctrl_cfg["moment_max"]),
            acc_x_max=ctrl_cfg["acc_xy_max"] * drone.gravity,
            max_moment_rate=as_float(ctrl_cfg["max_moment_rates"]),
        )

    return controller_factory
