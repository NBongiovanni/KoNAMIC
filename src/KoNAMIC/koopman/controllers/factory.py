from __future__ import annotations

from typing import TypeAlias

import numpy as np

from KoNAMIC import config
from KoNAMIC.core.control.config import KlqrControllerConfig, KmpcControllerConfig
from KoNAMIC.core.control.mpc_solver import AcadosMPCSolver
from KoNAMIC.core.scaling import DatasetScalers
from KoNAMIC.koopman.controllers.lqr import KoopmanLQRController
from KoNAMIC.koopman.controllers.mpc import KoopmanMPCController
from KoNAMIC.koopman.models import SensorKoopModel, VisionKoopModel
from KoNAMIC.koopman.models.model_config import ModelConfig


KoopmanControllerConfig: TypeAlias = KmpcControllerConfig | KlqrControllerConfig
KoopmanModel: TypeAlias = SensorKoopModel | VisionKoopModel


def build_koopman_controller(
    modality: config.Modality,
    data_scalers: DatasetScalers,
    model_config: ModelConfig,
    controller_config: KoopmanControllerConfig,
    closed_loop_paths,
    koop_model: KoopmanModel,
) -> KoopmanMPCController | KoopmanLQRController:
    return build_koopman_controller_for_dir(
        modality=modality,
        controller_dir=closed_loop_paths.eval_dir,
        data_scalers=data_scalers,
        model_config=model_config,
        controller_config=controller_config,
        koop_model=koop_model,
    )


def build_koopman_controller_for_dir(
    *,
    modality: config.Modality,
    controller_dir,
    data_scalers: DatasetScalers,
    model_config: ModelConfig,
    controller_config: KoopmanControllerConfig,
    koop_model: KoopmanModel,
) -> KoopmanMPCController | KoopmanLQRController:
    if isinstance(controller_config, KmpcControllerConfig):
        solver_backend = AcadosMPCSolver(controller_dir, controller_config)
        return KoopmanMPCController(
            modality=modality,
            model_config=model_config,
            controller_config=controller_config,
            solver=solver_backend,
            koop_model=koop_model,
            scalers=data_scalers,
            control_run_dir=controller_dir,
        )

    if isinstance(controller_config, KlqrControllerConfig):
        u_min, u_max = build_klqr_input_bounds(controller_config, int(koop_model.u_dim))
        return KoopmanLQRController(
            model_config=model_config,
            koop_model=koop_model,
            scalers=data_scalers,
            q_diag=np.asarray(controller_config.q_diag, dtype=float),
            r_diag=np.asarray(controller_config.r_diag, dtype=float),
            u_min=u_min,
            u_max=u_max,
        )

    raise TypeError(
        "Unsupported Koopman controller config: "
        f"{type(controller_config).__name__}."
    )


def build_klqr_input_bounds(
    controller_config: KlqrControllerConfig,
    u_dim: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    constraints = controller_config.constraints
    if not constraints.use_inputs_constraints:
        return None, None

    force_limits = constraints.force_limits
    torque_limits = constraints.torque_limits
    if torque_limits is None:
        raise ValueError("controller.constraints.torque_limits is required for KLQR.")

    if u_dim == 2:
        u_min = np.array([force_limits[0], torque_limits[0]], dtype=float)
        u_max = np.array([force_limits[1], torque_limits[1]], dtype=float)
    elif u_dim == 4:
        u_min = np.array(
            [force_limits[0], torque_limits[0], torque_limits[0], torque_limits[0]],
            dtype=float,
        )
        u_max = np.array(
            [force_limits[1], torque_limits[1], torque_limits[1], torque_limits[1]],
            dtype=float,
        )
    else:
        raise ValueError(f"Unsupported u_dim={u_dim} for KLQR input bounds.")

    return u_min, u_max
