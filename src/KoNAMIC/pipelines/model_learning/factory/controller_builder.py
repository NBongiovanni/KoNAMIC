from __future__ import annotations

from KoNAMIC.core.control.controllers import KoopmanMPCController


def create_koopman_mpc_controller(
    model_params: dict,
    control_params: dict,
    solver_backend,
    koop_model,
    u_scaler,
    x_scaler,
) -> KoopmanMPCController:

    return KoopmanMPCController(
        model_params=model_params,
        control_params=control_params,
        solver_backend=solver_backend,
        koop_model=koop_model,
        u_scaler=u_scaler,
        x_scaler=x_scaler,
    )