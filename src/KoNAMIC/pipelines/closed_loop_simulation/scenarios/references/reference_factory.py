import copy

from KoNAMIC.core.drone import DroneSpec
from .ref_traj_builder_sensor import ReferenceTrajBuilderSensor
from .ref_traj_builder_state import StateReferenceBuilder


def create_reference_builder_factory(
    controller_type: str,
    modality: str,
    control_params: dict,
    drone: DroneSpec,
    koop_model=None,
    x_scaler=None,
    verbose=None,
):
    """
    Return a per-rollout factory that builds a reference object compatible
    with the selected controller API.

    Returned object from builder.build():
    - Koopman MPC: (state_ref_traj, im_ref_traj, z_ref_traj)
    - State-feedback baselines: state_ref_traj
    """

    controller_type = controller_type.lower()

    def factory():
        # ------------------------------------------------------------------
        # Full-state baselines
        # ------------------------------------------------------------------
        if controller_type in {"lqr", "pid", "full_state_nmpc"}:
            return StateReferenceBuilder(
                control_params=control_params,
                koop_model=None,
                specs=copy.deepcopy(control_params["x_ref"]["specs"]),
                drone_dim=drone.drone_dim,
                verbose=verbose,
            )

        # ------------------------------------------------------------------
        # Koopman MPC
        # ------------------------------------------------------------------
        if controller_type == "koopman_mpc":
            if modality == "sensor":
                if koop_model is None:
                    raise ValueError("koop_model is required for koopman_mpc reference generation.")
                if x_scaler is None:
                    raise ValueError("x_scaler is required for koopman_mpc sensor reference generation.")

                return ReferenceTrajBuilderSensor(
                    control_params=control_params,
                    koop_model=koop_model,
                    specs=copy.deepcopy(control_params["x_ref"]["specs"]),
                    drone_dim=drone.drone_dim,
                    x_scaler=x_scaler,
                    verbose=verbose,
                )

            if modality == "vision":
                raise NotImplementedError(
                    "ReferenceTrajBuilderVision is not implemented yet."
                )

            raise ValueError(f"Unsupported modality: {modality}")

        raise ValueError(f"Unsupported controller type: {controller_type}")

    return factory