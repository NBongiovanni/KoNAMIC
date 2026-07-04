from __future__ import annotations

from KoNAMIC.core.control.config import KmpcControllerConfig
from KoNAMIC.core.models.model_config import ModelConfig


def resolve_kmpc_eval_config(
    *,
    controller_config,
    model_config: ModelConfig,
) -> KmpcControllerConfig:
    if not isinstance(controller_config, KmpcControllerConfig):
        raise TypeError(
            "controller=kmpc requires KmpcControllerConfig, "
            f"got {type(controller_config).__name__}."
        )

    solver_profile = solver_options_profile_for_latent_dynamics(
        model_config.z_dynamics.model
    )
    return controller_config.with_solver_options_profile(solver_profile)


def validate_kmpc_eval_config(
    *,
    controller_config,
    model_config: ModelConfig,
    controller_variant: str | None,
) -> None:
    if not isinstance(controller_config, KmpcControllerConfig):
        raise TypeError(
            "controller=kmpc requires KmpcControllerConfig, "
            f"got {type(controller_config).__name__}."
        )

    expected_profile = solver_options_profile_for_latent_dynamics(
        model_config.z_dynamics.model
    )
    actual_profile = controller_config.solver_options_profile
    if actual_profile != expected_profile:
        raise ValueError(
            "Incompatible KMPC solver profile for closed-loop simulation. "
            f"model.z_dynamics.model={model_config.z_dynamics.model!r} requires "
            f"solver_options_profile={expected_profile!r}, got {actual_profile!r}. "
            "Choose a matching controller config or update its solver_options_profile."
        )

    allowed_cost_modes_by_variant = {
        "position_in_z": {"position_in_z"},
        "state_in_z": {"state_in_z", "position_in_z"},
        "full_latent": {"full_latent"},
        "structured_latent": {"structured_latent"},
    }
    if controller_variant in allowed_cost_modes_by_variant:
        cost_mode = controller_config.cost.mode
        allowed_modes = allowed_cost_modes_by_variant[controller_variant]
        if cost_mode not in allowed_modes:
            raise ValueError(
                "Incompatible KMPC controller variant for closed-loop simulation. "
                f"controller_variant={controller_variant!r} loaded cost.mode={cost_mode!r}; "
                f"expected one of {sorted(allowed_modes)}."
            )


def solver_options_profile_for_latent_dynamics(latent_dynamics: str) -> str:
    if latent_dynamics == "linear":
        return "linear_latent_medium"
    if latent_dynamics == "bilinear":
        return "bilinear_latent_medium"
    raise ValueError(
        "Cannot derive KMPC solver_options_profile from "
        f"model.z_dynamics.model={latent_dynamics!r}."
    )
