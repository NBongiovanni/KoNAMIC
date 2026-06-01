import numpy as np

from ..trajectory_profiles import Profile
from ..sensor_generation_config import SensorGenerationConfig
from .signals import build_multistep_signal, low_pass_filter_reference
# ============================================================
# 2D reference generation
# ============================================================

def generate_reference_2d_user(
    *,
    cfg: SensorGenerationConfig,
    time: np.ndarray,
    profile: Profile,
    x0: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Reference compacte pour planar quadrotor.

    Convention supposée :
        state = [y, z, theta, y_dot, z_dot, theta_dot]

    columns:
        ref_user = [y_ref, z_ref]
    """
    n_steps = len(time)
    ref = np.zeros((n_steps, 2), dtype=float)

    # Initialisation à la position initiale
    ref[:, 0] = x0[0]  # y
    ref[:, 1] = x0[1]  # z

    n_segments = int(rng.integers(3, 7))

    if profile == "hover":
        pass

    elif profile == "step_z":
        ref[:, 1] = build_multistep_signal(
            n_samples=n_steps,
            n_segments=n_segments,
            max_abs=cfg.z_ref_max,
            rng=rng,
        )

    elif profile in ("step_y", "step_x"):
        # Pour le planar, votre convention actuelle semble utiliser y,z.
        # Si vous préférez x,z, renommez simplement y_ref_max en x_ref_max
        # dans le YAML et dans les labels.
        max_abs = getattr(cfg, "y_ref_max", cfg.x_ref_max)
        ref[:, 0] = build_multistep_signal(
            n_samples=n_steps,
            n_segments=n_segments,
            max_abs=cfg.z_ref_max,
            rng=rng,
        )

    elif profile in ("step_yz", "step_xz", "step_xyz"):
        max_abs = getattr(cfg, "y_ref_max", cfg.x_ref_max)
        ref[:, 0] = build_multistep_signal(
            n_samples=n_steps,
            n_segments=n_segments,
            max_abs=cfg.z_ref_max,
            rng=rng,
        )

        ref[:, 1] = build_multistep_signal(
            n_samples=n_steps,
            n_segments=n_segments,
            max_abs=cfg.z_ref_max,
            rng=rng,
        )

    else:
        raise ValueError(f"Unsupported 2D reference profile: {profile}")

    return low_pass_filter_reference(
        ref=ref,
        x0_ref=np.array([x0[0], x0[1]], dtype=float),
        cfg=cfg,
        rng=rng,
    )