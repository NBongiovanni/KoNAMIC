import numpy as np

from ..sensor_generation_config import SensorGenerationConfig
from ..trajectory_profiles import Profile
from .signals import build_multistep_signal, low_pass_filter_reference

# ============================================================
# 3D reference generation
# ============================================================

def generate_reference_3d_user(
    *,
    cfg: SensorGenerationConfig,
    time: np.ndarray,
    profile: Profile,
    x0: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Reference compacte pour quadrotor 3D.

    Convention :
        state = [x, y, z, phi, theta, psi, vx, vy, vz, p, q, r]

    columns:
        ref_user = [x_ref, y_ref, z_ref]
    """
    n_steps = len(time)
    ref = np.zeros((n_steps, 3), dtype=float)

    ref[:, 0] = x0[0]  # x
    ref[:, 1] = x0[1]  # y
    ref[:, 2] = x0[2]  # z

    n_segments = int(rng.integers(3, 7))

    if profile == "hover":
        pass

    elif profile == "step_z":
        ref[:, 2] = build_multistep_signal(
            n_samples=n_steps,
            n_segments=n_segments,
            max_abs=cfg.z_ref_max,
            rng=rng,
        )

    elif profile == "step_x":
        ref[:, 0] = build_multistep_signal(
            n_samples=n_steps,
            n_segments=n_segments,
            max_abs=cfg.z_ref_max,
            rng=rng,
        )

    elif profile == "step_y":
        ref[:, 1] = build_multistep_signal(
            n_samples=n_steps,
            n_segments=n_segments,
            max_abs=cfg.z_ref_max,
            rng=rng,
        )

    elif profile == "step_xyz":
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
        ref[:, 2] = build_multistep_signal(
            n_samples=n_steps,
            n_segments=n_segments,
            max_abs=cfg.z_ref_max,
            rng=rng,
        )

    else:
        raise ValueError(f"Unsupported 3D reference profile: {profile}")

    return low_pass_filter_reference(
        ref=ref,
        x0_ref=np.array([x0[0], x0[1], x0[2]], dtype=float),
        cfg=cfg,
        rng=rng,
    )

