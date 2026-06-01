import numpy as np

from KoNAMIC.core.drone import DroneSpec
from ..sensor_generation_config import SensorGenerationConfig
from ..trajectory_profiles import Profile
from .spatial import generate_reference_3d_user
from .planar import generate_reference_2d_user

def rand_uniform_sym(rng: np.random.Generator, max_abs: float) -> float:
    return float(rng.uniform(-max_abs, max_abs))


# ============================================================
# Public API
# ============================================================

def generate_reference(
    *,
    cfg: SensorGenerationConfig,
    time: np.ndarray,
    profile: Profile,
    x0: np.ndarray,
    rng: np.random.Generator,
    drone: DroneSpec,
) -> np.ndarray:
    """
    Generate a compact user-level reference.

    For drone_dim == 2:
        ref_user = [y_ref, z_ref]

    For drone_dim == 3:
        ref_user = [x_ref, y_ref, z_ref]

    This reference is not necessarily the one directly passed
    to the controller. Use build_controller_reference(...) for that.
    """
    if drone.drone_dim == 2:
        return generate_reference_2d_user(
            cfg=cfg,
            time=time,
            profile=profile,
            x0=x0,
            rng=rng,
        )

    if drone.drone_dim == 3:
        return generate_reference_3d_user(
            cfg=cfg,
            time=time,
            profile=profile,
            x0=x0,
            rng=rng,
        )

    raise ValueError(f"Unsupported drone_dim: {drone.drone_dim}")
