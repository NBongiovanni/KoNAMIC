import numpy as np

from KoNAMIC.core.drone import DroneSpec
from .sensor_generation_config import SensorGenerationConfig
from .profile import Profile


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


def build_controller_reference(
    *,
    ref_user: np.ndarray,
    drone,
) -> np.ndarray:
    """
    Convert a compact user-level reference into a full state-shaped
    reference expected by the PID controllers.

    For 2D:
        state = [y, z, theta, y_dot, z_dot, theta_dot]
        ref_user = [y_ref, z_ref]
        ref_controller = [y_ref, z_ref, theta_ref, 0, 0, 0]

    For 3D:
        state = [x, y, z, phi, theta, psi, vx, vy, vz, p, q, r]
        ref_user = [x_ref, y_ref, z_ref]
        ref_controller = [x_ref, y_ref, z_ref, phi_ref, theta_ref, psi_ref, 0, ..., 0]
    """
    ref_user = np.asarray(ref_user, dtype=float)

    if ref_user.ndim != 2:
        raise ValueError(f"ref_user must be 2D, got shape {ref_user.shape}")

    n_steps = ref_user.shape[0]
    ref_controller = np.zeros((n_steps, drone.x_dim), dtype=float)

    if drone.drone_dim == 2:
        if ref_user.shape[1] != 2:
            raise ValueError(
                f"For drone_dim=2, ref_user must have shape (T, 2), "
                f"got {ref_user.shape}"
            )

        # Convention planar actuelle :
        # x_state = [y, z, theta, y_dot, z_dot, theta_dot]
        ref_controller[:, 0] = ref_user[:, 0]  # y_ref
        ref_controller[:, 1] = ref_user[:, 1]  # z_ref

        return ref_controller

    if drone.drone_dim == 3:
        if ref_user.shape[1] != 3:
            raise ValueError(
                f"For drone_dim=3, ref_user must have shape (T, 3), "
                f"got {ref_user.shape}"
            )

        # Convention 3D :
        # x_state = [x, y, z, phi, theta, psi, vx, vy, vz, p, q, r]
        ref_controller[:, 0] = ref_user[:, 0]  # x_ref
        ref_controller[:, 1] = ref_user[:, 1]  # y_ref
        ref_controller[:, 2] = ref_user[:, 2]  # z_ref

        # phi_ref, theta_ref, psi_ref restent à 0.
        # Le contrôleur peut ensuite écrire ses consignes internes
        # phi/theta dans ref_controller[:, 3:5] si nécessaire.

        return ref_controller

    raise ValueError(f"Unsupported drone_dim: {drone.drone_dim}")


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

    if profile == "step_z":
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


# ============================================================
# Shared utilities
# ============================================================

def low_pass_filter_reference(
    *,
    ref: np.ndarray,
    x0_ref: np.ndarray,
    cfg: SensorGenerationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Apply the same first-order smoothing used in the original implementation.
    """
    ref = np.asarray(ref, dtype=float)
    filtered = np.zeros_like(ref)

    tau = float(rng.uniform(cfg.tau_ref_min, cfg.tau_ref_max))
    alpha = np.exp(-cfg.dt / tau)

    filtered[0] = x0_ref

    for k in range(1, ref.shape[0]):
        filtered[k] = alpha * filtered[k - 1] + (1.0 - alpha) * ref[k]

    return filtered


def build_multistep_signal(
    n_samples: int,
    n_segments: int,
    max_abs: float,
    rng: np.random.Generator,
) -> np.ndarray:
    signal = np.zeros(n_samples, dtype=float)
    lengths = random_segment_lengths(n_samples, n_segments, rng)

    idx = 0
    for length in lengths:
        value = rand_uniform_sym(rng, max_abs)
        signal[idx:idx + length] = value
        idx += length

    return signal


def random_segment_lengths(
    n_samples: int,
    n_segments: int,
    rng: np.random.Generator,
) -> np.ndarray:
    weights = rng.random(n_segments)
    weights /= weights.sum()

    lengths = np.maximum(1, np.floor(weights * n_samples).astype(int))

    deficit = n_samples - lengths.sum()

    while deficit > 0:
        i = rng.integers(0, n_segments)
        lengths[i] += 1
        deficit -= 1

    while deficit < 0:
        i = rng.integers(0, n_segments)
        if lengths[i] > 1:
            lengths[i] -= 1
            deficit += 1

    return lengths