import numpy as np

from .config import DataGenerationConfig
from .profile import Profile


def rand_uniform_sym(rng: np.random.Generator, max_abs: float) -> float:
    return float(rng.uniform(-max_abs, max_abs))


def ref6_to_ref12(ref6: np.ndarray) -> np.ndarray:
    """
    Adaptateur pour PIDPosAttController.

    ref6:
        [x_ref, y_ref, z_ref, psi_ref, phi_ref_internal, theta_ref_internal]

    ref12:
        même convention que l'état:
        [x, y, z, phi, theta, psi, vx, vy, vz, p, q, r]
    """
    ref12 = np.zeros((ref6.shape[0], 12), dtype=float)
    ref12[:, 0] = ref6[:, 0]
    ref12[:, 1] = ref6[:, 1]
    ref12[:, 2] = ref6[:, 2]
    ref12[:, 3] = ref6[:, 4]
    ref12[:, 4] = ref6[:, 5]
    ref12[:, 5] = ref6[:, 3]
    return ref12


def generate_reference_6d(
    cfg: DataGenerationConfig,
    time: np.ndarray,
    profile: Profile,
    x0: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Ref compacte type Matlab:
    columns = [x_ref, y_ref, z_ref, psi_ref, phi_ref_internal, theta_ref_internal]
    """
    n_steps = len(time)
    ref = np.zeros((n_steps, 6), dtype=float)

    ref[:, 0] = x0[0]
    ref[:, 1] = x0[1]
    ref[:, 2] = x0[2]
    ref[:, 3] = 0.0

    n_segments = int(rng.integers(3, 7))

    if profile == "step_z":
        ref[:, 2] = build_multistep_signal(n_steps, n_segments, cfg.z_ref_max, rng)

    elif profile == "step_x":
        ref[:, 0] = build_multistep_signal(n_steps, n_segments, cfg.x_ref_max, rng)

    elif profile == "step_y":
        ref[:, 1] = build_multistep_signal(n_steps, n_segments, cfg.y_ref_max, rng)

    elif profile == "step_xyz":
        ref[:, 0] = build_multistep_signal(n_steps, n_segments, cfg.x_ref_max, rng)
        ref[:, 1] = build_multistep_signal(n_steps, n_segments, cfg.y_ref_max, rng)
        ref[:, 2] = build_multistep_signal(n_steps, n_segments, cfg.z_ref_max, rng)

    tau = float(rng.uniform(cfg.tau_ref_min, cfg.tau_ref_max))
    alpha = np.exp(-cfg.dt / tau)

    filtered = np.zeros_like(ref)
    filtered[0, 0:3] = x0[0:3]
    filtered[0, 3:6] = 0.0

    for k in range(1, n_steps):
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