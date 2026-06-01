import numpy as np

from ..sensor_generation_config import SensorGenerationConfig


def rand_uniform_sym(rng: np.random.Generator, max_abs: float) -> float:
    return float(rng.uniform(-max_abs, max_abs))


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
