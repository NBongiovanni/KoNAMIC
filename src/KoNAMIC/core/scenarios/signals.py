import numpy as np


def rand_uniform_sym(rng: np.random.Generator, max_abs: float) -> float:
    return float(rng.uniform(-max_abs, max_abs))


def low_pass_filter_reference(
    *,
    ref: np.ndarray,
    x0_ref: np.ndarray,
    dt: float,
    tau_ref_min: float,
    tau_ref_max: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Apply first-order smoothing to a reference trajectory.
    """
    if tau_ref_min <= 0.0 or tau_ref_max <= 0.0:
        raise ValueError(
            f"Time constants must be positive, got "
            f"tau_ref_min={tau_ref_min}, tau_ref_max={tau_ref_max}"
        )

    if tau_ref_min > tau_ref_max:
        raise ValueError(
            f"tau_ref_min must be <= tau_ref_max, got "
            f"tau_ref_min={tau_ref_min}, tau_ref_max={tau_ref_max}"
        )

    ref = np.asarray(ref, dtype=float)
    x0_ref = np.asarray(x0_ref, dtype=float).reshape(-1)

    if ref.ndim != 2:
        raise ValueError(f"ref must be 2D, got shape {ref.shape}")

    if x0_ref.shape != (ref.shape[1],):
        raise ValueError(
            f"x0_ref must have shape ({ref.shape[1]},), got {x0_ref.shape}"
        )

    filtered = np.zeros_like(ref)

    tau = float(rng.uniform(tau_ref_min, tau_ref_max))
    alpha = np.exp(-dt / tau)

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


def get_reference_smoothing_config(references: dict) -> tuple[float, float]:
    smoothing = references["reference_smoothing"]
    return (
        float(smoothing["time_constant_min"]),
        float(smoothing["time_constant_max"]),
    )