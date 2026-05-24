import numpy as np

# === Fonctions par type de trajectoire ===
def traj_constant(n_steps: int, value: float) -> np.ndarray:
    return np.full(n_steps, value)


def traj_ramp(
        num_steps: int,
        dt: float,
        start: float,
        end: float,
        t_start: float,
        t_end: float
) -> np.ndarray:
    """
    Generate a 1D ramp trajectory between two values.

    The trajectory starts at `start` and linearly interpolates to `end`
    between times `t_start` and `t_end`. Outside this interval, the
    trajectory is constant (`start` before `t_start`, `end` after `t_end`).

    Parameters
    ----------
    num_steps : int
        Number of discrete time steps in the trajectory.
    dt : float
        Time step size (seconds).
    start : float
        Initial value of the trajectory.
    end : float
        Final value of the trajectory after the ramp.
    t_start : float
        Time (seconds) at which the ramp begins.
    t_end : float
        Time (seconds) at which the ramp ends.

    Returns
    -------
    np.ndarray
        Array of shape (num_steps, ) containing the generated trajectory.
    """
    arr = np.full(num_steps, start, dtype=float)
    i0 = int(np.clip(np.floor(t_start / dt), 0, num_steps - 1))
    i1 = int(np.clip(np.ceil(t_end / dt), 0, num_steps - 1))
    if i1 > i0:
        arr[i0:i1 + 1] = np.linspace(start, end, i1 - i0 + 1)
        arr[i1 + 1:] = end
    else:
        arr[i0:] = end
    return arr


def traj_sine(
        n_steps: int,
        dt: float,
        amplitude: float,
        freq_hz: float,
        phase: float = 0.0
) -> np.ndarray:
    t = np.arange(n_steps) * dt
    return amplitude * np.sin(2 * np.pi * freq_hz * t + phase)

def traj_cosine(
        n_steps: int,
        dt: float,
        amplitude: float,
        freq_hz: float,
        phase: float = 0.0
) -> np.ndarray:
    t = np.arange(n_steps) * dt
    return amplitude * np.cos(2 * np.pi * freq_hz * t + phase)

def traj_sine_cosine(
        n_steps: int,
        dt: float,
        amplitude: float,
        freq_hz: float,
        phase: float = 0.0
) -> np.ndarray:
    t = np.arange(n_steps) * dt
    return amplitude * np.sin(2 * np.pi * freq_hz * t + phase) * np.cos(2 * np.pi * freq_hz * t + phase)