import numpy as np


def compute_nrmse_fit(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Compute the normalized RMSE and Fit (%) using the trajectory-specific
    MATLAB-style definition:

        NRMSE = ||y_pred - y_true|| / ||y_true - mean(y_true)||
        Fit = (1 - NRMSE) * 100

    Args:
        pred: numpy array of shape (N, 1)
        true: numpy array of shape (N, 1)

    Returns:
        float: Fit percentage
    """
    pred = np.asarray(pred).reshape(-1)
    true = np.asarray(true).reshape(-1)

    rmse = np.sqrt(np.mean((pred - true) ** 2))

    traj_std = np.std(true)

    if traj_std < 1e-12:
        return np.nan

    nrmse = rmse / traj_std
    fit = (1 - nrmse) * 100

    return fit


def compute_rmse(pred, true):
    err = pred - true
    return np.sqrt(np.mean(err**2))


def compute_mae_per_state(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """
    Mean Absolute Error per dimension.

    Args:
        pred: (N, D)
        true: (N, D)

    Returns:
        mae: (D,)
    """
    pred = np.asarray(pred)
    true = np.asarray(true)
    return np.mean(np.abs(pred - true), axis=0)


def compute_mae_global(pred: np.ndarray, true: np.ndarray) -> float:
    pred = np.asarray(pred)
    true = np.asarray(true)
    return float(np.mean(np.abs(pred - true)))


def compute_mae_scalar(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Compute Mean Absolute Error (MAE) as a scalar value.

    This function is intended for single-dimension evaluation
    (e.g. one state component at a time).

    Args:
        pred: numpy array of shape (N,) or (N,1)
        true: numpy array of shape (N,) or (N,1)

    Returns:
        float: MAE value
    """
    pred = np.asarray(pred)
    true = np.asarray(true)

    if pred.shape != true.shape:
        raise ValueError("pred and true must have the same shape.")

    # Flatten to 1D for safety
    err = pred.reshape(-1) - true.reshape(-1)

    mae = np.mean(np.abs(err))
    return float(mae)


def format_scientific_latex(value: float, precision: int = 1) -> str:
    """
    Format a number as a LaTeX scientific notation string:
    1.23e-04 -> 1.23 × 10^{-4}
    """
    if value == 0:
        return "0"

    s = f"{value:.{precision}e}"   # ex: '1.02e-01'
    mantissa, exponent = s.split("e")
    exponent = int(exponent)       # remove leading zeros

    return rf"{mantissa} \times 10^{{{exponent}}}"