
from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def _as2d(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square 2D array (n,n).")
    return A

def _qr_rank(M: np.ndarray, tol: float = 1e-10) -> int:
    if M.size == 0:
        return 0
    q, r = np.linalg.qr(M)
    diag = np.abs(np.diag(r))
    return int(np.sum(diag > tol))

def _add_matrix_to_linear_basis(mat: np.ndarray, basis_cols: List[np.ndarray], tol: float) -> bool:
    """
    Keep a linear basis of matrices in R^{n x n} using QR on column-stacked vecs.
    Returns True if 'mat' increased the basis rank.
    """
    v = mat.reshape(-1, 1)
    if not basis_cols:
        basis_cols.append(v)
        return True
    M_prev = np.hstack(basis_cols)
    r_prev = _qr_rank(M_prev, tol)
    r_new  = _qr_rank(np.hstack([M_prev, v]), tol)
    if r_new > r_prev:
        basis_cols.append(v)
        return True
    return False

def mat_power(A: np.ndarray, k: int) -> np.ndarray:
    """Efficient integer power for square matrix with k>=0."""
    if k < 0:
        raise ValueError("k must be >= 0")
    if k == 0:
        return np.eye(A.shape[0])
    if k == 1:
        return A
    # fast exponentiation
    result = np.eye(A.shape[0])
    base = A.copy()
    n = k
    while n > 0:
        if n & 1:
            result = result @ base
        base = base @ base
        n >>= 1
    return result

def plot_discrete_eigs(
        A,
        ax=None,
        title=None,
        show_unit_circle=True,
        equal_aspect=True,
        grid=True,
        savepath=None,
        dpi=300,
):
    """
    Plot eigenvalues of a discrete-time system matrix A in the complex plane.

    Parameters
    ----------
    A : array_like, shape (n, n)
        Square state-transition matrix.
    ax : matplotlib.axes.Axes or None
        If provided, plot into this axis; otherwise create a new figure.
    title : str or None
        Title for the plot.
    show_unit_circle : bool
        If True, draw the unit circle |z|=1.
    equal_aspect : bool
        If True, set equal aspect ratio.
    grid : bool
        If True, show grid.
    savepath : str or None
        If provided, save the figure to this path.
    dpi : int
        Resolution for saved figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax  : matplotlib.axes.Axes
    eigs : np.ndarray
        Eigenvalues of A.
    """
    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be a square 2D array, got shape {A.shape}.")

    eigs = np.linalg.eigvals(A)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots()
        created_fig = True
    else:
        fig = ax.figure

    # Eigenvalues
    ax.scatter(eigs.real, eigs.imag, marker='x')

    # Unit circle
    if show_unit_circle:
        theta = np.linspace(0, 2 * np.pi, 400)
        ax.plot(np.cos(theta), np.sin(theta), linestyle='--')

    # Axes
    ax.axhline(0.0, linewidth=1)
    ax.axvline(0.0, linewidth=1)

    ax.set_xlabel("Real")
    ax.set_ylabel("Imag")

    if title is None:
        title = "Eigenvalues of A (discrete-time)"
    ax.set_title(title)

    if grid:
        ax.grid(True)

    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")

    # Limits
    r = np.max(np.r_[1.0, np.abs(eigs)])
    pad = 0.1 * r
    lim = r + pad
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    # --- SAVE HERE ---
    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

    if created_fig:
        plt.tight_layout()

    return fig, ax, eigs

def save_discrete_eigs(A, filepath, model_name=None):
    """
    Compute and save eigenvalues of a discrete-time system matrix A.

    Parameters
    ----------
    A : array_like, shape (n, n)
        Square state-transition matrix.
    filepath : str or Path
        Path to output file (.npz recommended).
    model_name : str or None
        Optional model identifier saved as metadata.

    Saves
    -----
    - eigs : complex ndarray
    - model_name : str (if provided)
    """
    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square, got shape {A.shape}")

    eigs = np.linalg.eigvals(A)

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        filepath,
        eigs=eigs,
        model_name=model_name if model_name is not None else "",
        system_type="discrete",
    )