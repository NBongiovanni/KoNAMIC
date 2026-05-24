from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
import numpy as np


def plot_and_save_eigs_compare(
        eigs_a,
        eigs_b,
        label_a="Linear",
        label_b="Bilinear",
        title="Eigenvalues comparison",
        savepath=None,
        dpi=300,
        show_unit_circle=True,
        grid=True,
        equal_aspect=True,
):
    """
    Plot and (optionally) save a comparison of two eigenvalue sets in the complex plane.

    Parameters
    ----------
    eigs_a, eigs_b : array_like of complex, shape (n,)
        Eigenvalues to compare.
    label_a, label_b : str
        Legend labels.
    title : str
        Figure title.
    savepath : str or Path or None
        If provided, saves the figure to this path (e.g., .pdf or .png).
    dpi : int
        Save resolution (for raster formats like PNG).
    show_unit_circle : bool
        Draw unit circle if True.
    grid : bool
        Show grid if True.
    equal_aspect : bool
        Enforce equal axis aspect ratio if True.

    Returns
    -------
    fig, ax
    """
    eigs_a = np.asarray(eigs_a).astype(np.complex128).ravel()
    eigs_b = np.asarray(eigs_b).astype(np.complex128).ravel()

    fig, ax = plt.subplots()

    # Linear: croix (couleur explicitée)
    ax.scatter(
        eigs_a.real,
        eigs_a.imag,
        marker="x",
        s=45,
        linewidths=1.5,
        c="C0",
        label=label_a,
        zorder=3,
    )

    # Bilinear: cercles vides (CONTOUR forcé)
    ax.scatter(
        eigs_b.real,
        eigs_b.imag,
        marker="o",
        s=55,
        facecolors="none",
        edgecolors="C1",
        linewidths=1.5,
        label=label_b,
        zorder=4,
    )

    # Unit circle
    if show_unit_circle:
        theta = np.linspace(0, 2 * np.pi, 400)
        ax.plot(np.cos(theta), np.sin(theta), linestyle="--", linewidth=1)

    # Axes through origin
    ax.axhline(0.0, linewidth=1)
    ax.axvline(0.0, linewidth=1)

    ax.set_xlabel("Real")
    ax.set_ylabel("Imag")
    ax.set_title(title)
    ax.legend()

    if grid:
        ax.grid(True)

    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")

    # Common limits based on both sets + unit circle
    r = np.max(np.r_[1.0, np.abs(eigs_a), np.abs(eigs_b)])
    pad = 0.1 * r
    lim = r + pad
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    fig.tight_layout()

    # Save (if requested)
    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

    return fig, ax

