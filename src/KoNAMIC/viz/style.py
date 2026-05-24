from pathlib import Path
from typing import Any, Dict
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from typing import Sequence, Union

AxesLike = Union[Axes, np.ndarray, Sequence[Axes]]

COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

GT_COLOR="black"

DEFAULT_RC_PARAMS: Dict[str, Any] = {
    "text.usetex": True,
    "font.family": "serif",
    "axes.labelsize": 11,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
    "lines.linewidth": 1.1,
}


def save_figure(fig: plt.Figure, save_dir: Path, filename: str) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / filename, dpi=120, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


def format_sci_latex(x: float, precision: int = 1) -> str:
    a = f"{x:.{precision}e}".split("e")
    coeff = float(a[0])
    exp = int(a[1])
    return rf"{coeff:.{precision}f}\times 10^{{{exp}}}"


def _as_axes_list(axes: AxesLike) -> list[Axes]:
    if isinstance(axes, np.ndarray):
        return [ax for ax in axes.ravel()]
    if isinstance(axes, (list, tuple)):
        return list(axes)
    return [axes]