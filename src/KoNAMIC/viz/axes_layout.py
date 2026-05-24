from __future__ import annotations

from typing import List, Tuple
import numpy as np

def get_shared_ylim_groups_state(
        drone_dim: int,
        only_position=False
) -> List[Tuple[int, ...]]:
    """
    Groupes 1-based des composantes de x qui doivent partager la même échelle (ylim).
    """
    if drone_dim == 2:
        # x = [1..6] : (1-2) et (5-6)
        if not only_position:
            return [(1, 2), (4, 5)]
        else:
            return [(1, 2)]
    if drone_dim == 3:
        # x = [1..12] : (1-2-3), (4-5-6), (7-8-9), (10-11-12)
        if not only_position:
            return [(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)]
        else:
            return [(1, 2, 3), (4, 5, 6)]
    raise ValueError(f"Invalid drone dimension {drone_dim}")


def apply_shared_ylims(
        axes: np.ndarray,  # shape (n_rows, n_cols)
        x_gt: np.ndarray,  # shape (T, x_dim)
        x_pred: np.ndarray | None,  # shape (T, x_dim) ou None
        groups_1based: list[tuple[int, ...]],
        pad_frac: float = 0.05,
        min_pad_abs: float = 1e-9,
) -> None:
    """
    Fixe des ylim identiques pour chaque groupe, en prenant le min/max sur gt et pred.
    """
    n_rows, n_cols = axes.shape
    x_dim = x_gt.shape[1]

    def ax_of_dim0(d0: int):
        # même mapping que dans ton code
        row = d0 % n_rows
        col = d0 // n_rows
        return axes[row, col]

    for grp in groups_1based:
        dims0 = [g - 1 for g in grp]  # -> 0-based
        assert all(0 <= d < x_dim for d in dims0)

        series = [x_gt[:, dims0]]
        if x_pred is not None:
            series.append(x_pred[:, dims0])

        y = np.concatenate(series, axis=0)  # (T_total, len(grp))
        y_min = float(np.nanmin(y))
        y_max = float(np.nanmax(y))

        # padding (évite un axe plat)
        span = max(y_max - y_min, min_pad_abs)
        pad = pad_frac * span
        lo, hi = y_min - pad, y_max + pad

        # for d0 in dims0:
        #     ax_of_dim0(d0).set_ylim(lo, hi)