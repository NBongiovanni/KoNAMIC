from pathlib import Path
from typing import Sequence, Tuple, Optional, cast

from matplotlib.axes import Axes
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from PIL import Image

def centered_crop_box(
        w: int, h: int, crop_w: int, crop_h: int
) -> Tuple[int, int, int, int]:
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    return (left, top, left + crop_w, top + crop_h)


def load_steps(im_dir: Path, step_count: int) -> Sequence[Path]:
    paths = [im_dir / f"step_{k:04d}_overlay.png" for k in range(step_count)]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing images in {im_dir}: {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    return paths


def create_snapshots_figure(
        im_dir: Path,
        out_path: Path,
        label: str,
        step_count: int,
        dt: float = 0.05,
        dpi: int = 300,
        crop_w: Optional[int] = 300,
        crop_h: Optional[int] = 300,
        grid_step: int = 32,
        n_rows: int = 1,
        stride: int = 1,
) -> None:

    if stride < 1:
        raise ValueError("stride must be >= 1")

    full_seq = load_steps(im_dir, step_count)

    indices = list(range(0, step_count, stride))
    seq = [full_seq[i] for i in indices]

    n_display = len(seq)

    if n_rows not in (1, 2):
        raise ValueError("n_rows must be 1 or 2")

    n_cols = (
        n_display
        if n_rows == 1
        else (n_display + 1) // 2
    )

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 1.15, n_rows * 1.75),
        constrained_layout=False,
        squeeze=False,
    )

    axes = cast(NDArray[Axes], axes)

    for k_display, (k_real, img_path) in enumerate(zip(indices, seq)):
        row = 0 if n_rows == 1 else k_display // n_cols
        col = k_display if n_rows == 1 else k_display % n_cols

        img = Image.open(img_path)
        W, H = img.size
        if crop_w is not None and crop_h is not None:
            img = img.crop(centered_crop_box(W, H, crop_w=crop_w, crop_h=crop_h))

        ax = axes[row, col]
        ax.imshow(img, interpolation="nearest")

        ax.set_xticks(range(0, img.width, grid_step))
        ax.set_yticks(range(0, img.height, grid_step))
        ax.grid(color="gray", linestyle="-", linewidth=0.3, alpha=0.3)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        if k_real == 0 and label:
            ax.text(
                0.05, 0.95, label,
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=8,
                color="black",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2),
            )

        t = k_real * dt
        ax.text(
            0.5,
            -0.10,
            rf"$t={t:.2f}\,\mathrm{{s}}$",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
            clip_on=False,
        )

    # Supprimer axes vides si nombre impair
    total_slots = n_rows * n_cols
    for k in range(step_count, total_slots):
        row = k // n_cols
        col = k % n_cols
        axes[row, col].axis("off")

    ax0 = cast(Axes, axes[0, 0])
    pos = ax0.get_position()  # Bbox en coordonnées figure (0..1)
    # position de la légende dans la 1ère vignette (en coordonnées figure)
    x0 = 0.012
    y0 = 0.66
    x1 = x0 + 0.048

    # ligne principale
    fig.add_artist(mlines.Line2D(
        [x0, x1],
        [y0, y0],
        transform=fig.transFigure,
        color="0.2", linewidth=1.2)
    )

    # embouts verticaux
    cap = 0.02 * pos.height
    fig.add_artist(mlines.Line2D(
        [x0, x0],
        [y0 - cap, y0 + cap],
        transform=fig.transFigure,
        color="0.2",
        linewidth=1.2)
    )
    fig.add_artist(mlines.Line2D(
        [x1, x1],
        [y0 - cap, y0 + cap],
        transform=fig.transFigure,
        color="0.2",
        linewidth=1.2)
    )

    # texte
    fig.text(
        (x0 + x1) / 2,
        y0 + 0.02 * pos.height,
        r"$0.3\,\mathrm{m}$",
        ha="center",
        va="bottom",
        fontsize=7,
        color="0.2",
    )

    # marges compactes
    fig.subplots_adjust(left=0.01, right=0.995, top=0.98, bottom=0.22, wspace=0.02)
    fig.savefig(out_path, dpi=dpi, pad_inches=0.02)
    plt.close(fig)