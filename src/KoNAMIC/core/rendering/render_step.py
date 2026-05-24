from dataclasses import dataclass
from pathlib import Path
import numpy as np
import cv2
from torch import Tensor

from .render_utils import upscale, draw_grid, draw_scale_bar, draw_label

class PredShim:
    def __init__(self, y: Tensor):
        # y expected shape (1, 1, H, W) logits
        self.y = y


class GTShim:
    def __init__(self, y: Tensor):
        # y expected shape (2, 1, H, W); render_step_paper uses channel 1
        self.y = y

# Couleurs officielles Python en RGB puis converties en BGR pour OpenCV
BLUE  = (48, 105, 152)[::-1]  # RGB (48,105,152) → BGR (152,105,48)
YELLOW= (255, 212,  59)[::-1]  # RGB (255,212,59) → BGR (59,212,255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thk = 2


@dataclass(frozen=True)
class RenderConfig:
    show_gt: bool = False
    show_pred: bool = True
    save_overlay: bool = False
    save_gt: bool = True
    save_pred: bool = False
    upscale_factor: int = 4
    prefix: str = ""
    show_grid: bool = True
    grid_spacing: int = 16
    grid_thickness: int = 1

    # scale bar
    show_scale: bool = True
    meters_per_pixel: float = 2.0/128.0
    scale_bar_m: float = 0.25
    scale_origin: str = "bl"

    # top-left label
    show_label: bool = True
    label_margin: int = 12
    label_font_scale: float = 0.8
    label_thickness: int = 2


def _to_u8(im01: np.ndarray) -> np.ndarray:
    im01 = np.clip(im01, 0.0, 1.0)
    return (im01 * 255.0).astype(np.uint8)


def render_step_paper(
        pred: PredShim,
        g_t: GTShim,
        filename: Path,
        output_dir: Path,
        label_text: str,
        cfg: RenderConfig = RenderConfig(),
) -> None:
    # --- extract ---
    im_pred = pred.y.squeeze().detach().cpu().numpy()
    im_gt   = g_t.y[1:2].squeeze().detach().cpu().numpy()

    im_pred_u8 = _to_u8(im_pred)
    im_gt_u8   = _to_u8(im_gt)

    h, w = im_gt_u8.shape
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(filename).stem
    ext  = Path(filename).suffix or ".png"

    # --- optional: save raw vision ---
    if cfg.save_gt:
        cv2.imwrite(
            str(output_dir / f"{cfg.prefix}{stem}_gt{ext}"),
            upscale(im_gt_u8, cfg.upscale_factor)
        )
    if cfg.save_pred:
        cv2.imwrite(
            str(output_dir / f"{cfg.prefix}{stem}_pred{ext}"),
            upscale(im_pred_u8, cfg.upscale_factor)
        )

    # --- overlay ---
    if cfg.save_overlay:
        if not (cfg.show_gt or cfg.show_pred):
            # rien à afficher -> soit on n’écrit rien, soit on écrit un fond blanc (au choix)
            return

        overlay = np.full((h, w, 3), 255, np.uint8)

        if cfg.show_gt:
            # GT en magenta (R+B diminués)
            overlay[..., 2] = np.clip(overlay[..., 2] - im_gt_u8, 0, 255)  # R
            overlay[..., 0] = np.clip(overlay[..., 0] - im_gt_u8, 0, 255)  # B

        if cfg.show_pred:
            # Pred en cyan/vert (G+B diminués) — ici tu fais G et B, OK
            overlay[..., 1] = np.clip(overlay[..., 1] - im_pred_u8, 0, 255)  # G
            overlay[..., 0] = np.clip(overlay[..., 0] - im_pred_u8, 0, 255)  # B

        overlay_up = upscale(overlay, cfg.upscale_factor)

        if cfg.show_grid:
            overlay_up = draw_grid(
                overlay_up,
                spacing=cfg.grid_spacing * cfg.upscale_factor,
                thickness=cfg.grid_thickness
            )

        if cfg.show_scale:
            # IMPORTANT: meters_per_pixel doit être celui de l'image overlay_up
            # Si tu connais meters_per_pixel à la résolution native (128x128), alors:
            # meters_per_pixel_up = meters_per_pixel_native / upscale_factor
            meters_per_pixel_up = cfg.meters_per_pixel / cfg.upscale_factor
            overlay_up = draw_scale_bar(
                overlay_up,
                meters_per_pixel=meters_per_pixel_up,
                bar_length_m=cfg.scale_bar_m,
                origin=cfg.scale_origin
            )

        if cfg.show_label:
            overlay_up = draw_label(
                overlay_up,
                text=label_text,
                margin=cfg.label_margin,
                font_scale=cfg.label_font_scale,
                thickness=cfg.label_thickness,
            )

        cv2.imwrite(str(output_dir / f"{cfg.prefix}{stem}_overlay{ext}"), overlay_up)