from __future__ import annotations
from pathlib import Path
from typing import Union

import numpy as np
import torch

from .render_step import render_step_paper, PredShim, GTShim

def _ensure_3d(arr: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """
    Normalize input to torch tensor with shape (T, H, W), dtype float32 in [0,1] if possible.
    Accepts (T,H,W) or (T,1,H,W), numpy or torch, uint8 or float.
    """
    if isinstance(arr, np.ndarray):
        t = torch.from_numpy(arr)
    else:
        t = arr

    # Move to CPU for I/O and OpenCV interoperability
    t = t.detach().cpu()

    # Squeeze channel if present as singleton
    if t.ndim == 4 and t.shape[1] == 1:
        t = t.squeeze(1)
    if t.ndim != 3:
        raise ValueError(f"Expected (T,H,W) or (T,1,H,W); got shape={tuple(t.shape)}")

    # Convert to float in [0,1] if needed
    if t.dtype in (torch.uint8, torch.int16, torch.int32, torch.int64):
        t = t.to(torch.float32) / 255.0
    else:
        t = t.to(torch.float32)
        t = torch.clamp(t, 0.0, 1.0)
    return t


def render_trajectory_sequence(
        pred_imgs: Union[np.ndarray, torch.Tensor],
        gt_imgs:   Union[np.ndarray, torch.Tensor],
        label_text: str,
        path_results: Union[str, Path],
) -> None:
    """
    Sauvegarde une image par pas de temps en utilisant `render_step_paper`.
    """
    # Normalisation des entrées en (T,H,W) float32 [0,1]
    pred = _ensure_3d(pred_imgs)
    gt   = _ensure_3d(gt_imgs)
    end = pred.shape[0]

    # Dossier de sortie
    out_dir = Path(path_results)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Boucle temporelle
    for t in range(end):
        pred_t = pred[t]        # (H,W) float in [0,1] or logits
        gt_t   = gt[t]          # (H,W) float in [0,1]

        # Préparer les tenseurs pour le shim, shapes attendues par render_step_paper
        images = pred_t.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

        # Ground truth: construire un tenseur (2,1,H,W) dont le canal 1 contient gt_t
        # Le canal 0 n'est pas utilisé par render_step_paper.
        h, w = gt_t.shape[-2:]
        gt_y = torch.zeros((2, 1, h, w), dtype=torch.float32)
        gt_y[1, 0] = gt_t

        pred_obj = PredShim(y=images)
        gt_obj   = GTShim(y=gt_y)

        # Nom de fichier
        filename = Path(f"step_{t:04d}.png")

        # Appel à la fonction d'affichage/écriture
        render_step_paper(
            pred=pred_obj,
            g_t=gt_obj,
            filename=filename,
            output_dir=out_dir,
            label_text=label_text,
        )


