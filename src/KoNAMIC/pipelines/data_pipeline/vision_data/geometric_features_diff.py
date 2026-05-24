from __future__ import annotations

from torch import Tensor
import torch

def compute_centroids_diff(images: Tensor) -> Tensor:
    """
    vision: (B, N, 1, H, W)  valeurs >= 0
    returns:
        centroids: (B, N, 1, 2) with [x, y]
        valid:     (B, N, 1)    True si masse > 0
    """
    assert images.dim() == 5 and images.shape[2] == 1, "Expected (B,N,1,H,W)"
    B, N, _, H, W = images.shape
    device, dtype = images.device, images.dtype

    # Grilles de coordonnées (broadcastables sur (B,N,1,H,W))
    y_coords = torch.arange(H, device=device, dtype=dtype).view(1, 1, 1, H, 1)
    x_coords = torch.arange(W, device=device, dtype=dtype).view(1, 1, 1, 1, W)

    # Masse totale par élément (B,N,1,1,1)
    m00 = images.sum(dim=(-2, -1), keepdim=True)
    mask = (m00 > 0)

    safe_m00 = torch.where(mask, m00, torch.ones_like(m00))

    # Sommes pondérées (B,N,1,1,1)
    sum_y = (images * y_coords).sum(dim=(-2, -1), keepdim=True)
    sum_x = (images * x_coords).sum(dim=(-2, -1), keepdim=True)

    # Coordonnées (B,N,1) après suppression UNIQUEMENT des dims H,W
    y_c = (sum_y / safe_m00).squeeze(-1).squeeze(-1)  # (B,N,1)
    x_c = (sum_x / safe_m00).squeeze(-1).squeeze(-1)  # (B,N,1)

    # Valeur neutre quand invalid (ici 0)
    mask_squeezed = mask.squeeze(-1).squeeze(-1)    # (B,N,1)
    x_c = torch.where(mask_squeezed, x_c, torch.zeros_like(x_c))
    y_c = torch.where(mask_squeezed, y_c, torch.zeros_like(y_c))
    return torch.stack([x_c, y_c], dim=-1)


def compute_angles_diff(images: Tensor) -> Tensor:
    """
    vision: (B, N, 1, H, W)
    returns:
        angles: (B, N, 1) en radians
        mask:   (B, N, 1) True si masse > 0
    """
    B, N, _, H, W = images.shape
    device, dtype = images.device, images.dtype

    # Coordonnées
    y_coords = torch.arange(H, device=device, dtype=dtype).view(1, 1, 1, H, 1)
    x_coords = torch.arange(W, device=device, dtype=dtype).view(1, 1, 1, 1, W)

    # Masse totale
    m00 = images.sum(dim=(-2, -1), keepdim=True)  # (B,N,1,1,1)
    mask = (m00 > 0)
    safe_m00 = torch.where(mask, m00, torch.ones_like(m00))

    # Moments centraux d’ordre 2
    x_mean = (images * x_coords).sum(dim=(-2, -1), keepdim=True) / safe_m00
    y_mean = (images * y_coords).sum(dim=(-2, -1), keepdim=True) / safe_m00
    x = x_coords - x_mean
    y = y_coords - y_mean
    mu20 = (images * x**2).sum(dim=(-2, -1))
    mu02 = (images * y**2).sum(dim=(-2, -1))
    mu11 = (images * x * y).sum(dim=(-2, -1))

    # Orientation = 0.5 * atan2(2*mu11, mu20 - mu02)
    angles = (-1) * 0.5 * torch.atan2(2 * mu11, mu20 - mu02)  # (B,N,1)

    mask_squeezed = mask.squeeze(-1).squeeze(-1)  # (B,N,1)
    return torch.where(mask_squeezed, angles, torch.zeros_like(angles))