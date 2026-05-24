from __future__ import annotations

import numpy as np
import cv2
import torch
from torch import Tensor


def compute_centroids_robust(
        images: torch.Tensor,
        *,
        use_otsu: bool = True,
        threshold: int = 32,          # used if use_otsu=False, in [0..255]
        dilate_ksize: int = 7,        # 3 or 5 typically; 0 disables dilation
        dilate_iter: int = 2,
        openclose_ksize: int = 3,     # 0 disables open/close
        open_iter: int = 1,
        close_iter: int = 1,
        min_area: int = 15,
        use_bbox_center: bool = True, # True: very stable; False: center of mass
        enable_tracking: bool = True,
        max_jump: float | None = 25.0 # pixels; None disables gating
) -> torch.Tensor:
    """
    Robust centroid extraction for ground-truth vision (non-differentiable).
    """
    assert images.dim() == 5 and images.shape[2] == 1, f"Expected (B,N,1,H,W), got {images.shape}"
    B, N, _, H, W = images.shape

    imgs = images.detach().cpu().numpy()  # (B,N,1,H,W)

    # Convert to uint8 [0,255]
    if imgs.max() <= 1.5:
        imgs_u8 = (imgs * 255.0).astype(np.uint8)
    else:
        imgs_u8 = np.clip(imgs, 0, 255).astype(np.uint8)

    centroids = np.zeros((B, N, 1, 2), dtype=np.float32)

    # Kernels
    k_oc = None
    if openclose_ksize and openclose_ksize > 0:
        k_oc = np.ones((openclose_ksize, openclose_ksize), np.uint8)

    k_dil = None
    if dilate_ksize and dilate_ksize > 0:
        k_dil = np.ones((dilate_ksize, dilate_ksize), np.uint8)

    for b in range(B):
        cx_prev, cy_prev = None, None

        for n in range(N):
            im = imgs_u8[b, n, 0]  # (H,W)

            # 1) Binarize
            if use_otsu:
                _, bw = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                _, bw = cv2.threshold(im, int(threshold), 255, cv2.THRESH_BINARY)

            # 2) Cleanup open/close (optional)
            if k_oc is not None:
                if open_iter and open_iter > 0:
                    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k_oc, iterations=int(open_iter))
                if close_iter and close_iter > 0:
                    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k_oc, iterations=int(close_iter))

            # 3) Dilate (KEY for thin objects / aliasing)
            if k_dil is not None and dilate_iter and dilate_iter > 0:
                bw = cv2.dilate(bw, k_dil, iterations=int(dilate_iter))

            # 4) Connected components
            num_labels, labels, stats, cents = cv2.connectedComponentsWithStats(bw, connectivity=8)

            # No foreground
            if num_labels <= 1:
                cx, cy = W / 2.0, H / 2.0
                centroids[b, n, 0] = (cx, cy)
                cx_prev, cy_prev = cx, cy
                continue

            # Build candidates (exclude background label 0)
            candidates = []
            for lab in range(1, num_labels):
                area = int(stats[lab, cv2.CC_STAT_AREA])
                if area < min_area:
                    continue

                x = int(stats[lab, cv2.CC_STAT_LEFT])
                y = int(stats[lab, cv2.CC_STAT_TOP])
                w = int(stats[lab, cv2.CC_STAT_WIDTH])
                h = int(stats[lab, cv2.CC_STAT_HEIGHT])

                if use_bbox_center:
                    cx = x + w / 2.0
                    cy = y + h / 2.0
                else:
                    cx, cy = float(cents[lab, 0]), float(cents[lab, 1])

                candidates.append((lab, area, cx, cy, x, y, w, h))

            if not candidates:
                cx, cy = W / 2.0, H / 2.0
            else:
                if (not enable_tracking) or (cx_prev is None):
                    # Init: pick largest area
                    cand = max(candidates, key=lambda t: t[1])
                    cx, cy = cand[2], cand[3]
                else:
                    # Track: pick closest centroid to previous
                    cand = min(
                        candidates,
                        key=lambda t: (t[2] - cx_prev) ** 2 + (t[3] - cy_prev) ** 2
                    )
                    cx, cy = cand[2], cand[3]

                    # Optional gating: if jump too large, fallback to largest component
                    if max_jump is not None:
                        d = float(np.hypot(cx - cx_prev, cy - cy_prev))
                        if d > float(max_jump):
                            cand = max(candidates, key=lambda t: t[1])
                            cx, cy = cand[2], cand[3]

            centroids[b, n, 0, 0] = cx
            centroids[b, n, 0, 1] = cy
            cx_prev, cy_prev = cx, cy

    return torch.from_numpy(centroids).to(images.device)


def compute_angles_robust(
    images: Tensor,
    *,
    use_otsu: bool = True,
    threshold: int = 32,
    openclose_ksize: int = 3,
    open_iter: int = 1,
    close_iter: int = 1,
    dilate_ksize: int = 7,
    dilate_iter: int = 2,
    min_area: int = 15,
    enable_tracking: bool = True,
    max_jump: float | None = 25.0,          # pixels gating (comme centroids_gt)
    unwrap_with_prev: bool = True,          # évite les flips +/- pi
    method: str = "pca",                    # "pca" ou "fitline"
) -> Tensor:
    """
    Robust angle extraction (non-differentiable).
    vision: (B, N, 1, H, W) in [0,1] or [0,255]
    returns: angles (B, N, 1) radians, range approx [-pi/2, pi/2] (mod pi)
    """
    assert images.dim() == 5 and images.shape[2] == 1, f"Expected (B,N,1,H,W), got {images.shape}"
    B, N, _, H, W = images.shape
    device = images.device

    imgs = images.detach().cpu().numpy()
    if imgs.max() <= 1.5:
        imgs_u8 = (imgs * 255.0).astype(np.uint8)
    else:
        imgs_u8 = np.clip(imgs, 0, 255).astype(np.uint8)

    # kernels
    k_oc = np.ones((openclose_ksize, openclose_ksize), np.uint8) if openclose_ksize and openclose_ksize > 0 else None
    k_dil = np.ones((dilate_ksize, dilate_ksize), np.uint8) if dilate_ksize and dilate_ksize > 0 else None

    angles = np.zeros((B, N, 1), dtype=np.float32)

    def wrap_pi(a: float) -> float:
        # wrap to [-pi, pi]
        return float((a + np.pi) % (2 * np.pi) - np.pi)

    def best_angle_mod_pi(theta: float, theta_prev: float) -> float:
        """
        Angles are equivalent modulo pi for an unoriented principal axis.
        Choose theta or theta+pi to minimize diff to prev (with wrapping).
        """
        t0 = wrap_pi(theta)
        t1 = wrap_pi(theta + np.pi)
        d0 = abs(wrap_pi(t0 - theta_prev))
        d1 = abs(wrap_pi(t1 - theta_prev))
        return t0 if d0 <= d1 else t1

    for b in range(B):
        cx_prev, cy_prev = None, None
        ang_prev = None

        for n in range(N):
            im = imgs_u8[b, n, 0]

            # 1) binarize
            if use_otsu:
                _, bw = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                _, bw = cv2.threshold(im, int(threshold), 255, cv2.THRESH_BINARY)

            # 2) open/close
            if k_oc is not None:
                if open_iter and open_iter > 0:
                    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k_oc, iterations=int(open_iter))
                if close_iter and close_iter > 0:
                    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k_oc, iterations=int(close_iter))

            # 3) dilate (utile si contours aliasés / segments fins)
            if k_dil is not None and dilate_iter and dilate_iter > 0:
                bw = cv2.dilate(bw, k_dil, iterations=int(dilate_iter))

            # 4) connected components -> pick target component
            num_labels, labels, stats, cents = cv2.connectedComponentsWithStats(bw, connectivity=8)

            if num_labels <= 1:
                angles[b, n, 0] = 0.0
                cx_prev, cy_prev = W / 2.0, H / 2.0
                ang_prev = 0.0
                continue

            candidates = []
            for lab in range(1, num_labels):
                area = int(stats[lab, cv2.CC_STAT_AREA])
                if area < min_area:
                    continue
                cx, cy = float(cents[lab, 0]), float(cents[lab, 1])
                x = int(stats[lab, cv2.CC_STAT_LEFT])
                y = int(stats[lab, cv2.CC_STAT_TOP])
                w = int(stats[lab, cv2.CC_STAT_WIDTH])
                h = int(stats[lab, cv2.CC_STAT_HEIGHT])
                candidates.append((lab, area, cx, cy, x, y, w, h))

            if not candidates:
                angles[b, n, 0] = float(ang_prev) if ang_prev is not None else 0.0
                continue

            if (not enable_tracking) or (cx_prev is None):
                lab, area, cx, cy, x, y, w, h = max(candidates, key=lambda t: t[1])
            else:
                lab, area, cx, cy, x, y, w, h = min(
                    candidates, key=lambda t: (t[2] - cx_prev) ** 2 + (t[3] - cy_prev) ** 2
                )
                if max_jump is not None:
                    d = float(np.hypot(cx - cx_prev, cy - cy_prev))
                    if d > float(max_jump):
                        lab, area, cx, cy, x, y, w, h = max(candidates, key=lambda t: t[1])

            comp = (labels == lab).astype(np.uint8)

            # 5) angle estimation
            theta = None

            if method.lower() == "fitline":
                # contour-based line fit (souvent très stable)
                contours, _ = cv2.findContours(
                    comp,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_NONE
                )
                if contours:
                    cnt = max(contours, key=cv2.contourArea)
                    if len(cnt) >= 2:
                        vx, vy, x0, y0 = cv2.fitLine(
                            cnt,
                            cv2.DIST_L2,
                            0,
                            0.01,
                            0.01
                        )
                        theta = float(np.arctan2(vy, vx))  # angle de l'axe principal
            else:
                # PCA sur les pixels du composant (très robuste pour forme remplie)
                ys, xs = np.nonzero(comp)
                if len(xs) >= 2:
                    X = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
                    X -= X.mean(axis=0, keepdims=True)
                    C = (X.T @ X) / max(1, (X.shape[0] - 1))
                    eigvals, eigvecs = np.linalg.eigh(C)  # ascending
                    v = eigvecs[:, np.argmax(eigvals)]    # principal axis
                    theta = float(np.arctan2(v[1], v[0]))

            if theta is None or not np.isfinite(theta):
                theta = float(ang_prev) if ang_prev is not None else 0.0

            # 6) unwrap modulo pi (évite flips)
            if unwrap_with_prev and (ang_prev is not None):
                theta = best_angle_mod_pi(theta, float(ang_prev))
            else:
                theta = wrap_pi(theta)

            angles[b, n, 0] = theta
            cx_prev, cy_prev = cx, cy
            ang_prev = theta

    return torch.from_numpy(angles).to(device)