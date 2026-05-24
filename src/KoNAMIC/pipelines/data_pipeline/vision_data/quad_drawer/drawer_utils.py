from typing import Optional, Tuple

import cv2
import numpy as np

def clip_segment_to_unit_square(
        p0: np.ndarray,
        p1: np.ndarray,
        xmin: float = -0.5, xmax: float = 0.5,
        ymin: float = -0.5, ymax: float = 0.5,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Clip un segment 2D contre le rectangle [xmin,xmax] x [ymin,ymax].
    Implémentation Cohen–Sutherland.

    Args:
        p0, p1: array-like shape (2,) représentant (h,v).
    Returns:
        (q0, q1) clippé (np.ndarray float32) ou None si segment hors du rectangle.
    """
    p0 = np.asarray(p0, dtype=np.float32).copy()
    p1 = np.asarray(p1, dtype=np.float32).copy()

    INSIDE, LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 4, 8

    def out_code(x: float, y: float) -> int:
        code = INSIDE
        if x < xmin: code |= LEFT
        elif x > xmax: code |= RIGHT
        if y < ymin: code |= BOTTOM
        elif y > ymax: code |= TOP
        return code

    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])

    c0 = out_code(x0, y0)
    c1 = out_code(x1, y1)

    # Boucle de clipping
    while True:
        # Trivial accept
        if (c0 | c1) == 0:
            return (np.array([x0, y0], dtype=np.float32),
                    np.array([x1, y1], dtype=np.float32))

        # Trivial reject
        if (c0 & c1) != 0:
            return None

        # Choisit un point hors-champ
        out = c0 if c0 != 0 else c1

        # Calcule intersection avec un bord
        if out & TOP:
            # y = ymax
            if y1 == y0:
                return None
            x = x0 + (x1 - x0) * (ymax - y0) / (y1 - y0)
            y = ymax
        elif out & BOTTOM:
            # y = ymin
            if y1 == y0:
                return None
            x = x0 + (x1 - x0) * (ymin - y0) / (y1 - y0)
            y = ymin
        elif out & RIGHT:
            # x = xmax
            if x1 == x0:
                return None
            y = y0 + (y1 - y0) * (xmax - x0) / (x1 - x0)
            x = xmax
        else:  # LEFT
            # x = xmin
            if x1 == x0:
                return None
            y = y0 + (y1 - y0) * (xmin - x0) / (x1 - x0)
            x = xmin

        # Remplace le point hors-champ par l'intersection et recalcule outcode
        if out == c0:
            x0, y0 = x, y
            c0 = out_code(x0, y0)
        else:
            x1, y1 = x, y
            c1 = out_code(x1, y1)


def extrude_segments_z(segments_xy: list[tuple[np.ndarray, np.ndarray]], h: float) -> list[
    tuple[np.ndarray, np.ndarray]]:
    """
    Prend une liste de segments (définis par 2 points 3D) supposés dans z=0,
    et renvoie une version extrudée en z (top/bottom + arêtes verticales).
    """
    z_top = +0.5 * h
    z_bot = -0.5 * h

    out = []
    for p0, p1 in segments_xy:
        p0 = p0.astype(np.float32).copy()
        p1 = p1.astype(np.float32).copy()

        # points top/bottom
        p0t, p1t = p0.copy(), p1.copy()
        p0b, p1b = p0.copy(), p1.copy()
        p0t[2], p1t[2] = z_top, z_top
        p0b[2], p1b[2] = z_bot, z_bot

        # segments "dessus" + "dessous"
        out.append((p0t, p1t))
        out.append((p0b, p1b))

        # arêtes verticales aux extrémités
        out.append((p0b, p0t))
        out.append((p1b, p1t))
    return out


def project_segments_to_view(segments3d_world: np.ndarray, view_spec) -> np.ndarray:
    """
    Projette des segments 3D monde vers une vue 2D définie par view_spec.

    Args:
        segments3d_world: (S, 2, 3)
        view_spec: objet avec axis_h, axis_v, flip_h, flip_v, scale, offset

    Returns:
        segments2d: (S, 2, 2) avec coordonnées (h, v) dans le repère 2D "image-monde"
                    (pas encore en pixels; ton pipeline actuel peut ensuite scaler/offset/clip).
    """
    assert segments3d_world.ndim == 3 and segments3d_world.shape[-2:] == (2, 3)

    axis_map = {"x": 0, "y": 1, "z": 2}
    axis_h = axis_map[view_spec.axis_h]
    axis_v = axis_map[view_spec.axis_v]

    seg2d = segments3d_world[..., [axis_h, axis_v]].astype(np.float32)  # (S,2,2)

    # flips éventuels
    if getattr(view_spec, "flip_h", False):
        seg2d[..., 0] *= -1.0
    if getattr(view_spec, "flip_v", False):
        seg2d[..., 1] *= -1.0

    # scale + offset (dans le même esprit que ton code actuel)
    s = float(getattr(view_spec, "scale", 1.0))
    off = np.array([view_spec.offset_h, view_spec.offset_v], dtype=np.float32)

    seg2d = seg2d * s + off  # (S,2,2)
    return seg2d


def postprocess(img: np.ndarray, save_size: int) -> np.ndarray:
    """
    Inversion / blur / conversion dtype / normalisation.
    """
    # img_inv = 255 - img # color inversion
    # img_inv = cv2.GaussianBlur(img_inv, (3, 3), sigmaX=0.6)

    img_resized = cv2.resize(
        img,
        (save_size, save_size),
        interpolation=cv2.INTER_LINEAR
    )
    return img_resized


def box_segments(
        center: np.ndarray,
        size_xyz: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Retourne les 12 arêtes d'un parallélépipède axis-aligned.
    center: (3,)
    size_xyz: (3,) = (sx, sy, sz) longueurs totales
    """
    c = np.asarray(center, dtype=np.float32)
    sx, sy, sz = (np.asarray(size_xyz, dtype=np.float32) * 0.5)

    # 8 sommets
    v000 = c + np.array([-sx, -sy, -sz], np.float32)
    v100 = c + np.array([+sx, -sy, -sz], np.float32)
    v110 = c + np.array([+sx, +sy, -sz], np.float32)
    v010 = c + np.array([-sx, +sy, -sz], np.float32)

    v001 = c + np.array([-sx, -sy, +sz], np.float32)
    v101 = c + np.array([+sx, -sy, +sz], np.float32)
    v111 = c + np.array([+sx, +sy, +sz], np.float32)
    v011 = c + np.array([-sx, +sy, +sz], np.float32)

    # 12 arêtes
    edges = [
        (v000, v100), (v100, v110), (v110, v010), (v010, v000),  # bas
        (v001, v101), (v101, v111), (v111, v011), (v011, v001),  # haut
        (v000, v001), (v100, v101), (v110, v111), (v010, v011),  # verticales
    ]
    return edges


def fill_closed_contours_bw(
        img_u8: np.ndarray,
        close_gaps: bool = True,
        close_ksize: int = 3,
        close_iter: int = 1
) -> np.ndarray:
    """
    img_u8: uint8, fond noir (0), traits blancs (255)
    retourne: fond noir, intérieur rempli blanc
    """
    assert img_u8.dtype == np.uint8 and img_u8.ndim == 2

    # Binarise (utile si jamais il y a eu interpolation avant)
    _, bw = cv2.threshold(img_u8, 127, 255, cv2.THRESH_BINARY)

    # Optionnel: boucher des micro-gaps
    if close_gaps:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=close_iter)

    h, w = bw.shape
    ff = bw.copy()

    # Flood-fill l'extérieur (fond = 0) avec 128
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(ff, mask, seedPoint=(0, 0), newVal=128)

    # Intérieur = pixels restés à 0 (fond non atteint car enfermé)
    interior = (ff == 0)
    ff[interior] = 255          # remplir
    ff[ff == 128] = 0           # remettre l'extérieur en noir

    # Réinjecter les traits
    out = cv2.bitwise_or(ff, bw)
    return out