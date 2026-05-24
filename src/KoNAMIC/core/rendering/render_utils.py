import cv2

# Couleurs officielles Python en RGB puis converties en BGR pour OpenCV
BLUE  = (48, 105, 152)[::-1]  # RGB (48,105,152) → BGR (152,105,48)
YELLOW= (255, 212,  59)[::-1]  # RGB (255,212,59) → BGR (59,212,255)
RED = (0, 0, 255)

def draw_grid(img, spacing=10, thickness=1):
    """Dessine un quadrillage régulier sur `img`."""
    color = BLUE
    h, w = img.shape[:2]
    for x in range(0, w, spacing):
        cv2.line(img, (x, 0), (x, h), color, thickness)
    for y in range(0, h, spacing):
        cv2.line(img, (0, y), (w, y), color, thickness)
    return img


def upscale(img, factor=4):
    """Upscale par interpolation nearest neighbor."""
    h, w = img.shape[:2]
    return cv2.resize(
        img,
        dsize=(w * factor, h * factor),
        interpolation=cv2.INTER_NEAREST
    )

def draw_scale_bar(
        img,
        meters_per_pixel: float,
        bar_length_m: float = 1.0,
        origin: str = "bl",  # bottom-left
        margin: int = 12,
        thickness: int = 3,
        font_scale: float = 0.8,
        font_thk: int = 2,
):
    """
    Dessine une barre d'échelle de longueur `bar_length_m` (en mètres).
    `meters_per_pixel` doit correspondre à l'image SUR LAQUELLE tu dessines
    (donc après upscale si tu dessines après upscale).
    """
    h, w = img.shape[:2]

    # longueur en pixels
    bar_px = int(round(bar_length_m / meters_per_pixel))
    bar_px = max(bar_px, 1)

    # position
    if origin == "bl":
        x1, y1 = margin, h - margin
    elif origin == "br":
        x1, y1 = w - margin - bar_px, h - margin
    elif origin == "tl":
        x1, y1 = margin, margin + thickness
    elif origin == "tr":
        x1, y1 = w - margin - bar_px, margin + thickness
    else:
        raise ValueError(f"origin invalide: {origin}")

    x2, y2 = x1 + bar_px, y1

    # barre (jaune)
    cv2.line(img, (x1, y1), (x2, y2), RED, thickness, cv2.LINE_AA)

    # petite hauteur des barres verticales
    cap_height = 8

    # extrémité gauche
    cv2.line(
        img,
        (x1, y1 - cap_height),
        (x1, y1 + cap_height),
        RED,
        thickness,
        cv2.LINE_AA,
    )

    # extrémité droite
    cv2.line(
        img,
        (x2, y2 - cap_height),
        (x2, y2 + cap_height),
        RED,
        thickness,
        cv2.LINE_AA,
    )

    # label
    label = f"{bar_length_m:g} m"
    (tw, th), _ = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        font_thk
    )
    tx = x1
    ty = y1 - 6  # un peu au-dessus
    # petit fond blanc pour lisibilité
    cv2.rectangle(img, (tx - 4, ty - th - 6), (tx + tw + 4, ty + 4), (255, 255, 255), -1)
    cv2.putText(
        img,
        label,
        (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        font_thk,
        cv2.LINE_AA
    )
    return img


def draw_label(
        img,
        text: str,
        margin: int = 12,
        font_scale: float = 0.8,
        thickness: int = 2,
):
    """
    Dessine un label en haut à gauche avec fond blanc.
    """
    if not text:
        return img

    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)

    x = margin
    y = margin + th

    # fond blanc
    cv2.rectangle(
        img,
        (x - 6, y - th - 6),
        (x + tw + 6, y + 6),
        (255, 255, 255),
        -1
    )

    # texte noir
    cv2.putText(
        img,
        text,
        (x, y),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA
    )

    return img