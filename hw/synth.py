"""Fake a *photographed, hand-filled* template using a handwriting-style
system font. Used to build the bundled demo profile and to test the
extraction pipeline end-to-end without a real scan."""

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import charset as C
from .fonts import HAND_FONTS, SYMBOL_FONTS, font_for_char, pil_font
from .template import render_page


def _cap_ratio(path):
    f = pil_font(path, 200)
    box = f.getbbox("H", anchor="ls")
    return (box[3] - box[1]) / 200


def fill_page(page_id, chars, rng, fonts=("Inkfree.ttf",) + HAND_FONTS + SYMBOL_FONTS):
    img = render_page(page_id, chars).convert("RGB")
    for i, ch in enumerate(chars):
        path = font_for_char(ch, prefer=fonts)
        if not path:
            continue
        size = int(C.EM * rng.uniform(0.92, 1.05) / _cap_ratio(path))
        f = pil_font(path, size)
        S = size * 3
        glyph = Image.new("L", (S, S), 0)
        ImageDraw.Draw(glyph).text((size, 2 * size), ch, fill=255, font=f, anchor="ls")
        glyph = glyph.rotate(rng.normal(0, 3), resample=Image.BICUBIC, center=(size * 1.5, 2 * size))
        bbox = glyph.getbbox()
        if not bbox:
            continue
        x, y = C.cell_origin(i)
        gx = int(x + C.CELL_W / 2 - (bbox[2] - bbox[0]) / 2 - bbox[0] + rng.normal(0, 4))
        gy = int(y + C.BASELINE_Y - 2 * size + rng.normal(0, 2))
        ink = Image.new("RGB", glyph.size, (30, 38, 105))
        img.paste(ink, (gx, gy), glyph)
    return img


def fake_photo(page, rng):
    """Perspective, uneven light, noise and JPEG, on a dark table."""
    arr = np.asarray(page)[:, :, ::-1].copy()
    h, w = arr.shape[:2]
    pad = int(0.08 * w)
    W, H = w + 2 * pad, h + 2 * pad
    j = lambda: rng.uniform(-0.03, 0.03) * w
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[pad + j(), pad + j()], [pad + w + j(), pad + j()],
                      [pad + w + j(), pad + h + j()], [pad + j(), pad + h + j()]])
    M = cv2.getPerspectiveTransform(src, dst)
    photo = cv2.warpPerspective(arr, M, (W, H), borderValue=(55, 50, 45))
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    light = 0.78 + 0.22 * (xx / W) * (1 - 0.4 * yy / H)
    photo = photo.astype(np.float32) * light[..., None]
    photo += rng.normal(0, 4, photo.shape)
    photo = cv2.GaussianBlur(np.clip(photo, 0, 255).astype(np.uint8), (3, 3), 0.8)
    ok, enc = cv2.imencode(".jpg", photo, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def synth_sources(seed=3, copies=1, fonts=None):
    rng = np.random.default_rng(seed)
    out = []
    for pid, chars in enumerate(C.pages_for(copies)):
        page = fill_page(pid, chars, rng, **({"fonts": fonts} if fonts else {}))
        out.append((fake_photo(page, rng), f"synthetic_page{pid + 1}.jpg"))
    return out
