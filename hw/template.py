"""Printable template sheet: one box per character, corner markers for
perspective correction, and binary page-id squares."""

from PIL import Image, ImageDraw

from . import charset as C
from .fonts import font_for_char, label_font, pil_font

GUIDE = (205, 205, 205)
GUIDE_LIGHT = (225, 225, 225)
BORDER = (170, 170, 170)
LABEL = (140, 140, 140)


def _dashed(draw, x0, x1, y, color, dash=8, gap=6):
    x = x0
    while x < x1:
        draw.line([(x, y), (min(x + dash, x1), y)], fill=color, width=1)
        x += dash + gap


def render_page(page_id, chars):
    img = Image.new("RGB", (C.PAGE_W, C.PAGE_H), "white")
    d = ImageDraw.Draw(img)

    for cx, cy in C.MARKER_CENTERS:
        h = C.MARKER / 2
        d.rectangle([cx - h, cy - h, cx + h - 1, cy + h - 1], fill="black")

    for b in range(C.ID_BITS):
        x = C.ID_X0 + b * (C.ID_SIZE + C.ID_GAP)
        box = [x, C.ID_Y, x + C.ID_SIZE - 1, C.ID_Y + C.ID_SIZE - 1]
        if (page_id >> b) & 1:
            d.rectangle(box, fill="black")
        else:
            d.rectangle(box, outline="black", width=3)

    d.text((150, 105), f"Handwriting template  -  page {page_id + 1}", fill="black", font=label_font(34))
    d.text((150, 155), "Write one character per box with a dark pen, sitting on the solid baseline.",
           fill=(90, 90, 90), font=label_font(22))
    d.text((150, 185), "Capitals reach the top line; small letters reach the dashed line. "
                       "Keep all four black corner squares in the photo.",
           fill=(90, 90, 90), font=label_font(22))

    for i, ch in enumerate(chars):
        x, y = C.cell_origin(i)
        d.rectangle([x, y, x + C.CELL_W, y + C.CELL_H], outline=BORDER, width=1)
        d.line([(x, y + C.LABEL_H), (x + C.CELL_W, y + C.LABEL_H)], fill=GUIDE_LIGHT, width=1)
        path = font_for_char(ch, prefer=("arial.ttf", "seguisym.ttf", "DejaVuSans.ttf", "cambria.ttc"))
        f = pil_font(path, 20) if path else label_font(20)
        d.text((x + 8, y + 3), ch, fill=LABEL, font=f)
        name = f"U+{ord(ch):04X}"
        d.text((x + C.CELL_W - 62, y + 7), name, fill=GUIDE, font=label_font(13))
        _dashed(d, x + 6, x + C.CELL_W - 6, y + C.CAP_Y, GUIDE_LIGHT)
        _dashed(d, x + 6, x + C.CELL_W - 6, y + C.XHEIGHT_Y, GUIDE_LIGHT)
        d.line([(x + 6, y + C.BASELINE_Y), (x + C.CELL_W - 6, y + C.BASELINE_Y)], fill=GUIDE, width=2)
        _dashed(d, x + 6, x + C.CELL_W - 6, y + C.DESC_Y, GUIDE_LIGHT)
    return img


def render_template(copies=1):
    return [render_page(i, chars) for i, chars in enumerate(C.pages_for(copies))]


def save_template_pdf(path, copies=1):
    pages = render_template(copies)
    pages[0].save(path, "PDF", save_all=True, append_images=pages[1:], resolution=C.DPI)
    return path
