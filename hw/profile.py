"""A handwriting profile = the glyph images extracted from a user's template."""

import json
import os
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import charset as C
from .fonts import font_for_char, pil_font


@dataclass
class Glyph:
    alpha: np.ndarray   # float32 0..1, cropped to the ink
    top: float          # y of the image's top edge relative to the baseline (negative = above)
    fallback: bool = False

    @property
    def w(self):
        return self.alpha.shape[1]

    @property
    def h(self):
        return self.alpha.shape[0]


# Characters that can stand in for each other when the user skipped a box.
SUBSTITUTES = {
    "−": "-", "–": "-", "—": "-", "‐": "-", "∗": "*", "⋅": "·", "•": "·", "∣": "|",
    "‘": "'", "’": "'", "“": '"', "”": '"', "′": "'", "″": '"', "…": None,
    "×": "x", "ε": "∈", "ϵ": "ε", "φ": "ϕ", "ϕ": "φ", "⟨": "<", "⟩": ">",
    "∶": ":", "⁄": "/", "∖": "\\", "∼": "~", "〈": "<", "〉": ">",
    "O": "0", "0": "O", "l": "|", "∑": "Σ", "Σ": "∑", "∏": "Π", "∆": "Δ", "Δ": "∆",
}


class Profile:
    def __init__(self, directory):
        self.dir = directory
        with open(os.path.join(directory, "profile.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        self.name = self.meta.get("name", os.path.basename(directory))
        self.em = float(self.meta.get("em", C.EM))
        self.stroke = float(self.meta.get("stroke", 4.0))
        self._cache = {}
        self._fallback = {}

    def has(self, ch):
        return ch in self.meta["glyphs"]

    def variants(self, ch):
        """All glyph variants for `ch` (user-written if possible)."""
        if ch in self._cache:
            return self._cache[ch]
        out = []
        for v in self.meta["glyphs"].get(ch, []):
            a = cv2.imread(os.path.join(self.dir, "glyphs", v["file"]), cv2.IMREAD_GRAYSCALE)
            if a is not None:
                out.append(Glyph(a.astype(np.float32) / 255.0, float(v["top"])))
        if not out:
            sub = SUBSTITUTES.get(ch)
            if sub and sub != ch and self.has(sub):
                out = self.variants(sub)
            elif ch == "…" and self.has("."):
                out = [self._join("...")]
        if not out:
            g = self.fallback_glyph(ch)
            out = [g] if g is not None else []
        self._cache[ch] = out
        return out

    def _join(self, text):
        gs = [self.variants(c)[0] for c in text]
        gap = int(self.em * 0.12)
        top = min(g.top for g in gs)
        bot = max(g.top + g.h for g in gs)
        w = sum(g.w for g in gs) + gap * (len(gs) - 1)
        a = np.zeros((int(bot - top), w), np.float32)
        x = 0
        for g in gs:
            y = int(g.top - top)
            a[y:y + g.h, x:x + g.w] = np.maximum(a[y:y + g.h, x:x + g.w], g.alpha)
            x += g.w + gap
        return Glyph(a, top)

    def fallback_glyph(self, ch):
        """Render a missing character from a system font at the profile's scale."""
        if ch in self._fallback:
            return self._fallback[ch]
        path = font_for_char(ch)
        g = None
        if path:
            size = int(self.em / 0.7)
            font = pil_font(path, size)
            S = size * 3
            img = Image.new("L", (S, S), 0)
            ImageDraw.Draw(img).text((size, 2 * size), ch, fill=255, font=font, anchor="ls")
            bbox = img.getbbox()
            if bbox:
                a = np.asarray(img.crop(bbox), np.float32) / 255.0
                g =Glyph(a, float(bbox[1] - 2 * size), fallback=True)
        self._fallback[ch] = g
        return g

    def summary(self):
        return {
            "name": self.name,
            "glyphs": len(self.meta["glyphs"]),
            "variants": sum(len(v) for v in self.meta["glyphs"].values()),
            "missing": "".join(c for c in C.CHARSET if c not in self.meta["glyphs"]),
        }

    def contact_sheet(self, cols=16, cell=72):
        chars = [c for c in C.CHARSET]
        rows = (len(chars) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * cell, rows * cell), "white")
        d = ImageDraw.Draw(sheet)
        for i, ch in enumerate(chars):
            r, c = divmod(i, cols)
            x, y = c * cell, r * cell
            d.rectangle([x, y, x + cell - 1, y + cell - 1], outline=(230, 230, 230))
            if not self.has(ch):
                d.line([(x + 8, y + 8), (x + cell - 8, y + cell - 8)], fill=(240, 200, 200), width=2)
                continue
            g = self.variants(ch)[0]
            s = min((cell - 10) / max(g.w, 1), (cell - 10) / max(g.h, 1), 0.55)
            a = cv2.resize(g.alpha, (max(1, int(g.w * s)), max(1, int(g.h * s))), interpolation=cv2.INTER_AREA)
            ink = Image.fromarray((255 - a * 220).astype(np.uint8))
            base = y + int(cell * 0.68)
            gy = int(base + g.top * s)
            gy = min(max(gy, y + 2), y + cell - a.shape[0] - 2)
            sheet.paste(ink.convert("RGB"), (x + (cell - a.shape[1]) // 2, gy), Image.fromarray((a * 255).astype(np.uint8)))
            d.line([(x + 4, base), (x + cell - 4, base)], fill=(200, 220, 245))
        return sheet


def list_profiles(root):
    out = []
    if os.path.isdir(root):
        for n in sorted(os.listdir(root)):
            if os.path.exists(os.path.join(root, n, "profile.json")):
                out.append(n)
    return out
