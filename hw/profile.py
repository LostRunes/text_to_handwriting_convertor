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


# How each character is anchored vertically. Users rarely write exactly on the
# template's guide lines, so placement is re-derived from the character class.
XHEIGHT_LETTERS = set("acemnorsuvwxz")
CAP_LETTERS = set("ABCDEFHIKLMNOPRSTUVWXZ")
DESCENDERS = set("fgjpqyβγζημξρφχψ")   # f: cursive f loops below the line
BASELINE = (set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdehiklmnorstuvwxz0123456789")
            | set(".!?:%&#$@∂∇∀∃∅∠∴∑∏αδεθλνπστωΓΔΘΛΣΦΨΩ"))
AXIS = set("-+=±∓×÷·~≠≈≡≤≥<>→←↔⇒⇔∈∉⊂⊆∪∩∝∞")
HIGH = set("'\"°′^*`")
TALL = set("()[]{}|/\√∫∮")
SMALL_PUNCT = set(".,·'`:;")

# Groups of characters that should share a height ("consistency" evens them out).
SIZE_CLASSES = {
    "x": set("acemnorsuvwxz"),
    "asc": set("bdhklt"),
    "desc": set("gjpqy"),
    "cap": set("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    "digit": set("0123456789"),
}
SLANTED = set().union(*SIZE_CLASSES.values()) | set("fi")


def stroke_width(a):
    b = (a > 0.5).astype(np.uint8)
    cs, _ = cv2.findContours(b, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    per = sum(cv2.arcLength(c, True) for c in cs)
    return 2 * float(b.sum()) / per if per else 0.0


def slant(a):
    m = cv2.moments(a)
    return m["mu11"] / m["mu02"] if m["mu02"] > 1e-6 else 0.0


def shear(a, k):
    """Shear horizontally by k (x' = x - k*(y - h/2)), growing the image to fit."""
    h, w = a.shape
    extra = int(np.ceil(abs(k) * h / 2)) + 1
    M = np.float32([[1, -k, k * h / 2 + extra], [0, 1, 0]])
    out = cv2.warpAffine(a, M, (w + 2 * extra, h), flags=cv2.INTER_LINEAR)
    cols = np.nonzero(out.max(axis=0) > 0.02)[0]
    return out[:, cols[0]:cols[-1] + 1] if len(cols) else out


class Profile:
    def __init__(self, directory, consistency=0.7):
        self.dir = directory
        self.consistency = float(consistency)
        with open(os.path.join(directory, "profile.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        self.name = self.meta.get("name", os.path.basename(directory))
        self.stroke = float(self.meta.get("stroke", 4.0))
        self._cache = {}
        self._fallback = {}
        self._measure()

    def _first(self, ch):
        v = self.meta["glyphs"].get(ch)
        return v[0] if v else None

    def _measure(self):
        """Find the writer's real cap height, x-height and baseline offset."""
        caps = [g["h"] for g in map(self._first, CAP_LETTERS) if g]
        xs = [g["h"] for g in map(self._first, XHEIGHT_LETTERS) if g]
        bottoms = [g["top"] + g["h"] for g in map(self._first, CAP_LETTERS | XHEIGHT_LETTERS) if g]
        self.em = float(np.median(caps)) if len(caps) >= 5 else float(self.meta.get("em", C.EM))
        self.xh = float(np.median(xs)) if len(xs) >= 5 else self.em * 0.6
        self.baseline_shift = float(np.median(bottoms)) if len(bottoms) >= 5 else 0.0

        self.class_h, self.class_of = {}, {}
        for name, chars in SIZE_CLASSES.items():
            hs = [g["h"] for g in map(self._first, chars) if g]
            if len(hs) >= 4:
                self.class_h[name] = float(np.median(hs))
                for c in chars:
                    self.class_of[c] = name
        slants = []
        for c in SLANTED:
            g = self._first(c)
            if g:
                a = cv2.imread(os.path.join(self.dir, "glyphs", g["file"]), cv2.IMREAD_GRAYSCALE)
                if a is not None:
                    slants.append(slant(a.astype(np.float32) / 255.0))
        self.slant = float(np.median(slants)) if len(slants) >= 8 else 0.0

    def set_consistency(self, c):
        c = float(min(max(c, 0.0), 1.0))
        if abs(c - self.consistency) > 1e-6:
            self.consistency = c
            self._cache.clear()

    def _even_stroke(self, a, strength):
        """Nudge a glyph's pen width toward the writer's typical width."""
        sw = stroke_width(a)
        if not sw or strength <= 0:
            return a
        r = sw / self.stroke
        k = np.ones((3, 3), np.uint8)
        if r < 0.78:
            a = a + (cv2.dilate(a, k) - a) * min(1.0, (0.95 - r) * 2.5) * strength
        elif r > 1.3 and min(a.shape) > 3 * self.stroke:
            a = a + (cv2.erode(a, k) - a) * min(1.0, (r - 1.15) * 1.5) * strength
        return a

    def _normalize(self, ch, a, top):
        """Even out size, slant, darkness and pen width of one glyph.
        `top` is relative to the template baseline; returns the new (a, top)."""
        c = self.consistency
        if c <= 0:
            return a, top
        ink = a[a > 0.15]
        if ink.size:
            p = float(np.percentile(ink, 95))
            if p < 0.92:
                a = np.clip(a * (1 + (1 / max(p, 0.3) - 1) * c), 0, 1)
        cls = self.class_of.get(ch)
        if cls:
            f = float(np.clip((self.class_h[cls] / a.shape[0]) ** c, 0.65, 1.5))
            if abs(np.log(f)) > 0.04:
                rel = top - self.baseline_shift
                a = cv2.resize(a, (max(1, round(a.shape[1] * f)), max(1, round(a.shape[0] * f))),
                               interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
                a = np.clip(a, 0, 1)
                top = rel * f + self.baseline_shift
        if ch in SLANTED and self.slant:
            d = slant(a) - self.slant
            if abs(d) > 0.12:
                a = shear(a, float(np.clip(d, -0.35, 0.35)) * c)
        return self._even_stroke(a, c), top

    def _place(self, ch, a, top):
        """Vertical position (top relative to baseline) for a glyph of `ch`."""
        h = a.shape[0]
        if ch in BASELINE:
            return -h
        if ch in DESCENDERS:
            return top - self.baseline_shift
        if ch in AXIS:
            return -self.xh * 0.5 - h / 2
        if ch in HIGH:
            return -self.em * 1.02
        if ch == ",":
            return -h * 0.3
        if ch == ";":
            return -h * 0.72
        if ch == "_":
            return self.em * 0.08
        if ch in TALL:
            return -self.em * 0.42 - h / 2
        return top - self.baseline_shift

    def _plausible(self, ch, a):
        """Reject glyphs that clearly don't match their box (e.g. '10' in the '.' box)."""
        if ch in ":;":
            return a.shape[1] <= max(self.em * 0.35, self.stroke * 4)
        if ch in SMALL_PUNCT:
            return max(a.shape) <= max(self.em * 0.42, self.stroke * 4)
        return True

    def _dot(self):
        d = max(self.stroke * 1.8, 4.0)
        n = int(np.ceil(d)) + 4
        yy, xx = np.mgrid[0:n, 0:n] - (n - 1) / 2
        a = np.clip(d / 2 - np.hypot(xx, yy) + 0.5, 0, 1).astype(np.float32)
        return a

    def has(self, ch):
        return any(True for _ in self._user_glyphs(ch))

    def _user_glyphs(self, ch):
        for v in self.meta["glyphs"].get(ch, []):
            a = cv2.imread(os.path.join(self.dir, "glyphs", v["file"]), cv2.IMREAD_GRAYSCALE)
            if a is None:
                continue
            a = a.astype(np.float32) / 255.0
            if self._plausible(ch, a):
                a, top = self._normalize(ch, a, float(v["top"]))
                yield Glyph(a.astype(np.float32), float(self._place(ch, a, top)))

    def variants(self, ch):
        """All glyph variants for `ch` (user-written if possible)."""
        if ch in self._cache:
            return self._cache[ch]
        out = list(self._user_glyphs(ch))
        if not out:
            sub = SUBSTITUTES.get(ch)
            if sub and sub != ch and self.has(sub):
                out = self.variants(sub)
            elif ch == "…" and self.has("."):
                out = [self._join("...")]
            elif ch in ".·":
                a = self._dot()
                out = [Glyph(a, self._place(ch, a, 0.0))]
        if not out:
            g = self.fallback_glyph(ch)
            out = [g] if g is not None else []
        self._cache[ch] = out
        return out

    def anchors(self, g):
        """Entry and exit points (glyph pixel coords) used to join cursive letters:
        the leftmost / rightmost ink inside the x-height band."""
        if hasattr(g, "_anchors"):
            return g._anchors
        mask = g.alpha > 0.4
        rows = np.arange(g.h) + g.top                    # baseline-relative y of each row
        band = (rows >= -self.xh * 1.15) & (rows <= self.xh * 0.12)
        m = mask & band[:, None]
        if not m.any():
            m = mask
        cols = np.nonzero(m.any(axis=0))[0]
        res = []
        for c0, sl in ((cols[0], slice(cols[0], cols[0] + 3)), (cols[-1], slice(max(cols[-1] - 2, 0), cols[-1] + 1))):
            ys = np.nonzero(m[:, sl].any(axis=1))[0]
            res.append((float(c0), float(ys.max() * 0.6 + ys.mean() * 0.4)))
        g._anchors = (res[0], res[1])
        return g._anchors

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
                a = self._even_stroke(a, 1.0)          # match the writer's pen width
                g = Glyph(a, float(bbox[1] - 2 * size), fallback=True)
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
