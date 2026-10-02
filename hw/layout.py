"""TeX-like box layout drawn with the user's glyphs.

Every box has width `w`, ascent `a` (above the baseline) and descent `d`
(below), and draws itself with draw(canvas, x, baseline_y). Randomness
(jitter, wobble) is decided when the box is built, so measuring and drawing
agree.
"""

import math

import cv2
import numpy as np

from .mathparse import REL_CHARS


# ---------------------------------------------------------------------------
# canvas + context
# ---------------------------------------------------------------------------
class Canvas:
    """Ink density (0..1) for one page."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.a = np.zeros((h, w), np.float32)

    def paste(self, alpha, x, y, intensity=1.0):
        x, y = int(round(x)), int(round(y))
        h, w = alpha.shape
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, self.w), min(y + h, self.h)
        if x1 <= x0 or y1 <= y0:
            return
        src = alpha[y0 - y:y1 - y, x0 - x:x1 - x]
        dst = self.a[y0:y1, x0:x1]
        np.maximum(dst, src * intensity, out=dst)

    def stroke(self, pts, width, intensity=0.95):
        pts = np.asarray(pts, np.float64)
        pad = int(width) + 3
        x0, y0 = np.floor(pts.min(axis=0)).astype(int) - pad
        x1, y1 = np.ceil(pts.max(axis=0)).astype(int) + pad
        patch = np.zeros((y1 - y0, x1 - x0), np.uint8)
        local = np.round((pts - [x0, y0]) * 16).astype(np.int32)
        cv2.polylines(patch, [local], False, 255, max(1, int(round(width))), cv2.LINE_AA, shift=4)
        self.paste(patch.astype(np.float32) / 255.0, x0, y0, intensity)


class Ctx:
    """Shared rendering state: profile, scale, randomness, pen settings."""

    def __init__(self, profile, em, rng, messiness=1.0):
        self.profile = profile
        self.em = em                          # cap height on the page, px, at size 1
        self.k = em / profile.em              # page px per template px
        self.rng = rng
        self.m = messiness
        self.stroke = max(1.2, profile.stroke * self.k)
        self._last = {}
        self._resized = {}
        minus = profile.variants("-")
        if minus and not minus[0].fallback:
            g = minus[0]
            self.axis_ratio = min(max(-(g.top + g.h / 2) / profile.em, 0.2), 0.45)
        else:
            self.axis_ratio = 0.32

    def axis(self, size):
        return self.axis_ratio * self.em * size

    def pick(self, ch):
        vs = self.profile.variants(ch)
        if not vs:
            return None
        if len(vs) == 1:
            return vs[0]
        i = int(self.rng.integers(len(vs)))
        if i == self._last.get(ch):
            i = (i + 1) % len(vs)
        self._last[ch] = i
        return vs[i]

    def resized(self, g, sx, sy):
        key = (id(g), round(sx, 3), round(sy, 3))
        r = self._resized.get(key)
        if r is None:
            w, h = max(1, int(round(g.w * sx))), max(1, int(round(g.h * sy)))
            interp = cv2.INTER_AREA if sx * sy < 1 else cv2.INTER_CUBIC
            r = np.clip(cv2.resize(g.alpha, (w, h), interpolation=interp), 0, 1)
            if len(self._resized) > 4000:
                self._resized.clear()
            self._resized[key] = r
        return r

    def wobble_line(self, pts, amp=None, step=6.0):
        """Subdivide a polyline and add smooth, hand-like deviation."""
        amp = self.stroke * 0.35 * self.m if amp is None else amp
        out = []
        pts = [np.asarray(p, np.float64) for p in pts]
        for p, q in zip(pts[:-1], pts[1:]):
            seg = q - p
            L = float(np.hypot(*seg)) or 1.0
            n = max(2, int(L / step))
            normal = np.array([-seg[1], seg[0]]) / L
            ph1, ph2 = self.rng.uniform(0, 2 * np.pi, 2)
            f1, f2 = self.rng.uniform(0.6, 1.4), self.rng.uniform(1.8, 3.2)
            bow = self.rng.normal(0, amp * 0.8)
            for t in np.linspace(0, 1, n, endpoint=False):
                off = amp * (0.6 * math.sin(f1 * np.pi * t + ph1) + 0.4 * math.sin(f2 * np.pi * t + ph2))
                off = off * min(1.0, L / 40) + bow * math.sin(np.pi * t)
                out.append(p + seg * t + normal * off)
        out.append(pts[-1])
        return out


# ---------------------------------------------------------------------------
# boxes
# ---------------------------------------------------------------------------
class Box:
    w = a = d = 0.0

    def draw(self, cv, x, y):
        pass

    @property
    def height(self):
        return self.a + self.d


class Space(Box):
    def __init__(self, w):
        self.w = w


class Bullet(Box):
    """A hand-dotted list bullet."""

    def __init__(self, ctx, size):
        self.ctx = ctx
        self.r = ctx.em * size * 0.09
        self.w = ctx.em * size * 0.5
        self.a = ctx.em * size * 0.5
        self.lift = ctx.axis(size)

    def draw(self, cv, x, y):
        cx, cy, r = x + self.w / 2, y - self.lift, self.r
        pts = [(cx + r * 0.6 * math.cos(t), cy + r * 0.6 * math.sin(t)) for t in np.linspace(0, 2 * np.pi, 9)]
        cv.stroke(pts, max(self.ctx.stroke, r * 1.4))


class Phantom(Box):
    def __init__(self, inner):
        self.w, self.a, self.d = inner.w, inner.a, inner.d


class GlyphBox(Box):
    def __init__(self, ctx, ch, size=1.0, bold=False, fit=None, center_on=None, scale=1.0,
                 tight=False):
        """fit=(top, bottom): stretch the glyph vertically to that span (delimiters).
        center_on: y (relative to baseline) the glyph's middle is moved to."""
        self.ctx, self.ch, self.bold = ctx, ch, bold
        g = ctx.pick(ch)
        self.g = g
        rng, m = ctx.rng, ctx.m
        em = ctx.em * size
        if g is None:
            self.w = em * 0.5
            return
        s = ctx.k * size * scale
        sx = sy = s
        if fit is not None:
            top, bot = fit
            sy = max((bot - top) / g.h, s * 0.5)
            sx = s * (1 + 0.3 * (sy / s - 1)) if sy > s else s
        self.sx, self.sy = sx, sy
        self.jit = 1 + rng.normal(0, 0.03 * m)
        self.rot = rng.normal(0, 1.4 * m) if fit is None else rng.normal(0, 0.6 * m)
        self.dy = rng.normal(0, 0.022 * m) * em
        self.intensity = float(np.clip(rng.normal(0.93, 0.04), 0.8, 1.0))
        self.sb = em * (0.03 if tight else 0.065) * (1 + rng.uniform(-0.3, 0.3) * m)
        self.ink_w = g.w * sx
        self.top = g.top * sy
        if fit is not None:
            self.top = fit[0]
            self.dy *= 0.3
        if center_on is not None:
            self.top = center_on - g.h * sy / 2
        self.w = self.ink_w + 2 * self.sb
        self.a = max(0.0, -self.top)
        self.d = max(0.0, self.top + g.h * sy)

    def draw(self, cv, x, y):
        if self.g is None:
            return
        img = self.ctx.resized(self.g, self.sx * self.jit, self.sy * self.jit)
        h, w = img.shape
        cx = x + self.sb + self.ink_w / 2
        cy = y + self.top + self.g.h * self.sy / 2 + self.dy
        if abs(self.rot) > 0.15:
            M = cv2.getRotationMatrix2D((w / 2, h / 2), self.rot, 1.0)
            c, s_ = abs(M[0, 0]), abs(M[0, 1])
            W, H = int(w * c + h * s_) + 2, int(w * s_ + h * c) + 2
            M[0, 2] += W / 2 - w / 2
            M[1, 2] += H / 2 - h / 2
            img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR)
            h, w = H, W
        cv.paste(img, cx - w / 2, cy - h / 2, self.intensity)
        if self.bold:
            off = max(1.0, self.ctx.stroke * 0.55)
            cv.paste(img, cx - w / 2 + off, cy - h / 2 + off * 0.3, self.intensity)


class HBox(Box):
    def __init__(self, items):
        self.items = [b for b in items if b is not None]
        self.w = sum(b.w for b in self.items)
        self.a = max((b.a for b in self.items), default=0.0)
        self.d = max((b.d for b in self.items), default=0.0)

    def draw(self, cv, x, y):
        for b in self.items:
            b.draw(cv, x, y)
            x += b.w


class Raised(Box):
    """Draw `inner` shifted up by `dy` (negative = down)."""

    def __init__(self, inner, dy):
        self.inner, self.dy = inner, dy
        self.w = inner.w
        self.a = max(0.0, inner.a + dy)
        self.d = max(0.0, inner.d - dy)

    def draw(self, cv, x, y):
        self.inner.draw(cv, x, y - self.dy)


class Frac(Box):
    def __init__(self, ctx, num, den, size, bar=True):
        self.ctx, self.num, self.den, self.bar = ctx, num, den, bar
        em = ctx.em * size
        self.pad = em * 0.12
        self.gap = em * (0.16 if bar else 0.08)
        self.axis = ctx.axis(size)
        self.inner_w = max(num.w, den.w) + 2 * self.pad
        self.margin = em * 0.1
        self.w = self.inner_w + 2 * self.margin
        self.a = self.axis + self.gap + num.d + num.a
        self.d = den.a + den.d + self.gap - self.axis
        r = ctx.rng
        self.bar_jit = (r.normal(0, em * 0.05 * ctx.m), r.normal(0, em * 0.05 * ctx.m),
                        r.normal(0, ctx.stroke * 0.8 * ctx.m))

    def draw(self, cv, x, y):
        x0 = x + self.margin
        by = y - self.axis
        self.num.draw(cv, x0 + (self.inner_w - self.num.w) / 2, by - self.gap - self.num.d)
        self.den.draw(cv, x0 + (self.inner_w - self.den.w) / 2, by + self.gap + self.den.a)
        if self.bar:
            j0, j1, tilt = self.bar_jit
            pts = self.ctx.wobble_line([(x0 + j0, by + tilt / 2), (x0 + self.inner_w + j1, by - tilt / 2)])
            cv.stroke(pts, self.ctx.stroke)


class Scripts(Box):
    def __init__(self, ctx, base, sup, sub, size):
        self.base, self.sup, self.sub = base, sup, sub
        em = ctx.em * size
        self.up = self.down = 0.0
        if sup is not None:
            self.up = max(0.45 * em, base.a - 0.45 * sup.a)
        if sub is not None:
            self.down = max(0.22 * em, base.d - 0.3 * sub.d) if sup is None else max(0.28 * em, base.d * 0.8)
        if sup is not None and sub is not None:
            clash = (sub.a - self.down) - (self.up - sup.d) + 0.12 * em
            if clash > 0:
                self.up += clash / 2
                self.down += clash / 2
        self.kern = em * 0.02
        sw = max(sup.w if sup else 0, sub.w if sub else 0)
        self.w = base.w + self.kern + sw
        self.a = max(base.a, self.up + sup.a if sup else 0)
        self.d = max(base.d, self.down + sub.d if sub else 0)

    def draw(self, cv, x, y):
        self.base.draw(cv, x, y)
        sx = x + self.base.w + self.kern
        if self.sup is not None:
            self.sup.draw(cv, sx, y - self.up)
        if self.sub is not None:
            self.sub.draw(cv, sx - self.kern * 2, y + self.down)


class Limits(Box):
    def __init__(self, ctx, op, upper, lower, size):
        self.op, self.upper, self.lower = op, upper, lower
        self.gap = ctx.em * size * 0.15
        self.w = max(op.w, upper.w if upper else 0, lower.w if lower else 0)
        self.a = op.a + (self.gap + upper.height if upper else 0)
        self.d = op.d + (self.gap + lower.height if lower else 0)

    def draw(self, cv, x, y):
        self.op.draw(cv, x + (self.w - self.op.w) / 2, y)
        if self.upper:
            self.upper.draw(cv, x + (self.w - self.upper.w) / 2, y - self.op.a - self.gap - self.upper.d)
        if self.lower:
            self.lower.draw(cv, x + (self.w - self.lower.w) / 2, y + self.op.d + self.gap + self.lower.a)


class Sqrt(Box):
    def __init__(self, ctx, body, index, size):
        self.ctx, self.body, self.index = ctx, body, index
        em = ctx.em * size
        self.gap = em * 0.14
        self.pad = em * 0.08
        self.hook = em * 0.55
        self.lead = max(0.0, (index.w - self.hook * 0.45) if index else 0.0)
        self.w = self.lead + self.hook + body.w + 2 * self.pad
        self.top = body.a + self.gap
        self.a = self.top + ctx.stroke
        self.d = body.d + em * 0.05
        if index is not None:
            self.a = max(self.a, em * 0.35 + index.height)
        r = ctx.rng
        self.j = r.normal(0, em * 0.03 * ctx.m, 4)

    def draw(self, cv, x, y):
        em_top = y - self.top
        bot = y + self.body.d + self.ctx.em * 0.02
        H = bot - em_top
        hs = min(H, self.ctx.em * 0.9)
        x0 = x + self.lead
        j = self.j
        pts = [
            (x0, bot - hs * 0.42 + j[0]),
            (x0 + self.hook * 0.18, bot - hs * 0.52),
            (x0 + self.hook * 0.45 + j[1], bot),
            (x0 + self.hook + j[2], em_top),
            (x + self.w - self.pad * 0.5 + j[3], em_top + j[3] * 0.5),
        ]
        cv.stroke(self.ctx.wobble_line(pts, step=5), self.ctx.stroke)
        self.body.draw(cv, x0 + self.hook + self.pad, y)
        if self.index is not None:
            self.index.draw(cv, x, bot - hs * 0.55 - self.index.d)


class Accent(Box):
    def __init__(self, ctx, kind, body, size):
        self.ctx, self.kind, self.body = ctx, kind, body
        em = ctx.em * size
        self.em = em
        self.gap = em * 0.12
        self.w, self.a, self.d = body.w, body.a, body.d
        self.mark = None
        if kind in ("bar", "vec", "lvec", "dot", "ddot"):
            self.a = body.a + self.gap + em * (0.22 if "vec" in kind else 0.1)
        elif kind in ("hat", "tilde"):
            ch = "^" if kind == "hat" else "~"
            self.mark = GlyphBox(ctx, ch, size * 0.7, tight=True)
            if self.mark.g is not None and self.mark.g.fallback:
                self.mark = None
            self.a = body.a + self.gap + em * 0.25
        elif kind == "under":
            self.d = body.d + self.gap + ctx.stroke
        r = ctx.rng
        self.j = r.normal(0, em * 0.04 * ctx.m, 3)

    def draw(self, cv, x, y):
        b, c = self.body, self.ctx
        b.draw(cv, x, y)
        top = y - b.a - self.gap
        x0, x1 = x + self.em * 0.08 + self.j[0], x + b.w - self.em * 0.06 + self.j[1]
        k = self.kind
        if k in ("bar", "vec", "lvec"):
            ly = top - (self.em * 0.08 if "vec" in k else 0)
            cv.stroke(c.wobble_line([(x0, ly + self.j[2] * 0.3), (x1, ly - self.j[2] * 0.3)]), c.stroke)
            if "vec" in k:
                h = self.em * 0.14
                tip, sgn = ((x1, ly), -1) if k == "vec" else ((x0, ly), 1)
                cv.stroke([(tip[0] + sgn * h * 1.2, tip[1] - h), tip, (tip[0] + sgn * h * 1.2, tip[1] + h)], c.stroke)
        elif k in ("dot", "ddot"):
            cx = x + b.w / 2
            xs = [cx] if k == "dot" else [cx - self.em * 0.12, cx + self.em * 0.12]
            for px in xs:
                r = max(1.0, c.stroke * 0.7)
                cv.stroke([(px - r * 0.3, top - r), (px + r * 0.3, top - r * 0.8)], c.stroke * 1.3)
        elif k in ("hat", "tilde"):
            if self.mark is not None:
                self.mark.draw(cv, x + (b.w - self.mark.w) / 2, top + self.mark.d + self.em * 0.02)
            else:
                cx, w = x + b.w / 2, min(b.w * 0.45, self.em * 0.3)
                if k == "hat":
                    pts = [(cx - w, top), (cx, top - self.em * 0.2), (cx + w, top)]
                else:
                    pts = [(cx - w + i * w / 4, top - self.em * 0.1 + math.sin(i * math.pi / 4) * -self.em * 0.07)
                           for i in range(9)]
                cv.stroke(c.wobble_line(pts, amp=c.stroke * 0.15), c.stroke)
        elif k == "under":
            ly = y + b.d + self.gap
            cv.stroke(c.wobble_line([(x0, ly), (x1, ly + self.j[2] * 0.3)]), c.stroke)
        elif k == "cancel":
            cv.stroke(c.wobble_line([(x0, y + b.d), (x1, y - b.a)]), c.stroke)


def delim_glyph(ctx, ch, size, top, bot):
    """A delimiter stretched to cover [top, bot] (relative to baseline)."""
    if not ch:
        return Space(0)
    if ch == "‖":
        return HBox([delim_glyph(ctx, "|", size, top, bot), delim_glyph(ctx, "|", size, top, bot)])
    ch = {"⟨": "<", "⟩": ">"}.get(ch, ch) if not ctx.profile.has(ch) else ch
    normal = GlyphBox(ctx, ch, size, tight=True)
    if normal.g is None or (normal.top <= top + ctx.em * size * 0.08 and
                            normal.top + normal.g.h * normal.sy >= bot - ctx.em * size * 0.08):
        return normal
    return GlyphBox(ctx, ch, size, fit=(top, bot), tight=True)


class Delim(Box):
    def __init__(self, ctx, l, body, r, size):
        axis = ctx.axis(size)
        pad = ctx.em * size * 0.1
        span = max(body.a - axis, body.d + axis, ctx.em * size * 0.55)
        top, bot = -(axis + span) - pad, span - axis + pad
        self.box = HBox([delim_glyph(ctx, l, size, top, bot), Space(ctx.em * size * 0.04), body,
                         Space(ctx.em * size * 0.04), delim_glyph(ctx, r, size, top, bot)])
        self.w, self.a, self.d = self.box.w, self.box.a, self.box.d

    def draw(self, cv, x, y):
        self.box.draw(cv, x, y)


class Grid(Box):
    """Matrix / cases / inline table, vertically centred on the math axis."""

    def __init__(self, ctx, rows, size, align="c", col_gap=0.9, center=True):
        self.rows, self.align = rows, align
        em = ctx.em * size
        ncol = max(len(r) for r in rows) if rows else 0
        self.cw = [max((r[i].w for r in rows if i < len(r)), default=0) for i in range(ncol)]
        self.ra = [max((c.a for c in r), default=em * 0.7) for r in rows]
        self.rd = [max((c.d for c in r), default=em * 0.2) for r in rows]
        self.rgap = em * 0.35
        self.cgap = em * col_gap
        self.w = sum(self.cw) + self.cgap * max(ncol - 1, 0)
        H = sum(self.ra) + sum(self.rd) + self.rgap * max(len(rows) - 1, 0)
        shift = ctx.axis(size) + H / 2 if center else self.ra[0] if rows else 0
        self.top = -shift
        self.a = shift
        self.d = H - shift

    def draw(self, cv, x, y):
        yy = y + self.top
        for r, ra, rd in zip(self.rows, self.ra, self.rd):
            base = yy + ra
            xx = x
            for i, cell in enumerate(r):
                if self.align == "l":
                    off = 0
                elif self.align == "r":
                    off = self.cw[i] - cell.w
                else:
                    off = (self.cw[i] - cell.w) / 2
                cell.draw(cv, xx + off, base)
                xx += self.cw[i] + self.cgap
            yy = base + rd + self.rgap


class Boxed(Box):
    def __init__(self, ctx, body, size):
        self.ctx, self.body = ctx, body
        self.pad = ctx.em * size * 0.25
        self.w = body.w + 2 * self.pad + ctx.em * size * 0.1
        self.a = body.a + self.pad
        self.d = body.d + self.pad
        self.j = ctx.rng.normal(0, ctx.em * size * 0.05 * ctx.m, 8)

    def draw(self, cv, x, y):
        c, j = self.ctx, self.j
        x0, x1 = x + c.em * 0.05, x + self.w - c.em * 0.05
        y0, y1 = y - self.a, y + self.d
        o = c.em * 0.08  # corners overshoot, like a real hand-drawn box
        for p, q in (((x0 - o, y0 + j[0]), (x1 + o * 0.5, y0 + j[1])),
                     ((x1 + j[2], y0 - o), (x1 + j[3], y1 + o * 0.5)),
                     ((x1 + o, y1 + j[4]), (x0 - o * 0.5, y1 + j[5])),
                     ((x0 + j[6], y1 + o), (x0 + j[7], y0 - o * 0.5))):
            cv.stroke(c.wobble_line([p, q]), c.stroke)
        self.body.draw(cv, x + self.pad + c.em * 0.05, y)


# ---------------------------------------------------------------------------
# math node -> box
# ---------------------------------------------------------------------------
def script_size(size):
    return max(0.45, size * (0.7 if size > 0.75 else 0.8))


def text_box(ctx, s, size, bold=False):
    items = []
    for ch in s:
        if ch == " ":
            items.append(Space(ctx.em * size * 0.4))
        else:
            items.append(GlyphBox(ctx, ch, size, bold=bold))
    return HBox(items)


BIGOP_SCALE = {"∫": (1.9, 1.35), "∬": (1.9, 1.35), "∭": (1.9, 1.35), "∮": (1.9, 1.35)}
INTEGRALS = set(BIGOP_SCALE)


def build_math(ctx, node, size=1.0, display=False):
    """Convert a parsed math node into a box."""
    t = node[0]
    em = ctx.em * size

    if t == "row":
        return build_row(ctx, node[1], size, display)
    if t in ("sym", "punct"):
        return GlyphBox(ctx, node[1], size)
    if t in ("op", "rel"):
        return GlyphBox(ctx, node[1], size)
    if t == "text":
        return text_box(ctx, node[1], size)
    if t == "func":
        return text_box(ctx, node[1], size)
    if t == "space":
        return Space(em * node[1])
    if t == "bigop":
        disp_f, text_f = BIGOP_SCALE.get(node[1], (1.45, 1.1))
        f = disp_f if display else text_f
        return GlyphBox(ctx, node[1], size, scale=f, center_on=-ctx.axis(size))
    if t == "big":
        axis = ctx.axis(size)
        half = em * 0.6 * node[2]
        return delim_glyph(ctx, node[1], size, -axis - half, -axis + half)
    if t == "frac":
        inner = size if display else max(0.5, size * 0.78)
        num = build_math(ctx, node[1], inner, False)
        den = build_math(ctx, node[2], inner, False)
        return Frac(ctx, num, den, size, bar=node[3])
    if t == "scripts":
        base_node, sup_n, sub_n = node[1], node[2], node[3]
        ss = script_size(size)
        sup = build_math(ctx, sup_n, ss) if sup_n is not None else None
        sub = build_math(ctx, sub_n, ss) if sub_n is not None else None
        base = build_math(ctx, base_node, size, display)
        limits_style = (base_node[0] == "bigop" and base_node[1] not in INTEGRALS and display) or \
                       (base_node[0] == "func" and base_node[2])   # lim: limits below, as written by hand
        if limits_style:
            return Limits(ctx, base, sup, sub, size)
        return Scripts(ctx, base, sup, sub, size)
    if t == "sqrt":
        body = build_math(ctx, node[1], size, display)
        idx = build_math(ctx, node[2], script_size(script_size(size))) if node[2] else None
        return Sqrt(ctx, body, idx, size)
    if t == "accent":
        return Accent(ctx, node[1], build_math(ctx, node[2], size, display), size)
    if t == "delim":
        body = build_math(ctx, node[2], size, display)
        return Delim(ctx, node[1], body, node[3], size)
    if t == "matrix":
        rows, l, r, align = node[1], node[2], node[3], node[4]
        inner = size if display else max(0.55, size * 0.85)
        cells = [[build_math(ctx, c, inner, False) for c in row] for row in rows]
        grid = Grid(ctx, cells, size, align=align, col_gap=1.1 if align == "l" else 0.9)
        return Delim(ctx, l, grid, r, size)
    if t == "table":
        rows = align_table(ctx, node[1], size, display)
        return Grid(ctx, [[r] for r in rows], size, align="l")
    if t == "boxed":
        return Boxed(ctx, build_math(ctx, node[1], size, display), size)
    if t == "phantom":
        return Phantom(build_math(ctx, node[1], size, display))
    if t == "newline":
        return Space(em * 0.5)
    return Space(0)


def build_row(ctx, items, size, display):
    em = ctx.em * size
    boxes = []
    prev = None                       # type of the previous atom, for spacing
    k = 1.0 if size >= 0.9 else 0.5
    for n in items:
        t = n[0]
        kind = t
        if t == "scripts":
            kind = n[1][0]
        box = build_math(ctx, n, size, display)
        if kind == "op":
            unary = prev in (None, "op", "rel", "punct", "open")
            if unary:
                boxes.append(box)
            else:
                boxes += [Space(em * 0.2 * k), box, Space(em * 0.2 * k)]
        elif kind == "rel":
            boxes += [Space(em * 0.28 * k), box, Space(em * 0.28 * k)]
        elif kind == "punct":
            boxes += [box, Space(em * 0.2 * k)]
        elif kind in ("func", "bigop"):
            if prev not in (None, "op", "rel", "open", "space"):
                boxes.append(Space(em * 0.12))
            boxes += [box, Space(em * 0.1)]
        else:
            boxes.append(box)
        if t == "sym" and n[1] in "([{":
            kind = "open"
        prev = kind
    return HBox(boxes)


def align_table(ctx, rows, size, display):
    """Rows of an aligned environment -> one HBox per row, columns lined up.
    Columns alternate right/left alignment like amsmath `aligned`."""
    cells = [[build_math(ctx, c, size, display) for c in row] for row in rows]
    ncol = max((len(r) for r in cells), default=0)
    widths = [max((r[i].w for r in cells if i < len(r)), default=0) for i in range(ncol)]
    out = []
    for r in cells:
        items = []
        for i in range(ncol):
            cell = r[i] if i < len(r) else Space(0)
            slack = widths[i] - cell.w
            if ncol == 1:
                items.append(cell)
                continue
            if i % 2 == 0:
                items += [Space(slack), cell]
            else:
                items += [cell, Space(slack)]
                if i < ncol - 1:
                    items.append(Space(ctx.em * size * 1.5))
        out.append(HBox(items))
    return out


def split_display(ctx, node, size):
    """A display equation as a list of line boxes (one per row for aligned
    environments / explicit line breaks)."""
    if node[0] == "row" and len(node[1]) == 1 and node[1][0][0] == "table":
        node = node[1][0]
    if node[0] == "table":
        return align_table(ctx, node[1], size, True)
    return [build_math(ctx, node, size, True)]


def break_at_relations(ctx, node, size, max_w):
    """Split an over-wide display row into several lines, breaking before
    top-level relations (=, <, ...) and, if still needed, before + / -.
    Continuation lines are indented like a hand-written derivation."""
    if node[0] != "row":
        return None

    def split(items, kinds):
        parts, cur = [], []
        for n in items:
            if n[0] in kinds and cur and cur[-1][0] not in ("rel", "op"):
                parts.append(cur)
                cur = []
            cur.append(n)
        parts.append(cur)
        return parts

    parts = []
    for chunk in split(node[1], ("rel",)):
        if build_math(ctx, ("row", chunk), size, True).w > max_w * 0.85:
            parts += split(chunk, ("op",))
        else:
            parts.append(chunk)
    if len(parts) < 2:
        return None
    indent = ctx.em * size * 1.5
    lines, acc = [], []
    for p in parts:
        room = max_w - (indent if lines else 0)
        if acc and build_math(ctx, ("row", acc + p), size, True).w > room:
            lines.append(acc)
            acc = list(p)
        else:
            acc = acc + p
    lines.append(acc)
    out = []
    for k, items in enumerate(lines):
        box = build_math(ctx, ("row", items), size, True)
        out.append(HBox([Space(indent), box]) if k else box)
    return out
