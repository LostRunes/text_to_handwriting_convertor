"""Render pasted text (markdown + LaTeX, e.g. a ChatGPT answer) as
handwritten pages."""

import re
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from . import charset as C
from .layout import (Bullet, Canvas, Ctx, GlyphBox, HBox, Space, break_at_relations, build_math,
                     split_display, text_box)
from .mathparse import parse_math

# ---------------------------------------------------------------------------
# options
# ---------------------------------------------------------------------------
INKS = {"blue": (22, 45, 140), "black": (28, 28, 34), "red": (170, 30, 35),
        "green": (20, 105, 60), "pencil": (85, 85, 90)}


@dataclass
class Options:
    ink: str = "blue"
    paper: str = "ruled"          # ruled | plain | grid
    size: float = 1.0             # handwriting size relative to line spacing
    messiness: float = 1.0        # 0 = neat, 2 = sloppy
    weight: float = 1.0           # pen thickness multiplier
    line_spacing: int = 72        # px between ruled lines (200 DPI)
    display_align: str = "left"   # left | center
    seed: int = 7
    page_w: int = C.PAGE_W
    page_h: int = C.PAGE_H

    @classmethod
    def from_dict(cls, d):
        o = cls()
        for k, v in (d or {}).items():
            if hasattr(o, k) and v not in (None, ""):
                t = type(getattr(o, k))
                setattr(o, k, t(v))
        return o

    def ink_rgb(self):
        if self.ink in INKS:
            return INKS[self.ink]
        m = re.fullmatch(r"#?([0-9a-fA-F]{6})", self.ink.strip())
        if m:
            h = m.group(1)
            return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
        return INKS["blue"]


# ---------------------------------------------------------------------------
# markdown-ish document parsing
# ---------------------------------------------------------------------------
SUPERSCRIPTS = dict(zip("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱ", "0123456789+-=()ni"))
SUBSCRIPTS = dict(zip("₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₒₓᵢⱼₙ", "0123456789+-=()aeoxijn"))

DISPLAY_RE = re.compile(
    r"\$\$(.+?)\$\$|\\\[(.+?)\\\]|"
    r"(\\begin\{(align\*?|equation\*?|gather\*?|multline\*?|eqnarray\*?)\}.+?\\end\{\4\})",
    re.S)
INLINE_MATH_RE = re.compile(r"\\\((.+?)\\\)|\$(?![\s$])([^$\n]+?)(?<![\s\\])\$(?!\d)")
EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]")


@dataclass
class Block:
    kind: str                     # para | display | blank | table
    inline: list = field(default_factory=list)
    latex: str = ""
    indent: float = 0.0           # in em
    label: str = ""
    heading: int = 0
    rows: list = field(default_factory=list)


def _clean_text(s):
    s = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", s)            # images
    s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s)             # links
    s = s.replace("`", "")
    s = re.sub(r"(?<![\w*])\*(?![\s*])(.+?)(?<![\s*])\*(?![\w*])", r"\1", s)   # *italic*
    s = re.sub(r"(?<!\w)_(?!\s)(.+?)(?<!\s)_(?!\w)", r"\1", s)                  # _italic_
    s = re.sub(r"\\([*_#`\[\]()>~|])", r"\1", s)
    s = EMOJI_RE.sub("", s)
    return s


def _scripts_to_math(s):
    """'x² + y₁' -> text + inline-math tokens for the unicode scripts."""
    out, buf = [], ""
    i = 0
    while i < len(s):
        c = s[i]
        if c in SUPERSCRIPTS or c in SUBSCRIPTS:
            table, op = (SUPERSCRIPTS, "^") if c in SUPERSCRIPTS else (SUBSCRIPTS, "_")
            j = i
            while j < len(s) and s[j] in table:
                j += 1
            if buf:
                out.append(("text", buf))
                buf = ""
            out.append(("math", "{}" + op + "{" + "".join(table[ch] for ch in s[i:j]) + "}"))
            i = j
        else:
            buf += c
            i += 1
    if buf:
        out.append(("text", buf))
    return out


def parse_inline(s):
    """-> list of ('text', str, bold) and ('math', latex)."""
    s = s.replace("\\$", "\x01")
    tokens = []
    pos = 0
    for m in INLINE_MATH_RE.finditer(s):
        if m.start() > pos:
            tokens.append(("text", s[pos:m.start()]))
        tokens.append(("math", (m.group(1) or m.group(2)).replace("\x01", "\\$")))
        pos = m.end()
    if pos < len(s):
        tokens.append(("text", s[pos:]))

    out, bold = [], False
    for t in tokens:
        if t[0] == "math":
            out.append(t)
            continue
        for part in re.split(r"(\*\*|__)", t[1]):
            if part in ("**", "__"):
                bold = not bold
                continue
            part = _clean_text(part.replace("\x01", "$"))
            for sub in _scripts_to_math(part):
                out.append((sub[0], sub[1], bold) if sub[0] == "text" else sub)
    return out


def parse_document(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
    displays = []

    def stash(m):
        displays.append(m.group(1) or m.group(2) or m.group(3))
        return f"\n\x00{len(displays) - 1}\x00\n"

    text = DISPLAY_RE.sub(stash, text)
    blocks = []
    in_code = False
    table_rows = []

    def flush_table():
        if table_rows:
            blocks.append(Block("table", rows=list(table_rows)))
            table_rows.clear()

    for raw in text.split("\n"):
        line = raw.rstrip()
        stripped = line.strip()
        if stripped.startswith("```"):
            flush_table()
            in_code = not in_code
            continue
        if in_code:
            if not stripped:
                blocks.append(Block("blank"))
            else:
                ind = (len(line) - len(line.lstrip())) / 4
                blocks.append(Block("para", inline=[("text", line.strip(), False)], indent=ind))
            continue

        if stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2:
            if re.fullmatch(r"\|?[\s:|-]+\|?", stripped):
                continue                                   # |---|---| separator
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            table_rows.append([parse_inline(c) for c in cells])
            continue
        flush_table()

        m = re.fullmatch(r"\x00(\d+)\x00", stripped)
        if m:
            blocks.append(Block("display", latex=displays[int(m.group(1))]))
            continue
        if not stripped:
            if blocks and blocks[-1].kind != "blank":
                blocks.append(Block("blank"))
            continue
        if re.fullmatch(r"(-{3,}|\*{3,}|_{3,})", stripped):
            if blocks and blocks[-1].kind != "blank":
                blocks.append(Block("blank"))
            continue

        indent_spaces = len(line) - len(line.lstrip())
        body = stripped
        while body.startswith(">"):
            body = body[1:].lstrip()

        h = re.match(r"(#{1,6})\s+(.*)", body)
        if h:
            blocks.append(Block("para", inline=parse_inline(h.group(2).strip("# ")),
                                heading=len(h.group(1))))
            continue
        b = re.match(r"([-*+•])\s+(.*)", body)
        if b:
            blocks.append(Block("para", inline=parse_inline(b.group(2)), label="•",
                                indent=indent_spaces / 2 * 1.2 + 0.3))
            continue
        n = re.match(r"(\d+[.)]|[a-zA-Z][.)]|\([a-zA-Z0-9]+\))\s+(.*)", body)
        if n:
            blocks.append(Block("para", inline=parse_inline(n.group(2)), label=n.group(1),
                                indent=indent_spaces / 2 * 1.2))
            continue
        blocks.append(Block("para", inline=parse_inline(body), indent=indent_spaces / 4))
    flush_table()
    return blocks


# ---------------------------------------------------------------------------
# page composition
# ---------------------------------------------------------------------------
class PageWriter:
    def __init__(self, profile, opt):
        self.opt = opt
        self.rng = np.random.default_rng(opt.seed)
        self.L = opt.line_spacing
        em = self.L * 0.5 * opt.size
        self.ctx = Ctx(profile, em, self.rng, opt.messiness)
        self.em = em
        self.W, self.H = opt.page_w, opt.page_h
        self.first_rule = int(self.L * 3.4)
        self.left = int(self.W * 0.125)
        self.right = self.W - int(self.W * 0.055)
        self.bottom = self.H - int(self.L * 1.4)
        self.pages = []
        self._new_page()

    # -- vertical flow --------------------------------------------------
    def _new_page(self):
        self.canvas = Canvas(self.W, self.H)
        self.pages.append(self.canvas)
        self.prev_bottom = self.first_rule - self.L * 0.9

    def _baseline_for(self, a):
        need = self.prev_bottom + a + self.em * 0.12
        k = max(0, int(np.ceil((need - self.first_rule) / self.L)))
        return self.first_rule + k * self.L

    def place(self, a, d):
        """Return the baseline y for a line of ascent a / descent d."""
        y = self._baseline_for(a)
        if y + d > self.bottom and self.prev_bottom > self.first_rule:
            self._new_page()
            y = self._baseline_for(a)
        self.prev_bottom = y + d
        return y

    def skip_line(self):
        self.prev_bottom += self.L * 0.7

    # -- horizontal drawing with hand drift -----------------------------
    def draw_line(self, units, x0, y, underline=False):
        """units: list of (x_offset, box)."""
        m = self.opt.messiness
        slope = np.tan(np.radians(self.rng.normal(0, 0.3 * m)))
        drift = 0.0
        x0 += self.rng.normal(0, 0.06 * m) * self.em
        end = x0
        for dx, box in units:
            drift = float(np.clip(drift + self.rng.normal(0, 0.018 * m) * self.em, -0.08 * self.em, 0.08 * self.em))
            bx = x0 + dx
            box.draw(self.canvas, bx, y + slope * dx + drift)
            end = bx + box.w
        if underline and units:
            pts = self.ctx.wobble_line([(x0, y + self.em * 0.32), (end, y + self.em * 0.32 + slope * (end - x0))])
            self.canvas.stroke(pts, self.ctx.stroke * 1.1)

    # -- inline content -> units ----------------------------------------
    def make_units(self, inline, size):
        """Group inline tokens into unbreakable words: list of [space_before, boxes, kind]."""
        units, cur, space = [], [], False

        def flush():
            nonlocal cur, space
            if cur:
                units.append([space, cur])
            cur, space = [], False

        for tok in inline:
            if tok[0] == "math":
                try:
                    node = parse_math(tok[1])
                    cur.append(("math", node))
                except Exception:
                    cur.append(("text", tok[1], False))
                continue
            for part in re.split(r"(\s+)", tok[1]):
                if not part:
                    continue
                if part.isspace():
                    flush()
                    space = True
                else:
                    cur.append(("text", part, tok[2]))
        flush()

        out = []
        for space, pieces in units:
            boxes = []
            for p in pieces:
                if p[0] == "math":
                    boxes.append(("math", p[1], build_math(self.ctx, p[1], size, False)))
                else:
                    boxes.append(("text", p[1], text_box(self.ctx, p[1], size, bold=p[2])))
            out.append((space, boxes))
        return out

    def _fit_piece(self, piece, avail, size):
        kind, src, box = piece
        if box.w <= avail:
            return [box]
        if kind == "math":
            s = max(0.45, size * avail / box.w * 0.97)
            return [build_math(self.ctx, src, s, False)]
        # break an over-long word into character chunks
        chunks, cur = [], []
        for g in box.items:
            if cur and sum(b.w for b in cur) + g.w > avail:
                chunks.append(HBox(cur))
                cur = []
            cur.append(g)
        if cur:
            chunks.append(HBox(cur))
        return chunks

    def write_paragraph(self, block):
        size = {1: 1.3, 2: 1.18, 3: 1.08}.get(block.heading, 1.04 if block.heading else 1.0)
        if block.heading:
            block.inline = [(t[0], t[1], True) if t[0] == "text" else t for t in block.inline]
            if block.heading <= 2 and self.prev_bottom > self.first_rule:
                self.skip_line()
        indent = block.indent * self.em
        x_start = self.left + indent
        label_w = 0.0
        label_box = None
        if block.label:
            label_box = Bullet(self.ctx, size) if block.label == "•" else text_box(self.ctx, block.label, size)
            label_w = max(label_box.w + self.em * 0.35, self.em * 1.1)

        text_x = x_start + label_w
        avail = self.right - text_x
        units = self.make_units(block.inline, size)

        lines, cur, x = [], [], 0.0
        for space, pieces in units:
            boxes = []
            for p in pieces:
                boxes += self._fit_piece(p, avail, size)
            word = HBox(boxes) if len(boxes) == 1 or sum(b.w for b in boxes) <= avail else None
            gap = self.em * 0.42 * (1 + self.rng.normal(0, 0.12 * self.opt.messiness)) if (space and cur) else 0
            if word is not None:
                if cur and x + gap + word.w > avail:
                    lines.append(cur)
                    cur, x, gap = [], 0.0, 0
                cur.append((x + gap, word))
                x += gap + word.w
            else:
                for b in boxes:                         # long word split into chunks
                    if cur and x + gap + b.w > avail:
                        lines.append(cur)
                        cur, x, gap = [], 0.0, 0
                    cur.append((x + gap, b))
                    x += gap + b.w
                    gap = 0
        if cur:
            lines.append(cur)
        if not lines:
            lines = [[]]

        for i, line in enumerate(lines):
            a = max([b.a for _, b in line] + [self.em * size])
            d = max([b.d for _, b in line] + [self.em * 0.25])
            y = self.place(a, d)
            if i == 0 and label_box is not None:
                self.draw_line([(0, label_box)], x_start, y)
            self.draw_line(line, text_x, y, underline=block.heading in (1, 2))

    def write_display(self, block):
        try:
            node = parse_math(block.latex)
        except Exception:
            self.write_paragraph(Block("para", inline=[("text", block.latex, False)]))
            return
        indent = self.em * 1.6
        avail = self.right - self.left - indent
        rows = []
        for row in split_display(self.ctx, node, 1.0):
            if row.w <= avail:
                rows.append(row)
                continue
            broken = break_at_relations(self.ctx, node, 1.0, avail) if node[0] == "row" else None
            if broken and all(r.w <= avail for r in broken):
                rows += broken
                continue
            s = max(0.45, avail / row.w * 0.97)
            rows += split_display(self.ctx, node, s)
        for r in rows:
            y = self.place(r.a, r.d)
            if self.opt.display_align == "center":
                x = self.left + (self.right - self.left - r.w) / 2
            else:
                x = self.left + indent
            self.draw_line([(0, r)], x, y)

    def write_table(self, block):
        size = 0.95
        grid = []
        for ri, row in enumerate(block.rows):
            cells = []
            for cell in row:
                items = []
                for j, (space, pieces) in enumerate(self.make_units(cell, size)):
                    if j and space:
                        items.append(Space(self.em * 0.4))
                    items += [p[2] for p in pieces]
                if ri == 0:
                    for it in items:
                        for g in getattr(it, "items", []):
                            if isinstance(g, GlyphBox):
                                g.bold = True
                cells.append(HBox(items))
            grid.append(cells)
        ncol = max(len(r) for r in grid)
        widths = [max((r[i].w for r in grid if i < len(r)), default=0) for i in range(ncol)]
        gap = self.em * 1.4
        total = sum(widths) + gap * (ncol - 1)
        avail = self.right - self.left
        if total > avail:
            # too wide for the page: write each row as a sentence instead
            for row in block.rows:
                inline = []
                for i, cell in enumerate(row):
                    if i:
                        inline.append(("text", "  |  ", False))
                    inline += cell
                self.write_paragraph(Block("para", inline=inline))
            return
        for ri, row in enumerate(grid):
            a = max([c.a for c in row] + [self.em])
            d = max([c.d for c in row] + [self.em * 0.25])
            y = self.place(a, d)
            units, x = [], 0.0
            for i, c in enumerate(row):
                units.append((x, c))
                x += widths[i] + gap
            self.draw_line(units, self.left, y, underline=ri == 0)

    def write(self, blocks):
        for b in blocks:
            if b.kind == "blank":
                self.skip_line()
            elif b.kind == "display":
                self.write_display(b)
            elif b.kind == "table":
                self.write_table(b)
            else:
                self.write_paragraph(b)

    # -- final raster ---------------------------------------------------
    def paper(self):
        o, W, H, L = self.opt, self.W, self.H, self.L
        rng = np.random.default_rng(o.seed + 1000)
        base = np.empty((H, W, 3), np.float32)
        base[:] = (252, 250, 244)
        fibre = cv2.resize(rng.normal(0, 1, (H // 6, W // 6)).astype(np.float32), (W, H),
                           interpolation=cv2.INTER_CUBIC)
        base += fibre[..., None] * 2.2
        lines = np.zeros((H, W), np.float32)
        if o.paper == "ruled":
            y = self.first_rule - L * 2
            while y < H - L * 0.5:
                cv2.line(lines, (0, int(y)), (W, int(y)), 1.0, 2, cv2.LINE_AA)
                y += L
            blue = np.array([150, 185, 225], np.float32)
            base = base * (1 - lines[..., None] * 0.75) + blue * lines[..., None] * 0.75
            margin = np.zeros((H, W), np.float32)
            mx = self.left - int(self.em * 0.9)
            cv2.line(margin, (mx, 0), (mx, H), 1.0, 2, cv2.LINE_AA)
            red = np.array([225, 120, 125], np.float32)
            base = base * (1 - margin[..., None] * 0.8) + red * margin[..., None] * 0.8
        elif o.paper == "grid":
            step = L // 2
            for y in range(int(self.first_rule % step), H, step):
                cv2.line(lines, (0, y), (W, y), 1.0, 1, cv2.LINE_AA)
            for x in range(int(self.left % step), W, step):
                cv2.line(lines, (x, 0), (x, H), 1.0, 1, cv2.LINE_AA)
            grey = np.array([170, 200, 215], np.float32)
            base = base * (1 - lines[..., None] * 0.6) + grey * lines[..., None] * 0.6
        return base

    def finish(self):
        o = self.opt
        ink = np.array(o.ink_rgb(), np.float32)
        paper = self.paper()
        rng = np.random.default_rng(o.seed + 2000)
        out = []
        for cv in self.pages:
            a = cv.a
            if o.weight > 1.01:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                w = o.weight - 1
                while w > 0:
                    a = a + (cv2.dilate(a, k) - a) * min(w, 1.0)
                    w -= 1
            elif o.weight < 0.99:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                a = a + (cv2.erode(a, k) - a) * (1 - o.weight)
            a = cv2.GaussianBlur(a, (0, 0), 0.55)
            grain = cv2.GaussianBlur(rng.normal(0, 1, a.shape).astype(np.float32), (0, 0), 1.2)
            a = np.clip(a * (0.94 + grain * 0.09), 0, 1)[..., None] * 0.96
            img = paper * (1 - a) + ink * a
            out.append(Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)))
        return out


def render_text(text, profile, options=None):
    opt = options if isinstance(options, Options) else Options.from_dict(options)
    writer = PageWriter(profile, opt)
    writer.write(parse_document(text))
    return writer.finish()


def save_pdf(pages, path):
    pages[0].save(path, "PDF", save_all=True, append_images=pages[1:], resolution=C.DPI)
    return path
