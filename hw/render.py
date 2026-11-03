"""Render pasted text (markdown + LaTeX, e.g. a ChatGPT answer) as
handwritten pages."""

import re
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from . import charset as C
from .figures import Pen, build_flow, build_graph
from .layout import (Bullet, Canvas, Ctx, GlyphBox, HBox, Space, break_at_relations, build_math,
                     colored, split_display, text_box)
from .mathparse import parse_math

# ---------------------------------------------------------------------------
# options
# ---------------------------------------------------------------------------
INKS = {"blue": (22, 45, 140), "black": (28, 28, 34), "red": (170, 30, 35),
        "green": (20, 105, 60), "pencil": (85, 85, 90), "orange": (205, 95, 20),
        "purple": (95, 40, 140), "pink": (200, 50, 120), "brown": (110, 65, 30),
        "violet": (110, 50, 170), "gray": (110, 110, 115), "grey": (110, 110, 115),
        "teal": (15, 110, 115), "darkblue": (15, 30, 95), "lightblue": (40, 110, 200)}


def resolve_ink(name, default=INKS["blue"]):
    if not name:
        return default
    n = str(name).strip().lower()
    if n in INKS:
        return INKS[n]
    m = re.fullmatch(r"#?([0-9a-f]{6}|[0-9a-f]{3})", n)
    if m:
        h = m.group(1)
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    return default


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
    cursive: bool = False         # join lowercase letters with connecting strokes
    consistency: float = 0.7      # 0 = glyphs exactly as written, 1 = fully evened out
    heading_ink: str = ""         # "" = same as ink
    math_ink: str = ""
    page_w: int = C.PAGE_W
    page_h: int = C.PAGE_H

    @classmethod
    def from_dict(cls, d):
        o = cls()
        for k, v in (d or {}).items():
            if hasattr(o, k) and v is not None:
                t = type(getattr(o, k))
                if v == "" and t is not str:
                    continue
                if t is bool and isinstance(v, str):
                    v = v.strip().lower() in ("1", "true", "yes", "on")
                setattr(o, k, t(v))
        return o

    def ink_rgb(self):
        return resolve_ink(self.ink)


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

# Pen-colour markup:  [red]..[/red]   [color=#aa0033]..[/color]
#                     <span style="color:red">..</span>   <font color="red">..</font>
_CNAMES = "|".join(sorted(INKS, key=len, reverse=True))
COLOR_OPEN = re.compile(
    r"\[(?:colou?r|ink|pen)\s*[=:]\s*[\"']?(#?[\w]+)[\"']?\s*\]"
    rf"|\[({_CNAMES})\]"
    r"|<span[^>]*?colou?r\s*:\s*(#?\w+)[^>]*>"
    r"|<font[^>]*?color\s*=\s*[\"']?(#?\w+)[\"']?[^>]*>", re.I)
COLOR_CLOSE = re.compile(rf"\[/(?:colou?r|ink|pen|{_CNAMES})\]|</span>|</font>", re.I)
MARK_RE = re.compile("(\x02[^\x03]*\x03|\x04)")

GRAPH_LANGS = {"graph", "plot", "chart", "function", "functions", "bar", "barchart", "pie", "piechart"}
FLOW_LANGS = {"diagram", "flow", "flowchart", "flowchart-td", "flowchart-lr"}


def mark_colors(text):
    text = COLOR_OPEN.sub(lambda m: "\x02" + next(g for g in m.groups() if g) + "\x03", text)
    return COLOR_CLOSE.sub("\x04", text)


def consume_marks(s, stack):
    """Apply colour markers in `s` to the stack; return s without them."""
    for m in MARK_RE.findall(s):
        if m == "\x04":
            if stack:
                stack.pop()
        else:
            stack.append(m[1:-1])
    return MARK_RE.sub("", s)


@dataclass
class Block:
    kind: str                     # para | display | blank | table | code | graph | flow
    inline: list = field(default_factory=list)
    latex: str = ""
    indent: float = 0.0           # in em
    label: str = ""
    heading: int = 0
    rows: list = field(default_factory=list)
    lines: list = field(default_factory=list)
    source: str = ""
    color: str = None


def _clean_text(s):
    s = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", s)            # images
    s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s)             # links
    s = s.replace("`", "")
    s = re.sub(r"(?<![\w*])\*(?![\s*])(.+?)(?<![\s*])\*(?![\w*])", r"\1", s)   # *italic*
    s = re.sub(r"(?<!\w)_(?!\s)(.+?)(?<!\s)_(?!\w)", r"\1", s)                  # _italic_
    s = re.sub(r"\\([*_#`\[\]()>~|])", r"\1", s)
    s = re.sub(r"</?(b|i|u|em|strong|br)\s*/?>", "", s, flags=re.I)
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


def parse_inline(s, stack=None):
    """-> list of ('text', str, bold, color) and ('math', latex, color).
    `stack` is the pen-colour stack; colour markers in `s` update it."""
    stack = stack if stack is not None else []
    out, bold = [], [False]
    for seg in MARK_RE.split(s):
        if not seg:
            continue
        if MARK_RE.fullmatch(seg):
            consume_marks(seg, stack)
            continue
        _parse_segment(seg, stack[-1] if stack else None, bold, out)
    return out


def _parse_segment(s, color, bold, out):
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
    for t in tokens:
        if t[0] == "math":
            out.append(("math", t[1], color))
            continue
        for part in re.split(r"(\*\*|__)", t[1]):
            if part in ("**", "__"):
                bold[0] = not bold[0]
                continue
            part = _clean_text(part.replace("\x01", "$"))
            for sub in _scripts_to_math(part):
                if sub[0] == "text":
                    out.append(("text", sub[1], bold[0], color))
                else:
                    out.append(("math", sub[1], color))


def _code_block(lang, body, color):
    lang = lang.lower()
    first = next((l.strip() for l in body if l.strip()), "")
    # figure sources keep colour tags in plain "[red]" form for their own parser
    src = re.sub("\x02([^\x03]*)\x03", r"[\1]", "\n".join(body)).replace("\x04", "")
    if lang in GRAPH_LANGS:
        return Block("graph", source=src, color=color)
    if lang in FLOW_LANGS:
        return Block("flow", source=src, color=color)
    if lang == "mermaid":
        if re.match(r"(graph|flowchart)\b", first, re.I):
            return Block("flow", source=src, color=color)
        if re.match(r"pie\b", first, re.I):
            return Block("graph", source=src, color=color)
    return Block("code", lines=[MARK_RE.sub("", l) for l in body], color=color)


def parse_document(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
    text = mark_colors(text)
    displays = []

    def stash(m):
        latex = m.group(1) or m.group(2) or m.group(3)
        marks = "".join(MARK_RE.findall(latex))
        opens = "".join(k for k in MARK_RE.findall(latex) if k != "\x04")
        closes = marks.count("\x04")
        displays.append(MARK_RE.sub("", latex))
        return f"\n{opens}\n\x00{len(displays) - 1}\x00\n" + "\x04" * closes + "\n"

    text = DISPLAY_RE.sub(stash, text)
    blocks = []
    stack = []
    code = None                    # (lang, lines, color) while inside a ``` fence
    table_rows = []
    table_color = [None]

    def flush_table():
        if table_rows:
            blocks.append(Block("table", rows=list(table_rows), color=table_color[0]))
            table_rows.clear()

    for raw in text.split("\n"):
        line = raw.rstrip()
        stripped = line.strip()
        fence = re.match(r"^\s*(```|~~~)\s*([\w+-]*)", MARK_RE.sub("", line))
        if fence:
            if code is None:
                flush_table()
                consume_marks(line, stack)
                code = (fence.group(2), [], stack[-1] if stack else None)
            else:
                blocks.append(_code_block(*code))
                code = None
                consume_marks(line, stack)
            continue
        if code is not None:
            code[1].append(line)
            continue

        # a line holding only colour markers just changes the pen
        if MARK_RE.search(stripped) and not MARK_RE.sub("", stripped).strip():
            consume_marks(stripped, stack)
            continue

        plain = MARK_RE.sub("", stripped)
        if plain.startswith("|") and plain.endswith("|") and plain.count("|") >= 2:
            if re.fullmatch(r"\|?[\s:|-]+\|?", plain):
                continue                                   # |---|---| separator
            if not table_rows:
                table_color[0] = stack[-1] if stack else None
            cells = [c.strip() for c in re.split(r"(?<!\\)\|", stripped.strip().strip("|"))]
            table_rows.append([parse_inline(c, stack) for c in cells])
            continue
        flush_table()

        m = re.fullmatch(r"\x00(\d+)\x00", stripped)
        if m:
            blocks.append(Block("display", latex=displays[int(m.group(1))], color=stack[-1] if stack else None))
            continue
        if not plain:
            consume_marks(stripped, stack)
            if blocks and blocks[-1].kind != "blank":
                blocks.append(Block("blank"))
            continue
        if re.fullmatch(r"(-{3,}|\*{3,}|_{3,})", plain):
            if blocks and blocks[-1].kind != "blank":
                blocks.append(Block("blank"))
            continue

        indent_spaces = len(line) - len(line.lstrip())
        body = stripped
        while MARK_RE.sub("", body).startswith(">"):
            body = body.replace(">", "", 1).lstrip()

        lead = ""
        mm = re.match(r"^((?:\x02[^\x03]*\x03|\x04)*)(.*)$", body, re.S)
        lead, rest = mm.group(1), mm.group(2)
        h = re.match(r"(#{1,6})\s+(.*)", rest)
        if h:
            blocks.append(Block("para", inline=parse_inline(lead + h.group(2).strip("# "), stack),
                                heading=len(h.group(1))))
            continue
        b = re.match(r"([-*+•])\s+(.*)", rest)
        if b:
            blocks.append(Block("para", inline=parse_inline(lead + b.group(2), stack), label="•",
                                indent=indent_spaces / 2 * 1.2 + 0.3))
            continue
        n = re.match(r"(\d+[.)]|[a-zA-Z][.)]|\([a-zA-Z0-9]+\))\s+(.*)", rest)
        if n:
            blocks.append(Block("para", inline=parse_inline(lead + n.group(2), stack), label=n.group(1),
                                indent=indent_spaces / 2 * 1.2))
            continue
        blocks.append(Block("para", inline=parse_inline(body, stack), indent=indent_spaces / 4))
    if code is not None:
        blocks.append(_code_block(*code))
    flush_table()
    return blocks


# ---------------------------------------------------------------------------
# monospace / ASCII-art code blocks
# ---------------------------------------------------------------------------
H_CH = set("─━═-")
V_CH = set("│┃║|")
BOX_CH = {"┌": "RD", "┐": "LD", "└": "RU", "┘": "LU", "├": "UDR", "┤": "UDL", "┬": "LRD",
          "┴": "LRU", "┼": "LRUD", "╭": "RD", "╮": "LD", "╰": "RU", "╯": "LU", "╔": "RD",
          "╗": "LD", "╚": "RU", "╝": "LU", "╠": "UDR", "╣": "UDL", "╦": "LRD", "╩": "LRU",
          "╬": "LRUD", "┏": "RD", "┓": "LD", "┗": "RU", "┛": "LU"}
ARROW_CH = {"→": "R", "►": "R", "▶": "R", "←": "L", "◄": "L", "◀": "L", "↑": "U", "▲": "U",
            "↓": "D", "▼": "D"}


def classify_ascii(grid):
    """Per cell: (line directions, arrow direction) for box-drawing / ASCII art."""
    def at(r, c):
        if 0 <= r < len(grid) and 0 <= c < len(grid[r]):
            return grid[r][c]
        return " "

    horiz = H_CH | set(BOX_CH) | {"+"}
    vert = V_CH | set(BOX_CH) | {"+"}
    out = {}
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            dirs, arrow = "", None
            if ch in BOX_CH:
                dirs = BOX_CH[ch]
            elif ch in "─━═":
                dirs = "LR"
            elif ch in "│┃║":
                dirs = "UD"
            elif ch == "-":
                if at(r, c - 1) in horiz | {">", "<"} or at(r, c + 1) in horiz | {">", "<"}:
                    dirs = "LR"
            elif ch == "|":
                dirs = "UD"
            elif ch == "+":
                if at(r, c - 1) in horiz - {"+"}:
                    dirs += "L"
                if at(r, c + 1) in horiz - {"+"}:
                    dirs += "R"
                if at(r - 1, c) in vert - {"+"}:
                    dirs += "U"
                if at(r + 1, c) in vert - {"+"}:
                    dirs += "D"
            elif ch in ARROW_CH:
                arrow = ARROW_CH[ch]
            elif ch == ">" and at(r, c - 1) in H_CH:
                arrow = "R"
            elif ch == "<" and at(r, c + 1) in H_CH:
                arrow = "L"
            elif ch in "vV" and at(r - 1, c) in V_CH and not at(r, c - 1).isalnum() and not at(r, c + 1).isalnum():
                arrow = "D"
            elif ch == "^" and at(r + 1, c) in V_CH:
                arrow = "U"
            if dirs or arrow:
                out[(r, c)] = (dirs, arrow)
    return out


# ---------------------------------------------------------------------------
# page composition
# ---------------------------------------------------------------------------
class PageWriter:
    def __init__(self, profile, opt):
        self.opt = opt
        self.rng = np.random.default_rng(opt.seed)
        if hasattr(profile, "set_consistency"):
            profile.set_consistency(opt.consistency)
        self.L = opt.line_spacing
        em = self.L * 0.5 * opt.size
        self.ctx = Ctx(profile, em, self.rng, opt.messiness, opt.cursive)
        self.em = em
        self.W, self.H = opt.page_w, opt.page_h
        self.first_rule = int(self.L * 3.4)
        self.left = int(self.W * 0.125)
        self.right = self.W - int(self.W * 0.055)
        self.bottom = self.H - int(self.L * 1.4)
        self.pages = []
        self.notes = []
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

    def stroke(self, pts, color=None, width=1.0):
        prev = self.canvas.color
        self.canvas.color = color
        self.canvas.stroke(self.ctx.wobble_line(pts), self.ctx.stroke * width)
        self.canvas.color = prev

    # -- horizontal drawing with hand drift -----------------------------
    def draw_line(self, units, x0, y, underline=False, color=None):
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
            self.stroke([(x0, y + self.em * 0.32), (end, y + self.em * 0.32 + slope * (end - x0))], color, 1.1)

    # -- text helpers (also used by figures) -----------------------------
    def make_units(self, inline, size, default_color=None):
        """Group inline tokens into unbreakable words: [(space_before, [(kind, src, box, color)])]."""
        units, cur, space = [], [], False
        math_ink = self.opt.math_ink or None

        def flush():
            nonlocal cur, space
            if cur:
                units.append([space, cur])
            cur, space = [], False

        for tok in inline:
            if tok[0] == "math":
                col = tok[2] or math_ink or default_color
                try:
                    cur.append(("math", parse_math(tok[1]), col))
                except Exception:
                    cur.append(("text", tok[1], False, col))
                continue
            for part in re.split(r"(\s+)", tok[1]):
                if not part:
                    continue
                if part.isspace():
                    flush()
                    space = True
                else:
                    cur.append(("text", part, tok[2], tok[3] or default_color))
        flush()

        out = []
        for space, pieces in units:
            boxes = []
            for p in pieces:
                if p[0] == "math":
                    boxes.append(("math", p[1], build_math(self.ctx, p[1], size, False), p[2]))
                else:
                    boxes.append(("text", p[1], text_box(self.ctx, p[1], size, bold=p[2]), p[3]))
            out.append((space, boxes))
        return out

    def _fit_piece(self, piece, avail, size):
        kind, src, box, color = piece
        if box.w <= avail:
            return [colored(box, color)]
        if kind == "math":
            s = max(0.45, size * avail / box.w * 0.97)
            return [colored(build_math(self.ctx, src, s, False), color)]
        chunks, cur = [], []
        for g in box.items:
            if cur and sum(b.w for b in cur) + g.w > avail:
                chunks.append(colored(HBox(cur), color))
                cur = []
            cur.append(g)
        if cur:
            chunks.append(colored(HBox(cur), color))
        return chunks

    def wrap_units(self, units, avail, size):
        """Greedy word wrap -> list of lines, each a list of (x, box)."""
        lines, cur, x = [], [], 0.0
        for space, pieces in units:
            boxes = []
            for p in pieces:
                boxes += self._fit_piece(p, avail, size)
            word = HBox(boxes) if len(boxes) == 1 or sum(b.w for b in boxes) <= avail else None
            gap = self.em * size * 0.42 * (1 + self.rng.normal(0, 0.12 * self.opt.messiness)) if (space and cur) else 0
            if word is not None:
                if cur and x + gap + word.w > avail:
                    lines.append(cur)
                    cur, x, gap = [], 0.0, 0
                cur.append((x + gap, word))
                x += gap + word.w
            else:
                for b in boxes:
                    if cur and x + gap + b.w > avail:
                        lines.append(cur)
                        cur, x, gap = [], 0.0, 0
                    cur.append((x + gap, b))
                    x += gap + b.w
                    gap = 0
        if cur:
            lines.append(cur)
        return lines

    @staticmethod
    def line_box(line):
        items, x = [], 0.0
        for dx, b in line:
            if dx > x:
                items.append(Space(dx - x))
            items.append(b)
            x = dx + b.w
        return HBox(items)

    # T-interface for figures
    def text(self, s, size):
        units = self.make_units(parse_inline(mark_colors(s)), size)
        return self.line_box(self.wrap_units(units, 1e9, size)[0]) if units else Space(0)

    def math(self, latex, size):
        return build_math(self.ctx, parse_math(latex), size, False)

    def lines(self, s, size, width):
        units = self.make_units(parse_inline(mark_colors(s)), size)
        return [self.line_box(l) for l in self.wrap_units(units, width, size)] or [Space(0)]

    # -- blocks -----------------------------------------------------------
    def write_paragraph(self, block):
        size = {1: 1.3, 2: 1.18, 3: 1.08}.get(block.heading, 1.04 if block.heading else 1.0)
        color = None
        if block.heading:
            hcol = self.opt.heading_ink or None
            block.inline = [("text", t[1], True, t[3] or hcol) if t[0] == "text" else ("math", t[1], t[2] or hcol)
                            for t in block.inline]
            colors = [t[-1] for t in block.inline if t[-1]]
            color = colors[0] if colors else None
            if block.heading <= 2 and self.prev_bottom > self.first_rule:
                self.skip_line()
        indent = block.indent * self.em
        x_start = self.left + indent
        label_w = 0.0
        label_box = None
        if block.label:
            label_box = Bullet(self.ctx, size) if block.label == "•" else text_box(self.ctx, block.label, size)
            first = next((t[-1] for t in block.inline if t[-1]), None)
            label_box = colored(label_box, first)
            label_w = max(label_box.w + self.em * 0.35, self.em * 1.1)

        text_x = x_start + label_w
        avail = self.right - text_x
        lines = self.wrap_units(self.make_units(block.inline, size), avail, size) or [[]]

        for i, line in enumerate(lines):
            a = max([b.a for _, b in line] + [self.em * size])
            d = max([b.d for _, b in line] + [self.em * 0.25])
            y = self.place(a, d)
            if i == 0 and label_box is not None:
                self.draw_line([(0, label_box)], x_start, y)
            self.draw_line(line, text_x, y, underline=block.heading in (1, 2), color=color)

    def write_display(self, block):
        try:
            node = parse_math(block.latex)
        except Exception:
            self.write_paragraph(Block("para", inline=[("text", block.latex, False, block.color)]))
            return
        color = block.color or self.opt.math_ink or None
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
            self.draw_line([(0, colored(r, color))], x, y)

    def write_table(self, block):
        """Hand-ruled table; long cells wrap inside their column."""
        avail = self.right - self.left - self.em * 0.3
        rows_tok = [[[("text", t[1], True, t[3]) if (ri == 0 and t[0] == "text") else t for t in cell]
                     for cell in row] for ri, row in enumerate(block.rows)]
        ncol = max(len(r) for r in rows_tok)
        layout = None
        for size in (0.92, 0.82, 0.72, 0.62):
            em = self.em * size
            pad = em * 0.4
            cells = [[self.make_units(c, size) for c in row] for row in rows_tok]

            def nat(units):
                w = 0.0
                for j, (space, pieces) in enumerate(units):
                    w += sum(p[2].w for p in pieces) + (em * 0.42 if j and space else 0)
                return w

            def minw(units):
                return max((sum(p[2].w for p in pieces) for _, pieces in units), default=0.0)

            natural = [max((nat(r[i]) for r in cells if i < len(r)), default=0) for i in range(ncol)]
            minimum = [min(max((minw(r[i]) for r in cells if i < len(r)), default=0), avail / ncol)
                       for i in range(ncol)]
            space = avail - 2 * pad * ncol
            if sum(natural) <= space:
                widths = natural
            elif sum(minimum) <= space:
                extra = space - sum(minimum)
                flex = [n - m for n, m in zip(natural, minimum)]
                widths = [m + extra * f / (sum(flex) or 1) for m, f in zip(minimum, flex)]
            else:
                continue
            widths = [max(w, em) for w in widths]
            layout = (size, em, pad, cells, widths)
            break
        if layout is None:
            for row in block.rows:
                inline = []
                for i, cell in enumerate(row):
                    if i:
                        inline.append(("text", "  |  ", False, block.color))
                    inline += cell
                self.write_paragraph(Block("para", inline=inline))
            return

        size, em, pad, cells, widths = layout
        xs = [self.left]
        for w in widths:
            xs.append(xs[-1] + w + 2 * pad)
        col = block.color
        seg = {"top": None, "bottom": None, "canvas": None}

        def close_segment():
            if seg["top"] is None:
                return
            cv = self.canvas
            self.canvas = seg["canvas"]
            for x in xs:
                self.stroke([(x, seg["top"]), (x, seg["bottom"])], col)
            self.canvas = cv
            seg["top"] = None

        for ri, row in enumerate(cells):
            wrapped = [self.wrap_units(row[i], widths[i], size) if i < len(row) else [] for i in range(ncol)]
            nlines = max(1, max(len(w) for w in wrapped))
            if self.prev_bottom + nlines * self.L + pad > self.bottom and self.prev_bottom > self.first_rule:
                close_segment()
                self._new_page()
            y = None
            for k in range(nlines):
                parts = [w[k] if k < len(w) else [] for w in wrapped]
                a = max([b.a for p in parts for _, b in p] + [em])
                d = max([b.d for p in parts for _, b in p] + [em * 0.25])
                y = self.place(min(a, em * 1.15) + (pad * 0.25 if k == 0 else 0), d)
                if k == 0:
                    top = y - a - pad * 0.75
                    if seg["top"] is None:
                        seg.update(top=top, canvas=self.canvas)
                        self.stroke([(xs[0], top), (xs[-1], top)], col)
                for i, p in enumerate(parts):
                    if p:
                        self.draw_line(p, xs[i] + pad, y)
            bottom = y + max(d, em * 0.3) + pad * 0.5
            self.stroke([(xs[0], bottom), (xs[-1], bottom)], col, 1.3 if ri == 0 else 1.0)
            seg["bottom"] = bottom
            self.prev_bottom = bottom - em * 0.2
        close_segment()

    def write_code(self, block):
        lines = [l.rstrip() for l in block.lines]
        while lines and not lines[-1].strip():
            lines.pop()
        while lines and not lines[0].strip():
            lines.pop(0)
        if not lines:
            return
        common = min((len(l) - len(l.lstrip()) for l in lines if l.strip()), default=0)
        grid = [l[common:] for l in lines]
        size = 0.85
        cw = self.em * size * 0.62
        x0 = self.left + self.em * 0.6
        longest = max(len(r) for r in grid)
        if longest * cw > self.right - x0:
            f = (self.right - x0) / (longest * cw)
            size *= f
            cw *= f
        art = classify_ascii(grid)
        axis = self.ctx.axis(size)
        # keep a drawing on one page if it fits on one
        need = len(grid) * self.L
        if art and self.prev_bottom + need > self.bottom and need < self.bottom - self.first_rule:
            self._new_page()
        rows = []                                       # (canvas, center_y)
        for r, row in enumerate(grid):
            glyphs = []
            for c, ch in enumerate(row):
                if ch == " " or (r, c) in art:
                    continue
                g = GlyphBox(self.ctx, ch, size, tight=True)
                glyphs.append((c * cw + (cw - g.w) / 2, g))
            a = max([g.a for _, g in glyphs] + [self.em * size * 0.8])
            d = max([g.d for _, g in glyphs] + [self.em * size * 0.25])
            y = self.place(a, d)
            rows.append((self.canvas, y - axis))
            if glyphs:
                self.draw_line([(dx, colored(g, block.color)) for dx, g in glyphs], x0, y)

        def half_gap(r, k):
            j = r + k
            if 0 <= j < len(rows) and rows[j][0] is rows[r][0]:
                return abs(rows[j][1] - rows[r][1]) / 2
            return self.em * size * 0.45

        hsegs, vsegs = {}, {}
        for (r, c), (dirs, arrow) in art.items():
            cx = x0 + c * cw + cw / 2
            cy = rows[r][1]
            for dch in dirs:
                if dch == "L":
                    hsegs.setdefault(r, []).append((cx - cw / 2, cx))
                elif dch == "R":
                    hsegs.setdefault(r, []).append((cx, cx + cw / 2))
                elif dch == "U":
                    vsegs.setdefault((c, id(rows[r][0])), []).append((cy - half_gap(r, -1), cy, rows[r][0]))
                elif dch == "D":
                    vsegs.setdefault((c, id(rows[r][0])), []).append((cy, cy + half_gap(r, 1), rows[r][0]))
            if arrow:
                cv0 = self.canvas
                self.canvas = rows[r][0]
                P = Pen(self.ctx)
                s = self.em * size * 0.3
                vec = {"R": (1, 0), "L": (-1, 0), "U": (0, -1), "D": (0, 1)}[arrow]
                tip = (cx + vec[0] * cw * 0.45, cy + vec[1] * half_gap(r, 1 if arrow == "D" else -1) * 0.8)
                if arrow in "RL":
                    hsegs.setdefault(r, []).append(tuple(sorted((cx - vec[0] * cw / 2, tip[0]))))
                else:
                    a_, b_ = sorted((cy - vec[1] * half_gap(r, -vec[1]), tip[1]))
                    vsegs.setdefault((c, id(rows[r][0])), []).append((a_, b_, rows[r][0]))
                P.arrow_head(tip, vec, size=s, color=block.color)
                for op in P.ops:
                    op(self.canvas, 0, 0)
                self.canvas = cv0

        def merged(ints):
            ints = sorted(ints)
            out = [list(ints[0])]
            for a, b in ints[1:]:
                if a <= out[-1][1] + 1.5:
                    out[-1][1] = max(out[-1][1], b)
                else:
                    out.append([a, b])
            return out

        cv0 = self.canvas
        for r, ints in hsegs.items():
            self.canvas = rows[r][0]
            for a, b in merged(ints):
                if b - a > 1:
                    self.stroke([(a, rows[r][1]), (b, rows[r][1])], block.color)
        for (c, _), ints in vsegs.items():
            cx = x0 + c * cw + cw / 2
            self.canvas = ints[0][2]
            for a, b in merged([(a, b) for a, b, _ in ints]):
                if b - a > 1:
                    self.stroke([(cx, a), (cx, b)], block.color)
        self.canvas = cv0

    def write_figure(self, block):
        max_w = self.right - self.left - self.em
        max_h = self.bottom - self.first_rule - self.L
        builder = build_graph if block.kind == "graph" else build_flow
        try:
            fig, errors = builder(self.ctx, self, block.source, max_w, max_h, block.color)
        except Exception as e:                            # never lose the page over a bad figure
            fig, errors = None, [str(e)]
        if errors:
            self.notes += [f"{block.kind}: {e}" for e in errors]
        if fig is None:
            self.write_code(Block("code", lines=block.source.split("\n"), color=block.color))
            return
        y = self.place(fig.a + self.em * 0.3, self.em * 0.3)
        x = self.left + max(self.em * 0.5, (self.right - self.left - fig.w) / 2)
        fig.draw(self.canvas, x, y)
        self.prev_bottom = y + self.em * 0.3

    def write(self, blocks):
        for b in blocks:
            if b.kind == "blank":
                self.skip_line()
            elif b.kind == "display":
                self.write_display(b)
            elif b.kind == "table":
                self.write_table(b)
            elif b.kind == "code":
                self.write_code(b)
            elif b.kind in ("graph", "flow"):
                self.write_figure(b)
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

    def _ink_alpha(self, a, rng):
        o = self.opt
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        if o.weight > 1.01:
            w = o.weight - 1
            while w > 0:
                a = a + (cv2.dilate(a, k) - a) * min(w, 1.0)
                w -= 1
        elif o.weight < 0.99:
            a = a + (cv2.erode(a, k) - a) * (1 - o.weight)
        a = cv2.GaussianBlur(a, (0, 0), 0.55)
        grain = cv2.GaussianBlur(rng.normal(0, 1, a.shape).astype(np.float32), (0, 0), 1.2)
        return np.clip(a * (0.94 + grain * 0.09), 0, 1)[..., None] * 0.96

    def finish(self):
        default = np.array(self.opt.ink_rgb(), np.float32)
        paper = self.paper()
        rng = np.random.default_rng(self.opt.seed + 2000)
        out = []
        for cv in self.pages:
            img = paper.copy()
            for color, layer in sorted(cv.layers.items(), key=lambda kv: kv[0] is not None):
                if color is not None and not layer.any():
                    continue
                ink = np.array(resolve_ink(color, tuple(default)), np.float32)
                a = self._ink_alpha(layer, rng)
                img = img * (1 - a) + ink * a
            out.append(Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)))
        return out


def render_text(text, profile, options=None, notes=None):
    opt = options if isinstance(options, Options) else Options.from_dict(options)
    writer = PageWriter(profile, opt)
    writer.write(parse_document(text))
    if notes is not None:
        notes += writer.notes
    return writer.finish()


def save_pdf(pages, path):
    pages[0].save(path, "PDF", save_all=True, append_images=pages[1:], resolution=C.DPI)
    return path
