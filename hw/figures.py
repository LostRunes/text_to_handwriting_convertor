"""Hand-drawn figures: function graphs, bar / pie charts and flowcharts.

Each builder returns a `Figure` box (width w, height a, d = 0) whose drawing
is fully decided at build time, so it can be measured, placed and paginated
like any other line of handwriting.

`T` is the text helper passed in by the page writer:
    T.text(str, size)  -> box   (handwriting, inline $math$ allowed)
    T.math(latex, size) -> box
    T.lines(str, size, width) -> [box, ...]   (word-wrapped)
"""

import ast
import math
import re
from fractions import Fraction

import numpy as np

from .layout import Box


# ---------------------------------------------------------------------------
# drawing primitives
# ---------------------------------------------------------------------------
class Figure(Box):
    def __init__(self, w, h, ops):
        self.w, self.a, self.d = w, h, 0.0
        self.ops = ops          # list of callables op(cv, ox, oy)

    def draw(self, cv, x, y):
        for op in self.ops:
            op(cv, x, y - self.a)


def _v(p):
    return np.asarray(p, np.float64)


class Pen:
    """Records hand-drawn strokes relative to a figure's top-left corner."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.ops = []

    def _add(self, pts, width=1.0, intensity=0.95, color=None):
        pts = np.asarray(pts, np.float64)
        w = self.ctx.stroke * width

        def op(cv, ox, oy, pts=pts, w=w, intensity=intensity, color=color):
            prev = cv.color
            if color:
                cv.color = color
            cv.stroke(pts + [ox, oy], w, intensity)
            cv.color = prev
        self.ops.append(op)

    def line(self, p, q, width=1.0, amp=None, color=None, intensity=0.95):
        if amp is None and np.hypot(q[0] - p[0], q[1] - p[1]) > self.ctx.em * 1.5:
            for pts, wf, inten in self.ctx.hand_rule(p, q):          # axes, box sides, bars
                self._add(pts, width * wf, min(intensity, inten), color)
            return
        self._add(self.ctx.wobble_line([p, q], amp=amp), width, intensity, color)

    def poly(self, pts, width=1.0, amp=None, color=None, closed=False, intensity=0.95):
        pts = [tuple(p) for p in pts]
        if closed:
            pts = pts + [pts[0]]
        self._add(self.ctx.wobble_line(pts, amp=amp), width, intensity, color)

    def smooth(self, pts, width=1.0, color=None, intensity=0.95):
        """A long sampled curve: add gentle low-frequency tremor only."""
        pts = np.asarray(pts, np.float64)
        if len(pts) < 2:
            return
        r = self.ctx.rng
        t = np.linspace(0, 1, len(pts))
        amp = self.ctx.stroke * 0.3 * self.ctx.m
        ph = r.uniform(0, 2 * np.pi, 2)
        n = amp * (np.sin(2 * np.pi * (1.3 * t) + ph[0]) * 0.6 + np.sin(2 * np.pi * (3.7 * t) + ph[1]) * 0.4)
        pts = pts + np.stack([n * 0.4, n], axis=1)
        self._add(pts, width, intensity, color)

    def dashed(self, p, q, width=1.0, dash=None, color=None):
        p, q = _v(p), _v(q)
        L = float(np.hypot(*(q - p)))
        dash = dash or self.ctx.em * 0.3
        n = max(1, int(L / (dash * 1.7)))
        for i in range(n + 1):
            a = i * dash * 1.7 / L
            b = min(1.0, a + dash / L)
            if a < 1:
                self.line(p + (q - p) * a, p + (q - p) * b, width, amp=self.ctx.stroke * 0.1, color=color)

    def arrow_head(self, tip, direction, size=None, color=None, width=1.0):
        d = _v(direction)
        n = np.hypot(*d)
        if n == 0:
            return
        d = d / n
        size = size or self.ctx.em * 0.32
        nrm = np.array([-d[1], d[0]])
        tip = _v(tip)
        a = tip - d * size + nrm * size * 0.48
        b = tip - d * size - nrm * size * 0.48
        self._add([a, tip, b], width, 0.95, color)

    def ellipse(self, cx, cy, rx, ry, color=None, width=1.0):
        r = self.ctx.rng
        start = r.uniform(0, 2 * np.pi)
        sweep = 2 * np.pi + r.uniform(0.08, 0.3)          # pen overlaps where it started
        t = np.linspace(start, start + sweep, max(24, int((rx + ry) / 2)))
        k = 1 + r.normal(0, 0.012 * self.ctx.m) * np.sin(t * 2 + r.uniform(0, 6))
        pts = np.stack([cx + rx * k * np.cos(t), cy + ry * k * np.sin(t)], axis=1)
        self._add(pts, width, 0.95, color)

    def rect(self, x0, y0, x1, y1, color=None, width=1.0, overshoot=True):
        o = self.ctx.em * 0.07 if overshoot else 0
        j = lambda: self.ctx.rng.normal(0, self.ctx.em * 0.03 * self.ctx.m)
        self.line((x0 - o, y0 + j()), (x1 + o * 0.5, y0 + j()), width, color=color)
        self.line((x1 + j(), y0 - o), (x1 + j(), y1 + o * 0.5), width, color=color)
        self.line((x1 + o, y1 + j()), (x0 - o * 0.5, y1 + j()), width, color=color)
        self.line((x0 + j(), y1 + o), (x0 + j(), y0 - o * 0.5), width, color=color)

    def round_rect(self, x0, y0, x1, y1, r, color=None):
        r = min(r, (x1 - x0) / 2, (y1 - y0) / 2)
        pts = []
        for cx, cy, a0 in ((x1 - r, y0 + r, -90), (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180)):
            for a in np.linspace(a0, a0 + 90, 6):
                pts.append((cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a))))
        self.poly(pts, closed=True, amp=self.ctx.stroke * 0.25, color=color)

    def text(self, box, x, baseline, color=None):
        def op(cv, ox, oy, box=box, x=x, b=baseline, color=color):
            prev = cv.color
            if color:
                cv.color = color
            box.draw(cv, ox + x, oy + b)
            cv.color = prev
        self.ops.append(op)

    def fill_hatch(self, x0, y0, x1, y1, color=None, spacing=None):
        spacing = spacing or self.ctx.em * 0.32
        h = y1 - y0
        k = x0 - h
        while k < x1:
            a = (max(k, x0), y1 - (max(k, x0) - k))
            b_x = min(k + h, x1)
            b = (b_x, y1 - (b_x - k))
            if b[0] - a[0] > 2:
                self.line(a, b, 0.7, amp=self.ctx.stroke * 0.15, color=color, intensity=0.6)
            k += spacing


# ---------------------------------------------------------------------------
# safe maths expressions  ("y = 2x^2 - sin(x)", "\frac{1}{x}", "|x|", ...)
# ---------------------------------------------------------------------------
def _sec(x):
    return 1 / np.cos(x)


FUNCS = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "sec": _sec, "csc": lambda x: 1 / np.sin(x),
    "cot": lambda x: 1 / np.tan(x), "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan,
    "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan, "sinh": np.sinh,
    "cosh": np.cosh, "tanh": np.tanh, "exp": np.exp, "ln": np.log, "log": np.log10,
    "log10": np.log10, "log2": np.log2, "sqrt": np.sqrt, "abs": np.abs, "floor": np.floor,
    "ceil": np.ceil, "sign": np.sign, "sgn": np.sign, "round": np.round,
}
CONSTS = {"pi": np.pi, "e": np.e, "tau": 2 * np.pi}
VARS = {"x", "y", "t"}
NAMES = sorted(set(FUNCS) | set(CONSTS) | VARS, key=len, reverse=True)

_TOK = re.compile(r"\s*(\d+\.?\d*(?:[eE][-+]?\d+)?|\.\d+|[A-Za-z_]+|\*\*|[-+*/()^,%])")


def _latex_to_plain(s):
    s = s.replace("\\left", "").replace("\\right", "").replace("\\,", "").replace("\\!", "")
    s = s.replace("\\cdot", "*").replace("\\times", "*").replace("\\div", "/")
    for _ in range(6):
        s2 = re.sub(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"((\1)/(\2))", s)
        s2 = re.sub(r"\\sqrt\s*\[([^\]]*)\]\s*\{([^{}]*)\}", r"((\2)**(1/(\1)))", s2)
        s2 = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", s2)
        if s2 == s:
            break
        s = s2
    s = re.sub(r"\\(operatorname\{(\w+)\}|(\w+))", lambda m: m.group(2) or m.group(3), s)
    s = s.replace("{", "(").replace("}", ")")
    for a, b in (("π", "pi"), ("·", "*"), ("×", "*"), ("−", "-"), ("÷", "/"), ("²", "^2"), ("³", "^3"), ("√", "sqrt")):
        s = s.replace(a, b)
    s = re.sub(r"\|([^|]+)\|", r"abs(\1)", s)
    return s


def _split_ident(word):
    """'xsinx' -> ['x', 'sin', 'x'] using known names; unknown letters stay single."""
    out, i = [], 0
    while i < len(word):
        for n in NAMES:
            if word.startswith(n, i):
                out.append(n)
                i += len(n)
                break
        else:
            out.append(word[i])
            i += 1
    return out


def to_python(src):
    s = _latex_to_plain(src).strip()
    toks = []
    pos = 0
    while pos < len(s):
        m = _TOK.match(s, pos)
        if not m and not s[pos:].strip():
            break
        if not m:
            raise ValueError(f"can't read '{s[pos:]}'")
        t = m.group(1)
        pos = m.end()
        if t[0].isalpha() or t[0] == "_":
            toks += _split_ident(t)
        else:
            toks.append("**" if t == "^" else t)
    # functions written without brackets: "sin x" -> sin(x), "sin 2x" -> sin(2*x)
    fixed = []
    i = 0
    while i < len(toks):
        t = toks[i]
        fixed.append(t)
        if t in FUNCS and i + 1 < len(toks) and toks[i + 1] != "(":
            j = i + 1
            arg = []
            while j < len(toks) and (re.match(r"[\d.]", toks[j]) or toks[j] in VARS or toks[j] in CONSTS):
                arg.append(toks[j])
                j += 1
            if j < len(toks) and toks[j] == "**" and j + 1 < len(toks):
                arg += toks[j:j + 2]
                j += 2
            if arg:
                fixed += ["("] + arg + [")"]
                i = j
                continue
        i += 1
    # implicit multiplication
    out = []
    for t in fixed:
        if out:
            p = out[-1]
            left = re.match(r"[\d.]", p) or p == ")" or p in VARS or p in CONSTS
            right = re.match(r"[\d.]", t) or t == "(" or t in VARS or t in CONSTS or t in FUNCS
            if left and right:
                out.append("*")
        out.append(t)
    return " ".join(out)


class _Floats(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return ast.copy_location(ast.Constant(float(node.value)), node)
        raise ValueError("bad constant")


_ALLOWED = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load, ast.Constant,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.USub, ast.UAdd)


def compile_expr(src):
    """Compile a maths expression in x / y into a numpy function f(x, y)."""
    tree = ast.parse(to_python(src), mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            raise ValueError(f"not allowed: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in FUNCS and node.id not in CONSTS and node.id not in VARS:
            raise ValueError(f"unknown name {node.id}")
        if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or node.func.id not in FUNCS or node.keywords):
            raise ValueError("bad call")
    tree = ast.fix_missing_locations(_Floats().visit(tree))
    code = compile(tree, "<expr>", "eval")
    uses = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}

    def f(x, y=0.0):
        ns = dict(FUNCS)
        ns.update(CONSTS)
        ns.update(x=x, y=y, t=x)
        with np.errstate(all="ignore"):
            try:
                r = eval(code, {"__builtins__": {}}, ns)
            except (OverflowError, ZeroDivisionError, ValueError, TypeError):
                return np.full(np.shape(x), np.nan)
        return np.broadcast_to(np.asarray(r, np.float64), np.shape(x)).copy()
    f.uses = uses
    return f


def eval_number(src):
    v = compile_expr(src)(np.zeros(1))[0]
    if not np.isfinite(v):
        raise ValueError("not a number")
    return float(v)


# ---------------------------------------------------------------------------
# graph spec
# ---------------------------------------------------------------------------
_NUM = r"[-+]?[\w.π\\{}()/*^ ]+?"
RANGE_RE = re.compile(rf"^(x|y)\s*(?:range|domain|limits?)?\s*(?:[:=]|from|in|between)\s*\[?\s*({_NUM})\s*(?:\.\.|to|,|and|…)\s*({_NUM})\s*\]?$", re.I)
COLOR_TAIL = re.compile(r"\s*(?:\[\s*(#?\w+)\s*\]|[,;]?\s*(?:colou?r|ink|pen)\s*[:=]\s*(#?\w+))\s*$", re.I)
PAIR_RE = re.compile(r"\(\s*([^,()]+(?:\([^()]*\))?[^,()]*)\s*,\s*([^()]+?)\s*\)")


class GraphSpec:
    def __init__(self):
        self.curves = []       # (label_latex, f, implicit: bool, color)
        self.points = []       # (x, y, label)
        self.polylines = []    # ([(x, y)...], color)
        self.bars = []         # (label, value)
        self.pie = []          # (label, value)
        self.xr = self.yr = None
        self.title = self.xlabel = self.ylabel = ""
        self.grid = False
        self.pi_axis = False
        self.errors = []


def _pairs(s):
    out = []
    for a, b in PAIR_RE.findall(s):
        try:
            out.append((eval_number(a), eval_number(b)))
        except ValueError:
            pass
    return out


def _kv_list(s):
    out = []
    for part in re.split(r"[,;]\s*(?![^()]*\))", s):
        m = re.match(r"\s*\"?([^=:\"]+?)\"?\s*[=:]\s*([-+\d.eE]+)\s*%?\s*$", part)
        if m:
            out.append((m.group(1).strip(), float(m.group(2))))
    return out


def parse_graph(text):
    g = GraphSpec()
    lines = [l.strip() for l in text.replace(";", "\n").split("\n")] if "pie" not in text.lower() else \
        [l.strip() for l in text.split("\n")]
    for line in lines:
        if not line or line.startswith(("#", "%%", "//")):
            continue
        low = line.lower()
        key, _, val = line.partition(":")
        k = key.strip().lower()
        if k == "title" or low.startswith("pie title") or low.startswith("title "):
            g.title = val.strip() if k == "title" else re.sub(r"^(pie\s+)?title\s+", "", line, flags=re.I)
            continue
        if k in ("xlabel", "x label", "x-axis", "x axis"):
            g.xlabel = val.strip()
            continue
        if k in ("ylabel", "y label", "y-axis", "y axis"):
            g.ylabel = val.strip()
            continue
        if k == "grid":
            g.grid = val.strip().lower() not in ("off", "no", "false", "0")
            continue
        if k in ("points", "point", "scatter", "plot points"):
            for x, y in _pairs(val):
                g.points.append((x, y, ""))
            continue
        if k in ("data", "line", "lines", "polyline"):
            col = COLOR_TAIL.search(val)
            pts = _pairs(val)
            if pts:
                g.polylines.append((pts, (col.group(1) or col.group(2)) if col else None))
            continue
        if k in ("bar", "bars", "bar chart", "histogram"):
            g.bars += _kv_list(val)
            continue
        if k in ("pie", "pie chart"):
            g.pie += _kv_list(val)
            continue
        if low in ("pie", "pie chart", "pie showdata"):
            g.pie = g.pie or []
            continue
        m = re.match(r'^\s*"([^"]+)"\s*:\s*([-+\d.eE]+)\s*$', line)       # mermaid pie slice
        if m:
            g.pie.append((m.group(1), float(m.group(2))))
            continue
        m = RANGE_RE.match(line)
        if m:
            try:
                lo, hi = eval_number(m.group(2)), eval_number(m.group(3))
                if hi > lo:
                    if m.group(1).lower() == "x":
                        g.xr = (lo, hi)
                        g.pi_axis = bool(re.search(r"pi|π", m.group(2) + m.group(3)))
                    else:
                        g.yr = (lo, hi)
                continue
            except ValueError:
                pass
        _add_curve(g, line)
    return g


def _add_curve(g, line):
    color = None
    cm = COLOR_TAIL.search(line)
    if cm:
        color = cm.group(1) or cm.group(2)
        line = line[:cm.start()]
    line = re.sub(r"^(plot|graph|draw|curve)\s*:?\s*", "", line, flags=re.I).strip().rstrip(",.")
    if not line:
        return
    label = line
    if "=" in line:
        lhs, rhs = line.split("=", 1)
        lhs_s = lhs.strip().replace(" ", "")
        if lhs_s in ("y", "f(x)", "g(x)", "h(x)") or re.fullmatch(r"[a-zA-Z]\(x\)", lhs_s):
            try:
                f = compile_expr(rhs)
                if "y" in f.uses:
                    raise ValueError
                g.curves.append((label, f, False, color))
            except (ValueError, SyntaxError) as e:
                g.errors.append(f"couldn't understand '{line}'")
            return
        try:
            fl, fr = compile_expr(lhs), compile_expr(rhs)
            g.curves.append((label, (lambda x, y, fl=fl, fr=fr: fl(x, y) - fr(x, y)), True, color))
        except (ValueError, SyntaxError) as e:
            g.errors.append(f"couldn't understand '{line}'")
        return
    try:
        f = compile_expr(line)
        if "y" in f.uses:
            raise ValueError("needs '= …'")
        g.curves.append(("y = " + line, f, False, color))
    except (ValueError, SyntaxError) as e:
        g.errors.append(f"couldn't understand '{line}'")


# ---------------------------------------------------------------------------
# axes helpers
# ---------------------------------------------------------------------------
def nice_step(span, target=6):
    raw = span / target
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


def fmt(v):
    if abs(v) < 1e-12:
        return "0"
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.3g}"


def pi_label(v):
    fr = Fraction(v / np.pi).limit_denominator(4)
    n, d = fr.numerator, fr.denominator
    if n == 0:
        return "0"
    sign = "-" if n < 0 else ""
    n = abs(n)
    num = "\\pi" if n == 1 else f"{n}\\pi"
    return f"{sign}{num}" if d == 1 else f"{sign}{num}/{d}"


def _auto_y(g, xs):
    vals = []
    for _, f, implicit, _ in g.curves:
        if not implicit:
            y = f(xs)
            vals.append(y[np.isfinite(y)])
    for x, y, _ in g.points:
        vals.append(np.array([y]))
    for pts, _ in g.polylines:
        vals.append(np.array([p[1] for p in pts]))
    v = np.concatenate(vals) if vals else np.array([])
    if v.size == 0:
        return (-5.0, 5.0)
    lo, hi = np.percentile(v, 2), np.percentile(v, 98)
    lo, hi = min(lo, v.min() if v.size < 50 else lo), max(hi, v.max() if v.size < 50 else hi)
    if hi - lo < 1e-9:
        lo, hi = lo - 1, hi + 1
    span = hi - lo
    if lo > 0 and lo < span * 0.6:
        lo = 0.0
    if hi < 0 and -hi < span * 0.6:
        hi = 0.0
    return (lo - span * 0.08, hi + span * 0.1)


# ---------------------------------------------------------------------------
# graph builder
# ---------------------------------------------------------------------------
def build_graph(ctx, T, text, max_w, max_h, color=None):
    g = parse_graph(text)
    if g.pie:
        return build_pie(ctx, T, g, max_w, max_h, color)
    if g.bars:
        return build_bars(ctx, T, g, max_w, max_h, color)
    if not (g.curves or g.points or g.polylines):
        return None, g.errors or ["nothing to plot"]

    em = ctx.em
    P = Pen(ctx)
    W = min(max_w, em * 24)
    H = min(max_h, W * 0.68)
    title_h = em * 1.6 if g.title else 0
    ml, mr, mt, mb = em * 1.6, em * 1.6, em * 1.0 + title_h, em * 1.5
    pw, ph = W - ml - mr, H - mt - mb

    if g.xr:
        xr = g.xr
    else:
        xs_all = [p[0] for p in g.points] + [p[0] for pl, _ in g.polylines for p in pl]
        if xs_all:
            lo, hi = min(xs_all), max(xs_all)
            pad = max((hi - lo) * 0.12, 0.5)
            xr = (min(lo - pad, 0) if lo > 0 and lo < (hi - lo) else lo - pad, hi + pad)
        else:
            xr = (-5.0, 5.0)
    xs = np.linspace(xr[0], xr[1], 900)
    has_implicit = any(c[2] for c in g.curves)
    if g.yr:
        yr = g.yr
    elif has_implicit and not any(not c[2] for c in g.curves):
        # equal scale on both axes (circles stay round), y covering the same span as x
        half_y = max(abs(xr[0]), abs(xr[1]))
        half_x = half_y * pw / ph
        cx = (xr[0] + xr[1]) / 2
        xr = (cx - half_x, cx + half_x)
        yr = (-half_y, half_y)
    else:
        yr = _auto_y(g, xs)

    X = lambda x: ml + (x - xr[0]) / (xr[1] - xr[0]) * pw
    Y = lambda y: mt + (yr[1] - y) / (yr[1] - yr[0]) * ph
    ax_y = Y(0) if yr[0] <= 0 <= yr[1] else mt + ph            # where the x-axis runs
    ax_x = X(0) if xr[0] <= 0 <= xr[1] else ml                 # where the y-axis runs

    if g.title:
        tb = T.text(g.title, 0.85)
        P.text(tb, (W - tb.w) / 2, em * 0.95, color)

    ext = em * 0.55
    if g.grid:
        for v in np.arange(math.ceil(xr[0] / nice_step(xr[1] - xr[0])) * nice_step(xr[1] - xr[0]), xr[1], nice_step(xr[1] - xr[0])):
            P.line((X(v), mt), (X(v), mt + ph), 0.5, amp=0.3, intensity=0.3, color=color)
        for v in np.arange(math.ceil(yr[0] / nice_step(yr[1] - yr[0])) * nice_step(yr[1] - yr[0]), yr[1], nice_step(yr[1] - yr[0])):
            P.line((ml, Y(v)), (ml + pw, Y(v)), 0.5, amp=0.3, intensity=0.3, color=color)

    # axes with arrow heads
    P.line((ml - ext * 0.4, ax_y), (ml + pw + ext, ax_y), color=color)
    P.arrow_head((ml + pw + ext, ax_y), (1, 0), color=color)
    P.line((ax_x, mt + ph + ext * 0.4), (ax_x, mt - ext), color=color)
    P.arrow_head((ax_x, mt - ext), (0, -1), color=color)
    xl = T.text(g.xlabel or "x", 0.75)
    P.text(xl, ml + pw + ext - xl.w * 0.3, ax_y + em * 0.95, color)
    yl = T.text(g.ylabel or "y", 0.75)
    P.text(yl, ax_x + em * 0.3, mt - ext + em * 0.25, color)

    # ticks
    tick = em * 0.16
    if g.pi_axis:
        step = next(s for s in (np.pi / 4, np.pi / 2, np.pi, 2 * np.pi, 4 * np.pi) if (xr[1] - xr[0]) / s <= 9)
        xt = [(v, T.math(pi_label(v), 0.55)) for v in np.arange(math.ceil(xr[0] / step - 1e-9) * step, xr[1] + 1e-9, step)]
    else:
        step = nice_step(xr[1] - xr[0], 7)
        xt = [(v, T.math(fmt(v), 0.55)) for v in np.arange(math.ceil(xr[0] / step - 1e-9) * step, xr[1] + 1e-9, step)]
    ystep = nice_step(yr[1] - yr[0], 6)
    yt = [(v, T.math(fmt(v), 0.55)) for v in np.arange(math.ceil(yr[0] / ystep - 1e-9) * ystep, yr[1] + 1e-9, ystep)]

    def thin(ticks, spacing):
        widest = max((b.w for _, b in ticks), default=0)
        k = 1
        while ticks and widest + em * 0.25 > spacing * k:
            k += 1
        return ticks[::k] if k > 1 else ticks

    xt = thin(xt, abs(X(step) - X(0)))
    origin_done = False
    for v, b in xt:
        px = X(v)
        if abs(v) < 1e-9 and xr[0] < 0 < xr[1] and yr[0] < 0 < yr[1]:
            if not origin_done:
                ob = T.math("O", 0.55)
                P.text(ob, ax_x - ob.w - em * 0.08, ax_y + em * 0.62, color)
                origin_done = True
            continue
        P.line((px, ax_y - tick), (px, ax_y + tick), 0.8, amp=0.2, color=color)
        P.text(b, px - b.w / 2, ax_y + tick + em * 0.55, color)
    for v, b in yt:
        if abs(v) < 1e-9 and xr[0] < 0 < xr[1]:
            continue
        py = Y(v)
        P.line((ax_x - tick, py), (ax_x + tick, py), 0.8, amp=0.2, color=color)
        P.text(b, ax_x - tick - b.w - em * 0.08, py + em * 0.2, color)

    # curves
    def in_view(y):
        return (y >= yr[0] - (yr[1] - yr[0]) * 0.02) & (y <= yr[1] + (yr[1] - yr[0]) * 0.02)

    label_spots = []
    for label, f, implicit, ccol in g.curves:
        col = ccol or color
        runs = []
        if implicit:
            runs = _implicit_runs(f, xr, yr, X, Y)
        else:
            y = f(xs)
            ok = np.isfinite(y) & in_view(y)
            jump = np.abs(np.diff(y, prepend=y[0])) > (yr[1] - yr[0]) * 0.6
            run = []
            for i in range(len(xs)):
                if ok[i] and not jump[i]:
                    run.append((X(xs[i]), Y(y[i])))
                else:
                    if len(run) > 1:
                        runs.append(run)
                    run = [(X(xs[i]), Y(y[i]))] if ok[i] else []
            if len(run) > 1:
                runs.append(run)
        for r in runs:
            P.smooth(r, 1.05, color=col)
        if runs:
            end = max((p for r in runs for p in r), key=lambda p: p[0] - abs(p[1] - (mt + ph / 2)) * 0.05)
            label_spots.append((label, end, col))

    for pts, ccol in g.polylines:
        P.poly([(X(x), Y(y)) for x, y in pts], amp=ctx.stroke * 0.2, color=ccol or color)
        for x, y in pts:
            P.ellipse(X(x), Y(y), ctx.stroke * 0.9, ctx.stroke * 0.9, color=ccol or color, width=1.3)

    for x, y, _ in g.points:
        if xr[0] <= x <= xr[1] and yr[0] <= y <= yr[1]:
            P.ellipse(X(x), Y(y), ctx.stroke * 1.0, ctx.stroke * 1.0, color=color, width=1.6)
            if len(g.points) <= 6:
                lb = T.math(f"({fmt(x)}, {fmt(y)})", 0.5)
                P.text(lb, min(X(x) + em * 0.25, W - lb.w), Y(y) - em * 0.3, color)

    used = []
    for label, (px, py), col in label_spots:
        try:
            lb = T.math(label, 0.6)
        except Exception:
            lb = T.text(label, 0.6)
        lx = min(px + em * 0.15, W - lb.w)
        ly = min(max(py - em * 0.35, mt + em * 0.5), mt + ph)
        while any(abs(ly - u) < em * 0.75 and abs(lx - ux) < lb.w for ux, u in used):
            ly += em * 0.8
        used.append((lx, ly))
        P.text(lb, lx, ly, col)
    return Figure(W, H, P.ops), g.errors


def _implicit_runs(F, xr, yr, X, Y, n=260):
    try:
        from skimage.measure import find_contours
    except ImportError:
        return []
    xs = np.linspace(xr[0], xr[1], n)
    ys = np.linspace(yr[0], yr[1], n)
    gx, gy = np.meshgrid(xs, ys)
    v = F(gx, gy)
    if not np.isfinite(v).any():
        return []
    v = np.where(np.isfinite(v), v, np.nanmax(np.abs(v[np.isfinite(v)])) + 1)
    runs = []
    for c in find_contours(v, 0.0):
        r, k = c[:, 0], c[:, 1]
        x = xr[0] + k / (n - 1) * (xr[1] - xr[0])
        y = yr[0] + r / (n - 1) * (yr[1] - yr[0])
        if len(x) > 2:
            runs.append([(X(a), Y(b)) for a, b in zip(x, y)])
    return runs


def build_bars(ctx, T, g, max_w, max_h, color):
    em = ctx.em
    P = Pen(ctx)
    n = len(g.bars)
    W = min(max_w, max(em * 12, em * (2.2 * n + 4)), em * 26)
    H = min(max_h, em * 11)
    title_h = em * 1.6 if g.title else 0
    ml, mr, mt, mb = em * 2.2, em * 0.8, em * 1.2 + title_h, em * 1.6
    pw, ph = W - ml - mr, H - mt - mb
    vmax = max(max(v for _, v in g.bars), 0)
    vmin = min(min(v for _, v in g.bars), 0)
    step = nice_step((vmax - vmin) or 1, 5)
    top = math.ceil(vmax / step) * step if vmax > 0 else 0
    bot = math.floor(vmin / step) * step if vmin < 0 else 0
    Y = lambda v: mt + (top - v) / ((top - bot) or 1) * ph
    if g.title:
        tb = T.text(g.title, 0.85)
        P.text(tb, (W - tb.w) / 2, em * 0.95, color)
    P.line((ml, Y(0)), (ml + pw + em * 0.4, Y(0)), color=color)
    P.line((ml, mt + ph + em * 0.1), (ml, mt - em * 0.5), color=color)
    P.arrow_head((ml, mt - em * 0.5), (0, -1), color=color)
    for v in np.arange(bot, top + step / 2, step):
        if abs(v) < 1e-12:
            continue
        b = T.math(fmt(v), 0.55)
        P.line((ml - em * 0.14, Y(v)), (ml + em * 0.14, Y(v)), 0.8, amp=0.2, color=color)
        P.text(b, ml - em * 0.22 - b.w, Y(v) + em * 0.2, color)
    zb = T.math("0", 0.55)
    P.text(zb, ml - em * 0.22 - zb.w, Y(0) + em * 0.2, color)
    slot = pw / n
    bw = slot * 0.58
    lab_size = 0.6
    for i, (label, v) in enumerate(g.bars):
        x0 = ml + slot * i + (slot - bw) / 2
        y0, y1 = sorted((Y(v), Y(0)))
        P.rect(x0, y0, x0 + bw, y1, color=color, overshoot=False)
        if y1 - y0 > em * 0.3:
            P.fill_hatch(x0, y0, x0 + bw, y1, color=color)
        vb = T.math(fmt(v), 0.55)
        P.text(vb, x0 + (bw - vb.w) / 2, y0 - em * 0.2, color)
        lb = T.text(label, lab_size)
        if lb.w > slot * 0.95:                            # shrink rather than break the word
            lb = T.text(label, max(0.32, lab_size * slot * 0.95 / lb.w))
        P.text(lb, ml + slot * i + (slot - lb.w) / 2, Y(0) + em * 0.75 if v >= 0 else Y(0) - em * 0.3, color)
    if g.xlabel:
        xb = T.text(g.xlabel, 0.65)
        P.text(xb, ml + (pw - xb.w) / 2, H - em * 0.05, color)
    return Figure(W, H + (em * 0.4 if g.xlabel else 0), P.ops), g.errors


def build_pie(ctx, T, g, max_w, max_h, color):
    em = ctx.em
    P = Pen(ctx)
    total = sum(max(v, 0) for _, v in g.pie) or 1
    title_h = em * 1.6 if g.title else 0
    W = min(max_w, em * 22)
    R = min(em * 3.6, (max_h - title_h) / 2 - em * 1.2)
    H = title_h + 2 * R + em * 2.4
    cx, cy = W / 2, title_h + em * 1.2 + R
    if g.title:
        tb = T.text(g.title, 0.85)
        P.text(tb, (W - tb.w) / 2, em * 0.95, color)
    P.ellipse(cx, cy, R, R, color=color, width=1.1)
    a = -np.pi / 2
    for label, v in g.pie:
        frac = max(v, 0) / total
        P.line((cx, cy), (cx + R * math.cos(a), cy + R * math.sin(a)), color=color)
        mid = a + frac * np.pi
        lb = T.text(f"{label} ({frac * 100:.0f}%)", 0.6)
        lx = cx + (R + em * 0.35) * math.cos(mid)
        ly = cy + (R + em * 0.35) * math.sin(mid)
        lx = lx if math.cos(mid) >= 0 else lx - lb.w
        lx = min(max(lx, 0), W - lb.w)
        P.text(lb, lx, ly + em * 0.25 * (1 + math.sin(mid)), color)
        a += frac * 2 * np.pi
    return Figure(W, H, P.ops), g.errors


# ---------------------------------------------------------------------------
# flowcharts  (mermaid "graph TD" / "flowchart LR" subset, or "A -> B" lines)
# ---------------------------------------------------------------------------
ARROW_RE = re.compile(
    r"\s*(?:--\s*(?P<l1>[^->|][^>]*?)\s*(?P<a1>-->|---|->)"
    r"|-\.\s*(?P<l3>[^.>|-][^>]*?)\s*\.->"
    r"|(?P<a2><-->|-\.->|-\.-|-->|==>|===|---|<->|->|=>|→|⟶)\s*(?:\|(?P<l2>[^|]*)\|)?)\s*")
SHAPES = [("((", "))", "circle"), ("([", "])", "stadium"), ("[[", "]]", "rect"), ("[(", ")]", "rect"),
          ("{{", "}}", "hexagon"), ("[/", "/]", "para"), ("[\\", "\\]", "para"), ("[/", "\\]", "para"),
          ("[", "]", "rect"), ("(", ")", "round"), ("{", "}", "diamond"), (">", "]", "rect")]


def _node_ref(s):
    s = s.strip().strip(";")
    m = re.match(r"^([\w.\-]+)\s*(.*)$", s, re.S)
    if m and m.group(2):
        rest = m.group(2).strip()
        for o, c, shape in SHAPES:
            if rest.startswith(o) and rest.endswith(c) and len(rest) >= len(o) + len(c):
                text = rest[len(o):len(rest) - len(c)].strip().strip('"')
                return m.group(1), text, shape
    text = s.strip('"')
    return s, text, None


def parse_flow(text):
    direction = "TD"
    nodes, order, edges = {}, [], []

    def node(ref):
        nid, txt, shape = _node_ref(ref)
        if not nid:
            return None
        if nid not in nodes:
            nodes[nid] = {"text": txt or nid, "shape": shape or "rect"}
            order.append(nid)
        elif shape:
            nodes[nid].update(text=txt or nid, shape=shape)
        return nid

    for raw in re.split(r"[\n;]", text):
        line = raw.strip()
        if not line or line.startswith("%%"):
            continue
        m = re.match(r"^(graph|flowchart)\s*(TD|TB|BT|LR|RL)?\s*$", line, re.I)
        if m:
            direction = "LR" if (m.group(2) or "TD").upper() in ("LR", "RL") else "TD"
            continue
        if re.match(r"^(direction|classDef|class|style|linkStyle|click|subgraph|end)\b", line):
            if line.startswith("direction"):
                direction = "LR" if re.search(r"LR|RL", line) else "TD"
            continue
        line = re.sub(r":::\w+", "", line)
        pos, prev_ids, pending = 0, None, None
        matches = list(ARROW_RE.finditer(line))
        if not matches:
            for part in line.split("&"):
                node(part)
            continue
        segs = []
        for mm in matches:
            segs.append(line[pos:mm.start()])
            segs.append(mm)
            pos = mm.end()
        segs.append(line[pos:])
        for s in segs:
            if isinstance(s, str):
                ids = [node(p) for p in s.split("&") if p.strip()]
                ids = [i for i in ids if i]
                if prev_ids and pending is not None and ids:
                    for a in prev_ids:
                        for b in ids:
                            edges.append(dict(a=a, b=b, **pending))
                prev_ids, pending = ids, None
            else:
                arrow = s.group("a1") or s.group("a2") or ".->"
                label = (s.group("l1") or s.group("l2") or s.group("l3") or "").strip().strip('"')
                pending = dict(label=label, directed=">" in arrow or arrow in ("→", "⟶"),
                               both=arrow.startswith("<"), dashed="." in arrow,
                               thick=arrow.startswith("="))
    for n in nodes.values():
        n["text"] = re.sub(r"<br\s*/?>", "\n", n["text"])
    return direction, nodes, order, edges


def _ranks(order, edges):
    out = {n: [] for n in order}
    for i, e in enumerate(edges):
        if e["a"] != e["b"]:
            out[e["a"]].append(i)
    state, back = {}, set()
    for root in order:
        if state.get(root):
            continue
        stack = [(root, iter(out[root]))]
        state[root] = 1
        while stack:
            u, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                state[u] = 2
                stack.pop()
                continue
            v = edges[nxt]["b"]
            if state.get(v) == 1:
                back.add(nxt)
            elif not state.get(v):
                state[v] = 1
                stack.append((v, iter(out[v])))
    rank = {n: 0 for n in order}
    for _ in range(len(order)):
        changed = False
        for i, e in enumerate(edges):
            if i in back or e["a"] == e["b"]:
                continue
            if rank[e["b"]] < rank[e["a"]] + 1:
                rank[e["b"]] = rank[e["a"]] + 1
                changed = True
        if not changed:
            break
    return rank, back


def _layout_flow(ctx, T, direction, nodes, order, edges, size):
    em = ctx.em * size
    boxes = {}
    for nid in order:
        n = nodes[nid]
        lines = []
        for para in n["text"].split("\n"):
            lines += T.lines(para, size, em * 8.5)
        tw = max((b.w for b in lines), default=em)
        lh = em * 1.35
        th = lh * max(1, len(lines))
        shape = n["shape"]
        w, h = tw + em * 1.0, th + em * 0.7
        if shape == "diamond":
            w, h = tw * 1.45 + em * 1.2, th * 1.5 + em * 1.0
        elif shape == "circle":
            w = h = max(w, h) * 1.12
        elif shape in ("stadium", "hexagon", "para"):
            w += em * 0.8
        boxes[nid] = dict(lines=lines, w=w, h=h, tw=tw, th=th, lh=lh, shape=shape)

    rank, back = _ranks(order, edges)
    nr = max(rank.values(), default=0) + 1
    layers = [[n for n in order if rank[n] == r] for r in range(nr)]
    pos = {n: i for layer in layers for i, n in enumerate(layer)}
    preds = {n: [] for n in order}
    succs = {n: [] for n in order}
    for i, e in enumerate(edges):
        if i not in back and e["a"] != e["b"]:
            preds[e["b"]].append(e["a"])
            succs[e["a"]].append(e["b"])
    for sweep in range(4):
        rng_layers = range(1, nr) if sweep % 2 == 0 else range(nr - 2, -1, -1)
        for r in rng_layers:
            nb = preds if sweep % 2 == 0 else succs
            layers[r].sort(key=lambda n: (np.mean([pos[u] for u in nb[n]]) if nb[n] else pos[n]))
            for i, n in enumerate(layers[r]):
                pos[n] = i

    has_labels = any(e["label"] for e in edges)
    along_gap = em * (2.4 if has_labels else 1.7)
    across_gap = em * 1.3
    centers = {}
    if direction == "TD":
        y = 0.0
        widths = [sum(boxes[n]["w"] for n in L) + across_gap * (len(L) - 1) for L in layers]
        total_w = max(widths, default=0)
        for L, lw in zip(layers, widths):
            hmax = max(boxes[n]["h"] for n in L)
            x = (total_w - lw) / 2
            for n in L:
                centers[n] = (x + boxes[n]["w"] / 2, y + hmax / 2)
                x += boxes[n]["w"] + across_gap
            y += hmax + along_gap
        total_h = y - along_gap
    else:
        x = 0.0
        heights = [sum(boxes[n]["h"] for n in L) + across_gap * (len(L) - 1) for L in layers]
        total_h = max(heights, default=0)
        for L, lh in zip(layers, heights):
            wmax = max(boxes[n]["w"] for n in L)
            y = (total_h - lh) / 2
            for n in L:
                centers[n] = (x + wmax / 2, y + boxes[n]["h"] / 2)
                y += boxes[n]["h"] + across_gap
            x += wmax + along_gap * 1.2
        total_w = x - along_gap * 1.2
    # room for back edges that loop around the side
    side = em * 1.6 if back else 0
    return boxes, centers, total_w + side + em * 0.4, total_h + em * 0.4, rank, back


def _boundary(b, c, d):
    """Distance from centre c along unit direction d to the shape edge."""
    hw, hh = b["w"] / 2, b["h"] / 2
    dx, dy = abs(d[0]) + 1e-9, abs(d[1]) + 1e-9
    if b["shape"] == "circle":
        return 1 / math.sqrt((dx / hw) ** 2 + (dy / hh) ** 2)
    if b["shape"] == "diamond":
        return 1 / (dx / hw + dy / hh)
    return min(hw / dx, hh / dy)


def build_flow(ctx, T, text, max_w, max_h, color=None):
    direction, nodes, order, edges = parse_flow(text)
    if not order:
        return None, ["no boxes or arrows found"]
    size = 0.85
    for attempt in range(2):
        tries = [direction] if direction == "TD" else ["LR", "TD"]
        best = None
        for d in tries:
            lay = _layout_flow(ctx, T, d, nodes, order, edges, size)
            scale = min(1.0, max_w / lay[2], max_h / lay[3])
            if best is None or scale > best[0] + 0.05:
                best = (scale, d, lay)
        scale, direction, lay = best
        if scale >= 0.98 or attempt == 1:
            break
        size = max(0.45, size * scale * 0.98)
    boxes, centers, W, H, rank, back = lay
    em = ctx.em * size
    P = Pen(ctx)
    off = em * 0.2

    for nid in order:
        b = boxes[nid]
        cx, cy = centers[nid]
        cx += off
        cy += off
        x0, y0, x1, y1 = cx - b["w"] / 2, cy - b["h"] / 2, cx + b["w"] / 2, cy + b["h"] / 2
        s = b["shape"]
        if s == "diamond":
            P.poly([(cx, y0), (x1, cy), (cx, y1), (x0, cy)], closed=True, color=color)
        elif s == "circle":
            P.ellipse(cx, cy, b["w"] / 2, b["h"] / 2, color=color)
        elif s == "round":
            P.round_rect(x0, y0, x1, y1, em * 0.35, color=color)
        elif s == "stadium":
            P.round_rect(x0, y0, x1, y1, b["h"] / 2, color=color)
        elif s == "hexagon":
            k = em * 0.5
            P.poly([(x0 + k, y0), (x1 - k, y0), (x1, cy), (x1 - k, y1), (x0 + k, y1), (x0, cy)], closed=True, color=color)
        elif s == "para":
            k = em * 0.4
            P.poly([(x0 + k, y0), (x1, y0), (x1 - k, y1), (x0, y1)], closed=True, color=color)
        else:
            P.rect(x0, y0, x1, y1, color=color)
        ty = cy - b["th"] / 2
        for i, lb in enumerate(b["lines"]):
            P.text(lb, cx - lb.w / 2, ty + b["lh"] * i + em * 1.0, color)

    for i, e in enumerate(edges):
        if e["a"] == e["b"]:
            continue
        ba, bb = boxes[e["a"]], boxes[e["b"]]
        ca, cb = _v(centers[e["a"]]) + off, _v(centers[e["b"]]) + off
        gap = ctx.stroke * 1.5
        is_back = i in back or (rank[e["b"]] <= rank[e["a"]] and direction == "TD" and abs(ca[0] - cb[0]) < em)
        if is_back:
            # loop round the right-hand side
            if direction == "TD":
                p0 = ca + [ba["w"] / 2 + gap, 0]
                p1 = cb + [bb["w"] / 2 + gap, 0]
                xr = max(p0[0], p1[0]) + em * 1.2
                pts = [p0, (xr, p0[1]), (xr, p1[1]), p1]
                tip_dir = (-1, 0)
            else:
                p0 = ca + [0, ba["h"] / 2 + gap]
                p1 = cb + [0, bb["h"] / 2 + gap]
                yb = max(p0[1], p1[1]) + em * 1.0
                pts = [p0, (p0[0], yb), (p1[0], yb), p1]
                tip_dir = (0, -1)
            P.poly(pts, color=color, width=1.6 if e["thick"] else 1.0)
            if e["directed"]:
                P.arrow_head(p1, tip_dir, size=em * 0.3, color=color)
            mid = _v(pts[1]) * 0.5 + _v(pts[2]) * 0.5
        else:
            d = cb - ca
            n = np.hypot(*d)
            if n < 1:
                continue
            d = d / n
            p0 = ca + d * (_boundary(ba, ca, d) + gap)
            p1 = cb - d * (_boundary(bb, cb, -d) + gap)
            if e["dashed"]:
                P.dashed(p0, p1, color=color)
            else:
                P.line(p0, p1, 1.6 if e["thick"] else 1.0, color=color)
            if e["directed"]:
                P.arrow_head(p1, d, size=em * 0.3, color=color)
            if e["both"]:
                P.arrow_head(p0, -d, size=em * 0.3, color=color)
            mid = (p0 + p1) / 2
            tip_dir = d
        if e["label"]:
            lb = T.text(e["label"], size * 0.75)
            if abs(tip_dir[1]) > abs(tip_dir[0]):          # vertical edge: label to the right
                P.text(lb, mid[0] + em * 0.25, mid[1] + em * 0.25, color)
            else:                                          # horizontal edge: label above
                P.text(lb, mid[0] - lb.w / 2, mid[1] - em * 0.25, color)
    fig = Figure(W + off, H + off, P.ops)
    if fig.w > max_w * 1.02 or fig.a > max_h * 1.02:
        return None, ["diagram too large for the page"]
    return fig, []
