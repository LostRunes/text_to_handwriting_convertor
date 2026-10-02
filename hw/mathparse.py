"""A forgiving LaTeX-math parser producing a small node tree.

Nodes are tuples:
  ('row', [nodes])              ('sym', ch)            ('op', ch)   binary op
  ('rel', ch)                   ('punct', ch)          ('text', str)
  ('func', name, limits)        ('bigop', ch)          ('space', em)
  ('frac', num, den, bar)       ('scripts', base, sup, sub)
  ('sqrt', body, index)         ('accent', kind, body) ('delim', l, body, r)
  ('matrix', rows, l, r, align) ('table', rows)        ('boxed', body)
  ('newline',)                  ('phantom', body)      ('big', ch, factor)
It never raises on bad input: unknown commands render as their name.
"""

GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "varepsilon": "ε",
    "zeta": "ζ", "eta": "η", "theta": "θ", "vartheta": "θ", "iota": "ι", "kappa": "κ",
    "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "pi": "π", "varpi": "π", "rho": "ρ",
    "varrho": "ρ", "sigma": "σ", "varsigma": "ς", "tau": "τ", "upsilon": "υ", "phi": "φ",
    "varphi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ", "Pi": "Π",
    "Sigma": "Σ", "Upsilon": "Υ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
}

BINOPS = {
    "times": "×", "div": "÷", "pm": "±", "mp": "∓", "cdot": "·", "ast": "∗", "star": "⋆",
    "circ": "∘", "bullet": "·", "cup": "∪", "cap": "∩", "setminus": "∖", "oplus": "⊕",
    "otimes": "⊗", "wedge": "∧", "land": "∧", "vee": "∨", "lor": "∨",
}

RELS = {
    "leq": "≤", "le": "≤", "geq": "≥", "ge": "≥", "neq": "≠", "ne": "≠", "approx": "≈",
    "equiv": "≡", "sim": "∼", "simeq": "≃", "cong": "≅", "propto": "∝", "in": "∈",
    "notin": "∉", "ni": "∋", "subset": "⊂", "subseteq": "⊆", "supset": "⊃", "supseteq": "⊇",
    "to": "→", "rightarrow": "→", "leftarrow": "←", "gets": "←", "leftrightarrow": "↔",
    "Rightarrow": "⇒", "implies": "⇒", "Leftarrow": "⇐", "Leftrightarrow": "⇔", "iff": "⇔",
    "mapsto": "↦", "longrightarrow": "→", "Longrightarrow": "⇒", "longleftrightarrow": "↔",
    "ll": "≪", "gg": "≫", "perp": "⊥", "parallel": "∥", "mid": "|", "leqslant": "≤",
    "geqslant": "≥", "coloneqq": "≔", "uparrow": "↑", "downarrow": "↓",
}

SYMBOLS = {
    "infty": "∞", "partial": "∂", "nabla": "∇", "forall": "∀", "exists": "∃", "emptyset": "∅",
    "varnothing": "∅", "angle": "∠", "degree": "°", "circ": "∘", "therefore": "∴",
    "because": "∵", "ldots": "…", "cdots": "⋯", "dots": "…", "vdots": "⋮", "ddots": "⋱",
    "prime": "′", "hbar": "ℏ", "ell": "ℓ", "neg": "¬", "lnot": "¬", "top": "⊤", "bot": "⊥",
    "triangle": "△", "square": "□", "checkmark": "✓", "aleph": "ℵ", "Re": "ℜ", "Im": "ℑ",
    "lbrace": "{", "rbrace": "}", "langle": "⟨", "rangle": "⟩", "vert": "|", "Vert": "‖",
    "lvert": "|", "rvert": "|", "lVert": "‖", "rVert": "‖", "lfloor": "⌊", "rfloor": "⌋",
    "lceil": "⌈", "rceil": "⌉", "backslash": "\\", "percent": "%", "dagger": "†",
}

BIGOPS = {"sum": "∑", "prod": "∏", "coprod": "∐", "int": "∫", "iint": "∬", "iiint": "∭",
          "oint": "∮", "bigcup": "⋃", "bigcap": "⋂", "bigoplus": "⊕", "bigotimes": "⊗"}

FUNCS = {"sin", "cos", "tan", "cot", "sec", "csc", "arcsin", "arccos", "arctan", "sinh",
         "cosh", "tanh", "coth", "log", "ln", "lg", "exp", "det", "dim", "ker", "deg", "gcd",
         "lcm", "arg", "hom", "Pr", "mod", "bmod", "tr", "rank", "sgn"}
LIMFUNCS = {"lim", "max", "min", "sup", "inf", "limsup", "liminf", "argmax", "argmin"}

SPACES = {",": 0.17, ":": 0.22, ">": 0.22, ";": 0.28, "!": -0.1, " ": 0.3, "quad": 1.0,
          "qquad": 2.0, "enspace": 0.5, "thinspace": 0.17, "medspace": 0.22,
          "thickspace": 0.28, "enskip": 0.5, "hfill": 1.0}

ACCENTS = {"vec": "vec", "overrightarrow": "vec", "hat": "hat", "widehat": "hat",
           "bar": "bar", "overline": "bar", "underline": "under", "dot": "dot",
           "ddot": "ddot", "tilde": "tilde", "widetilde": "tilde", "cancel": "cancel",
           "bcancel": "cancel", "xcancel": "cancel", "check": "hat", "breve": "tilde",
           "overleftarrow": "lvec", "underbrace": "under", "overbrace": "bar"}

TEXTCMDS = {"text", "textrm", "textit", "textbf", "textsf", "texttt", "mbox", "hbox",
            "textnormal", "textup", "emph"}
FONTCMDS = {"mathrm", "mathbf", "mathit", "mathsf", "mathtt", "mathcal", "mathbb",
            "mathfrak", "mathscr", "boldsymbol", "bm", "mathnormal", "pmb", "bf", "rm", "it"}
IGNORE = {"displaystyle", "textstyle", "scriptstyle", "scriptscriptstyle", "limits",
          "nolimits", "left.", "right.", "nonumber", "notag", "label", "tag", "centering",
          "small", "large", "Large", "normalsize", "strut", "mathstrut", "relax", "allowbreak"}
BIGSIZES = {"big": 1.25, "Big": 1.6, "bigg": 2.0, "Bigg": 2.4}

REL_CHARS = set("=<>≤≥≠≈≡∼≃≅∝∈∉∋⊂⊆⊃⊇→←↔⇒⇐⇔↦≪≫⊥∥≔:")
BIN_CHARS = set("+-−×÷±∓·∗⋆∘∪∩∖⊕⊗∧∨*")

MATRIX_ENVS = {"matrix": ("", ""), "pmatrix": ("(", ")"), "bmatrix": ("[", "]"),
               "Bmatrix": ("{", "}"), "vmatrix": ("|", "|"), "Vmatrix": ("‖", "‖"),
               "smallmatrix": ("", ""), "array": ("", ""), "cases": ("{", ""),
               "dcases": ("{", ""), "rcases": ("", "}")}
ALIGN_ENVS = {"aligned", "align", "align*", "alignat", "alignat*", "split", "gathered",
              "gather", "gather*", "equation", "equation*", "eqnarray", "eqnarray*",
              "multline", "multline*", "flalign", "flalign*", "subequations"}


class MathParser:
    def __init__(self, s):
        self.s = s
        self.i = 0

    # -- low level ---------------------------------------------------------
    def peek(self):
        return self.s[self.i] if self.i < len(self.s) else ""

    def skip_ws(self):
        while self.i < len(self.s) and self.s[self.i].isspace():
            self.i += 1

    def at_cmd(self, name):
        j = self.i + 1 + len(name)
        return (self.s.startswith("\\" + name, self.i)
                and (j >= len(self.s) or not self.s[j].isalpha()))

    def read_cmd(self):
        """Assumes s[i] == '\\'. Returns the command name."""
        j = self.i + 1
        if j < len(self.s) and self.s[j].isalpha():
            while j < len(self.s) and self.s[j].isalpha():
                j += 1
            name = self.s[self.i + 1:j]
            if j < len(self.s) and self.s[j] == "*" and name in {"operatorname"}:
                j += 1
        else:
            name = self.s[j:j + 1]
            j += 1
        self.i = j
        return name

    def read_raw_group(self):
        self.skip_ws()
        if self.peek() != "{":
            # single token
            if self.peek() == "\\":
                return "\\" + self.read_cmd()
            ch = self.peek()
            self.i += 1
            return ch
        depth, j = 0, self.i
        while j < len(self.s):
            if self.s[j] == "\\":
                j += 2
                continue
            if self.s[j] == "{":
                depth += 1
            elif self.s[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        raw = self.s[self.i + 1:j]
        self.i = j + 1
        return raw

    def read_optional(self):
        self.skip_ws()
        if self.peek() != "[":
            return None
        self.i += 1
        node = self.parse_row(stops={"]"})
        if self.peek() == "]":
            self.i += 1
        return node

    def read_delim(self):
        self.skip_ws()
        c = self.peek()
        if c == "\\":
            name = self.read_cmd()
            if name in ("{", "}", "|"):
                return {"|": "‖"}.get(name, name)
            return SYMBOLS.get(name, RELS.get(name, ""))
        self.i += 1
        return "" if c == "." else c

    # -- grammar -----------------------------------------------------------
    def at_stop(self, stops):
        c = self.peek()
        if not c:
            return True
        if c in stops and c in "}]&":
            return True
        if "\\\\" in stops and self.s.startswith("\\\\", self.i):
            return True
        if "right" in stops and self.at_cmd("right"):
            return True
        if "end" in stops and self.at_cmd("end"):
            return True
        return False

    def parse(self):
        return self.parse_row(stops=())

    def parse_row(self, stops):
        items = []
        while True:
            self.skip_ws()
            if self.at_stop(stops):
                break
            before = self.i
            atom = self.parse_atom(stops)
            if atom is None:
                if self.i == before:      # safety: always make progress
                    self.i += 1
                continue
            atom = self.parse_scripts(atom)
            items.append(atom)
        return ("row", items)

    def parse_scripts(self, base):
        sup = sub = None
        while True:
            self.skip_ws()
            c = self.peek()
            if c == "^":
                self.i += 1
                arg = self.parse_arg()
                sup = arg if sup is None else ("row", [sup, arg])
            elif c == "_":
                self.i += 1
                sub = self.parse_arg()
            elif c == "'":
                n = 0
                while self.peek() == "'":
                    n += 1
                    self.i += 1
                p = ("row", [("sym", "′")] * n)
                sup = p if sup is None else ("row", [p, sup])
            else:
                break
        if sup is None and sub is None:
            return base
        return ("scripts", base, sup, sub)

    def parse_arg(self):
        self.skip_ws()
        c = self.peek()
        if c == "{":
            self.i += 1
            r = self.parse_row(stops={"}"})
            if self.peek() == "}":
                self.i += 1
            return r
        if c == "\\":
            node = self.parse_atom(stops=())
            return node if node is not None else ("row", [])
        if not c:
            return ("row", [])
        self.i += 1
        return self.char_node(c)

    @staticmethod
    def char_node(c):
        if c in REL_CHARS:
            return ("rel", c)
        if c in BIN_CHARS:
            return ("op", c)
        if c in ",;":
            return ("punct", c)
        return ("sym", c)

    def parse_atom(self, stops):
        c = self.peek()
        if c == "{":
            self.i += 1
            r = self.parse_row(stops={"}"})
            if self.peek() == "}":
                self.i += 1
            return r
        if c in "}]&":
            self.i += 1
            return None
        if c in "^_":
            return ("row", [])           # script with empty base; parse_scripts picks it up
        if c == "~":
            self.i += 1
            return ("space", 0.3)
        if c == "\\":
            if self.s.startswith("\\\\", self.i):
                self.i += 2
                self.read_optional()
                return ("newline",)
            return self.parse_command(self.read_cmd())
        self.i += 1
        return self.char_node(c)

    def parse_command(self, name):
        if not name:
            return None
        if name in ("frac", "dfrac", "tfrac", "cfrac"):
            return ("frac", self.parse_arg(), self.parse_arg(), True)
        if name in ("binom", "dbinom", "tbinom", "choose"):
            return ("delim", "(", ("frac", self.parse_arg(), self.parse_arg(), False), ")")
        if name == "sqrt":
            idx = self.read_optional()
            return ("sqrt", self.parse_arg(), idx)
        if name == "left":
            l = self.read_delim()
            body = self.parse_row(stops={"right"})
            r = ""
            if self.at_cmd("right"):
                self.read_cmd()
                r = self.read_delim()
            return ("delim", l, body, r)
        if name == "right":
            self.read_delim()
            return None
        if name in BIGSIZES or name[:-1] in BIGSIZES and name[-1] in "lrm":
            factor = BIGSIZES.get(name, BIGSIZES.get(name[:-1], 1.3))
            d = self.read_delim()
            return ("big", d, factor) if d else None
        if name in TEXTCMDS:
            return ("text", self.read_raw_group())
        if name == "operatorname":
            return ("func", self.read_raw_group(), False)
        if name in FONTCMDS:
            return self.parse_arg()
        if name == "begin":
            return self.parse_env(self.read_raw_group())
        if name == "end":
            self.read_raw_group()
            return None
        if name in ACCENTS:
            return ("accent", ACCENTS[name], self.parse_arg())
        if name in ("boxed", "fbox", "framebox"):
            if name == "fbox":
                return ("boxed", ("text", self.read_raw_group()))
            return ("boxed", self.parse_arg())
        if name in ("color",):
            self.read_raw_group()
            return None
        if name in ("textcolor", "colorbox"):
            self.read_raw_group()
            return self.parse_arg()
        if name in ("phantom", "hphantom", "vphantom"):
            return ("phantom", self.parse_arg())
        if name in ("label", "tag", "hspace", "vspace", "kern", "mkern"):
            self.read_raw_group()
            return ("space", 0.3) if name in ("hspace", "kern", "mkern") else None
        if name == "not":
            nxt = self.parse_atom(stops=())
            neg = {"=": "≠", "∈": "∉", "<": "≮", ">": "≯", "≡": "≢", "⊂": "⊄", "⊆": "⊈"}
            if nxt and nxt[0] in ("rel", "sym") and nxt[1] in neg:
                return ("rel", neg[nxt[1]])
            return ("accent", "cancel", nxt) if nxt else None
        if name in ("pmod",):
            return ("row", [("space", 0.5), ("sym", "("), ("func", "mod", False), ("space", 0.2),
                            self.parse_arg(), ("sym", ")")])
        if name in SPACES:
            return ("space", SPACES[name])
        if name in IGNORE:
            return None
        if name in GREEK:
            return ("sym", GREEK[name])
        if name in BIGOPS:
            return ("bigop", BIGOPS[name])
        if name in LIMFUNCS:
            return ("func", name, True)
        if name in FUNCS:
            return ("func", name, False)
        if name in BINOPS:
            return ("op", BINOPS[name])
        if name in RELS:
            return ("rel", RELS[name])
        if name in SYMBOLS:
            return ("sym", SYMBOLS[name])
        if name in "{}_%$&#|":
            return ("sym", {"|": "‖"}.get(name, name))
        return ("text", name)

    def parse_env(self, env):
        env = env.strip()
        if env == "array" or env.startswith("alignat"):
            self.read_raw_group()                     # column spec / column count
        if env in MATRIX_ENVS:
            rows = self.parse_table()
            l, r = MATRIX_ENVS[env]
            align = "l" if "cases" in env else "c"
            return ("matrix", rows, l, r, align)
        if env in ALIGN_ENVS:
            return ("table", self.parse_table())
        # unknown environment: just parse its content
        return ("table", self.parse_table())

    def parse_table(self):
        rows, row = [], []
        while True:
            cell = self.parse_row(stops={"&", "\\\\", "end"})
            row.append(cell)
            if self.peek() == "&":
                self.i += 1
                continue
            if self.s.startswith("\\\\", self.i):
                self.i += 2
                self.read_optional()
                rows.append(row)
                row = []
                continue
            if self.at_cmd("end"):
                self.read_cmd()
                self.read_raw_group()
            rows.append(row)
            break
        # drop a trailing empty row (from a final \\)
        if rows and all(not c[1] for c in rows[-1]) and len(rows) > 1:
            rows.pop()
        return rows


def parse_math(s):
    """Parse LaTeX math. A top-level `\\\\` turns the result into a table of rows."""
    node = MathParser(s).parse()
    items = node[1]
    if any(n[0] == "newline" for n in items):
        rows, cur = [], []
        for n in items:
            if n[0] == "newline":
                rows.append([("row", cur)])
                cur = []
            else:
                cur.append(n)
        if cur:
            rows.append([("row", cur)])
        return ("table", rows)
    return node
