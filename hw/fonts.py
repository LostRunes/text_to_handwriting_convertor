"""Locating system fonts (for template labels and for characters the user
never wrote) and checking which characters each font covers."""

import os
from functools import lru_cache

from fontTools.ttLib import TTFont
from PIL import ImageFont


def _font_dirs():
    dirs = []
    windir = os.environ.get("WINDIR")
    if windir:
        dirs.append(os.path.join(windir, "Fonts"))
        local = os.environ.get("LOCALAPPDATA")
        if local:
            dirs.append(os.path.join(local, "Microsoft", "Windows", "Fonts"))
    dirs += ["/usr/share/fonts", "/usr/local/share/fonts", os.path.expanduser("~/.fonts"),
             "/Library/Fonts", "/System/Library/Fonts", os.path.expanduser("~/Library/Fonts")]
    try:
        import matplotlib
        dirs.append(os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf"))
    except ImportError:
        pass
    return [d for d in dirs if os.path.isdir(d)]


@lru_cache(maxsize=None)
def find_font(*names):
    """First existing font file among `names` (case-insensitive basenames)."""
    wanted = [n.lower() for n in names]
    found = {}
    for d in _font_dirs():
        for root, _, files in os.walk(d):
            for f in files:
                lf = f.lower()
                if lf in wanted and lf not in found:
                    found[lf] = os.path.join(root, f)
    for n in wanted:
        if n in found:
            return found[n]
    return None


@lru_cache(maxsize=None)
def coverage(path):
    try:
        return frozenset(TTFont(path, fontNumber=0, lazy=True).getBestCmap().keys())
    except Exception:
        return frozenset()


@lru_cache(maxsize=None)
def pil_font(path, size):
    return ImageFont.truetype(path, size)


# Handwriting-looking fonts first, then broad-coverage symbol fonts.
HAND_FONTS = ("segoepr.ttf", "Inkfree.ttf", "comic.ttf")
SYMBOL_FONTS = ("seguisym.ttf", "DejaVuSans.ttf", "cambria.ttc", "arial.ttf",
                "NotoSansMath-Regular.ttf", "NotoSans-Regular.ttf", "LiberationSans-Regular.ttf")


def font_for_char(ch, prefer=HAND_FONTS + SYMBOL_FONTS):
    for name in prefer:
        path = find_font(name)
        if path and ord(ch) in coverage(path):
            return path
    return None


def label_font(size):
    path = find_font("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf")
    return pil_font(path, size) if path else ImageFont.load_default()
