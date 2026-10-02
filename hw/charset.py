"""Characters collected by the template sheet, and the sheet geometry.

All geometry is in pixels of the canonical template image (A4 at 200 DPI).
Extraction warps every uploaded photo back onto this exact geometry, so the
template and the extractor must agree on these numbers.
"""

UPPER = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
LOWER = list("abcdefghijklmnopqrstuvwxyz")
DIGITS = list("0123456789")
PUNCT = list(".,;:!?'\"()[]{}-_/\\|@#%&*+=<>^~$")
MATH = list("×÷±∓·≠≈≡≤≥∞√∫∮∑∏∂∇→←↔⇒⇔∈∉⊂⊆∪∩∀∃∅∠°′∴∝")
GREEK = list("αβγδεζηθλμνξπρστφχψωΓΔΘΛΣΦΨΩ")

CHARSET = UPPER + LOWER + DIGITS + PUNCT + MATH + GREEK

# --- page geometry (A4 @ 200 DPI) ---------------------------------------
DPI = 200
PAGE_W, PAGE_H = 1654, 2339

MARKER = 70            # side of the solid corner squares
MARKER_MARGIN = 50     # distance of a marker from the page edge
MARKER_CENTERS = [
    (MARKER_MARGIN + MARKER / 2, MARKER_MARGIN + MARKER / 2),                    # top-left
    (PAGE_W - MARKER_MARGIN - MARKER / 2, MARKER_MARGIN + MARKER / 2),           # top-right
    (PAGE_W - MARKER_MARGIN - MARKER / 2, PAGE_H - MARKER_MARGIN - MARKER / 2),  # bottom-right
    (MARKER_MARGIN + MARKER / 2, PAGE_H - MARKER_MARGIN - MARKER / 2),           # bottom-left
]

# page-id bits: filled square = 1, outlined square = 0
ID_BITS = 4
ID_SIZE = 28
ID_GAP = 16
ID_Y = 62
ID_X0 = PAGE_W // 2 - (ID_BITS * ID_SIZE + (ID_BITS - 1) * ID_GAP) // 2

COLS, ROWS = 8, 10
PER_PAGE = COLS * ROWS
GRID_X, GRID_Y = 110, 250
CELL_W, CELL_H = 179, 190

LABEL_H = 28           # strip at the top of each cell holding the printed label
CAP_Y = 55             # guide lines, measured from the cell top
XHEIGHT_Y = 95
BASELINE_Y = 135
DESC_Y = 175
EM = BASELINE_Y - CAP_Y  # cap height in template pixels (80)


def pages_for(copies=1):
    """Char list for each template page. copies>1 repeats the set so the
    writer gives several variants of every character."""
    base = [CHARSET[i:i + PER_PAGE] for i in range(0, len(CHARSET), PER_PAGE)]
    return base * max(1, copies)


def chars_for_page(page_id):
    base = pages_for(1)
    return base[page_id % len(base)]


def cell_origin(index):
    r, c = divmod(index, COLS)
    return GRID_X + c * CELL_W, GRID_Y + r * CELL_H
