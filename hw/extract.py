"""Turn photos/scans of a filled-in template into a glyph library (profile).

Pipeline per image: find the four corner squares -> perspective-warp to the
canonical template -> read the page-id bits -> flatten lighting -> cut every
cell, isolate the ink, and store it as an alpha mask positioned relative to
the cell's baseline.
"""

import json
import os

import cv2
import numpy as np

from . import charset as C


class ExtractError(Exception):
    pass


def _load(src):
    if isinstance(src, np.ndarray):
        return src
    if isinstance(src, (bytes, bytearray)):
        img = cv2.imdecode(np.frombuffer(src, np.uint8), cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        raise ExtractError("Could not read image")
    return img


def load_images(src, filename=""):
    """An upload may be an image or a (scanned) PDF with several pages."""
    if isinstance(src, (bytes, bytearray)) and (filename.lower().endswith(".pdf") or src[:4] == b"%PDF"):
        try:
            import fitz
        except ImportError as e:
            raise ExtractError("PDF upload needs PyMuPDF (pip install pymupdf)") from e
        out = []
        with fitz.open(stream=bytes(src), filetype="pdf") as doc:
            for page in doc:
                pix = page.get_pixmap(dpi=C.DPI)
                arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)
                out.append(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR if pix.n == 3 else cv2.COLOR_RGBA2BGR))
        return out
    return [_load(src)]


def find_markers(gray):
    """Centres of the four corner squares, ordered TL, TR, BR, BL."""
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    min_area = (min(h, w) * 0.012) ** 2
    cands = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.06 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        x, y, bw, bh = cv2.boundingRect(approx)
        if not 0.6 < bw / bh < 1.6 or area / (bw * bh) < 0.75:
            continue
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [c], -1, 255, -1)
        if cv2.mean(th, mask=mask)[0] < 200:      # must be solid, not an outline
            continue
        m = cv2.moments(c)
        cands.append((area, m["m10"] / m["m00"], m["m01"] / m["m00"]))
    if len(cands) < 4:
        raise ExtractError("Could not find the four black corner squares. "
                           "Make sure the whole sheet is in the photo and well lit.")
    biggest = max(a for a, _, _ in cands)
    cands = [(x, y) for a, x, y in cands if a > 0.35 * biggest]
    corners = [(0, 0), (w, 0), (w, h), (0, h)]
    picked = []
    for cx, cy in corners:
        best = min(cands, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
        picked.append(best)
    if len(set(picked)) < 4:
        raise ExtractError("Corner squares are ambiguous - retake the photo straight on.")
    return np.float32(picked)


def rectify(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    src = find_markers(gray)
    dst = np.float32(C.MARKER_CENTERS)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(gray, M, (C.PAGE_W, C.PAGE_H), flags=cv2.INTER_CUBIC,
                               borderValue=255)


def flatten_light(gray):
    """Divide by an estimate of the paper brightness -> 1.0 paper, ~0 ink."""
    g = gray.astype(np.float32)
    bg = cv2.dilate(g, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
    bg = cv2.GaussianBlur(bg, (0, 0), 15)
    return np.clip(g / np.maximum(bg, 1), 0, 1)


def read_page_id(norm):
    pid = 0
    for b in range(C.ID_BITS):
        x = C.ID_X0 + b * (C.ID_SIZE + C.ID_GAP)
        q = C.ID_SIZE // 4
        patch = norm[C.ID_Y + q:C.ID_Y + C.ID_SIZE - q, x + q:x + C.ID_SIZE - q]
        if patch.mean() < 0.5:
            pid |= 1 << b
    return pid


INK_THRESHOLD = 0.62


def cut_glyph(norm, index):
    """Return (alpha float32 HxW, top_rel_baseline) or None for an empty box."""
    x, y = C.cell_origin(index)
    inset = 6
    y0, y1 = y + C.LABEL_H + 3, y + C.CELL_H - inset
    x0, x1 = x + inset, x + C.CELL_W - inset
    cell = norm[y0:y1, x0:x1]
    mask = (cell < INK_THRESHOLD).astype(np.uint8)

    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    ch, cw = mask.shape
    keep = np.zeros(n, bool)
    for i in range(1, n):
        sx, sy, sw, sh, area = stats[i]
        if area < 12:
            continue                                  # speck
        touches = sx == 0 or sy == 0 or sx + sw >= cw or sy + sh >= ch
        if touches and ((sw > 0.5 * cw and sh < 9) or (sh > 0.5 * ch and sw < 9)):
            continue                                  # leftover box border
        keep[i] = True
    mask = keep[lab].astype(np.uint8)
    if mask.sum() < 25:
        return None

    ys, xs = np.nonzero(mask)
    pad = 2
    by0, by1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, ch)
    bx0, bx1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, cw)

    ink_level = float(np.median(cell[mask > 0]))
    alpha = np.clip((1.0 - cell) / max(1.0 - ink_level, 0.2), 0, 1) ** 0.8
    near = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
    alpha = (alpha * near)[by0:by1, bx0:bx1].astype(np.float32)

    top_rel = (y0 + by0) - (y + C.BASELINE_Y)
    return alpha, int(top_rel)


def estimate_stroke(masks):
    area = perim = 0.0
    for m in masks:
        b = (m > 0.5).astype(np.uint8)
        area += b.sum()
        cs, _ = cv2.findContours(b, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        perim += sum(cv2.arcLength(c, True) for c in cs)
    return float(2 * area / perim) if perim else 4.0


def extract_profile(sources, out_dir, name, append=False, log=print):
    """sources: list of (bytes|path|ndarray, filename). Returns a report dict."""
    os.makedirs(os.path.join(out_dir, "glyphs"), exist_ok=True)
    meta_path = os.path.join(out_dir, "profile.json")
    meta = {"name": name, "em": C.EM, "glyphs": {}}
    if append and os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

    report = {"pages": [], "errors": []}
    masks = []
    for src, fname in sources:
        try:
            images = load_images(src, fname)
        except ExtractError as e:
            report["errors"].append(f"{fname}: {e}")
            continue
        for k, img in enumerate(images):
            label = fname if len(images) == 1 else f"{fname} p{k + 1}"
            try:
                warped = rectify(img)
            except ExtractError as e:
                report["errors"].append(f"{label}: {e}")
                continue
            norm = flatten_light(warped)
            pid = read_page_id(norm)
            chars = C.chars_for_page(pid)
            got = []
            for i, ch in enumerate(chars):
                res = cut_glyph(norm, i)
                if res is None:
                    continue
                alpha, top = res
                variants = meta["glyphs"].setdefault(ch, [])
                fn = f"{ord(ch):05X}_{len(variants)}.png"
                cv2.imwrite(os.path.join(out_dir, "glyphs", fn), (alpha * 255).astype(np.uint8))
                variants.append({"file": fn, "top": top, "w": int(alpha.shape[1]), "h": int(alpha.shape[0])})
                masks.append(alpha)
                got.append(ch)
            report["pages"].append({"file": label, "page_id": pid, "extracted": len(got)})
            log(f"{label}: template page {pid + 1}, {len(got)} glyphs")

    if masks:
        meta["stroke"] = estimate_stroke(masks)
    meta.setdefault("stroke", 4.0)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    report["glyph_count"] = len(meta["glyphs"])
    report["missing"] = "".join(c for c in C.CHARSET if c not in meta["glyphs"])
    return report
