"""Vectorise a profile's glyphs into an installable TrueType font."""

import cv2
import numpy as np
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen

UPM = 1000
CAP = 700          # cap height in font units
ASCENT, DESCENT = 950, -320
UPSAMPLE = 3


def _signed_area(pts):
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def _glyph_contours(glyph, scale, lsb_px):
    """Outline contours in font units (y up), correctly oriented for TrueType."""
    a = cv2.resize(glyph.alpha, None, fx=UPSAMPLE, fy=UPSAMPLE, interpolation=cv2.INTER_CUBIC)
    a = cv2.GaussianBlur(a, (0, 0), UPSAMPLE * 0.5)
    mask = (a > 0.45).astype(np.uint8)
    contours, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    out = []
    if hier is None:
        return out
    for c, h in zip(contours, hier[0]):
        if len(c) < 8:
            continue
        c = cv2.approxPolyDP(c, UPSAMPLE * 0.6, True).reshape(-1, 2).astype(np.float64)
        if len(c) < 3:
            continue
        px = c[:, 0] / UPSAMPLE + lsb_px
        py = c[:, 1] / UPSAMPLE + glyph.top
        pts = np.stack([px * scale, -py * scale], axis=1)
        is_hole = h[3] != -1
        area = _signed_area(pts)
        # TrueType: outer contours clockwise (negative area), holes counter-clockwise
        if (not is_hole and area > 0) or (is_hole and area < 0):
            pts = pts[::-1]
        out.append(pts)
    return out


def _draw_smooth(pen, pts):
    """Treat polygon vertices as quadratic off-curve points -> smooth B-spline."""
    n = len(pts)
    mids = [((pts[i] + pts[(i + 1) % n]) / 2) for i in range(n)]
    r = lambda p: (int(round(p[0])), int(round(p[1])))
    pen.moveTo(r(mids[-1]))
    for i in range(n):
        pen.qCurveTo(r(pts[i]), r(mids[i]))
    pen.closePath()


def export_ttf(profile, path, family=None):
    family = family or f"{profile.name} Hand"
    scale = CAP / profile.em
    side = 0.07 * profile.em

    order = [".notdef", "space"]
    cmap = {32: "space"}
    glyf, metrics = {}, {}

    pen = TTGlyphPen(None)
    glyf[".notdef"] = pen.glyph()
    metrics[".notdef"] = (int(UPM * 0.5), 0)
    glyf["space"] = TTGlyphPen(None).glyph()
    metrics["space"] = (int(profile.em * 0.42 * scale), 0)

    for ch in profile.meta["glyphs"]:
        if ch == " ":
            continue
        g = profile.variants(ch)[0]
        name = f"uni{ord(ch):04X}"
        pen = TTGlyphPen(None)
        xmin = None
        for pts in _glyph_contours(g, scale, side):
            _draw_smooth(pen, pts)
            m = pts[:, 0].min()
            xmin = m if xmin is None else min(xmin, m)
        order.append(name)
        cmap[ord(ch)] = name
        glyf[name] = pen.glyph()
        adv = int(round((g.w + 2 * side) * scale))
        metrics[name] = (adv, int(round(xmin)) if xmin is not None else 0)

    fb = FontBuilder(UPM, isTTF=True)
    fb.setupGlyphOrder(order)
    fb.setupCharacterMap(cmap)
    fb.setupGlyf(glyf)
    fb.setupHorizontalMetrics(metrics)
    fb.setupHorizontalHeader(ascent=ASCENT, descent=DESCENT)
    safe = "".join(ch for ch in family if ch.isalnum()) or "Handwriting"
    fb.setupNameTable({"familyName": family, "styleName": "Regular",
                         "uniqueFontIdentifier": f"{safe}-Regular", "fullName": f"{family} Regular",
                         "psName": f"{safe}-Regular", "version": "Version 1.0"})
    fb.setupOS2(sTypoAscender=ASCENT, sTypoDescender=DESCENT, sTypoLineGap=100,
                usWinAscent=ASCENT + 50, usWinDescent=-DESCENT + 50,
                sCapHeight=CAP, sxHeight=int(CAP * 0.55), achVendID="HAND")
    fb.setupPost()
    fb.save(path)
    return path
