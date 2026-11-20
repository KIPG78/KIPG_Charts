"""Scanline coverage rasteriser for flat vector shapes.

Why not just draw big and shrink?  At 30,000 px a supersampled bitmap would be
hundreds of megapixels per band.  This rasteriser never builds one.  It works
the way a font rasteriser does:

  * horizontally the coverage is EXACT (area of the pixel covered by the span),
  * vertically it averages `ss` sub-scanlines per pixel row.

The trick that makes it fast is a difference array.  A span [xa, xb) on one
sub-scanline is written as four small numbers:

      d[floor(xa)]   += 1 - frac(xa)        d[floor(xb)]   -= 1 - frac(xb)
      d[floor(xa)+1] += frac(xa)            d[floor(xb)+1] -= frac(xb)

and a running sum of `d` gives the exact coverage of every pixel on that line.
Contours carry a sign (+1 add, -1 cut), so rings and cut-outs are just two
contours, and overlapping "+" contours union correctly because each
sub-scanline is clamped to [0, 1] before the vertical average.

A shape is a list of contours.  A contour is one of

    ("poly",    sign, [(x, y), ...])           any simple polygon
    ("ellipse", sign, cx, cy, rx, ry)          axis-aligned ellipse

All coordinates are in output pixels, pixel (i, j) covering [i, i+1) x [j, j+1).
"""
from __future__ import annotations

import numpy as np

Contour = tuple


def _poly_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def contour_bbox(c: Contour) -> tuple[float, float, float, float]:
    """(xmin, ymin, xmax, ymax) of one contour."""
    if c[0] == "poly":
        p = np.asarray(c[2], dtype=np.float64)
        return float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())
    _, _, cx, cy, rx, ry = c
    return cx - rx, cy - ry, cx + rx, cy + ry


def shape_bbox(contours: list[Contour], width: int, height: int) -> tuple[int, int, int, int]:
    """Integer pixel bbox (left, top, right, bottom) of the positive contours,
    clipped to the canvas.  right/bottom are exclusive."""
    boxes = [contour_bbox(c) for c in contours if c[1] > 0]
    if not boxes:
        return 0, 0, 0, 0
    x0 = max(0, int(np.floor(min(b[0] for b in boxes))))
    y0 = max(0, int(np.floor(min(b[1] for b in boxes))))
    x1 = min(width, int(np.ceil(max(b[2] for b in boxes))))
    y1 = min(height, int(np.ceil(max(b[3] for b in boxes))))
    if x1 <= x0 or y1 <= y0:
        return 0, 0, 0, 0
    return x0, y0, x1, y1


def _crossings(c: Contour, ys: np.ndarray):
    """Yield (row_index_array, x_array, weight_array) for one contour.

    `ys` are the sub-scanline centre heights.  weight is +1 where the fill
    switches on going left to right and -1 where it switches off, already
    multiplied by the contour sign.
    """
    sign = float(c[1])
    if c[0] == "ellipse":
        _, _, cx, cy, rx, ry = c
        t = (ys - cy) / ry
        inside = np.abs(t) < 1.0
        idx = np.flatnonzero(inside)
        if idx.size == 0:
            return
        half = rx * np.sqrt(1.0 - t[idx] ** 2)
        yield idx, cx - half, np.full(idx.size, sign)
        yield idx, cx + half, np.full(idx.size, -sign)
        return

    pts = np.asarray(c[2], dtype=np.float64)
    # Normalise the outline direction so the rule below holds for any input.
    if _poly_area(pts) < 0:
        pts = pts[::-1]
    nxt = np.roll(pts, -1, axis=0)
    for (xa, ya), (xb, yb) in zip(pts, nxt):
        if ya == yb:
            continue  # horizontal edges never cross a sub-scanline centre
        lo, hi = (ya, yb) if ya < yb else (yb, ya)
        # half-open [lo, hi) so a vertex shared by two edges is counted once
        idx = np.flatnonzero((ys >= lo) & (ys < hi))
        if idx.size == 0:
            continue
        x = xa + (ys[idx] - ya) * (xb - xa) / (yb - ya)
        # Positive shoelace area with y growing downward means the outline
        # runs clockwise on screen.  Scanning left to right, the first edge
        # met is the one heading UP the screen, so that edge switches the
        # fill on and a downward edge switches it off.
        direction = -1.0 if yb > ya else 1.0
        yield idx, x, np.full(idx.size, sign * direction)


def coverage(contours: list[Contour], x0: int, y0: int, x1: int, y1: int, ss: int = 16) -> np.ndarray:
    """Coverage of the window [x0, x1) x [y0, y1) as uint8 (0..255)."""
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return np.zeros((max(h, 0), max(w, 0)), dtype=np.uint8)
    m = h * ss
    ys = y0 + (np.arange(m, dtype=np.float64) + 0.5) / ss
    d = np.zeros((m, w + 2), dtype=np.float32)
    for c in contours:
        for rows, xs, wts in _crossings(c, ys):
            xs = np.clip(xs - x0, 0.0, float(w))
            xi = np.floor(xs).astype(np.int64)
            fr = (xs - xi).astype(np.float32)
            wts = wts.astype(np.float32)
            np.add.at(d, (rows, xi), wts * (1.0 - fr))
            np.add.at(d, (rows, xi + 1), wts * fr)
    np.cumsum(d, axis=1, out=d)
    np.clip(d, 0.0, 1.0, out=d)
    cov = d[:, :w].reshape(h, ss, w).mean(axis=1)
    return np.rint(cov * 255.0).astype(np.uint8)
