"""Scanline coverage rasteriser for flat vector shapes.

Why not just draw big and shrink?  At 30,000 px a supersampled bitmap would be
hundreds of megapixels per band.  This rasteriser never builds one.  It works
the way a font rasteriser does: each pixel row is cut into `ss` thin slabs, and
for every slab it computes where the shape's outline crosses.

  * Horizontally the coverage is EXACT: the area of each pixel covered by the
    span, not a count of samples.
  * Vertically a slab counts for exactly as much of its height as the edge
    actually spans.  So a flat edge (the base of a triangle) that falls part
    way through a pixel row is exact too, and straight sloping edges are
    integrated by the midpoint rule, which is exact for a straight line.

The trick that makes it fast is a difference array.  A span [xa, xb) on one
slab is written as four small numbers:

      d[floor(xa)]   += 1 - frac(xa)        d[floor(xb)]   -= 1 - frac(xb)
      d[floor(xa)+1] += frac(xa)            d[floor(xb)+1] -= frac(xb)

and a running sum of `d` gives the coverage of every pixel on that slab.
Contours carry a sign (+1 add, -1 cut), so rings and cut-outs are just two
contours, and overlapping "+" contours union correctly because each slab is
clamped to [0, 1] before the vertical average.

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


def _slabs(lo: float, hi: float, y0: float, m: int, ss: int):
    """Slabs of the window that overlap the height range [lo, hi].

    Returns (row indices, weight = fraction of the slab covered, mid-height of
    the covered part).  Slab i spans [y0 + i/ss, y0 + (i+1)/ss)."""
    first = max(0, int(np.floor((lo - y0) * ss)))
    last = min(m - 1, int(np.ceil((hi - y0) * ss)) - 1)
    if last < first:
        return None
    idx = np.arange(first, last + 1)
    top = y0 + idx / ss
    a = np.maximum(top, lo)
    b = np.minimum(top + 1.0 / ss, hi)
    wt = (b - a) * ss
    keep = wt > 0
    if not keep.any():
        return None
    return idx[keep], wt[keep], 0.5 * (a[keep] + b[keep])


def _crossings(c: Contour, y0: float, m: int, ss: int):
    """Yield (row_index_array, x_array, weight_array) for one contour.

    weight is positive where the fill switches on going left to right and
    negative where it switches off, scaled by the contour sign and by the
    fraction of the slab the edge spans.
    """
    sign = float(c[1])
    if c[0] == "ellipse":
        _, _, cx, cy, rx, ry = c
        s = _slabs(cy - ry, cy + ry, y0, m, ss)
        if s is None:
            return
        idx, wt, ymid = s
        t = (ymid - cy) / ry
        half = rx * np.sqrt(np.clip(1.0 - t * t, 0.0, None))
        yield idx, cx - half, sign * wt
        yield idx, cx + half, -sign * wt
        return

    pts = np.asarray(c[2], dtype=np.float64)
    # Normalise the outline direction so the rule below holds for any input.
    if _poly_area(pts) < 0:
        pts = pts[::-1]
    nxt = np.roll(pts, -1, axis=0)
    for (xa, ya), (xb, yb) in zip(pts, nxt):
        if ya == yb:
            continue  # a flat edge has no height: its neighbours carry it
        s = _slabs(min(ya, yb), max(ya, yb), y0, m, ss)
        if s is None:
            continue
        idx, wt, ymid = s
        x = xa + (ymid - ya) * (xb - xa) / (yb - ya)
        # Positive shoelace area with y growing downward means the outline
        # runs clockwise on screen.  Scanning left to right, the first edge
        # met is the one heading UP the screen, so that edge switches the
        # fill on and a downward edge switches it off.
        direction = -1.0 if yb > ya else 1.0
        yield idx, x, sign * direction * wt


def coverage_float(contours: list[Contour], x0: int, y0: int, x1: int, y1: int, ss: int = 16,
                   dtype=np.float32) -> np.ndarray:
    """Coverage of the window [x0, x1) x [y0, y1) as fractions 0..1."""
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return np.zeros((max(h, 0), max(w, 0)), dtype=dtype)
    m = h * ss
    d = np.zeros((m, w + 2), dtype=dtype)
    for c in contours:
        for rows, xs, wts in _crossings(c, float(y0), m, ss):
            xs = np.clip(xs - x0, 0.0, float(w))
            xi = np.floor(xs).astype(np.int64)
            fr = (xs - xi).astype(dtype)
            wts = wts.astype(dtype)
            np.add.at(d, (rows, xi), wts * (1.0 - fr))
            np.add.at(d, (rows, xi + 1), wts * fr)
    np.cumsum(d, axis=1, out=d)
    np.clip(d, 0.0, 1.0, out=d)
    return d[:, :w].reshape(h, ss, w).mean(axis=1)


def coverage(contours: list[Contour], x0: int, y0: int, x1: int, y1: int, ss: int = 16) -> np.ndarray:
    """Coverage of the window [x0, x1) x [y0, y1) as uint8 (0..255)."""
    return np.rint(coverage_float(contours, x0, y0, x1, y1, ss) * 255.0).astype(np.uint8)
