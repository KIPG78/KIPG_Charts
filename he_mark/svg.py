"""Emit the same layer stack as an SVG, the resolution-independent master.

The SVG and the PSD are built from one list of contours, so they cannot drift
apart.  Each layer becomes one <path> inside a named <g>.  Positive contours
are written clockwise and cut-outs counter-clockwise, so fill-rule="nonzero"
gives exactly what the rasteriser computes: union of the adds minus the cuts.
"""
from __future__ import annotations

from .raster import _poly_area

import numpy as np


def _fmt(v: float) -> str:
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _poly_path(pts, clockwise: bool) -> str:
    p = np.asarray(pts, dtype=np.float64)
    # positive shoelace area with y down == clockwise on screen
    if (_poly_area(p) > 0) != clockwise:
        p = p[::-1]
    return "M" + " L".join(f"{_fmt(x)},{_fmt(y)}" for x, y in p) + " Z"


def _ellipse_path(cx, cy, rx, ry, clockwise: bool) -> str:
    sweep = 1 if clockwise else 0
    return (f"M{_fmt(cx - rx)},{_fmt(cy)} "
            f"A{_fmt(rx)},{_fmt(ry)} 0 1 {sweep} {_fmt(cx + rx)},{_fmt(cy)} "
            f"A{_fmt(rx)},{_fmt(ry)} 0 1 {sweep} {_fmt(cx - rx)},{_fmt(cy)} Z")


def contours_path(contours) -> str:
    parts = []
    for c in contours:
        cw = c[1] > 0
        if c[0] == "poly":
            parts.append(_poly_path(c[2], cw))
        else:
            parts.append(_ellipse_path(c[2], c[3], c[4], c[5], cw))
    return " ".join(parts)


def _slug(name: str) -> str:
    # Pixelmator Pro shows an SVG element's id as its layer name, so keep it readable
    return "".join(ch if ch.isalnum() else "-" for ch in name).strip("-")


def to_svg(items: list[dict], width: float, height: float, title: str = "", background: str | None = None) -> str:
    def hexc(rgb):
        return "#%02X%02X%02X" % tuple(rgb)

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_fmt(width)} {_fmt(height)}" '
           f'width="{_fmt(width)}" height="{_fmt(height)}">']
    if title:
        out.append(f"  <title>{title}</title>")
    if background:
        out.append(f'  <rect width="100%" height="100%" fill="{background}"/>')

    def emit(seq, depth):
        pad = "  " * depth
        for it in seq:
            if "group" in it:
                out.append(f'{pad}<g id="{_slug(it["group"])}">')
                emit(it["children"], depth + 1)
                out.append(f"{pad}</g>")
            else:
                out.append(f'{pad}<path id="{_slug(it["name"])}" fill="{hexc(it["rgb"])}" fill-rule="nonzero" '
                           f'd="{contours_path(it["contours"])}"/>')

    emit(items, 1)
    out.append("</svg>")
    return "\n".join(out) + "\n"


def scale_items(items: list[dict], s: float, dx: float = 0.0, dy: float = 0.0) -> list[dict]:
    """Copy of the stack with every coordinate mapped to x*s+dx, y*s+dy."""
    def sc(c):
        if c[0] == "poly":
            return ("poly", c[1], [(x * s + dx, y * s + dy) for x, y in c[2]])
        return ("ellipse", c[1], c[2] * s + dx, c[3] * s + dy, c[4] * s, c[5] * s)

    out = []
    for it in items:
        if "group" in it:
            out.append({"group": it["group"], "children": scale_items(it["children"], s, dx, dy)})
        else:
            out.append({**it, "contours": [sc(c) for c in it["contours"]]})
    return out
