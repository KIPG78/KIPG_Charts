"""Small in-memory renders (previews, thumbnails, test references).

Uses the same rasteriser and the same compositing arithmetic as the PSD band
renderer, so a preview is a faithful miniature of the big file.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .raster import coverage, shape_bbox
from .render import flatten
from .svg import scale_items


def composite_premult(items: list[dict], width: int, height: int, ss: int = 32) -> np.ndarray:
    """float32 (H, W, 4): premultiplied RGB in 0..255, alpha in 0..1."""
    rgb = np.zeros((height, width, 3), dtype=np.float32)
    alpha = np.zeros((height, width), dtype=np.float32)
    for layer in flatten(items):
        x0, y0, x1, y1 = shape_bbox(layer["contours"], width, height)
        if x1 <= x0 or y1 <= y0:
            continue
        a = coverage(layer["contours"], x0, y0, x1, y1, ss).astype(np.float32) / 255.0
        sl = (slice(y0, y1), slice(x0, x1))
        rgb[sl] = rgb[sl] * (1 - a)[..., None] + a[..., None] * np.asarray(layer["rgb"], dtype=np.float32)
        alpha[sl] = alpha[sl] * (1 - a) + a
    return np.dstack([rgb, alpha])


def render_rgba(items: list[dict], design_w: float, design_h: float, px_height: int, ss: int = 32) -> Image.Image:
    """Straight-alpha RGBA preview, `px_height` pixels tall."""
    s = px_height / design_h
    w = int(round(design_w * s))
    pm = composite_premult(scale_items(items, s), w, px_height, ss)
    a = pm[..., 3]
    safe = np.where(a > 0, a, 1.0)
    rgb = np.rint(pm[..., :3] / safe[..., None]).clip(0, 255).astype(np.uint8)
    return Image.fromarray(np.dstack([rgb, np.rint(a * 255).astype(np.uint8)]), "RGBA")


def on_background(img: Image.Image, colour=(255, 255, 255)) -> Image.Image:
    bg = Image.new("RGBA", img.size, tuple(colour) + (255,))
    bg.alpha_composite(img)
    return bg.convert("RGB")
