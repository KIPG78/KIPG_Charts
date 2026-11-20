"""Render a stack of flat-colour vector layers to a layered PSD, band by band.

The picture is never held whole.  The canvas is cut into horizontal bands; each
band is rasterised and compressed in a worker process and comes back as a few
kilobytes of PackBits rows, which the writer appends in order.  Memory stays
flat whether the canvas is 2,000 px or 30,000 px.

A document is a bottom-to-top list of items:

    {"name": str, "rgb": (r, g, b), "contours": [...]}              a layer
    {"group": str, "children": [ ...layers, bottom to top... ]}     a folder
"""
from __future__ import annotations

import multiprocessing as mp
import time

import numpy as np

from .psd_writer import PsdDocument, packbits_row
from .raster import coverage, shape_bbox

_STATE: dict = {}


def flatten(items: list[dict]) -> list[dict]:
    """Pixel layers only, bottom to top."""
    out = []
    for it in items:
        if "group" in it:
            out.extend(flatten(it["children"]))
        else:
            out.append(it)
    return out


def _init(layers, width, height, ss):
    _STATE.update(layers=layers, width=width, height=height, ss=ss)


def _band(args):
    y0, y1 = args
    layers, width, ss = _STATE["layers"], _STATE["width"], _STATE["ss"]
    h = y1 - y0
    acc_rgb = np.zeros((h, width, 3), dtype=np.float32)   # premultiplied
    acc_a = np.zeros((h, width), dtype=np.float32)
    layer_rows: dict[int, list[bytes]] = {}
    for idx, layer in enumerate(layers):
        left, top, right, bottom = layer["bbox"]
        a0, a1 = max(y0, top), min(y1, bottom)
        if a1 <= a0 or right <= left:
            continue
        cov = coverage(layer["contours"], left, a0, right, a1, ss)
        layer_rows[idx] = [packbits_row(r) for r in cov]
        a = cov.astype(np.float32) / 255.0
        sl = (slice(a0 - y0, a1 - y0), slice(left, right))
        inv = 1.0 - a
        rgb = np.asarray(layer["rgb"], dtype=np.float32)
        acc_rgb[sl] = acc_rgb[sl] * inv[..., None] + a[..., None] * rgb
        acc_a[sl] = acc_a[sl] * inv + a
    # Photoshop stores the flattened picture matted on white, with the true
    # transparency in a fourth channel.
    matte = np.rint(acc_rgb + 255.0 * (1.0 - acc_a)[..., None]).clip(0, 255).astype(np.uint8)
    alpha = np.rint(acc_a * 255.0).clip(0, 255).astype(np.uint8)
    merged = [[packbits_row(r) for r in matte[:, :, c]] for c in range(3)]
    merged.append([packbits_row(r) for r in alpha])
    return y0, y1, layer_rows, merged


def render_psd(items: list[dict], width: int, height: int, path: str, *, ss: int = 16,
               band: int = 48, ppi: float = 300.0, workers: int | None = None,
               tmpdir: str | None = None, progress: bool = True) -> dict:
    layers = flatten(items)
    for layer in layers:
        layer["bbox"] = shape_bbox(layer["contours"], width, height)

    doc = PsdDocument(width, height, ppi=ppi, tmpdir=tmpdir)
    handles: dict[int, object] = {}

    def add(seq):
        for it in seq:
            if "group" in it:
                doc.begin_group()
                add(it["children"])
                doc.end_group(it["group"])
            else:
                i = next(k for k, l in enumerate(layers) if l is it)
                handles[i] = doc.add_layer(it["name"], it["rgb"], it["bbox"])

    add(items)

    bands = [(y, min(y + band, height)) for y in range(0, height, band)]
    slim = [{"contours": l["contours"], "rgb": l["rgb"], "bbox": l["bbox"]} for l in layers]
    t0 = time.time()
    workers = workers or max(1, mp.cpu_count())
    with mp.Pool(workers, initializer=_init, initargs=(slim, width, height, ss)) as pool:
        for n, (y0, y1, layer_rows, merged) in enumerate(pool.imap(_band, bands, chunksize=1)):
            for idx, rows in layer_rows.items():
                handles[idx].add_encoded_rows(rows)
            for c in range(4):
                doc.add_merged_rows(c, merged[c])
            if progress and (n % max(1, len(bands) // 20) == 0 or n == len(bands) - 1):
                print(f"  band {n + 1}/{len(bands)}  rows {y1}/{height}  {time.time() - t0:.0f}s", flush=True)
    size = doc.save(path)
    return {"path": path, "bytes": size, "seconds": time.time() - t0, "width": width, "height": height,
            "layers": [{"name": l["name"], "bbox": l["bbox"]} for l in layers]}
