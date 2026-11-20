#!/usr/bin/env python3
"""Fast self-tests for the rasteriser, the PackBits coder and the PSD writer.

    python3 selftest.py
"""
import os
import tempfile

import numpy as np

from he_mark.psd_writer import packbits_const, packbits_row, unpackbits
from he_mark.raster import coverage, coverage_float, shape_bbox
from he_mark.render import render_psd


def test_packbits():
    rng = np.random.default_rng(1)
    for trial in range(600):
        n = int(rng.integers(1, 900))
        kind = trial % 4
        if kind == 0:
            row = rng.integers(0, 256, n, dtype=np.uint8)
        elif kind == 1:
            row = np.repeat(rng.integers(0, 256, n // 37 + 1, dtype=np.uint8), 37)[:n]
        elif kind == 2:
            row = np.full(n, int(rng.integers(0, 256)), dtype=np.uint8)
        else:
            row = np.zeros(n, dtype=np.uint8)
            a = int(rng.integers(0, n))
            row[a:] = 255
            row[max(0, a - 2):a] = 77
        assert unpackbits(packbits_row(row), n) == row.tobytes(), (trial, n)
    for n in (1, 2, 3, 127, 128, 129, 255, 256, 257, 30000):
        assert unpackbits(packbits_const(7, n), n) == bytes([7]) * n
    assert len(packbits_const(0, 30000)) == 470
    return "600 random rows and 10 constant rows round-trip"


def test_raster_exact():
    W = H = 400
    # a rectangle with all four edges at fractional positions must be exact
    # even with ONE slab per row: that is the vertical-weighting rule at work
    rect = [("poly", 1, [(10.25, 20.3), (300.75, 20.3), (300.75, 150.7), (10.25, 150.7)])]
    for ss in (1, 4, 16):
        c = coverage_float(rect, 0, 0, W, H, ss, dtype=np.float64)
        assert abs(c[20, 100] - 0.7) < 1e-12 and abs(c[150, 100] - 0.7) < 1e-12, ss
        assert abs(c[100, 10] - 0.75) < 1e-12 and abs(c[100, 300] - 0.75) < 1e-12, ss
        assert abs(c[20, 10] - 0.7 * 0.75) < 1e-12, ss
        assert abs(c.sum() - (300.75 - 10.25) * (150.7 - 20.3)) < 1e-8, ss
    worst = 0.0
    rng = np.random.default_rng(7)
    for _ in range(200):
        pts = rng.uniform(5, 395, (3, 2))
        x, y = pts[:, 0], pts[:, 1]
        area = abs(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
        tri = [("poly", 1, [tuple(p) for p in pts])]
        got = coverage_float(tri, 0, 0, W, H, 16, dtype=np.float64).sum()
        worst = max(worst, abs(got - area))
        rev = [("poly", 1, [tuple(p) for p in pts[::-1]])]
        assert np.array_equal(coverage(tri, 0, 0, W, H, 16), coverage(rev, 0, 0, W, H, 16))
    assert worst < 1e-6, worst          # total area of a triangle is exact
    ell = coverage_float([("ellipse", 1, 200.5, 190.25, 150.0, 90.0)], 0, 0, W, H, 16, dtype=np.float64).sum()
    assert abs(ell - np.pi * 150 * 90) / (np.pi * 150 * 90) < 2e-5
    ring = [("ellipse", 1, 200, 200, 150, 90), ("ellipse", -1, 200, 200, 140, 80)]
    got = coverage_float(ring, 0, 0, W, H, 16, dtype=np.float64).sum()
    assert abs(got - np.pi * (150 * 90 - 140 * 80)) < 0.5
    uni = [("poly", 1, [(50, 50), (250, 50), (250, 250), (50, 250)]),
           ("poly", 1, [(150, 150), (350, 150), (350, 350), (150, 350)])]
    c = coverage(uni, 0, 0, W, H, 16)
    assert c.max() == 255 and abs(c.sum() / 255 - 70000) < 1e-6
    tri = [("poly", 1, [(37.3, 20.1), (380.7, 350.9), (15.2, 300.4)])]
    full = coverage(tri, 0, 0, W, H, 16)
    assert np.array_equal(full[100:260, 30:300], coverage(tri, 30, 100, 300, 260, 16))
    assert shape_bbox(tri, W, H) == (15, 20, 381, 351)
    return f"fractional rectangle exact at 1 slab/row; 200 random triangles, worst area error {worst:.1e} px"


def test_psd_roundtrip():
    from psd_tools import PSDImage
    items = [
        {"name": "Ground", "rgb": (27, 37, 56), "contours": [("ellipse", 1, 320, 240, 300, 200)]},
        {"group": "Ladder", "children": [
            {"name": "Rung 1", "rgb": (145, 29, 15), "contours": [("poly", 1, [(320, 60), (520, 240), (120, 240)])]},
            {"name": "Rung 2 é", "rgb": (30, 70, 167), "contours": [("poly", 1, [(320, 420), (420, 300), (220, 300)])]},
        ]},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "t.psd")
        render_psd(items, 640, 480, path, band=32, progress=False, workers=2)
        psd = PSDImage.open(path)
        assert psd.size == (640, 480) and psd.channels == 4 and psd.depth == 8
        names = [(l.kind, l.name) for l in psd.descendants()]
        assert names == [("pixel", "Ground"), ("group", "Ladder"), ("pixel", "Rung 1"), ("pixel", "Rung 2 é")], names
        for layer, rgb in zip([l for l in psd.descendants() if l.kind == "pixel"],
                              [(27, 37, 56), (145, 29, 15), (30, 70, 167)]):
            arr = np.asarray(layer.topil())
            assert np.unique(arr[..., :3].reshape(-1, 3), axis=0).tolist() == [list(rgb)]
            assert arr[..., 3].max() == 255
    return "layered PSD with a folder and a Unicode name reads back through psd-tools"


if __name__ == "__main__":
    for t in (test_packbits, test_raster_exact, test_psd_roundtrip):
        print("ok  ", t.__name__, "-", t())
