#!/usr/bin/env python3
"""Build the reversed-ladder HE mark: SVG master, PNG previews, layered PSDs.

    python3 build.py                      # SVG + previews + 16,380 px PSD
    python3 build.py --px 30000           # the PSD format's ceiling
    python3 build.py --px 16380 30000     # both
    python3 build.py --variant B          # one of the alternate readings

Everything comes from one Spec (he_mark/design.py).  The SVG and the PSD are
written from the same list of contours, so they cannot disagree.
"""
from __future__ import annotations

import argparse
import io
import json
import os

from he_mark.design import Spec, build
from he_mark.preview import on_background, render_rgba
from he_mark.render import render_psd
from he_mark.svg import scale_items, to_svg

NAME = "HE_Geometric_C_Ledger_A_Roofline_Stairs_LadderReversed"

VARIANTS = {
    "A": ("Rungs: separate equilateral triangles", Spec()),
    "B": ("Stairs: each triangle overlaps the next by a third", Spec(gap=-1 / 3)),
    "C": ("Rooflines: open chevrons instead of solid triangles", Spec(style="chevron")),
    "D": ("Wide roofs: 90 degree apex in a rounder oval", Spec(apex_deg=90, aspect=1.3)),
}


def canvas_px(spec: Spec, px_height: int) -> tuple[int, int, float, float]:
    """Pixel canvas for a given height, and the scale / x-offset that centre
    the design on it when the width does not come out whole."""
    s = px_height / spec.height
    w = int(round(spec.width * s))
    return w, px_height, s, (w - spec.width * s) / 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--px", type=int, nargs="*", default=[16380], help="PSD heights in pixels (max 30000)")
    ap.add_argument("--variant", default="A", choices=sorted(VARIANTS))
    ap.add_argument("--out", default="out")
    ap.add_argument("--ss", type=int, default=16, help="sub-scanlines per pixel row")
    ap.add_argument("--no-psd", action="store_true")
    args = ap.parse_args()

    label, spec = VARIANTS[args.variant]
    items, solved = build(spec)
    stem = NAME if args.variant == "A" else f"{NAME}_alt{args.variant}"
    os.makedirs(args.out, exist_ok=True)

    with open(os.path.join(args.out, stem + ".svg"), "w") as f:
        f.write(to_svg(items, spec.width, spec.height, title=f"HE mark, reversed ladder ({label})"))

    preview = render_rgba(items, spec.width, spec.height, 3000, ss=32)
    preview.save(os.path.join(args.out, stem + "_preview.png"))
    on_background(preview).save(os.path.join(args.out, stem + "_preview_on_white.png"))

    with open(os.path.join(args.out, stem + "_construction.json"), "w") as f:
        json.dump(solved, f, indent=2)

    if args.no_psd:
        return
    thumb_img = on_background(render_rgba(items, spec.width, spec.height, 160, ss=32))
    buf = io.BytesIO()
    thumb_img.save(buf, "JPEG", quality=90)
    for px in args.px:
        w, h, s, dx = canvas_px(spec, px)
        path = os.path.join(args.out, f"{stem}_{h}px.psd")
        print(f"PSD {w} x {h} px ({w * h / 1e6:.0f} megapixels) -> {path}", flush=True)
        info = render_psd(scale_items(items, s, dx, 0.0), w, h, path, ss=args.ss,
                          thumbnail=(buf.getvalue(), thumb_img.width, thumb_img.height))
        print(f"  wrote {info['bytes'] / 1e6:.1f} MB in {info['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
