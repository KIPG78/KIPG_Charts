#!/usr/bin/env python3
"""Check a built PSD against the design, pixel by pixel.

The file is parsed and its PackBits data decoded by psd-tools, an independent
reader: none of this script's reading path uses he_mark's writer.

    python3 verify.py out/<name>_16380px.psd [more.psd ...]

What is checked
  1. header, resources and the layer tree (names, order, folders);
  2. every layer's colour channels are one flat colour: the specified one;
  3. every layer's alpha, shrunk by an integer factor with a box filter,
     equals a fresh low-resolution render of that shape.  Area coverage is
     linear, so the two must agree to within rounding;
  4. the layers, recomposited AT FULL RESOLUTION the way any layered editor
     would, give exactly the stored flattened picture.  (3) and (4) together
     mean the flattened picture is the design;
  5. the picture is mirror-symmetric left/right and top/bottom;
  6. transparent pixels of the flattened picture are white (Photoshop's matte).

A note on what is NOT a pass criterion: the flattened colour compared with a
low-resolution composite.  Where three partly covered layers share one pixel
(the two side corners of the crossing) a coarse composite is simply less
accurate than the big file, so that number is reported, not judged.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
from psd_tools import PSDImage

from he_mark.design import Spec, build
from he_mark.raster import coverage_float, shape_bbox
from he_mark.render import flatten
from he_mark.svg import scale_items

from build import VARIANTS, canvas_px


def reference_lowres(items, design_w, design_h, w, h, ss=128, band=64):
    """Premultiplied RGB (0..255) and alpha (0..1) at w x h, rendered in bands
    with many sub-scanlines so it can serve as ground truth."""
    s = h / design_h
    dx = (w - design_w * s) / 2
    layers = flatten(scale_items(items, s, dx, 0.0))
    rgb = np.zeros((h, w, 3), dtype=np.float64)
    alpha = np.zeros((h, w), dtype=np.float64)
    per_layer = []
    for layer in layers:
        x0, y0, x1, y1 = shape_bbox(layer["contours"], w, h)
        a_full = np.zeros((h, w), dtype=np.float64)
        for b0 in range(y0, y1, band):
            b1 = min(y1, b0 + band)
            # float coverage, not the 8-bit one, to keep the reference clean
            a_full[b0:b1, x0:x1] = _coverage_float(layer["contours"], x0, b0, x1, b1, ss)
        per_layer.append(a_full)
        rgb = rgb * (1 - a_full)[..., None] + a_full[..., None] * np.asarray(layer["rgb"], dtype=np.float64)
        alpha = alpha * (1 - a_full) + a_full
    return rgb, alpha, layers, per_layer


def _coverage_float(contours, x0, y0, x1, y1, ss):
    # float64 and many slabs: a clean reference, never rounded to 8 bits
    return coverage_float(contours, x0, y0, x1, y1, ss, dtype=np.float64)


def box(arr: np.ndarray, f: int) -> np.ndarray:
    h, w = arr.shape
    return arr.reshape(h // f, f, w // f, f).sum(axis=(1, 3), dtype=np.uint64) / float(f * f)


def decode_channels(channel_data_list, width, height, header):
    return [np.frombuffer(c.get_data(width, height, header.depth, header.version), dtype=np.uint8).reshape(height, width)
            for c in channel_data_list]


def verify(path: str, variant: str = "A") -> dict:
    t0 = time.time()
    report: dict = {"file": path, "checks": {}}
    ok = report["checks"]
    psd = PSDImage.open(path)
    hdr = psd._record.header
    W, H = psd.width, psd.height
    report["canvas_px"] = [W, H]
    report["megapixels"] = round(W * H / 1e6, 1)
    ok["header: 8BPS v1, RGB, 8-bit, 4 channels"] = (hdr.version == 1 and hdr.color_mode == 3 and hdr.depth == 8
                                                    and hdr.channels == 4)
    ids = sorted(int(k) for k in psd.image_resources.keys())
    ok["resources 1005 / 1036 / 1039 / 1057 present"] = ids == [1005, 1036, 1039, 1057]
    ok["resolution is 300 ppi"] = psd.image_resources.get_data(1005).horizontal == 300 * 65536

    label, spec = VARIANTS[variant]
    items, solved = build(spec)
    expect = [(l["name"], tuple(l["rgb"])) for l in flatten(items)]

    # ---- layer tree ------------------------------------------------------
    tree = []

    def walk(group, depth):
        for layer in group:
            tree.append((depth, layer.kind, layer.name))
            if layer.is_group():
                walk(layer, depth + 1)

    walk(psd, 0)
    report["tree"] = [f"{'  ' * d}{k}: {n}" for d, k, n in tree]
    pixel_layers = [l for l in psd.descendants() if l.kind == "pixel"]
    ok["layer names and order match the design"] = [l.name for l in pixel_layers] == [n for n, _ in expect]
    ok["two folders: Lower ladder, Upper ladder"] = [n for d, k, n in tree if k == "group"] == ["Lower ladder", "Upper ladder"]
    ok["all layers visible, normal blend, 100% opacity"] = all(
        l.visible and l.opacity == 255 and l.blend_mode.name == "NORMAL" for l in pixel_layers)

    # ---- pick the largest integer shrink factor that lands near 3000 px ---
    f = max(k for k in range(1, 41) if H % k == 0 and W % k == 0 and H // k >= 2000)
    w, h = W // f, H // f
    report["shrink_factor"] = f
    report["compared_at_px"] = [w, h]
    ref_rgb, ref_a, ref_layers, ref_alphas = reference_lowres(items, spec.width, spec.height, w, h)
    ref_matte = ref_rgb + 255.0 * (1 - ref_a)[..., None]

    # ---- stored flattened picture -----------------------------------------
    merged = psd._record.image_data.get_data(hdr)            # psd-tools decodes the RLE
    R, G, B, A = [np.frombuffer(c, dtype=np.uint8).reshape(H, W) for c in merged]
    worst_a = float(np.abs(box(A, f) - ref_a * 255.0).max())
    low_rgb_diff = np.max([np.abs(box(ch, f) - ref_matte[..., c]) for c, ch in enumerate((R, G, B))], axis=0)
    report["flattened_alpha_vs_design_max_diff_levels"] = round(worst_a, 3)
    report["flattened_colour_vs_coarse_composite"] = {
        "max_diff_levels": round(float(low_rgb_diff.max()), 3),
        "coarse_pixels_over_1_level": int((low_rgb_diff > 1.0).sum()),
        "note": "informational: a coarse composite is less accurate than the file where three layer edges meet"}
    ok["flattened alpha equals the design (within 1 level of 255)"] = worst_a <= 1.0

    clear = A == 0
    ok["transparent pixels are white (Photoshop matte)"] = bool(
        (R[clear] == 255).all() and (G[clear] == 255).all() and (B[clear] == 255).all())
    solid = A == 255
    cols = np.unique(np.stack([R[solid][::997], G[solid][::997], B[solid][::997]], axis=1), axis=0)
    palette = {tuple(c) for _, c in expect}
    ok["every palette colour appears in the flattened picture"] = palette <= {tuple(int(v) for v in c) for c in cols}
    lr = max(int(np.abs(ch.astype(np.int16) - ch[:, ::-1].astype(np.int16)).max()) for ch in (R, G, B, A))
    tb = max(int(np.abs(ch.astype(np.int16) - ch[::-1, :].astype(np.int16)).max()) for ch in (R, G, B, A))
    report["mirror_max_diff_levels"] = {"left_right": lr, "top_bottom": tb}
    ok["mirror-symmetric left/right and top/bottom, all four channels (within 1 level)"] = lr <= 1 and tb <= 1
    del clear, solid

    # ---- each layer --------------------------------------------------------
    recs = psd._record.layer_and_mask_information.layer_info
    flat_ok, alpha_worst, k = True, 0.0, 0
    layer_rows, decoded = [], []
    for rec, chans in zip(recs.layer_records, recs.channel_image_data):
        lw, lh = rec.right - rec.left, rec.bottom - rec.top
        if lw == 0 or lh == 0:
            continue                                         # folder markers
        name, colour = expect[k]
        ids_ = [ci.id for ci in rec.channel_info]
        data = dict(zip(ids_, decode_channels(chans, lw, lh, hdr)))
        flat = all(int(data[c].min()) == int(data[c].max()) == colour[c] for c in (0, 1, 2))
        flat_ok &= flat
        full = np.zeros((H, W), dtype=np.uint8)
        full[rec.top:rec.bottom, rec.left:rec.right] = data[-1]
        d = float(np.abs(box(full, f) / 255.0 - ref_alphas[k]).max() * 255.0)
        alpha_worst = max(alpha_worst, d)
        layer_rows.append({"layer": name, "bounds_ltrb": [rec.left, rec.top, rec.right, rec.bottom],
                           "colour": "#%02X%02X%02X" % colour, "flat_colour": flat,
                           "alpha_vs_design_max_diff": round(d, 3)})
        decoded.append((rec.left, rec.top, rec.right, rec.bottom, colour, data[-1]))
        del full, data
        k += 1
    report["layers"] = layer_rows
    ok["every layer is one flat colour, the specified one"] = flat_ok and k == len(expect)
    ok["every layer's alpha equals the design (within 1 level)"] = alpha_worst <= 1.0

    # ---- recomposite the decoded layers at full resolution ------------------
    worst_full, over1 = 0, 0
    step = 500
    for y0 in range(0, H, step):
        y1 = min(H, y0 + step)
        acc_rgb = np.zeros((y1 - y0, W, 3), dtype=np.float32)
        acc_a = np.zeros((y1 - y0, W), dtype=np.float32)
        for left, top, right, bottom, colour, alpha in decoded:
            a0, a1 = max(y0, top), min(y1, bottom)
            if a1 <= a0:
                continue
            a = alpha[a0 - top:a1 - top].astype(np.float32) / 255.0
            sl = (slice(a0 - y0, a1 - y0), slice(left, right))
            acc_rgb[sl] = acc_rgb[sl] * (1 - a)[..., None] + a[..., None] * np.asarray(colour, dtype=np.float32)
            acc_a[sl] = acc_a[sl] * (1 - a) + a
        matte = np.rint(acc_rgb + 255.0 * (1 - acc_a)[..., None])
        d = np.zeros((y1 - y0, W), dtype=np.float32)
        for c, ch in enumerate((R, G, B)):
            np.maximum(d, np.abs(matte[..., c] - ch[y0:y1]), out=d)
        np.maximum(d, np.abs(np.rint(acc_a * 255.0) - A[y0:y1]), out=d)
        worst_full = max(worst_full, int(d.max()))
        over1 += int((d > 1).sum())
    report["layers_recomposited_at_full_resolution_vs_flattened"] = {
        "max_diff_levels": worst_full, "pixels_over_1_level": over1, "pixels_compared": W * H}
    ok["layers recomposite to the stored flattened picture at full resolution (within 1 level)"] = worst_full <= 1

    report["passed"] = all(ok.values())
    report["seconds"] = round(time.time() - t0, 1)
    return report


if __name__ == "__main__":
    paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    all_ok = True
    for p in paths:
        variant = p.split("_alt")[1][0] if "_alt" in p else "A"
        rep = verify(p, variant)
        all_ok &= rep["passed"]
        print(json.dumps(rep, indent=2))
        with open(p[:-4] + "_verification.json", "w") as fh:
            json.dump(rep, fh, indent=2)
    sys.exit(0 if all_ok else 1)
