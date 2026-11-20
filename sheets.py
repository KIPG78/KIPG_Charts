#!/usr/bin/env python3
"""Two picture sheets that go with the mark.

    python3 sheets.py            # writes into out/

  *_understanding.png   what was asked, drawn as before -> after
  *_options.png         the chosen reading (A) beside three alternates (B, C, D)

The "before" panel is an assumption drawn from the request's wording, not a
copy of the original file, and the sheet says so.
"""
from __future__ import annotations

import os
from dataclasses import replace

from PIL import Image, ImageDraw, ImageFont

from build import NAME, VARIANTS
from he_mark.design import Spec, build, solve
from he_mark.preview import render_rgba

INTER = "/usr/share/fonts/opentype/inter/Inter-{}.otf"
PAGE = (250, 249, 245)
INK = (27, 37, 56)
MUTED = (104, 110, 122)
RED = (145, 29, 15)
RULE = (221, 218, 210)


def font(weight: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(INTER.format(weight), size)
    except OSError:
        return ImageFont.load_default(size)


def before_items(spec: Spec) -> list[dict]:
    """The request read backwards: both ladders point the same way, so the
    upper one is a plain repeat of the lower one (smallest triangle on top)."""
    s = solve(replace(spec, cross=0.0))
    cx, cy = spec.width / 2, spec.height / 2
    last = s["rungs"][-1]
    shift = last["base"] + spec.gap * last["height"]
    items = [{"name": "Oval", "rgb": spec.ground, "contours": [("ellipse", 1, cx, cy, spec.oval_a, spec.oval_b)]}]
    for r in s["rungs"]:
        for dy in (0.0, -shift):
            ty, by, hw = cy + r["tip"] + dy, cy + r["base"] + dy, r["half_base"]
            items.append({"name": "rung", "rgb": spec.figure, "contours": [("poly", 1, [(cx, ty), (cx + hw, by), (cx - hw, by)])]})
    return items


def mark(items, spec: Spec, px_height: int) -> Image.Image:
    return render_rgba(items, spec.width, spec.height, px_height, ss=16)


def wrap(draw, text, fnt, width):
    lines, cur = [], ""
    for word in text.split():
        trial = (cur + " " + word).strip()
        if draw.textlength(trial, font=fnt) <= width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    return lines + [cur]


def paragraph(draw, xy, text, fnt, fill, width, leading=1.38):
    x, y = xy
    for line in wrap(draw, text, fnt, width):
        draw.text((x, y), line, font=fnt, fill=fill)
        y += int(fnt.size * leading)
    return y


def understanding_sheet(path: str) -> None:
    spec = VARIANTS["A"][1]
    mh = 1500
    before = mark(before_items(spec), spec, mh)
    after = mark(build(spec)[0], spec, mh)
    mw = before.width
    m, gap = 130, 420
    W = 2 * m + 2 * mw + gap
    H = 2440
    page = Image.new("RGB", (W, H), PAGE)
    d = ImageDraw.Draw(page)
    d.text((m, 110), "WHAT I UNDERSTOOD", font=font("SemiBold", 34), fill=RED)
    d.text((m, 168), "Turn the upper ladder so the two smallest triangles meet.", font=font("Bold", 68), fill=INK)
    top = 380
    x1, x2 = m, m + mw + gap
    page.paste(before, (x1, top), before)
    page.paste(after, (x2, top), after)
    # arrow between the panels
    ay = top + mh // 2
    ax0, ax1 = x1 + mw + 90, x2 - 90
    d.line([(ax0, ay), (ax1 - 20, ay)], fill=RED, width=10)
    d.polygon([(ax1, ay), (ax1 - 46, ay - 30), (ax1 - 46, ay + 30)], fill=RED)
    lab = font("SemiBold", 34)
    txt = "rotate 180°"
    d.text(((ax0 + ax1) / 2 - d.textlength(txt, font=lab) / 2, ay - 76), txt, font=lab, fill=RED)
    cy = top + mh + 56
    cap, body = font("Bold", 44), font("Regular", 36)
    d.text((x1, cy), "Before, as I read it", font=cap, fill=INK)
    paragraph(d, (x1, cy + 72),
              "Both ladders point the same way. The upper one repeats the lower one, so its smallest "
              "triangle sits at the top of the oval. This panel is my assumption: I could not open "
              "your original file.", body, MUTED, mw)
    d.text((x2, cy), "After, as built", font=cap, fill=INK)
    paragraph(d, (x2, cy + 72),
              "The upper ladder is turned over. Its smallest triangle is now in the middle, facing the "
              "smallest of the lower part. Their tips cross at the exact centre (red), and each rung "
              "grows toward the edge of the oval.", body, MUTED, mw)
    page.save(path)


def options_sheet(path: str) -> None:
    order = ["A", "B", "C", "D"]
    mh = 1200
    marks = {k: mark(build(VARIANTS[k][1])[0], VARIANTS[k][1], mh) for k in order}
    m, gap = 130, 110
    col = max(im.width for im in marks.values())
    W = 2 * m + 4 * col + 3 * gap
    H = 2060
    page = Image.new("RGB", (W, H), PAGE)
    d = ImageDraw.Draw(page)
    d.text((m, 110), "FOUR READINGS OF THE SAME REQUEST", font=font("SemiBold", 34), fill=RED)
    d.text((m, 168), "A is built at full size. B, C and D are one command away.", font=font("Bold", 68), fill=INK)
    top = 380
    notes = {
        "A": "Ladder. Separate triangles, each two thirds of the next. Built as the PSD.",
        "B": "Stairs. Each triangle overlaps the next by a third, giving one stepped outline.",
        "C": "Rooflines. Open chevrons, so the centre becomes two crossing bars.",
        "D": "Wide roofs. A right-angle apex in a rounder oval.",
    }
    for i, k in enumerate(order):
        x = m + i * (col + gap)
        im = marks[k]
        page.paste(im, (x + (col - im.width) // 2, top), im)
        y = top + mh + 50
        d.text((x, y), k, font=font("Bold", 84), fill=RED if k == "A" else INK)
        if k == "A":
            d.text((x + 86, y + 34), "chosen", font=font("SemiBold", 34), fill=RED)
        d.line([(x, y + 122), (x + col, y + 122)], fill=RULE, width=3)
        paragraph(d, (x, y + 150), notes[k], font("Regular", 36), MUTED, col)
    page.save(path)


if __name__ == "__main__":
    os.makedirs("out", exist_ok=True)
    understanding_sheet(os.path.join("out", NAME + "_understanding.png"))
    options_sheet(os.path.join("out", NAME + "_options.png"))
    print("sheets written")
