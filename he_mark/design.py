"""HE mark, Geometric C / Ledger A / Roofline Stairs: the reversed ladder.

THE FIGURE
    An oval.  Inside it, on the vertical axis, two ladders of triangles.
    The smallest triangle of each ladder sits in the middle and points at the
    other; their tips cross at the exact centre.  Every next rung is larger,
    and the largest pair has its outer corners on a line parallel to the oval's
    edge.

THE LAW (why there is nothing to tune)
    A ladder is ONE triangle repeated by ONE operation: scale by r about one
    fixed point V on the axis.  If rung 1 has its tip at distance t and its
    base at distance s from V, rung k has them at t*r^(k-1) and s*r^(k-1).
    So sizes, positions and gaps are all the same geometric sequence, and the
    stepped outline of the ladder lies on two straight rays out of V.

    The ladder used to run the other way.  Reversing it is that same matrix
    read backwards: ratio q = 2/3 (shrinking toward the edge) becomes
    r = 1/q = 3/2 (growing toward the edge).  Nothing else changes.

    V lies on the far side of the centre, inside the opposite ladder.  That is
    the whole reason the two smallest triangles intersect: each ladder
    converges on a point that belongs to the other.

    Two thirds appears three times, by construction:
      * each rung is 2/3 the size of the next one out,
      * 2/3 of each smallest triangle stays on its own side of the centre,
      * the crossing (the red lozenge) is 2/3 of that triangle's height.

THE FIT (one equation, one unknown)
    The outer corners of the largest rung must land on the inset ellipse
    (x/a')^2 + (y/b')^2 = 1.  Both the corner's x and y are a known number
    times h1, the height of the smallest triangle, so

        h1 = 1 / sqrt( (W_N / a')^2 + (U_N / b')^2 )

    with W_N = r^(N-1) * tan(phi) and
         U_N = cross_keep + (r + gap) * (r^(N-1) - 1) / (r - 1).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

# HE palette, from the brand's own specification of the mark colours.
HE_NAVY = (0x1B, 0x25, 0x38)   # ground of the mark
HE_BLUE = (0x1E, 0x46, 0xA7)
HE_RED = (0x91, 0x1D, 0x0F)    # the single signal
HE_INK = (0x11, 0x11, 0x11)
BONE = (0xF3, 0xF2, 0xEE)      # light figure colour on the navy ground


@dataclass
class Spec:
    # canvas, in design units (the PSD scales these to pixels)
    oval_b: float = 1400.0          # oval semi-height
    aspect: float = 14.0 / 9.0      # oval height / width  (semi-width 900)
    pad: float = 100.0              # clear space; canvas comes out 2000 x 3000 (2:3)
    # ladder
    rungs: int = 4                  # triangles per ladder
    ratio: float = 2.0 / 3.0        # q: each rung is q times the next one out
    apex_deg: float = 60.0          # full apex angle; 60 = equilateral
    gap: float = 0.25               # air after a rung, as a fraction of its height
    cross: float = 1.0 / 3.0        # how far a smallest tip passes the centre, / h1
    inset: float = 0.10             # corner ellipse sits this fraction inside the oval
    style: str = "solid"            # "solid" triangles or open "chevron" rooflines
    stroke: float = 0.30            # chevron bar thickness, as a fraction of rung height
    # colours
    ground: tuple = HE_NAVY
    figure: tuple = BONE
    signal: tuple = HE_RED
    notes: dict = field(default_factory=dict)

    @property
    def oval_a(self) -> float:
        return self.oval_b / self.aspect

    @property
    def width(self) -> float:
        return 2 * self.oval_a + 2 * self.pad

    @property
    def height(self) -> float:
        return 2 * self.oval_b + 2 * self.pad


def solve(spec: Spec) -> dict:
    """All construction numbers, in design units, y measured from the centre
    outward (the same numbers serve the upper and the lower ladder)."""
    r = 1.0 / spec.ratio
    n = spec.rungs
    tan_phi = math.tan(math.radians(spec.apex_deg) / 2)
    a_in = spec.oval_a * (1 - spec.inset)
    b_in = spec.oval_b * (1 - spec.inset)
    keep = 1.0 - spec.cross                       # part of rung 1 on its own side
    geo = (r ** (n - 1) - 1) / (r - 1)
    w_n = r ** (n - 1) * tan_phi                  # half-base of the last rung / h1
    u_n = keep + (r + spec.gap) * geo             # its base distance / h1
    h1 = 1.0 / math.sqrt((w_n / a_in) ** 2 + (u_n / b_in) ** 2)

    rungs = []
    tip = -spec.cross * h1                        # rung 1's tip is past the centre
    h = h1
    for k in range(1, n + 1):
        base = tip + h
        rungs.append({"k": k, "tip": tip, "base": base, "height": h, "half_base": h * tan_phi})
        tip = base + spec.gap * h                 # next tip: one gap beyond this base
        h *= r
    # fixed point of the scaling: solve tip2 - V = r * (tip1 - V)
    t1, t2 = rungs[0]["tip"], (rungs[1]["tip"] if n > 1 else rungs[0]["tip"] * r)
    v = (t2 - r * t1) / (1 - r)
    return {
        "r": r, "q": spec.ratio, "h1": h1, "tan_phi": tan_phi, "a_in": a_in, "b_in": b_in,
        "vanishing_point": v, "rungs": rungs,
        "lozenge_height": 2 * spec.cross * h1,
        "lozenge_half_width": spec.cross * h1 * tan_phi,
        "corner": (rungs[-1]["half_base"], rungs[-1]["base"]),
    }


def build(spec: Spec) -> tuple[list[dict], dict]:
    """Layer stack (bottom to top) in design units, plus the solved numbers."""
    s = solve(spec)
    cx, cy = spec.width / 2, spec.height / 2

    def tri(rung, direction):
        # direction -1: upper ladder (tip points down at the centre);
        # direction +1: lower ladder (tip points up at the centre)
        ty = cy + direction * rung["tip"]
        by = cy + direction * rung["base"]
        hw = rung["half_base"]
        outer = ("poly", 1, [(cx, ty), (cx + hw, by), (cx - hw, by)])
        if spec.style != "chevron":
            return [outer]
        # open roofline: cut a similar triangle out of the base side, leaving
        # two bars of constant thickness that meet at the tip
        cut = rung["height"] * (1 - spec.stroke)
        ity = by - direction * cut
        ihw = cut * s["tan_phi"]
        return [outer, ("poly", -1, [(cx, ity), (cx + ihw, by), (cx - ihw, by)])]

    upper = [{"name": f"Upper rung {r['k']}", "rgb": spec.figure, "contours": tri(r, -1)} for r in s["rungs"]]
    lower = [{"name": f"Lower rung {r['k']}", "rgb": spec.figure, "contours": tri(r, +1)} for r in s["rungs"]]
    lz_h, lz_w = s["lozenge_height"] / 2, s["lozenge_half_width"]
    lozenge = {"name": "Crossing", "rgb": spec.signal,
               "contours": [("poly", 1, [(cx, cy - lz_h), (cx + lz_w, cy), (cx, cy + lz_h), (cx - lz_w, cy)])]}
    oval = {"name": "Oval", "rgb": spec.ground, "contours": [("ellipse", 1, cx, cy, spec.oval_a, spec.oval_b)]}

    items = [
        oval,
        # largest first, so the smallest rung is the top layer of each folder
        {"group": "Lower ladder", "children": lower[::-1]},
        {"group": "Upper ladder", "children": upper[::-1]},
        lozenge,
    ]
    s["canvas"] = (spec.width, spec.height)
    s["centre"] = (cx, cy)
    s["spec"] = {k: v for k, v in asdict(spec).items() if k != "notes"}
    return items, s
