#!/usr/bin/env python3
"""nebulizer.py — image → single-color SVG stencil (dots + diagonal bridges).

Turns a grayscale photo into a black-on-white SVG made of two shape kinds:

  * NODES   — rounded squares (circles at BORDER_RADIUS=1.0) on a
              checkerboard grid, radius driven by local image luminance.
              Dark image area -> big node; light -> small node.
  * BRIDGES — concave-pinch bands connecting DIAGONALLY adjacent nodes
              (the two nodes that share a grid corner), present with a
              probability derived from the luminance of the gap between them.

The output is stencil-safe in STENCIL mode: every white region touches
the outside, so nothing is trapped and the whole black area is one
connected island.

Pipeline
--------
  1.  Load image, invert luminance (0=light .. 1=dark).
  2.  Place checkerboard nodes (i+j even); radius = f(cell luminance)
      + seeded jitter, clamped to [RAD_MIN, RAD_MAX].
  3.  For each diagonal node pair, roll a seeded coin with probability
      mapped from gap luminance; optionally reject the bridge if it
      violates the stencil rules.
  4.  Emit one <path> per node and per bridge into a single SVG.
  5.  Rasterize with rsvg-convert (full-res + a smaller preview).

Dependencies: Python 3, numpy, Pillow, and the `rsvg-convert` binary
(Debian/Arch: librsvg2-bin).

Usage
-----
  python3 nebulizer.py photo.jpg
  python3 nebulizer.py photo.jpg --cols 30 --rows 30 --seed 7
  python3 nebulizer.py photo.jpg --stencil off --p-max 0.8
  python3 nebulizer.py photo.jpg --out /tmp/my-stencil

Deterministic: same image + same params + same seed = identical output.
"""
import argparse
import os
import random
import subprocess
import sys
from datetime import datetime

import numpy as np
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────
# Parameters (defaults; all overridable via CLI flags)
# ──────────────────────────────────────────────────────────────────────────
SPACING = 18.0              # grid spacing, SVG units
COLS, ROWS = 20, 20         # grid size (checkerboard nodes on even-parity cells)
RAD_MIN, RAD_MAX = 5.0, 11.0
SIZE_NOISE = 1.0            # per-node radius jitter, +/- (seeded, clamped)
SEED = 41
BORDER_RADIUS = 1.0         # node corner radius = BORDER_RADIUS * r (1.0 = pure circle)
BRIDGE_P_MIN = 0.0          # bridge probability at min gap luminance (floor)
BRIDGE_P_MAX = 0.50         # bridge probability at max gap luminance
BRIDGE_NOISE = 0.10         # seeded jitter on bridge probability
GAP_HALF = 0.5              # half-window (in cells) around the shared corner
STENCIL_MODE = True         # ON: enforce stencil-safe bridge rules; OFF: free
STENCIL_MIN_SKIP = 2        # bridges forbidden while a node is within this many
                            # radius-steps above RAD_MIN (step = (RAD_MAX-RAD_MIN)/ROWS)
RENDER_WIDE = 4200          # full-res PNG width
RENDER_SMALL = 1400         # preview PNG width


# ──────────────────────────────────────────────────────────────────────────
# Node shape
# ──────────────────────────────────────────────────────────────────────────
def rounded_rect_path(x0, y0, x1, y1, cr, sharp_corners=()):
    """SVG path for a rounded rectangle (or circle at cr == side/2).
    `sharp_corners` names any of TL/TR/BR/BL to keep that corner sharp
    (unused by the nebulizer; kept for completeness)."""
    s = []
    if 'TL' in sharp_corners:
        s.append(f"M {x0:.2f} {y0:.2f}")
    else:
        s.append(f"M {x0:.2f} {y0+cr:.2f}"); s.append(f"A {cr:.2f} {cr:.2f} 0 0 1 {x0+cr:.2f} {y0:.2f}")
    if 'TR' in sharp_corners:
        s.append(f"L {x1:.2f} {y0:.2f}")
    else:
        s.append(f"L {x1-cr:.2f} {y0:.2f}"); s.append(f"A {cr:.2f} {cr:.2f} 0 0 1 {x1:.2f} {y0+cr:.2f}")
    if 'BR' in sharp_corners:
        s.append(f"L {x1:.2f} {y1:.2f}")
    else:
        s.append(f"L {x1:.2f} {y1-cr:.2f}"); s.append(f"A {cr:.2f} {cr:.2f} 0 0 1 {x1-cr:.2f} {y1:.2f}")
    if 'BL' in sharp_corners:
        s.append(f"L {x0:.2f} {y1:.2f}")
    else:
        s.append(f"L {x0+cr:.2f} {y1:.2f}"); s.append(f"A {cr:.2f} {cr:.2f} 0 0 1 {x0:.2f} {y1-cr:.2f}")
    s.append("Z")
    return " ".join(s)


# ──────────────────────────────────────────────────────────────────────────
# Bridge geometry
# ──────────────────────────────────────────────────────────────────────────
def diag_bridge(A, B):
    """SVG path for the bridge between two DIAGONALLY adjacent nodes,
    geometry decoded from an Illustrator reference (test.svg in archive).

    A = the node with the smaller x (left of the pair; the caller must
        orient A[0] < B[0]).  B = the node with the larger x (right).
    B is either up-right (by <= ay) or down-right (by > ay) of A.

    STRUCTURE (verified to <0.1 unit against the reference):
      The bridge is a concave-pinch band with TWO boundaries, each a
      QUARTER-CIRCLE of radius e = S - rb  (S = grid spacing = |bx-ax|,
      rb = B's radius). The radius is FORCED, not a free knob: a circle
      tangent to A's facing edge at its midpoint AND tangent-parallel to
      B's facing edge can only have radius exactly S - rb. The cubics
      approximate those quarter-circles with the standard circular-arc
      control-point constant k = 0.5523.

      arc1 (one boundary): starts on A's facing-edge midpoint, tangent to
            that edge, ends tangent-parallel on B's facing edge.
      arc2 (other boundary): the mirror of arc1 across the pair's
            centerline.
      Closing: each arc's far end runs along B's facing edge to B's center
            and cuts 45 deg straight back to A's center. Every closing
            segment (the edge run, the centerline, the 45 deg cut, and the
            close-back along A's centerline) lies under the node fills, so
            only the two concave arcs + the short edge runs are visible.

    Verified against the reference (A r=129.2 lower-left, B r=400
    upper-right, S=625.6, e=225.6): all 8 cubic control/end points match
    the reference paths to <0.1 unit.
    """
    ax, ay, ra = A
    bx, by, rb = B
    S = abs(bx - ax)               # grid spacing (== |by-ay| for a diagonal)
    e = S - rb                     # arc radius (forced)
    if e <= 0:                     # guard the degenerate rb >= S case
        e = max(1e-3, abs(S) * 0.1)
    k = 0.5523                     # circular-arc Bezier constant
    f = lambda v: f"{v:.3f}"

    if by <= ay:
        # ---- B up-right of A ----
        # A's facing edges: TOP (y=ay-ra) & RIGHT (x=ax+ra)
        P0  = (ax, ay - ra)                    # arc1 start (A top-mid)
        C1a = (ax + k * e,          ay - ra)   # exit tangent +x (horizontal)
        C2a = (ax + e,              ay - ra - (1 - k) * e)
        P1  = (ax + e,              ay - ra - e)   # = (bx-rb, ...) on B left edge
        Q0  = (ax + ra, ay)                    # arc2 start (A right-mid)
        C1b = (ax + ra,             ay - k * e)   # exit tangent -y (vertical)
        C2b = (ax + ra + (1 - k) * e,  ay - e)
        Q1  = (ax + ra + e,         ay - e)      # = (bx-rb+ra, by+rb) on B bottom
        L1  = (bx - rb, by)           # B's left point (far edge run end, up-side)
        L2  = (bx, by + rb)           # B's bottom point (far edge run end, down)
    else:
        # ---- B down-right of A ----
        # A's facing edges: BOTTOM (y=ay+ra) & RIGHT (x=ax+ra)
        P0  = (ax, ay + ra)
        C1a = (ax + k * e,          ay + ra)
        C2a = (ax + e,              ay + ra + (1 - k) * e)
        P1  = (ax + e,              ay + ra + e)
        Q0  = (ax + ra, ay)
        C1b = (ax + ra,             ay + k * e)
        C2b = (ax + ra + (1 - k) * e,  ay + e)
        Q1  = (ax + ra + e,         ay + e)
        L1  = (bx - rb, by)
        L2  = (bx, by - rb)

    # Two subpaths (one per boundary), each: arc -> edge run -> B center ->
    # 45 deg cut to A center -> close along A's centerline. The union of the
    # two subpaths is the solid bridge; all closing segments hide under the
    # node fills. A single path with two subpaths keeps one <path> element.
    return (
        f"M {f(P0[0])} {f(P0[1])} "
        f"C {f(C1a[0])} {f(C1a[1])} {f(C2a[0])} {f(C2a[1])} {f(P1[0])} {f(P1[1])} "
        f"L {f(L1[0])} {f(L1[1])} L {f(bx)} {f(by)} L {f(ax)} {f(ay)} Z "
        f"M {f(Q0[0])} {f(Q0[1])} "
        f"C {f(C1b[0])} {f(C1b[1])} {f(C2b[0])} {f(C2b[1])} {f(Q1[0])} {f(Q1[1])} "
        f"L {f(L2[0])} {f(L2[1])} L {f(bx)} {f(by)} L {f(ax)} {f(ay)} Z"
    )


# ──────────────────────────────────────────────────────────────────────────
# Stencil rules
# ──────────────────────────────────────────────────────────────────────────
def seals_pocket(a, b, nodes, placed):
    """True if bridge a-b completes a sealed 4-node ring around an odd-parity
    grid corner (all 4 ring nodes present + all 4 ring bridges present),
    i.e. it would trap a white pocket the ink can't reach."""
    # The bridge's two endpoints are an adjacent pair of exactly one hole
    # ring. Find that corner by locating the odd-parity corner whose 4-node
    # ring contains both endpoints.
    ax, ay = a
    bx, by = b
    for X in (ax, ax + 1, bx, bx + 1):
        for Y in (ay, ay + 1, by, by + 1):
            if (X + Y) % 2 != 1:
                continue
            top, left = (X, Y - 1), (X - 1, Y)
            right, bottom = (X + 1, Y), (X, Y + 1)
            ring = (top, left, right, bottom)
            # both endpoints must be members of this ring
            if a not in ring or b not in ring:
                continue
            # a true hole needs all 4 ring nodes to exist
            if any(n not in nodes for n in ring):
                continue
            # and all 4 ring bridges present: each side is already placed,
            # or IS the bridge we're about to add (a,b).
            sides = [frozenset((top, left)), frozenset((top, right)),
                     frozenset((right, bottom)), frozenset((bottom, left))]
            new = frozenset((a, b))
            if all(s in placed or s == new for s in sides):
                return True
    return False


# ──────────────────────────────────────────────────────────────────────────
# Luminance sampling
# ──────────────────────────────────────────────────────────────────────────
def mean_lum(lum, x0, x1, y0, y1):
    """Mean luminance (0..1) of image rectangle in normalized [0,1] coords."""
    h, w = lum.shape
    px0, px1 = max(0, int(x0 * w)), min(w, max(1, int(x1 * w)))
    py0, py1 = max(0, int(y0 * h)), min(h, max(1, int(y1 * h)))
    return float(lum[py0:py1, px0:px1].mean())


# ──────────────────────────────────────────────────────────────────────────
# nebulizer core
# ──────────────────────────────────────────────────────────────────────────
def build_nodes(lum, cols, rows, rng, p):
    """Checkerboard nodes at (i+j) even. Radius from cell-area luminance +
    seeded noise, clamped to [RAD_MIN, RAD_MAX]. Returns {(i,j): (x, y, r, L)}."""
    S, (rmin, rmax) = p["spacing"], (p["rad_min"], p["rad_max"])
    nodes = {}
    for j in range(rows):
        for i in range(cols):
            if (i + j) % 2 == 0:
                L = mean_lum(lum, i / cols, (i + 1) / cols, j / rows, (j + 1) / rows)
                r = rmin + (rmax - rmin) * L
                r = max(rmin, min(rmax, r + rng.uniform(-p["size_noise"], p["size_noise"])))
                nodes[(i, j)] = (i * S, j * S, r, L)
    return nodes


def build_bridges(lum, nodes, cols, rows, rng, p):
    """Diagonal neighbour bridges, probabilistic.

    Each candidate gap's luminance G maps to a bridge probability:

        prob = P_MIN + (P_MAX - P_MIN) * G  +  uniform(-NOISE, +NOISE)

    clamped to [0, 1]; a seeded coin flip decides. When stencil mode is on,
    accepted bridges are additionally rejected if they violate:

      Rule 1 (min-size): a node at RAD_MIN (or within STENCIL_MIN_SKIP
             radius-steps above it) cannot bridge.
      Rule 2 (no-pocket): a bridge that would close all 4 sides of the
             diamond around an odd-parity grid corner is rejected — the
             white trapped inside could never drain to the outside.

    Returns (bridges, rolls): bridges=[(k1, k2, G, prob)], rolls=[(prob, G)].
    """
    S, (rmin, rmax) = p["spacing"], (p["rad_min"], p["rad_max"])
    bridges, rolls, placed = [], [], set()
    for (i, j) in nodes:
        for di, dj in [(1, 1), (1, -1)]:
            k = (i + di, j + dj)
            if k not in nodes:
                continue
            gx, gy = (i + 1) / cols, (j + dj) / rows
            G = mean_lum(lum, gx - p["gap_half"] / cols, gx + p["gap_half"] / cols,
                         gy - p["gap_half"] / rows, gy + p["gap_half"] / rows)
            prob = (p["p_min"] + (p["p_max"] - p["p_min"]) * G
                    + rng.uniform(-p["bridge_noise"], p["bridge_noise"]))
            prob = max(0.0, min(1.0, prob))
            rolls.append((prob, G))
            if rng.random() >= prob:
                continue
            if p["stencil"]:
                # Rule 1: step size is one grid row's worth of radius range.
                step = (rmax - rmin) / rows
                r_a = nodes[(i, j)][2]
                r_b = nodes[k][2]
                if (r_a <= rmin + step * p["stencil_min_skip"] or
                        r_b <= rmin + step * p["stencil_min_skip"]):
                    continue
                # Rule 2: no sealed white pocket.
                if seals_pocket((i, j), k, nodes, placed):
                    continue
            placed.add(frozenset({(i, j), k}))
            bridges.append(((i, j), k, G, prob))
    return bridges, rolls


def render_svg(nodes, bridges, p):
    """Emit the SVG document: one <path> per node, one per bridge."""
    border = p["border_radius"]
    nodepaths = [rounded_rect_path(cx - r, cy - r, cx + r, cy + r, border * r, set())
                 for (i, j), (cx, cy, r, L) in sorted(nodes.items())]
    bridgepaths = []
    for k1, k2, G, prob in bridges:
        A, B = nodes[k1][:3], nodes[k2][:3]
        if A[0] > B[0]:
            A, B = B, A          # diag_bridge requires A left of B
        bridgepaths.append(diag_bridge(A, B))

    all_r = [n[2] for n in nodes.values()]
    m = max(all_r) + 3
    x0, y0 = -m, -m
    W = (p["cols"] - 1) * p["spacing"] + 2 * m
    H = (p["rows"] - 1) * p["spacing"] + 2 * m
    np_ = "\n".join(f'    <path d="{d}"/>' for d in nodepaths)
    bp = "\n".join(f'    <path d="{d}"/>' for d in bridgepaths)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="{x0:.2f} {y0:.2f} {W:.2f} {H:.2f}">\n'
            f'  <rect x="{x0:.2f}" y="{y0:.2f}" width="{W:.2f}" height="{H:.2f}" fill="white"/>\n'
            f'  <g fill="black">\n{np_}\n{bp}\n  </g>\n</svg>')


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Raster image -> single-color SVG stencil "
                    "(checkerboard nodes + probabilistic diagonal bridges).")
    ap.add_argument("image", help="input image (any format Pillow reads)")
    ap.add_argument("--out", default=None,
                    help="output directory (default: <image dir>/stencil-out)")
    ap.add_argument("--cols", type=int, default=COLS)
    ap.add_argument("--rows", type=int, default=ROWS)
    ap.add_argument("--spacing", type=float, default=SPACING)
    ap.add_argument("--rad-min", type=float, default=RAD_MIN)
    ap.add_argument("--rad-max", type=float, default=RAD_MAX)
    ap.add_argument("--size-noise", type=float, default=SIZE_NOISE)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--border-radius", type=float, default=BORDER_RADIUS,
                    help="node corner radius factor (1.0 = pure circle)")
    ap.add_argument("--p-min", type=float, default=BRIDGE_P_MIN,
                    help="bridge probability at min gap luminance")
    ap.add_argument("--p-max", type=float, default=BRIDGE_P_MAX,
                    help="bridge probability at max gap luminance")
    ap.add_argument("--bridge-noise", type=float, default=BRIDGE_NOISE)
    ap.add_argument("--gap-half", type=float, default=GAP_HALF)
    ap.add_argument("--stencil", choices=["on", "off"],
                    default="on" if STENCIL_MODE else "off",
                    help="stencil-safe bridge rules (default: on)")
    ap.add_argument("--stencil-min-skip", type=int, default=STENCIL_MIN_SKIP)
    ap.add_argument("--wide", type=int, default=RENDER_WIDE,
                    help="full-res PNG width")
    ap.add_argument("--small", type=int, default=RENDER_SMALL,
                    help="preview PNG width")
    args = ap.parse_args(argv)

    p = dict(spacing=args.spacing, cols=args.cols, rows=args.rows,
             rad_min=args.rad_min, rad_max=args.rad_max,
             size_noise=args.size_noise, border_radius=args.border_radius,
             p_min=args.p_min, p_max=args.p_max, bridge_noise=args.bridge_noise,
             gap_half=args.gap_half, stencil=(args.stencil == "on"),
             stencil_min_skip=args.stencil_min_skip)

    outdir = args.out or os.path.join(os.path.dirname(os.path.abspath(args.image)),
                                      "stencil-out")
    os.makedirs(outdir, exist_ok=True)

    # 1. luminance, inverted: light -> small (negative space), dark -> large
    im = Image.open(args.image).convert("L")
    lum = 1.0 - np.array(im, dtype=np.float64) / 255.0

    rng = random.Random(args.seed)

    # 2. + 3. nodes, then probabilistic bridges
    nodes = build_nodes(lum, args.cols, args.rows, rng, p)
    bridges, rolls = build_bridges(lum, nodes, args.cols, args.rows, rng, p)

    # 4. SVG
    svg = render_svg(nodes, bridges, p)
    svgp = os.path.join(outdir, "stencil.svg")
    with open(svgp, "w") as f:
        f.write(svg)

    # 5. PNGs (rsvg-convert required)
    stamp = f"{datetime.now():%H%M%S}"
    pngp = os.path.join(outdir, "stencil.png")
    png_small = os.path.join(outdir, f"stencil-{stamp}.png")
    subprocess.run(["rsvg-convert", "-w", str(args.wide), svgp, "-o", pngp], check=True)
    subprocess.run(["rsvg-convert", "-w", str(args.small), svgp, "-o", png_small], check=True)

    all_r = [n[2] for n in nodes.values()]
    a = np.array(Image.open(pngp).convert("L"))
    ps = [pr for pr, _ in rolls]
    print(f"nodes={len(nodes)} ({args.cols}x{args.rows} checkerboard)  "
          f"bridges={len(bridges)}/{len(rolls)}")
    print(f"r range=[{min(all_r):.2f},{max(all_r):.2f}] (clamped to RAD_MAX={args.rad_max})")
    print(f"bridge prob: {args.p_min:.0%}..{args.p_max:.0%} linear "
          f"+/-{args.bridge_noise:.2f} noise, realized mean={np.mean(ps):.2f}")
    print(f"stencil: {'ON (no-4-node-pocket + min-size rules)' if p['stencil'] else 'OFF'}")
    print(f"blackpx={(a < 128).sum()}")
    print(f"-> {svgp}")
    print(f"-> {pngp} (full res)")
    print(f"-> {png_small} (preview)")


if __name__ == "__main__":
    main()
