#!/usr/bin/env python3
"""Step 8: image-driven checkerboard with PROBABILISTIC bridges.

Same as step7, but bridge presence is no longer a hard luminance cutoff.
Each gap's luminance G maps to a bridge probability:

    p = BRIDGE_P_MIN + (BRIDGE_P_MAX - BRIDGE_P_MIN) * G
      + seeded noise in [-BRIDGE_NOISE, +BRIDGE_NOISE],  clamped [0, 1]

A seeded coin flip per gap decides presence.

- G = 1.0 (max luminance) -> ~50%
- G = 0.0 (white / negative space) -> ~10% floor
"""
import os
import random
import subprocess

import numpy as np
from PIL import Image

from step6 import diag_bridge
import step3e

IMG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Hermes.jpg")
S = 18.0                 # grid spacing
COLS, ROWS = 20, 20      # grid size
RAD_MIN, RAD_MAX = 5.0, 11.0
SIZE_NOISE = 1.0         # per-node random radius jitter, +/- (seeded, clamped to [RAD_MIN, RAD_MAX])
SEED = 41
BORDER_RADIUS = 1.0      # global: 1.0 = max = pure circle
BRIDGE_P_MIN = 0.0       # bridge probability at min luminance (floor)
BRIDGE_P_MAX = 0.50      # bridge probability at max luminance
BRIDGE_NOISE = 0.10      # seeded jitter on bridge probability
GAP_HALF = 0.5           # half-cell window (in cells) around the shared corner

# ── Stencil rules ──────────────────────────────────────────────────────────
STENCIL_MODE = True      # ON: enforce stencil-safe bridge rules; OFF: free
STENCIL_MIN_SKIP = 2     # bridges forbidden while a node is within this many
                         # radius-steps above RAD_MIN (step = (RAD_MAX-RAD_MIN)/ROWS)
                         # (1 = only the very smallest; 2 = smallest + one step up)


def _seals_pocket(a, b, nodes, placed):
    """True if bridge a-b completes a sealed 4-node ring around an odd-parity
    grid corner (all 4 ring nodes present + all 4 ring bridges present)."""
    # The bridge's two endpoints are an adjacent pair of exactly one hole ring.
    # Find that corner by locating the odd-parity corner whose 4-node ring
    # contains both endpoints.
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


def _mean_lum(lum, x0, x1, y0, y1):
    """Mean luminance (0..1) of image rectangle in normalized [0,1] coords."""
    h, w = lum.shape
    px0, px1 = max(0, int(x0 * w)), min(w, max(1, int(x1 * w)))
    py0, py1 = max(0, int(y0 * h)), min(h, max(1, int(y1 * h)))
    return float(lum[py0:py1, px0:px1].mean())


def main():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "step8")
    os.makedirs(outdir, exist_ok=True)

    im = Image.open(IMG).convert("L")
    # INVERTED: light image areas -> small (negative space), dark -> large
    lum = 1.0 - np.array(im, dtype=np.float64) / 255.0

    rng = random.Random(SEED)

    # Checkerboard nodes: (i+j) even. Size from cell-area luminance +
    # seeded noise, clamped to [RAD_MIN, RAD_MAX].
    nodes = {}
    for j in range(ROWS):
        for i in range(COLS):
            if (i + j) % 2 == 0:
                L = _mean_lum(lum, i / COLS, (i + 1) / COLS, j / ROWS, (j + 1) / ROWS)
                r = RAD_MIN + (RAD_MAX - RAD_MIN) * L
                r = max(RAD_MIN, min(RAD_MAX, r + rng.uniform(-SIZE_NOISE, SIZE_NOISE)))
                nodes[(i, j)] = (i * S, j * S, r, L)

    # Bridges: diagonal neighbours, PROBABILISTIC.
    # p = linear map of gap luminance [P_MIN..P_MAX] + seeded noise, clamped.
    #
    # Stencil rules (STENCIL_MODE on):
    #   1. MIN-SIZE: a node within STENCIL_MIN_SKIP steps above RAD_MIN can't bridge
    #   2. NO-POCKET: a bridge is rejected if it would close all 4 sides of the
    #      diamond at its inner corner -> white can't escape
    bridges, rolls = [], []
    placed = set()  # canonical node pairs of bridges accepted
    for (i, j), node in nodes.items():
        for di, dj in [(1, 1), (1, -1)]:
            k = (i + di, j + dj)
            if k not in nodes:
                continue
            gx, gy = (i + 1) / COLS, (j + dj) / ROWS
            G = _mean_lum(lum, gx - GAP_HALF / COLS, gx + GAP_HALF / COLS,
                          gy - GAP_HALF / ROWS, gy + GAP_HALF / ROWS)
            p = (BRIDGE_P_MIN + (BRIDGE_P_MAX - BRIDGE_P_MIN) * G
                 + rng.uniform(-BRIDGE_NOISE, BRIDGE_NOISE))
            p = max(0.0, min(1.0, p))
            rolls.append((p, G))
            if rng.random() >= p:
                continue
            if STENCIL_MODE:
                # Rule 1: a node at minimum size (or within STENCIL_MIN_SKIP
                # steps above it) cannot bridge. Step size is one grid row's
                # worth of radius range.
                step = (RAD_MAX - RAD_MIN) / ROWS
                r_a = nodes[(i, j)][2]
                r_b = nodes[k][2]
                if (r_a <= RAD_MIN + step * STENCIL_MIN_SKIP or
                        r_b <= RAD_MIN + step * STENCIL_MIN_SKIP):
                    continue
                # Rule 2: no sealed white pocket. A hole forms at an odd-parity
                # grid corner (X,Y) when all 4 even-parity nodes around it exist
                # AND all 4 diagonal bridges between them are present. Our new
                # bridge seals exactly one such corner; reject it if that corner
                # would be fully closed (nodes + the other 3 bridges).
                if _seals_pocket((i, j), k, nodes, placed):
                    continue
            placed.add(frozenset({(i, j), k}))
            bridges.append(((i, j), k, G, p))

    nodepaths, bridgepaths, info = [], [], []
    for (i, j), (cx, cy, r, L) in sorted(nodes.items()):
        cr = BORDER_RADIUS * r
        nodepaths.append(step3e.rounded_rect_path(cx - r, cy - r, cx + r, cy + r, cr, set()))

    for (k1, k2, G, p) in bridges:
        A, B = nodes[k1][:3], nodes[k2][:3]
        if A[0] > B[0]:
            A, B = B, A
        bow = 0.5 * min(A[2], B[2])
        bridgepaths.append(diag_bridge(A, B, bow))
        info.append(f"  {k1}-{k2}  gapLum={G:.2f}  p={p:.2f}")

    all_r = [n[2] for n in nodes.values()]
    m = max(all_r) + 3
    x0, y0 = -m, -m
    W = (COLS - 1) * S + 2 * m
    H = (ROWS - 1) * S + 2 * m
    np_ = "\n".join(f'    <path d="{p}"/>' for p in nodepaths)
    bp = "\n".join(f'    <path d="{p}"/>' for p in bridgepaths)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0:.2f} {y0:.2f} {W:.2f} {H:.2f}">
  <rect x="{x0:.2f}" y="{y0:.2f}" width="{W:.2f}" height="{H:.2f}" fill="white"/>
  <g fill="black">
{np_}
{bp}
  </g>
</svg>'''

    svgp = os.path.join(outdir, "step8.svg")
    pngp = os.path.join(outdir, "step8.png")
    # Timestamped display copy: the gateway dedups MEDIA: tags by path, so
    # re-rendering to a fixed filename gets silently swallowed. A fresh
    # name per render keeps every send distinct.
    from datetime import datetime
    png_small = os.path.join(outdir, f"step8-{datetime.now():%H%M%S}.png")
    with open(svgp, "w") as f:
        f.write(svg)
    subprocess.run(["rsvg-convert", "-w", "4200", svgp, "-o", pngp], check=True)
    subprocess.run(["rsvg-convert", "-w", "1400", svgp, "-o", png_small], check=True)
    a = np.array(Image.open(pngp).convert("L"))
    ps = [p for p, _ in rolls]
    print(f"nodes={len(nodes)} ({COLS}x{ROWS} checkerboard)  bridges={len(bridgepaths)}/{len(rolls)}")
    print(f"r range=[{min(all_r):.2f},{max(all_r):.2f}] (clamped to RAD_MAX={RAD_MAX})")
    print(f"bridge prob: {BRIDGE_P_MIN:.0%}..{BRIDGE_P_MAX:.0%} linear +/-{BRIDGE_NOISE:.2f} noise, "
          f"realized mean={np.mean(ps):.2f}")
    print(f"stencil: {'ON (no-4-node-pocket + min-size rules)' if STENCIL_MODE else 'OFF'}")
    print(f"blackpx={(a < 128).sum()}")
    print(f"-> {pngp} (full res)")
    print(f"-> {png_small} (chat copy)")


if __name__ == "__main__":
    main()
