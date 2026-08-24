#!/usr/bin/env python3
"""Step 7: image-driven checkerboard digitizer.

Template: Hermes.jpg (200x201). Grid: 20x10 checkerboard.

Mapping:
- Node size   = mean luminance of the cell's image area -> radius [6.0, 9.0]
- Bridge      = mean luminance of the gap area around the shared corner
                between two diagonal cells -> bridge present if >= 0.5

Geometry: identical to step6 (circular nodes, 1.0r grip, cubic Bezier
pinch bridges, size-mapped bow).
"""
import os
import random
import subprocess

import numpy as np
from PIL import Image

from step6 import bow_for, diag_bridge
import step3e

IMG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Hermes.jpg")
S = 18.0                 # grid spacing
COLS, ROWS = 20, 20      # grid size
RAD_MIN, RAD_MAX = 5.0, 11.0
SIZE_NOISE = 1.0          # per-node random radius jitter, +/- this (seeded, clamped to [RAD_MIN, RAD_MAX])
SEED = 41
BORDER_RADIUS = 1.0      # global: 1.0 = max = pure circle
BRIDGE_THRESH = 0.5      # gap luminance threshold for bridge presence
GAP_HALF = 0.5           # half-cell window (in cells) around the shared corner


def _mean_lum(lum, x0, x1, y0, y1):
    """Mean luminance (0..1) of image rectangle in normalized [0,1] coords."""
    h, w = lum.shape
    px0, px1 = max(0, int(x0 * w)), min(w, max(1, int(x1 * w)))
    py0, py1 = max(0, int(y0 * h)), min(h, max(1, int(y1 * h)))
    return float(lum[py0:py1, px0:px1].mean())


def main():
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "step7")
    os.makedirs(outdir, exist_ok=True)

    im = Image.open(IMG).convert("L")
    # INVERTED: light image areas -> small (negative space), dark -> large
    lum = 1.0 - np.array(im, dtype=np.float64) / 255.0

    # Checkerboard nodes: (i+j) even. Size from cell-area luminance +
    # seeded noise, clamped to [RAD_MIN, RAD_MAX].
    rng = random.Random(SEED)
    nodes = {}
    for j in range(ROWS):
        for i in range(COLS):
            if (i + j) % 2 == 0:
                L = _mean_lum(lum, i / COLS, (i + 1) / COLS, j / ROWS, (j + 1) / ROWS)
                r = RAD_MIN + (RAD_MAX - RAD_MIN) * L
                r = max(RAD_MIN, min(RAD_MAX, r + rng.uniform(-SIZE_NOISE, SIZE_NOISE)))
                nodes[(i, j)] = (i * S, j * S, r, L)

    # Bridges: diagonal neighbours. Presence from gap-area luminance around
    # the shared grid corner.
    bridges = []
    for (i, j), node in nodes.items():
        for di, dj in [(1, 1), (1, -1)]:
            k = (i + di, j + dj)
            if k in nodes:
                # shared corner at grid point (i+1, j+dj)
                gx, gy = (i + 1) / COLS, (j + dj) / ROWS
                G = _mean_lum(lum, gx - GAP_HALF / COLS, gx + GAP_HALF / COLS,
                              gy - GAP_HALF / ROWS, gy + GAP_HALF / ROWS)
                if G >= BRIDGE_THRESH:
                    bridges.append(((i, j), k, G))

    nodepaths, bridgepaths, info = [], [], []
    for (i, j), (cx, cy, r, L) in sorted(nodes.items()):
        cr = BORDER_RADIUS * r
        nodepaths.append(step3e.rounded_rect_path(cx - r, cy - r, cx + r, cy + r, cr, set()))

    for (k1, k2, G) in bridges:
        A, B = nodes[k1][:3], nodes[k2][:3]
        if A[0] > B[0]:
            A, B = B, A
        bow = bow_for(A[2], B[2])
        bridgepaths.append(diag_bridge(A, B, bow))
        info.append(f"  {k1}-{k2}  r={A[2]:.2f}+{B[2]:.2f}  gapLum={G:.2f}  bow={bow:.2f}")

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

    svgp = os.path.join(outdir, "step7.svg")
    pngp = os.path.join(outdir, "step7.png")
    with open(svgp, "w") as f:
        f.write(svg)
    subprocess.run(["rsvg-convert", "-w", "1400", svgp, "-o", pngp], check=True)
    a = np.array(Image.open(pngp).convert("L"))
    ls = [n[3] for n in nodes.values()]
    print(f"nodes={len(nodes)} ({COLS}x{ROWS} checkerboard)  bridges={len(bridgepaths)}")
    print(f"r range=[{min(all_r):.2f},{max(all_r):.2f}]  cellLum range=[{min(ls):.2f},{max(ls):.2f}]")
    print(f"bridge threshold: gap luminance >= {BRIDGE_THRESH}")
    print(f"blackpx={(a < 128).sum()}")
    print(f"-> {pngp}")


if __name__ == "__main__":
    main()
