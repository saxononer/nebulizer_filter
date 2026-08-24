#!/usr/bin/env python3
"""Step 6: 5x5 CHECKERBOARD test — varying node sizes, random diagonal bridges.

Topology (per Saxon's correction):
- Nodes on a 5x5 grid at (i,j) where (i+j)%2==0  -> a checkerboard:
  one point filled, one empty, alternating. 13 nodes on 5x5.
- Bridges connect DIAGONALLY adjacent filled nodes (corner-to-corner),
  NOT edge-sharing neighbours. This is the approved step5 configuration.

Bridge geometry (step5, APPROVED):
- Each node grips the bridge along the two edges adjacent to its
  corner facing the gap. Max grip = full straight edge = 1.25r
  (from the facing corner out to where the adjacent border-radius
  arc begins).
- Two cubic Beziers (lower + upper boundary), each tangent-parallel
  to both node edges at its endpoints. Control points bow toward the
  gap center -> concave pinch.
- Bow: LINEAR function of the two node sizes, clamped to [5, 9],
  then capped at 1.25*min(r1, r2) (smaller node's grip) so the
  control point never overshoots the node edge.
"""
import math
import os
import random
import subprocess

import numpy as np
from PIL import Image

import step3e

S = 18.0                 # grid spacing
N = 5                    # grid size
RAD_MIN, RAD_MAX = 6.0, 9.0
BORDER_RADIUS = 1.0      # GLOBAL: border-radius factor of r. 1.0 = max = pure circle
BOW_MIN, BOW_MAX = 5.0, 9.0
SIZE_MIN, SIZE_MAX = 6.0, 22.0   # node-sum range mapping to full bow range
SEED = 41
N_BRIDGES = 8


def bow_for(r1, r2):
    """Linear map: sum of node radii -> bow, clamped to [5, 9],
    then capped at the smaller node's grip (1.25r) so the control
    point never overshoots the node edge."""
    total = r1 + r2
    t = (total - SIZE_MIN) / (SIZE_MAX - SIZE_MIN)
    t = max(0.0, min(1.0, t))
    bow = BOW_MIN + (BOW_MAX - BOW_MIN) * t
    return min(bow, 1.25 * min(r1, r2))


def diag_bridge(A, B, bow=None):
    """Bridge between two DIAGONALLY adjacent nodes, geometry decoded from
    test.svg (Illustrator reference).

    A = the node with the smaller x (left of the pair; the caller orients
        A[0] < B[0]).  B = the node with the larger x (right of the pair).
    B is either up-right (by <= ay) or down-right (by > ay) of A.

    DECODED STRUCTURE (verified to <0.1 unit against test.svg):
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

    Verified against test.svg (A r=129.2 lower-left, B r=400 upper-right,
    S=625.6, e=225.6): all 8 cubic control/end points match the reference
    paths to <0.1 unit.

    `bow` is accepted for signature compatibility but unused — the arc
    radius is fully determined by node positions and radii.
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


def main():
    rng = random.Random(SEED)
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "step6")
    os.makedirs(outdir, exist_ok=True)

    # Checkerboard nodes: (i+j) even
    nodes = {}
    for j in range(N):
        for i in range(N):
            if (i + j) % 2 == 0:
                nodes[(i, j)] = (i * S, j * S, rng.uniform(RAD_MIN, RAD_MAX))

    # Candidate bridges: diagonal neighbours (both filled)
    pairs = set()
    for (i, j) in nodes:
        for di, dj in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            k = (i + di, j + dj)
            if k in nodes:
                pairs.add(frozenset([(i, j), k]))
    pair_list = [tuple(sorted(p)) for p in pairs]
    chosen = rng.sample(pair_list, min(N_BRIDGES, len(pair_list)))

    nodepaths, bridgepaths, info = [], [], []
    for (i, j), (cx, cy, r) in sorted(nodes.items()):
        cr = BORDER_RADIUS * r
        nodepaths.append(step3e.rounded_rect_path(cx - r, cy - r, cx + r, cy + r, cr, set()))

    for (k1, k2) in chosen:
        A, B = nodes[k1], nodes[k2]
        # orient so A is upper-left
        if A[0] > B[0]:
            A, B = B, A
        bow = bow_for(A[2], B[2])
        bridgepaths.append(diag_bridge(A, B, bow))
        info.append(f"  {k1}-{k2}  r={A[2]:.2f}+{B[2]:.2f}  bow={bow:.2f}")

    all_r = [n[2] for n in nodes.values()]
    x_min = -max(all_r) - 3
    y_min = x_min
    x_max = (N - 1) * S + max(all_r) + 3
    W = x_max - x_min
    np_ = "\n".join(f'    <path d="{p}"/>' for p in nodepaths)
    bp = "\n".join(f'    <path d="{p}"/>' for p in bridgepaths)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x_min:.2f} {y_min:.2f} {W:.2f} {W:.2f}">
  <rect x="{x_min:.2f}" y="{y_min:.2f}" width="{W:.2f}" height="{W:.2f}" fill="white"/>
  <g fill="black">
{np_}
{bp}
  </g>
</svg>'''

    svgp = os.path.join(outdir, "step6.svg")
    pngp = os.path.join(outdir, "step6.png")
    with open(svgp, "w") as f:
        f.write(svg)
    subprocess.run(["rsvg-convert", "-w", "1000", svgp, "-o", pngp], check=True)
    a = np.array(Image.open(pngp).convert("L"))
    print(f"nodes={len(nodes)} (checkerboard)  bridges={len(bridgepaths)} (diagonal)")
    print(f"r range=[{min(all_r):.2f},{max(all_r):.2f}]")
    print("bridges (seeded):")
    print("\n".join(info))
    print(f"blackpx={(a < 128).sum()}")
    print(f"-> {pngp}")


if __name__ == "__main__":
    main()
