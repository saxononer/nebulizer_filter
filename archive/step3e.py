#!/usr/bin/env python3
"""Step 3e: bridge replicated from Saxon's Inkscape reference (saxonedit.svg).

Exact geometry extracted from the reference SVG path data:
  - center node  : (10,10)-(30,30), r=10, BR corner SHARP at C=(30,30)
  - BR node      : (34,34)-(46,46), r=6,  TL corner SHARP at B=(34,34)
  - bridge path  :
      M (30, 30-g)          on center's RIGHT edge, g above corner
      L (30, 30)            down to center's sharp corner C
      L (30-g, 30)          left along center bottom edge, g from corner
      A R R 0 0 sweep -> (34, 34+g)   ONE concave arc to BR left edge
      L (34, 34)            up to BR's sharp corner B
      L (34+g, 34)          right along BR top edge, g from corner
      A R R 0 0 sweep -> (30, 30-g)   ONE concave arc back to start
      Z

Key properties (verified numerically from reference):
  - Each arc leaves the node edge TANGENTIALLY (tangent parallel to the edge)
  - Each arc is CONCAVE (bows toward the gap center (32,32))
  - The two sharp corners C and B sit on the diagonal between the arcs,
    so the bridge reads as one continuous shape with the nodes.

Knobs:
  --grip  : distance along each flat edge from corner to attach point (~4.6 ref)
  --arcR  : radius of the boundary arcs (~9.1 ref)
"""
import argparse
import math
import os
import subprocess

import numpy as np
from PIL import Image


def concave_arc_d(p0, p1, R):
    """SVG arc command from p0 to p1, radius R, choosing the sweep flag so
    the arc is CONCAVE toward the gap center (32,32) (bows inward -> pinch).

    The two candidate arc CENTERS are chord_mid +/- perp*a. An arc bulges
    AWAY from its center, so the concave-toward-(32,32) arc is the one whose
    center is FARTHEST from (32,32). The center at chord_mid + perp*a
    (perp = p0->p1 rotated +90 deg, clockwise in y-down) corresponds to
    sweep=1.
    """
    dx, dy = p1[0]-p0[0], p1[1]-p0[1]
    d = math.hypot(dx, dy)
    if d >= 2*R:
        R = d/2 + 0.001
    a = math.sqrt(max(R*R - d*d/4, 0.0))
    mx, my = (p0[0]+p1[0])/2, (p0[1]+p1[1])/2
    ux, uy = -dy/d, dx/d
    c_plus = (mx+ux*a, my+uy*a)   # sweep=1 center
    c_minus = (mx-ux*a, my-uy*a)  # sweep=0 center
    d_plus = math.hypot(c_plus[0]-32, c_plus[1]-32)
    d_minus = math.hypot(c_minus[0]-32, c_minus[1]-32)
    sweep = 1 if d_plus > d_minus else 0  # farthest center -> concave
    return f"A {R:.3f} {R:.3f} 0 0 {sweep} {p1[0]:.3f} {p1[1]:.3f}"


def rounded_rect_path(x0, y0, x1, y1, cr, sharp_corners=()):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spacing", type=float, default=20.0)
    ap.add_argument("--r-center", type=float, default=10.0)
    ap.add_argument("--r-diag", type=float, default=6.0)
    ap.add_argument("--radius", type=float, default=0.75)
    ap.add_argument("--grip", type=float, default=4.6,
                    help="distance along each flat edge from corner to attach point")
    ap.add_argument("--arcR", type=float, default=9.1,
                    help="radius of the boundary arcs")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "step3e"))
    args = ap.parse_args()

    S, rc, rd = args.spacing, args.r_center, args.r_diag
    cr_c, cr_d = args.radius*rc, args.radius*rd
    g = args.grip

    C = (S+rc, S+rc)          # (30,30) center sharp corner
    B = (2*S-rd, 2*S-rd)      # (34,34) BR sharp corner

    U0 = (C[0], C[1]-g)       # (30, 30-g) on center right edge
    L0 = (C[0]-g, C[1])       # (30-g, 30) on center bottom edge
    B_left = (B[0], B[1]+g)   # (34, 34+g) on BR left edge
    B_top = (B[0]+g, B[1])    # (34+g, 34) on BR top edge

    # Lower boundary: L0 -> B_left (concave arc, bows toward (32,32))
    arcL = concave_arc_d(L0, B_left, args.arcR)
    # Upper boundary: B_top -> U0 (concave arc, bows toward (32,32))
    arcU = concave_arc_d(B_top, U0, args.arcR)

    d = (f"M {U0[0]:.3f} {U0[1]:.3f} "
         f"L {C[0]:.3f} {C[1]:.3f} "
         f"L {L0[0]:.3f} {L0[1]:.3f} "
         + arcL + " "
         f"L {B[0]:.3f} {B[1]:.3f} "
         f"L {B_top[0]:.3f} {B_top[1]:.3f} "
         + arcU + " Z")

    nodes = []
    cx, cy = S, S
    nodes.append(rounded_rect_path(cx-rc, cy-rc, cx+rc, cy+rc, cr_c, {'BR'}))
    cx, cy = 2*S, 2*S
    nodes.append(rounded_rect_path(cx-rd, cy-rd, cx+rd, cy+rd, cr_d, {'TL'}))
    for (gx, gy) in [(0,0), (2*S,0), (0,2*S)]:
        nodes.append(rounded_rect_path(gx-rd, gy-rd, gx+rd, gy+rd, cr_d, set()))

    pad = 3
    x_min, y_min = -rd-pad, -rd-pad
    x_max, y_max = 2*S+rd+pad, 2*S+rd+pad
    W, H = x_max-x_min, y_max-y_min

    nodepaths = "\n".join(f'    <path d="{n}"/>' for n in nodes)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W:.2f}mm" height="{H:.2f}mm" viewBox="{x_min:.2f} {y_min:.2f} {W:.2f} {H:.2f}">
  <rect x="{x_min:.2f}" y="{y_min:.2f}" width="{W:.2f}" height="{H:.2f}" fill="white"/>
  <g fill="black">
{nodepaths}
    <path d="{d}"/>
  </g>
</svg>'''

    os.makedirs(args.out, exist_ok=True)
    svgp = os.path.join(args.out, "step3e.svg")
    pngp = os.path.join(args.out, "step3e.png")
    with open(svgp, "w") as f:
        f.write(svg)
    subprocess.run(["rsvg-convert", "-w", "900", svgp, "-o", pngp], check=True)
    a = np.array(Image.open(pngp).convert("L"))
    print(f"U0{tuple(round(v,2) for v in U0)} L0{tuple(round(v,2) for v in L0)} "
          f"Bleft{tuple(round(v,2) for v in B_left)} Btop{tuple(round(v,2) for v in B_top)}")
    print(f"blackpx={(a<128).sum()}  -> {pngp}")


if __name__ == "__main__":
    main()
