#!/usr/bin/env python3
"""
size.py — STEP 3: distance-based cell sizing + checkerboard → SVG.

Each on-cell's rendered size is determined by how deep it sits inside
its connected component:
  - Edge cells (close to boundary) → small shapes.
  - Component center (deepest point) → full-size shapes.
  - The transition is a POWER CURVE over the full depth of the component.

SVG-native output:
  --radius 1.0   → <circle>  (perfect circles, dot-matrix)
  --radius 0.0   → <rect>    (sharp squares)
  0 < r < 1     → <rect rx>  (rounded squares)

All coordinates in mm. File is CNC-ready.

Usage:
  size.py grid24/grid.npy --out out_sized --width 200 --radius 1.0
  size.py grid24/grid.npy --out out_sized --width 200 --radius 0.5
  size.py grid24/grid.npy --out out_sized --power 1.15 --min-size 0.62 --radius 0.5
"""
import argparse
import os

import numpy as np
from scipy.ndimage import distance_transform_edt, label


def compute_sized(g: np.ndarray, power: float, min_size: float) -> np.ndarray:
    """
    Compute per-cell size (0..1) using per-component normalized distance.
    g: binary 2D array (0/1)
    Returns: float array same shape, 0 for off-cells, 0..1 for on-cells.
    """
    dist = distance_transform_edt(g)
    labels, n_comp = label(g)

    sized = np.zeros_like(g, dtype=np.float64)

    for comp_id in range(1, n_comp + 1):
        mask = labels == comp_id
        max_d = dist[mask].max()
        if max_d < 0.01:
            sized[mask] = min_size
            continue
        norm = np.clip(dist / max_d, 0, 1)
        curve = norm ** power
        sized[mask] = min_size + (1.0 - min_size) * curve[mask]

    sized[g == 0] = 0
    return sized


def emit_connections(sized: np.ndarray, width_mm: float, seed: int = 42,
                     bridge_frac: float = 0.45,
                     p_min: float = 0.15, p_max: float = 0.85,
                     curve: float = 0.25) -> tuple[str, int]:
    """
    Generate neighbor connections (variable count per dot) as tapered paths.

    Connection probability is SIZE-DRIVEN: each node's chance of linking to a
    given neighbor = lerp(p_min, p_max, normalized_size). Big blobs get dense
    linkages, small edge blobs stay sparse.

    Each bridge is a filled shape: a tapered band from blob A to blob B whose
    width at each end = bridge_frac * (that blob's rendered size), capped with
    a semicircle. The taper + caps make the bridge flow into the blobs as one
    continuous path instead of a separate line.
    Returns (svg_fragment, n_connections).
    """
    import math
    import random
    rng = random.Random(seed)
    gh, gw = sized.shape
    cell_mm = width_mm / gw
    pad = cell_mm / 2

    on_cells = [(r, c) for r in range(gh) for c in range(gw) if sized[r, c] > 0.01]
    on_set = set(on_cells)
    diags = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    # normalize node size for the probability distribution
    on_vals = sized[sized > 0.01]
    smin, smax = float(on_vals.min()), float(on_vals.max())

    def norm_size(s: float) -> float:
        if smax <= smin:
            return 1.0
        return (s - smin) / (smax - smin)

    drawn = set()
    lines = []
    n_conn = 0

    def center(r, c):
        return (pad + c * cell_mm + cell_mm / 2,
                pad + r * cell_mm + cell_mm / 2)

    for (r, c) in on_cells:
        neighbors = [(r + dr, c + dc) for dr, dc in diags
                     if (r + dr, c + dc) in on_set]
        if not neighbors:
            continue
        # size-driven probability: big node -> links to most neighbors
        p = p_min + (p_max - p_min) * norm_size(sized[r, c])
        # keep (dr, dc) with each neighbor so we know which side faces a hole
        cands = [((r + dr, c + dc), dr, dc) for dr, dc in diags
                 if (r + dr, c + dc) in on_set]
        chosen = [c for c in cands if rng.random() < p]
        for ((nr, nc), dr, dc) in chosen:
            pair = frozenset([(r, c), (nr, nc)])
            if pair in drawn:
                continue
            drawn.add(pair)

            ax, ay = center(r, c)
            bx, by = center(nr, nc)
            dx, dy = bx - ax, by - ay
            L = math.hypot(dx, dy)
            if L < 1e-6:
                continue
            ux, uy = dx / L, dy / L
            vx, vy = -uy, ux  # perpendicular

            # half-widths scale with each blob's rendered size
            hA = bridge_frac * (sized[r, c] * cell_mm) / 2
            hB = bridge_frac * (sized[nr, nc] * cell_mm) / 2
            if hA < 0.05 or hB < 0.05:
                continue

            # The two cells that a diagonal bridge A(r,c)->B(r+dr,c+dc) actually
            # borders are the axis-aligned cells (r+dr,c) and (r,c+dc). In a
            # checkerboard both are OFF (they are orthogonal neighbors of an
            # on-cell), so both long edges face holes. Determine which edge
            # (+v / -v) each borders via the sign of its offset dotted with v,
            # and bow that edge if the cell is off (outside grid counts as off).
            def off(rr, cc):
                if not (0 <= rr < gh and 0 <= cc < gw):
                    return True
                return sized[rr, cc] < 0.01

            v_off = False
            n_off = False
            for (br, bc) in ((r + dr, c), (r, c + dc)):
                ox, oy = bc - c, br - r
                side = ox * (-uy) + oy * ux  # dot with v
                if off(br, bc):
                    if side > 0:
                        v_off = True
                    else:
                        n_off = True

            midx, midy = (ax + bx) / 2, (ay + by) / 2
            # Each long edge bows OUTWARD on its own side: the +v edge's
            # control point is pushed further along +v, the -v edge's further
            # along -v, each by `curve * L`. Four bridges meeting around a gap
            # then sweep their inner corners into a circle (verified
            # empirically: +bow = round holes, -bow = diamonds; offset is ±v,
            # NOT ±u — ±u would S-warp the edge along its length).
            bow_v = curve * L if v_off else 0.0
            bow_n = curve * L if n_off else 0.0
            c1x = midx + ((hA + hB) / 2 + bow_v) * vx   # +v edge control
            c1y = midy + ((hA + hB) / 2 + bow_v) * vy
            c2x = midx - ((hA + hB) / 2 + bow_n) * vx   # -v edge control
            c2y = midy - ((hA + hB) / 2 + bow_n) * vy

            p1x, p1y = ax + hA * vx, ay + hA * vy   # A, +v side
            p2x, p2y = bx + hB * vx, by + hB * vy   # B, +v side
            p3x, p3y = bx - hB * vx, by - hB * vy   # B, -v side
            p4x, p4y = ax - hA * vx, ay - hA * vy   # A, -v side

            # Path order: p1 -> p2 (+v edge) -> cap B -> p3 -> p4 (-v edge) -> cap A
            # +v edge: quadratic through c1.  -v edge: quadratic through c2.
            lines.append(
                f'    <path d="M {p1x:.3f} {p1y:.3f} '
                f'Q {c1x:.3f} {c1y:.3f} {p2x:.3f} {p2y:.3f} '
                f'A {hB:.3f} {hB:.3f} 0 0 1 {p3x:.3f} {p3y:.3f} '
                f'Q {c2x:.3f} {c2y:.3f} {p4x:.3f} {p4y:.3f} '
                f'A {hA:.3f} {hA:.3f} 0 0 1 {p1x:.3f} {p1y:.3f} Z"/>')
            n_conn += 1

    return "\n".join(lines), n_conn


def emit_svg(sized: np.ndarray, width_mm: float, radius: float,
             connect: bool = False, seed: int = 42,
             bridge_frac: float = 0.45,
             p_min: float = 0.15, p_max: float = 0.85,
             curve: float = 0.25) -> tuple[str, int, int]:
    """
    Emit SVG from the sized grid.
    Returns (svg_string, n_shapes, n_connections).
    """
    gh, gw = sized.shape
    height_mm = width_mm * gh / gw
    cell_mm = width_mm / gw

    pad = cell_mm / 2
    svg_w = width_mm + 2 * pad
    svg_h = height_mm + 2 * pad

    lines = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" '
                 f'width="{svg_w:.3f}mm" height="{svg_h:.3f}mm" '
                 f'viewBox="0 0 {svg_w:.3f} {svg_h:.3f}">')
    lines.append(f'  <rect width="100%" height="100%" fill="white"/>')

    # connections layer (drawn first, under dots)
    n_conn = 0
    if connect:
        conn_svg, n_conn = emit_connections(sized, width_mm, seed, bridge_frac,
                                            p_min, p_max, curve)
        if conn_svg:
            lines.append(f'  <g id="connections" fill="black">')
            lines.append(conn_svg)
            lines.append(f'  </g>')

    # dots layer
    lines.append(f'  <g fill="black" id="dots">')

    n = 0
    for r in range(gh):
        for c in range(gw):
            s = sized[r, c]
            if s < 0.01:
                continue
            cx = pad + c * cell_mm + cell_mm / 2
            cy = pad + r * cell_mm + cell_mm / 2
            side = s * cell_mm

            if radius >= 0.99:
                rad = side / 2
                lines.append(f'    <circle cx="{cx:.3f}" cy="{cy:.3f}" '
                             f'r="{rad:.3f}"/>')
            elif radius <= 0.01:
                x = cx - side / 2
                y = cy - side / 2
                lines.append(f'    <rect x="{x:.3f}" y="{y:.3f}" '
                             f'width="{side:.3f}" height="{side:.3f}"/>')
            else:
                x = cx - side / 2
                y = cy - side / 2
                rx = radius * side / 2
                lines.append(f'    <rect x="{x:.3f}" y="{y:.3f}" '
                             f'width="{side:.3f}" height="{side:.3f}" '
                             f'rx="{rx:.3f}" ry="{rx:.3f}"/>')
            n += 1

    lines.append('  </g>')
    lines.append('</svg>')
    return "\n".join(lines), n, n_conn


def render_preview(svg_path: str, png_path: str, width_px: int = 800):
    """Render SVG → PNG for quick visual check."""
    import subprocess
    subprocess.run(
        ["rsvg-convert", "-w", str(width_px), svg_path, "-o", png_path],
        check=True,
    )


def main():
    ap = argparse.ArgumentParser(description="distance-based cell sizing → SVG")
    ap.add_argument("grid", help="path to grid.npy or grid.csv (binary 0/1)")
    ap.add_argument("--out", default=None, help="output dir")
    ap.add_argument("--width", type=float, default=200.0,
                    help="SVG width in mm (default 200)")
    ap.add_argument("--power", type=float, default=2.5,
                    help="curve exponent: >1 concave (only center full), "
                         "=1 linear, <1 convex (default 2.5)")
    ap.add_argument("--min-size", type=float, default=0.25,
                    help="minimum size for edge cells (0.0..1.0, default 0.25)")
    ap.add_argument("--checker", action="store_true", default=True,
                    help="apply checkerboard mask (default on)")
    ap.add_argument("--no-checker", dest="checker", action="store_false",
                    help="disable checkerboard mask")
    ap.add_argument("--flip", action="store_true",
                    help="flip checkerboard parity")
    ap.add_argument("--radius", type=float, default=0.0,
                    help="corner roundness 0..1 (0=square, 1=circle) (default 0)")
    ap.add_argument("--connect", action="store_true",
                    help="enable random neighbor connections (1-3 per dot)")
    ap.add_argument("--seed", type=int, default=42,
                    help="random seed for connections (default 42)")
    ap.add_argument("--bridge", type=float, default=0.45,
                    help="bridge width as fraction of each blob's size "
                         "(tapered: each end scales with its own blob) (default 0.45)")
    ap.add_argument("--p-min", type=float, default=0.15,
                    help="connection probability for the SMALLEST blob (default 0.15)")
    ap.add_argument("--p-max", type=float, default=0.85,
                    help="connection probability for the LARGEST blob (default 0.85)")
    ap.add_argument("--curve", type=float, default=0.25,
                    help="bridge edge curvature 0..1 (0=straight, ~0.25=smooth "
                         "arcs/round holes, >0.4=pinches/scallops) (default 0.25)")
    args = ap.parse_args()

    if args.grid.endswith(".csv"):
        g = np.loadtxt(args.grid, dtype=np.uint8, delimiter=",")
    else:
        g = np.load(args.grid)
    gh, gw = g.shape

    # per-component normalized distance → size
    sized = compute_sized(g, power=args.power, min_size=args.min_size)

    # checkerboard
    if args.checker:
        rows, cols = np.indices((gh, gw))
        parity = (rows + cols) % 2
        mask = (parity == (1 if args.flip else 0))
        sized = np.where(mask, sized, 0)

    n_on = int((sized > 0).sum())
    vals = sized[sized > 0]
    print(f"[size] {gh}x{gw} grid, {n_on} cells on")
    print(f"[size] power={args.power} min-size={args.min_size} "
          f"checker={'on' if args.checker else 'off'}")
    print(f"[size] size range: {vals.min():.2f} .. {vals.max():.2f} "
          f"(mean {vals.mean():.2f})")
    print(f"[size] radius={args.radius} width={args.width}mm")

    od = args.out or os.path.dirname(os.path.abspath(args.grid)) or "."
    os.makedirs(od, exist_ok=True)

    # save sized grid (for downstream steps like connection)
    np.save(os.path.join(od, "sized.npy"), sized)

    # emit SVG (primary output)
    svg_str, n_shapes, n_conn = emit_svg(sized, args.width, args.radius,
                                         connect=args.connect, seed=args.seed,
                                         bridge_frac=args.bridge,
                                         p_min=args.p_min, p_max=args.p_max,
                                         curve=args.curve)
    svg_path = os.path.join(od, "sized.svg")
    with open(svg_path, "w") as f:
        f.write(svg_str)
    print(f"[emit] {svg_path} ({n_shapes} shapes, {n_conn} connections, "
          f"{os.path.getsize(svg_path) / 1024:.1f} KB)")

    # render PNG preview
    png_path = os.path.join(od, "sized.png")
    render_preview(svg_path, png_path)
    print(f"[emit] {png_path} (preview)")

    # ASCII
    with open(os.path.join(od, "sized.txt"), "w") as f:
        for row in sized:
            line = ""
            for v in row:
                if v < 0.01:
                    line += "."
                elif v < 0.4:
                    line += "o"
                elif v < 0.7:
                    line += "O"
                else:
                    line += "#"
            f.write(line + "\n")


if __name__ == "__main__":
    main()
