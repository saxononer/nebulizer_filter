#!/usr/bin/env python3
"""
blob.py — STEP 4 (METABALL): distance-field fusion → smooth SVG.

Replaces the bridge approach. Every on-cell is a metaball whose strength
scales with its distance-derived size. The black region is where the summed
inverse-square field crosses a threshold. Boundaries between lobes are
concave (soap-film) by construction; gaps where 3-4 lobes meet are
curvilinear.

    F(x,y) = sum_i r_i^2 / d_i^2        (r_i in mm, d in mm)
    black  = F >= thresh

Pipeline: grid.npy --(size)--> strengths --(field)--> threshold
          --(marching squares)--> polylines --(Chaikin + Catmull-Rom)--> SVG

No checkerboard: the field fuses all lobes of the figure into one mass,
which is the target aesthetic.

Usage:
  blob.py test/grid24/grid.npy --out test/blob1 --width 200 \
      --radius 0.45 --thresh 2.2 --power 1.15 --min-size 0.62
"""
import argparse
import os

import numpy as np
from scipy.ndimage import distance_transform_edt, label

from size import compute_sized, render_preview


# ── contour extraction (marching squares) ────────────────────────────

def marching_squares(f: np.ndarray, thresh: float,
                     smooth_sigma: float = 0.6,
                     min_loop_pts: int = 24) -> list[np.ndarray]:
    """
    Extract closed contours where field `f` crosses `thresh`.

    Marching squares interpolates the crossing point on each cell edge from
    the actual field values, so the contour follows the field's true shape —
    including the concave soap-film necks between metaballs. (A pixel-edge
    tracer like Moore neighbourhood produces an axis-aligned staircase and
    can't represent concave necks.)

    Two robustness measures:
      * a light Gaussian blur of the field first kills the near-threshold
        micro-wiggles (degenerate cells) that fragment the contour;
      * contour segments are assembled into cycles as an UNDIRECTED graph
        (Hierholzer), so edge-table orientation can't break the walk.

    Returns list of (N,2) float arrays in pixel coords (x=col, y=row),
    each a closed loop.
    """
    from scipy.ndimage import gaussian_filter
    if smooth_sigma > 0:
        f = gaussian_filter(f, sigma=smooth_sigma)

    h, w = f.shape
    if h < 2 or w < 2:
        return []

    def lerp(v0, v1, p0, p1):
        denom = v1 - v0
        t = 0.5 if abs(denom) < 1e-12 else (thresh - v0) / denom
        t = min(max(t, 0.0), 1.0)
        return (p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1]))

    segs = []
    # precompute edge points only for cells that actually have a crossing
    for i in range(h - 1):
        for j in range(w - 1):
            tl, tr = f[i, j], f[i, j + 1]
            bl, br = f[i + 1, j], f[i + 1, j + 1]
            m = (0 if tl >= thresh else 1) | (0 if tr >= thresh else 2) \
                | (0 if br >= thresh else 4) | (0 if bl >= thresh else 8)
            if m == 0 or m == 15:
                continue
            P = {
                "T": lerp(tl, tr, (j, i), (j + 1, i)),
                "R": lerp(tr, br, (j + 1, i), (j + 1, i + 1)),
                "B": lerp(bl, br, (j, i + 1), (j + 1, i + 1)),
                "L": lerp(tl, bl, (j, i), (j, i + 1)),
            }
            center = (tl + tr + br + bl) / 4
            hi = center >= thresh
            if m in (1, 14, 2, 13, 4, 11, 8, 7, 3, 12, 6, 9):
                single = {1: ("L", "T"), 14: ("T", "L"), 2: ("T", "R"),
                          13: ("R", "T"), 4: ("B", "R"), 11: ("R", "B"),
                          8: ("L", "B"), 7: ("B", "L"), 3: ("L", "R"),
                          12: ("R", "L"), 6: ("T", "B"), 9: ("L", "R")}
                a, b = single[m]
                segs.append((P[a], P[b]))
            elif m == 10:
                if hi:
                    segs.append((P["L"], P["T"])); segs.append((P["R"], P["B"]))
                else:
                    segs.append((P["L"], P["B"])); segs.append((P["T"], P["R"]))
            elif m == 5:
                if hi:
                    segs.append((P["T"], P["R"])); segs.append((P["L"], P["B"]))
                else:
                    segs.append((P["L"], P["T"])); segs.append((P["R"], P["B"]))

    if not segs:
        return []

    # assemble into closed cycles via undirected graph + Hierholzer
    key = lambda p: (round(p[0], 4), round(p[1], 4))
    from collections import defaultdict
    edges = defaultdict(list)
    for (a, b) in segs:
        ka, kb = key(a), key(b)
        edges[ka].append(kb)
        edges[kb].append(ka)

    loops = []
    for start in list(edges.keys()):
        if not edges.get(start):
            continue
        stack = [start]
        circuit = []
        while stack:
            v = stack[-1]
            if edges[v]:
                stack.append(edges[v].pop())
            else:
                circuit.append(stack.pop())
        circuit.reverse()
        if len(circuit) >= min_loop_pts and circuit[0] == circuit[-1]:
            pts = circuit[:-1]
            loops.append(np.array(pts))

    # dedupe
    seen = set()
    uniq = []
    for lp in loops:
        sig = (round(lp[0, 0], 2), round(lp[0, 1], 2), len(lp))
        if sig in seen:
            continue
        seen.add(sig)
        uniq.append(lp)
    return uniq


# ── smoothing ────────────────────────────────────────────────────────

def chaikin(pts: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Chaikin corner cutting on a closed loop (in pixel coords)."""
    p = pts.astype(float)
    for _ in range(iterations):
        n = len(p)
        q = np.empty((n * 2, 2))
        for i in range(n):
            p0 = p[i]
            p1 = p[(i + 1) % n]
            q[2 * i] = 0.75 * p0 + 0.25 * p1
            q[2 * i + 1] = 0.25 * p0 + 0.75 * p1
        p = q
    return p


def catmull_rom_path(pts: np.ndarray, scale: float) -> str:
    """Closed Catmull-Rom loop -> cubic bezier SVG path (mm coords)."""
    n = len(pts)
    if n < 3:
        return ""
    # resample evenly so the spline doesn't kink at uneven marching-squares steps
    d = np.hypot(np.diff(np.vstack([pts, pts[:1]]), axis=0)[:, 0],
                 np.diff(np.vstack([pts, pts[:1]]), axis=0)[:, 1])
    cum = np.concatenate([[0], np.cumsum(d)])
    total = cum[-1]
    m = max(n, 32)
    ts = np.linspace(0, total, m, endpoint=False)
    xs = np.interp(ts, cum, np.append(pts[:, 0], pts[0, 0]))
    ys = np.interp(ts, cum, np.append(pts[:, 1], pts[0, 1]))
    r = np.column_stack([xs, ys]) * scale

    parts = [f"M {r[0, 0]:.3f} {r[0, 1]:.3f}"]
    for i in range(m):
        p0 = r[(i - 1) % m]
        p1 = r[i]
        p2 = r[(i + 1) % m]
        p3 = r[(i + 2) % m]
        c1 = p1 + (p2 - p0) / 6
        c2 = p2 - (p3 - p1) / 6
        parts.append(f"C {c1[0]:.3f} {c1[1]:.3f} {c2[0]:.3f} {c2[1]:.3f} "
                     f"{p2[0]:.3f} {p2[1]:.3f}")
    parts.append("Z")
    return " ".join(parts)


# ── main pipeline ────────────────────────────────────────────────────

def build_field(strengths: np.ndarray, width_mm: float,
                samples: int = 8) -> tuple[np.ndarray, float]:
    """
    strengths: grid-shaped array of per-cell radii in mm (0 = off).
    Field is extended by one full cell on every side so edge lobes are
    never clipped. Returns (field array, mm-per-pixel).
    """
    gh, gw = strengths.shape
    cell_mm = width_mm / gw
    h, w = (gh + 2) * samples, (gw + 2) * samples
    mpp = cell_mm / samples  # mm per pixel

    # node centers in pixel coords (field space), offset by one cell
    cx = (np.arange(gw) + 1.5) * samples
    cy = (np.arange(gh) + 1.5) * samples
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))

    F = np.zeros((h, w))
    for r in range(gh):
        for c in range(gw):
            rad = strengths[r, c]
            if rad <= 0:
                continue
            dx = xs - cx[c]
            dy = ys - cy[r]
            d2 = (dx * dx + dy * dy) * (mpp * mpp) + 1e-9
            F += (rad * rad) / d2
    return F, mpp


def main():
    ap = argparse.ArgumentParser(description="metaball fusion → smooth SVG")
    ap.add_argument("grid", help="grid.npy or grid.csv (binary 0/1)")
    ap.add_argument("--out", default=None, help="output dir")
    ap.add_argument("--width", type=float, default=200.0,
                    help="SVG width in mm (default 200)")
    ap.add_argument("--radius", type=float, default=0.45,
                    help="base metaball radius as fraction of cell "
                         "(default 0.45 = tuned match #3)")
    ap.add_argument("--thresh", type=float, default=2.2,
                    help="field threshold (default 2.2 = tuned match #3)")
    ap.add_argument("--power", type=float, default=1.15,
                    help="sizing curve exponent (default 1.15)")
    ap.add_argument("--min-size", type=float, default=0.62,
                    help="min size fraction for edge cells (default 0.62)")
    ap.add_argument("--samples", type=int, default=8,
                    help="field samples per cell (default 8, higher=smoother)")
    ap.add_argument("--smooth", type=int, default=2,
                    help="Chaikin iterations 0-3 (default 2)")
    args = ap.parse_args()

    g = (np.loadtxt(args.grid, dtype=np.uint8, delimiter=",")
         if args.grid.endswith(".csv") else np.load(args.grid))
    gh, gw = g.shape

    sized = compute_sized(g, power=args.power, min_size=args.min_size)
    vals = sized[sized > 0]
    cell_mm = args.width / gw
    strengths = sized * args.radius * cell_mm  # per-cell radius in mm

    od = args.out or os.path.dirname(os.path.abspath(args.grid)) or "."
    os.makedirs(od, exist_ok=True)

    print(f"[blob] {gh}x{gw} grid, {int((sized > 0).sum())} lobes")
    print(f"[blob] radius={args.radius} (of cell) thresh={args.thresh} "
          f"power={args.power} min-size={args.min_size}")
    print(f"[blob] lobe radius range: "
          f"{strengths[strengths > 0].min():.2f} .. "
          f"{strengths[strengths > 0].max():.2f} mm")

    F, mpp = build_field(strengths, args.width, args.samples)
    black = F >= args.thresh
    print(f"[blob] field {F.shape[1]}x{F.shape[0]} px, "
          f"black fraction {black.mean():.2f}")

    loops = marching_squares(F, args.thresh)
    print(f"[blob] {len(loops)} contour loops")

    # canvas = full field extent (figure sits one cell in from the edge)
    svg_w = F.shape[1] * mpp
    svg_h = F.shape[0] * mpp

    paths = []
    for loop in loops:
        if args.smooth > 0:
            loop = chaikin(loop, args.smooth)
        d = catmull_rom_path(loop, mpp)
        if d:
            paths.append(f'    <path d="{d}"/>')

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" '
               f'width="{svg_w:.3f}mm" height="{svg_h:.3f}mm" '
               f'viewBox="0 0 {svg_w:.3f} {svg_h:.3f}">')
    svg.append(f'  <rect width="100%" height="100%" fill="white"/>')
    svg.append(f'  <g fill="black" fill-rule="evenodd">')
    svg.extend(paths)
    svg.append(f'  </g>')
    svg.append(f'</svg>')
    svg_str = "\n".join(svg)

    svg_path = os.path.join(od, "blob.svg")
    with open(svg_path, "w") as f:
        f.write(svg_str)
    print(f"[emit] {svg_path} ({len(paths)} paths, "
          f"{os.path.getsize(svg_path) / 1024:.1f} KB)")

    png_path = os.path.join(od, "blob.png")
    render_preview(svg_path, png_path)
    print(f"[emit] {png_path} (preview)")


if __name__ == "__main__":
    main()
