#!/usr/bin/env python3
"""
digitize.py — procedural generator for the organic black-dot stencil
effect. Binary black/negative-space output, guaranteed stencil-safe.

Pipeline:
  1. ingest    SVG (rsvg-convert) or PNG/JPG (Pillow) -> content mask
  2. mask      crop to content bounding box
  3. sample    Poisson-disc points inside mask (or regular grid)
  4. size      radius = density field x noise x edge falloff x random
  5. render    irregular blobs stamped into a binary buffer (OR-merge)
  6. topology  MANDATORY cleanup (iterated to stability):
                a. negative channels thinner than --min-channel closed
                   (morphological closing with a disk)
                b. black features smaller than --min-black removed
                c. enclosed negative holes filled with black
                   (flood fill from image border; unreachable -> black)
  7. emit      binary PNG + vectorized SVG (Suzuki-Abe contours)
  8. validate  hard re-check; refuses to exit 0 on failure

Dependencies: numpy, scipy, Pillow. External tool: rsvg-convert (SVG).
All lengths in millimetres; --dpi sets the raster scale.

Usage:
  python3 digitize.py input.png -o out/ --width 1200 --height 1500 \
      --dpi 150 --density 0.55 --min-dot 0.8 --max-blob 9 \
      --irregularity 0.35 --merge-prob 0.12 --noise-scale 0.18 \
      --edge-falloff 12 --min-channel 3 --min-black 2 --seed 42
"""
import argparse
import math
import os
import random
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

# ------------------------------------------------------------------ utils

def mm2px(mm, dpi):
    return mm * dpi / 25.4


def log(msg):
    print(msg, flush=True)


def disk_selem(r):
    r = int(round(r))
    if r <= 0:
        return np.ones((1, 1), bool)
    s = 2 * r + 1
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


# ------------------------------------------------------------------ ingest

def ingest_mask(path, target_w, target_h):
    """Return content mask (bool) at target resolution.
    Content = the figure. Polarity auto-fixed if the figure is
    light-on-dark (content would be the frame majority)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".svg":
        png = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        subprocess.run(
            ["rsvg-convert", "-w", str(target_w), "-h", str(target_h),
             "-o", png, path],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        img = Image.open(png).convert("RGBA")
        os.unlink(png)
        content = np.asarray(img.getchannel("A"), np.float32) / 255.0 > 0.5
    elif ext in (".png", ".jpg", ".jpeg", ".webp"):
        img = Image.open(path).convert("RGBA")
        img = img.resize((target_w, target_h), Image.LANCZOS)
        g = np.asarray(img.convert("L"), np.float32) / 255.0
        alpha = np.asarray(img.getchannel("A"), np.float32) / 255.0
        content = (g < 0.5) & (alpha > 0.5)
    else:
        raise SystemExit(f"unsupported input: {path}")

    if content.mean() > 0.5:
        content = ~content
        log("[mask] polarity inverted (figure was light-on-dark)")
    return content


def crop_to_content(mask, pad_px):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise SystemExit("input mask is empty — nothing to digitize")
    y0, y1 = max(ys.min() - pad_px, 0), min(ys.max() + pad_px, mask.shape[0])
    x0, x1 = max(xs.min() - pad_px, 0), min(xs.max() + pad_px, mask.shape[1])
    return mask[y0:y1 + 1, x0:x1 + 1]


# ------------------------------------------------------------------ sampling

def poisson_disc(w, h, mask, r_min, rng, k=30):
    """Bridson's Poisson-disk sampling restricted to mask pixels.
    Grid cell = r_min/sqrt(2); lookup window +/-2 cells (points within
    r_min can straddle two cell boundaries)."""
    pts = []
    cell = r_min / math.sqrt(2)
    grid = {}
    active = []

    def try_place(x, y):
        if x < 0 or y < 0 or x >= w - 1 or y >= h - 1:
            return False
        if not mask[int(y), int(x)]:
            return False
        gx, gy = int(x // cell), int(y // cell)
        for dy in (-2, -1, 0, 1, 2):
            for dx in (-2, -1, 0, 1, 2):
                for (px, py, _) in grid.get((gx + dx, gy + dy), ()):
                    if (px - x) ** 2 + (py - y) ** 2 < r_min * r_min:
                        return False
        return True

    def add(x, y):
        i = len(pts)
        pts.append((x, y))
        active.append(i)
        gx, gy = int(x // cell), int(y // cell)
        grid.setdefault((gx, gy), []).append((x, y, i))

    guard = 0
    while guard < 10000:
        guard += 1
        x = rng.uniform(0, w)
        y = rng.uniform(0, h)
        if try_place(x, y):
            add(x, y)
            break
    else:
        raise RuntimeError("poisson: could not seed")

    while active:
        ai = rng.randrange(len(active))
        x, y = pts[active[ai]]
        placed = False
        for _ in range(k):
            ang = rng.uniform(0, 2 * math.pi)
            rad = r_min * rng.uniform(1.0, 2.0)
            nx, ny = x + rad * math.cos(ang), y + rad * math.sin(ang)
            if try_place(nx, ny):
                add(nx, ny)
                placed = True
                break
        if not placed:
            active.pop(ai)
    return pts


def grid_points(w, h, mask, spacing, rng):
    """Regular grid with slight jitter (the geometric style)."""
    pts = []
    j = spacing * 0.06
    for gy in range(0, int(h // spacing) + 1):
        for gx in range(0, int(w // spacing) + 1):
            x = gx * spacing + spacing / 2 + rng.uniform(-j, j)
            y = gy * spacing + spacing / 2 + rng.uniform(-j, j)
            if 0 <= x < w and 0 <= y < h and mask[int(y), int(x)]:
                pts.append((x, y))
    return pts


# ------------------------------------------------------------------ noise

def value_noise_2d(shape, scale, rng):
    """Multi-octave value noise, normalized to [0,1]."""
    out = np.zeros(shape, dtype=np.float32)
    amp = 1.0
    total = 0.0
    for o in range(3):
        res = max(2, int(round(scale * (2 ** o))))
        gw, gh = shape[1] // res + 2, shape[0] // res + 2
        g = rng.random((gh, gw), dtype=np.float32)
        yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
        fy, fx = yy / res, xx / res
        gy0, gx0 = fy.astype(int), fx.astype(int)
        ty, tx = fy - gy0, fx - gx0
        ty = ty * ty * (3 - 2 * ty)
        tx = tx * tx * (3 - 2 * tx)
        a = g[gy0, gx0]
        b = g[gy0, gx0 + 1]
        c = g[gy0 + 1, gx0]
        d = g[gy0 + 1, gx0 + 1]
        out += amp * (a * (1 - tx) * (1 - ty) + b * tx * (1 - ty) +
                      c * (1 - tx) * ty + d * tx * ty)
        total += amp
        amp *= 0.5
    return out / total


def angular_profile(seed, n_theta):
    """Periodic 1D noise in [0,1] for blob-boundary irregularity."""
    rng = random.Random(seed)
    m = 8
    base = np.array([rng.random() for _ in range(m)], dtype=np.float32)
    idx = np.linspace(0, m, n_theta, endpoint=False)
    i0 = idx.astype(int) % m
    i1 = (i0 + 1) % m
    t = idx - i0
    t = t * t * (3 - 2 * t)
    return (base[i0] * (1 - t) + base[i1] * t).astype(np.float32)


# ------------------------------------------------------------------ render

def render_blobs(w, h, points, radii, profiles, n_theta):
    """Stamp irregular blobs (scanline polygon fill) into a bool buffer."""
    buf = np.zeros((h, w), dtype=bool)
    theta = np.linspace(0, 2 * math.pi, n_theta, endpoint=False)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    for (x, y), r, prof in zip(points, radii, profiles):
        if r < 0.5:
            continue
        rr = r * prof  # pre-shaped radius per angle
        bx = np.round(x + rr * cos_t).astype(np.int32)
        by = np.round(y + rr * sin_t).astype(np.int32)
        y0, y1 = by.min(), by.max()
        if y0 < 0 or y1 >= h:
            continue
        x0, x1 = bx.min(), bx.max()
        if x0 < 0 or x1 >= w:
            continue
        # scanline: for each row, find x-intervals inside the polygon
        # use edge crossing against each row line
        rows = {}
        for i in range(n_theta):
            a_x, a_y = bx[i], by[i]
            b_x, b_y = bx[(i + 1) % n_theta], by[(i + 1) % n_theta]
            if a_y == b_y:
                continue
            lo, hi = (a_y, b_y) if a_y < b_y else (b_y, a_y)
            for yy in range(max(lo, y0), min(hi, y1 + 1, h)):
                t = (yy - a_y) / (b_y - a_y)
                rows.setdefault(yy, []).append(a_x + t * (b_x - a_x))
        for yy, xs in rows.items():
            xs.sort()
            for i in range(0, len(xs) - 1, 2):
                xa = max(int(xs[i]), x0, 0)
                xb = min(int(xs[i + 1]), x1, w - 1)
                if xa <= xb:
                    buf[yy, xa:xb + 1] = True
    return buf


# ------------------------------------------------------------------ topology

def fill_enclosed_holes(black):
    """6c: flood negative space from the border; unreachable -> black."""
    neg = ~black
    if not neg.any():
        return black, 0
    reached, _ = ndi.label(neg, structure=np.ones((3, 3), bool))
    border_labels = np.unique(np.concatenate([
        reached[0, :], reached[-1, :], reached[:, 0], reached[:, -1]]))
    border_labels = border_labels[border_labels > 0]
    holes = np.isin(reached, [l for l in np.unique(reached[reached > 0])
                              if l not in set(border_labels.tolist())])
    n = int(holes.sum())
    if n:
        black = black | holes
    return black, n


def close_thin_channels(black, min_ch_px):
    """6a: close negative channels thinner than min_ch_px.
    Morphological closing of the black set with a disk of radius
    min_ch/2 — any negative region that fits inside that disk is
    filled (its blobs merge)."""
    if min_ch_px < 1:
        return black, 0
    r = max(1, int(round(min_ch_px / 2)))
    selem = disk_selem(r)
    closed = ndi.binary_closing(black, structure=selem)
    n = int((closed & ~black).sum())
    return (black | closed), n


def remove_small_features(black, min_area_px):
    """6b: remove black components with area < min_area_px."""
    labels, n = ndi.label(black, structure=np.ones((3, 3), bool))
    if n == 0:
        return black, 0
    sizes = ndi.sum(np.ones_like(labels), labels, range(1, n + 1))
    small = [i + 1 for i, s in enumerate(sizes) if s < min_area_px]
    if small:
        black = black & ~np.isin(labels, small)
    return black, len(small)


def topology_pass(black, min_ch_px, min_area_px, max_iter=10, close=True):
    """Iterate cleanup to stability.

    MANDATORY: fill enclosed negative holes (flood from border; unreachable
    -> black). This is the stencil-safety rule — a fully sealed negative
    pocket is impossible to cut.

    OPTIONAL (close=True): close negative channels thinner than min_ch_px and
    remove black features smaller than min_area_px. For a dot pattern these
    must be tiny (sliver cleanup only) or OFF, or they fuse the dots."""
    rep = {"holes_filled": 0, "channels_closed": 0,
           "features_removed": 0, "iterations": 0}
    for it in range(max_iter):
        rep["iterations"] = it + 1
        before = black.copy()
        if close:
            black, n = close_thin_channels(black, min_ch_px)
            rep["channels_closed"] += n
            black, n = remove_small_features(black, min_area_px)
            rep["features_removed"] += n
        black, n = fill_enclosed_holes(black)
        rep["holes_filled"] += n
        if (black == before).all():
            break
    return black, rep


def validate(black, min_area_px):
    """Hard final check. Returns list of violations (empty = pass)."""
    v = []
    neg = ~black
    if neg.any():
        reached, _ = ndi.label(neg, structure=np.ones((3, 3), bool))
        border = set(np.unique(np.concatenate([
            reached[0, :], reached[-1, :], reached[:, 0], reached[:, -1]]))
            .tolist())
        all_labs = set(np.unique(reached).tolist())
        enclosed = [l for l in all_labs if l > 0 and l not in border]
        if enclosed:
            n = int(np.isin(reached, enclosed).sum())
            v.append(f"enclosed negative pixels: {n}")
    labels, n = ndi.label(black, structure=np.ones((3, 3), bool))
    if n:
        sizes = ndi.sum(np.ones_like(labels), labels, range(1, n + 1))
        small = int((sizes < min_area_px).sum())
        if small:
            v.append(f"black features below min size: {small}")
    return v


# ------------------------------------------------------------------ svg
# The SVG is emitted directly from the raster via run-length encoding:
# one <rect> per horizontal black run. O(n), exact 1:1 with the PNG,
# and rect edges are ideal for a CNC stencil cut.

def mask_to_rle(black):
    """Run-length encode the black mask into horizontal runs.

    Returns a list of (y, x0, x1) where the run spans pixels x0..x1-1."""
    rows = []
    ys = np.flatnonzero(black.any(axis=1))
    for y in ys:
        row = black[y].view(np.int8)
        padded = np.concatenate(([-1], row, [0]))
        diff = np.diff(padded)
        starts = np.flatnonzero(diff == 1)
        ends = np.flatnonzero(diff == -1)
        for s, e in zip(starts, ends):
            rows.append((int(y), int(s), int(e)))
    return rows


def rle_to_svg(rows, w, h, width_mm, height_mm):
    """Emit RLE runs as black SVG rects on a white sheet."""
    parts = []
    for (y, s, e) in rows:
        parts.append(f'  <rect x="{s}" y="{y}" width="{e - s}" height="1"/>')
    body = "\n".join(parts)
    return (
        f'<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width_mm}mm" height="{height_mm}mm" '
        f'viewBox="0 0 {w} {h}">\n'
        f'  <rect width="{w}" height="{h}" fill="#ffffff"/>\n'
        f'{body}\n'
        f'</svg>\n'
    )


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="SVG or PNG/JPG figure")
    ap.add_argument("-o", "--out", default="out", help="output directory")
    ap.add_argument("--width", type=float, default=1200,
                    help="target width mm (default 1200)")
    ap.add_argument("--height", type=float, default=1500,
                    help="target height mm (default 1500)")
    ap.add_argument("--dpi", type=float, default=150,
                    help="raster scale dpi (default 150)")
    ap.add_argument("--density", type=float, default=0.55,
                    help="pattern density 0..1 (default 0.55)")
    ap.add_argument("--gap", type=float, default=2.2,
                    help="center spacing in x-min-dot-radii; higher = bigger "
                         "dots + fewer of them (default 2.2)")
    ap.add_argument("--fill", type=float, default=0.72,
                    help="dot size as fraction of cell; 1.0 = touching, "
                         "<1 = white gaps (default 0.72)")
    ap.add_argument("--min-dot", type=float, default=3.0,
                    help="base dot radius mm (default 3.0)")
    ap.add_argument("--max-blob", type=float, default=9,
                    help="(legacy) max blob radius mm — radii now capped by gap")
    ap.add_argument("--irregularity", type=float, default=0.35,
                    help="boundary irregularity 0..1 (default 0.35)")
    ap.add_argument("--merge-prob", type=float, default=0.12,
                    help="prob of large merged blob (default 0.12)")
    ap.add_argument("--noise-scale", type=float, default=0.18,
                    help="noise field scale (default 0.18)")
    ap.add_argument("--edge-falloff", type=float, default=12,
                    help="edge falloff width mm (default 12)")
    ap.add_argument("--min-channel", type=float, default=0.0,
                    help="sliver cleanup: close negative channels thinner "
                         "than this mm (default 0 = off)")
    ap.add_argument("--min-black", type=float, default=0.0,
                    help="sliver cleanup: remove black features smaller than "
                         "this diameter mm (default 0 = off)")
    ap.add_argument("--close", dest="close", action="store_true",
                    default=False,
                    help="enable sliver cleanup (channel-close + small-feature "
                         "removal). The mandatory enclosure-fill ALWAYS runs. "
                         "For dot patterns keep this OFF.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--style", choices=["poisson", "grid"],
                    default="poisson")
    ap.add_argument("--bbox", default=None,
                    help="input crop x0,y0,x1,y1 as 0-100 percents")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    W = int(round(mm2px(args.width, args.dpi)))
    H = int(round(mm2px(args.height, args.dpi)))
    log(f"[scale] {args.width}x{args.height}mm @ {args.dpi}dpi "
        f"-> {W}x{H}px")

    mask = ingest_mask(args.input, W, H)
    if args.bbox:
        x0, y0, x1, y1 = [float(v) for v in args.bbox.split(",")]
        mask = mask[int(y0 / 100 * H):int(y1 / 100 * H),
                    int(x0 / 100 * W):int(x1 / 100 * W)]
    pad = int(mm2px(args.min_dot, args.dpi))
    mask = crop_to_content(mask, pad)
    H, W = mask.shape
    log(f"[mask] content {W}x{H}px, fill {mask.mean():.0%}")

    min_r_px = mm2px(args.min_dot, args.dpi)
    # spacing = minimum CENTER-to-CENTER distance between dots (px).
    # `gap` = spacing in units of the base dot radius, so the base dot
    # diameter fills (2/gap) of the cell.
    spacing = min_r_px * args.gap
    # Dots stay separate as long as r < spacing/2. fill = r / (spacing/2),
    # i.e. 1.0 = dots just touching, <1 = white gaps.
    r_max = spacing * 0.5 * args.fill
    if args.style == "poisson":
        log(f"[sample] poisson-disc spacing={spacing:.1f}px "
            f"(gap {args.gap}x base, fill {args.fill:.2f}) ...")
        pts = poisson_disc(W, H, mask, spacing, rng)
    else:
        log(f"[sample] grid spacing={spacing * 1.2:.1f}px ...")
        pts = grid_points(W, H, mask, spacing * 1.2, rng)
    log(f"[sample] {len(pts)} points")

    # 4. size — radius as a fraction of the cell, with organic variation.
    # Strong size range (pinpricks -> big blobs) but capped at r_max so the
    # pattern never fuses into a solid mass.
    ns = max(0.02, args.noise_scale)
    field = value_noise_2d((H, W), ns * W / 8, np_rng)
    radii, profiles = [], []
    n_theta = 48
    for (x, y) in pts:
        f = field[int(y), int(x)]
        # base fraction of the cell, modulated by noise for organic clumping
        frac = args.fill * (0.45 + 0.95 * f)
        frac *= rng.uniform(0.75, 1.15)
        if rng.random() < args.merge_prob:
            frac *= rng.uniform(1.25, 1.6)  # occasional big amoeba blob
        frac = min(max(frac, 0.30 * args.fill), 1.0)
        r = spacing * 0.5 * frac
        prof_base = angular_profile(rng.randrange(10**6), n_theta)
        # dimensionless per-angle shape (~0.65..1.35); render_blobs applies
        # it as rr = r * prof, so store the shape NOT r*shape.
        prof = 1 + args.irregularity * (prof_base - 0.5) * 2
        radii.append(r)
        profiles.append(prof)
    log(f"[size] radius {min(radii):.2f}..{max(radii):.2f}px "
        f"(cell {spacing:.2f}px, cap {r_max:.2f}px)")

    # 5. render
    log("[render] stamping blobs ...")
    black = render_blobs(W, H, pts, radii, profiles, n_theta)
    log(f"[render] black coverage {black.mean():.0%}")

    # 6. topology (mandatory). Pad with a white border first so the
    # flood-fill's "outside" is the figure's background, not the canvas edge.
    # Without this, the whole figure fills the crop and every gap between
    # dots is "unreachable" -> everything fuses into a solid mass.
    min_ch_px = mm2px(args.min_channel, args.dpi)
    min_area_px = math.pi * (mm2px(args.min_black, args.dpi) / 2) ** 2
    pad = max(2, int(spacing))
    padded = np.pad(black, pad, mode="constant", constant_values=False)
    log(f"[topology] pad={pad}px min-channel={min_ch_px:.1f}px "
        f"min-area={min_area_px:.0f}px close={'on' if args.close else 'off'} ...")
    padded, rep = topology_pass(padded, min_ch_px, min_area_px,
                                close=args.close)
    black = padded[pad:-pad, pad:-pad]
    for k, v in rep.items():
        log(f"[topology] {k}: {v}")

    # 8. validate (hard)
    viol = validate(black, min_area_px)
    if viol:
        for v in viol:
            log(f"[VALIDATION FAILED] {v}")
        sys.exit(1)
    log("[validate] PASS — stencil-safe")

    # 7. emit
    os.makedirs(args.out, exist_ok=True)
    out_png = os.path.join(args.out, "out.png")
    out_svg = os.path.join(args.out, "out.svg")
    out_rep = os.path.join(args.out, "out.report.txt")

    img = Image.fromarray(np.where(black, 0, 255).astype(np.uint8), "L")
    img.save(out_png, dpi=(args.dpi, args.dpi))
    log(f"[emit] {out_png} ({os.path.getsize(out_png)} bytes)")

    log("[emit] run-length encoding ...")
    rows = mask_to_rle(black)
    log(f"[emit] {len(rows)} runs")
    out_w_mm = W * args.dpi / 25.4
    out_h_mm = H * args.dpi / 25.4
    svg = rle_to_svg(rows, W, H, out_w_mm, out_h_mm)
    with open(out_svg, "w") as f:
        f.write(svg)
    log(f"[emit] {out_svg} ({os.path.getsize(out_svg)} bytes)")

    report = (
        f"seed {args.seed}\ninput {args.input}\n"
        f"output {out_w_mm:.1f}x{out_h_mm:.1f}mm @ {args.dpi}dpi\n"
        f"style {args.style}\npoints {len(pts)}\n"
        f"black coverage {black.mean():.2%}\n"
        f"holes_filled {rep['holes_filled']}\n"
        f"channels_closed {rep['channels_closed']}\n"
        f"features_removed {rep['features_removed']}\n"
        f"iterations {rep['iterations']}\n"
        f"runs {len(rows)}\nvalidation PASS\n"
    )
    with open(out_rep, "w") as f:
        f.write(report)
    log(f"[emit] {out_rep}")
    log("[done]")


if __name__ == "__main__":
    main()
