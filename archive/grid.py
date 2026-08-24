#!/usr/bin/env python3
"""
grid.py — STEP 1: rasterize an SVG into a binary grid (pixelization).

What it does, nothing more:
  1. Render the SVG to a smooth high-res raster (anti-aliased alpha).
  2. Crop to the shape's bounding box.
  3. Downsample to a grid of `res` cells wide (height follows aspect).
     Each cell is the AREA-AVERAGE of the shape inside it (0.0..1.0).
  4. Threshold at `--threshold` -> each cell becomes 0 or 1.
     1 = filled (part of the shape), 0 = empty (background).

That's it. No dots, no topology, no sizing. Just the shape turned into a
binary matrix. Everything downstream builds on this grid.

Outputs (in --out):
  grid.txt   ASCII art of the grid (# = filled, space = empty)
  grid.png   the grid as image blocks (each cell = a square)
  grid.npy   the raw 0/1 matrix (numpy)
  grid.csv   the raw 0/1 matrix (CSV)

Usage:
  grid.py input.svg --res 64
  grid.py input.svg --res 120 --threshold 0.5 --out out/
"""
import argparse
import os
import subprocess
import sys

import numpy as np


def render_svg_alpha(svg_path: str, target_width_px: int = 2000):
    """Render SVG -> (H x W) float array of alpha 0..1 via rsvg-convert."""
    png = svg_path + ".tmp.png"
    subprocess.run(
        ["rsvg-convert", "-w", str(target_width_px),
         "--keep-aspect-ratio", svg_path, "-o", png],
        check=True,
    )
    from PIL import Image
    im = Image.open(png).convert("RGBA")
    os.remove(png)
    a = np.asarray(im, dtype=np.float64) / 255.0
    # alpha channel; fall back to luminance if no alpha
    alpha = a[..., 3] if a.shape[2] == 4 else a.mean(axis=2)
    return alpha


def crop_to_content(mask: np.ndarray, pad: int = 0):
    """Crop a 2D mask to its non-zero bounding box (+ optional pad)."""
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return mask, (0, 0, mask.shape[1], mask.shape[0])
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, mask.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, mask.shape[1])
    return mask[y0:y1, x0:x1], (x0, y0, x1, y1)


def area_average_grid(alpha: np.ndarray, gw: int, gh: int) -> np.ndarray:
    """
    Downsample `alpha` (HxW, 0..1) to a gw x gh grid by exact area average.
    Uses an integral image (cumulative sum) so each output cell is the true
    mean of every sub-pixel that falls inside it. gw x gh cells.
    """
    H, W = alpha.shape
    # integral image with a 0 border for easy rectangle sums
    S = np.zeros((H + 1, W + 1), dtype=np.float64)
    S[1:, 1:] = np.cumsum(np.cumsum(alpha, axis=0), axis=1)

    # cell i covers rows [y0,y1), cols [x0,x1) in the source
    row = np.linspace(0, H, gh + 1)
    col = np.linspace(0, W, gw + 1)
    y0 = row[:-1].astype(np.int64)[:, None]          # (gh,1)
    y1 = row[1:].astype(np.int64)[:, None]           # (gh,1)
    x0 = col[:-1].astype(np.int64)[None, :]          # (1,gw)
    x1 = col[1:].astype(np.int64)[None, :]           # (1,gw)

    top = S[y0, x0]
    bot = S[y1, x1]
    lft = S[y0, x1]
    rgt = S[y1, x0]
    area = (y1 - y0) * (x1 - x0)
    grid = (bot - lft - rgt + top) / area
    return grid


def main():
    ap = argparse.ArgumentParser(description="SVG -> binary grid (pixelization)")
    ap.add_argument("svg")
    ap.add_argument("--res", type=int, default=64,
                    help="number of grid cells across the width (default 64)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="cell fill fraction that counts as 'on' (default 0.5)")
    ap.add_argument("--out", default="grid_out", help="output dir")
    ap.add_argument("--render-w", type=int, default=2000,
                    help="internal render width in px (default 2000)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # 1. render + 2. crop
    alpha = render_svg_alpha(args.svg, args.render_w)
    crop, bbox = crop_to_content(alpha)
    H, W = crop.shape
    print(f"[render] {args.svg} -> alpha {alpha.shape[::-1]} (w x h px)")
    print(f"[crop]   content bbox x{bbox[0]}..{bbox[2]} y{bbox[1]}..{bbox[3]} "
          f"-> {W}x{H}px")

    # 3. grid (width = res, height follows aspect)
    res = args.res
    gh = max(1, round(res * H / W))
    grid = area_average_grid(crop, res, gh)          # gh x res, 0..1
    print(f"[grid]   {res} x {gh} cells  (fill range "
          f"{grid.min():.2f}..{grid.max():.2f})")

    # 4. threshold -> binary
    binary = (grid >= args.threshold).astype(np.uint8)
    n_on = int(binary.sum())
    print(f"[binary] threshold {args.threshold} -> {n_on}/{binary.size} cells "
          f"on ({100.0 * n_on / binary.size:.1f}%)")

    # outputs
    npy = os.path.join(args.out, "grid.npy")
    np.save(npy, binary)

    txt = os.path.join(args.out, "grid.txt")
    with open(txt, "w") as f:
        for row in binary:
            f.write("".join("#" if c else " " for c in row) + "\n")

    csv = os.path.join(args.out, "grid.csv")
    np.savetxt(csv, binary, fmt="%d", delimiter=",")

    png = os.path.join(args.out, "grid.png")
    from PIL import Image
    scale = 12  # px per cell in the preview
    big = np.kron(binary, np.ones((scale, scale), dtype=np.uint8))
    rgb = np.zeros((*big.shape, 3), dtype=np.uint8)
    rgb[big == 0] = 255
    rgb[big == 1] = 0
    Image.fromarray(rgb).save(png)

    print(f"[emit] {txt}\n[emit] {png}\n[emit] {npy}\n[emit] {csv}")

    # ASCII preview (downsampled if too wide to read in a terminal)
    print("\n" + "=" * 44)
    print(f"  BINARY GRID  {res} x {gh}   ({n_on} cells on)")
    print("=" * 44)
    print(ascii_preview(binary, max_cols=60))


def ascii_preview(binary: np.ndarray, max_cols: int = 60) -> str:
    """Render the binary grid as ASCII, downsampled to fit `max_cols` wide."""
    gh, gw = binary.shape
    if gw <= max_cols:
        return "\n".join("".join("#" if c else "." for c in row)
                         for row in binary)
    # area-average down to max_cols wide
    new_gh = max(1, round(gh * max_cols / gw))
    small = area_average_grid(binary.astype(np.float64), max_cols, new_gh)
    return "\n".join("".join("#" if c > 0.5 else "." for c in row)
                     for row in small)


if __name__ == "__main__":
    main()
