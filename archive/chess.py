#!/usr/bin/env python3
"""
chess.py — STEP 2: apply a checkerboard (diagonal) mask to the binary grid.

Takes the 0/1 grid from grid.py and clears a checkerboard of cells:
only cells where (row + col) is EVEN survive. Result: no two filled
pixels touch diagonally — every filled pixel has its 4 diagonal
neighbors clear. Exactly the pattern of the 5 on a die.

Usage:
  chess.py grid48/grid.npy --out out_chess
  chess.py grid48/grid.npy --flip    # keep the other half (odd parity)
"""
import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser(description="checkerboard-mask a binary grid")
    ap.add_argument("grid", help="path to grid.npy (or grid.csv)")
    ap.add_argument("--out", default=None, help="output dir (default: alongside input)")
    ap.add_argument("--flip", action="store_true",
                    help="keep (row+col) ODD cells instead of EVEN")
    args = ap.parse_args()

    if args.grid.endswith(".csv"):
        g = np.loadtxt(args.grid, dtype=np.uint8, delimiter=",")
    else:
        g = np.load(args.grid)

    gh, gw = g.shape
    # checkerboard: (row + col) % 2 == 0  ->  like the 5 on a die
    rows, cols = np.indices((gh, gw))
    mask = ((rows + cols) % 2 == (1 if args.flip else 0))

    out = (g & mask).astype(np.uint8)
    before, after = int(g.sum()), int(out.sum())

    od = args.out or os.path.dirname(os.path.abspath(args.grid)) or "."
    os.makedirs(od, exist_ok=True)

    npy = os.path.join(od, "grid_chess.npy")
    np.save(npy, out)

    txt = os.path.join(od, "grid_chess.txt")
    with open(txt, "w") as f:
        for row in out:
            f.write("".join("#" if c else " " for c in row) + "\n")

    csv = os.path.join(od, "grid_chess.csv")
    np.savetxt(csv, out, fmt="%d", delimiter=",")

    from PIL import Image
    scale = 12
    big = np.kron(out, np.ones((scale, scale), dtype=np.uint8))
    rgb = np.zeros((*big.shape, 3), dtype=np.uint8)
    rgb[big == 0] = 255
    rgb[big == 1] = 0
    png = os.path.join(od, "grid_chess.png")
    Image.fromarray(rgb).save(png)

    print(f"[chess] {before} on-cells -> {after} (kept "
          f"{100.0 * after / max(before, 1):.1f}%)  parity={'odd' if args.flip else 'even'}")
    print(f"[emit] {png}\n[emit] {txt}\n[emit] {npy}")


if __name__ == "__main__":
    main()
