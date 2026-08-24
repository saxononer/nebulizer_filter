#!/usr/bin/env python3
"""
node.py — the LOCKED 2-node / multi-node primitive (2026-08-23).

Design contract (agreed with Saxon):
  * Nodes are circles of radius r at a FIXED spacing. Spacing does NOT vary;
    the radius (size) is what changes.
  * Bridging is an EXPLICIT, TOGGLEABLE per-pair decision. It is NOT a
    metaball field: node-to-node distance never determines whether a bridge
    exists or how thick it is.
  * Bridge thickness = r1 + r2 (sum of the two node radii). Bigger nodes
    -> fatter bridge.
  * The bridge is a capsule (bar with rounded ends) laid center-to-center.
    Its ends hide inside the nodes, so the only visible boundary is the
    concave soap-film pinch where the bar meets each node.

Field (threshold 1.0):
  F(x,y) =  SUM over nodes   r_i^2 / dist^2(x, c_i)
          + SUM over bridges  rb^2  / dist^2(x, capsule_i)
  where rb = (r1+r2)/2 for the pair. Each term == 1 exactly on its own
  isosurface (the circle of radius r / the capsule of half-thickness rb),
  and the black region is F >= 1. Because the terms are summed (classic
  metaball union), the boundary pinches IN concavely where a capsule meets
  a node — the approved "soap-film" look — without distance coupling.

Contour: direct marching-squares walk (interior-on-left orientation table,
derived geometrically so shared edges never create T-junctions).

SVG: closed loops -> Catmull-Rom beziers (reuses blob.catmull_rom_path).
PNG: rsvg-convert + a printed black-pixel count, so a blank render is caught
immediately instead of eyeballed.
"""
import argparse
import os
import subprocess
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blob import catmull_rom_path


# ── field ────────────────────────────────────────────────────────────

def _ball_term(gx, gy, cx, cy, r):
    d2 = (gx - cx) ** 2 + (gy - cy) ** 2 + 1e-9
    return r * r / d2


def _capsule_term(gx, gy, ax, ay, bx, by, rb):
    """Capsule between (ax,ay)-(bx,by), half-thickness rb.
    Term == 1 exactly at distance rb from the axis segment."""
    abx, aby = bx - ax, by - ay
    t = np.clip(((gx - ax) * abx + (gy - ay) * aby) / (abx * abx + aby * aby
               + 1e-12), 0.0, 1.0)
    px = gx - (ax + t * abx)
    py = gy - (ay + t * aby)
    d2 = px * px + py * py + 1e-9
    return rb * rb / d2


def build_field(nodes, bridges, density=40.0, bounds=None):
    """
    nodes   : list of (cx, cy, r)
    bridges : list of (ax, ay, bx, by, rb)   # rb = half-thickness
    bounds  : (x0, x1, y0, y1) world-mm, default computed from nodes
    density : samples per mm
    Returns (F, x0, y0, mpp).
    """
    mpp = 1.0 / density
    if bounds is None:
        if nodes:
            xs = [c[0] for c in nodes]
            ys = [c[1] for c in nodes]
            rs = [c[2] for c in nodes]
        else:
            xs, ys, rs = [0], [0], [1]
        for b in bridges:
            xs += [b[0], b[2]]
            ys += [b[1], b[3]]
        m = max([max(rs), max([b[4] for b in bridges] or [1])] + [1]) * 1.5
        x0, x1 = min(xs) - m, max(xs) + m
        y0, y1 = min(ys) - m, max(ys) + m
    nx = int(round((x1 - x0) / mpp))
    ny = int(round((y1 - y0) / mpp))
    xs = (np.arange(nx) + 0.5) * mpp + x0
    ys = (np.arange(ny) + 0.5) * mpp + y0
    gx, gy = np.meshgrid(xs, ys)
    F = np.zeros_like(gx)
    for (cx, cy, r) in nodes:
        F = F + _ball_term(gx, gy, cx, cy, r)
    for (ax, ay, bx, by, rb) in bridges:
        if rb > 0:
            F = F + _capsule_term(gx, gy, ax, ay, bx, by, rb)
    return F, x0, y0, mpp


# ── contour: direct marching-squares walk ────────────────────────────

def marching_squares(f, thresh):
    """Closed contours where f crosses thresh -> list of (N,2) float arrays
    in pixel coords. Interior (f>=thresh) kept on the LEFT while walking
    (derived orientation table => consistent cycle, no T-junctions)."""
    h, w = f.shape
    if h < 2 or w < 2:
        return []

    tl = f[:-1, :-1]; tr = f[:-1, 1:]
    bl = f[1:, :-1]; br = f[1:, 1:]
    tl -= thresh; tr -= thresh; bl -= thresh; br -= thresh

    T = np.zeros((h - 1, w - 1, 2))
    R = np.zeros((h - 1, w - 1, 2))
    B = np.zeros((h - 1, w - 1, 2))
    L = np.zeros((h - 1, w - 1, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        tt = np.clip(np.where(np.abs(tr - tl) > 1e-12, tl / (tl - tr), 0.5), 0, 1)
        trr = np.clip(np.where(np.abs(br - tr) > 1e-12, tr / (tr - br), 0.5), 0, 1)
        bb = np.clip(np.where(np.abs(br - bl) > 1e-12, bl / (bl - br), 0.5), 0, 1)
        ll = np.clip(np.where(np.abs(bl - tl) > 1e-12, tl / (tl - bl), 0.5), 0, 1)
    T[:, :, 0] = np.arange(w - 1)[None, :] + tt; T[:, :, 1] = np.arange(h - 1)[:, None]
    R[:, :, 0] = np.arange(w - 1)[None, :] + 1.0; R[:, :, 1] = np.arange(h - 1)[:, None] + trr
    B[:, :, 0] = np.arange(w - 1)[None, :] + bb; B[:, :, 1] = np.arange(h - 1)[:, None] + 1.0
    L[:, :, 0] = np.arange(w - 1)[None, :]; L[:, :, 1] = np.arange(h - 1)[:, None] + ll

    case = (tl > 0) * 1 | (tr > 0) * 2 | (br > 0) * 4 | (bl > 0) * 8
    centre = (f[:-1, :-1] + f[:-1, 1:] + f[1:, :-1] + f[1:, 1:]) / 4 - thresh
    case = np.where((case == 5) & (centre > 0), 10, case)
    case = np.where((case == 10) & (centre <= 0), 5, case)

    # interior-on-LEFT orientation (derived, see git history / _orient.py)
    simple = {
        1: ("L", "T"), 14: ("T", "L"),
        2: ("T", "R"), 13: ("R", "T"),
        4: ("R", "B"), 11: ("B", "R"),
        8: ("B", "L"), 7: ("L", "B"),
        3: ("L", "R"), 12: ("R", "L"),
        6: ("T", "B"), 9: ("B", "T"),
    }
    E = {"T": T, "R": R, "B": B, "L": L}
    key = lambda p: (round(p[0], 5), round(p[1], 5))
    start_pts, end_pts = [], []

    for i in range(h - 1):
        for j in range(w - 1):
            c = int(case[i, j])
            if c == 0 or c == 15:
                continue
            P = {k: (E[k][i, j, 0], E[k][i, j, 1]) for k in "TRBL"}
            if c in simple:
                a, b = simple[c]
                start_pts.append(P[a]); end_pts.append(P[b])
            elif c == 5:
                start_pts += [P["T"], P["L"]]; end_pts += [P["R"], P["B"]]
            elif c == 10:
                start_pts += [P["T"], P["B"]]; end_pts += [P["L"], P["R"]]

    if not start_pts:
        return []
    nxt = {}
    for a, b in zip(start_pts, end_pts):
        ka, kb = key(a), key(b)
        if ka in nxt and nxt[ka] != kb:
            continue
        nxt[ka] = kb
    loops, visited = [], set()
    for s in list(nxt.keys()):
        if s in visited:
            continue
        cycle = []
        cur = s
        while cur is not None and cur not in visited:
            visited.add(cur)
            cycle.append(cur)
            cur = nxt.get(cur)
        if len(cycle) >= 8 and nxt.get(cycle[-1]) == cycle[0]:
            loops.append(np.array(cycle, dtype=float))
    return [lp for lp in loops if len(lp) >= 16]


# ── svg + png ────────────────────────────────────────────────────────

def emit_svg(loops, mpp, x0, y0, out_path, pad=2.0):
    if not loops:
        with open(out_path, "w") as f:
            f.write('<svg xmlns="http://www.w3.org/2000/svg" width="10mm" '
                    'height="10mm" viewBox="0 0 10 10">'
                    '<rect width="100%" height="100%" fill="white"/></svg>')
        return 10.0, 10.0
    allpts = np.vstack(loops)
    minx = allpts[:, 0].min() * mpp + x0 - pad
    maxx = allpts[:, 0].max() * mpp + x0 + pad
    miny = allpts[:, 1].min() * mpp + y0 - pad
    maxy = allpts[:, 1].max() * mpp + y0 + pad
    W, H = maxx - minx, maxy - miny
    paths = []
    for loop in loops:
        p = loop.copy()
        p[:, 0] = p[:, 0] * mpp + x0 - minx
        p[:, 1] = p[:, 1] * mpp + y0 - miny
        dstr = catmull_rom_path(p, 1.0)
        if dstr:
            paths.append(f'    <path d="{dstr}"/>')
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.3f}mm" '
           f'height="{H:.3f}mm" viewBox="0 0 {W:.3f} {H:.3f}">']
    svg.append('  <rect width="100%" height="100%" fill="white"/>')
    svg.append('  <g fill="black">')
    svg.extend(paths)
    svg.append('  </g>')
    svg.append('</svg>')
    with open(out_path, "w") as f:
        f.write("\n".join(svg))
    return W, H


def to_png(svg_path, png_path, width_px=1200):
    subprocess.run(["rsvg-convert", "-w", str(width_px),
                    svg_path, "-o", png_path], check=True)
    a = np.array(Image.open(png_path).convert("L"))
    return int((a < 128).sum())
