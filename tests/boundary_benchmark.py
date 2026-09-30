"""
Offline boundary-quality benchmark for Traditional V2 (no Earth Engine needed).

It builds a synthetic scene whose TRUE water polygons are known exactly, turns it
into what Earth Engine would hand the backend (pixel-edge polygon fragments cut by
a tile grid), runs the real merge_fragments + refine code on it, and measures the
error against the truth.

Scene (metres, 2000 x 2000 domain, tile seams at x=1000 and y=1000):
  * lake with a wavy shoreline and a 35 m radius island, straddling the x seam
  * river (40 m wide) that runs along the y seam and crosses the x seam
  * two ponds separated by a 25 m dam (must stay TWO bodies)
  * four small ponds (2,000-2,800 m2) at arbitrary sub-pixel positions

Scenarios
  legacy_10m     : original v2 behaviour (10 m raster, no halo, +/-20 m closing, 8 m DP)
  refined_10m    : 10 m raster + halo tiles + exact merge + constrained smoothing
  refined_5m_sub : 5 m sub-pixel raster (bilinear-interpolated score) + same vector stage

Run:  python tests/boundary_benchmark.py
"""
import math
import os
import sys
from contextlib import contextmanager

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from config import settings as cfg
from geospatial import boundary_refine, merge

DOMAIN_M = 2000.0
TILE_M = 1000.0


# ---------------------------------------------------------------- truth ----
def build_truth():
    th = np.linspace(0, 2 * math.pi, 720, endpoint=False)
    r = 280 * (1 + 0.15 * np.sin(3 * th) + 0.08 * np.sin(5 * th + 1.0))
    lake = Polygon(np.c_[1000 + r * np.cos(th), 600 + r * np.sin(th)])
    island = Point(1010, 620).buffer(35, 32)          # 3.8 ha hole, must be kept
    lake = lake.difference(island)

    xs = np.linspace(0, DOMAIN_M, 400)
    river = LineString(np.c_[xs, 1000 + 120 * np.sin(xs / 150.0)]).buffer(20, cap_style="flat")

    pond_a = Point(1500, 1600).buffer(70, 64)
    pond_b = Point(1500 + 140 + 25, 1600).buffer(70, 64)   # 25 m dam
    truth = {"lake": lake, "river": river, "pond_a": pond_a, "pond_b": pond_b}
    # small ponds (2,000-2,800 m2 = 20-28 native pixels) at arbitrary sub-pixel offsets
    for k, (x, y, r) in enumerate([(253.3, 1651.7, 25), (702.9, 251.1, 28), (1747.4, 303.6, 30), (307.7, 1403.2, 26)]):
        truth[f"small_{k}"] = Point(x, y).buffer(r, 32)
    return truth


def coverage_grid(truth_union, pix, ss=6):
    """Fraction of each pix x pix cell covered by truth (mixed-pixel model)."""
    n = int(DOMAIN_M / pix)
    off = (np.arange(n * ss) + 0.5) * pix / ss
    X, Y = np.meshgrid(off, off)
    inside = shapely.contains_xy(truth_union, X.ravel(), Y.ravel()).reshape(n * ss, n * ss)
    return inside.reshape(n, ss, n, ss).mean(axis=(1, 3))     # [row=y, col=x]


def bilinear_up(a, f):
    ny, nx = a.shape
    ys = (np.arange(ny * f) + 0.5) / f - 0.5
    xs = (np.arange(nx * f) + 0.5) / f - 0.5
    y0 = np.clip(np.floor(ys).astype(int), 0, ny - 2)
    x0 = np.clip(np.floor(xs).astype(int), 0, nx - 2)
    wy = np.clip(ys - y0, 0, 1)[:, None]
    wx = np.clip(xs - x0, 0, 1)[None, :]
    return ((1 - wy) * (1 - wx) * a[np.ix_(y0, x0)] + (1 - wy) * wx * a[np.ix_(y0, x0 + 1)]
            + wy * (1 - wx) * a[np.ix_(y0 + 1, x0)] + wy * wx * a[np.ix_(y0 + 1, x0 + 1)])


# ------------------------------------------------- "Earth Engine" output ----
def mask_to_fragments(mask, pix, window, tile_id):
    """Pixel-edge polygons (what reduceToVectors returns) for the pixels whose
    centre lies inside `window` = (x0, y0, x1, y1). One fragment per component."""
    x0, y0, x1, y1 = window
    n = mask.shape[0]
    cx = (np.arange(n) + 0.5) * pix
    colsel = (cx >= x0) & (cx < x1)
    boxes = []
    for j in np.where((cx >= y0) & (cx < y1))[0]:
        row = mask[j] & colsel
        if not row.any():
            continue
        d = np.diff(np.r_[0, row.astype(int), 0])
        for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
            boxes.append(shapely.box(s * pix, j * pix, e * pix, (j + 1) * pix))
    if not boxes:
        return []
    u = unary_union(boxes)
    parts = list(u.geoms) if u.geom_type == "MultiPolygon" else [u]
    return [{"geometry": p, "area_m2": p.area, "tile_id": tile_id,
             "stats": {"mean": 0.9, "median": 0.9, "min": 0.6, "max": 0.99, "p50": 1.0, "p75": 0.95, "p90": 0.8}}
            for p in parts if p.area > 0]


def scan(mask, pix, halo_m):
    frags = []
    for ti, (tx, ty) in enumerate([(0, 0), (TILE_M, 0), (0, TILE_M), (TILE_M, TILE_M)]):
        win = (tx - halo_m, ty - halo_m, tx + TILE_M + halo_m, ty + TILE_M + halo_m)
        frags += mask_to_fragments(mask, pix, win, f"T{ti + 1:03d}")
    return frags


# -------------------------------------------------------------- metrics ----
def _boundary_points(geom, step=1.0):
    pts = []
    for p in (geom.geoms if hasattr(geom, "geoms") else [geom]):
        for ring in [p.exterior, *p.interiors]:
            n = max(2, int(ring.length / step))
            pts.append(shapely.line_interpolate_point(ring, np.linspace(0, ring.length, n, endpoint=False)))
    return np.concatenate(pts)


def _dist_to(boundary_geom, pts):
    return shapely.distance(pts, boundary_geom.boundary)


def metrics(bodies, truth):
    tu = unary_union(list(truth.values()))
    out = unary_union(bodies)
    inter = tu.intersection(out).area
    m = {
        "n_bodies": len(bodies), "n_truth": len(truth),
        "iou": inter / tu.union(out).area,
        "area_err_pct": 100 * (out.area - tu.area) / tu.area,
        "valid": all(b.is_valid for b in bodies),
        "vertices": sum(len(r.coords) for b in bodies for r in [b.exterior, *b.interiors]),
    }
    d_out = _dist_to(tu, _boundary_points(out))          # output boundary -> truth boundary
    d_tru = _dist_to(out, _boundary_points(tu))          # truth boundary -> output boundary
    d = np.r_[d_out, d_tru]
    m.update(mean_err_m=float(d.mean()), rms_err_m=float(np.sqrt((d ** 2).mean())), p95_err_m=float(np.percentile(d, 95)))
    # seam quality: boundary error within 15 m of a tile seam line
    P = _boundary_points(out)
    xy = shapely.get_coordinates(P)
    near = (np.abs(xy[:, 0] - TILE_M) < 15) | (np.abs(xy[:, 1] - TILE_M) < 15)
    m["seam_rms_err_m"] = float(np.sqrt((_dist_to(tu, P[near]) ** 2).mean())) if near.any() else float("nan")
    # topology
    dam_kept = not (unary_union([b for b in bodies if b.distance(truth["pond_a"]) < 1]).intersects(truth["pond_b"].buffer(-5)))
    m["dam_kept"] = bool(dam_kept)
    island_c = Point(1010, 620)
    m["island_kept"] = any(b.contains(island_c) is False and b.distance(island_c) < 40 for b in bodies)
    m["overlaps"] = sum(1 for i in range(len(bodies)) for j in range(i + 1, len(bodies)) if bodies[i].intersects(bodies[j]))
    return m


# ------------------------------------------------------------ pipelines ----
@contextmanager
def config(**kw):
    old = {k: getattr(cfg, k) for k in kw}
    for k, v in kw.items():
        setattr(cfg, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(cfg, k, v)


def run_pipeline(frags, pixel_m):
    groups = merge.merge_fragments(frags)
    out = []
    for g in groups:
        out += [p for p, _ in boundary_refine.refine_geometry_report(g["geometry"], pixel_size_m=pixel_m)]
    return [p for p in out if p.area >= cfg.MIN_WATER_BODY_AREA_M2]


def run_all(noise=0.0, seed=1):
    truth = build_truth()
    tu = unary_union(list(truth.values()))
    cov10 = coverage_grid(tu, 10.0)
    if noise:
        cov10 = np.clip(cov10 + np.random.default_rng(seed).normal(0, noise, cov10.shape), 0, 1)
    m10 = cov10 >= 0.5
    m5 = bilinear_up(cov10, 2) >= 0.5
    results = {}
    with config(BOUNDARY_REFINEMENT_ENABLED=False):
        results["legacy_10m"] = metrics(run_pipeline(scan(m10, 10.0, 0.0), 10.0), truth)
    with config(BOUNDARY_REFINEMENT_ENABLED=True):
        results["refined_10m"] = metrics(run_pipeline(scan(m10, 10.0, 40.0), 10.0), truth)
        results["refined_5m_sub"] = metrics(run_pipeline(scan(m5, 5.0, 40.0), 5.0), truth)
    return results


if __name__ == "__main__":
    for noise in (0.0, 0.05):
        print(f"\n=== synthetic scene, sensor noise sigma={noise} ===")
        res = run_all(noise)
        keys = ["n_bodies", "iou", "area_err_pct", "mean_err_m", "rms_err_m", "p95_err_m",
                "seam_rms_err_m", "dam_kept", "island_kept", "overlaps", "valid", "vertices"]
        print(f"{'':16s}" + "".join(f"{k:>15s}" for k in keys))
        for name, r in res.items():
            print(f"{name:16s}" + "".join(f"{(r[k] if isinstance(r[k], (bool, int)) else round(r[k], 3))!s:>15s}" for k in keys))
