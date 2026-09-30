"""
Boundary drawing for Traditional Model v2.

Two modes, chosen by config BOUNDARY_REFINEMENT ("refined" default | "legacy"):

LEGACY  (unchanged from the previous version)
  vector-side open (erode -> dilate), small holes filled, then Douglas-Peucker
  at SIMPLIFY_TOLERANCE_M (8 m by default). On a 10 m raster an 8 m tolerance
  moves the shoreline by ~0.8 pixel - more than the quantisation error it was
  meant to hide - which is why real shorelines came out as ~45 m straight chords.

REFINED
  The polygon that arrives is a pixel-edge staircase. The goal is the smooth line
  the staircase is *sampling*, without ever leaving the pixel's own uncertainty:

    1. small holes filled (same rule as legacy: noise, real islands are kept)
    2. every edge is densified to <= REFINE_STEP_PX
    3. REFINE_SMOOTH_PASSES passes of [1 2 1]/4 smoothing; after EVERY pass each
       vertex is pulled back to within REFINE_MAX_SHIFT_PX of its raster position
       (vertices start ON the raster boundary, so the result stays within that
       distance of it - a hard bound, no buffer/overlay needed)
    4. topology-preserving Douglas-Peucker at REFINE_SIMPLIFY_PX (a fraction of a
       pixel, not 8 m)
    5. safety net: invalid result -> make_valid; implied mean boundary shift
       (|area change| / perimeter) above REFINE_MAX_MEAN_SHIFT_PX, an empty
       result, a split, or a hole/island count change -> the un-smoothed
       raster polygon is kept

  Because MAX_SHIFT + SIMPLIFY < 0.5 px, two bodies that are at least one pixel
  apart in the raster cannot touch or overlap after refinement.

Works on Polygon or MultiPolygon (UTM metres). refine_geometry() returns a list of
valid Polygons; refine_geometry_report() also returns per-polygon diagnostics.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import shapely
from shapely.geometry import Polygon

from config import settings as cfg
from geospatial.merge import flatten_polygons

logger = logging.getLogger("traditional_v2_backend.boundary_refine")

METHOD_LEGACY = "legacy_simplify_dp"
METHOD_REFINED = "pixel_smooth_clamped"
METHOD_FALLBACK = "raster_polygon_unsmoothed"


# ============================================================
# LEGACY (verbatim)
# ============================================================

def fill_small_holes(poly: Polygon, max_hole_area_m2: float = cfg.FILL_HOLES_MAX_AREA_M2) -> Polygon:
    if not poly.interiors or max_hole_area_m2 <= 0:
        return poly
    kept = [r for r in poly.interiors if Polygon(r).area >= max_hole_area_m2]
    return poly if len(kept) == len(poly.interiors) else Polygon(poly.exterior, kept)


def despeckle(geom, radius_m: float = cfg.DESPECKLE_RADIUS_M):
    """Vector-side open (erode -> dilate): pinches off spurs/whiskers
    narrower than ~2*radius_m without shrinking or reshaping anything
    wider than that. Falls back to the original geometry whenever the
    erosion empties it (a body thinner than the radius everywhere,
    e.g. a genuinely narrow canal) so this can only remove noise, never
    a real — if slender — water body.
    """
    if radius_m <= 0 or geom is None or geom.is_empty:
        return geom
    try:
        opened = geom.buffer(-radius_m).buffer(radius_m)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[boundary_refine] despeckle failed ({exc!r}); keeping original geometry")
        return geom
    return opened if opened is not None and not opened.is_empty else geom


def refine_polygon(poly: Polygon) -> List[Polygon]:
    try:
        poly = fill_small_holes(poly)
        pieces = _repair(poly.simplify(cfg.SIMPLIFY_TOLERANCE_M, preserve_topology=True)) or [poly]
        return [fill_small_holes(p) for p in pieces if p.is_valid and p.area > 0] or [poly]
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[boundary_refine] failed ({exc!r}); keeping original polygon")
        return [poly]


def _repair(geom) -> List[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if not geom.is_valid:
        geom = geom.buffer(0)
    return [p for p in flatten_polygons(geom) if p.area > 0]


def _refine_geometry_legacy(geom) -> List[Polygon]:
    geom = despeckle(geom)
    out: List[Polygon] = []
    for poly in flatten_polygons(geom):
        out.extend(refine_polygon(poly))
    return out


# ============================================================
# REFINED
# ============================================================

def _densify(ring_xy: np.ndarray, step: float) -> np.ndarray:
    """Open ring (n,2, no repeated last point) -> same path with extra points so no
    gap exceeds `step`. Original vertices are all kept (corners are never cut here)."""
    nxt = np.roll(ring_xy, -1, axis=0)
    seg = nxt - ring_xy
    length = np.hypot(seg[:, 0], seg[:, 1])
    parts = np.maximum(1, np.ceil(length / step).astype(int))
    idx = np.repeat(np.arange(len(ring_xy)), parts)
    k = np.arange(parts.sum()) - np.repeat(np.cumsum(parts) - parts, parts)
    t = (k / np.repeat(parts, parts))[:, None]
    return ring_xy[idx] + seg[idx] * t


def _smooth_ring(ring, step: float, max_shift: float, passes: int) -> Optional[np.ndarray]:
    xy = np.asarray(ring.coords)[:-1]
    if len(xy) < 3:
        return None
    anchor = _densify(xy, step)          # points lying ON the raster boundary
    pts = anchor.copy()
    for _ in range(passes):
        pts = 0.25 * np.roll(pts, 1, axis=0) + 0.5 * pts + 0.25 * np.roll(pts, -1, axis=0)
        delta = pts - anchor
        dist = np.hypot(delta[:, 0], delta[:, 1])
        over = dist > max_shift
        if over.any():                   # hard clamp: never leave the pixel's own uncertainty
            pts[over] = anchor[over] + delta[over] * (max_shift / dist[over])[:, None]
    return pts


def _smooth_polygon(base: Polygon, px: float) -> Optional[Polygon]:
    step = cfg.REFINE_STEP_PX * px
    shift = cfg.REFINE_MAX_SHIFT_PX * px
    shell = _smooth_ring(base.exterior, step, shift, cfg.REFINE_SMOOTH_PASSES)
    if shell is None:
        return None
    holes = []
    for interior in base.interiors:
        h = _smooth_ring(interior, step, shift, cfg.REFINE_SMOOTH_PASSES)
        if h is not None:
            holes.append(h)
    smooth = Polygon(shell, holes)
    if cfg.REFINE_SIMPLIFY_PX > 0:
        smooth = smooth.simplify(cfg.REFINE_SIMPLIFY_PX * px, preserve_topology=True)
    return smooth


def _info(method: str, original: Polygon, refined: Polygon, repaired: bool, fallback: Optional[str] = None) -> Dict:
    return {
        "method": method,
        "original_area": float(original.area),
        "original_perimeter": float(original.length),
        "refined_area": float(refined.area),
        "refined_perimeter": float(refined.length),
        "repaired": bool(repaired),
        "fallback": fallback,
    }


def _refine_polygon_v2(poly: Polygon, px: float) -> List[Tuple[Polygon, Dict]]:
    original = poly
    repaired = False
    if not poly.is_valid:
        fixed = shapely.make_valid(poly)
        parts = [p for p in flatten_polygons(fixed) if p.area > 0]
        repaired = True
        if len(parts) != 1:
            return [(p, _info(METHOD_FALLBACK, p, p, True, "invalid_input_split")) for p in parts]
        poly = original = parts[0]

    base = fill_small_holes(poly)          # noise removal, same rule as legacy; NOT counted as smoothing
    try:
        smooth = _smooth_polygon(base, px)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[boundary_refine] smoothing failed ({exc!r}); keeping raster polygon")
        smooth = None
    if smooth is None or smooth.is_empty:
        return [(base, _info(METHOD_FALLBACK, original, base, repaired, "smoothing_failed"))]

    if not smooth.is_valid:
        pieces = [p for p in flatten_polygons(shapely.make_valid(smooth)) if p.area > 0]
        repaired = True
    else:
        pieces = [smooth]

    total = sum(p.area for p in pieces)
    mean_shift_px = abs(total - base.area) / base.length / px if base.length > 0 else 1.0
    holes_before = len(base.interiors)
    holes_after = sum(len(p.interiors) for p in pieces)
    if not pieces:
        reason = "empty_result"
    elif mean_shift_px > cfg.REFINE_MAX_MEAN_SHIFT_PX:
        reason = f"mean_shift_{mean_shift_px:.2f}px"
    elif len(pieces) > 1 or holes_after != holes_before:
        reason = "topology_changed"
    else:
        reason = None
    if reason:
        return [(base, _info(METHOD_FALLBACK, original, base, repaired, reason))]
    return [(p, _info(METHOD_REFINED, original, p, repaired)) for p in pieces]


def refine_geometry_report(geom, pixel_size_m: Optional[float] = None) -> List[Tuple[Polygon, Dict]]:
    """[(Polygon, diagnostics)] for a merged water body (Polygon or MultiPolygon, UTM m).

    pixel_size_m: size of the raster pixel the geometry was vectorised from
    (the smallest ground dimension); defaults to cfg.PROCESS_SCALE_M.
    """
    if not cfg.BOUNDARY_REFINEMENT_ENABLED:
        out = []
        geom_d = despeckle(geom)
        for poly in flatten_polygons(geom_d):
            poly_in = fill_small_holes(poly)
            for p in refine_polygon(poly):
                out.append((p, _info(METHOD_LEGACY, poly_in, p, False)))
        return out

    px = float(pixel_size_m or cfg.PROCESS_SCALE_M)
    geom = despeckle(geom, cfg.REFINE_DESPECKLE_RADIUS_M)
    out: List[Tuple[Polygon, Dict]] = []
    for poly in flatten_polygons(geom):
        try:
            out.extend(_refine_polygon_v2(poly, px))
        except Exception as exc:  # noqa: BLE001 - one bad polygon must never fail the analysis
            logger.warning(f"[boundary_refine] polygon skipped smoothing ({exc!r}); keeping raster polygon")
            out.append((poly, _info(METHOD_FALLBACK, poly, poly, False, "exception")))
    return out


def refine_geometry(geom, pixel_size_m: Optional[float] = None) -> List[Polygon]:
    """Same contract as before: list of valid Polygons."""
    if not cfg.BOUNDARY_REFINEMENT_ENABLED:
        return _refine_geometry_legacy(geom)
    return [p for p, _ in refine_geometry_report(geom, pixel_size_m)]
