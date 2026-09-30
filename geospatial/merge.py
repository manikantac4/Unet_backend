"""
Merges the polygon fragments that come back from the tiled Earth
Engine scan into one coherent set of water bodies.

Working in a metric CRS (UTM), so buffer distances are true metres:

  grow by d  ->  union  ->  shrink by d

heals the seams where a lake was cut in two by the tile grid and
bridges 1-pixel gaps, while bodies further than 2*d apart stay
separate. (Same "closing" trick Mini-Backend and the U-Net pipeline
use, but each fragment is assigned to its merged body through a
spatial index instead of an all-pairs scan, so thousands of
fragments stay fast.)

That "closing" is the LEGACY merge (BOUNDARY_REFINEMENT=legacy). It heals seams,
but it also moves every concave shoreline by up to 20 m and welds separate
bodies that are < 40 m apart. The default REFINED merge below never moves a
shoreline: fragments are snapped to a mm grid and unioned exactly, and only
fragments from DIFFERENT tiles that a seam left slightly apart are bridged.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

from config import settings as cfg

logger = logging.getLogger("traditional_v2_backend.merge")

# Used ONLY when Earth Engine returned no statistics for a body. The
# body is then flagged `probability_stats_unavailable` so nobody reads
# these as measured values.
FALLBACK_STATS = {"mean": 0.70, "median": 0.70, "min": 0.50, "max": 0.90, "p50": 1.0, "p75": 0.50, "p90": 0.20}


def flatten_polygons(geom) -> List[Polygon]:
    """Polygon / MultiPolygon / GeometryCollection -> flat list of non-empty Polygons."""
    if geom is None or geom.is_empty:
        return []
    kind = geom.geom_type
    if kind == "Polygon":
        return [geom]
    if kind in ("MultiPolygon", "GeometryCollection"):
        out: List[Polygon] = []
        for part in geom.geoms:
            out.extend(flatten_polygons(part))
        return out
    return []


def _query_indices(tree: STRtree, geoms: list, probe) -> List[int]:
    """Version-agnostic STRtree query (shapely 2.x returns indices, 1.8 returns geometries)."""
    hits = tree.query(probe)
    if len(hits) == 0:
        return []
    first = hits[0]
    if isinstance(first, (int, np.integer)):
        return [int(i) for i in hits]
    lookup = {id(g): i for i, g in enumerate(geoms)}
    return [lookup[id(g)] for g in hits if id(g) in lookup]


def repair_polygons(geom) -> Tuple[List[Polygon], bool]:
    """Valid Polygons from any geometry + whether a repair was actually needed.

    shapely.make_valid keeps every lobe of a bow-tie / self-touching ring;
    the old `buffer(0)` silently drops some of them (i.e. loses water area).
    """
    if geom is None or geom.is_empty:
        return [], False
    if geom.is_valid:
        return flatten_polygons(geom), False
    return [p for p in flatten_polygons(shapely.make_valid(geom)) if p.area > 0], True


def _snap(geom):
    grid = cfg.SEAM_SNAP_GRID_M
    if grid <= 0:
        return geom
    try:
        snapped = shapely.set_precision(geom, grid)
    except Exception:  # noqa: BLE001
        return geom
    return snapped if snapped is not None and not snapped.is_empty else geom


def _bridge(g, h, d: float):
    """Polygon(s) filling the gap between two nearby, non-touching polygons.

    Everything is computed inside a small window around the gap (never on the full
    outline of a 100k-vertex lake): the window is the overlap of both envelopes
    grown by d, padded by 2d so the closing sees all the geometry it needs.
    The result is confined to points within 2d of BOTH polygons: every point of a
    gap of width <= 2d is within 2d of both sides, while a notch that only one
    polygon has (nothing across from it) is left alone.
    """
    ga, gb = g.bounds, h.bounds
    x0, y0 = max(ga[0], gb[0]) - d, max(ga[1], gb[1]) - d
    x1, y1 = min(ga[2], gb[2]) + d, min(ga[3], gb[3]) + d
    win = (x0 - 2 * d, y0 - 2 * d, x1 + 2 * d, y1 + 2 * d)
    gl, hl = shapely.clip_by_rect(g, *win), shapely.clip_by_rect(h, *win)
    if gl.is_empty or hl.is_empty:
        return []
    pair = unary_union([gl, hl])
    closed = pair.buffer(d).buffer(-d)                     # closing: contains both polygons + the gap
    # Deliberately NOT `closed - pair`: that leaves a hairline between the fill and the
    # polygons, unary_union then refuses to fuse them and the fill becomes an orphan.
    # `closed` overlaps both polygons, so the union below is robust.
    strip = gl.buffer(2 * d).intersection(hl.buffer(2 * d))
    return [_snap(p) for p in flatten_polygons(closed.intersection(strip))]


def _seam_fills(fragments: List[dict], geoms: list, d: float) -> list:
    """Thin bridges between fragments of DIFFERENT tiles that a tile seam left
    a hair apart (<= 2*d). Within-tile neighbours are never bridged."""
    if d <= 0 or len(geoms) < 2:
        return []
    tile_of = [f.get("tile_id") for f in fragments]
    tree = STRtree(geoms)
    fills: list = []
    for i, g in enumerate(geoms):
        minx, miny, maxx, maxy = g.bounds
        for j in _query_indices(tree, geoms, shapely.box(minx - 2 * d, miny - 2 * d, maxx + 2 * d, maxy + 2 * d)):
            if j <= i or tile_of[i] == tile_of[j]:
                continue
            h = geoms[j]
            try:
                if g.intersects(h) or g.distance(h) > 2 * d:
                    continue                   # touching / overlapping = already seamless; far = separate bodies
                fills.extend(_bridge(g, h, d))
            except Exception as exc:  # noqa: BLE001 - GEOS overlay can fail on degenerate input
                logger.warning(f"[merge] seam bridge skipped for one fragment pair ({exc!r})")
    return fills


def _merge_fragments_exact(fragments: List[dict]) -> List[dict]:
    geoms = [_snap(f["geometry"]) for f in fragments]
    fills = _seam_fills(fragments, geoms, cfg.SEAM_HEAL_DISTANCE_M)
    bodies = [p for p in flatten_polygons(unary_union(geoms + fills)) if p.area > 0]
    if not bodies:
        return []

    tree = STRtree(bodies)
    members: Dict[int, List[dict]] = {i: [] for i in range(len(bodies))}
    for fragment in fragments:
        probe = fragment["geometry"].representative_point()
        candidates = _query_indices(tree, bodies, probe)
        owner = next((i for i in candidates if bodies[i].covers(probe)), None)
        if owner is None and candidates:            # probe fell in a snap-rounding sliver: nearest body wins
            nearest = min(candidates, key=lambda i: bodies[i].distance(probe))
            owner = nearest if bodies[nearest].distance(probe) <= 0.1 else None
        if owner is not None:
            members[owner].append(fragment)
    return [
        {"geometry": body, "source_fragments": members[i]}
        for i, body in enumerate(bodies)
        if members[i]
    ]


def merge_fragments(fragments: List[dict], connect_distance_m: float = cfg.MERGE_CONNECT_DISTANCE_M) -> List[dict]:
    """Dispatch: refined (exact union) by default, legacy closing when
    BOUNDARY_REFINEMENT=legacy. Same input / output shape either way."""
    if not fragments:
        return []
    if cfg.BOUNDARY_REFINEMENT_ENABLED:
        try:
            return _merge_fragments_exact(fragments)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[merge] exact merge failed ({exc!r}); falling back to the legacy closing merge")
    return _merge_fragments_legacy(fragments, connect_distance_m)


def _merge_fragments_legacy(fragments: List[dict], connect_distance_m: float = cfg.MERGE_CONNECT_DISTANCE_M) -> List[dict]:
    """
    fragments: [{"geometry": Polygon (UTM metres), "area_m2": float, "tile_id": str, "stats": {...}}, ...]
    returns:   [{"geometry": Polygon (UTM), "source_fragments": [fragment, ...]}, ...]
    """
    if not fragments:
        return []

    grown = [f["geometry"].buffer(connect_distance_m) for f in fragments]
    closed = unary_union(grown).buffer(-connect_distance_m)
    bodies = [p for p in flatten_polygons(closed) if p.area > 0]
    if not bodies:
        return []

    tree = STRtree(bodies)
    members: Dict[int, List[dict]] = {i: [] for i in range(len(bodies))}
    for fragment in fragments:
        probe = fragment["geometry"].representative_point()
        for idx in _query_indices(tree, bodies, probe):
            if bodies[idx].covers(probe):
                members[idx].append(fragment)
                break

    return [
        {"geometry": body, "source_fragments": members[i]}
        for i, body in enumerate(bodies)
        if members[i]
    ]


def _valid(x: Optional[float]) -> bool:
    return x is not None and np.isfinite(x)


def combine_probability_stats(source_fragments: List[dict]) -> dict:
    """Probability statistics for one water body.

    One fragment  -> its own Earth Engine statistics, unchanged.
    Several       -> area-weighted mean / fractions, exact min / max,
                     area-weighted median (approximation, flagged by
                     the caller). Fragments with missing statistics
                     are ignored; if none have any, FALLBACK_STATS is
                     returned and `available` is False.
    """
    usable = [f for f in source_fragments if _valid(f["stats"].get("mean"))]
    if not usable:
        return {**FALLBACK_STATS, "available": False}

    if len(usable) == 1:
        s = usable[0]["stats"]
        return {
            "mean": s["mean"],
            "median": s["median"] if _valid(s.get("median")) else s["mean"],
            "min": s["min"] if _valid(s.get("min")) else s["mean"],
            "max": s["max"] if _valid(s.get("max")) else s["mean"],
            "p50": s["p50"] if _valid(s.get("p50")) else float(s["mean"] >= 0.5),
            "p75": s["p75"] if _valid(s.get("p75")) else float(s["mean"] >= 0.75),
            "p90": s["p90"] if _valid(s.get("p90")) else float(s["mean"] >= 0.90),
            "available": True,
        }

    total = sum(f["area_m2"] for f in usable) or 1.0

    def wavg(key: str) -> float:
        num = 0.0
        for f in usable:
            v = f["stats"].get(key)
            v = f["stats"]["mean"] if not _valid(v) else v
            num += v * f["area_m2"]
        return num / total

    mins = [f["stats"]["min"] for f in usable if _valid(f["stats"].get("min"))]
    maxs = [f["stats"]["max"] for f in usable if _valid(f["stats"].get("max"))]
    return {
        "mean": wavg("mean"),
        "median": wavg("median"),
        "min": min(mins) if mins else wavg("mean"),
        "max": max(maxs) if maxs else wavg("mean"),
        "p50": wavg("p50"),
        "p75": wavg("p75"),
        "p90": wavg("p90"),
        "available": True,
    }
