"""
Tiled, parallel water-polygon extraction from Earth Engine.

Same idea as Mini-Backend/services/tiled_water_detection.py (scan the
AOI in ~10 km cells so no single reduceToVectors() call gets big
enough to hit Earth Engine's limits), with four upgrades:

  * Cells are scanned IN PARALLEL (thread pool). v1 scanned them one
    after another, which is what made large radii slow.
  * Cells that lie entirely outside the circular AOI are skipped
    (a circle only fills ~78% of its bounding box).
  * Each cell is retried with back-off, and split into four sub-cells
    when Earth Engine reports it is too big (>5000 features, memory,
    timeout). A failed cell is reported, never silently dropped, and
    never aborts the rest of the scan.
  * Per-polygon confidence statistics are computed inside Earth Engine
    (reduceRegions) in the same call, so nothing has to be downloaded
    or uploaded again afterwards.
  * (refined mode) Every tile is vectorised over its rectangle EXPANDED by
    TILE_HALO_M, so neighbouring tiles overlap. A tile edge is therefore never
    the edge of a water polygon: the neighbour covers that strip completely and
    the exact union in geospatial/merge.py removes the cut. Vectorisation runs
    at PROCESS_SCALE_M (5 m) instead of the native 10 m.
"""

import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Optional, Tuple

import ee

from config import settings as cfg

logger = logging.getLogger("traditional_v2_backend.tiles")

Bounds = Tuple[float, float, float, float]  # west, south, east, north

# Substrings of Earth Engine error messages that mean "this request is
# too big — make it smaller", as opposed to a transient network error.
_TOO_BIG_HINTS = (
    "5000", "too many elements", "too many pixels", "memory", "timed out", "time out",
    "too large", "response size", "payload size", "output too large",
)


# ============================================================
# GRID
# ============================================================

def _km_to_deg_lat(km: float) -> float:
    return km / 111.32


def _km_to_deg_lon(km: float, lat_deg: float) -> float:
    return km / (111.32 * max(0.1, math.cos(math.radians(lat_deg))))


def generate_tiles(west: float, south: float, east: float, north: float, tile_km: float) -> List[Bounds]:
    """Splits the bbox into ~tile_km cells that exactly cover it (no gaps, no overlap)."""
    mid_lat = (south + north) / 2.0
    lat_step = _km_to_deg_lat(tile_km)
    lon_step = _km_to_deg_lon(tile_km, mid_lat)

    tiles: List[Bounds] = []
    lat = south
    while lat < north - 1e-9:
        lat_end = min(lat + lat_step, north)
        lon = west
        while lon < east - 1e-9:
            lon_end = min(lon + lon_step, east)
            tiles.append((lon, lat, lon_end, lat_end))
            lon = lon_end
        lat = lat_end
    return tiles or [(west, south, east, north)]


def _tile_touches_circle(tile: Bounds, lat: float, lon: float, radius_km: float) -> bool:
    """True if the tile rectangle intersects the circle (equirectangular approximation)."""
    w, s, e, n = tile
    nearest_lon = min(max(lon, w), e)
    nearest_lat = min(max(lat, s), n)
    dx = (nearest_lon - lon) * 111.32 * math.cos(math.radians(lat))
    dy = (nearest_lat - lat) * 111.32
    return math.hypot(dx, dy) <= radius_km + 0.05


def plan_tiles(latitude: float, longitude: float, radius_km: float, tile_km: float = cfg.TILE_KM) -> List[Bounds]:
    dlat = _km_to_deg_lat(radius_km)
    dlon = _km_to_deg_lon(radius_km, latitude)
    tiles = generate_tiles(longitude - dlon, latitude - dlat, longitude + dlon, latitude + dlat, tile_km)
    return [t for t in tiles if _tile_touches_circle(t, latitude, longitude, radius_km)]


def _split_four(tile: Bounds) -> List[Bounds]:
    w, s, e, n = tile
    mx, my = (w + e) / 2.0, (s + n) / 2.0
    return [(w, s, mx, my), (mx, s, e, my), (w, my, mx, n), (mx, my, e, n)]


def _with_halo(tile: Bounds) -> Bounds:
    """Tile rectangle grown by TILE_HALO_M on every side (identity when 0)."""
    if cfg.TILE_HALO_M <= 0:
        return tile
    w, s, e, n = tile
    dlat = _km_to_deg_lat(cfg.TILE_HALO_M / 1000.0)
    dlon = _km_to_deg_lon(cfg.TILE_HALO_M / 1000.0, (s + n) / 2.0)
    return (w - dlon, s - dlat, e + dlon, n + dlat)


# ============================================================
# ONE TILE (Earth Engine)
# ============================================================

def _num(value) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _fetch_once(mask: ee.Image, stats_image: ee.Image, prob: ee.Image, tile: Bounds, tile_id: str) -> List[dict]:
    geom = ee.Geometry.Rectangle(list(_with_halo(tile)), None, False)

    vectors = mask.reduceToVectors(
        geometry=geom,
        scale=cfg.PROCESS_SCALE_M,
        geometryType="polygon",
        labelProperty="water",
        maxPixels=int(1e9),
        tileScale=4,
    )
    # bands [prob, p50, p75, p90] -> properties named after the bands
    vectors = stats_image.reduceRegions(
        collection=vectors, reducer=ee.Reducer.mean(), scale=cfg.PIXEL_SIZE_M, tileScale=4,
    )
    # properties: median, min, max
    vectors = prob.reduceRegions(
        collection=vectors,
        reducer=ee.Reducer.median().combine(reducer2=ee.Reducer.minMax(), outputPrefix="", sharedInputs=True),
        scale=cfg.PIXEL_SIZE_M,
        tileScale=4,
    )
    raw = vectors.getInfo() or {}

    fragments = []
    for feature in raw.get("features", []):
        geometry = feature.get("geometry")
        if not geometry or not geometry.get("coordinates"):
            continue
        p = feature.get("properties") or {}
        mean = _num(p.get("prob", p.get("mean")))
        fragments.append({
            "geometry": geometry,
            "tile_id": tile_id,
            "stats": {
                "mean": mean,
                "median": _num(p.get("median")),
                "min": _num(p.get("min")),
                "max": _num(p.get("max")),
                "p50": _num(p.get("p50")),
                "p75": _num(p.get("p75")),
                "p90": _num(p.get("p90")),
            },
        })
    return fragments


def _fetch_tile(mask, stats_image, prob, tile: Bounds, tile_id: str, depth: int = 0) -> List[dict]:
    last_exc: Optional[Exception] = None
    for attempt in range(1, cfg.TILE_MAX_ATTEMPTS + 1):
        try:
            return _fetch_once(mask, stats_image, prob, tile, tile_id)
        except Exception as exc:  # noqa: BLE001 — EE raises many exception types
            last_exc = exc
            text = str(exc).lower()
            if depth < cfg.TILE_MAX_SPLIT_DEPTH and any(h in text for h in _TOO_BIG_HINTS):
                logger.info(f"[tiles] {tile_id} too big for one request ({exc}); splitting into 4")
                out: List[dict] = []
                for i, child in enumerate(_split_four(tile)):
                    out.extend(_fetch_tile(mask, stats_image, prob, child, f"{tile_id}.{i + 1}", depth + 1))
                return out
            if attempt < cfg.TILE_MAX_ATTEMPTS:
                time.sleep(2 ** attempt)
    raise RuntimeError(str(last_exc))


# ============================================================
# WHOLE AOI
# ============================================================

def scan_aoi(
    mask: ee.Image,
    stats_image: ee.Image,
    prob: ee.Image,
    latitude: float,
    longitude: float,
    radius_km: float,
    log: Optional[Callable[[str], None]] = None,
) -> dict:
    """Scans every tile of the AOI in parallel.

    Returns {"fragments": [...], "tile_count": int, "tile_errors": [...]}.
    """
    log = log or (lambda m: logger.info(m))
    tiles = plan_tiles(latitude, longitude, radius_km)
    ids = [f"T{i + 1:03d}" for i in range(len(tiles))]
    total = len(tiles)
    log(f"scanning {total} tile(s) of ~{cfg.TILE_KM:g} km with {cfg.TILE_WORKERS} parallel workers")

    fragments: List[dict] = []
    errors: List[dict] = []
    done = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=max(1, cfg.TILE_WORKERS)) as pool:
        futures = {
            pool.submit(_fetch_tile, mask, stats_image, prob, tile, tid): tid
            for tile, tid in zip(tiles, ids)
        }
        for future in as_completed(futures):
            tid = futures[future]
            done += 1
            try:
                fragments.extend(future.result())
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[tiles] {tid} failed after retries: {exc}")
                errors.append({"patch_id": tid, "stage": "tile_vectorize", "error": str(exc)})

            if done % cfg.PROGRESS_LOG_EVERY_N_TILES == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0.0
                eta = (total - done) / rate / 60.0 if rate > 0 else float("nan")
                log(f"progress {done}/{total} tiles ({rate:.2f}/s, ETA {eta:.1f} min, "
                    f"{len(fragments)} raw fragments, {len(errors)} tile errors)")

    return {"fragments": fragments, "tile_count": total, "tile_errors": errors}
