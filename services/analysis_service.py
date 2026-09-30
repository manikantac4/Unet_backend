"""
End-to-end Traditional Model v2 pipeline.

Same process shape as Mini-Backend ("Traditional v1"): classification
runs inside Earth Engine, only vectors come back. Same INPUT and
OUTPUT contract as Unet_backend's run_analysis():

    input : latitude, longitude, radius_km, start_date, end_date, max_cloud_percentage
    output: dict with analysis_id, aoi, scene_metadata, analysis_crs, water_bodies,
            polygons_wgs84, patch_count, patch_errors, timings, model_threshold

Field mapping onto the U-Net vocabulary (so any client works with both):
    patch_count / patch_errors  -> vector tiles scanned / tiles that failed
    model_threshold             -> 0.5 on the confidence surface
    mean_probability etc.       -> index-ensemble confidence (see gee_processor.py)
"""

import logging
import math
import time
import traceback
import uuid

from shapely.geometry import Point, shape
from shapely.ops import transform as shp_transform

import gee_processor
from config import settings as cfg
from geospatial.boundary_refine import refine_geometry_report
from geospatial.measurements import get_transformer, measure_polygon, utm_epsg_for_lonlat
from geospatial.merge import combine_probability_stats, flatten_polygons, merge_fragments, repair_polygons
from ranking.ranking_engine import rank_water_bodies
from services.tiled_water_detection import scan_aoi
from water_bodies.schema import WaterBody, compute_quality_flags

logger = logging.getLogger("traditional_v2_backend.analysis")
ENGINE_NAME = cfg.ENGINE_NAME

EDGE_MARGIN_M = 15.0


def _say(analysis_id: str, message: str) -> None:
    line = f"[{ENGINE_NAME} {analysis_id}] {message}"
    print(line, flush=True)
    logger.info(line)


def _fail(analysis_id: str, stage: str, exc: Exception) -> None:
    print(f"[{ENGINE_NAME} {analysis_id}] STAGE '{stage}' FAILED: {exc}", flush=True)
    print(traceback.format_exc(), flush=True)
    logger.error(f"[{ENGINE_NAME} {analysis_id}] STAGE '{stage}' FAILED: {exc}")


def _to_metric_fragments(raw_fragments: list, to_utm) -> list:
    """GeoJSON (lon/lat) fragments -> shapely Polygons in UTM metres."""
    out = []
    for f in raw_fragments:
        try:
            geom = shp_transform(to_utm.transform, shape(f["geometry"]))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[{ENGINE_NAME}] bad fragment geometry skipped: {exc}")
            continue
        repaired = False
        if cfg.BOUNDARY_REFINEMENT_ENABLED:
            polys, repaired = repair_polygons(geom)      # make_valid keeps every lobe; buffer(0) can drop some
        else:
            if not geom.is_valid:
                geom = geom.buffer(0)
            polys = flatten_polygons(geom)
        for poly in polys:
            out.append({
                "geometry": poly, "area_m2": poly.area,
                "tile_id": f["tile_id"], "stats": f["stats"], "repaired": repaired,
            })
    return out


def run_analysis_v2(
    latitude: float,
    longitude: float,
    radius_km: float,
    start_date: str = "2024-01-01",
    end_date: str = "2024-06-30",
    max_cloud_percentage: float = 20.0,
) -> dict:
    analysis_id = "tv2-" + str(uuid.uuid4())[:8]
    t_start = time.time()
    timings: dict = {}

    print("=" * 70, flush=True)
    _say(analysis_id, f"REQUEST — lat={latitude} lon={longitude} radius_km={radius_km} "
                      f"{start_date}..{end_date} max_cloud={max_cloud_percentage}")
    print("=" * 70, flush=True)

    analysis_dir = cfg.STORAGE_DIR / analysis_id
    for sub in ("polygons", "statistics", "metadata"):
        (analysis_dir / sub).mkdir(parents=True, exist_ok=True)

    # ---- 1. Earth Engine + composite -----------------------------------
    t0 = time.time()
    _say(analysis_id, "STAGE 1/5: Earth Engine init + cloud-masked composite — starting...")
    try:
        gee_processor.initialize_earth_engine()
        aoi = gee_processor.create_aoi(latitude, longitude, radius_km)
        comp = gee_processor.build_composite(aoi, start_date, end_date, max_cloud_percentage)
    except Exception as exc:
        _fail(analysis_id, "composite", exc)
        raise
    timings["composite_s"] = time.time() - t0
    _say(analysis_id, f"STAGE 1/5: SUCCESS — composite of {comp['image_count']} scene(s) "
                      f"[{comp['window']['note']}] in {timings['composite_s']:.1f}s")

    # ---- 2. Classification graph (lazy) --------------------------------
    try:
        layers = gee_processor.classify(comp["composite"], aoi)
    except Exception as exc:
        _fail(analysis_id, "classify", exc)
        raise
    _say(analysis_id, "STAGE 2/5: SUCCESS — fused-score + hysteresis classifier built (evaluated server-side)")

    # ---- 3. Tiled vectorisation ----------------------------------------
    t0 = time.time()
    _say(analysis_id, "STAGE 3/5: TILED VECTORISATION — starting...")
    try:
        scan = scan_aoi(
            layers["mask"], layers["stats_image"], layers["prob"],
            latitude, longitude, radius_km, log=lambda m: _say(analysis_id, f"STAGE 3/5: {m}"),
        )
    except Exception as exc:
        _fail(analysis_id, "tile_scan", exc)
        raise
    timings["vectorize_s"] = time.time() - t0
    tile_count, tile_errors = scan["tile_count"], scan["tile_errors"]
    _say(analysis_id, f"STAGE 3/5: SUCCESS — {len(scan['fragments'])} raw fragments from "
                      f"{tile_count - len(tile_errors)}/{tile_count} tiles in {timings['vectorize_s']:.1f}s")

    if tile_count > 0 and len(tile_errors) == tile_count:
        raise RuntimeError(
            "Every Earth Engine tile request failed — check Earth Engine authentication / quota. "
            f"First error: {tile_errors[0]['error']}"
        )

    # ---- 4. Merge + refine + measure -----------------------------------
    t0 = time.time()
    _say(analysis_id, "STAGE 4/5: MERGE + REFINE + MEASURE — starting...")
    try:
        utm_epsg = utm_epsg_for_lonlat(longitude, latitude)
        to_utm = get_transformer("EPSG:4326", utm_epsg)
        to_wgs84 = get_transformer(utm_epsg, "EPSG:4326")

        fragments = _to_metric_fragments(scan["fragments"], to_utm)
        groups = merge_fragments(fragments)

        cx, cy = to_utm.transform(longitude, latitude)
        aoi_utm = Point(cx, cy).buffer(radius_km * 1000.0, 128)
        aoi_inner = aoi_utm.buffer(-EDGE_MARGIN_M)

        # Ground size of one raster pixel as vectorised by Earth Engine: the default
        # lon/lat grid at PROCESS_SCALE_M is `scale` m N-S but only scale*cos(lat) m E-W.
        # The smaller side bounds how far a vertex may move (see boundary_refine).
        pixel_m = cfg.PROCESS_SCALE_M * min(1.0, math.cos(math.radians(latitude)))
        pixel_area_m2 = cfg.PROCESS_SCALE_M * pixel_m
        _say(analysis_id, f"STAGE 4/5: boundary method = {'refined' if cfg.BOUNDARY_REFINEMENT_ENABLED else 'legacy'} "
                          f"(raster {cfg.PROCESS_SCALE_M:g} m, tile halo {cfg.TILE_HALO_M:g} m)")

        water_bodies, polygons_wgs84 = [], {}
        counter = 0
        for group in groups:
            frags = group["source_fragments"]
            stats = combine_probability_stats(frags)
            tile_ids = sorted({f["tile_id"].split(".")[0] for f in frags})
            frags_repaired = any(f.get("repaired") for f in frags)

            for poly_utm, info in refine_geometry_report(group["geometry"], pixel_size_m=pixel_m):
                if poly_utm.area < cfg.MIN_WATER_BODY_AREA_M2:
                    continue
                counter += 1
                m = measure_polygon(poly_utm, to_wgs84)
                flags = compute_quality_flags(
                    area_m2=m["area_m2"], mean_probability=stats["mean"], fraction_p75=stats["p75"],
                    was_repaired=bool(frags_repaired or info["repaired"]),
                    touches_aoi_edge=not aoi_inner.contains(poly_utm),
                )
                if m["compactness"] < cfg.RIVER_COMPACTNESS_MAX and m["area_m2"] >= cfg.RIVER_MIN_AREA_M2:
                    flags.append("river_like")   # v1's river-vs-lake shape rule
                if len(frags) > 1:
                    flags.append("merged_across_patches_approx_probability")
                if not stats["available"]:
                    flags.append("probability_stats_unavailable")

                wb = WaterBody(
                    id=f"WB-{counter:03d}",
                    area_m2=float(m["area_m2"]), area_ha=float(m["area_ha"]), area_km2=float(m["area_km2"]),
                    perimeter_m=float(m["perimeter_m"]),
                    centroid_lat=float(m["centroid_lat"]), centroid_lon=float(m["centroid_lon"]),
                    bbox_wgs84=tuple(float(v) for v in m["bbox_wgs84"]),
                    compactness=float(m["compactness"]), elongation=float(m["elongation"]),
                    bounding_box_ratio=float(m["bounding_box_ratio"]),
                    convex_hull_ratio=float(m["convex_hull_ratio"]),
                    mean_probability=float(stats["mean"]), median_probability=float(stats["median"]),
                    min_probability=float(stats["min"]), max_probability=float(stats["max"]),
                    fraction_p50=float(stats["p50"]), fraction_p75=float(stats["p75"]),
                    fraction_p90=float(stats["p90"]),
                    source_patch_ids=tile_ids, quality_flags=flags,
                    original_area=float(info["original_area"]),
                    refined_area=float(m["area_m2"]),
                    boundary_perimeter=float(m["perimeter_m"]),
                    confidence=float(stats["mean"]),
                    boundary_refinement_method=f"{info['method']}@{cfg.PROCESS_SCALE_M:g}m",
                    component_pixel_count=int(round(info["original_area"] / pixel_area_m2)),
                    boundary_pixel_count=int(round(info["original_perimeter"] / pixel_m)),
                )
                water_bodies.append(wb)
                polygons_wgs84[wb.id] = shp_transform(to_wgs84.transform, poly_utm)

        ranked = rank_water_bodies(water_bodies)
    except Exception as exc:
        _fail(analysis_id, "merge_refine_measure", exc)
        raise
    timings["merge_s"] = time.time() - t0
    _say(analysis_id, f"STAGE 4/5: SUCCESS — {len(ranked)} water bodies in {timings['merge_s']:.2f}s")

    timings["total_s"] = time.time() - t_start
    print("=" * 70, flush=True)
    _say(analysis_id, f"COMPLETE in {timings['total_s']:.1f}s — {len(ranked)} water bodies, "
                      f"{len(tile_errors)} tile errors")
    print("=" * 70, flush=True)

    return {
        "analysis_id": analysis_id,
        "aoi": {
            "latitude": latitude, "longitude": longitude, "radius_km": radius_km,
            "image_count": comp["image_count"], "composite_window": comp["window"],
        },
        "scene_metadata": {"image_count": comp["image_count"], **comp["window"]},
        "analysis_crs": utm_epsg,
        "water_bodies": ranked,
        "polygons_wgs84": polygons_wgs84,
        "patch_count": tile_count,
        "patch_errors": tile_errors,
        "timings": timings,
        "model_threshold": cfg.MODEL_THRESHOLD,
    }
