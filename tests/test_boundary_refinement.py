"""
Offline tests for the boundary-refinement changes (no Earth Engine needed).

Run:  cd Traditional-V2-Backend && python -m pytest tests -q
"""
import io
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import shapely
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union

import boundary_benchmark as bench
from config import settings as cfg
from geospatial import boundary_refine, merge

STATS = {"mean": 0.9, "median": 0.9, "min": 0.6, "max": 0.99, "p50": 1.0, "p75": 0.95, "p90": 0.8}


def frag(geom, tile):
    return {"geometry": geom, "area_m2": geom.area, "tile_id": tile, "stats": STATS}


def staircase_disc(cx, cy, r, pix):
    """Pixel-edge polygon of a disc, i.e. what reduceToVectors hands us."""
    n = int(2 * r / pix) + 4
    xs = cx - n * pix / 2 + (np.arange(n) + 0.5) * pix
    boxes = [box(x - pix / 2, y - pix / 2, x + pix / 2, y + pix / 2)
             for x in xs for y in xs if (x - cx) ** 2 + (y - cy) ** 2 <= r * r]
    return unary_union(boxes)


# ------------------------------------------------------------- merging ----
def test_exact_merge_is_seamless_and_never_moves_a_shoreline():
    lake = Point(500, 500).buffer(300, 64)
    left, right = lake.intersection(box(0, 0, 500, 1000)), lake.intersection(box(500, 0, 1000, 1000))
    groups = merge.merge_fragments([frag(left, "T001"), frag(right, "T002")])
    assert len(groups) == 1
    # exact up to the 1 cm snap grid (~0.2 m2 on a 283,000 m2 lake), not "+/- 20 m closing"
    assert abs(groups[0]["geometry"].area - lake.area) < 1.0
    assert groups[0]["geometry"].symmetric_difference(lake).area < 1.0


def test_overlapping_halo_fragments_dissolve():
    lake = Point(500, 500).buffer(300, 64)
    a = lake.intersection(box(0, 0, 540, 1000))                            # tile A + 40 m halo
    b = lake.intersection(box(460, 0, 1000, 1000))                         # tile B + 40 m halo
    groups = merge.merge_fragments([frag(a, "T001"), frag(b, "T002")])
    assert len(groups) == 1 and abs(groups[0]["geometry"].area - lake.area) < 1.0


def test_seam_gap_bridged_only_across_tiles():
    gap = 8.0                                                              # < 2 * SEAM_HEAL_DISTANCE_M
    left, right = box(0, 0, 100, 100), box(100 + gap, 0, 200 + gap, 100)
    healed = merge.merge_fragments([frag(left, "T001"), frag(right, "T002")])
    assert len(healed) == 1                                                                # a seam artefact: healed
    assert healed[0]["geometry"].area > left.area + right.area + 0.9 * gap * 100          # the WHOLE 8 m strip is filled
    assert healed[0]["geometry"].area < left.area + right.area + gap * 100 + 1.0           # ...and nothing outside it
    assert len(merge.merge_fragments([frag(left, "T001"), frag(right, "T001")])) == 2   # a real gap: kept


def test_dam_between_ponds_is_not_welded():
    a, b = box(0, 0, 100, 100), box(125, 0, 225, 100)                      # 25 m dam, same tile
    assert len(merge.merge_fragments([frag(a, "T001"), frag(b, "T001")])) == 2
    old = cfg.BOUNDARY_REFINEMENT_ENABLED
    try:
        cfg.BOUNDARY_REFINEMENT_ENABLED = False                            # legacy closing welds them (documented change)
        assert len(merge.merge_fragments([frag(a, "T001"), frag(b, "T001")])) == 1
    finally:
        cfg.BOUNDARY_REFINEMENT_ENABLED = old


def test_stats_are_still_combined_across_fragments():
    lake = Point(500, 500).buffer(300, 64)
    left, right = lake.intersection(box(0, 0, 500, 1000)), lake.intersection(box(500, 0, 1000, 1000))
    g = merge.merge_fragments([frag(left, "T001"), frag(right, "T002")])[0]
    assert len(g["source_fragments"]) == 2
    assert merge.combine_probability_stats(g["source_fragments"])["available"]


# ---------------------------------------------------------- refinement ----
def test_vertices_never_move_more_than_the_cap():
    pix = 10.0
    raw = staircase_disc(500, 500, 180, pix)
    (poly, info), = boundary_refine.refine_geometry_report(raw, pixel_size_m=pix)
    assert info["method"] == boundary_refine.METHOD_REFINED
    pts = shapely.line_interpolate_point(poly.exterior, np.linspace(0, poly.exterior.length, 4000, endpoint=False))
    worst = shapely.distance(pts, raw.boundary).max()
    assert worst <= (cfg.REFINE_MAX_SHIFT_PX + cfg.REFINE_SIMPLIFY_PX) * pix + 1e-6, worst


def test_smoothing_is_closer_to_truth_than_the_raw_staircase():
    pix, truth = 10.0, Point(503.3, 497.1).buffer(180, 128)
    raw = staircase_disc(503.3, 497.1, 180, pix)
    (poly, _), = boundary_refine.refine_geometry_report(raw, pixel_size_m=pix)

    def rms(g):
        pts = shapely.line_interpolate_point(g.exterior, np.linspace(0, g.exterior.length, 3000, endpoint=False))
        return float(np.sqrt((shapely.distance(pts, truth.boundary) ** 2).mean()))
    assert rms(poly) < 0.7 * rms(raw)


def test_bodies_one_pixel_apart_stay_disjoint():
    pix = 10.0
    a, b = staircase_disc(300, 300, 90, pix), staircase_disc(300 + 180 + pix, 300, 90, pix)   # 1 px gap
    assert a.distance(b) >= pix - 1e-9
    ra = boundary_refine.refine_geometry(a, pixel_size_m=pix)
    rb = boundary_refine.refine_geometry(b, pixel_size_m=pix)
    assert all(not p.intersects(q) and p.distance(q) > 0 for p in ra for q in rb)


def test_island_kept_and_smoothed_small_hole_filled():
    pix = 10.0
    shell = staircase_disc(500, 500, 300, pix)
    big_island = staircase_disc(500, 500, 35, pix)                         # ~3,800 m2 > 2,000 m2: real island
    speck = box(700, 500, 710, 510)                                        # 100 m2: noise
    raw = shell.difference(big_island).difference(speck)
    (poly, info), = boundary_refine.refine_geometry_report(raw, pixel_size_m=pix)
    assert len(poly.interiors) == 1 and poly.is_valid and info["method"] == boundary_refine.METHOD_REFINED


def test_invalid_input_is_repaired_and_flagged_not_dropped():
    bowtie = Polygon([(0, 0), (100, 100), (100, 0), (0, 100)])            # self-intersecting
    out = boundary_refine.refine_geometry_report(bowtie, pixel_size_m=10.0)
    assert out and all(p.is_valid for p, _ in out) and all(i["repaired"] for _, i in out)
    assert sum(p.area for p, _ in out) > 0.9 * shapely.make_valid(bowtie).area


def test_refined_outputs_are_valid_polygons_for_multipolygon_input():
    mp = shapely.MultiPolygon([box(0, 0, 100, 100), box(300, 0, 420, 90)])
    out = boundary_refine.refine_geometry(mp)
    assert len(out) == 2 and all(p.is_valid and p.area > 5000 for p in out)


def test_legacy_switch_restores_old_geometry_pipeline():
    raw = staircase_disc(500, 500, 180, 10.0)
    old = cfg.BOUNDARY_REFINEMENT_ENABLED
    try:
        cfg.BOUNDARY_REFINEMENT_ENABLED = False
        (p, info), = boundary_refine.refine_geometry_report(raw, pixel_size_m=10.0)
        assert info["method"] == boundary_refine.METHOD_LEGACY
        assert p.equals_exact(boundary_refine.refine_polygon(boundary_refine.despeckle(raw))[0], 0)
    finally:
        cfg.BOUNDARY_REFINEMENT_ENABLED = old


# ----------------------------------------------------------- benchmark ----
def test_refined_pipeline_beats_legacy_on_synthetic_truth():
    r = bench.run_all(noise=0.05)
    leg, r10, r5 = r["legacy_10m"], r["refined_10m"], r["refined_5m_sub"]
    assert r10["rms_err_m"] < 0.65 * leg["rms_err_m"]
    assert r5["rms_err_m"] < 0.80 * r10["rms_err_m"]
    assert abs(r5["area_err_pct"]) < abs(leg["area_err_pct"])
    assert r10["seam_rms_err_m"] < leg["seam_rms_err_m"]
    for res in (r10, r5):
        assert res["valid"] and res["overlaps"] == 0 and res["dam_kept"] and res["island_kept"]
        assert res["n_bodies"] == res["n_truth"]
    assert not leg["dam_kept"]                                             # the old +/-20 m closing welded the ponds


# -------------------------------------------------------- API contract ----
ORIGINAL_CSV_COLUMNS = [
    "id", "area_m2", "area_ha", "area_km2", "perimeter_m", "centroid_lat", "centroid_lon", "bbox_wgs84",
    "compactness", "elongation", "bounding_box_ratio", "convex_hull_ratio", "mean_probability",
    "median_probability", "min_probability", "max_probability", "fraction_p50", "fraction_p75",
    "fraction_p90", "source_patch_ids", "quality_flags", "ranking_score", "rank",
]
DIAGNOSTIC_FIELDS = ["original_area", "refined_area", "boundary_perimeter", "confidence",
                     "boundary_refinement_method", "component_pixel_count", "boundary_pixel_count"]


def test_api_contract_unchanged_and_diagnostics_added():
    import pandas as pd
    from fastapi.testclient import TestClient

    import gee_processor
    from schemas.response import WaterBodyResponse
    from services import analysis_service
    from test_pipeline_offline import LAT, LON, _lonlat_box

    frags = [
        {"geometry": _lonlat_box(-3000, -2000, 200, 300), "tile_id": "T001", "stats": STATS},
        {"geometry": _lonlat_box(-2800, -2000, 200, 300), "tile_id": "T002", "stats": STATS},
        {"geometry": _lonlat_box(2000, 1000, 120, 90), "tile_id": "T003", "stats": STATS},
    ]
    gee_processor.initialize_earth_engine = lambda: None
    gee_processor.create_aoi = lambda *a, **k: object()
    gee_processor.build_composite = lambda *a, **k: {"composite": object(), "image_count": 3, "window": {"note": "x"}}
    gee_processor.classify = lambda *a, **k: {"mask": 1, "stats_image": 2, "prob": 3}
    analysis_service.scan_aoi = lambda *a, **k: {"fragments": frags, "tile_count": 4, "tile_errors": []}

    from app import app
    c = TestClient(app)
    d = c.post("/analysis", json={"latitude": LAT, "longitude": LON, "radius_km": 10}).json()

    # response model: exactly the fields it always had (diagnostics are NOT added to WaterBodyResponse)
    assert set(d["water_bodies"][0]) == set(WaterBodyResponse.model_fields)
    assert not set(DIAGNOSTIC_FIELDS) & set(d["water_bodies"][0])
    # GeoJSON properties + CSV: old fields untouched, diagnostics appended
    props = d["geojson"]["features"][0]["properties"]
    assert set(ORIGINAL_CSV_COLUMNS) <= set(props) and set(DIAGNOSTIC_FIELDS) <= set(props)
    for name in ("water_bodies.csv", "ranked_water_bodies.csv"):
        df = pd.read_csv(io.StringIO(c.get(f"/downloads/{d['analysis_id']}/{name}").text))
        assert list(df.columns) == ORIGINAL_CSV_COLUMNS + DIAGNOSTIC_FIELDS
    f = props
    assert f["boundary_refinement_method"].startswith(boundary_refine.METHOD_REFINED)
    assert f["refined_area"] == f["area_m2"] and f["boundary_perimeter"] == f["perimeter_m"]
    assert f["component_pixel_count"] > 0 and f["boundary_pixel_count"] > 0
    assert math.isclose(f["confidence"], f["mean_probability"])
