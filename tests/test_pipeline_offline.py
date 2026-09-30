"""
Offline tests: no Earth Engine needed. The two Earth Engine touch-points
(composite/classify and the tile scan) are replaced with synthetic data,
everything downstream (merge, refine, measure, rank, API schema) is real.

Run:  cd Traditional-V2-Backend && python -m pytest tests -q   (or python tests/test_pipeline_offline.py)
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shapely.geometry import MultiPolygon, Polygon, box

from geospatial.boundary_refine import refine_geometry
from services.tiled_water_detection import generate_tiles, plan_tiles

LAT, LON = 16.5, 80.6


def _lonlat_box(cx_m, cy_m, w_m, h_m):
    dlat = 1 / 111320.0
    dlon = 1 / (111320.0 * math.cos(math.radians(LAT)))
    x0, y0 = LON + cx_m * dlon, LAT + cy_m * dlat
    b = box(x0, y0, x0 + w_m * dlon, y0 + h_m * dlat)
    return b.__geo_interface__


def test_refine_handles_multipolygon():
    # the exact failure from the field log
    mp = MultiPolygon([box(0, 0, 100, 100), box(300, 0, 420, 90)])
    out = refine_geometry(mp)
    assert len(out) == 2 and all(p.is_valid and p.area > 5000 for p in out)
    holey = Polygon(box(0, 0, 200, 200).exterior, [box(50, 50, 60, 60).exterior.coords])
    out = refine_geometry(holey)
    assert out and not out[0].interiors  # 100 m2 hole filled


def test_tiles_cover_circle_and_skip_outside():
    tiles = plan_tiles(LAT, LON, 25)
    assert 20 < len(tiles) < 40            # ~ pi*25^2/100 = 20 + edge cells, not the 1309 patches
    full = generate_tiles(LON - .3, LAT - .3, LON + .3, LAT + .3, 10)
    assert len(tiles) < len(generate_tiles(LON - .23, LAT - .23, LON + .23, LAT + .23, 10)) + 1
    assert full


def test_full_api_with_mocked_earth_engine(monkeypatch=None):
    import gee_processor
    from services import analysis_service
    from fastapi.testclient import TestClient

    lake = _lonlat_box(-3000, -2000, 400, 300)          # one lake cut in two by a tile seam
    left = _lonlat_box(-3000, -2000, 200, 300)
    right = _lonlat_box(-2800, -2000, 200, 300)
    pond = _lonlat_box(2000, 1000, 120, 90)             # 10,800 m2 pond
    speck = _lonlat_box(500, 500, 30, 30)               # 900 m2 -> must be dropped
    st = {"mean": 0.9, "median": 0.92, "min": 0.6, "max": 0.99, "p50": 1.0, "p75": 0.95, "p90": 0.8}
    frags = [
        {"geometry": left, "tile_id": "T001", "stats": st},
        {"geometry": right, "tile_id": "T002", "stats": dict(st, mean=0.7, p75=0.5)},
        {"geometry": pond, "tile_id": "T003", "stats": st},
        {"geometry": speck, "tile_id": "T003", "stats": st},
        {"geometry": pond, "tile_id": "T009", "stats": {k: None for k in st}},  # stats-less duplicate area
    ]

    gee_processor.initialize_earth_engine = lambda: None
    gee_processor.create_aoi = lambda *a, **k: object()
    gee_processor.build_composite = lambda *a, **k: {
        "composite": object(), "image_count": 7,
        "window": {"start_date": "2024-01-01", "end_date": "2024-06-30", "max_cloud_percentage": 20, "note": "requested window"}}
    gee_processor.classify = lambda *a, **k: {"mask": 1, "stats_image": 2, "prob": 3}
    analysis_service.scan_aoi = lambda *a, **k: {"fragments": frags, "tile_count": 25, "tile_errors": []}

    from app import app
    c = TestClient(app)
    r = c.post("/analysis", json={"latitude": LAT, "longitude": LON, "radius_km": 10})
    assert r.status_code == 200, r.text
    d = r.json()

    unet_keys = {"analysis_id", "aoi", "analysis_crs", "model_threshold", "patch_count",
                 "patch_error_count", "timings", "water_bodies", "summary", "geojson"}
    assert unet_keys <= set(d) and d["engine"] == "traditional_v2"
    wbs = d["water_bodies"]
    assert len(wbs) == 2, [w["area_m2"] for w in wbs]                     # lake healed into ONE, pond kept, speck dropped
    big = max(wbs, key=lambda w: w["area_m2"])
    assert abs(big["area_m2"] - 120000) / 120000 < 0.08
    assert big["source_patch_ids"] == ["T001", "T002"]
    assert "merged_across_patches_approx_probability" in big["quality_flags"]
    assert {w["rank"] for w in wbs} == {1, 2}
    assert len(d["geojson"]["features"]) == 2
    assert d["patch_count"] == 25 and d["patch_error_count"] == 0
    for path in ("water_bodies.geojson", "water_bodies.csv", "ranked_water_bodies.csv", "statistics.csv"):
        assert c.get(f"/downloads/{d['analysis_id']}/{path}").status_code == 200
    assert c.get("/statistics", params={"analysis_id": d["analysis_id"]}).status_code == 200
    assert c.get("/ranking", params={"analysis_id": d["analysis_id"]}).status_code == 200
    print("OK", [(w["id"], round(w["area_m2"]), w["quality_flags"]) for w in wbs])


if __name__ == "__main__":
    test_refine_handles_multipolygon(); test_tiles_cover_circle_and_skip_outside(); test_full_api_with_mocked_earth_engine()
    print("all offline tests passed")
