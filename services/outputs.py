"""
Generates downloadable output files from a completed analysis.

Only formats that can be built correctly from what this pipeline
actually produces are generated — per the brief, "only generate
formats that are actually supported and valid" rather than faking a
broken/empty file.

IMPLEMENTED: water_bodies.geojson, water_bodies.csv,
ranked_water_bodies.csv, statistics.csv.

NOT YET IMPLEMENTED (documented, not faked): water_probability.tif /
water_mask.tif would require assembling one common full-AOI raster
grid across all overlapping patches (a real reprojection/resampling
task, not just concatenation) — this pass keeps probability rasters
per-patch only (used for water-body statistics) and does not mosaic
them into one exportable GeoTIFF. water_bodies.gpkg,
analysis_report.pdf, and visualization.png are likewise left for a
follow-up pass. See ARCHITECTURE.md "Known limitations".
"""

import json

import pandas as pd

from water_bodies.schema import WaterBody


def water_bodies_to_geojson(water_bodies: list, polygons_wgs84: dict) -> dict:
    """
    polygons_wgs84: {wb.id: shapely Polygon in WGS84} — the caller
    (routes/prediction.py) keeps geometry alongside the WaterBody
    objects returned by the pipeline; this function assembles the
    final FeatureCollection.
    """
    features = []
    for wb in water_bodies:
        geom = polygons_wgs84.get(wb.id)
        if geom is None:
            continue
        features.append({
            "type": "Feature",
            "geometry": json.loads(json.dumps(geom.__geo_interface__)),
            "properties": wb.to_dict(),
        })
    return {"type": "FeatureCollection", "features": features}


def water_bodies_to_dataframe(water_bodies: list) -> pd.DataFrame:
    rows = [wb.to_dict() for wb in water_bodies]
    return pd.DataFrame(rows)


def write_analysis_outputs(analysis_dir, water_bodies: list, polygons_wgs84: dict):
    """Writes water_bodies.geojson, water_bodies.csv,
    ranked_water_bodies.csv, statistics.csv into
    analysis_dir/polygons and analysis_dir/statistics."""

    geojson = water_bodies_to_geojson(water_bodies, polygons_wgs84)
    geojson_path = analysis_dir / "polygons" / "water_bodies.geojson"
    geojson_path.write_text(json.dumps(geojson, indent=2))

    df = water_bodies_to_dataframe(water_bodies)

    csv_path = analysis_dir / "statistics" / "water_bodies.csv"
    df.to_csv(csv_path, index=False)

    ranked_df = df.sort_values("rank", na_position="last")
    ranked_csv_path = analysis_dir / "statistics" / "ranked_water_bodies.csv"
    ranked_df.to_csv(ranked_csv_path, index=False)

    stats_summary = _build_stats_summary(water_bodies)
    stats_path = analysis_dir / "statistics" / "statistics.csv"
    pd.DataFrame([stats_summary]).to_csv(stats_path, index=False)

    return {
        "water_bodies.geojson": geojson_path,
        "water_bodies.csv": csv_path,
        "ranked_water_bodies.csv": ranked_csv_path,
        "statistics.csv": stats_path,
    }


def _build_stats_summary(water_bodies: list) -> dict:
    if not water_bodies:
        return {
            "total_water_bodies": 0, "total_water_area_m2": 0, "total_water_area_ha": 0,
            "largest_water_body_id": None, "largest_water_body_area_ha": None,
            "average_area_ha": 0, "highest_scoring_water_body_id": None,
            "highest_scoring_water_body_score": None, "mean_segmentation_score": 0,
            "high_quality_count": 0, "flagged_count": 0,
        }

    total_area_m2 = sum(wb.area_m2 for wb in water_bodies)
    largest = max(water_bodies, key=lambda wb: wb.area_m2)
    best = max(water_bodies, key=lambda wb: wb.ranking_score or 0)

    return {
        "total_water_bodies": len(water_bodies),
        "total_water_area_m2": total_area_m2,
        "total_water_area_ha": total_area_m2 / 10_000.0,
        "largest_water_body_id": largest.id,
        "largest_water_body_area_ha": largest.area_ha,
        "average_area_ha": (total_area_m2 / 10_000.0) / len(water_bodies),
        "highest_scoring_water_body_id": best.id,
        "highest_scoring_water_body_score": best.ranking_score,
        "mean_segmentation_score": sum(wb.mean_probability for wb in water_bodies) / len(water_bodies),
        "high_quality_count": sum(1 for wb in water_bodies if not wb.quality_flags),
        "flagged_count": sum(1 for wb in water_bodies if wb.quality_flags),
    }
