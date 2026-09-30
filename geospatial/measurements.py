"""
Geospatially accurate area/perimeter/centroid/shape measurements.

Per the project brief: area/perimeter must NOT be
"number_of_pixels x arbitrary pixel size" unless that's actually
valid for the raster's CRS. Here, polygons coming out of
polygonize.py are already in a real projected CRS (UTM, chosen per
AOI — see satellite/tiler.py's use of a metric grid), so their
.area / .length in Shapely ARE already in m^2 / m directly, with no
extra reprojection needed at measurement time.

For user-facing coordinates (centroid lat/lon), everything is
transformed to WGS84 via pyproj — established library, not
hand-rolled trigonometry.
"""

import math

import pyproj
from shapely.geometry import Polygon
from shapely.ops import transform as shapely_transform


def utm_epsg_for_lonlat(lon: float, lat: float) -> str:
    """Auto-select the appropriate UTM zone EPSG code for a WGS84
    point — used when polygons need to be measured in a metric CRS
    but were produced in degrees."""
    zone = int(math.floor((lon + 180) / 6) + 1)
    hemisphere = "326" if lat >= 0 else "327"  # EPSG 326xx (N) / 327xx (S)
    return f"EPSG:{hemisphere}{zone:02d}"


def get_transformer(src_crs: str, dst_crs: str) -> pyproj.Transformer:
    return pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)


def measure_polygon(
    polygon_metric: Polygon,
    to_wgs84_transformer: pyproj.Transformer,
) -> dict:
    """
    polygon_metric: a Shapely Polygon already in a projected (metric)
    CRS (e.g. UTM) — .area is m^2, .length is m directly.
    to_wgs84_transformer: a pyproj.Transformer from that metric CRS
    to EPSG:4326, for user-facing lat/lon output.

    Returns area/perimeter/centroid/bbox/shape-descriptor metrics.
    """
    area_m2 = polygon_metric.area
    perimeter_m = polygon_metric.length

    centroid_metric = polygon_metric.centroid
    centroid_wgs84 = shapely_transform(to_wgs84_transformer.transform, centroid_metric)

    bounds_metric = polygon_metric.bounds  # (minx, miny, maxx, maxy)
    polygon_wgs84 = shapely_transform(to_wgs84_transformer.transform, polygon_metric)
    bounds_wgs84 = polygon_wgs84.bounds

    # ---- shape descriptors (geometric only — not a quality claim) ----
    compactness = (4 * math.pi * area_m2 / (perimeter_m ** 2)) if perimeter_m > 0 else 0.0

    convex_hull = polygon_metric.convex_hull
    convex_hull_ratio = (area_m2 / convex_hull.area) if convex_hull.area > 0 else 0.0

    minx, miny, maxx, maxy = bounds_metric
    bbox_w = maxx - minx
    bbox_h = maxy - miny
    bbox_area = bbox_w * bbox_h
    bounding_box_ratio = (area_m2 / bbox_area) if bbox_area > 0 else 0.0
    elongation = (max(bbox_w, bbox_h) / min(bbox_w, bbox_h)) if min(bbox_w, bbox_h) > 0 else 0.0

    return {
        "area_m2": area_m2,
        "area_ha": area_m2 / 10_000.0,
        "area_km2": area_m2 / 1_000_000.0,
        "perimeter_m": perimeter_m,
        "centroid_lat": centroid_wgs84.y,
        "centroid_lon": centroid_wgs84.x,
        "bbox_wgs84": bounds_wgs84,  # (minx, miny, maxx, maxy) = (west, south, east, north)
        "compactness": compactness,        # 1.0 = perfect circle
        "circularity": compactness,        # alias, same formula
        "elongation": elongation,          # bbox long/short side ratio
        "bounding_box_ratio": bounding_box_ratio,  # area / bbox area
        "convex_hull_ratio": convex_hull_ratio,    # area / convex-hull area
    }
