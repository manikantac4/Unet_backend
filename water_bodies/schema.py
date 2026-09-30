"""
WaterBody data model.

Deliberately kept separate from raw segmentation/mask output — the
model answers "where is water"; this schema is what downstream
geospatial analytics/ranking/API/frontend actually consume.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple


@dataclass
class WaterBody:
    id: str  # "WB-001", "WB-002", ...

    # Geometry (geometry object itself kept separately by the caller;
    # this schema carries serializable measurements only)
    area_m2: float
    area_ha: float
    area_km2: float
    perimeter_m: float

    centroid_lat: float
    centroid_lon: float
    bbox_wgs84: Tuple[float, float, float, float]  # west, south, east, north

    # Shape descriptors (geometric only — not a water-quality claim)
    compactness: float
    elongation: float
    bounding_box_ratio: float
    convex_hull_ratio: float

    # Segmentation-quality statistics, computed from the model's
    # probability map within this water body's footprint. Explicitly
    # named "segmentation_score" / "model_probability" rather than
    # "confidence" — this is raw sigmoid output, not calibrated
    # confidence.
    mean_probability: float
    median_probability: float
    min_probability: float
    max_probability: float
    fraction_p50: float  # fraction of pixels with probability >= 0.50
    fraction_p75: float
    fraction_p90: float

    source_patch_ids: List[str] = field(default_factory=list)
    quality_flags: List[str] = field(default_factory=list)

    ranking_score: Optional[float] = None
    rank: Optional[int] = None

    # ---- Boundary-refinement diagnostics (additive; appended AFTER every existing
    # field so existing CSV column order is unchanged; None when not available).
    original_area: Optional[float] = None            # m2, raster polygon before smoothing
    refined_area: Optional[float] = None             # m2, polygon that is returned
    boundary_perimeter: Optional[float] = None       # m, refined boundary incl. holes (== perimeter_m)
    confidence: Optional[float] = None               # mean fused-score probability (== mean_probability)
    boundary_refinement_method: Optional[str] = None
    component_pixel_count: Optional[int] = None      # ~ original_area / pixel area (derived from the polygon)
    boundary_pixel_count: Optional[int] = None       # ~ raw boundary length / pixel size (= raw pixel edges)

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================
# QUALITY FLAGS
#
# Explicit, documented — not hidden inside unexplained logic. These
# exist specifically because the brief requires avoiding overstating
# model certainty, given the documented small-water-body weakness.
# ============================================================

QUALITY_FLAG_SMALL_AREA = "small_area"          # < 1000 m^2 — model is measurably weaker here
QUALITY_FLAG_LOW_PROBABILITY = "low_probability"  # mean probability close to threshold
QUALITY_FLAG_LOW_CONFIDENCE_FRACTION = "low_high_confidence_fraction"  # little of it is P>=0.75
QUALITY_FLAG_REPAIRED_GEOMETRY = "repaired_geometry"  # polygon needed shapely validity repair
QUALITY_FLAG_EDGE_OF_TILE = "edge_of_scan_area"  # touches the boundary of the scanned AOI


SMALL_AREA_THRESHOLD_M2 = 1000.0
LOW_PROBABILITY_THRESHOLD = 0.60
LOW_HIGH_CONFIDENCE_FRACTION_THRESHOLD = 0.20


def compute_quality_flags(
    area_m2: float,
    mean_probability: float,
    fraction_p75: float,
    was_repaired: bool,
    touches_aoi_edge: bool,
) -> List[str]:
    flags = []
    if area_m2 < SMALL_AREA_THRESHOLD_M2:
        flags.append(QUALITY_FLAG_SMALL_AREA)
    if mean_probability < LOW_PROBABILITY_THRESHOLD:
        flags.append(QUALITY_FLAG_LOW_PROBABILITY)
    if fraction_p75 < LOW_HIGH_CONFIDENCE_FRACTION_THRESHOLD:
        flags.append(QUALITY_FLAG_LOW_CONFIDENCE_FRACTION)
    if was_repaired:
        flags.append(QUALITY_FLAG_REPAIRED_GEOMETRY)
    if touches_aoi_edge:
        flags.append(QUALITY_FLAG_EDGE_OF_TILE)
    return flags
