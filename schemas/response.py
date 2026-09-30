"""Pydantic response models for the U-Net water-body API."""

from typing import List, Optional, Tuple
from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    radius_km: float = Field(..., gt=0, le=100)
    start_date: str = "2024-01-01"
    end_date: str = "2024-06-30"
    max_cloud_percentage: float = Field(20.0, ge=0, le=100)


class WaterBodyResponse(BaseModel):
    id: str
    area_m2: float
    area_ha: float
    area_km2: float
    perimeter_m: float
    centroid_lat: float
    centroid_lon: float
    bbox_wgs84: Tuple[float, float, float, float]
    compactness: float
    elongation: float
    bounding_box_ratio: float
    convex_hull_ratio: float
    mean_probability: float
    median_probability: float
    min_probability: float
    max_probability: float
    fraction_p50: float
    fraction_p75: float
    fraction_p90: float
    source_patch_ids: List[str]
    quality_flags: List[str]
    ranking_score: Optional[float] = None
    rank: Optional[int] = None


class AnalysisSummary(BaseModel):
    total_water_bodies: int
    total_water_area_m2: float
    total_water_area_ha: float
    largest_water_body_id: Optional[str]
    largest_water_body_area_ha: Optional[float]
    average_area_ha: float
    highest_scoring_water_body_id: Optional[str]
    highest_scoring_water_body_score: Optional[float]
    mean_segmentation_score: float
    high_quality_count: int   # no quality_flags at all
    flagged_count: int        # at least one quality_flag


class AnalysisResponse(BaseModel):
    analysis_id: str
    aoi: dict
    analysis_crs: str
    model_threshold: float
    patch_count: int
    patch_error_count: int
    timings: dict
    water_bodies: List[WaterBodyResponse]
    summary: AnalysisSummary
    geojson: dict


class ErrorResponse(BaseModel):
    status: str = "error"
    message: str
    stage: Optional[str] = None
