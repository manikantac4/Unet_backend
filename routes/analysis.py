"""
API routes for the standalone Traditional Model v2 backend.

Route paths deliberately mirror Unet_backend/routes/prediction.py's
paths exactly (POST /analysis, GET /analysis/{id}, /water-bodies,
/statistics, /ranking, /downloads/...) — same names, just served by
a completely separate process/port, since this is now its own
backend rather than a sub-router mounted inside Unet_backend.

Response model: AnalysisResponseV2 = AnalysisResponse (this
project's own schemas/response.py, not imported from Unet_backend)
plus one extra `engine` field. See services/traditional_v2_service.py
for how the Earth-Engine-side detection feeds the same
water_bodies/schema.py, geospatial/measurements.py and
ranking/ranking_engine.py logic Unet_backend uses.
"""

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from config.settings import STORAGE_DIR
from services.analysis_service import run_analysis_v2, ENGINE_NAME
from services.outputs import write_analysis_outputs, _build_stats_summary, water_bodies_to_geojson
from schemas.response import (
    AnalysisRequest, AnalysisResponse, AnalysisSummary,
    WaterBodyResponse, ErrorResponse,
)

logger = logging.getLogger("traditional_v2_backend.routes_analysis")
router = APIRouter(tags=["traditional-v2"])

_ANALYSIS_CACHE: dict = {}


class AnalysisResponseV2(AnalysisResponse):
    engine: str = ENGINE_NAME


def _to_response(result: dict) -> AnalysisResponseV2:
    water_bodies = result["water_bodies"]
    summary_dict = _build_stats_summary(water_bodies)
    geojson = water_bodies_to_geojson(water_bodies, result.get("polygons_wgs84", {}))

    return AnalysisResponseV2(
        analysis_id=result["analysis_id"],
        aoi=result["aoi"],
        analysis_crs=result["analysis_crs"],
        model_threshold=result["model_threshold"],
        patch_count=result["patch_count"],
        patch_error_count=len(result["patch_errors"]),
        timings=result["timings"],
        water_bodies=[WaterBodyResponse(**wb.to_dict()) for wb in water_bodies],
        summary=AnalysisSummary(**summary_dict),
        geojson=geojson,
    )


@router.post("/analysis", response_model=AnalysisResponseV2, responses={500: {"model": ErrorResponse}})
def create_analysis(request: AnalysisRequest):
    print(f"[traditional_v2 route] REQUEST RECEIVED: {(request.model_dump() if hasattr(request, 'model_dump') else request.dict())}", flush=True)
    try:
        result = run_analysis_v2(
            latitude=request.latitude,
            longitude=request.longitude,
            radius_km=request.radius_km,
            start_date=request.start_date,
            end_date=request.end_date,
            max_cloud_percentage=request.max_cloud_percentage,
        )
    except RuntimeError as exc:
        print(f"[traditional_v2 route] REQUEST FAILED (input problem): {exc}", flush=True)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        print(f"[traditional_v2 route] REQUEST FAILED (server error): {exc}", flush=True)
        logger.exception("Traditional v2 analysis failed")
        raise HTTPException(status_code=500, detail=str(exc))

    analysis_dir = STORAGE_DIR / result["analysis_id"]
    files = write_analysis_outputs(
        analysis_dir, result["water_bodies"], result["polygons_wgs84"]
    )

    _ANALYSIS_CACHE[result["analysis_id"]] = {"result": result, "files": files}

    print(f"[traditional_v2 route] RESPONSE SENT: analysis_id={result['analysis_id']} "
          f"water_bodies={len(result['water_bodies'])}", flush=True)
    return _to_response(result)


def _get_cached(analysis_id: str) -> dict:
    entry = _ANALYSIS_CACHE.get(analysis_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Unknown analysis_id: {analysis_id}")
    return entry


@router.get("/analysis/{analysis_id}", response_model=AnalysisResponseV2)
def get_analysis(analysis_id: str):
    entry = _get_cached(analysis_id)
    return _to_response(entry["result"])


@router.get("/water-bodies", response_model=list)
def list_water_bodies(analysis_id: str):
    entry = _get_cached(analysis_id)
    return [WaterBodyResponse(**wb.to_dict()) for wb in entry["result"]["water_bodies"]]


@router.get("/water-bodies/{water_body_id}", response_model=WaterBodyResponse)
def get_water_body(water_body_id: str, analysis_id: str):
    entry = _get_cached(analysis_id)
    for wb in entry["result"]["water_bodies"]:
        if wb.id == water_body_id:
            return WaterBodyResponse(**wb.to_dict())
    raise HTTPException(status_code=404, detail=f"Unknown water body: {water_body_id}")


@router.get("/statistics", response_model=AnalysisSummary)
def get_statistics(analysis_id: str):
    entry = _get_cached(analysis_id)
    return AnalysisSummary(**_build_stats_summary(entry["result"]["water_bodies"]))


@router.get("/ranking", response_model=list)
def get_ranking(analysis_id: str, limit: int = 10):
    entry = _get_cached(analysis_id)
    ranked = sorted(
        entry["result"]["water_bodies"],
        key=lambda wb: wb.rank if wb.rank is not None else 10**9,
    )
    return [WaterBodyResponse(**wb.to_dict()) for wb in ranked[:limit]]


@router.get("/downloads/{analysis_id}/{filename}")
def download_file(analysis_id: str, filename: str):
    entry = _get_cached(analysis_id)
    files = entry["files"]
    if filename not in files:
        raise HTTPException(
            status_code=404,
            detail=f"'{filename}' was not generated for this analysis. "
                   f"Available: {list(files.keys())}",
        )
    return FileResponse(path=str(files[filename]), filename=filename)
