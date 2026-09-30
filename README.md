# Traditional Model v2 Backend (port 8002)

Same input/output contract as `Unet_backend` (POST `/analysis` with
`latitude, longitude, radius_km[, start_date, end_date, max_cloud_percentage]`;
response = U-Net `AnalysisResponse` + `engine`). Process shape follows
`Mini-Backend` (Traditional v1): classification runs inside Earth Engine, only polygons come back.

## Run
    cd Traditional-V2-Backend
    pip install -r requirements.txt
    earthengine authenticate        # once; GEE_PROJECT_ID is in .env
    uvicorn app:app --port 8002
    python tests/test_pipeline_offline.py   # offline tests, no GEE needed

## What was wrong in the old run (from the log) and the fix
| Problem in log | Cause | Fix |
|---|---|---|
| 829 of 1309 patches "Tile contains NaN or infinite values" | one "best" Sentinel-2 scene doesn't cover the AOI; SCL-masked holes | cloud-masked multi-date median composite (`gee_processor.build_composite`), no patch downloads |
| ~half of land patches labelled water (30-50k px at score -0.78) | per-patch Otsu always splits a histogram | fixed core + grow hysteresis thresholds, AWEI_sh + SRTM slope shadow rejection |
| `'MultiPolygon' object has no attribute 'interiors'` | refine assumed Polygon | `geospatial/boundary_refine.py` works on lists of Polygons |
| 19 min, thousands of rasterio TIFF warnings | per-patch GeoTIFF downloads | server-side processing, parallel 10 km tile vectorisation |

`patch_count` / `patch_error_count` in the response now mean vector tiles scanned / failed.
Tunables (thresholds, weights, tile size) are in `config/settings.py` and overridable via env vars.
