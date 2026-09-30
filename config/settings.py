"""
Central configuration for the Traditional Model v2 backend.

Every tunable lives here (with an env-var override), so nothing is
scattered through the pipeline code.

Design note — why this file looks different from the old version
----------------------------------------------------------------
The previous v2 downloaded every 256x256 patch of the AOI from Earth
Engine to this machine and classified it with NumPy. In the field
log that meant 1,309 patches, ~830 of which failed ("Tile contains NaN
or infinite values") because a single Sentinel-2 scene rarely covers
a whole 25-50 km AOI, and ~19 minutes of wall-clock time.

v2 now follows the Mini-Backend ("Traditional v1") process shape:
the classification runs INSIDE Earth Engine on a cloud-masked
multi-date composite (full AOI coverage, no per-patch downloads), and
only the resulting water polygons come back to Python. What v2 keeps
from the old design is the *fused index score* idea, and what it adds
over v1 is described in gee_processor.py.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

STORAGE_DIR = BASE_DIR / "storage"   # analysis/{id}/... job outputs
CACHE_DIR = BASE_DIR / "cache"

ENGINE_NAME = "traditional_v2"

# ============================================================
# EARTH ENGINE
# ============================================================

GEE_PROJECT_ID = os.getenv("GEE_PROJECT_ID", "").strip().strip('"').strip("'")
GEE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON", "")

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
DEM_COLLECTION = "USGS/SRTMGL1_003"

# Sentinel-2 SCL classes removed before compositing:
#   0 no-data, 1 saturated/defective, 3 cloud shadow,
#   8/9 cloud (medium/high), 10 thin cirrus, 11 snow/ice.
# Class 2 ("dark area pixels") is kept on purpose — it contains real
# dark/deep water.
SCL_INVALID_CLASSES = [0, 1, 3, 8, 9, 10, 11]

# If the requested window has no usable scenes, retry with a looser
# cloud filter and then the whole calendar year before giving up.
FALLBACK_MAX_CLOUD_PERCENTAGE = 60.0

# ============================================================
# WATER-EVIDENCE SCORE (fused index ensemble)
#
#   S = wNDWI*NDWI + wMNDWI*MNDWI + wAWEI*clip(AWEI_nsh/0.25,-1,1)
#       - wNDVI*max(NDVI,0) - wNDBI*max(NDBI,0)
#
# Typical values: clear water 0.35-0.55, turbid water 0.0-0.15,
# bare soil/built-up/vegetation -0.2 to -0.8.
# ============================================================

W_NDWI = float(os.getenv("W_NDWI", "0.30"))
W_MNDWI = float(os.getenv("W_MNDWI", "0.35"))
W_AWEI = float(os.getenv("W_AWEI", "0.25"))
W_NDVI_PENALTY = float(os.getenv("W_NDVI_PENALTY", "0.20"))
W_NDBI_PENALTY = float(os.getenv("W_NDBI_PENALTY", "0.10"))
AWEI_SCALE = 0.25  # AWEI_nsh in reflectance units: water ~ +0.05..0.2, land ~ -0.3..-1.5

# ============================================================
# HYSTERESIS DECISION (replaces the old per-patch Otsu threshold)
#
# WHY: Otsu always splits a histogram in two. On a patch with no
# water it still "finds" a threshold, so half the land got labelled
# water (the log shows 30-50k "water" pixels per 65k-pixel patch at
# thresholds like -0.78). A fixed, physically anchored pair of
# thresholds cannot invent water where none exists.
#
#   CORE    : score >= SCORE_CORE_MIN  AND  every strict rule below
#   GROWN   : region-grow from the core through pixels with
#             score >= SCORE_GROW_MIN  (catches turbid/shallow
#             margins that are real water, but only when they touch
#             confident water)
# ============================================================

SCORE_CORE_MIN = float(os.getenv("SCORE_CORE_MIN", "0.05"))
SCORE_GROW_MIN = float(os.getenv("SCORE_GROW_MIN", "-0.05"))

CORE_MIN_MNDWI = float(os.getenv("CORE_MIN_MNDWI", "0.0"))
CORE_MAX_NDVI = float(os.getenv("CORE_MAX_NDVI", "0.25"))
CORE_MAX_NDBI = float(os.getenv("CORE_MAX_NDBI", "0.15"))
# AWEI_sh separates water from shadow (terrain / building / cloud).
CORE_MIN_AWEI_SH = float(os.getenv("CORE_MIN_AWEI_SH", "0.0"))

GROW_MAX_NDVI = float(os.getenv("GROW_MAX_NDVI", "0.35"))
GROW_MAX_NDBI = float(os.getenv("GROW_MAX_NDBI", "0.20"))
GROW_ITERATIONS = int(os.getenv("GROW_ITERATIONS", "5"))
GROW_RADIUS_PX = int(os.getenv("GROW_RADIUS_PX", "2"))

# Growth used to admit any low-vegetation/low-built-up pixel with
# score >= SCORE_GROW_MIN, with no requirement that it show ANY real
# water signal. On turbid/braided rivers that let growth crawl into
# plain damp soil and sandbar fringes (score there is often just as
# ambiguous as a real turbid margin), which is what turned a single
# river into a web of speckled false rings. Real turbid/shallow water
# almost always keeps at least a weak positive MNDWI or AWEI reading;
# dry/damp bare soil usually does not. Requiring one of the two still
# recovers genuine margins (the reason v2 has hysteresis at all) while
# rejecting land that only *looks* ambiguous on the fused score.
GROW_MIN_MNDWI = float(os.getenv("GROW_MIN_MNDWI", "-0.05"))
GROW_MIN_AWEI_SCALED = float(os.getenv("GROW_MIN_AWEI_SCALED", "-0.10"))

# Water is flat. Slopes above this are terrain shadow, not water.
MAX_SLOPE_DEG = float(os.getenv("MAX_SLOPE_DEG", "10"))

# ---- Mini-Backend (v1) rule set — the BASE mask, so v2 boundaries match v1 ----
V1_NDWI_MIN = float(os.getenv("V1_NDWI_MIN", "0.10"))
V1_MNDWI_MIN = float(os.getenv("V1_MNDWI_MIN", "0.02"))
V1_AWEI_MIN = float(os.getenv("V1_AWEI_MIN", "0.02"))
V1_NDWI_SECONDARY_MIN = float(os.getenv("V1_NDWI_SECONDARY_MIN", "0.0"))
V1_MAX_NDVI = float(os.getenv("V1_MAX_NDVI", "0.30"))
V1_MAX_NDBI = float(os.getenv("V1_MAX_NDBI", "0.15"))

# Raster clean-up (Earth Engine side), in 10 m pixels.
# Opening (erode -> dilate) runs BEFORE closing. It strips 1-pixel
# speckle and thin whiskers — the raster source of the tiny false
# rings — without touching a real body wider than 2*radius px. Set to
# 0 to disable and fall back to the old closing-only behaviour.
OPENING_RADIUS_PX = int(os.getenv("OPENING_RADIUS_PX", "1"))
CLOSING_RADIUS_PX = int(os.getenv("CLOSING_RADIUS_PX", "1"))
MIN_CONNECTED_PIXELS = int(os.getenv("MIN_CONNECTED_PIXELS", "15"))  # 800 m2

# ============================================================
# CONFIDENCE ("probability") SURFACE
#
# logistic((S - SCORE_GROW_MIN) / CONFIDENCE_SCALE): 0.5 exactly at
# the grow threshold, ~0.82 at the core threshold, ~0.98 for clear
# water. Fixed (not patch-dependent), so the same pixel always gets
# the same confidence regardless of what else is in the scan.
# ============================================================

CONFIDENCE_SCALE = float(os.getenv("CONFIDENCE_SCALE", "0.08"))
MODEL_THRESHOLD = 0.5   # semantics identical to the U-Net's `model_threshold`

# ============================================================
# TILED VECTORIZATION (Earth Engine -> polygons)
# ============================================================

PIXEL_SIZE_M = 10
TILE_KM = float(os.getenv("TILE_KM", "10"))
TILE_WORKERS = int(os.getenv("TILE_WORKERS", "3"))   # 6 triggered HTTP 429 (Earth Engine concurrency limit)
TILE_MAX_ATTEMPTS = int(os.getenv("TILE_MAX_ATTEMPTS", "5"))
TILE_MAX_SPLIT_DEPTH = int(os.getenv("TILE_MAX_SPLIT_DEPTH", "2"))
PROGRESS_LOG_EVERY_N_TILES = int(os.getenv("PROGRESS_LOG_EVERY_N_TILES", "5"))

# ============================================================
# POLYGON POST-PROCESSING (Python side, true metres in UTM)
# ============================================================

# Fragments closer than this are joined (heals tile seams and 1-pixel
# gaps; separate bodies further apart stay separate).
MERGE_CONNECT_DISTANCE_M = float(os.getenv("MERGE_CONNECT_DISTANCE_M", "20"))
FILL_HOLES_MAX_AREA_M2 = float(os.getenv("FILL_HOLES_MAX_AREA_M2", "2000"))
# Vector-side open (erode -> dilate): the Shapely counterpart of
# OPENING_RADIUS_PX above, for any thin spur that still slips through
# the raster stage (e.g. re-introduced at a tile seam during
# merge_fragments). Half a pixel-width — small enough to leave real
# shorelines untouched, big enough to pinch off anything under ~10 m
# wide. Set to 0 to disable.
DESPECKLE_RADIUS_M = float(os.getenv("DESPECKLE_RADIUS_M", "5"))
# Boundary drawing = Mini-Backend's: simplify (metres), no corner-cutting.
SIMPLIFY_TOLERANCE_M = float(os.getenv("SIMPLIFY_TOLERANCE_M", "8"))
RIVER_COMPACTNESS_MAX = 0.22   # v1 river/lake rule
RIVER_MIN_AREA_M2 = 3000.0
MIN_WATER_BODY_AREA_M2 = float(os.getenv("MIN_WATER_BODY_AREA_M2", "1000"))

# ============================================================
# BOUNDARY REFINEMENT
#
# One master switch.  BOUNDARY_REFINEMENT=legacy restores the previous
# behaviour exactly (10 m raster, no tile overlap, +/-20 m closing merge,
# 8 m Douglas-Peucker), which is what you A/B against.
#
#   refined (default):
#     1. Sentinel-2 bands are bilinear-resampled BEFORE compositing, so the
#        20 m SWIR bands (B11/B12) no longer arrive as 2x2 blocks.
#     2. The classifier is evaluated + vectorised at PROCESS_SCALE_M (5 m):
#        every pixel radius / area in the graph is rescaled so the physical
#        distances (grow reach, opening, closing, min size) are UNCHANGED;
#        only the location of the threshold crossing becomes sub-pixel.
#     3. Tiles overlap by TILE_HALO_M so a seam is never a polygon edge.
#     4. Fragments are merged by EXACT union (no shoreline-moving buffer).
#     5. Boundaries are smoothed with a displacement-clamped filter instead
#        of a coarse Douglas-Peucker.
# ============================================================

BOUNDARY_REFINEMENT_MODE = os.getenv("BOUNDARY_REFINEMENT", "refined").strip().lower()
BOUNDARY_REFINEMENT_ENABLED = BOUNDARY_REFINEMENT_MODE != "legacy"
_R = BOUNDARY_REFINEMENT_ENABLED

# Grid the classifier is evaluated and vectorised at. PIXEL_SIZE_M (10 m) stays
# the native Sentinel-2 grid that all *_PX settings above are expressed in.
PROCESS_SCALE_M = float(os.getenv("PROCESS_SCALE_M", "5" if _R else "10"))
S2_BILINEAR_RESAMPLE = os.getenv("S2_BILINEAR_RESAMPLE", "1" if _R else "0") == "1"

# Tile overlap (metres) added on every side of every vectorisation tile.
TILE_HALO_M = float(os.getenv("TILE_HALO_M", "40" if _R else "0"))

# Merge: fragments are snapped to this grid (mm-level float noise from the
# EE -> UTM reprojection) and unioned exactly. Fragments from DIFFERENT tiles
# that are still separated by a gap <= 2 * SEAM_HEAL_DISTANCE_M are bridged.
# Fragments of the same tile are never bridged, so real separate bodies (a
# dam, a levee, a road) keep their separation.
SEAM_SNAP_GRID_M = float(os.getenv("SEAM_SNAP_GRID_M", "0.01"))
SEAM_HEAL_DISTANCE_M = float(os.getenv("SEAM_HEAL_DISTANCE_M", "6"))

# Smoothing of the pixel staircase. All distances are in PIXELS of the raster the
# polygon came from, so the same values are right for 10 m and 5 m rasters.
#   REFINE_STEP_PX          edges are densified to at most this spacing
#   REFINE_SMOOTH_PASSES    [1 2 1]/4 smoothing passes
#   REFINE_MAX_SHIFT_PX     hard cap on how far ANY vertex may move from where the
#                           raster put it. Keep < 0.5: two bodies that are >= 1 px
#                           apart in the raster can then never touch or overlap.
#   REFINE_SIMPLIFY_PX      topology-preserving Douglas-Peucker on the smoothed line
#   REFINE_MAX_MEAN_SHIFT_PX  safety net on the IMPLIED mean boundary shift,
#                           |area change| / perimeter. (A relative area threshold would
#                           wrongly reject small ponds: a 30-pixel pond legitimately
#                           loses ~5 % area from corner rounding while its boundary moves
#                           0.07 px on average.) Above this the un-smoothed polygon is kept.
REFINE_STEP_PX = float(os.getenv("REFINE_STEP_PX", "0.5"))
REFINE_SMOOTH_PASSES = int(os.getenv("REFINE_SMOOTH_PASSES", "8"))
REFINE_MAX_SHIFT_PX = float(os.getenv("REFINE_MAX_SHIFT_PX", "0.4"))
REFINE_SIMPLIFY_PX = float(os.getenv("REFINE_SIMPLIFY_PX", "0.05"))
REFINE_MAX_MEAN_SHIFT_PX = float(os.getenv("REFINE_MAX_MEAN_SHIFT_PX", "0.25"))
# The raster stage already opens the mask (OPENING_RADIUS_PX); a second vector-side
# opening only rounds real corners, so it is off in refined mode.
REFINE_DESPECKLE_RADIUS_M = float(os.getenv("REFINE_DESPECKLE_RADIUS_M", "0"))

# ============================================================
# GEOSPATIAL / RANKING  (identical to Unet_backend so results line up)
# ============================================================

OUTPUT_CRS_WGS84 = "EPSG:4326"

RANKING_WEIGHTS = {
    "area_score": 0.35,
    "probability_score": 0.35,
    "geometry_score": 0.15,
    "high_confidence_fraction": 0.15,
}

assert abs(sum(RANKING_WEIGHTS.values()) - 1.0) < 1e-9, "RANKING_WEIGHTS must sum to 1.0"
assert SCORE_CORE_MIN > SCORE_GROW_MIN, "core threshold must be stricter than grow threshold"
assert CORE_MAX_NDVI <= GROW_MAX_NDVI and CORE_MAX_NDBI <= GROW_MAX_NDBI, (
    "core rules must be at least as strict as grow rules (core must be a subset of grow)"
)
assert 1.0 <= PROCESS_SCALE_M <= PIXEL_SIZE_M, "PROCESS_SCALE_M must be between 1 m and the native 10 m"
assert REFINE_MAX_SHIFT_PX + REFINE_SIMPLIFY_PX < 0.5, (
    "REFINE_MAX_SHIFT_PX + REFINE_SIMPLIFY_PX must stay below 0.5 px, otherwise two bodies "
    "one pixel apart could be pushed into each other"
)

# ============================================================
# STORAGE
# ============================================================

STORAGE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
