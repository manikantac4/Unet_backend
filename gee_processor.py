"""
Earth Engine side of Traditional Model v2.

Same process shape as Mini-Backend/gee_processor.py ("Traditional
v1"): everything heavy — compositing, index maths, classification,
clean-up — is expressed as a lazy Earth Engine graph and evaluated
server-side. Nothing is downloaded pixel-by-pixel; only vector
polygons (plus per-polygon confidence statistics) come back.

What v2 does BETTER than v1 (that is where the extra accuracy comes from)
------------------------------------------------------------------------
1. Fused score instead of hard AND/OR thresholds. NDWI, MNDWI and
   AWEI_nsh are blended into one continuous water-evidence score with
   vegetation / built-up evidence subtracted, so one borderline index
   can no longer flip a pixel on its own.
2. Hysteresis (core + region-grow) instead of a single cut-off.
   Confident water is found with strict rules; turbid / shallow /
   weedy margins are recovered only where they touch confident water.
   v1 either misses those margins (strict) or floods soil/shadow
   (loose).
3. Shadow rejection. AWEI_sh (the shadow-suppressing variant) and an
   SRTM slope mask remove terrain/building/cloud-shadow false
   positives, which are the dominant error of a plain NDWI/MNDWI
   rule in hilly or built-up areas. Water is flat; steep pixels are
   never water.
4. A real, fixed confidence surface (logistic of the fused score) so
   each polygon carries mean/median/min/max and P>=0.5/0.75/0.9
   statistics, the same fields the U-Net response carries.
5. Growth requires a real (if weak) MNDWI/AWEI signal, not just a
   permissive score, and an opening (despeckle) pass runs before the
   existing closing pass. Both target the same failure: on turbid,
   braided rivers the old hysteresis alone could crawl into damp
   soil/sandbar fringes and leave 1-pixel speckle, which turned one
   water body into dozens of tiny false rings when vectorised. See
   GROW_MIN_MNDWI / GROW_MIN_AWEI_SCALED / OPENING_RADIUS_PX in
   config/settings.py.

What it deliberately does NOT do: per-patch Otsu. Otsu always splits
the histogram in two, so on land-only patches it labelled ~half the
pixels "water" (visible in the old log as 30-50k water px per
65k-px patch at scores around -0.78).
"""

import json
import logging
import threading
from typing import Optional

import ee

from config import settings as cfg

logger = logging.getLogger("traditional_v2_backend.gee_processor")

_init_lock = threading.Lock()
_initialized = False

REFLECTANCE_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]


# ============================================================
# INITIALISATION  (same env-var contract as Mini-Backend/app.py)
# ============================================================

def initialize_earth_engine() -> None:
    """Idempotent, thread-safe Earth Engine initialization.

    Production / Render:
        GEE_PROJECT_ID + GOOGLE_APPLICATION_CREDENTIALS_JSON

    Local development:
        GEE_PROJECT_ID + the normal Earth Engine CLI authentication
        (`earthengine authenticate`), or the same service-account JSON.

    The service-account JSON is read from the environment so no credential
    file needs to be committed to the repository.
    """
    global _initialized

    if _initialized:
        return

    with _init_lock:
        if _initialized:
            return

        try:
            service_account_json = (cfg.GEE_SERVICE_ACCOUNT_JSON or "").strip()
            project_id = (cfg.GEE_PROJECT_ID or "").strip()

            # Production path: Render supplies the complete service-account
            # JSON through GOOGLE_APPLICATION_CREDENTIALS_JSON.
            if service_account_json:
                if not project_id:
                    raise RuntimeError(
                        "GEE_PROJECT_ID must be set when "
                        "GOOGLE_APPLICATION_CREDENTIALS_JSON is provided."
                    )

                try:
                    info = json.loads(service_account_json)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        "GOOGLE_APPLICATION_CREDENTIALS_JSON is not valid JSON."
                    ) from exc

                required_keys = ("client_email", "private_key")
                missing = [key for key in required_keys if not info.get(key)]
                if missing:
                    raise RuntimeError(
                        "GOOGLE_APPLICATION_CREDENTIALS_JSON is missing: "
                        + ", ".join(missing)
                    )

                credentials = ee.ServiceAccountCredentials(
                    info["client_email"],
                    key_data=json.dumps(info),
                )

                ee.Initialize(
                    credentials=credentials,
                    project=project_id,
                )

            # Local development path: use the credentials configured by the
            # Earth Engine CLI / local Google credentials.
            elif project_id:
                ee.Initialize(project=project_id)

            else:
                raise RuntimeError(
                    "Earth Engine is not configured. Set GEE_PROJECT_ID and, "
                    "for Render/production, GOOGLE_APPLICATION_CREDENTIALS_JSON."
                )

        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                "Google Earth Engine initialization failed. "
                "For Render, verify GEE_PROJECT_ID and "
                "GOOGLE_APPLICATION_CREDENTIALS_JSON. "
                f"Cause: {exc}"
            ) from exc

        _initialized = True
        logger.info(
            "Google Earth Engine initialized (project=%s)",
            project_id or "default",
        )


# ============================================================
# AOI
# ============================================================

def create_aoi(latitude: float, longitude: float, radius_km: float) -> ee.Geometry:
    if not -90 <= latitude <= 90:
        raise ValueError("Latitude must be between -90 and 90.")
    if not -180 <= longitude <= 180:
        raise ValueError("Longitude must be between -180 and 180.")
    if radius_km <= 0:
        raise ValueError("Radius must be greater than 0 km.")
    return ee.Geometry.Point([float(longitude), float(latitude)]).buffer(float(radius_km) * 1000.0)


# ============================================================
# COMPOSITE  (cloud/shadow-masked multi-date median = full AOI coverage)
# ============================================================

def _mask_invalid_scl(image: ee.Image) -> ee.Image:
    scl = image.select("SCL")
    valid = ee.Image.constant(1)
    for scl_class in cfg.SCL_INVALID_CLASSES:
        valid = valid.And(scl.neq(scl_class))
    if not cfg.S2_BILINEAR_RESAMPLE:
        return image.updateMask(valid)
    # Refined mode. Earth Engine resamples with nearest-neighbour by default, so the
    # 20 m bands (B11 / B12 -> MNDWI, AWEI, NDBI) reach a finer grid as 2x2 blocks and
    # the shoreline inherits their staircase. Bilinear interpolation is applied to the
    # reflectance bands only. `valid` was built from the untouched SCL band above, so
    # the categorical class codes are never interpolated (a bilinear 4-vs-9 blend would
    # otherwise turn a cloud edge into a "valid" class 6.5).
    return image.select(REFLECTANCE_BANDS).resample("bilinear").updateMask(valid)


def _collection(aoi, start_date: str, end_date: str, max_cloud: float) -> ee.ImageCollection:
    return (
        ee.ImageCollection(cfg.S2_COLLECTION)
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
    )


def build_composite(aoi, start_date: str, end_date: str, max_cloud_percentage: float) -> dict:
    """Median composite of cloud-masked Sentinel-2 SR reflectance.

    Why a composite and not "the single best scene" (old v2): one
    Sentinel-2 granule almost never covers a whole 25-100 km circle, and
    after SCL masking it has holes. Both showed up as thousands of
    NaN "download failed" patches. The median over all clear
    observations covers the whole AOI and is robust to residual
    clouds/shadows in any single acquisition.

    Falls back (looser cloud filter, then the full calendar year) if
    the requested window has no usable scenes.
    """
    year_start = f"{start_date[:4]}-01-01"
    year_end = f"{end_date[:4]}-12-31"
    attempts = [
        (start_date, end_date, float(max_cloud_percentage), "requested window"),
        (start_date, end_date, max(float(max_cloud_percentage), cfg.FALLBACK_MAX_CLOUD_PERCENTAGE),
         "requested window, relaxed cloud filter"),
        (year_start, year_end, cfg.FALLBACK_MAX_CLOUD_PERCENTAGE, "full calendar year, relaxed cloud filter"),
    ]

    chosen: Optional[tuple] = None
    count = 0
    for s, e, cloud, label in attempts:
        count = int(_collection(aoi, s, e, cloud).size().getInfo() or 0)
        logger.info(f"[composite] {label}: {s}..{e} cloud<={cloud} -> {count} scene(s)")
        if count > 0:
            chosen = (s, e, cloud, label)
            break

    if chosen is None:
        raise RuntimeError(
            "No Sentinel-2 images were found for the requested location, radius and date range "
            "(even after relaxing the cloud filter and widening to the full year)."
        )

    s, e, cloud, label = chosen
    collection = _collection(aoi, s, e, cloud).map(_mask_invalid_scl)
    composite = collection.select(REFLECTANCE_BANDS).median().multiply(0.0001).toFloat()

    return {
        "composite": composite,
        "image_count": count,
        "window": {"start_date": s, "end_date": e, "max_cloud_percentage": cloud, "note": label},
    }


# ============================================================
# CLASSIFICATION
# ============================================================

def _scaled_px(native_px: int) -> int:
    """cfg pixel radii are defined on the native 10 m grid. When the graph is
    evaluated at PROCESS_SCALE_M (5 m) the same physical distance is more pixels.
    Identity when PROCESS_SCALE_M == PIXEL_SIZE_M, so legacy behaviour is unchanged."""
    if native_px <= 0:
        return 0
    return max(1, int(round(native_px * cfg.PIXEL_SIZE_M / cfg.PROCESS_SCALE_M)))


def classify(composite: ee.Image, aoi: ee.Geometry) -> dict:
    """Builds the lazy water mask + confidence images.

    Returns:
        mask        — ee.Image, 1 where water, masked elsewhere (clipped to AOI)
        stats_image — ee.Image bands [prob, p50, p75, p90] for per-polygon means
        prob        — ee.Image band  'prob' (for median / min / max)
    """
    blue = composite.select("B2")
    green = composite.select("B3")
    nir = composite.select("B8")
    swir1 = composite.select("B11")
    swir2 = composite.select("B12")

    ndwi = composite.normalizedDifference(["B3", "B8"]).rename("NDWI")
    mndwi = composite.normalizedDifference(["B3", "B11"]).rename("MNDWI")
    ndvi = composite.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndbi = composite.normalizedDifference(["B11", "B8"]).rename("NDBI")

    # AWEI_nsh = 4(G - SWIR1) - (0.25 NIR + 2.75 SWIR2)   [same formula as the U-Net's channel 15]
    awei_nsh = (
        green.subtract(swir1).multiply(4.0)
        .subtract(nir.multiply(0.25))
        .subtract(swir2.multiply(2.75))
        .rename("AWEI")
    )
    # AWEI_sh = B + 2.5 G - 1.5 (NIR + SWIR1) - 0.25 SWIR2   [shadow-suppressing variant]
    awei_sh = (
        blue.add(green.multiply(2.5))
        .subtract(nir.add(swir1).multiply(1.5))
        .subtract(swir2.multiply(0.25))
        .rename("AWEI_SH")
    )

    awei_scaled = awei_nsh.divide(cfg.AWEI_SCALE).clamp(-1, 1)

    score = (
        ndwi.multiply(cfg.W_NDWI)
        .add(mndwi.multiply(cfg.W_MNDWI))
        .add(awei_scaled.multiply(cfg.W_AWEI))
        .subtract(ndvi.max(0).multiply(cfg.W_NDVI_PENALTY))
        .subtract(ndbi.max(0).multiply(cfg.W_NDBI_PENALTY))
        .rename("score")
    )

    # Water is flat: terrain shadow lives on slopes.
    slope = ee.Terrain.slope(ee.Image(cfg.DEM_COLLECTION)).unmask(0)
    flat = slope.lt(cfg.MAX_SLOPE_DEG)

    core = (
        score.gte(cfg.SCORE_CORE_MIN)
        .And(mndwi.gt(cfg.CORE_MIN_MNDWI))
        .And(ndvi.lt(cfg.CORE_MAX_NDVI))
        .And(ndbi.lt(cfg.CORE_MAX_NDBI))
        .And(awei_sh.gt(cfg.CORE_MIN_AWEI_SH))
        .And(flat)
    )
    core = core.unmask(0)

    grow_allowed = (
        score.gte(cfg.SCORE_GROW_MIN)
        .And(ndvi.lt(cfg.GROW_MAX_NDVI))
        .And(ndbi.lt(cfg.GROW_MAX_NDBI))
        # A borderline SCORE alone doesn't tell you whether a pixel is
        # turbid water or just damp/bare soil — both can sit in the same
        # -0.2..0.0 band. Require at least a weak but real water signal
        # (MNDWI or AWEI) too, so growth still recovers genuine turbid
        # margins but stops crawling into plain sandbar/field pixels
        # that only happen to share a low score. This is what was
        # turning one messy river into dozens of speckled false rings.
        .And(mndwi.gt(cfg.GROW_MIN_MNDWI).Or(awei_scaled.gt(cfg.GROW_MIN_AWEI_SCALED)))
    ).unmask(0)   # no slope rule here: DEM noise at shorelines must not clip the border

    # Mini-Backend (v1) rule set: the base mask, so boundaries follow v1.
    v1_mask = (
        ndwi.gt(cfg.V1_NDWI_MIN).And(mndwi.gt(cfg.V1_MNDWI_MIN))
        .Or(awei_nsh.gt(cfg.V1_AWEI_MIN).And(ndwi.gt(cfg.V1_NDWI_SECONDARY_MIN)))
        .And(ndvi.lt(cfg.V1_MAX_NDVI))
        .And(ndbi.lt(cfg.V1_MAX_NDBI))
    ).unmask(0)

    # Region-grow: repeatedly dilate the accepted region by a few pixels,
    # keeping only pixels the relaxed rules allow. (Approximates
    # morphological reconstruction without an expensive connected-components pass.)
    grown = core.Or(v1_mask)   # v1 water + v2 confident core, then grow into turbid margins
    grow_r, open_r, close_r = _scaled_px(cfg.GROW_RADIUS_PX), _scaled_px(cfg.OPENING_RADIUS_PX), _scaled_px(cfg.CLOSING_RADIUS_PX)
    area_scale = (cfg.PIXEL_SIZE_M / cfg.PROCESS_SCALE_M) ** 2      # pixels per native pixel (1 at 10 m, 4 at 5 m)
    min_px = max(1, int(round(cfg.MIN_CONNECTED_PIXELS * area_scale)))
    for _ in range(cfg.GROW_ITERATIONS):
        grown = (
            grown.focal_max(radius=grow_r, kernelType="circle", units="pixels")
            .And(grow_allowed)
            .Or(core).Or(v1_mask)
        )

    # Opening (erode -> dilate) BEFORE closing: strips isolated 1-px
    # speckle and thin whiskers that a noisy/turbid scene leaves behind
    # (the raster source of the "web of tiny rings" look) without
    # shrinking a real body wider than 2*OPENING_RADIUS_PX. Closing
    # (dilate -> erode) then heals small real gaps in what's left.
    # Order matters — closing first would weld speckle onto the body
    # before opening gets a chance to remove it.
    if open_r > 0:
        grown = (
            grown.focal_min(radius=open_r, kernelType="circle", units="pixels")
            .focal_max(radius=open_r, kernelType="circle", units="pixels")
        )

    # Small closing (fills 1-px speckle inside water), then drop tiny blobs.
    if close_r > 0:
        grown = (
            grown.focal_max(radius=close_r, kernelType="circle", units="pixels")
            .focal_min(radius=close_r, kernelType="circle", units="pixels")
        )

    connected = grown.connectedPixelCount(max(min_px * 2, int(round(64 * area_scale))), False)  # same as v1
    cleaned = grown.updateMask(connected.gte(min_px))
    mask = cleaned.selfMask().clip(aoi).rename("water")

    # Confidence surface: logistic centred on the grow threshold.
    z = score.subtract(cfg.SCORE_GROW_MIN).divide(cfg.CONFIDENCE_SCALE)
    prob = ee.Image.constant(1).divide(ee.Image.constant(1).add(z.multiply(-1).exp())).rename("prob")
    stats_image = ee.Image.cat([
        prob,
        prob.gte(0.50).rename("p50"),
        prob.gte(0.75).rename("p75"),
        prob.gte(0.90).rename("p90"),
    ])

    return {"mask": mask, "stats_image": stats_image, "prob": prob}
