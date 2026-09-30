"""
Traditional Model v2 — standalone FastAPI backend (port 8002).

Own process, own port; shares no code or process with Unet_backend.
Same request/response contract as Unet_backend (POST /analysis, plus
/water-bodies, /statistics, /ranking, /downloads/...) with one extra
`engine` field.

    cd Traditional-V2-Backend
    pip install -r requirements.txt
    earthengine authenticate          # once per machine (or set service-account env)
    uvicorn app:app --host 0.0.0.0 --port 8002

Set GEE_PROJECT_ID in .env (already provided).
"""

import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from routes.analysis import router  # noqa: E402

print("[app] Traditional v2 (Earth Engine index-ensemble) backend starting — no model to load.", flush=True)

app = FastAPI(
    title="Traditional Model v2 Backend",
    description=(
        "Non-neural water-body detection. Fused NDWI/MNDWI/AWEI score with "
        "hysteresis, shadow/slope rejection, computed inside Earth Engine on a "
        "cloud-masked composite. Request/response schema matches Unet_backend "
        "(plus an `engine` field)."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "service": "Traditional Model v2 Backend",
        "engine": "traditional_v2",
        "endpoints": {"analysis": "/analysis"},
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok", "engine": "traditional_v2"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8002")))
