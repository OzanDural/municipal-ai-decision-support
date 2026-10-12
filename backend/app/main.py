import time
from contextlib import asynccontextmanager

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
)
from fastapi.middleware.cors import (
    CORSMiddleware,
)
from sqlalchemy import text

from backend.app.api.analytics import (
    router as analytics_router,
)
from backend.app.api.auth import (
    router as auth_router,
)
from backend.app.api.complaints import (
    router as complaints_router,
)
from backend.app.api.locations import (
    router as locations_router,
)
from backend.app.api.manager_assistant import (
    router as manager_assistant_router,
)
from backend.app.api.service_areas import (
    router as service_areas_router,
)
from backend.app.core.config import settings
from backend.app.db.database import engine
from backend.app.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
)
from backend.app.services.classifier import (
    BerturkClassifier,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    classifier = BerturkClassifier()

    load_start = time.perf_counter()

    classifier.load()

    load_seconds = (
        time.perf_counter()
        - load_start
    )

    app.state.classifier = classifier
    app.state.model_load_seconds = load_seconds

    yield

    if classifier.device.type == "cuda":
        import torch

        torch.cuda.empty_cache()


app = FastAPI(
    title="Belediye AI Karar Destek API",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(
    complaints_router
)
app.include_router(
    auth_router
)
app.include_router(
    analytics_router
)
app.include_router(
    manager_assistant_router
)
app.include_router(
    service_areas_router
)
app.include_router(
    locations_router
)


@app.get("/health")
def health(request: Request):
    try:
        with engine.connect() as connection:
            database_user = connection.execute(
                text("SELECT current_user")
            ).scalar_one()

            database_name = connection.execute(
                text("SELECT current_database()")
            ).scalar_one()

            postgis_version = connection.execute(
                text("SELECT PostGIS_Version()")
            ).scalar_one()

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Database health check failed.",
        ) from exc

    classifier = request.app.state.classifier

    return {
        "status": "ok",

        "database": {
            "connected": True,
            "user": database_user,
            "name": database_name,
        },

        "postgis": {
            "available": True,
            "version": postgis_version,
        },

        "model": {
            "loaded": classifier.loaded,
            "device": str(
                classifier.device
            ),
            "version": settings.model_version,
            "load_seconds":
                request.app.state.model_load_seconds,
        },
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    payload: PredictionRequest,
    request: Request,
):
    classifier = (
        request.app.state.classifier
    )

    try:
        result = classifier.predict(
            payload.complaint_text
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    return PredictionResponse(
        predicted_category=(
            result["predicted_category"]
        ),

        secondary_category=(
            result["secondary_category"]
        ),

        model_score=(
            result["model_score"]
        ),

        second_model_score=(
            result["second_model_score"]
        ),

        score_margin=(
            result["score_margin"]
        ),

        review_required=(
            result["review_required"]
        ),

        review_reason=(
            result["review_reason"]
        ),

        model_version=(
            result["model_version"]
        ),
    )


app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],

    allow_credentials=True,

    allow_methods=[
        "*",
    ],

    allow_headers=[
        "*",
    ],
)
