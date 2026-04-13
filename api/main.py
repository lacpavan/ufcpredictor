from __future__ import annotations

from fastapi import FastAPI, HTTPException

from .schemas import HealthResponse, PredictionRequest, PredictionResponse
from .service import PredictionService


app = FastAPI(
    title="UFC Winner Predictor API",
    version="1.0.0",
    description="Production-style API for UFC winner prediction using the trained ML model.",
)

service: PredictionService | None = None


@app.on_event("startup")
def startup_event() -> None:
    global service
    service = PredictionService()


@app.get("/", tags=["meta"])
def root() -> dict:
    return {
        "message": "UFC Winner Predictor API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    if service is None:
        raise HTTPException(status_code=503, detail="Model service not initialized")
    return HealthResponse(status="ok", model_loaded=True, feature_count=len(service.feature_columns))


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(request: PredictionRequest) -> PredictionResponse:
    if service is None:
        raise HTTPException(status_code=503, detail="Model service not initialized")

    result = service.predict(
        red_corner=request.red_corner.model_dump(),
        blue_corner=request.blue_corner.model_dump(),
        include_explanations=request.include_explanations,
    )
    return PredictionResponse(**result)
