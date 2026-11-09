from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any

from fastapi import Body, FastAPI

from src.inference import DEFAULT_MODEL_PATH, load_model, predict_records
from src.schemas import *

MAX_BATCH_SIZE = 100
MODEL_PATH = Path(os.environ.get("API_MODEL_PATH", DEFAULT_MODEL_PATH))
model = load_model(MODEL_PATH)


app = FastAPI(
    title="API Sentinel ML",
    description="Classify REST API requests by attack type",
)


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health() -> HealthResponse:
    """Liveness probe."""
    return HealthResponse(status="ok")


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["predictions"],
)
async def predict(record: dict[str, Any]) -> dict[str, Any]:
    """Classify one nested REST request record."""
    results = predict_records([record], model)
    return results[0]


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["predictions"],
)
async def predict_batch(
    records: Annotated[
        list[dict[str, Any]],
        Body(min_length=1, max_length=MAX_BATCH_SIZE),
    ],
) -> BatchPredictionResponse:
    """Classify 1 to 100 nested REST request records in one call."""
    results = predict_records(records, model)
    return BatchPredictionResponse(predictions=results)
