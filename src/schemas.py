from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class RequestData(BaseModel):
    model_config = ConfigDict(extra="allow")

    method: str | None = Field(default=None, description="HTTP method.")
    url: str | None = Field(default=None, description="Request URL or path.")
    body: str | None = Field(default=None, description="Raw request body.")
    headers: dict[str, Any] = Field(
        default_factory=dict,
        description="HTTP headers. Header names are case-sensitive in the payload.",
    )


class ResponseData(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: str | None = None
    status_code: int | str | None = None
    body: str | None = None
    headers: dict[str, Any] = Field(default_factory=dict)


class PredictRequest(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "examples": [
                {
                    "request": {
                        "method": "GET",
                        "url": "/api/resource",
                        "headers": {"User-Agent": "example-client"},
                        "body": "",
                    },
                    "response": {
                        "status_code": 200,
                        "headers": {"Content-Type": "application/json"},
                        "body": "",
                    },
                }
            ]
        },
    )

    request: RequestData = Field(description="Incoming REST request.")
    response: ResponseData | None = Field(
        default=None,
        description="Optional response metadata, if available.",
    )


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: Literal["ok", "ready"]
