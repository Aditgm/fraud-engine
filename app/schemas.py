from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field, field_validator


class TransactionRequest(BaseModel):
    user_id: str = Field(..., examples=["user_1234"])
    amount: float = Field(..., gt=0)
    timestamp: datetime = Field(
        ..., description="Transaction timestamp - will be coerced to UTC"
    )

    @field_validator("timestamp")
    def enforce_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            # If no timezone provided, assume UTC
            return v.replace(tzinfo=timezone.utc)
        # Convert any provided timezone to UTC
        return v.astimezone(timezone.utc)


class PredictionResponse(BaseModel):
    user_id: str
    score: float
    threshold: float
    is_fraud: bool
    model_version: str
    features: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    redis_connected: bool
    model_source: str
