from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.responses import ORJSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from app.redis_client import InMemoryFeatureStore, RedisFeatureStore
from app.schemas import HealthResponse, PredictionResponse, TransactionRequest
from app.services.ml import compute_history_features, load_model_artifact, predict

REQUEST_COUNT = Counter("prediction_requests_total", "Total prediction requests")
FRAUD_DETECTED = Counter("fraud_detected_total", "Total predicted fraud events")
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Latency for prediction endpoint",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    window_size = int(os.getenv("REDIS_WINDOW_SIZE", "5"))

    redis_store: RedisFeatureStore | InMemoryFeatureStore = RedisFeatureStore(
        url=redis_url, window_size=window_size
    )
    try:
        await redis_store.connect()
        await redis_store.ping()
    except Exception:
        redis_store = InMemoryFeatureStore(window_size=window_size)
        await redis_store.connect()

    app.state.redis_store = redis_store

    model_path = os.getenv("MODEL_PATH", "artifacts/fraud_model.pkl")
    model_s3_uri = os.getenv("MODEL_S3_URI")
    artifact, source = load_model_artifact(
        model_path=model_path, model_s3_uri=model_s3_uri
    )

    app.state.model_artifact = artifact
    app.state.model_source = source

    try:
        yield
    finally:
        await app.state.redis_store.close()


app = FastAPI(
    title="Streaming Fraud Detection Engine",
    version="1.0.0",
    lifespan=lifespan,
    default_response_class=ORJSONResponse,
)
APP_DIR = Path(__file__).resolve().parent


def verify_api_key(
    x_api_key: str | None = Header(default=None, alias="x-api-key")
) -> None:
    expected_key = os.getenv("API_KEY", "local-dev-key")
    if not x_api_key or x_api_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key",
        )


@app.get("/", include_in_schema=False)
async def home() -> FileResponse:
    return FileResponse(APP_DIR / "index.html")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    redis_ok = False
    try:
        redis_ok = await app.state.redis_store.ping()
    except Exception:
        redis_ok = False

    model_loaded = bool(
        app.state.model_artifact and "model" in app.state.model_artifact
    )

    return HealthResponse(
        status="ok" if model_loaded and redis_ok else "degraded",
        model_loaded=model_loaded,
        redis_connected=redis_ok,
        model_source=str(app.state.model_source),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    dependencies=[Depends(verify_api_key)],
)
async def predict_endpoint(request: TransactionRequest) -> PredictionResponse:
    REQUEST_COUNT.inc()
    start = perf_counter()

    recent = await app.state.redis_store.get_recent_transactions(request.user_id)
    amounts = [float(item.get("amount", 0.0)) for item in recent]

    # Ensure timestamp is UTC - schema validation already ensures this
    utc_timestamp = request.timestamp
    features = compute_history_features(
        amounts=amounts,
        current_amount=request.amount,
        txn_hour=utc_timestamp.hour,
    )

    artifact = app.state.model_artifact
    score, is_fraud, threshold = predict(features, artifact)

    if is_fraud:
        FRAUD_DETECTED.inc()

    await app.state.redis_store.append_transaction(
        user_id=request.user_id,
        amount=request.amount,
        timestamp=request.timestamp.isoformat(),
    )

    PREDICTION_LATENCY.observe(perf_counter() - start)

    return PredictionResponse(
        user_id=request.user_id,
        score=score,
        threshold=threshold,
        is_fraud=is_fraud,
        model_version=str(artifact.get("model_version", "unknown")),
        features=features,
    )


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    return PlainTextResponse(
        generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST
    )
