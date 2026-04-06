from __future__ import annotations

import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import boto3
import joblib
import numpy as np
import pandas as pd


class FallbackModel:
    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        base = np.clip((data["amount"] / 2000.0) + (data["amount_to_avg_ratio"] / 10.0), 0.0, 1.0)
        probs = base.to_numpy(dtype=float)
        return np.column_stack([1.0 - probs, probs])


def load_model_from_s3(s3_uri: str) -> str:
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ValueError("MODEL_S3_URI must be a valid s3://bucket/key path")

    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    temp_dir = tempfile.mkdtemp(prefix="fraud_model_")
    local_path = os.path.join(temp_dir, "fraud_model.pkl")
    boto3.client("s3").download_file(bucket, key, local_path)
    return local_path


def load_model_artifact(model_path: str, model_s3_uri: str | None) -> tuple[dict, str]:
    source = "local"
    resolved_path = model_path

    if model_s3_uri:
        resolved_path = load_model_from_s3(model_s3_uri)
        source = "s3"

    if not os.path.exists(resolved_path):
        fallback_artifact = {
            "model": FallbackModel(),
            "feature_columns": [
                "amount",
                "txn_hour",
                "txn_count_last_n",
                "avg_amount_last_n",
                "std_amount_last_n",
                "min_amount_last_n",
                "max_amount_last_n",
                "amount_to_avg_ratio",
            ],
            "threshold": 0.8,
            "window_size": int(os.getenv("REDIS_WINDOW_SIZE", "5")),
            "metrics": {},
            "model_version": "fallback-v1",
        }
        return fallback_artifact, "fallback"

    artifact = joblib.load(resolved_path)
    artifact.setdefault("model_version", os.getenv("MODEL_VERSION", "xgboost-v1"))
    return artifact, source


def compute_history_features(amounts: list[float], current_amount: float, txn_hour: int) -> dict[str, float]:
    if not amounts:
        avg = 0.0
        std = 0.0
        min_val = 0.0
        max_val = 0.0
        count = 0.0
    else:
        arr = np.array(amounts, dtype=float)
        avg = float(arr.mean())
        std = float(arr.std())
        min_val = float(arr.min())
        max_val = float(arr.max())
        count = float(arr.size)

    return {
        "amount": float(current_amount),
        "txn_hour": float(txn_hour),
        "txn_count_last_n": count,
        "avg_amount_last_n": avg,
        "std_amount_last_n": std,
        "min_amount_last_n": min_val,
        "max_amount_last_n": max_val,
        "amount_to_avg_ratio": float(current_amount / (avg + 1e-6)),
    }


def predict(
    features: dict[str, float],
    model_artifact: dict,
) -> tuple[float, bool, float]:
    """
    Make a fraud prediction using the loaded model artifact
    
    Returns:
        tuple of (score, is_fraud, threshold)
    """
    model = model_artifact["model"]
    threshold = float(os.getenv("FRAUD_THRESHOLD", str(model_artifact.get("threshold", 0.5))))
    feature_columns = model_artifact["feature_columns"]

    model_input = pd.DataFrame([features])[feature_columns]
    score = float(model.predict_proba(model_input)[0, 1])
    is_fraud = score >= threshold
    
    return score, is_fraud, threshold