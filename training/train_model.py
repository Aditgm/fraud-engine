from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, precision_recall_curve
from xgboost import XGBClassifier

RATIO_CAP = 20.0

FEATURE_COLUMNS = [
    "amount",
    "txn_hour",
    "txn_count_last_n",
    "avg_amount_last_n",
    "std_amount_last_n",
    "min_amount_last_n",
    "max_amount_last_n",
    "amount_to_avg_ratio",
]


@dataclass
class TrainConfig:
    data_path: str
    model_output: str
    window_size: int
    test_size: float
    synthetic_users: int


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Train streaming fraud model with rolling features"
    )
    parser.add_argument("--data-path", required=True, help="Path to raw CSV data")
    parser.add_argument(
        "--model-output",
        default="artifacts/fraud_model.pkl",
        help="Output path for the serialized model artifact",
    )
    parser.add_argument(
        "--window-size", type=int, default=5, help="History window size"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Temporal validation ratio"
    )
    parser.add_argument(
        "--synthetic-users",
        type=int,
        default=5000,
        help="Number of synthetic users if dataset has no user_id",
    )
    args = parser.parse_args()
    return TrainConfig(
        data_path=args.data_path,
        model_output=args.model_output,
        window_size=args.window_size,
        test_size=args.test_size,
        synthetic_users=args.synthetic_users,
    )


def _infer_timestamp(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series, unit="s", origin="unix", errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def standardize_columns(df: pd.DataFrame, synthetic_users: int) -> pd.DataFrame:
    rename_map = {}

    for col in df.columns:
        lower = col.lower()
        if lower in {"class", "is_fraud", "fraud", "label", "target"}:
            rename_map[col] = "is_fraud"
        elif lower in {"amount", "amt", "transaction_amount", "transactionamt"}:
            rename_map[col] = "amount"
        elif lower in {"user_id", "userid", "customer_id", "card_id", "account_id"}:
            rename_map[col] = "user_id"
        elif lower in {"timestamp", "time", "datetime", "event_time", "transactiondt"}:
            rename_map[col] = "timestamp"

    df = df.rename(columns=rename_map)

    if "amount" not in df.columns:
        raise ValueError("Could not infer amount column")
    if "is_fraud" not in df.columns:
        raise ValueError("Could not infer label column (expected Class/is_fraud)")

    if "user_id" not in df.columns:
        df["user_id"] = (np.arange(len(df)) % max(1, synthetic_users)).astype(str)
    else:
        df["user_id"] = df["user_id"].astype(str)

    if "timestamp" not in df.columns:
        df["timestamp"] = pd.date_range("2024-01-01", periods=len(df), freq="s")
    else:
        df["timestamp"] = _infer_timestamp(df["timestamp"])
        if df["timestamp"].isna().all():
            df["timestamp"] = pd.date_range("2024-01-01", periods=len(df), freq="s")
        else:
            # Fill any malformed timestamps deterministically to preserve ordering.
            df["timestamp"] = df["timestamp"].ffill().bfill()

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    df["is_fraud"] = (
        pd.to_numeric(df["is_fraud"], errors="coerce").fillna(0).astype(int)
    )

    return df[["user_id", "timestamp", "amount", "is_fraud"]]


def build_rolling_features(df: pd.DataFrame, window_size: int) -> pd.DataFrame:
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    grouped = df.groupby("user_id", sort=False)["amount"]
    prev_amount = grouped.shift(1)
    rolling = prev_amount.groupby(df["user_id"]).rolling(
        window=window_size, min_periods=1
    )

    df["txn_count_last_n"] = rolling.count().reset_index(level=0, drop=True).fillna(0.0)
    df["avg_amount_last_n"] = rolling.mean().reset_index(level=0, drop=True).fillna(0.0)
    df["std_amount_last_n"] = rolling.std().reset_index(level=0, drop=True).fillna(0.0)
    df["min_amount_last_n"] = rolling.min().reset_index(level=0, drop=True).fillna(0.0)
    df["max_amount_last_n"] = rolling.max().reset_index(level=0, drop=True).fillna(0.0)

    df["txn_hour"] = df["timestamp"].dt.hour.astype(float)
    raw_ratio = df["amount"] / (df["avg_amount_last_n"] + 1e-6)
    df["amount_to_avg_ratio"] = np.clip(raw_ratio, 0.0, RATIO_CAP)

    return df


def temporal_split(
    df: pd.DataFrame, test_size: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = max(1, int(len(df) * (1 - test_size)))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        preds = (y_score >= 0.5).astype(int)
        return 0.5, float(f1_score(y_true, preds, zero_division=0))

    f1_values = (2 * precision[:-1] * recall[:-1]) / (
        precision[:-1] + recall[:-1] + 1e-8
    )
    best_idx = int(np.nanargmax(f1_values))
    return float(thresholds[best_idx]), float(f1_values[best_idx])


def train(config: TrainConfig) -> None:
    data_path = Path(config.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {data_path}")

    df_raw = pd.read_csv(data_path)
    df_std = standardize_columns(df_raw, synthetic_users=config.synthetic_users)
    df_feat = build_rolling_features(df_std, window_size=config.window_size)

    train_df, val_df = temporal_split(df_feat, test_size=config.test_size)

    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["is_fraud"]
    x_val = val_df[FEATURE_COLUMNS]
    y_val = val_df["is_fraud"]

    positives = int(y_train.sum())
    negatives = int((y_train == 0).sum())
    scale_pos_weight = float(max(1.0, negatives / max(1, positives)))

    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        reg_lambda=2.0,
        min_child_weight=5,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    val_scores = model.predict_proba(x_val)[:, 1]
    threshold, best_f1 = best_f1_threshold(y_val.to_numpy(), val_scores)
    val_preds = (val_scores >= threshold).astype(int)

    metrics = {
        "f1": float(f1_score(y_val, val_preds, zero_division=0)),
        "best_threshold": threshold,
        "best_f1_from_pr_curve": best_f1,
        "validation_rows": int(len(val_df)),
        "train_rows": int(len(train_df)),
        "scale_pos_weight": scale_pos_weight,
    }

    report = classification_report(y_val, val_preds, zero_division=0)
    print("Validation report:\n", report)
    print("Metrics:", json.dumps(metrics, indent=2))

    output_path = Path(config.model_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "threshold": threshold,
        "window_size": config.window_size,
        "metrics": metrics,
    }
    joblib.dump(artifact, output_path)

    metrics_path = output_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved model artifact to {output_path}")
    print(f"Saved metrics to {metrics_path}")


def main() -> None:
    config = parse_args()
    train(config)


if __name__ == "__main__":
    main()
