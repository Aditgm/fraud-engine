from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic transaction data for fraud model training"
    )
    parser.add_argument(
        "--output",
        default="artifacts/synthetic_transactions.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--rows", type=int, default=250_000, help="Number of transactions to generate"
    )
    parser.add_argument(
        "--users", type=int, default=4_000, help="Number of synthetic users"
    )
    parser.add_argument("--days", type=int, default=365, help="History span in days")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def make_dataset(rows: int, users: int, days: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Simulate a one-year timeline and user-specific behavior profiles.
    start = np.datetime64("2025-01-01T00:00:00")
    seconds_span = int(days * 24 * 60 * 60)

    user_ids = rng.integers(1, users + 1, size=rows)
    user_base = rng.lognormal(mean=0.1, sigma=0.35, size=users + 1)
    user_risk = rng.uniform(0.0, 1.0, size=users + 1)

    timestamps = start + rng.integers(0, seconds_span, size=rows).astype(
        "timedelta64[s]"
    )
    hours = (timestamps.astype("datetime64[h]").astype(int) % 24).astype(float)

    normal_amount = rng.lognormal(mean=3.1, sigma=0.8, size=rows)
    amount = normal_amount * user_base[user_ids]

    # Inject behavior spikes that resemble fraud bursts.
    burst_mask = rng.random(rows) < 0.015
    amount[burst_mask] *= rng.uniform(2.0, 5.0, size=int(burst_mask.sum()))

    # Fraud probability: logit model to keep prevalence realistic while preserving signals.
    amount_score = np.clip(np.log1p(amount) / np.log1p(4000.0), 0.0, 1.2)
    night_flag = ((hours < 5) | (hours > 22)).astype(float)
    burst_flag = burst_mask.astype(float)

    logit = (
        -4.2
        + 2.2 * (amount_score - 0.4)
        + 1.3 * night_flag
        + 1.1 * user_risk[user_ids]
        + 1.8 * burst_flag
    )
    fraud_prob = 1.0 / (1.0 + np.exp(-logit))
    fraud_prob = np.clip(fraud_prob, 0.001, 0.8)
    labels = (rng.random(rows) < fraud_prob).astype(int)

    df = pd.DataFrame(
        {
            "user_id": user_ids.astype(str),
            "timestamp": pd.to_datetime(timestamps).astype(str),
            "amount": amount.round(2),
            "is_fraud": labels,
        }
    )

    return df.sort_values("timestamp").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    df = make_dataset(rows=args.rows, users=args.users, days=args.days, seed=args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    fraud_rate = float(df["is_fraud"].mean())
    print(f"Saved synthetic dataset to {output_path}")
    print(f"Rows: {len(df):,}")
    print(f"Users: {df['user_id'].nunique():,}")
    print(f"Fraud rate: {fraud_rate:.4%}")


if __name__ == "__main__":
    main()
