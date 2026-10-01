"""Generate a synthetic financial-transactions dataset -> data/raw_transactions.csv"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PATH = Path("data/raw_transactions.csv")


def generate(n_samples: int = 20000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    avg_7d = rng.lognormal(mean=4.0, sigma=0.6, size=n_samples)            # typical spend
    amount = avg_7d * rng.lognormal(mean=0.0, sigma=0.8, size=n_samples)   # relative to typical spend
    hour = rng.integers(0, 24, size=n_samples)
    num_1h = rng.poisson(1.5, size=n_samples)
    risk = rng.beta(2, 5, size=n_samples)

    # Simple fraud rules -> probability: large vs. usual amount, night time,
    # bursts of transactions and risky locations all raise the odds.
    score = (
        -5.5
        + 2.5 * (amount / avg_7d > 3)
        + 2.0 * (amount > 400)
        + 2.0 * ((hour <= 5) | (hour >= 23))
        + 0.8 * np.clip(num_1h - 3, 0, None)
        + 5.0 * risk
    )
    prob = 1 / (1 + np.exp(-2.0 * score))  # sharpen: labels follow the rules more closely
    is_fraud = (rng.random(n_samples) < prob).astype(int)

    return pd.DataFrame(
        {
            "transaction_amount": amount.round(2),
            "time_of_day_hour": hour,
            "num_transactions_1h": num_1h,
            "avg_transaction_7d": avg_7d.round(2),
            "location_risk_score": risk.round(4),
            "is_fraud": is_fraud,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()

    df = generate(args.n_samples, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output} (fraud rate: {df['is_fraud'].mean():.2%})")


if __name__ == "__main__":
    main()
