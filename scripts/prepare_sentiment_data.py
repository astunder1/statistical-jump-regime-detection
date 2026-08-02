"""Prepare aligned market and sentiment features."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from regimejump.features import (
    PAPER_FEATURE_NAMES,
    compute_paper_features,
)
from regimejump.risk_free import risk_free_returns_for_dates
from regimejump.sentiment import (
    SENTIMENT_FEATURE_NAMES,
    compute_sentiment_features,
)

SPX_PATH = Path("data/spx_tr.parquet")
SENTIMENT_PATH = Path("data/Eq_macro_ns.csv")
TREASURY_PATH = Path("data/us_tbill_3m.parquet")
NASDAQ_PATH = Path("data/nasdaq100_tr.parquet")

TRAINING_LENGTH = 3000


def parse_args() -> argparse.Namespace:
    """Parse the selected market."""
    parser = argparse.ArgumentParser(
        description=("Prepare aligned price and sentiment features for a selected market.")
    )

    parser.add_argument(
        "--market",
        choices=["spx", "nasdaq100"],
        required=True,
    )

    return parser.parse_args()


def load_spx_returns() -> pd.Series:
    """Load simple S&P 500 total returns."""
    prices = pd.read_parquet(SPX_PATH)["Adj Close"]

    return prices.astype(float).pct_change().dropna().rename("equity_return")


def load_nasdaq_returns() -> pd.Series:
    """Load simple Nasdaq-100 total returns."""
    prices = pd.read_parquet(NASDAQ_PATH)["total_return_index"]

    return prices.astype(float).pct_change().dropna().rename("equity_return")


def load_market_returns(
    market: str,
) -> pd.Series:
    """Load total returns for a supported market."""
    if market == "spx":
        return load_spx_returns()

    if market == "nasdaq100":
        return load_nasdaq_returns()

    raise ValueError(f"unsupported market: {market}")


def prepare_dataset(
    market: str,
) -> pd.DataFrame:
    """Construct aligned price and sentiment features."""
    equity_returns = load_market_returns(market)

    treasury = pd.read_parquet(TREASURY_PATH)

    risk_free_returns = risk_free_returns_for_dates(
        treasury["annual_discount_yield_pct"],
        equity_returns.index,
    )

    excess_returns = (equity_returns - risk_free_returns).rename("excess_return")

    # Calculate price features before restricting the sample to the
    # sentiment period, preserving the available EWM history.
    price_features = compute_paper_features(excess_returns)

    sentiment_data = pd.read_csv(
        SENTIMENT_PATH,
        sep="\t",
        parse_dates=["date"],
        index_col="date",
    )

    sentiment_features = compute_sentiment_features(sentiment_data)

    prepared = pd.concat(
        [
            equity_returns,
            risk_free_returns,
            excess_returns,
            price_features,
        ],
        axis=1,
    ).join(
        sentiment_features,
        how="inner",
    )

    prepared = prepared.replace([np.inf, -np.inf], np.nan).dropna().sort_index()

    if len(prepared) < TRAINING_LENGTH:
        raise RuntimeError(f"the common sample is shorter than {TRAINING_LENGTH} observations")

    return prepared


def main() -> None:
    args = parse_args()

    prepared = prepare_dataset(args.market)

    output_path = Path(f"data/{args.market}_sentiment_features.csv")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    prepared.to_csv(output_path)

    first_signal_date = prepared.index[TRAINING_LENGTH - 1]

    print(f"Market: {args.market}")
    print(
        "Common period:",
        prepared.index[0].date(),
        "to",
        prepared.index[-1].date(),
    )
    print(f"Observations: {len(prepared)}")
    print(
        "First possible signal:",
        first_signal_date.date(),
    )

    print("\nPrepared columns:")
    print(
        [
            "equity_return",
            "risk_free_return",
            "excess_return",
            *PAPER_FEATURE_NAMES,
            *SENTIMENT_FEATURE_NAMES,
        ]
    )

    print(f"\nSaved data to {output_path}")


if __name__ == "__main__":
    main()
