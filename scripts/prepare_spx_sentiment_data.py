"""Prepare the common SPX and news-sentiment feature dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from regimejump.sentiment import (
    SENTIMENT_FEATURE_NAMES,
    compute_sentiment_features,
)

PAPER_DATA_PATH = Path("data/spx_excess_feature_states_lambda50.csv")
SENTIMENT_DATA_PATH = Path("data/Eq_macro_ns.csv")
OUTPUT_PATH = Path("data/spx_sentiment_features.csv")

TRAINING_LENGTH = 3000

PAPER_FEATURE_NAMES = (
    "downside_dev_10",
    "sortino_20",
    "sortino_60",
)


def main() -> None:
    paper_data = pd.read_csv(
        PAPER_DATA_PATH,
        parse_dates=["date"],
        index_col="date",
    )

    sentiment_data = pd.read_csv(
        SENTIMENT_DATA_PATH,
        sep="\t",
        parse_dates=["date"],
        index_col="date",
    )

    sentiment_features = compute_sentiment_features(sentiment_data)

    # Retain the equity log return and established paper features.
    combined = paper_data[
        [
            "return",
            *PAPER_FEATURE_NAMES,
        ]
    ].join(
        sentiment_features,
        how="inner",
    )

    combined = combined.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if len(combined) < TRAINING_LENGTH:
        raise RuntimeError("the common sample is shorter than the training window")

    first_signal_date = combined.index[TRAINING_LENGTH - 1]

    test_observations = len(combined) - TRAINING_LENGTH + 1
    approximate_test_years = test_observations / 252.0

    print("Input coverage:")
    print(
        "Paper features:",
        paper_data.index[0].date(),
        "to",
        paper_data.index[-1].date(),
        f"({len(paper_data)} observations)",
    )
    print(
        "Sentiment:",
        sentiment_data.index[0].date(),
        "to",
        sentiment_data.index[-1].date(),
        f"({len(sentiment_data)} observations)",
    )

    print("\nCommon sample:")
    print(
        combined.index[0].date(),
        "to",
        combined.index[-1].date(),
    )
    print(f"Observations: {len(combined)}")
    print(f"First possible signal with {TRAINING_LENGTH} observations: {first_signal_date.date()}")
    print(
        "Remaining test observations:",
        test_observations,
        f"(approximately {approximate_test_years:.1f} years)",
    )

    print("\nCombined feature columns:")
    print(
        [
            *PAPER_FEATURE_NAMES,
            *SENTIMENT_FEATURE_NAMES,
        ]
    )

    print("\nSentiment feature correlations:")
    print(combined[list(SENTIMENT_FEATURE_NAMES)].corr().round(3))

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(OUTPUT_PATH)

    print(f"\nSaved combined data to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
