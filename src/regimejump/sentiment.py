"""Feature engineering for aggregate news sentiment."""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_SENTIMENT_COLUMNS: tuple[str, ...] = (
    "avgSentimentClass",
    "numberOfEntries",
)

SENTIMENT_FEATURE_NAMES: tuple[str, ...] = (
    "macro_sentiment",
    "sentiment_momentum",
    "abnormal_news_attention",
)

SHORT_HALFLIFE = 5
LONG_HALFLIFE = 20
ATTENTION_HALFLIFE = 60


def compute_sentiment_features(
    sentiment_data: pd.DataFrame,
) -> pd.DataFrame:
    """Construct causal daily features from aggregate news sentiment.

    The output contains the daily sentiment level, short-minus-long EWM
    sentiment momentum, and news volume relative to its trailing EWM level.
    """
    if not isinstance(sentiment_data, pd.DataFrame):
        raise TypeError("sentiment_data must be a pandas DataFrame")

    missing_columns = [
        column for column in REQUIRED_SENTIMENT_COLUMNS if column not in sentiment_data.columns
    ]

    if missing_columns:
        raise ValueError(f"sentiment_data is missing required columns: {missing_columns}")

    if not sentiment_data.index.is_monotonic_increasing:
        raise ValueError("sentiment_data must be sorted by date")

    if sentiment_data.index.has_duplicates:
        raise ValueError("sentiment_data must have unique dates")

    sentiment = sentiment_data["avgSentimentClass"].astype(float)
    article_count = sentiment_data["numberOfEntries"].astype(float)

    if not np.isfinite(sentiment).all():
        raise ValueError("sentiment values must be finite")

    if not np.isfinite(article_count).all() or (article_count < 0).any():
        raise ValueError("numberOfEntries must contain finite non-negative values")

    short_sentiment = sentiment.ewm(
        halflife=SHORT_HALFLIFE,
        min_periods=1,
    ).mean()

    long_sentiment = sentiment.ewm(
        halflife=LONG_HALFLIFE,
        min_periods=1,
    ).mean()

    log_attention = np.log1p(article_count)

    expected_attention = log_attention.ewm(
        halflife=ATTENTION_HALFLIFE,
        min_periods=1,
    ).mean()

    return pd.DataFrame(
        {
            "macro_sentiment": sentiment,
            "sentiment_momentum": (short_sentiment - long_sentiment),
            "abnormal_news_attention": (log_attention - expected_attention),
        },
        index=sentiment_data.index,
    )
