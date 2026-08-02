"""Tests for aggregate news-sentiment feature construction."""

import numpy as np
import pandas as pd
import pytest

from regimejump.sentiment import (
    ATTENTION_HALFLIFE,
    LONG_HALFLIFE,
    SENTIMENT_FEATURE_NAMES,
    SHORT_HALFLIFE,
    compute_sentiment_features,
)


@pytest.fixture
def sentiment_data() -> pd.DataFrame:
    index = pd.date_range(
        "2020-01-01",
        periods=30,
        freq="B",
    )

    return pd.DataFrame(
        {
            "avgSentimentClass": np.linspace(
                -0.5,
                0.5,
                len(index),
            ),
            "numberOfEntries": np.arange(
                100,
                100 + len(index),
            ),
        },
        index=index,
    )


def test_sentiment_features_have_expected_structure(
    sentiment_data: pd.DataFrame,
):
    result = compute_sentiment_features(sentiment_data)

    assert list(result.columns) == list(SENTIMENT_FEATURE_NAMES)
    assert result.index.equals(sentiment_data.index)

    pd.testing.assert_series_equal(
        result["macro_sentiment"],
        sentiment_data["avgSentimentClass"].rename("macro_sentiment"),
    )


def test_sentiment_momentum_matches_direct_calculation(
    sentiment_data: pd.DataFrame,
):
    sentiment = sentiment_data["avgSentimentClass"]

    expected = (
        sentiment.ewm(
            halflife=SHORT_HALFLIFE,
            min_periods=1,
        ).mean()
        - sentiment.ewm(
            halflife=LONG_HALFLIFE,
            min_periods=1,
        ).mean()
    )

    result = compute_sentiment_features(sentiment_data)

    pd.testing.assert_series_equal(
        result["sentiment_momentum"],
        expected.rename("sentiment_momentum"),
    )


def test_abnormal_attention_matches_direct_calculation(
    sentiment_data: pd.DataFrame,
):
    log_attention = np.log1p(sentiment_data["numberOfEntries"])

    expected = (
        log_attention
        - log_attention.ewm(
            halflife=ATTENTION_HALFLIFE,
            min_periods=1,
        ).mean()
    )

    result = compute_sentiment_features(sentiment_data)

    pd.testing.assert_series_equal(
        result["abnormal_news_attention"],
        expected.rename("abnormal_news_attention"),
    )


def test_sentiment_features_do_not_use_future_values(
    sentiment_data: pd.DataFrame,
):
    original = compute_sentiment_features(sentiment_data)

    changed = sentiment_data.copy()
    changed.iloc[20:, 0] = 10.0
    changed.iloc[20:, 1] = 1_000_000

    perturbed = compute_sentiment_features(changed)

    pd.testing.assert_frame_equal(
        original.iloc[:20],
        perturbed.iloc[:20],
    )


def test_rejects_missing_required_columns(
    sentiment_data: pd.DataFrame,
):
    invalid = sentiment_data.drop(columns="numberOfEntries")

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        compute_sentiment_features(invalid)


@pytest.mark.parametrize(
    "article_count",
    [-1.0, np.inf],
)
def test_rejects_invalid_article_counts(
    sentiment_data: pd.DataFrame,
    article_count: float,
):
    invalid = sentiment_data.copy()
    invalid["numberOfEntries"] = invalid["numberOfEntries"].astype(float)
    invalid.iloc[
        0,
        invalid.columns.get_loc("numberOfEntries"),
    ] = article_count

    with pytest.raises(
        ValueError,
        match="numberOfEntries",
    ):
        compute_sentiment_features(invalid)


@pytest.mark.parametrize(
    "index",
    [
        pd.to_datetime(
            [
                "2020-01-02",
                "2020-01-01",
                "2020-01-03",
            ]
        ),
        pd.to_datetime(
            [
                "2020-01-01",
                "2020-01-01",
                "2020-01-02",
            ]
        ),
    ],
)
def test_rejects_invalid_date_order(
    index: pd.DatetimeIndex,
):
    data = pd.DataFrame(
        {
            "avgSentimentClass": [0.0, 0.1, -0.1],
            "numberOfEntries": [100, 110, 90],
        },
        index=index,
    )

    with pytest.raises(
        ValueError,
        match="sorted|unique",
    ):
        compute_sentiment_features(data)
