import pandas as pd
import numpy as np
import pytest

from regimejump.risk_free import discount_yield_to_daily_return, risk_free_returns_for_dates


def test_zero_discount_yield_produces_zero_return():
    yields = pd.Series(
        [0.0],
        index=pd.to_datetime(["2020-01-02"]),
    )

    result = discount_yield_to_daily_return(yields)

    assert result.iloc[0] == pytest.approx(0.0)
    assert result.name == "risk_free_return"
    assert result.index.equals(yields.index)


def test_positive_discount_yield_matches_manual_calculation():
    yields = pd.Series([7.92])

    result = discount_yield_to_daily_return(
        yields,
        maturity_days=91,
        trading_days=252,
    )

    discount_rate = 7.92 / 100.0
    price = 1.0 - discount_rate * 91 / 360
    holding_period_growth = 1.0 / price
    trading_periods = 91 * 252 / 365

    expected = holding_period_growth ** (1.0 / trading_periods) - 1.0

    assert result.iloc[0] == pytest.approx(expected)
    assert result.iloc[0] > 0.0


def test_alignment_uses_most_recent_past_yield():
    yields = pd.Series(
        [4.0, 6.0],
        index=pd.to_datetime(["2020-01-02", "2020-01-06"]),
    )

    dates = pd.to_datetime([
        "2020-01-02",
        "2020-01-03",
        "2020-01-06",
    ])

    result = risk_free_returns_for_dates(yields, dates)

    expected_yields = pd.Series(
        [4.0, 4.0, 6.0],
        index=dates,
    )
    expected = discount_yield_to_daily_return(expected_yields)

    pd.testing.assert_series_equal(result, expected)


def test_alignment_does_not_backfill_from_future():
    yields = pd.Series(
        [5.0],
        index=pd.to_datetime(["2020-01-06"]),
    )

    dates = pd.to_datetime([
        "2020-01-03",
        "2020-01-06",
    ])

    result = risk_free_returns_for_dates(yields, dates)

    assert pd.isna(result.iloc[0])
    assert pd.notna(result.iloc[1])


def test_alignment_fills_missing_quote_from_past():
    yields = pd.Series(
        [4.0, np.nan, 6.0],
        index=pd.to_datetime([
            "2020-01-02",
            "2020-01-03",
            "2020-01-06",
        ]),
    )

    dates = pd.to_datetime([
        "2020-01-02",
        "2020-01-03",
        "2020-01-06",
    ])

    result = risk_free_returns_for_dates(yields, dates)

    expected_yields = pd.Series(
        [4.0, 4.0, 6.0],
        index=dates,
    )
    expected = discount_yield_to_daily_return(expected_yields)

    pd.testing.assert_series_equal(result, expected)