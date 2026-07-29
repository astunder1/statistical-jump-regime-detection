"""Utilities for preparing risk-free returns."""

from __future__ import annotations

import pandas as pd


def discount_yield_to_daily_return(
    annual_discount_yield_pct: pd.Series,
    maturity_days: int = 91,
    trading_days: int = 252,
) -> pd.Series:
    """Convert a Treasury-bill discount yield into a daily return."""

    discount_rate = annual_discount_yield_pct / 100.0

    price = 1.0 - discount_rate * maturity_days / 360.0
    holding_period_growth = 1.0 / price

    trading_periods = maturity_days * trading_days / 365.0
    daily_return = holding_period_growth ** (1.0 / trading_periods) - 1.0

    return daily_return.rename("risk_free_return")


def risk_free_returns_for_dates(
    annual_discount_yield_pct: pd.Series,
    dates: pd.DatetimeIndex,
) -> pd.Series:
    """Create daily risk-free returns for the requested trading dates."""

    annual_discount_yield_pct = (
        annual_discount_yield_pct
        .sort_index()
        .ffill()
    )

    aligned_yield = annual_discount_yield_pct.reindex(
        dates,
        method="ffill",
    )

    return discount_yield_to_daily_return(aligned_yield)