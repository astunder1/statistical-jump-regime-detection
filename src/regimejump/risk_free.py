"""Treasury-bill yield conversion and causal date alignment."""

from __future__ import annotations

import pandas as pd


def discount_yield_to_daily_return(
    annual_discount_yield_pct: pd.Series,
    maturity_days: int = 91,
    trading_days: int = 252,
) -> pd.Series:
    """Convert annual Treasury-bill discount yields to daily returns.

    Yields are quoted as annual percentages on a 360-day discount basis.
    The implied holding-period return is converted to an equivalent
    return per trading day.
    """

    if maturity_days <= 0:
        raise ValueError("maturity_days must be positive")

    if trading_days <= 0:
        raise ValueError("trading_days must be positive")

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
    """Align risk-free returns to trading dates without look-ahead.

    Each requested date receives the most recently available Treasury
    yield. Dates before the first available quote remain missing.
    """

    annual_discount_yield_pct = annual_discount_yield_pct.sort_index().ffill()

    aligned_yield = annual_discount_yield_pct.reindex(
        dates,
        method="ffill",
    )

    return discount_yield_to_daily_return(aligned_yield)
