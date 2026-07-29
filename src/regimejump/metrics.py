"""Performance metrics for strategy returns."""

from __future__ import annotations

import pandas as pd
import numpy as np

def cumulative_wealth(
    net_returns: pd.Series,
    initial_wealth: float = 1.0,
) -> pd.Series:
    """Compound simple net returns into a wealth index."""

    if not isinstance(net_returns, pd.Series):
        raise TypeError("net_returns must be a pandas Series")

    if initial_wealth <= 0:
        raise ValueError("initial_wealth must be positive")

    valid_returns = net_returns.dropna().astype(float)

    valid_wealth = (
        initial_wealth
        * (1.0 + valid_returns).cumprod()
    )

    wealth = pd.Series(
        pd.NA,
        index=net_returns.index,
        dtype="Float64",
        name="wealth",
    )

    wealth.loc[valid_wealth.index] = valid_wealth

    return wealth


def drawdown_series(
    wealth: pd.Series,
) -> pd.Series:
    """Calculate percentage drawdown from the running wealth peak."""
    valid_wealth = wealth.dropna().astype(float)

    running_peak = valid_wealth.cummax()
    valid_drawdown = valid_wealth / running_peak - 1.0

    drawdown = pd.Series(
        pd.NA,
        index=wealth.index,
        dtype="Float64",
        name="drawdown",
    )

    drawdown.loc[valid_drawdown.index] = valid_drawdown

    return drawdown


def maximum_drawdown(
    wealth: pd.Series,
) -> float:
    """Return the most negative drawdown."""
    drawdown = drawdown_series(wealth).dropna()

    if drawdown.empty:
        raise ValueError("wealth must contain at least one valid value")

    return float(drawdown.min())


def annualized_return(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Calculate annualized compound return from simple returns."""
    valid_returns = returns.dropna().astype(float)

    if valid_returns.empty:
        raise ValueError("returns must contain at least one valid value")

    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    if (valid_returns < -1.0).any():
        raise ValueError("simple returns cannot be less than -1")

    total_growth = (1.0 + valid_returns).prod()
    n_periods = len(valid_returns)

    return float(
        total_growth ** (periods_per_year / n_periods) - 1.0
    )


def annualized_volatility(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Calculate annualized sample volatility."""
    valid_returns = returns.dropna().astype(float)

    if len(valid_returns) < 2:
        raise ValueError("returns must contain at least two valid values")

    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    return float(
        valid_returns.std(ddof=1)
        * np.sqrt(periods_per_year)
    )


def sharpe_ratio(
    returns: pd.Series,
    risk_free_returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Calculate the annualized Sharpe ratio."""
    if not returns.index.equals(risk_free_returns.index):
        raise ValueError("returns must have identical indexes")

    valid_returns = returns.dropna().astype(float)
    valid_risk_free = risk_free_returns.loc[
        valid_returns.index
    ].astype(float)

    if len(valid_returns) < 2:
        raise ValueError("returns must contain at least two valid values")

    mean_excess_return = (
        valid_returns - valid_risk_free
    ).mean()

    annualized_excess_return = (
        mean_excess_return * periods_per_year
    )

    volatility = annualized_volatility(
        valid_returns,
        periods_per_year=periods_per_year,
    )

    if volatility == 0:
        raise ValueError("Sharpe ratio is undefined for zero volatility")

    return float(annualized_excess_return / volatility)
