"""Backtesting utilities for the regime-switching strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd


def delayed_equity_weight(states: pd.Series, delay: int = 2) -> pd.Series:
    """Convert regime states into delayed equity allocations."""

    if not isinstance(states, pd.Series):
        raise TypeError("states must be a Series")

    if not isinstance(delay, int):
        raise TypeError("delay must be an int")

    if delay < 0:
        raise ValueError("delay must be >= 0")

    valid_states = states.dropna()

    if not valid_states.isin([0, 1]).all():
        raise ValueError("States must be either 0 or 1")

    signal_weight = states.map(
        {
            0: 1.0,
            1: 0.0,
        }
    ).astype("Float64")

    equity_weight = signal_weight.shift(delay)
    equity_weight.name = "equity_weight"

    return equity_weight


def calculate_turnover(
    equity_weight: pd.Series,
) -> pd.Series:
    """Calculate one-way portfolio turnover from equity weights."""

    valid_weights = equity_weight.dropna()

    if not valid_weights.isin([0.0, 1.0]).all():
        raise ValueError("equity_weight must contain only 0.0, 1.0, or missing values")

    valid_turnover = valid_weights.diff().abs()

    if not valid_turnover.empty:
        valid_turnover.iloc[0] = 0

    turnover = pd.Series(
        pd.NA,
        index=equity_weight.index,
        dtype="Float64",
        name="turnover",
    )

    turnover.loc[valid_turnover.index] = valid_turnover

    return turnover


def calculate_transaction_cost(
    turnover: pd.Series,
    cost_rate: float = 0.001,
) -> pd.Series:
    """Calculate transaction costs from one-way turnover."""

    if not np.isfinite(cost_rate) or cost_rate < 0:
        raise ValueError("cost_rate must be finite and non-negative")

    transaction_cost = turnover * cost_rate
    transaction_cost.name = "transaction_cost"

    return transaction_cost


def calculate_gross_return(
    equity_returns: pd.Series,
    risk_free_returns: pd.Series,
    equity_weight: pd.Series,
) -> pd.Series:
    """Combine equity and risk-free returns using the equity weight."""

    if not (
        equity_returns.index.equals(risk_free_returns.index)
        and equity_returns.index.equals(equity_weight.index)
    ):
        raise ValueError("all inputs must have identical indexes")

    gross_return = equity_weight * equity_returns + (1.0 - equity_weight) * risk_free_returns

    gross_return.name = "gross_return"
    return gross_return


def calculate_net_return(
    gross_return: pd.Series,
    transaction_cost: pd.Series,
) -> pd.Series:
    """Deduct transaction costs from gross strategy returns."""
    if not gross_return.index.equals(transaction_cost.index):
        raise ValueError("inputs must have identical indexes")

    net_return = gross_return - transaction_cost
    net_return.name = "net_return"

    return net_return


def run_zero_one_backtest(
    equity_returns: pd.Series,
    risk_free_returns: pd.Series,
    states: pd.Series,
    delay: int = 2,
    cost_rate: float = 0.001,
) -> pd.DataFrame:
    """Run the delayed equity/risk-free regime strategy."""

    equity_weight = delayed_equity_weight(states, delay)
    turnover = calculate_turnover(equity_weight)
    transaction_cost = calculate_transaction_cost(turnover, cost_rate)

    gross_return = calculate_gross_return(
        equity_returns,
        risk_free_returns,
        equity_weight,
    )

    net_return = calculate_net_return(
        gross_return,
        transaction_cost,
    )

    return pd.DataFrame(
        {
            "state": states,
            "equity_weight": equity_weight,
            "turnover": turnover,
            "transaction_cost": transaction_cost,
            "gross_return": gross_return,
            "net_return": net_return,
        }
    )
