"""EWM feature engineering for statistical jump models.

For a daily return series, builds three exponentially-weighted statistics
at each of a set of halflives (5, 10, 21 trading days by default):

- EWM mean of returns
- EWM downside deviation: EWM standard deviation of ``min(r, 0)``
- EWM mean of ``|r|``

This module only computes features from a *given* window of data; it does
not decide what's "available" at time t. Callers that need a look-ahead-free
standardization for out-of-sample evaluation should use
:mod:`regimejump.online`, which builds on top of these raw features.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

DEFAULT_HALFLIVES: tuple[int, ...] = (5, 10, 21)

PAPER_HALFLIVES: tuple[int, ...] = (20, 60, 120)

PAPER_FEATURE_NAMES: tuple[str, ...] = (
    "downside_dev_10",
    "sortino_20",
    "sortino_60",
)

def feature_names(halflives: Sequence[int] = DEFAULT_HALFLIVES) -> list[str]:
    """Column names produced by :func:`compute_features`, in order."""
    names = []
    for h in halflives:
        names.append(f"ewm_mean_{h}")
        names.append(f"ewm_downside_dev_{h}")
        names.append(f"ewm_abs_mean_{h}")
    return names

def compute_ewm_downside_deviation(
    returns: pd.Series,
    halflife: float,
    min_periods: int = 1,
) -> pd.Series:
    """Compute EWM downside deviation.

    Defined as the square root of the exponentially weighted second
    moment of negative returns.
    """
    negative_returns = returns.clip(upper=0.0)

    downside_second_moment = negative_returns.pow(2).ewm(
        halflife=halflife,
        min_periods=min_periods,
    ).mean()

    return np.sqrt(downside_second_moment)


def compute_features(
    returns: pd.Series,
    halflives: Sequence[int] = DEFAULT_HALFLIVES,
    min_periods: int = 1,
) -> pd.DataFrame:
    """Compute EWM return-based features at each halflife in ``halflives``.

    Parameters
    ----------
    returns:
        A pandas Series of simple or log returns, indexed by date.
    halflives:
        Halflives (in observations) at which to compute each statistic.
    min_periods:
        Minimum number of observations before a value is emitted (passed
        through to ``pandas.Series.ewm``); earlier rows are NaN.

    Returns
    -------
    DataFrame indexed like ``returns`` with columns
    ``feature_names(halflives)``.
    """
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series")

    downside = returns.clip(upper=0.0)
    abs_returns = returns.abs()

    columns = {}
    for h in halflives:
        columns[f"ewm_mean_{h}"] = returns.ewm(halflife=h, min_periods=min_periods).mean()
        columns[f"ewm_downside_dev_{h}"] = downside.ewm(
            halflife=h, min_periods=min_periods
        ).std()
        columns[f"ewm_abs_mean_{h}"] = abs_returns.ewm(
            halflife=h, min_periods=min_periods
        ).mean()

    return pd.DataFrame(columns, index=returns.index, columns=feature_names(halflives))

def compute_paper_features(
    returns: pd.Series,
    min_periods: int = 1,
) -> pd.DataFrame:
    """Compute the feature specification used by Shu et al. (2024)."""
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series")

    downside_dev_10 = compute_ewm_downside_deviation(
        returns,
        halflife=10,
        min_periods=min_periods,
    )

    downside_dev_20 = compute_ewm_downside_deviation(
        returns,
        halflife=20,
        min_periods=min_periods,
    )

    downside_dev_60 = compute_ewm_downside_deviation(
        returns,
        halflife=60,
        min_periods=min_periods,
    )

    ewm_return_20 = returns.ewm(
        halflife=20,
        min_periods=min_periods,
    ).mean()

    ewm_return_60 = returns.ewm(
        halflife=60,
        min_periods=min_periods,
    ).mean()

    return pd.DataFrame(
        {
            "downside_dev_10": downside_dev_10,
            "sortino_20": ewm_return_20 / downside_dev_20,
            "sortino_60": ewm_return_60 / downside_dev_60,
        },
        index=returns.index,
    )
