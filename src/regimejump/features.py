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

import pandas as pd

DEFAULT_HALFLIVES: tuple[int, ...] = (5, 10, 21)


def feature_names(halflives: Sequence[int] = DEFAULT_HALFLIVES) -> list[str]:
    """Column names produced by :func:`compute_features`, in order."""
    names = []
    for h in halflives:
        names.append(f"ewm_mean_{h}")
        names.append(f"ewm_downside_dev_{h}")
        names.append(f"ewm_abs_mean_{h}")
    return names


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
