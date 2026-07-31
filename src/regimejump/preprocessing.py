"""Training-window feature preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd


def standardize_from_training_window(
    training: pd.DataFrame,
    data: pd.DataFrame,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """Standardize data using statistics from a fixed training window.

    Column means and population standard deviations (``ddof=0``) are
    estimated from ``training`` and then applied unchanged to ``data``.
    Standard deviations are floored at ``eps`` so constant training
    features remain finite.
    """
    if not isinstance(training, pd.DataFrame):
        raise TypeError("training must be a pandas DataFrame")

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    if training.empty:
        raise ValueError("training must contain at least one observation")

    if not training.columns.equals(data.columns):
        raise ValueError("training and data must have identical columns")

    if not np.isfinite(training.to_numpy()).all():
        raise ValueError("training must contain only finite values")

    if not np.isfinite(data.to_numpy()).all():
        raise ValueError("data must contain only finite values")

    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")

    mean = training.mean()
    std = training.std(ddof=0).clip(lower=eps)

    return (data - mean) / std
