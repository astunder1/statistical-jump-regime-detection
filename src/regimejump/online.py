"""Look-ahead-free standardization and one-step-ahead online regime decisions.

Two independent pieces:

- :func:`expanding_standardize` turns raw features into z-scores using only
  data available up to and including each row -- safe for any out-of-sample
  evaluation, unlike full-sample standardization (which is only fine inside
  unit tests).
- :func:`online_state_decision` / :func:`greedy_online_path` implement the
  one-step online decision rule for a *fitted* jump model (fixed centroids):
  stay in the previous state unless another centroid is closer by more than
  the jump penalty. This is a greedy approximation to the full DP path,
  useful for genuinely online (no-relabeling-the-past) evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def expanding_standardize(
    data: pd.DataFrame | pd.Series,
    min_periods: int = 2,
    ddof: int = 0,
    eps: float = 1e-12,
) -> pd.DataFrame | pd.Series:
    """Standardize each column using only data up to and including that row.

    z_t = (x_t - mean(x_0..x_t)) / std(x_0..x_t)

    Only past and current observations are used at each row, so this is
    safe for out-of-sample use (no look-ahead). Rows before ``min_periods``
    observations are NaN. A tiny floor ``eps`` on the rolling std avoids
    division by zero when a window is (numerically) constant.
    """
    mean = data.expanding(min_periods=min_periods).mean()
    std = data.expanding(min_periods=min_periods).std(ddof=ddof)
    return (data - mean) / std.clip(lower=eps)


def online_state_decision(
    x: np.ndarray,
    centroids: np.ndarray,
    prev_state: int,
    jump_penalty: float,
) -> int:
    """One-step online regime decision for a single new observation.

    Stays in ``prev_state`` unless some other centroid's squared distance
    to ``x`` is smaller than the previous state's by more than
    ``jump_penalty``. This mirrors the trade-off in the jump-model
    objective (fit cost vs. switching penalty) but is computed greedily,
    looking only one step ahead rather than solving the full DP.
    """
    x = np.asarray(x, dtype=float)
    centroids = np.asarray(centroids, dtype=float)

    dists = 0.5 * np.sum((centroids - x) ** 2, axis=1)
    best = int(np.argmin(dists))

    if best == prev_state:
        return prev_state
    if dists[best] < dists[prev_state] - jump_penalty:
        return best
    return prev_state


def greedy_online_path(
    X: np.ndarray,
    centroids: np.ndarray,
    jump_penalty: float,
    init_state: int | None = None,
) -> np.ndarray:
    """Run :func:`online_state_decision` sequentially over a feature matrix.

    Parameters
    ----------
    X:
        (T, n_features) array of standardized features.
    centroids:
        (K, n_features) array of fixed regime centroids.
    jump_penalty:
        Non-negative switching penalty (same ``lambda`` as the jump model).
    init_state:
        State for the first observation. Defaults to the nearest centroid
        (no penalty applies before there is a "previous" state).

    Returns
    -------
    (T,) integer array of state assignments, causal at every step: state
    ``t`` depends only on ``X[0..t]``, never on future rows.
    """
    X = np.asarray(X, dtype=float)
    centroids = np.asarray(centroids, dtype=float)
    n_obs = X.shape[0]

    path = np.empty(n_obs, dtype=int)
    if n_obs == 0:
        return path

    if init_state is None:
        init_dists = np.sum((centroids - X[0]) ** 2, axis=1)
        init_state = int(np.argmin(init_dists))
    path[0] = init_state

    for t in range(1, n_obs):
        path[t] = online_state_decision(X[t], centroids, path[t - 1], jump_penalty)

    return path
