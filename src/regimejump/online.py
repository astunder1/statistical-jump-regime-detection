"""Causal standardization and online regime inference.

The module provides expanding-window standardization and two online
inference rules for fixed regime centroids:

- a greedy rule that updates the state one observation at a time;
- a rolling dynamic-programming rule that solves over a trailing window.

Neither inference rule changes states assigned to earlier observations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regimejump.jump import JumpModel


def expanding_standardize(
    data: pd.DataFrame | pd.Series,
    min_periods: int = 2,
    ddof: int = 0,
    eps: float = 1e-12,
) -> pd.DataFrame | pd.Series:
    """Standardize each column using only data up to and including that row.

    z_t = (x_t - mean(x_0..x_t)) / std(x_0..x_t)

    Rows before ``min_periods`` observations are NaN. The standard
    deviation is floored at ``eps`` to avoid division by zero.
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


def rolling_dp_online_path(
    X: np.ndarray,
    centroids: np.ndarray,
    jump_penalty: float,
    lookback: int = 3000,
) -> np.ndarray:
    """Infer states causally using dynamic programming over trailing windows.

    At each observation, the jump-model path is recomputed using at most
    ``lookback`` observations ending at the current row. Only the final
    state of that path is retained, so later observations cannot revise
    previously reported states.

    Parameters
    ----------
    X:
        ``(T, n_features)`` standardized feature matrix.
    centroids:
        ``(K, n_features)`` fixed regime centroids.
    jump_penalty:
        Non-negative switching penalty.
    lookback:
        Maximum number of observations included in each DP window.

    Returns
    -------
    ``(T,)`` integer array containing the causal state sequence.
    """
    X = np.asarray(X, dtype=float)
    centroids = np.asarray(centroids, dtype=float)

    if not isinstance(lookback, int) or lookback < 1:
        raise ValueError("lookback must be a positive integer")

    if X.ndim != 2 or centroids.ndim != 2:
        raise ValueError("X and centroids must be 2D arrays")

    if X.shape[1] != centroids.shape[1]:
        raise ValueError("X and centroids must have the same number of features")

    states = np.empty(len(X), dtype=int)

    model = JumpModel(
        n_states=len(centroids),
        jump_penalty=jump_penalty,
    )

    for t in range(len(X)):
        start = max(0, t - lookback + 1)
        window = X[start : t + 1]

        path, _ = model._dp_state_path(window, centroids)
        states[t] = path[-1]

    return states
