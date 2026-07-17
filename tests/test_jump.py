"""Correctness tests for JumpModel, written before the implementation.

These tests define what "correct" means for `_dp_state_path` and `fit`
(both left as documented stubs in jump.py). They are expected to fail with
NotImplementedError until those methods are implemented by hand.

Full-sample z-scoring is used here (never in a backtest / reported OOS
number) purely to keep these unit tests simple and self-contained.
"""

from __future__ import annotations

import numpy as np
import pytest

from regimejump.jump import JumpModel, jump_objective
from regimejump.online import greedy_online_path


def n_switches(labels: np.ndarray) -> int:
    return int(np.sum(np.asarray(labels)[1:] != np.asarray(labels)[:-1]))


def simulate_two_regime_returns(
    n_per_regime: int = 250, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Calm bull half followed by a volatile bear half.

    Returns the raw return series and the ground-truth regime label per
    observation (0 = calm bull, 1 = volatile bear).
    """
    rng = np.random.default_rng(seed)
    bull = rng.normal(loc=0.0008, scale=0.005, size=n_per_regime)
    bear = rng.normal(loc=-0.0015, scale=0.025, size=n_per_regime)
    returns = np.concatenate([bull, bear])
    true_labels = np.concatenate(
        [np.zeros(n_per_regime, dtype=int), np.ones(n_per_regime, dtype=int)]
    )
    return returns, true_labels


def zscore(X: np.ndarray) -> np.ndarray:
    """Full-sample standardization -- unit tests only, never OOS reporting."""
    return (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)


def simple_features_from_returns(returns: np.ndarray) -> np.ndarray:
    """A minimal 2-column feature set: level and local (rolling) volatility.

    Deliberately independent of regimejump.features, so these tests don't
    silently depend on a particular feature-engineering choice.
    """
    import pandas as pd

    r = pd.Series(returns)
    mean_5 = r.rolling(5, min_periods=1).mean()
    vol_5 = r.rolling(5, min_periods=1).std(ddof=0).fillna(0.0)
    return np.column_stack([mean_5.to_numpy(), vol_5.to_numpy()])


@pytest.fixture
def random_features():
    rng = np.random.default_rng(123)
    return rng.normal(size=(120, 3))


# ---- (a) lower penalty switches more than higher penalty -------------------


def test_lower_penalty_switches_more_than_higher_penalty():
    returns, _ = simulate_two_regime_returns()
    X = zscore(simple_features_from_returns(returns))

    model_low = JumpModel(n_states=2, jump_penalty=0.0, n_init=5, random_state=0).fit(X)
    model_high = JumpModel(n_states=2, jump_penalty=100.0, n_init=5, random_state=0).fit(X)

    assert n_switches(model_low.labels_) > n_switches(model_high.labels_)


# ---- (b) huge penalty forces zero switches ---------------------------------


def test_huge_penalty_forces_zero_switches():
    returns, _ = simulate_two_regime_returns()
    X = zscore(simple_features_from_returns(returns))

    model = JumpModel(n_states=2, jump_penalty=1e9, n_init=5, random_state=0).fit(X)

    assert n_switches(model.labels_) == 0


# ---- (c) >90% state accuracy on a simulated two-regime series -------------


def test_state_accuracy_on_simulated_two_regime_series():
    returns, true_labels = simulate_two_regime_returns()
    X = zscore(simple_features_from_returns(returns))

    model = JumpModel(n_states=2, jump_penalty=10.0, n_init=10, random_state=0).fit(X)
    model.relabel_by_feature(X, feature_idx=0)  # state 0 = high-mean = calm bull

    accuracy = np.mean(model.labels_ == true_labels)
    assert accuracy > 0.90, f"accuracy {accuracy:.3f} did not exceed 90%"


# ---- (d) DP path objective <= greedy online path objective ----------------


def test_dp_path_beats_or_matches_greedy_path(random_features):
    X = zscore(random_features)
    jump_penalty = 2.0

    rng = np.random.default_rng(7)
    # Arbitrary but fixed centroids -- we only need *some* centroids to
    # compare the two path-finding strategies against, not a fitted model.
    centroids = rng.normal(size=(3, X.shape[1]))

    model = JumpModel(n_states=3, jump_penalty=jump_penalty)
    dp_path, dp_cost = model._dp_state_path(X, centroids)

    greedy_path = greedy_online_path(X, centroids, jump_penalty=jump_penalty)
    greedy_cost = jump_objective(X, centroids, greedy_path, jump_penalty)

    assert dp_cost == pytest.approx(jump_objective(X, centroids, dp_path, jump_penalty))
    assert dp_cost <= greedy_cost + 1e-9


# ---- (e) gate against the `jumpmodels` PyPI reference implementation ------


@pytest.mark.skip(
    reason="Enable once JumpModel.fit is implemented, to gate against the "
    "jumpmodels reference implementation."
)
def test_matches_jumpmodels_reference():
    from jumpmodels.jump import JumpModel as ReferenceJumpModel  # noqa: F401

    returns, true_labels = simulate_two_regime_returns()
    X = zscore(simple_features_from_returns(returns))

    ours = JumpModel(n_states=2, jump_penalty=10.0, n_init=10, random_state=0).fit(X)
    ours.relabel_by_feature(X, feature_idx=0)

    # Placeholder comparison -- fill in once the reference API is wired up:
    # reference = ReferenceJumpModel(...).fit(X)
    # agreement = np.mean(ours.labels_ == reference.labels_)
    # assert agreement > 0.95
    raise NotImplementedError("wire up the jumpmodels reference comparison")
