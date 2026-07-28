"""Correctness tests for JumpModel, written before the implementation.

These tests define what "correct" means for `_dp_state_path` and `fit`
(both left as documented stubs in jump.py). They are expected to fail with
NotImplementedError until those methods are implemented by hand.

Full-sample z-scoring is used here (never in a backtest / reported OOS
number) purely to keep these unit tests simple and self-contained.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from regimejump.jump import JumpModel, jump_objective
from regimejump.online import greedy_online_path, rolling_dp_online_path


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


def test_matches_jumpmodels_reference():
    from jumpmodels.jump import JumpModel as ReferenceJumpModel

    returns, _ = simulate_two_regime_returns()
    X = zscore(simple_features_from_returns(returns))

    ours = JumpModel(
        n_states=2,
        jump_penalty=10.0,
        max_iter=50,
        n_init=10,
        random_state=0,
    ).fit(X)

    reference = ReferenceJumpModel(
        n_components=2,
        jump_penalty=10.0,
        cont=False,
        mode_loss=True,
        max_iter=50,
        n_init=10,
        random_state=0,
    ).fit(X, sort_by=None)

    # State numbers are arbitrary, so compare both possible alignments.
    direct_agreement = np.mean(ours.labels_ == reference.labels_)
    reversed_agreement = np.mean(ours.labels_ == (1 - reference.labels_))
    agreement = max(direct_agreement, reversed_agreement)

    assert agreement > 0.99
    assert ours.inertia_ == pytest.approx(reference.val_)


def test_dp_simple_switch():
    X = np.array([
        [0.0],
        [1.0],
        [9.0],
        [10.0],
    ])

    centroids = np.array([
        [0.0],
        [10.0],
    ])

    model = JumpModel(n_states=2, jump_penalty=4.0)
    path, cost = model._dp_state_path(X, centroids)

    expected_path = np.array([0, 0, 1, 1])
    expected_cost = 5.0

    np.testing.assert_array_equal(path, expected_path)
    assert cost == pytest.approx(expected_cost)


def test_dp_zero_penalty_chooses_nearest_centroid():
    X = np.array([
        [0.0],
        [9.0],
        [1.0],
        [10.0],
    ])

    centroids = np.array([
        [0.0],
        [10.0],
    ])

    model = JumpModel(n_states=2, jump_penalty=0.0)
    path, _ = model._dp_state_path(X, centroids)

    expected_path = np.array([0, 1, 0, 1])
    np.testing.assert_array_equal(path, expected_path)


def test_dp_large_penalty_prevents_switching():
    X = np.array([
        [0.0],
        [1.0],
        [9.0],
        [10.0],
    ])

    centroids = np.array([
        [0.0],
        [10.0],
    ])

    model = JumpModel(n_states=2, jump_penalty=1_000_000.0)
    path, _ = model._dp_state_path(X, centroids)

    assert np.all(path == path[0])


def test_dp_reported_cost_matches_objective():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(20, 3))
    centroids = rng.normal(size=(2, 3))
    penalty = 2.5

    model = JumpModel(n_states=2, jump_penalty=penalty)
    path, cost = model._dp_state_path(X, centroids)

    independently_calculated = jump_objective(
        X,
        centroids,
        path,
        penalty,
    )

    assert cost == pytest.approx(independently_calculated)


def brute_force_best_cost(
    X: np.ndarray,
    centroids: np.ndarray,
    jump_penalty: float,
) -> float:
    n_obs = len(X)
    n_states = len(centroids)
    best_cost = np.inf

    for candidate in product(range(n_states), repeat=n_obs):
        path = np.asarray(candidate, dtype=int)

        cost = jump_objective(
            X,
            centroids,
            path,
            jump_penalty,
        )

        best_cost = min(best_cost, cost)

    return float(best_cost)


@pytest.mark.parametrize("seed", range(20))
def test_dp_matches_brute_force(seed):
    rng = np.random.default_rng(seed)

    X = rng.normal(size=(6, 2))
    centroids = rng.normal(size=(3, 2))
    penalty = float(rng.uniform(0.0, 5.0))

    model = JumpModel(
        n_states=3,
        jump_penalty=penalty,
    )

    path, dp_cost = model._dp_state_path(X, centroids)

    brute_force_cost = brute_force_best_cost(
        X,
        centroids,
        penalty,
    )

    assert dp_cost == pytest.approx(brute_force_cost)

    assert jump_objective(
        X,
        centroids,
        path,
        penalty,
    ) == pytest.approx(brute_force_cost)


def test_dp_rejects_negative_penalty():
    X = np.array([[0.0], [1.0]])
    centroids = np.array([[0.0], [1.0]])

    model = JumpModel(n_states=2, jump_penalty=-1.0)

    with pytest.raises(ValueError, match="jump_penalty"):
        model._dp_state_path(X, centroids)


def test_dp_rejects_empty_X():
    X = np.empty((0, 1))
    centroids = np.array([[0.0], [1.0]])

    model = JumpModel(n_states=2)

    with pytest.raises(ValueError, match="at least one observation"):
        model._dp_state_path(X, centroids)


def test_dp_rejects_nan():
    X = np.array([[0.0], [np.nan]])
    centroids = np.array([[0.0], [1.0]])

    model = JumpModel(n_states=2)

    with pytest.raises(ValueError, match="finite"):
        model._dp_state_path(X, centroids)


def test_update_centroids_uses_state_means():
    X = np.array([
        [0.0, 2.0],
        [2.0, 4.0],
        [8.0, 10.0],
        [10.0, 12.0],
    ])

    labels = np.array([0, 0, 1, 1])

    old_centroids = np.array([
        [-5.0, -5.0],
        [20.0, 20.0],
    ])

    model = JumpModel(n_states=2)

    updated = model._update_centroids(
        X,
        labels,
        old_centroids,
    )

    expected = np.array([
        [1.0, 3.0],
        [9.0, 11.0],
    ])

    np.testing.assert_allclose(updated, expected)


def test_update_centroids_does_not_modify_old_centroids():
    X = np.array([
        [0.0],
        [2.0],
        [8.0],
        [10.0],
    ])
    labels = np.array([0, 0, 1, 1])
    old_centroids = np.array([[-5.0], [20.0]])
    original = old_centroids.copy()

    model = JumpModel(n_states=2)
    model._update_centroids(X, labels, old_centroids)

    np.testing.assert_array_equal(old_centroids, original)


def test_update_centroids_reseeds_empty_state_with_worst_fitted_point():
    X = np.array([
        [0.0],
        [1.0],
        [10.0],
    ])

    # Every observation is currently assigned to state 0.
    labels = np.array([0, 0, 0])

    old_centroids = np.array([
        [0.0],
        [50.0],
    ])

    model = JumpModel(n_states=2)
    updated = model._update_centroids(X, labels, old_centroids)

    # State 0 is updated to the mean: (0 + 1 + 10) / 3.
    assert updated[0, 0] == pytest.approx(11.0 / 3.0)

    # Relative to that mean, X=10 is the worst-fitted observation.
    assert updated[1, 0] == pytest.approx(10.0)


def test_update_centroids_uses_distinct_reseeds():
    X = np.array([
        [0.0],
        [1.0],
        [10.0],
    ])
    labels = np.array([0, 0, 0])

    old_centroids = np.array([
        [0.0],
        [50.0],
        [60.0],
    ])

    model = JumpModel(n_states=3)
    updated = model._update_centroids(X, labels, old_centroids)

    # Empty states should receive different observations.
    assert not np.array_equal(updated[1], updated[2])

@pytest.mark.parametrize(
    ("X", "message"),
    [
        (np.array([1.0, 2.0]), "2D"),
        (np.empty((0, 2)), "at least one observation"),
        (np.empty((2, 0)), "at least one feature"),
        (np.array([[0.0], [np.nan]]), "finite"),
    ],
)
def test_fit_rejects_invalid_X(X, message):
    model = JumpModel()

    with pytest.raises(ValueError, match=message):
        model.fit(X)


def test_fit_rejects_more_states_than_observations():
    X = np.array([[0.0], [1.0]])
    model = JumpModel(n_states=3)

    with pytest.raises(ValueError, match="cannot exceed"):
        model.fit(X)

@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_states": 0}, "n_states"),
        ({"jump_penalty": -1.0}, "jump_penalty"),
        ({"max_iter": 0}, "max_iter"),
        ({"n_init": 0}, "n_init"),
    ],
)
def test_fit_rejects_invalid_parameters(kwargs, message):
    X = np.array([[0.0], [1.0]])
    model = JumpModel(**kwargs)

    with pytest.raises(ValueError, match=message):
        model.fit(X)

def test_fit_sets_expected_attributes():
    X = np.array([
        [0.0],
        [0.2],
        [0.1],
        [9.9],
        [10.0],
        [10.1],
    ])

    model = JumpModel(
        n_states=2,
        jump_penalty=1.0,
        n_init=1,
        random_state=0,
    ).fit(X)

    assert model.centroids_.shape == (2, 1)
    assert model.labels_.shape == (len(X),)
    assert model.n_iter_ >= 1
    assert model.n_iter_ <= model.max_iter
    assert np.isfinite(model.inertia_)

    assert model.inertia_ == pytest.approx(
        jump_objective(
            X,
            model.centroids_,
            model.labels_,
            model.jump_penalty,
        )
    )


def test_single_restart_fit_is_reproducible():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(50, 2))

    model_a = JumpModel(
        n_states=2,
        jump_penalty=2.0,
        n_init=1,
        random_state=7,
    ).fit(X)

    model_b = JumpModel(
        n_states=2,
        jump_penalty=2.0,
        n_init=1,
        random_state=7,
    ).fit(X)

    np.testing.assert_allclose(
        model_a.centroids_,
        model_b.centroids_,
    )
    np.testing.assert_array_equal(
        model_a.labels_,
        model_b.labels_,
    )
    assert model_a.inertia_ == pytest.approx(model_b.inertia_)

def test_multiple_restarts_do_not_worsen_objective():
    rng = np.random.default_rng(321)
    X = rng.normal(size=(100, 3))

    model_one = JumpModel(
        n_states=3,
        jump_penalty=2.0,
        n_init=1,
        random_state=10,
    ).fit(X)

    model_many = JumpModel(
        n_states=3,
        jump_penalty=2.0,
        n_init=5,
        random_state=10,
    ).fit(X)

    assert model_many.inertia_ <= model_one.inertia_ + 1e-12

def test_fitted_labels_match_dp_path_for_fitted_centroids():
    rng = np.random.default_rng(456)
    X = rng.normal(size=(100, 3))

    model = JumpModel(
        n_states=3,
        jump_penalty=2.0,
        n_init=5,
        random_state=0,
    ).fit(X)

    expected_labels, expected_cost = model._dp_state_path(
        X,
        model.centroids_,
    )

    np.testing.assert_array_equal(model.labels_, expected_labels)
    assert model.inertia_ == pytest.approx(expected_cost)


def test_relabel_by_cumulative_return():
    model = JumpModel(n_states=2)
    model.centroids_ = np.array([[10.0], [20.0]])
    model.labels_ = np.array([0, 0, 1, 1])

    returns = np.array([-0.03, -0.02, 0.01, 0.02])

    model.relabel_by_cumulative_return(returns)

    np.testing.assert_array_equal(
        model.labels_,
        np.array([1, 1, 0, 0]),
    )
    np.testing.assert_array_equal(
        model.centroids_,
        np.array([[20.0], [10.0]]),
    )


def test_online_states_match_jumpmodels_reference():
    from jumpmodels.jump import JumpModel as ReferenceJumpModel

    returns, _ = simulate_two_regime_returns()
    X = zscore(simple_features_from_returns(returns))

    ours = JumpModel(
        n_states=2,
        jump_penalty=10.0,
        max_iter=50,
        n_init=10,
        random_state=0,
    ).fit(X)

    ours.relabel_by_cumulative_return(returns)

    reference = ReferenceJumpModel(
        n_components=2,
        jump_penalty=10.0,
        cont=False,
        mode_loss=True,
        max_iter=50,
        n_init=10,
        random_state=0,
    ).fit(
        X,
        ret_ser=returns,
        sort_by="cumret",
    )

    # Confirm state ordering and centroids match before comparing inference.
    np.testing.assert_allclose(
        ours.centroids_,
        reference.centers_,
    )

    # Create a nontrivial test sequence that moves between both centroids.
    rng = np.random.default_rng(9)

    X_test = np.vstack([
        ours.centroids_[0] + rng.normal(0.0, 0.1, size=(20, X.shape[1])),
        ours.centroids_[1] + rng.normal(0.0, 0.1, size=(30, X.shape[1])),
        ours.centroids_[0] + rng.normal(0.0, 0.1, size=(20, X.shape[1])),
    ])

    ours_online = rolling_dp_online_path(
        X_test,
        ours.centroids_,
        jump_penalty=10.0,
        lookback=len(X_test),
    )

    reference_online = np.asarray(
        reference.predict_online(X_test),
        dtype=int,
    )

    # Ensure this is not a trivial all-one-state comparison.
    assert np.sum(ours_online[1:] != ours_online[:-1]) > 0

    np.testing.assert_array_equal(
        ours_online,
        reference_online,
    )