import numpy as np
import pandas as pd
import pytest

from regimejump.jump import JumpModel
from regimejump.online import (
    expanding_standardize,
    greedy_online_path,
    online_state_decision,
    rolling_dp_online_path,
)


@pytest.fixture
def df():
    rng = np.random.default_rng(42)
    idx = pd.date_range("2020-01-01", periods=100, freq="B")
    return pd.DataFrame(
        {"a": rng.normal(0, 1, 100), "b": rng.normal(5, 2, 100)}, index=idx
    )


# ---- expanding_standardize -------------------------------------------------


def test_expanding_standardize_matches_manual_slice(df):
    z = expanding_standardize(df, min_periods=5)
    for t in (10, 50, 99):
        window = df.iloc[: t + 1]
        expected = (df.iloc[t] - window.mean()) / window.std(ddof=0)
        got = z.iloc[t]
        pd.testing.assert_series_equal(got, expected, check_names=False)


def test_expanding_standardize_no_lookahead(df):
    z_full = expanding_standardize(df, min_periods=5)

    # Perturb only rows after t=60; rows up to and including 60 must be
    # identical, since expanding_standardize at row t must not depend on
    # rows > t.
    perturbed = df.copy()
    perturbed.iloc[61:] = perturbed.iloc[61:] + 1000.0
    z_perturbed = expanding_standardize(perturbed, min_periods=5)

    pd.testing.assert_frame_equal(z_full.iloc[: 61], z_perturbed.iloc[: 61])


def test_expanding_standardize_leading_nans(df):
    z = expanding_standardize(df, min_periods=10)
    assert z.iloc[:9].isna().all().all()
    assert z.iloc[9:].notna().all().all()


def test_expanding_standardize_series_input():
    idx = pd.date_range("2020-01-01", periods=20, freq="B")
    s = pd.Series(np.arange(20, dtype=float), index=idx)
    z = expanding_standardize(s, min_periods=2)
    assert isinstance(z, pd.Series)
    assert z.notna().sum() == 19


# ---- online_state_decision --------------------------------------------------


def test_stays_when_improvement_below_penalty():
    centroids = np.array([[0.0, 0.0], [1.0, 0.0]])
    x = np.array([0.6, 0.0])  # dist to c0 = 0.36, dist to c1 = 0.16 -> improves by 0.20
    new_state = online_state_decision(x, centroids, prev_state=0, jump_penalty=0.25)
    assert new_state == 0


def test_switches_when_improvement_exceeds_penalty():
    centroids = np.array([[0.0, 0.0], [1.0, 0.0]])
    x = np.array([0.6, 0.0])  # improvement of 0.20 over jump_penalty=0.05
    new_state = online_state_decision(x, centroids, prev_state=0, jump_penalty=0.05)
    assert new_state == 1


def test_no_switch_when_already_best():
    centroids = np.array([[0.0, 0.0], [1.0, 0.0]])
    x = np.array([0.05, 0.0])
    new_state = online_state_decision(x, centroids, prev_state=0, jump_penalty=0.0)
    assert new_state == 0


def test_boundary_is_strict_inequality():
    # Improvement exactly equal to jump_penalty should NOT trigger a switch
    # (must be strictly smaller by more than the penalty).
    centroids = np.array([[0.0, 0.0], [1.0, 0.0]])
    x = np.array([0.6, 0.0])  # dist0=0.36, dist1=0.16, improvement=0.10
    new_state = online_state_decision(x, centroids, prev_state=0, jump_penalty=0.10)
    assert new_state == 0


# ---- greedy_online_path -----------------------------------------------------


def test_zero_penalty_matches_nearest_centroid():
    rng = np.random.default_rng(0)
    centroids = np.array([[0.0, 0.0], [5.0, 5.0]])
    X = rng.normal(loc=0, scale=1, size=(50, 2))
    X[25:] += 5  # push second half near the other centroid

    path = greedy_online_path(X, centroids, jump_penalty=0.0)
    dists = np.stack(
        [np.sum((X - c) ** 2, axis=1) for c in centroids], axis=1
    )
    expected = np.argmin(dists, axis=1)
    np.testing.assert_array_equal(path, expected)


def test_huge_penalty_never_switches_after_first():
    rng = np.random.default_rng(1)
    centroids = np.array([[0.0, 0.0], [10.0, 10.0]])
    X = rng.normal(loc=5, scale=1, size=(40, 2))

    path = greedy_online_path(X, centroids, jump_penalty=1e9)
    assert (path == path[0]).all()


def test_path_is_causal_no_lookahead():
    rng = np.random.default_rng(2)
    centroids = np.array([[0.0, 0.0], [5.0, 5.0]])
    X = rng.normal(loc=2, scale=1, size=(30, 2))

    path_full = greedy_online_path(X, centroids, jump_penalty=1.0)
    # Truncating the future must not change any already-computed state.
    path_prefix = greedy_online_path(X[:15], centroids, jump_penalty=1.0)
    np.testing.assert_array_equal(path_full[:15], path_prefix)


def test_empty_input_returns_empty_path():
    centroids = np.array([[0.0, 0.0], [1.0, 1.0]])
    path = greedy_online_path(np.empty((0, 2)), centroids, jump_penalty=1.0)
    assert path.shape == (0,)

def test_rolling_dp_matches_direct_window_calculation():
    rng = np.random.default_rng(10)
    X = rng.normal(size=(20, 2))
    centroids = rng.normal(size=(2, 2))
    penalty = 1.5
    lookback = 5

    states = rolling_dp_online_path(
        X,
        centroids,
        penalty,
        lookback,
    )

    model = JumpModel(n_states=2, jump_penalty=penalty)

    for t in range(len(X)):
        start = max(0, t - lookback + 1)
        expected_path, _ = model._dp_state_path(
            X[start : t + 1],
            centroids,
        )
        assert states[t] == expected_path[-1]


def test_rolling_dp_is_causal():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(30, 2))
    centroids = rng.normal(size=(2, 2))

    original = rolling_dp_online_path(X, centroids, 2.0, lookback=10)

    changed = X.copy()
    changed[20:] += 1000.0

    perturbed = rolling_dp_online_path(
        changed,
        centroids,
        2.0,
        lookback=10,
    )

    np.testing.assert_array_equal(original[:20], perturbed[:20])