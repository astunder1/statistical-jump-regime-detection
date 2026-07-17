import numpy as np
import pandas as pd
import pytest

from regimejump.features import DEFAULT_HALFLIVES, compute_features, feature_names


@pytest.fixture
def returns():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=200, freq="B")
    return pd.Series(rng.normal(0, 0.01, size=len(idx)), index=idx, name="ret")


def test_feature_names_order():
    names = feature_names((5, 10))
    assert names == [
        "ewm_mean_5",
        "ewm_downside_dev_5",
        "ewm_abs_mean_5",
        "ewm_mean_10",
        "ewm_downside_dev_10",
        "ewm_abs_mean_10",
    ]


def test_compute_features_shape_and_columns(returns):
    df = compute_features(returns)
    assert df.shape == (len(returns), 3 * len(DEFAULT_HALFLIVES))
    assert list(df.columns) == feature_names(DEFAULT_HALFLIVES)
    assert df.index.equals(returns.index)


def test_rejects_non_series():
    with pytest.raises(TypeError):
        compute_features(np.array([0.01, -0.02, 0.03]))


def test_ewm_mean_matches_pandas_directly(returns):
    df = compute_features(returns, halflives=(10,))
    expected = returns.ewm(halflife=10, min_periods=1).mean()
    pd.testing.assert_series_equal(
        df["ewm_mean_10"], expected.rename("ewm_mean_10"), check_names=False
    )


def test_abs_mean_matches_pandas_directly(returns):
    df = compute_features(returns, halflives=(21,))
    expected = returns.abs().ewm(halflife=21, min_periods=1).mean()
    pd.testing.assert_series_equal(
        df["ewm_abs_mean_21"], expected.rename("ewm_abs_mean_21"), check_names=False
    )


def test_downside_deviation_ignores_positive_returns():
    # An all-positive return series should clip to a constant zero series,
    # whose EWM std is zero (after the initial NaN warm-up).
    idx = pd.date_range("2020-01-01", periods=30, freq="B")
    all_positive = pd.Series(np.linspace(0.001, 0.02, len(idx)), index=idx)
    df = compute_features(all_positive, halflives=(5,))
    dd = df["ewm_downside_dev_5"].dropna()
    assert (dd.abs() < 1e-12).all()


def test_downside_deviation_only_sees_negative_part():
    # Two series that differ only in their positive returns should produce
    # identical downside deviation columns.
    idx = pd.date_range("2020-01-01", periods=60, freq="B")
    rng = np.random.default_rng(1)
    base = rng.normal(0, 0.01, size=len(idx))
    neg_mask = base < 0

    r1 = base.copy()
    r2 = base.copy()
    r2[~neg_mask] = r2[~neg_mask] + 0.05  # perturb only the positive returns

    s1 = pd.Series(r1, index=idx)
    s2 = pd.Series(r2, index=idx)

    df1 = compute_features(s1, halflives=(10,))
    df2 = compute_features(s2, halflives=(10,))

    pd.testing.assert_series_equal(
        df1["ewm_downside_dev_10"], df2["ewm_downside_dev_10"]
    )


def test_min_periods_produces_leading_nans(returns):
    df = compute_features(returns, halflives=(21,), min_periods=21)
    assert df["ewm_mean_21"].iloc[:20].isna().all()
    assert df["ewm_mean_21"].iloc[20:].notna().all()
