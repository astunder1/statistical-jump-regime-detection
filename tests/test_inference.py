from itertools import pairwise

import numpy as np
import pandas as pd

from regimejump.inference import six_month_refit_states


def make_data():
    rng = np.random.default_rng(0)
    index = pd.date_range(
        "2018-01-31",
        periods=24,
        freq="ME",
    )

    features = pd.DataFrame(
        rng.normal(size=(24, 3)),
        index=index,
        columns=["a", "b", "c"],
    )

    returns = pd.Series(
        rng.normal(size=24),
        index=index,
        name="return",
    )

    return features, returns


def test_states_begin_after_complete_training_window():
    features, returns = make_data()

    states, _ = six_month_refit_states(
        features,
        returns,
        jump_penalty=2.0,
        training_length=6,
        n_init=1,
    )

    assert states.iloc[:5].isna().all()
    assert states.iloc[5:].notna().all()


def test_refits_are_six_months_apart():
    features, returns = make_data()

    _, refit_dates = six_month_refit_states(
        features,
        returns,
        jump_penalty=2.0,
        training_length=6,
        n_init=1,
    )

    assert len(refit_dates) > 1

    for previous, current in pairwise(refit_dates):
        assert current >= previous + pd.DateOffset(months=6)


def test_six_month_inference_is_causal():
    features, returns = make_data()

    original, _ = six_month_refit_states(
        features,
        returns,
        jump_penalty=2.0,
        training_length=6,
        n_init=1,
    )

    changed_features = features.copy()
    changed_returns = returns.copy()

    changed_features.iloc[18:] += 1000.0
    changed_returns.iloc[18:] += 1000.0

    perturbed, _ = six_month_refit_states(
        changed_features,
        changed_returns,
        jump_penalty=2.0,
        training_length=6,
        n_init=1,
    )

    pd.testing.assert_series_equal(
        original.iloc[:18],
        perturbed.iloc[:18],
    )
