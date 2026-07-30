import pandas as pd
import pytest

from regimejump.selection import (
    monthly_selected_state_path,
    select_best_penalty,
)


def test_selects_penalty_with_highest_sharpe():
    scores = {
        0.0: 0.20,
        5.0: 0.35,
        15.0: 0.50,
        35.0: 0.42,
    }

    selected = select_best_penalty(scores)

    assert selected == 15.0


def test_ignores_nonfinite_sharpe():
    scores = {
        0.0: float("nan"),
        5.0: 0.30,
    }

    selected = select_best_penalty(scores)

    assert selected == 5.0


def test_rejects_when_no_finite_sharpe_exists():
    scores = {
        0.0: float("nan"),
        5.0: float("inf"),
    }

    with pytest.raises(ValueError, match="finite Sharpe"):
        select_best_penalty(scores)


def test_generate_candidate_state_paths(monkeypatch):
    from regimejump import selection

    dates = pd.bdate_range("2020-01-01", periods=5)
    features = pd.DataFrame({"x": range(5)}, index=dates)
    returns = pd.Series(0.0, index=dates)

    called_penalties = []

    def fake_refit_states(
        features,
        returns,
        jump_penalty,
        **kwargs,
    ):
        called_penalties.append(jump_penalty)

        states = pd.Series(
            int(jump_penalty),
            index=features.index,
            dtype="Int64",
            name="state",
        )

        return states, []

    monkeypatch.setattr(
        selection,
        "six_month_refit_states",
        fake_refit_states,
    )

    paths = selection.generate_candidate_state_paths(
        features=features,
        model_returns=returns,
        candidates=(0.0, 5.0),
    )

    assert called_penalties == [0.0, 5.0]
    assert set(paths) == {0.0, 5.0}
    assert (paths[5.0] == 5).all()


def test_generate_candidate_backtests():
    from regimejump import selection

    dates = pd.bdate_range("2020-01-01", periods=4)

    equity_returns = pd.Series(
        0.01,
        index=dates,
        name="equity_return",
    )

    risk_free_returns = pd.Series(
        0.0,
        index=dates,
        name="risk_free_return",
    )

    state_paths = {
        0.0: pd.Series(
            [0, 0, 0, 0],
            index=dates,
            dtype="Int64",
            name="state",
        ),
        5.0: pd.Series(
            [1, 1, 1, 1],
            index=dates,
            dtype="Int64",
            name="state",
        ),
    }

    backtests = selection.generate_candidate_backtests(
        equity_returns=equity_returns,
        risk_free_returns=risk_free_returns,
        state_paths=state_paths,
        delay=0,
    )

    assert set(backtests) == {0.0, 5.0}
    assert (backtests[0.0]["net_return"] == 0.01).all()
    assert (backtests[5.0]["net_return"] == 0.0).all()


def test_monthly_selection_applies_new_penalty_on_second_day(
    monkeypatch,
):
    from regimejump import selection

    dates = pd.bdate_range(
        "2018-01-01",
        "2020-01-31",
    )

    features = pd.DataFrame(
        {"feature": 0.0},
        index=dates,
    )

    returns = pd.Series(
        0.0,
        index=dates,
    )

    state_paths = {
        0.0: pd.Series(
            0,
            index=dates,
            dtype="Int64",
            name="state",
        ),
        5.0: pd.Series(
            1,
            index=dates,
            dtype="Int64",
            name="state",
        ),
    }

    candidate_backtests = {
        penalty: pd.DataFrame(
            {"net_return": 0.0},
            index=dates,
        )
        for penalty in state_paths
    }

    monkeypatch.setattr(
        selection,
        "generate_candidate_state_paths",
        lambda **kwargs: state_paths,
    )

    monkeypatch.setattr(
        selection,
        "generate_candidate_backtests",
        lambda **kwargs: candidate_backtests,
    )

    # December selection: lambda 0 wins.
    # January selection: lambda 5 wins.
    scores = iter([0.9, 0.1, 0.1, 0.9])

    monkeypatch.setattr(
        selection,
        "sharpe_ratio",
        lambda *args, **kwargs: next(scores),
    )

    states, _, _ = selection.monthly_selected_state_path(
        features=features,
        model_returns=returns,
        equity_returns=returns,
        risk_free_returns=returns,
        test_start="2020-01-01",
        test_end="2020-01-31",
        candidates=(0.0, 5.0),
        training_length=100,
        validation_years=1,
    )

    january_states = states.loc["2020-01"]

    assert january_states.iloc[0] == 0
    assert (january_states.iloc[1:] == 1).all()


def test_rejects_invalid_validation_years():
    dates = pd.bdate_range(
        "2020-01-01",
        periods=5,
    )

    features = pd.DataFrame(
        {"feature": 0.0},
        index=dates,
    )

    returns = pd.Series(
        0.0,
        index=dates,
    )

    with pytest.raises(
        ValueError,
        match="validation_years",
    ):
        monthly_selected_state_path(
            features=features,
            model_returns=returns,
            equity_returns=returns,
            risk_free_returns=returns,
            test_start="2020-01-01",
            test_end="2020-01-31",
            validation_years=0,
        )
