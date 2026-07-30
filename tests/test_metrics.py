import numpy as np
import pandas as pd
import pytest

from regimejump.metrics import (
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    cumulative_wealth,
    drawdown_series,
    expected_shortfall,
    maximum_drawdown,
    sharpe_ratio,
)


def test_cumulative_wealth_compounds_simple_returns():
    returns = pd.Series(
        [pd.NA, 0.10, -0.10, 0.05],
        dtype="Float64",
    )

    result = cumulative_wealth(
        returns,
        initial_wealth=1.0,
    )

    expected = pd.Series(
        [pd.NA, 1.10, 0.99, 1.0395],
        dtype="Float64",
        name="wealth",
    )

    pd.testing.assert_series_equal(result, expected)


def test_drawdown_and_maximum_drawdown():
    wealth = pd.Series(
        [pd.NA, 1.0, 1.2, 0.9, 1.0, 1.3],
        dtype="Float64",
        name="wealth",
    )

    drawdown = drawdown_series(wealth)

    assert drawdown.iloc[1] == pytest.approx(0.0)
    assert drawdown.iloc[2] == pytest.approx(0.0)
    assert drawdown.iloc[3] == pytest.approx(-0.25)
    assert drawdown.iloc[4] == pytest.approx(1.0 / 1.2 - 1.0)
    assert drawdown.iloc[5] == pytest.approx(0.0)

    assert maximum_drawdown(wealth) == pytest.approx(-0.25)


def test_annualized_return_compounds_returns():
    returns = pd.Series(
        [0.10, 0.05, -0.10, 0.02],
        dtype="Float64",
    )

    result = annualized_return(
        returns,
        periods_per_year=4,
    )

    expected = (1.10 * 1.05 * 0.90 * 1.02) - 1.0

    assert result == pytest.approx(expected)


def test_annualized_volatility():
    returns = pd.Series(
        [0.01, -0.01, 0.02, -0.02],
        dtype="Float64",
    )

    result = annualized_volatility(
        returns,
        periods_per_year=4,
    )

    expected = returns.astype(float).std(ddof=1) * np.sqrt(4)

    assert result == pytest.approx(expected)


def test_sharpe_ratio():
    index = pd.date_range("2020-01-01", periods=4)

    returns = pd.Series(
        [0.02, 0.00, 0.03, -0.01],
        index=index,
        dtype="Float64",
    )

    risk_free_returns = pd.Series(
        [0.001, 0.001, 0.001, 0.001],
        index=index,
        dtype="Float64",
    )

    result = sharpe_ratio(
        returns,
        risk_free_returns,
        periods_per_year=4,
    )

    expected = (
        (returns - risk_free_returns).mean() * 4 / (returns.astype(float).std(ddof=1) * np.sqrt(4))
    )

    assert result == pytest.approx(expected)


def test_calmar_ratio():
    returns = pd.Series(
        [0.10, -0.20, 0.10],
    )

    expected_return = annualized_return(
        returns,
        periods_per_year=3,
    )

    # Wealth: 1.10, 0.88, 0.968
    # Maximum drawdown: -20%
    expected = expected_return / 0.20

    assert calmar_ratio(
        returns,
        periods_per_year=3,
    ) == pytest.approx(expected)


def test_calmar_ratio_rejects_zero_drawdown():
    returns = pd.Series([0.01, 0.02, 0.03])

    with pytest.raises(
        ValueError,
        match="maximum drawdown is zero",
    ):
        calmar_ratio(returns)


def test_expected_shortfall():
    returns = pd.Series(
        [
            -0.10,
            -0.05,
            0.00,
            0.02,
            0.03,
        ]
    )

    # alpha=0.4 selects the worst two of five returns.
    assert expected_shortfall(
        returns,
        alpha=0.4,
    ) == pytest.approx(-0.075)


def test_expected_shortfall_uses_at_least_one_observation():
    returns = pd.Series(
        [
            -0.10,
            0.00,
            0.10,
        ]
    )

    assert expected_shortfall(
        returns,
        alpha=0.01,
    ) == pytest.approx(-0.10)


def test_expected_shortfall_rejects_invalid_alpha():
    returns = pd.Series([0.01, -0.01])

    with pytest.raises(
        ValueError,
        match="alpha",
    ):
        expected_shortfall(
            returns,
            alpha=0.0,
        )
