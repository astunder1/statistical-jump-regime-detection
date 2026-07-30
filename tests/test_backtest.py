"""Tests for the delayed binary regime-strategy backtest."""

import pandas as pd

from regimejump.backtest import (
    calculate_gross_return,
    calculate_net_return,
    calculate_transaction_cost,
    calculate_turnover,
    delayed_equity_weight,
    run_zero_one_backtest,
)

# ---------------------------------------------------------------------------
# Allocation and turnover
# ---------------------------------------------------------------------------


def test_delayed_equity_weight_applies_signal_two_days_later():
    index = pd.date_range("2020-01-01", periods=6, freq="B")

    states = pd.Series(
        [pd.NA, 0, 0, 1, 1, 0],
        index=index,
        dtype="Int64",
    )

    result = delayed_equity_weight(states, delay=2)

    expected = pd.Series(
        [pd.NA, pd.NA, pd.NA, 1.0, 1.0, 0.0],
        index=index,
        dtype="Float64",
        name="equity_weight",
    )

    pd.testing.assert_series_equal(result, expected)


def test_zero_delay_applies_signal_immediately():
    states = pd.Series([0, 1, 0], dtype="Int64")

    result = delayed_equity_weight(states, delay=0)

    expected = pd.Series(
        [1.0, 0.0, 1.0],
        dtype="Float64",
        name="equity_weight",
    )

    pd.testing.assert_series_equal(result, expected)


def test_delayed_weight_preserves_missing_states():
    states = pd.Series(
        [pd.NA, pd.NA, 0, 1],
        dtype="Int64",
    )

    result = delayed_equity_weight(states, delay=0)

    expected = pd.Series(
        [pd.NA, pd.NA, 1.0, 0.0],
        dtype="Float64",
        name="equity_weight",
    )

    pd.testing.assert_series_equal(result, expected)


def test_calculate_turnover_detects_allocation_changes():
    weights = pd.Series(
        [pd.NA, 1.0, 1.0, 0.0, 0.0, 1.0],
        dtype="Float64",
        name="equity_weight",
    )

    result = calculate_turnover(weights)

    expected = pd.Series(
        [pd.NA, 0.0, 0.0, 1.0, 0.0, 1.0],
        dtype="Float64",
        name="turnover",
    )

    pd.testing.assert_series_equal(result, expected)


# ---------------------------------------------------------------------------
# Return and cost calculations
# ---------------------------------------------------------------------------


def test_transaction_cost_is_turnover_times_rate():
    turnover = pd.Series(
        [pd.NA, 0.0, 1.0, 0.0, 1.0],
        dtype="Float64",
        name="turnover",
    )

    result = calculate_transaction_cost(
        turnover,
        cost_rate=0.001,
    )

    expected = pd.Series(
        [pd.NA, 0.0, 0.001, 0.0, 0.001],
        dtype="Float64",
        name="transaction_cost",
    )

    pd.testing.assert_series_equal(result, expected)


def test_gross_return_switches_between_equity_and_risk_free():
    index = pd.date_range("2020-01-01", periods=4, freq="B")

    equity_returns = pd.Series(
        [0.10, -0.02, 0.03, 0.04],
        index=index,
    )

    risk_free_returns = pd.Series(
        [0.001, 0.001, 0.001, 0.001],
        index=index,
    )

    equity_weight = pd.Series(
        [pd.NA, 1.0, 0.0, 1.0],
        index=index,
        dtype="Float64",
    )

    result = calculate_gross_return(
        equity_returns,
        risk_free_returns,
        equity_weight,
    )

    expected = pd.Series(
        [pd.NA, -0.02, 0.001, 0.04],
        index=index,
        dtype="Float64",
        name="gross_return",
    )

    pd.testing.assert_series_equal(result, expected)


def test_net_return_deducts_transaction_cost():
    gross_return = pd.Series(
        [pd.NA, 0.02, -0.01, 0.03],
        dtype="Float64",
        name="gross_return",
    )

    transaction_cost = pd.Series(
        [pd.NA, 0.0, 0.001, 0.0],
        dtype="Float64",
        name="transaction_cost",
    )

    result = calculate_net_return(
        gross_return,
        transaction_cost,
    )

    expected = pd.Series(
        [pd.NA, 0.02, -0.011, 0.03],
        dtype="Float64",
        name="net_return",
    )

    pd.testing.assert_series_equal(result, expected)


# ---------------------------------------------------------------------------
# End-to-end backtest
# ---------------------------------------------------------------------------


def test_run_zero_one_backtest_end_to_end():
    index = pd.date_range("2020-01-01", periods=5, freq="B")

    states = pd.Series(
        [0, 0, 1, 1, 0],
        index=index,
        dtype="Int64",
    )

    equity_returns = pd.Series(
        [0.01, 0.02, 0.03, -0.04, 0.05],
        index=index,
    )

    risk_free_returns = pd.Series(
        [0.001] * 5,
        index=index,
    )

    result = run_zero_one_backtest(
        equity_returns,
        risk_free_returns,
        states,
        delay=2,
        cost_rate=0.001,
    )

    expected_weight = pd.Series(
        [pd.NA, pd.NA, 1.0, 1.0, 0.0],
        index=index,
        dtype="Float64",
        name="equity_weight",
    )

    pd.testing.assert_series_equal(
        result["equity_weight"],
        expected_weight,
    )

    assert result.loc[index[4], "turnover"] == 1.0
    assert result.loc[index[4], "transaction_cost"] == 0.001
    assert result.loc[index[4], "gross_return"] == 0.001
    assert result.loc[index[4], "net_return"] == 0.0
