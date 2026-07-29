
"""Backtest the SPX zero-one regime strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from regimejump.risk_free import risk_free_returns_for_dates
from regimejump.backtest import run_zero_one_backtest

from regimejump.metrics import (
    annualized_return,
    annualized_volatility,
    cumulative_wealth,
    maximum_drawdown,
    sharpe_ratio,
)

TEST_START = "1990-01-01"
TEST_END = "2023-12-31"


def main() -> None:
    regime_data = pd.read_csv(
        "data/spx_selected_states.csv",
        parse_dates=["date"],
        index_col="date",
    )

    treasury_data = pd.read_parquet(
        "data/us_tbill_3m.parquet",
    )

    states = regime_data["state"]

    # The regime script saved log returns, whereas the backtest expects
    # simple returns.
    equity_returns = np.expm1(regime_data["return"])
    equity_returns.name = "equity_return"

    risk_free_returns = risk_free_returns_for_dates(
        treasury_data["annual_discount_yield_pct"],
        regime_data.index,
    )

    backtest = run_zero_one_backtest(
        equity_returns=equity_returns,
        risk_free_returns=risk_free_returns,
        states=states,
        delay=2,
        cost_rate=0.001,
    )

    backtest = backtest.dropna(subset=["net_return"])

    backtest = backtest.loc[TEST_START:TEST_END]

    print(backtest.head())
    print()
    print(f"Backtest start: {backtest.index[0].date()}")
    print(f"Backtest end:   {backtest.index[-1].date()}")
    print(f"Observations:   {len(backtest)}")

    wealth = cumulative_wealth(backtest["net_return"])

    ann_return = annualized_return(backtest["net_return"])
    ann_volatility = annualized_volatility(backtest["net_return"])
    sharpe = sharpe_ratio(
        backtest["net_return"],
        risk_free_returns.loc[backtest.index],
    )
    max_drawdown = maximum_drawdown(wealth)

    print("\nPerformance:")
    print(f"Annualized return:     {ann_return:.2%}")
    print(f"Annualized volatility: {ann_volatility:.2%}")
    print(f"Sharpe ratio:          {sharpe:.2f}")
    print(f"Maximum drawdown:      {max_drawdown:.2%}")
    print(f"Final wealth:          {wealth.dropna().iloc[-1]:.2f}")

    benchmark_returns = equity_returns.loc[backtest.index]
    benchmark_wealth = cumulative_wealth(benchmark_returns)

    benchmark_ann_return = annualized_return(benchmark_returns)
    benchmark_ann_volatility = annualized_volatility(benchmark_returns)
    benchmark_sharpe = sharpe_ratio(
        benchmark_returns,
        risk_free_returns.loc[backtest.index],
    )
    benchmark_max_drawdown = maximum_drawdown(benchmark_wealth)

    print("\nBuy-and-hold benchmark:")
    print(f"Annualized return:     {benchmark_ann_return:.2%}")
    print(f"Annualized volatility: {benchmark_ann_volatility:.2%}")
    print(f"Sharpe ratio:          {benchmark_sharpe:.2f}")
    print(f"Maximum drawdown:      {benchmark_max_drawdown:.2%}")
    print(
        f"Final wealth:          "
        f"{benchmark_wealth.dropna().iloc[-1]:.2f}"
    )

    equity_exposure = backtest["equity_weight"].mean()
    number_of_trades = int(backtest["turnover"].sum())
    total_transaction_cost = backtest["transaction_cost"].sum()
    gross_ann_return = annualized_return(backtest["gross_return"])

    print("\nStrategy diagnostics:")
    print(f"Equity exposure:          {equity_exposure:.2%}")
    print(f"Allocation changes:       {number_of_trades}")
    print(f"Gross annualized return:  {gross_ann_return:.2%}")
    print(f"Total transaction costs:  {total_transaction_cost:.2%}")

    strategy_mean_return = backtest["net_return"].mean() * 252
    benchmark_mean_return = benchmark_returns.mean() * 252
    average_risk_free = (
        risk_free_returns.loc[backtest.index].mean() * 252
    )

    print("\nSharpe components:")
    print(f"Strategy arithmetic return:  {strategy_mean_return:.2%}")
    print(f"Benchmark arithmetic return: {benchmark_mean_return:.2%}")
    print(f"Average risk-free return:     {average_risk_free:.2%}")
    print(
        "Strategy excess return:     "
        f"{strategy_mean_return - average_risk_free:.2%}"
    )
    print(
        "Benchmark excess return:    "
        f"{benchmark_mean_return - average_risk_free:.2%}"
    )


if __name__ == "__main__":
    main()