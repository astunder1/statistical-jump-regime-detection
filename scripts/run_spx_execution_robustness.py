"""Test SPX strategy robustness to execution delay and transaction costs."""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from regimejump.backtest import run_zero_one_backtest
from regimejump.metrics import (
    annualized_return,
    annualized_volatility,
    cumulative_wealth,
    maximum_drawdown,
    sharpe_ratio,
)
from regimejump.risk_free import risk_free_returns_for_dates

TEST_START = "1990-01-01"
TEST_END = "2023-12-31"

DELAYS = (1, 2, 3)
COST_RATES = (0.0, 0.001, 0.0025)


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

    # The stored return series contains log returns, whereas the
    # backtest operates on simple returns.
    equity_returns = np.expm1(regime_data["return"])
    equity_returns.name = "equity_return"

    risk_free_returns = risk_free_returns_for_dates(
        treasury_data["annual_discount_yield_pct"],
        regime_data.index,
    )

    scenario_backtests: dict[tuple[int, float], pd.DataFrame] = {}

    for delay, cost_rate in product(DELAYS, COST_RATES):
        backtest = run_zero_one_backtest(
            equity_returns=equity_returns,
            risk_free_returns=risk_free_returns,
            states=states,
            delay=delay,
            cost_rate=cost_rate,
        )

        backtest = backtest.loc[TEST_START:TEST_END]
        backtest = backtest.dropna(subset=["net_return"])

        scenario_backtests[(delay, cost_rate)] = backtest

    # Use exactly the same evaluation dates for every scenario. A longer
    # delay can otherwise remove an additional observation at the start.
    common_index: pd.Index | None = None

    for backtest in scenario_backtests.values():
        if common_index is None:
            common_index = backtest.index
        else:
            common_index = common_index.intersection(
                backtest.index,
            )

    if common_index is None or common_index.empty:
        raise RuntimeError("execution scenarios have no common observations")

    rows: list[dict[str, float | int]] = []

    for (delay, cost_rate), backtest in scenario_backtests.items():
        backtest = backtest.loc[common_index]

        net_returns = backtest["net_return"]
        wealth = cumulative_wealth(net_returns)

        years = len(backtest) / 252.0
        allocation_changes = int(backtest["turnover"].sum())

        rows.append(
            {
                "delay": delay,
                "cost_bps": cost_rate * 10_000,
                "observations": len(backtest),
                "annualized_return": annualized_return(net_returns),
                "annualized_volatility": annualized_volatility(
                    net_returns,
                ),
                "sharpe": sharpe_ratio(
                    net_returns,
                    risk_free_returns.loc[common_index],
                ),
                "maximum_drawdown": maximum_drawdown(
                    wealth,
                    initial_wealth=1.0,
                ),
                "annualized_turnover": (backtest["turnover"].sum() / (2.0 * years)),
                "equity_exposure": backtest["equity_weight"].mean(),
                "allocation_changes": allocation_changes,
            }
        )

    results = (
        pd.DataFrame(rows)
        .sort_values(
            ["delay", "cost_bps"],
        )
        .reset_index(drop=True)
    )

    output = Path("results/tables/spx_execution_robustness.csv")
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.round(6).to_csv(
        output,
        index=False,
    )

    display = results.copy()

    percentage_columns = [
        "annualized_return",
        "annualized_volatility",
        "maximum_drawdown",
        "annualized_turnover",
        "equity_exposure",
    ]

    for column in percentage_columns:
        display[column] = display[column].map(lambda value: f"{value:.2%}")

    display["sharpe"] = display["sharpe"].map(lambda value: f"{value:.2f}")

    display["cost_bps"] = display["cost_bps"].map(lambda value: f"{value:.0f}")

    print("SPX execution robustness:")
    print(f"Period: {common_index[0].date()} to {common_index[-1].date()}")
    print(f"Common observations: {len(common_index)}")
    print()
    print(display.to_string(index=False))
    print(f"\nSaved results to {output}")


if __name__ == "__main__":
    main()
