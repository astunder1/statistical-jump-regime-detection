"""Plot the final S&P 500 replication results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from regimejump.backtest import run_zero_one_backtest
from regimejump.metrics import cumulative_wealth
from regimejump.risk_free import risk_free_returns_for_dates

TEST_START = "1990-01-01"
TEST_END = "2023-12-31"


def main() -> None:
    regime_data = pd.read_csv(
        "data/spx_selected_states.csv",
        parse_dates=["date"],
        index_col="date",
    )

    treasury = pd.read_parquet("data/us_tbill_3m.parquet")

    equity_returns = np.expm1(regime_data["return"])

    risk_free_returns = risk_free_returns_for_dates(
        treasury["annual_discount_yield_pct"],
        regime_data.index,
    )

    backtest = run_zero_one_backtest(
        equity_returns=equity_returns,
        risk_free_returns=risk_free_returns,
        states=regime_data["state"],
        delay=2,
        cost_rate=0.001,
    )

    backtest = backtest.dropna(subset=["net_return"]).loc[TEST_START:TEST_END]

    benchmark_returns = equity_returns.loc[backtest.index]

    strategy_wealth = cumulative_wealth(backtest["net_return"]).astype(float)

    benchmark_wealth = cumulative_wealth(benchmark_returns).astype(float)

    print(
        "Plot period:",
        backtest.index[0].date(),
        "to",
        backtest.index[-1].date(),
    )
    print(
        "Final wealth:",
        f"strategy={strategy_wealth.iloc[-1]:.2f},",
        f"benchmark={benchmark_wealth.iloc[-1]:.2f}",
    )

    bear_allocation = backtest["equity_weight"] == 0.0

    fig, ax = plt.subplots(
        figsize=(12, 6),
    )

    ax.fill_between(
        backtest.index,
        0,
        1,
        where=bear_allocation,
        transform=ax.get_xaxis_transform(),
        color="lightcoral",
        alpha=0.25,
        step="post",
        label="Bear allocation",
    )

    ax.plot(
        strategy_wealth.index,
        strategy_wealth,
        label="Jump-model strategy",
        color="tab:blue",
        linewidth=1.5,
    )

    ax.plot(
        benchmark_wealth.index,
        benchmark_wealth,
        label="S&P 500 buy-and-hold",
        color="black",
        linewidth=1.2,
        alpha=0.8,
    )

    ax.set_yscale("log")

    ax.set_title("S&P 500 Jump-Model Strategy, 1990–2023")

    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative wealth (log scale)")

    ax.grid(
        True,
        which="both",
        alpha=0.25,
    )

    ax.legend(
        loc="upper left",
        frameon=True,
    )

    fig.tight_layout()

    output = Path("results/figures/spx_wealth.png")

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved figure to {output}")

    # To replicate the plot in the paper, using cumulative excess returns
    strategy_excess_return = backtest["net_return"] - risk_free_returns.loc[backtest.index]

    benchmark_excess_return = benchmark_returns - risk_free_returns.loc[backtest.index]

    strategy_cumulative_excess = strategy_excess_return.cumsum()

    benchmark_cumulative_excess = benchmark_excess_return.cumsum()

    bear_share = bear_allocation.mean()

    regime_shifts = int(backtest["equity_weight"].diff().abs().fillna(0.0).sum())

    fig, ax = plt.subplots(
        figsize=(12, 5),
    )

    ax.fill_between(
        backtest.index,
        0,
        1,
        where=bear_allocation,
        transform=ax.get_xaxis_transform(),
        color="lightcoral",
        alpha=0.40,
        step="post",
        label="Bear regime",
    )

    ax.plot(
        benchmark_cumulative_excess.index,
        benchmark_cumulative_excess,
        color="tab:blue",
        linewidth=1.0,
        label="S&P 500",
    )

    ax.plot(
        strategy_cumulative_excess.index,
        strategy_cumulative_excess,
        color="tab:orange",
        linewidth=1.0,
        label="JM strategy",
    )

    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    ax.set_title(
        "S&P 500 Online Jump-Model Regimes "
        f"(bear share: {bear_share:.1%}, "
        f"regime shifts: {regime_shifts})"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative excess return")
    ax.grid(alpha=0.20)
    ax.legend(loc="upper left")

    fig.tight_layout()

    comparison_output = Path("results/figures/spx_paper_comparison.png")

    fig.savefig(
        comparison_output,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved paper comparison to {comparison_output}")


if __name__ == "__main__":
    main()
