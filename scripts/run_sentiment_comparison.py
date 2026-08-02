"""Compare frozen price and sentiment models for a selected market."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from regimejump.backtest import run_zero_one_backtest
from regimejump.inference import six_month_refit_states
from regimejump.metrics import (
    annualized_return,
    annualized_volatility,
    cumulative_wealth,
    maximum_drawdown,
    sharpe_ratio,
)

PRICE_FEATURES = (
    "downside_dev_10",
    "sortino_20",
    "sortino_60",
)

# These settings are frozen from the S&P 500 experiment.
TRAINING_LENGTH = 3000
JUMP_PENALTY = 15.0
DELAY = 2
COST_RATE = 0.001
N_INIT = 10
RANDOM_STATE = 0

MOMENTUM_FEATURES = (
    *PRICE_FEATURES,
    "sentiment_momentum",
)


def parse_args() -> argparse.Namespace:
    """Parse the selected market."""
    parser = argparse.ArgumentParser(
        description=("Compare frozen price-only and sentiment-momentum jump models.")
    )

    parser.add_argument(
        "--market",
        choices=["spx", "nasdaq100"],
        required=True,
    )

    return parser.parse_args()


def summarize_strategy(
    returns: pd.Series,
    risk_free_returns: pd.Series,
    equity_weight: pd.Series,
    turnover: pd.Series,
) -> dict[str, float | int]:
    """Calculate performance and allocation diagnostics."""
    wealth = cumulative_wealth(returns)
    years = len(returns) / 252.0

    return {
        "observations": len(returns),
        "annualized_return": annualized_return(returns),
        "annualized_volatility": annualized_volatility(returns),
        "sharpe": sharpe_ratio(
            returns,
            risk_free_returns,
        ),
        "maximum_drawdown": maximum_drawdown(
            wealth,
            initial_wealth=1.0,
        ),
        "annualized_turnover": (turnover.sum() / (2.0 * years)),
        "equity_exposure": equity_weight.mean(),
        "allocation_changes": int(turnover.sum()),
    }


def main() -> None:
    args = parse_args()

    data_path = Path(f"data/{args.market}_sentiment_features.csv")

    table_output = Path(f"results/tables/{args.market}_sentiment_comparison.csv")

    daily_output = Path(f"data/{args.market}_sentiment_comparison.csv")

    data = pd.read_csv(
        data_path,
        parse_dates=["date"],
        index_col="date",
    )

    print("Common sample:")
    print(
        data.index[0].date(),
        "to",
        data.index[-1].date(),
    )
    print(f"Observations: {len(data)}")
    print(
        "First possible signal:",
        data.index[TRAINING_LENGTH - 1].date(),
    )

    equity_returns = data["equity_return"]
    risk_free_returns = data["risk_free_return"]
    excess_returns = data["excess_return"]

    feature_sets = {
        "price_only": data[list(PRICE_FEATURES)],
        "price_plus_momentum": data[list(MOMENTUM_FEATURES)],
    }

    states: dict[str, pd.Series] = {}
    backtests: dict[str, pd.DataFrame] = {}

    for model_name, features in feature_sets.items():
        print(f"\nGenerating {model_name} states...")

        model_states, refit_dates = six_month_refit_states(
            features=features,
            returns=excess_returns,
            jump_penalty=JUMP_PENALTY,
            training_length=TRAINING_LENGTH,
            n_init=N_INIT,
            random_state=RANDOM_STATE,
            verbose=True,
        )

        states[model_name] = model_states

        backtests[model_name] = run_zero_one_backtest(
            equity_returns=equity_returns,
            risk_free_returns=risk_free_returns,
            states=model_states,
            delay=DELAY,
            cost_rate=COST_RATE,
        ).dropna(subset=["net_return"])

        print(
            f"{model_name} refits:",
            len(refit_dates),
        )

    common_index = backtests["price_only"].index.intersection(
        backtests["price_plus_momentum"].index
    )

    if common_index.empty:
        raise RuntimeError("the two strategies have no common observations")

    for model_name in backtests:
        backtests[model_name] = backtests[model_name].loc[common_index]

    periods = {
        "full_sample": (
            common_index[0],
            common_index[-1],
        ),
        "pre_covid": (
            common_index[0],
            pd.Timestamp("2019-12-31"),
        ),
        "covid_and_after": (
            pd.Timestamp("2020-01-01"),
            common_index[-1],
        ),
    }

    rows: list[dict[str, str | float | int]] = []

    for period_name, (
        period_start,
        period_end,
    ) in periods.items():
        period_index = common_index[(common_index >= period_start) & (common_index <= period_end)]

        if period_index.empty:
            continue

        period_risk_free = risk_free_returns.loc[period_index]

        for model_name, backtest in backtests.items():
            summary = summarize_strategy(
                returns=backtest.loc[
                    period_index,
                    "net_return",
                ],
                risk_free_returns=period_risk_free,
                equity_weight=backtest.loc[
                    period_index,
                    "equity_weight",
                ],
                turnover=backtest.loc[
                    period_index,
                    "turnover",
                ],
            )

            rows.append(
                {
                    "period": period_name,
                    "model": model_name,
                    **summary,
                }
            )

        benchmark_summary = summarize_strategy(
            returns=equity_returns.loc[period_index],
            risk_free_returns=period_risk_free,
            equity_weight=pd.Series(
                1.0,
                index=period_index,
            ),
            turnover=pd.Series(
                0.0,
                index=period_index,
            ),
        )

        rows.append(
            {
                "period": period_name,
                "model": "buy_and_hold",
                **benchmark_summary,
            }
        )

    results = pd.DataFrame(rows)

    daily = pd.DataFrame(
        {
            "equity_return": equity_returns.loc[common_index],
            "risk_free_return": risk_free_returns.loc[common_index],
            "price_state": states["price_only"].loc[common_index],
            "momentum_state": states["price_plus_momentum"].loc[common_index],
            "price_weight": backtests["price_only"]["equity_weight"],
            "momentum_weight": backtests["price_plus_momentum"]["equity_weight"],
            "price_net_return": backtests["price_only"]["net_return"],
            "momentum_net_return": backtests["price_plus_momentum"]["net_return"],
        }
    )

    signal_disagreement = daily["price_state"] != daily["momentum_state"]

    allocation_disagreement = daily["price_weight"] != daily["momentum_weight"]

    print("\nModel disagreement:")
    print(
        "Signal disagreement:",
        f"{signal_disagreement.mean():.2%}",
        f"({signal_disagreement.sum()} days)",
    )
    print(
        "Allocation disagreement:",
        f"{allocation_disagreement.mean():.2%}",
        f"({allocation_disagreement.sum()} days)",
    )

    for model_name, state_series in states.items():
        valid_states = state_series.loc[common_index].astype(int)

        changes = valid_states[valid_states.ne(valid_states.shift())]

        print(f"\n{model_name} regime changes:")
        print(changes)

        table_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        daily_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    results.round(6).to_csv(
        table_output,
        index=False,
    )

    daily.to_csv(daily_output)

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

    print("\nSubperiod performance:")
    print(display.to_string(index=False))

    print(f"\nSaved table to {table_output}")
    print(f"Saved daily diagnostics to {daily_output}")


if __name__ == "__main__":
    main()
