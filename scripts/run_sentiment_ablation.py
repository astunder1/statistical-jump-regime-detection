"""Compare price and sentiment jump models across fixed penalties."""

from __future__ import annotations

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
from regimejump.selection import (
    PAPER_JUMP_PENALTIES,
    generate_candidate_state_paths,
)

DATA_PATH = Path("data/spx_sentiment_features.csv")
OUTPUT_PATH = Path("results/tables/spx_sentiment_ablations.csv")

TRAINING_LENGTH = 3000
DELAY = 2
COST_RATE = 0.001
N_INIT = 10
RANDOM_STATE = 0

PRICE_FEATURES = (
    "downside_dev_10",
    "sortino_20",
    "sortino_60",
)

SENTIMENT_FEATURES = (
    "macro_sentiment",
    "sentiment_momentum",
    "abnormal_news_attention",
)


def main() -> None:
    data = pd.read_csv(
        DATA_PATH,
        parse_dates=["date"],
        index_col="date",
    )

    equity_returns = data["equity_return"]
    risk_free_returns = data["risk_free_return"]
    model_returns = data["excess_return"]

    feature_sets = {
        "price_only": data[list(PRICE_FEATURES)],
        "sentiment_only": data[list(SENTIMENT_FEATURES)],
        "price_plus_level": data[
            [
                *PRICE_FEATURES,
                "macro_sentiment",
            ]
        ],
        "price_plus_momentum": data[
            [
                *PRICE_FEATURES,
                "sentiment_momentum",
            ]
        ],
        "price_plus_attention": data[
            [
                *PRICE_FEATURES,
                "abnormal_news_attention",
            ]
        ],
        "combined": data[
            [
                *PRICE_FEATURES,
                *SENTIMENT_FEATURES,
            ]
        ],
    }

    state_paths: dict[
        tuple[str, float],
        pd.Series,
    ] = {}

    for model_name, features in feature_sets.items():
        print(f"\nGenerating paths for {model_name}...")

        model_paths = generate_candidate_state_paths(
            features=features,
            model_returns=model_returns,
            candidates=PAPER_JUMP_PENALTIES,
            training_length=TRAINING_LENGTH,
            n_init=N_INIT,
            random_state=RANDOM_STATE,
            verbose=True,
        )

        for penalty, path in model_paths.items():
            state_paths[
                (
                    model_name,
                    float(penalty),
                )
            ] = path

    scenario_backtests: dict[
        tuple[str, float],
        pd.DataFrame,
    ] = {}

    for key, states in state_paths.items():
        backtest = run_zero_one_backtest(
            equity_returns=equity_returns,
            risk_free_returns=risk_free_returns,
            states=states,
            delay=DELAY,
            cost_rate=COST_RATE,
        )

        scenario_backtests[key] = backtest.dropna(subset=["net_return"])

    # Evaluate every model and penalty over identical dates.
    common_index: pd.Index | None = None

    for backtest in scenario_backtests.values():
        if common_index is None:
            common_index = backtest.index
        else:
            common_index = common_index.intersection(
                backtest.index,
            )

    if common_index is None or common_index.empty:
        raise RuntimeError("model scenarios have no common observations")

    rows: list[dict[str, str | float | int]] = []

    for (
        model_name,
        penalty,
    ), backtest in scenario_backtests.items():
        backtest = backtest.loc[common_index]

        net_returns = backtest["net_return"]
        wealth = cumulative_wealth(net_returns)

        years = len(backtest) / 252.0

        rows.append(
            {
                "model": model_name,
                "jump_penalty": penalty,
                "test_start": (common_index[0].date().isoformat()),
                "test_end": (common_index[-1].date().isoformat()),
                "observations": len(backtest),
                "annualized_return": annualized_return(net_returns),
                "annualized_volatility": (annualized_volatility(net_returns)),
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
                "allocation_changes": int(backtest["turnover"].sum()),
            }
        )

    benchmark_returns = equity_returns.loc[common_index]
    benchmark_wealth = cumulative_wealth(benchmark_returns)

    rows.append(
        {
            "model": "buy_and_hold",
            "jump_penalty": np.nan,
            "test_start": (common_index[0].date().isoformat()),
            "test_end": (common_index[-1].date().isoformat()),
            "observations": len(common_index),
            "annualized_return": annualized_return(benchmark_returns),
            "annualized_volatility": (annualized_volatility(benchmark_returns)),
            "sharpe": sharpe_ratio(
                benchmark_returns,
                risk_free_returns.loc[common_index],
            ),
            "maximum_drawdown": maximum_drawdown(
                benchmark_wealth,
                initial_wealth=1.0,
            ),
            "annualized_turnover": 0.0,
            "equity_exposure": 1.0,
            "allocation_changes": 0,
        }
    )

    results = (
        pd.DataFrame(rows)
        .sort_values(
            [
                "model",
                "jump_penalty",
            ]
        )
        .reset_index(drop=True)
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.round(6).to_csv(
        OUTPUT_PATH,
        index=False,
    )

    display = results[
        [
            "model",
            "jump_penalty",
            "annualized_return",
            "annualized_volatility",
            "sharpe",
            "maximum_drawdown",
            "annualized_turnover",
            "equity_exposure",
            "allocation_changes",
        ]
    ].copy()

    display["jump_penalty"] = display["jump_penalty"].map(
        lambda value: "-" if pd.isna(value) else f"{value:g}"
    )

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

    print("\nFixed-penalty comparison:")
    print(f"Period: {common_index[0].date()} to {common_index[-1].date()}")
    print(f"Observations: {len(common_index)}")
    print()
    print(display.to_string(index=False))
    print(f"\nSaved results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
