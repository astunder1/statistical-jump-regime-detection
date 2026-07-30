"""Generate SPX states with monthly jump-penalty selection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from regimejump.risk_free import risk_free_returns_for_dates
from regimejump.selection import monthly_selected_state_path

TEST_START = "1990-01-01"
TEST_END = "2023-12-31"


def main() -> None:
    data = pd.read_csv(
        "data/spx_excess_feature_states_lambda50.csv",
        parse_dates=["date"],
        index_col="date",
    )

    feature_columns = [
        "downside_dev_10",
        "sortino_20",
        "sortino_60",
    ]

    features = data[feature_columns]

    equity_returns = np.expm1(data["return"])
    equity_returns.name = "equity_return"

    treasury = pd.read_parquet("data/us_tbill_3m.parquet")

    risk_free_returns = risk_free_returns_for_dates(
        treasury["annual_discount_yield_pct"],
        data.index,
    )

    model_returns = equity_returns - risk_free_returns
    model_returns.name = "excess_return"

    states, selected_penalties, score_table = monthly_selected_state_path(
        features=features,
        model_returns=model_returns,
        equity_returns=equity_returns,
        risk_free_returns=risk_free_returns,
        test_start=TEST_START,
        test_end=TEST_END,
        training_length=3000,
        validation_years=8,
        delay=2,
        cost_rate=0.001,
        n_init=10,
        random_state=0,
        verbose=True,
    )

    results = data.loc[states.index, feature_columns].copy()
    results.insert(
        0,
        "return",
        data.loc[states.index, "return"],
    )
    results["state"] = states

    output_dir = Path("data")

    states_path = output_dir / "spx_selected_states.csv"
    penalties_path = output_dir / "spx_monthly_penalties.csv"
    scores_path = output_dir / "spx_penalty_sharpes.csv"

    results.to_csv(states_path)
    selected_penalties.to_csv(penalties_path)
    score_table.to_csv(scores_path)

    print("\nSelected-penalty frequencies:")
    print(selected_penalties.value_counts().sort_index())

    print(f"\nSaved states to {states_path}")
    print(f"Saved penalties to {penalties_path}")
    print(f"Saved scores to {scores_path}")


if __name__ == "__main__":
    main()
