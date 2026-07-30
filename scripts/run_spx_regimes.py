"""Generate exploratory SPX regimes with six-month model refitting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from regimejump.features import compute_paper_features
from regimejump.inference import six_month_refit_states
from regimejump.risk_free import risk_free_returns_for_dates

END_DATE = "2023-12-31"
TRAINING_LENGTH = 3000
JUMP_PENALTY = 50.0


def main() -> None:
    price_index = pd.read_parquet("data/spx.parquet")["Adj Close"].loc[:END_DATE]

    total_return_index = pd.read_parquet("data/spx_tr.parquet")["Adj Close"].loc[:END_DATE]

    price_returns = np.log(price_index).diff()
    total_returns = np.log(total_return_index).diff()

    first_total_return_date = total_returns.first_valid_index()

    returns = pd.concat(
        [
            price_returns.loc[price_returns.index < first_total_return_date],
            total_returns.loc[first_total_return_date:],
        ]
    )

    returns = returns.sort_index()
    returns.name = "return"

    treasury_data = pd.read_parquet("data/us_tbill_3m.parquet")

    risk_free_returns = risk_free_returns_for_dates(
        treasury_data["annual_discount_yield_pct"],
        returns.index,
    )

    # Convert equity log returns to simple returns before subtracting
    # the simple daily risk-free return.
    equity_simple_returns = np.expm1(returns)

    excess_returns = equity_simple_returns - risk_free_returns
    excess_returns.name = "excess_return"

    features = compute_paper_features(excess_returns)
    features = features.replace([np.inf, -np.inf], np.nan).dropna()

    # Ensure returns and features cover exactly the same dates.
    returns = returns.loc[features.index]
    excess_returns = excess_returns.loc[features.index]

    print(f"Valid observations: {len(features)}")
    print(f"Date range: {features.index[0].date()} to {features.index[-1].date()}")
    print("Running six-month refitting...")

    states, refit_dates = six_month_refit_states(
        features=features,
        returns=excess_returns,
        jump_penalty=JUMP_PENALTY,
        training_length=TRAINING_LENGTH,
        n_init=10,
        random_state=0,
        verbose=True,
    )

    results = features.copy()
    results.insert(0, "return", returns)
    results["state"] = states

    valid_states = states.dropna().astype(int)
    switches = valid_states[valid_states.ne(valid_states.shift())]

    output = Path("data/spx_excess_feature_states_lambda50.csv")
    results.to_csv(output)

    print(f"\nFirst signal: {valid_states.index[0].date()}")
    print(f"Refits: {len(refit_dates)}")
    print(f"Regime switches: {max(0, len(switches) - 1)}")
    print(f"Bear-state share: {(valid_states == 1).mean():.1%}")

    print("\nMost recent regime changes:")
    print(switches.tail(15))

    print(f"\nSaved results to {output}")


if __name__ == "__main__":
    main()
