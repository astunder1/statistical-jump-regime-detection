"""Generate exploratory SPX regimes with six-month model refitting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from regimejump.features import compute_paper_features
from regimejump.inference import six_month_refit_states

END_DATE = "2023-12-31"
TRAINING_LENGTH = 3000
JUMP_PENALTY = 50.0


def main() -> None:
    prices = pd.read_parquet("data/spx.parquet")["Adj Close"]
    prices = prices.loc[:END_DATE]

    # Public-data approximation: index log returns without risk-free adjustment.
    returns = np.log(prices).diff()
    returns.name = "return"

    features = compute_paper_features(returns)
    features = features.replace([np.inf, -np.inf], np.nan).dropna()

    # Ensure returns and features cover exactly the same dates.
    returns = returns.loc[features.index]

    print(f"Valid observations: {len(features)}")
    print(f"Date range: {features.index[0].date()} to {features.index[-1].date()}")
    print("Running six-month refitting...")

    states, refit_dates = six_month_refit_states(
        features=features,
        returns=returns,
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
    switches = valid_states[
        valid_states.ne(valid_states.shift())
    ]

    output = Path("data/spx_states_lambda50.csv")
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