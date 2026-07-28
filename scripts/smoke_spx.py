"""Exploratory SPX regime inference around the 2020 crash."""

from __future__ import annotations

import numpy as np
import pandas as pd

from regimejump.features import compute_paper_features
from regimejump.jump import JumpModel
from regimejump.online import rolling_dp_online_path
from regimejump.preprocessing import standardize_from_training_window

TRAIN_END = "2019-12-31"
TEST_END = "2020-06-30"
TRAIN_LENGTH = 3000
JUMP_PENALTY = 50.0


def main() -> None:
    prices = pd.read_parquet("data/spx.parquet")["Adj Close"]
    prices = prices.loc[:TEST_END]

    # Exploratory proxy: index log returns, without risk-free adjustment.
    returns = np.log(prices).diff().dropna()
    returns.name = "return"

    features = compute_paper_features(returns)
    features = features.replace([np.inf, -np.inf], np.nan).dropna()

    training = features.loc[:TRAIN_END].tail(TRAIN_LENGTH)

    if len(training) != TRAIN_LENGTH:
        raise RuntimeError(
            f"Expected {TRAIN_LENGTH} training rows, found {len(training)}"
        )

    # Include training history so each test date has a trailing DP window.
    inference_features = features.loc[training.index[0] : TEST_END]

    processed = standardize_from_training_window(
        training,
        inference_features,
    )
    processed_training = processed.loc[training.index]

    model = JumpModel(
        n_states=2,
        jump_penalty=JUMP_PENALTY,
        n_init=10,
        random_state=0,
    ).fit(processed_training.to_numpy())

    model.relabel_by_cumulative_return(
        returns.loc[training.index].to_numpy()
    )

    states = rolling_dp_online_path(
        processed.to_numpy(),
        model.centroids_,
        jump_penalty=JUMP_PENALTY,
        lookback=TRAIN_LENGTH,
    )

    states = pd.Series(states, index=processed.index, name="state")
    test_states = states.loc[states.index > TRAIN_END]

    switches = test_states[
        test_states.ne(test_states.shift())
    ]

    print("Training period:")
    print(training.index[0].date(), "to", training.index[-1].date())

    print("\nCentroids:")
    print(pd.DataFrame(
        model.centroids_,
        index=["bull", "bear"],
        columns=training.columns,
    ))

    print("\nTest-state counts:")
    print(test_states.value_counts().sort_index())

    print("\nRegime changes:")
    print(switches)


if __name__ == "__main__":
    main()