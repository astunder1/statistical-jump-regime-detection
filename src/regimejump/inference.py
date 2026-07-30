"""Scheduled model refitting and online regime inference."""

from __future__ import annotations

import numpy as np
import pandas as pd

from regimejump.jump import JumpModel
from regimejump.preprocessing import standardize_from_training_window


def six_month_refit_states(
    features: pd.DataFrame,
    returns: pd.Series,
    jump_penalty: float,
    training_length: int = 3000,
    n_init: int = 10,
    random_state: int | None = 0,
    verbose: bool = False,
) -> tuple[pd.Series, list[pd.Timestamp]]:
    """Generate causal states with model refitting every six months.

    At each refit, the function uses the latest ``training_length``
    observations to fit the standardization parameters, centroids, and
    state ordering. These fitted values remain fixed for the following
    six-month block. Daily states are inferred using dynamic programming
    over the trailing feature window.

    Parameters
    ----------
    features:
        Raw feature matrix indexed by unique, increasing dates.
    returns:
        Return series aligned with ``features`` and used to order the
        fitted states from bull to bear.
    jump_penalty:
        Nonnegative penalty applied to each state change.
    training_length:
        Number of observations in each training and inference window.
    n_init:
        Number of model initializations at each refit.
    random_state:
        Seed used for reproducible model initialization.
    verbose:
        Whether to print refit progress.

    Returns
    -------
    states:
        Nullable integer state series. Observations before the first
        complete training window remain missing.
    refit_dates:
        Dates on which the model was refitted.
    """
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")

    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series")

    if not isinstance(features.index, pd.DatetimeIndex):
        raise TypeError("features must have a DatetimeIndex")

    if not features.index.equals(returns.index):
        raise ValueError("features and returns must have identical indexes")

    if not features.index.is_monotonic_increasing:
        raise ValueError("features must be sorted by date")

    if not features.index.is_unique:
        raise ValueError("feature dates must be unique")

    if not isinstance(training_length, int) or training_length < 1:
        raise ValueError("training_length must be a positive integer")

    if len(features) < training_length:
        raise ValueError("not enough observations for the training window")

    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("features must contain only finite values")

    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("returns must contain only finite values")

    states = pd.Series(
        pd.NA,
        index=features.index,
        dtype="Int64",
        name="state",
    )
    refit_dates: list[pd.Timestamp] = []

    n_obs = len(features)
    refit_position = training_length - 1

    refit_number = 0

    while refit_position < n_obs:
        refit_number += 1
        refit_date = features.index[refit_position]
        refit_dates.append(refit_date)

        if verbose:
            print(
                f"[refit {refit_number}] fitting model at {refit_date.date()}...",
                flush=True,
            )

        training_start = refit_position - training_length + 1
        training = features.iloc[training_start : refit_position + 1]

        processed_training = standardize_from_training_window(
            training,
            training,
        )

        model = JumpModel(
            n_states=2,
            jump_penalty=jump_penalty,
            n_init=n_init,
            random_state=random_state,
        ).fit(processed_training.to_numpy())

        model.relabel_by_cumulative_return(
            returns.iloc[training_start : refit_position + 1].to_numpy()
        )

        next_refit_date = refit_date + pd.DateOffset(months=6)
        next_refit_position = int(
            features.index.searchsorted(
                next_refit_date,
                side="left",
            )
        )

        block_stop = min(next_refit_position, n_obs)

        for position in range(refit_position, block_stop):
            window_start = position - training_length + 1

            raw_window = features.iloc[window_start : position + 1]

            processed_window = standardize_from_training_window(
                training,
                raw_window,
            )

            path, _ = model._dp_state_path(
                processed_window.to_numpy(),
                model.centroids_,
            )

            states.iloc[position] = path[-1]

        if verbose and block_stop > refit_position:
            completed_date = features.index[block_stop - 1]

            print(
                f"[refit {refit_number}] completed signals through {completed_date.date()}",
                flush=True,
            )

        if next_refit_position >= n_obs:
            break

        refit_position = next_refit_position

    return states, refit_dates
