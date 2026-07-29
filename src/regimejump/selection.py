"""Jump-penalty selection for rolling out-of-sample inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
from collections.abc import Mapping, Sequence

from regimejump.backtest import run_zero_one_backtest
from regimejump.inference import six_month_refit_states
from regimejump.metrics import sharpe_ratio

PAPER_JUMP_PENALTIES: tuple[float, ...] = (
    0.0,
    5.0,
    15.0,
    35.0,
    70.0,
    150.0,
)


def select_best_penalty(
    sharpe_by_penalty: Mapping[float, float],
) -> float:
    """Select the candidate penalty with the highest validation Sharpe."""

    finite_scores = {
        float(penalty): float(sharpe)
        for penalty, sharpe in sharpe_by_penalty.items()
        if np.isfinite(sharpe)
    }

    if not finite_scores:
        raise ValueError("at least one finite Sharpe ratio is required")

    return max(
        finite_scores,
        key=finite_scores.get,
    )


def generate_candidate_state_paths(
    features: pd.DataFrame,
    model_returns: pd.Series,
    candidates: Sequence[float] = PAPER_JUMP_PENALTIES,
    training_length: int = 3000,
    n_init: int = 10,
    random_state: int | None = 0,
    verbose: bool = False,
) -> dict[float, pd.Series]:
    """Generate one full causal online state path per candidate penalty."""

    paths: dict[float, pd.Series] = {}

    for penalty in candidates:
        penalty = float(penalty)

        if verbose:
            print(f"Generating states for lambda={penalty:.1f}...")

        states, _ = six_month_refit_states(
            features=features,
            returns=model_returns,
            jump_penalty=penalty,
            training_length=training_length,
            n_init=n_init,
            random_state=random_state,
            verbose=verbose,
        )

        paths[penalty] = states

    return paths


def generate_candidate_backtests(
    equity_returns: pd.Series,
    risk_free_returns: pd.Series,
    state_paths: Mapping[float, pd.Series],
    delay: int = 2,
    cost_rate: float = 0.001,
) -> dict[float, pd.DataFrame]:
    """Backtest every fixed-penalty state path once."""

    backtests: dict[float, pd.DataFrame] = {}

    for penalty, states in state_paths.items():
        backtests[float(penalty)] = run_zero_one_backtest(
            equity_returns=equity_returns,
            risk_free_returns=risk_free_returns,
            states=states,
            delay=delay,
            cost_rate=cost_rate,
        )

    return backtests

def monthly_selected_state_path(
    features: pd.DataFrame,
    model_returns: pd.Series,
    equity_returns: pd.Series,
    risk_free_returns: pd.Series,
    test_start: str | pd.Timestamp,
    test_end: str | pd.Timestamp,
    candidates: Sequence[float] = PAPER_JUMP_PENALTIES,
    training_length: int = 3000,
    validation_years: int = 8,
    delay: int = 2,
    cost_rate: float = 0.001,
    n_init: int = 10,
    random_state: int | None = 0,
    verbose: bool = False,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Generate states using monthly trailing-Sharpe penalty selection."""

    state_paths = generate_candidate_state_paths(
        features=features,
        model_returns=model_returns,
        candidates=candidates,
        training_length=training_length,
        n_init=n_init,
        random_state=random_state,
        verbose=verbose,
    )

    candidate_backtests = generate_candidate_backtests(
        equity_returns=equity_returns,
        risk_free_returns=risk_free_returns,
        state_paths=state_paths,
        delay=delay,
        cost_rate=cost_rate,
    )

    test_start = pd.Timestamp(test_start)
    test_end = pd.Timestamp(test_end)

    test_dates = features.index[
        (features.index >= test_start)
        & (features.index <= test_end)
    ]

    if test_dates.empty:
        raise ValueError("test period contains no observations")

    test_start_position = int(
        features.index.searchsorted(test_dates[0])
    )

    pretest_start_position = max(
        0,
        test_start_position - delay,
    )

    pretest_dates = features.index[
        pretest_start_position:test_start_position
    ]

    output_dates = pretest_dates.append(test_dates)

    first_month = test_dates[0].to_period("M")
    last_month = test_dates[-1].to_period("M")

    # Include the preceding month because its selected penalty remains
    # active on the first trading day of the test period.
    selection_months = pd.period_range(
        first_month - 1,
        last_month,
        freq="M",
    )

    score_rows: dict[pd.Timestamp, dict[float, float]] = {}
    selected_values: dict[pd.Timestamp, float] = {}

    for month in selection_months:
        month_start = month.start_time

        end_position = int(
            features.index.searchsorted(
                month_start,
                side="left",
            ) - 1
        )

        if end_position < 0:
            raise ValueError(
                "not enough history before a selection month"
            )

        validation_end = features.index[end_position]
        validation_start = (
            validation_end
            - pd.DateOffset(years=validation_years)
        )

        validation_dates = features.index[
            (features.index >= validation_start)
            & (features.index <= validation_end)
        ]

        scores: dict[float, float] = {}

        for penalty, backtest in candidate_backtests.items():
            scores[float(penalty)] = sharpe_ratio(
                backtest.loc[
                    validation_dates,
                    "net_return",
                ],
                risk_free_returns.loc[validation_dates],
            )

        selected = select_best_penalty(scores)

        score_rows[month_start] = scores
        selected_values[month_start] = selected

        if verbose:
            print(
                f"{month_start.date()}: "
                f"selected lambda={selected:.1f}"
            )

    score_table = pd.DataFrame.from_dict(
        score_rows,
        orient="index",
    ).sort_index()

    score_table.index.name = "month"
    score_table.columns.name = "jump_penalty"

    selected_monthly = pd.Series(
        selected_values,
        name="selected_jump_penalty",
        dtype=float,
    ).sort_index()

    selected_monthly.index.name = "month"

    daily_penalty = pd.Series(
        index=output_dates,
        dtype=float,
        name="selected_jump_penalty",
    )

    initial_penalty = selected_monthly.loc[
        (first_month - 1).start_time
    ]

    daily_penalty.loc[pretest_dates] = initial_penalty

    for month in pd.period_range(
        first_month,
        last_month,
        freq="M",
    ):
        month_dates = test_dates[
            test_dates.to_period("M") == month
        ]

        if month_dates.empty:
            continue

        previous_penalty = selected_monthly.loc[
            (month - 1).start_time
        ]

        current_penalty = selected_monthly.loc[
            month.start_time
        ]

        # A month-end choice is available on the next trading day
        # and becomes applicable from the second trading day.
        daily_penalty.loc[month_dates[0]] = previous_penalty
        daily_penalty.loc[month_dates[1:]] = current_penalty

    selected_states = pd.Series(
        pd.NA,
        index=output_dates,
        dtype="Int64",
        name="state",
    )

    for penalty, path in state_paths.items():
        use_penalty = daily_penalty == float(penalty)

        selected_states.loc[use_penalty] = (
            path.loc[use_penalty.index[use_penalty]]
            .astype("Int64")
        )

    return selected_states, selected_monthly, score_table