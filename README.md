# regimejump

Statistical jump models for market regime detection. Built as a replication
of Shu, Yu & Kolm (2024), *"Downside risk reduction using regime-switching
signals: a statistical jump model approach"* (Journal of Asset Management),
extended with a strictly online evaluation protocol and a comparison against
a Gaussian Hidden Markov Model.

## Background

A statistical jump model (Nystrup, Lindström & Madsen, 2020) fits discrete
market regimes by penalized k-means over return-based features:

```
sum_t ||x_t - theta_{s_t}||^2 + lambda * sum_{t>=1} 1{s_t != s_{t-1}}
```

The `lambda` penalty discourages rapid state switching, producing far more
persistent regime labels than an unpenalized clustering or a plain HMM
Viterbi path. Fitting alternates:

1. **Centroid step** — given a state path, each regime's centroid is the
   mean of features assigned to it.
2. **State step** — given centroids, the optimal state path is found by a
   dynamic program (Viterbi-style), trading off per-observation fit against
   the switching penalty.

Multiple random restarts (k-means++ initialization) guard against local
optima; empty states are reseeded.

## Features

Daily return series are mapped to EWM statistics at halflives of 5, 10, and
21 trading days:

- EWM mean of returns
- EWM downside deviation (std of `min(r, 0)`)
- EWM mean of `|returns|`

Standardization is **expanding-window** (uses only data available up to
time `t`) for anything reported out of sample. Full-sample z-scoring is
permitted only inside unit tests, never in a backtest or reported result.

## Roadmap

- [x] 1. Repo scaffold
- [ ] 2. `scripts/download_data.py` — SPX/NDX/DAX daily data via Stooq,
      cached as parquet
- [ ] 3. `src/regimejump/features.py` — EWM feature engineering
- [ ] 4. `src/regimejump/online.py` — expanding standardization + one-step
      online regime decision + greedy online path runner
- [ ] 5. `src/regimejump/jump.py` — `JumpModel` (DP recursion and fit loop
      implemented by hand, not generated)
- [ ] 6. `tests/test_jump.py` — correctness tests written before the
      implementation
- [ ] 7. `src/regimejump/cv.py` — expanding-window walk-forward CV, lambda
      tuning
- [ ] 8. Replication backtest of the paper's regime-based SPX de-risking
      strategy, fully online
- [ ] 9. Extension: Gaussian HMM comparison (persistence, identification
      lag, strategy Sharpe / max drawdown)

## Hard constraints

- No look-ahead in any reported out-of-sample number.
- The core package (`regimejump`) depends only on numpy/pandas/scipy.
  `jumpmodels` and `hmmlearn` are used only in tests and in the extension
  comparison, never in the core.

## Development

```bash
pip install -e ".[dev,data,hmm]"
pytest
ruff check src tests
```

## Reference

Shu, M., Yu, W., & Kolm, P. N. (2024). Downside risk reduction using
regime-switching signals: a statistical jump model approach. *Journal of
Asset Management*.

Nystrup, P., Lindström, E., & Madsen, H. (2020). Learning hidden Markov
models with persistent states by penalizing jumps. *Expert Systems with
Applications*.
