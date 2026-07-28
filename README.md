# regimejump

A Python implementation and public-data replication of the statistical jump model used in Shu, Yu, and Mulvey (2024), *“Downside risk reduction using regime-switching signals: a statistical jump model approach.”*

The model identifies persistent bull and bear regimes by clustering return-based features while penalizing changes between states.

## Model

Given standardized features \(x_t\), the model minimizes

```text
sum_t 0.5 * ||x_t - theta_{s_t}||²
    + lambda * sum_{t>=1} 1{s_t != s_{t-1}}.
```

The jump penalty `lambda` controls regime persistence. Fitting alternates between:

1. Finding the optimal state path with dynamic programming.
2. Updating each state centroid from its assigned observations.

The implementation uses multiple k-means++ initializations and retains the solution with the lowest objective.

## Paper features

The final paper uses three features calculated from daily excess returns:

- EWM downside deviation, half-life 10
- EWM Sortino ratio, half-life 20
- EWM Sortino ratio, half-life 60

Downside deviation is defined as

```text
sqrt(EWM[min(return, 0)²]).
```

The implementation is available in `regimejump.features.compute_paper_features`.

## Replication design

The target replication follows the paper’s main setup:

- S&P 500, DAX, and Nikkei 225
- Two market states
- 3000-day training and inference windows
- Model refitting every six months
- Daily rolling-DP regime inference
- Monthly jump-penalty selection using an eight-year validation window
- Equity exposure in the bull state and risk-free exposure in the bear state
- Signal from day `t` applied starting on day `t+2`
- 10 bp cost per one-way trade

The original study uses proprietary total-return and risk-free-rate data. Results based on public sources will therefore be presented as an approximate data replication of the published methodology.

## Current status

Completed:

- [x] Statistical jump-model objective
- [x] `O(TK)` dynamic-programming state solver
- [x] Multi-start coordinate-descent fitting
- [x] Empty-state handling
- [x] Final-paper feature set
- [x] Brute-force DP correctness tests
- [x] Comparison with the `jumpmodels` reference implementation
- [x] Experimental greedy-online inference

Remaining:

- [ ] Public-data pipeline
- [ ] Rolling-DP online inference
- [ ] Six-month refitting schedule
- [ ] Monthly jump-penalty selection
- [ ] Delayed 0/1 strategy backtest
- [ ] Replication tables and figures
- [ ] Gaussian HMM comparison

The greedy online classifier and alternative short-horizon features are treated as extensions, not as part of the baseline replication.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Run the tests:

```powershell
pytest -q
```

Run static checks:

```powershell
ruff check src tests scripts
```

Optional data and HMM dependencies:

```powershell
python -m pip install -e ".[dev,data,hmm]"
```

## Reproducibility

All reported out-of-sample results will follow three rules:

- No future information may affect a historical signal.
- Preprocessing and hyperparameter selection use historical data only.
- Signal dates, execution dates, risk-free returns, and transaction costs are handled explicitly.

## References

Shu, Y., Yu, C., & Mulvey, J. M. (2024). Downside risk reduction using regime-switching signals: a statistical jump model approach. *Journal of Asset Management*, 25, 493–507.  
https://doi.org/10.1057/s41260-024-00376-x

Nystrup, P., Lindström, E., & Madsen, H. (2020). Learning hidden Markov models with persistent states by penalizing jumps. *Expert Systems with Applications*, 150, 113307.  
https://doi.org/10.1016/j.eswa.2020.113307