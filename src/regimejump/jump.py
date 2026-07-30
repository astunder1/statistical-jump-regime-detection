"""Statistical jump model (Nystrup, Lindstrom & Madsen, 2020).

Penalized k-means over return features:

    sum_t 0.5 * ||x_t - theta_{s_t}||^2 + lambda * sum_{t>=1} 1{s_t != s_{t-1}}

Fit by coordinate descent between two steps:

    1. given a state path, each state's centroid is the mean of the
       features assigned to it ("centroid step");
    2. given centroids, the optimal state path is found by dynamic
       programming ("state step").
"""

from __future__ import annotations

import numpy as np


def kmeans_plusplus_init(X: np.ndarray, n_states: int, rng: np.random.Generator) -> np.ndarray:
    """Initialize centroids using k-means++.

    The first centroid is sampled uniformly from ``X``. Each subsequent
    centroid is sampled with probability proportional to its squared
    distance from the nearest previously selected centroid.

    Parameters
    ----------
    X:
        Standardized feature matrix with shape
        ``(n_observations, n_features)``.
    n_states:
        Number of centroids to initialize.
    rng:
        NumPy random generator used for reproducible sampling.

    Returns
    -------
    np.ndarray
        Initial centroids with shape ``(n_states, n_features)``.
    """
    n_obs = X.shape[0]
    centroids = np.empty((n_states, X.shape[1]), dtype=float)

    first = rng.integers(n_obs)
    centroids[0] = X[first]

    closest_sq_dist = np.sum((X - centroids[0]) ** 2, axis=1)
    for k in range(1, n_states):
        total = closest_sq_dist.sum()
        if total <= 0:
            # All remaining points coincide with a chosen centroid; pick
            # uniformly at random to keep the routine well-defined.
            next_idx = rng.integers(n_obs)
        else:
            probs = closest_sq_dist / total
            next_idx = rng.choice(n_obs, p=probs)
        centroids[k] = X[next_idx]
        new_sq_dist = np.sum((X - centroids[k]) ** 2, axis=1)
        closest_sq_dist = np.minimum(closest_sq_dist, new_sq_dist)

    return centroids


def jump_objective(
    X: np.ndarray, centroids: np.ndarray, labels: np.ndarray, jump_penalty: float
) -> float:
    """Evaluate the jump-model objective for a fixed state path.

    The objective is

    ``0.5 * sum_t ||x_t - theta_{s_t}||^2
    + lambda * sum_{t>=1} 1{s_t != s_{t-1}}``.

    Parameters
    ----------
    X:
        Feature matrix with shape ``(n_observations, n_features)``.
    centroids:
        State centroids with shape ``(n_states, n_features)``.
    labels:
        State path with shape ``(n_observations,)``.
    jump_penalty:
        Nonnegative penalty applied to each state change.

    Returns
    -------
    float
        Objective value for the supplied path and centroids.
    """
    X = np.asarray(X, dtype=float)
    centroids = np.asarray(centroids, dtype=float)
    labels = np.asarray(labels)

    fit_cost = 0.5 * np.sum((X - centroids[labels]) ** 2)
    n_switches = np.sum(labels[1:] != labels[:-1])
    return float(fit_cost + jump_penalty * n_switches)


class JumpModel:
    """Discrete statistical jump model for market regime detection.

    The model alternates between dynamic-programming state assignment
    and centroid updates, retaining the best solution across multiple
    initializations.

    Parameters
    ----------
    n_states:
        Number of regimes.
    jump_penalty:
        Nonnegative penalty applied to each state change. Larger values
        produce more persistent state paths.
    max_iter:
        Maximum number of coordinate-descent iterations per restart.
    n_init:
        Number of k-means++ initializations.
    random_state:
        Seed used for reproducible centroid initialization.

    Attributes
    ----------
    centroids_:
        Fitted centroids with shape ``(n_states, n_features)``.
    labels_:
        In-sample state path with shape ``(n_observations,)``.
    n_iter_:
        Number of iterations used by the selected restart.
    inertia_:
        Objective value of the selected solution.
    """

    def __init__(
        self,
        n_states: int = 2,
        jump_penalty: float = 50.0,
        max_iter: int = 50,
        n_init: int = 10,
        random_state: int | None = None,
    ):
        self.n_states = n_states
        self.jump_penalty = jump_penalty
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state

        self.centroids_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.n_iter_: int | None = None
        self.inertia_: float | None = None

    def _dp_state_path(self, X: np.ndarray, centroids: np.ndarray) -> tuple[np.ndarray, float]:
        """Find the optimal state path for fixed centroids.

        The method minimizes

        ``0.5 * sum_t ||x_t - theta_{s_t}||^2
        + lambda * sum_{t>=1} 1{s_t != s_{t-1}}``

        using the dynamic-programming recursion

        ``V[0, k] = 0.5 * ||x_0 - theta_k||^2``

        and

        ``V[t, k] = 0.5 * ||x_t - theta_k||^2
        + min(V[t-1, k], min_j V[t-1, j] + lambda)``.

        The minimum over previous states is computed once per observation,
        giving complexity ``O(n_observations * n_states)``.

        Parameters
        ----------
        X:
            Standardized feature matrix with shape
            ``(n_observations, n_features)``.
        centroids:
            Current centroid estimates with shape
            ``(n_states, n_features)``.

        Returns
        -------
        path:
            Optimal state path with shape ``(n_observations,)``.
        cost:
            Objective value attained by the optimal path.
        """

        X = np.asarray(X, dtype=float)
        centroids = np.asarray(centroids, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        if centroids.ndim != 2:
            raise ValueError("centroids must be a 2D array")

        if X.shape[0] == 0:
            raise ValueError("X must contain at least one observation")

        if X.shape[1] != centroids.shape[1]:
            raise ValueError("X and centroids must have the same number of features")

        if centroids.shape[0] != self.n_states:
            raise ValueError("centroids must contain n_states rows")

        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values")

        if not np.isfinite(centroids).all():
            raise ValueError("centroids must contain only finite values")

        if not np.isfinite(self.jump_penalty) or self.jump_penalty < 0:
            raise ValueError("jump_penalty must be finite and non-negative")

        observation_costs = 0.5 * np.sum(
            (X[:, None, :] - centroids[None, :, :]) ** 2,
            axis=2,
        )

        n_obs, n_states = observation_costs.shape

        # value[t, k] is the minimum total cost of all paths that cover dates 0 to t
        # and end in state k on date t
        value = np.empty((n_obs, n_states), dtype=float)

        # predecessor[t, k] records the state at t - 1 that leads
        # to the cheapest partial path ending in state k at time t.
        predecessor = np.empty((n_obs, n_states), dtype=int)

        # Initialization at first date
        value[0] = observation_costs[0]
        predecessor[0] = -1

        for t in range(1, n_obs):
            previous_values = value[t - 1]

            best_previous_state = int(np.argmin(previous_values))
            best_switch_cost = previous_values[best_previous_state] + self.jump_penalty

            for k in range(n_states):
                stay_cost = previous_values[k]

                if stay_cost <= best_switch_cost:
                    transition_cost = stay_cost
                    predecessor[t, k] = k
                else:
                    transition_cost = best_switch_cost
                    predecessor[t, k] = best_previous_state

                value[t, k] = observation_costs[t, k] + transition_cost

        last_state = int(np.argmin(value[-1]))
        best_cost = float(value[-1, last_state])

        path = np.empty(n_obs, dtype=int)
        path[-1] = last_state
        for t in range(n_obs - 1, 0, -1):
            path[t - 1] = predecessor[t, path[t]]

        return path, best_cost

    def _update_centroids(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        old_centroids: np.ndarray,
    ) -> np.ndarray:
        """Update centroids and reseed states with no assigned observations."""

        X = np.asarray(X, dtype=float)
        labels = np.asarray(labels, dtype=int)
        old_centroids = np.asarray(old_centroids, dtype=float)

        new_centroids = old_centroids.copy()

        # First, update every nonempty state
        for k in range(self.n_states):
            members = X[labels == k]

            if len(members) > 0:
                new_centroids[k] = members.mean(axis=0)

        # Measure each observation against its updated assigned centroid
        residuals = 0.5 * np.sum(
            (X - new_centroids[labels]) ** 2,
            axis=1,
        )

        # Reseed empty states using poorly fitted observations
        empty_states = [k for k in range(self.n_states) if not np.any(labels == k)]

        available = residuals.copy()

        for k in empty_states:
            replacement_index = int(np.argmax(available))
            new_centroids[k] = X[replacement_index]

            # Avoid using the same observation for multiple empty states.
            available[replacement_index] = -np.inf

        return new_centroids

    def fit(self, X: np.ndarray) -> JumpModel:
        """Fit the model using multi-start coordinate descent.

        Each restart alternates between dynamic-programming state assignment
        and centroid updates. The solution with the lowest objective is retained.

        Parameters
        ----------
        X:
            Standardized feature matrix with shape
            ``(n_observations, n_features)``.

        Returns
        -------
        JumpModel
            The fitted model.
        """
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        if X.shape[0] == 0:
            raise ValueError("X must contain at least one observation")

        if X.shape[1] == 0:
            raise ValueError("X must contain at least one feature")

        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values")

        if not isinstance(self.n_states, int) or self.n_states < 1:
            raise ValueError("n_states must be a positive integer")

        if self.n_states > X.shape[0]:
            raise ValueError("n_states cannot exceed the number of observations")

        if not np.isfinite(self.jump_penalty) or self.jump_penalty < 0:
            raise ValueError("jump_penalty must be finite and non-negative")

        if not isinstance(self.max_iter, int) or self.max_iter < 1:
            raise ValueError("max_iter must be a positive integer")

        if not isinstance(self.n_init, int) or self.n_init < 1:
            raise ValueError("n_init must be a positive integer")

        rng = np.random.default_rng(self.random_state)

        best_centroids = None
        best_labels = None
        best_n_iter = None
        best_cost = np.inf

        for _ in range(self.n_init):
            centroids = kmeans_plusplus_init(
                X,
                n_states=self.n_states,
                rng=rng,
            )

            previous_labels = None

            for n_iter in range(1, self.max_iter + 1):
                labels, _ = self._dp_state_path(X, centroids)

                updated_centroids = self._update_centroids(
                    X,
                    labels,
                    centroids,
                )

                labels_unchanged = previous_labels is not None and np.array_equal(
                    labels, previous_labels
                )

                centroids = updated_centroids

                if labels_unchanged:
                    break

                previous_labels = labels.copy()

            restart_cost = jump_objective(
                X,
                centroids,
                labels,
                self.jump_penalty,
            )

            if restart_cost < best_cost:
                best_cost = restart_cost
                best_centroids = centroids.copy()
                best_labels = labels.copy()
                best_n_iter = n_iter

        self.centroids_ = best_centroids
        self.labels_ = best_labels
        self.n_iter_ = best_n_iter
        self.inertia_ = float(best_cost)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Decode the optimal batch state path for new observations.

        This method uses the full supplied sequence and is therefore
        non-causal.

        Parameters
        ----------
        X:
            Standardized feature matrix with shape
            ``(n_observations, n_features)``.

        Returns
        -------
        np.ndarray
            Optimal state path with shape ``(n_observations,)``.
        """

        if self.centroids_ is None:
            raise RuntimeError("JumpModel must be fitted before calling predict()")

        path, _ = self._dp_state_path(
            np.asarray(X, dtype=float),
            self.centroids_,
        )

        return path

    def relabel_by_feature(self, feature_idx: int = 0) -> JumpModel:
        """Order states by descending centroid value for one feature.

        State 0 is assigned to the state with the highest centroid value
        for ``feature_idx``.

        Parameters
        ----------
        feature_idx:
            Feature column used to order the centroids.

        Returns
        -------
        JumpModel
            The relabelled model.
        """
        if self.centroids_ is None or self.labels_ is None:
            raise RuntimeError("JumpModel must be fitted before relabeling")

        order = np.argsort(-self.centroids_[:, feature_idx])
        new_index_of_old = np.empty_like(order)
        new_index_of_old[order] = np.arange(len(order))

        self.centroids_ = self.centroids_[order]
        self.labels_ = new_index_of_old[self.labels_]
        return self

    def relabel_by_cumulative_return(
        self,
        returns: np.ndarray,
    ) -> JumpModel:
        """Order states by decreasing aggregate return.

        Returns are summed over observations assigned to each state. State 0
        is assigned to the state with the highest aggregate return.

        Parameters
        ----------
        returns:
            Return series aligned one-to-one with the fitted state labels.

        Returns
        -------
        JumpModel
            The relabelled model.
        """

        if self.centroids_ is None or self.labels_ is None:
            raise RuntimeError("JumpModel must be fitted before relabeling")

        returns = np.asarray(returns, dtype=float)

        if returns.ndim != 1:
            raise ValueError("returns must be one-dimensional")

        if len(returns) != len(self.labels_):
            raise ValueError("returns and labels must have the same length")

        if not np.isfinite(returns).all():
            raise ValueError("returns must contain only finite values")

        cumulative_returns = np.full(self.n_states, -np.inf)

        for k in range(self.n_states):
            members = self.labels_ == k

            if np.any(members):
                cumulative_returns[k] = returns[members].sum()

        order = np.argsort(-cumulative_returns)

        new_index_of_old = np.empty_like(order)
        new_index_of_old[order] = np.arange(self.n_states)

        self.centroids_ = self.centroids_[order]
        self.labels_ = new_index_of_old[self.labels_]

        return self
