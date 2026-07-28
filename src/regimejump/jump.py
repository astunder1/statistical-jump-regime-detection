"""Statistical jump model (Nystrup, Lindstrom & Madsen, 2020).

Penalized k-means over return features:

    sum_t 0.5 * ||x_t - theta_{s_t}||^2 + lambda * sum_{t>=1} 1{s_t != s_{t-1}}

Fit by coordinate descent between two steps:

    1. given a state path, each state's centroid is the mean of the
       features assigned to it ("centroid step");
    2. given centroids, the optimal state path is found by dynamic
       programming ("state step").

``JumpModel._dp_state_path`` and ``JumpModel.fit`` are intentionally left
unimplemented here -- they are the core of the project and are implemented
by hand, not generated. Everything else (construction, k-means++
initialization, relabeling, prediction on top of fitted centroids) is
ordinary plumbing and is provided.
"""

from __future__ import annotations

import numpy as np


def kmeans_plusplus_init(
    X: np.ndarray, n_states: int, rng: np.random.Generator
) -> np.ndarray:
    """k-means++ seeding: pick initial centroids spread out across ``X``.

    Standard k-means++ (Arthur & Vassilvitskii, 2007): the first centroid
    is a uniformly random row of ``X``; each subsequent centroid is drawn
    with probability proportional to its squared distance to the nearest
    centroid chosen so far.

    Parameters
    ----------
    X:
        (T, n_features) array of standardized features.
    n_states:
        Number of centroids to choose.
    rng:
        A numpy random Generator, for reproducibility under ``random_state``.

    Returns
    -------
    (n_states, n_features) array of initial centroids.
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
    """The penalized k-means objective for a given path and centroids.

        sum_t ||x_t - theta_{s_t}||^2 + lambda * sum_{t>=1} 1{s_t != s_{t-1}}

    A plain evaluation helper (not part of fitting): useful for comparing
    candidate paths -- e.g. the DP-optimal path against a greedy
    approximation -- under the same centroids and penalty.
    """
    X = np.asarray(X, dtype=float)
    centroids = np.asarray(centroids, dtype=float)
    labels = np.asarray(labels)

    fit_cost = 0.5 * np.sum((X - centroids[labels]) ** 2)
    n_switches = np.sum(labels[1:] != labels[:-1])
    return float(fit_cost + jump_penalty * n_switches)


class JumpModel:
    """Statistical jump model for discrete market regime detection.

    Parameters
    ----------
    n_states:
        Number of regimes ``K``.
    jump_penalty:
        The switching penalty ``lambda`` in the objective above. Larger
        values produce more persistent (slower-switching) regime paths.
    max_iter:
        Maximum number of centroid/state coordinate-descent iterations
        per restart.
    n_init:
        Number of random (k-means++) restarts; the restart with the
        lowest objective value is kept.
    random_state:
        Seed for reproducible initialization.

    Attributes (set by ``fit``)
    ----------------------------
    centroids_ : (n_states, n_features) array
    labels_ : (T,) int array, the in-sample optimal state path
    n_iter_ : int, iterations used by the winning restart
    inertia_ : float, the winning restart's objective value
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
        """Solve for the optimal state path given fixed centroids.

        Must find the path ``s_0, ..., s_{T-1}`` minimizing

            sum_t 0.5 * ||x_t - theta_{s_t}||^2 + lambda * sum_{t>=1} 1{s_t != s_{t-1}}

        via dynamic programming:

            V[0, k] = 0.5 * ||x_0 - theta_k||^2

            V[t, k] = 0.5 * ||x_t - theta_k||^2 + min(
                V[t-1, k],
                min_j V[t-1, j] + lambda,
            )

        Complexity requirement: O(T*K), not O(T*K^2) -- computing
        ``min_j V[t-1, j]`` once per row of ``V`` (rather than once per
        (t, k) pair) is what keeps this linear in K instead of quadratic.

        Parameters
        ----------
        X:
            (T, n_features) standardized feature matrix.
        centroids:
            (K, n_features) current centroid estimates.

        Returns
        -------
        path:
            (T,) int array, the optimal state assignment per row of ``X``.
        cost:
            The objective value achieved by ``path`` (fit cost + total
            switching penalty), for comparing restarts / convergence.
        """

        # First, ensure proper arrays created
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

        # At each date t and state k, compute observation cost
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

        # Begin the recurrence relation
        for t in range(1, n_obs):
            previous_values = value[t - 1]

            best_previous_state = int(np.argmin(previous_values))
            best_switch_cost = (
                previous_values[best_previous_state]
                + self.jump_penalty
            )

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

        # Backtrack to determine states
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
        """Update centroids from fixed state assignment
        Nonempty states updated to mean of their assigned observations.
        Empty states temporarily left at previous centroid.
        """

        # Ensure data types are correct
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
        empty_states = [
            k
            for k in range(self.n_states)
            if not np.any(labels == k)
        ]

        available = residuals.copy()

        for k in empty_states:
            replacement_index = int(np.argmax(available))
            new_centroids[k] = X[replacement_index]

            # Avoid using the same observation for multiple empty states.
            available[replacement_index] = -np.inf

        return new_centroids


    def fit(self, X: np.ndarray) -> "JumpModel":
        """Fit the jump model to ``X`` by coordinate descent.

        Expected structure (standard jump-model fitting, Nystrup et al.
        2020):

        1. For each of ``n_init`` restarts (use ``kmeans_plusplus_init``
           for initial centroids, seeded from ``self.random_state``):
           a. Alternate, up to ``max_iter`` times or until the state path
              stops changing:
              - state step: call ``self._dp_state_path(X, centroids)``;
              - centroid step: recompute each centroid as the mean of the
                rows of ``X`` currently assigned to it;
              - if a state ends up with no rows assigned, reseed its
                centroid (e.g. to the row with the largest cost under the
                current assignment) rather than leaving it empty.
           b. Track that restart's final objective value.
        2. Keep the restart with the lowest objective value; set
           ``self.centroids_``, ``self.labels_``, ``self.n_iter_``,
           ``self.inertia_`` from it.
        3. Return ``self`` (so ``model.fit(X).labels_`` works).

        Parameters
        ----------
        X:
            (T, n_features) standardized feature matrix. Standardization
            is the caller's responsibility (see :mod:`regimejump.online`
            for the look-ahead-free version).
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

        raise NotImplementedError("fit loop not implemented yet")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign the optimal (batch, non-causal) state path for ``X``.

        Requires the model to already be fitted (uses ``self.centroids_``
        and ``self.jump_penalty`` in a fresh call to ``_dp_state_path``).
        For genuinely online, causal predictions on new data use
        :func:`regimejump.online.greedy_online_path` instead.
        """
        if self.centroids_ is None:
            raise RuntimeError("JumpModel must be fitted before calling predict()")
        path, _ = self._dp_state_path(np.asarray(X, dtype=float), self.centroids_)
        return path

    def relabel_by_feature(self, X: np.ndarray, feature_idx: int = 0) -> "JumpModel":
        """Relabel fitted states so state 0 is the high-mean regime.

        States from coordinate descent are arbitrarily ordered (whichever
        restart / initialization happened to land on them). This reorders
        ``centroids_`` and remaps ``labels_`` so that state 0 always has
        the highest centroid value on ``feature_idx`` (by convention, an
        EWM-mean-of-returns feature, so state 0 is "bull"/calm and higher
        states are progressively more bearish/volatile).

        Parameters
        ----------
        X:
            Unused directly; kept in the signature for API symmetry with
            ``fit``/``predict`` and potential future feature-based
            relabeling rules. Present implementation only needs
            ``centroids_``.
        feature_idx:
            Column of ``centroids_`` to sort on, descending.
        """
        if self.centroids_ is None or self.labels_ is None:
            raise RuntimeError("JumpModel must be fitted before relabeling")

        order = np.argsort(-self.centroids_[:, feature_idx])
        new_index_of_old = np.empty_like(order)
        new_index_of_old[order] = np.arange(len(order))

        self.centroids_ = self.centroids_[order]
        self.labels_ = new_index_of_old[self.labels_]
        return self
