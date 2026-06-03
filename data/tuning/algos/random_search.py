"""
Random Search over arbitrary parameter bounds.

Default bounds are the same narrower range as GridSearch (link_bias ≤ 1,
svd_dims ≤ 64) for a fair budget comparison with 125 evaluations.
Pass custom bounds to search a different space (e.g. network params).
"""

import numpy as np

from . import BOUNDS, OptimizeResult

# Default narrower range kept for backwards compatibility with benchmark.ipynb.
_DEFAULT_BOUNDS = [
    (BOUNDS[0][0], 1.0),
    (int(BOUNDS[1][0]), 64),
    (BOUNDS[2][0], BOUNDS[2][1]),
]


class RandomSearch:
    def __init__(
        self,
        bounds=None,
        n_evaluations: int = 125,
        seed: int = 42,
        early_stop_patience: int | None = None,
        early_stop_min_delta: float = 1e-4,
        early_stop_min_iters: int = 0,
    ):
        self.bounds        = bounds if bounds is not None else _DEFAULT_BOUNDS
        self.n_evaluations = n_evaluations
        self.rng           = np.random.default_rng(seed)
        self.early_stop_patience  = early_stop_patience
        self.early_stop_min_delta = early_stop_min_delta
        self.early_stop_min_iters = early_stop_min_iters

    def optimize(self, objective) -> OptimizeResult:
        best_params  = None
        best_fitness = np.inf
        history: list[dict] = []
        best_ndcg_seen = -np.inf
        no_improve     = 0
        early_stop_active = (
            self.early_stop_patience is not None and self.early_stop_patience > 0
        )

        for i in range(self.n_evaluations):
            params = np.array(
                [self.rng.uniform(lo, hi) for (lo, hi) in self.bounds],
                dtype=float,
            )
            fit = objective(params)
            if fit < best_fitness:
                best_fitness = fit
                best_params  = params.tolist()

            current_best_ndcg = float(-best_fitness)
            history.append({
                "iteration": i,
                "best_ndcg": float(-best_fitness),
                "n_evals":   i + 1,
            })

            if early_stop_active:
                if current_best_ndcg > best_ndcg_seen + self.early_stop_min_delta:
                    best_ndcg_seen = current_best_ndcg
                    no_improve     = 0
                else:
                    no_improve += 1
                if (
                    no_improve >= self.early_stop_patience
                    and (i + 1) >= self.early_stop_min_iters
                ):
                    break

        default_lo = [b[0] for b in self.bounds]
        return OptimizeResult(
            best_params=best_params or default_lo,
            best_fitness=best_fitness,
            history=history,
        )
