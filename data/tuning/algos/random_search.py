"""
Random Search over link_bias × svd_dims × alpha.

Samples uniformly from the same narrow range as GridSearch for a fair
budget comparison (both default to 125 evaluations).
"""

import numpy as np

from . import BOUNDS, OptimizeResult

# Same narrower range as GridSearch for a fair budget comparison.
_RS_LINK_BIAS = (BOUNDS[0][0], 1.0)
_RS_SVD_DIMS  = (int(BOUNDS[1][0]), 64)
_RS_ALPHA     = (BOUNDS[2][0], BOUNDS[2][1])


class RandomSearch:
    def __init__(
        self,
        bounds=None,            # ignored — uses _RS_* ranges above
        n_evaluations: int = 125,
        seed: int = 42,
        early_stop_patience: int | None = None,
        early_stop_min_delta: float = 1e-4,
        early_stop_min_iters: int = 0,
    ):
        self.n_evaluations = n_evaluations
        self.rng = np.random.default_rng(seed)
        self.early_stop_patience = early_stop_patience
        self.early_stop_min_delta = early_stop_min_delta
        self.early_stop_min_iters = early_stop_min_iters

    def optimize(self, objective) -> OptimizeResult:
        best_params = None
        best_fitness = np.inf
        history: list[dict] = []
        best_ndcg_seen = -np.inf
        no_improve = 0
        early_stop_active = (
            self.early_stop_patience is not None and self.early_stop_patience > 0
        )

        for i in range(self.n_evaluations):
            link_bias = float(self.rng.uniform(*_RS_LINK_BIAS))
            svd_dims = int(self.rng.integers(_RS_SVD_DIMS[0], _RS_SVD_DIMS[1] + 1))
            alpha = float(self.rng.uniform(*_RS_ALPHA))
            params = np.array([link_bias, svd_dims, alpha], dtype=float)

            fit = objective(params)
            if fit < best_fitness:
                best_fitness = fit
                best_params = params.tolist()

            current_best_ndcg = float(-best_fitness)

            history.append({
                "iteration": i,
                "best_ndcg": float(-best_fitness),
                "n_evals": i + 1,
            })

            if early_stop_active:
                if current_best_ndcg > best_ndcg_seen + self.early_stop_min_delta:
                    best_ndcg_seen = current_best_ndcg
                    no_improve = 0
                else:
                    no_improve += 1
                if (
                    no_improve >= self.early_stop_patience
                    and (i + 1) >= self.early_stop_min_iters
                ):
                    break

        return OptimizeResult(
            best_params=best_params or [0.0, 8, 0.0],
            best_fitness=best_fitness,
            history=history,
        )
