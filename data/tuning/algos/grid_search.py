"""
Grid Search over link_bias × svd_dims × alpha.

Uses a narrower range than QPSO (link_bias ≤ 1, svd_dims ≤ 64) to keep
the total evaluation budget tractable at 125 points for n_per_dim=5.
"""

import itertools

import numpy as np

from . import BOUNDS, OptimizeResult

# Grid search uses a narrower range than full BOUNDS to keep the budget
# tractable — link_bias capped at 1.0, svd_dims capped at 64.
_GRID_LINK_BIAS = (BOUNDS[0][0], 1.0)
_GRID_SVD_DIMS  = (BOUNDS[1][0], 64)
_GRID_ALPHA     = (BOUNDS[2][0], BOUNDS[2][1])


class GridSearch:
    def __init__(
        self,
        bounds=None,    # ignored — uses _GRID_* ranges above
        n_per_dim: int = 5,
        seed: int = 42,  # unused, kept for API consistency
    ):
        self.n_per_dim = n_per_dim

    def _grid_points(self):
        link_bias_vals = np.linspace(*_GRID_LINK_BIAS, self.n_per_dim)
        svd_raw = np.linspace(*_GRID_SVD_DIMS, self.n_per_dim).astype(int)
        svd_vals = sorted(set(svd_raw.tolist()))   # deduplicate
        alpha_vals = np.linspace(*_GRID_ALPHA, self.n_per_dim)
        return list(itertools.product(link_bias_vals, svd_vals, alpha_vals))

    def optimize(self, objective) -> OptimizeResult:
        points = self._grid_points()
        best_params = None
        best_fitness = np.inf
        history: list[dict] = []
        n_evals = 0

        for i, params in enumerate(points):
            params_arr = np.array(params, dtype=float)
            fit = objective(params_arr)
            n_evals += 1
            if fit < best_fitness:
                best_fitness = fit
                best_params = params_arr.tolist()
            history.append({
                "iteration": i,
                "best_ndcg": float(-best_fitness),
                "n_evals": n_evals,
            })

        return OptimizeResult(
            best_params=best_params or [0.0, 8, 0.0],
            best_fitness=best_fitness,
            history=history,
        )
