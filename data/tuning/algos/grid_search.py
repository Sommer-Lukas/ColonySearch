"""
Grid Search over arbitrary parameter bounds.

Default bounds are the local-search BOUNDS (link_bias, svd_dims, alpha) with a
narrower range to keep the budget tractable at 125 points for n_per_dim=5.
Pass custom bounds to sweep a different parameter space (e.g. network params).
"""

import itertools

import numpy as np

from . import BOUNDS, OptimizeResult

# Default narrower range kept for backwards compatibility with benchmark.ipynb.
_DEFAULT_BOUNDS = [
    (BOUNDS[0][0], 1.0),    # link_bias capped at 1.0
    (BOUNDS[1][0], 64),     # svd_dims capped at 64
    (BOUNDS[2][0], BOUNDS[2][1]),  # alpha full range
]


class GridSearch:
    def __init__(
        self,
        bounds=None,
        n_per_dim: int = 5,
        seed: int = 42,  # unused, kept for API consistency
    ):
        self.bounds    = bounds if bounds is not None else _DEFAULT_BOUNDS
        self.n_per_dim = n_per_dim

    def _grid_points(self):
        axes = [np.linspace(lo, hi, self.n_per_dim) for (lo, hi) in self.bounds]
        return list(itertools.product(*axes))

    def optimize(self, objective) -> OptimizeResult:
        points = self._grid_points()
        best_params  = None
        best_fitness = np.inf
        history: list[dict] = []
        n_evals = 0

        for i, params in enumerate(points):
            params_arr = np.array(params, dtype=float)
            fit = objective(params_arr)
            n_evals += 1
            if fit < best_fitness:
                best_fitness = fit
                best_params  = params_arr.tolist()
            history.append({
                "iteration": i,
                "best_ndcg": float(-best_fitness),
                "n_evals":   n_evals,
            })

        default_lo = [b[0] for b in self.bounds]
        return OptimizeResult(
            best_params=best_params or default_lo,
            best_fitness=best_fitness,
            history=history,
        )
