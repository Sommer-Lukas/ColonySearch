"""
Standard (classical) Particle Swarm Optimisation (PSO).

Reference: Kennedy & Eberhart, "Particle swarm optimization", ICNN 1995.
Inertia-weight variant: Shi & Eberhart, 1998 — w decays linearly from
w_max to w_min over the run so early iterations explore, later ones exploit.

BOUNDS = [(0.0, 2.0), (8, 128), (0.0, 1.0)]  — [link_bias, svd_dims, alpha]
"""

import numpy as np

from . import BOUNDS, OptimizeResult


class PSO:
    def __init__(
        self,
        bounds: list[tuple] = BOUNDS,
        n_particles: int = 20,
        max_iter: int = 30,
        w_max: float = 0.9,
        w_min: float = 0.4,
        c1: float = 2.0,   # cognitive coefficient
        c2: float = 2.0,   # social coefficient
        seed: int = 42,
        early_stop_patience: int | None = None,
        early_stop_min_delta: float = 1e-4,
        early_stop_min_iters: int = 0,
    ):
        self.bounds = bounds
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        self.rng = np.random.default_rng(seed)
        self._n_dims = len(bounds)
        self.early_stop_patience = early_stop_patience
        self.early_stop_min_delta = early_stop_min_delta
        self.early_stop_min_iters = early_stop_min_iters

    def optimize(self, objective) -> OptimizeResult:
        """Minimise objective(params). objective must return a scalar."""
        lo = np.array([b[0] for b in self.bounds], dtype=float)
        hi = np.array([b[1] for b in self.bounds], dtype=float)
        span = hi - lo

        positions = self.rng.uniform(lo, hi, size=(self.n_particles, self._n_dims))
        # Initialise velocities in [-span, span] so particles can traverse the space in one step
        velocities = self.rng.uniform(-span, span, size=(self.n_particles, self._n_dims))

        pbest = positions.copy()
        pbest_fitness = np.full(self.n_particles, np.inf)
        gbest: np.ndarray | None = None
        gbest_fitness = np.inf

        history: list[dict] = []
        n_evals = 0
        best_ndcg_seen = -np.inf
        no_improve = 0
        early_stop_active = (
            self.early_stop_patience is not None and self.early_stop_patience > 0
        )

        for t in range(self.max_iter):
            # Linear inertia decay: high early (exploration) → low late (exploitation)
            w = self.w_max - (self.w_max - self.w_min) * (t / max(self.max_iter - 1, 1))

            for i in range(self.n_particles):
                fit = objective(positions[i])
                n_evals += 1
                if fit < pbest_fitness[i]:
                    pbest_fitness[i] = fit
                    pbest[i] = positions[i].copy()
                if fit < gbest_fitness:
                    gbest_fitness = fit
                    gbest = positions[i].copy()

            current_best_ndcg = float(-gbest_fitness) if gbest is not None else -np.inf

            for i in range(self.n_particles):
                r1 = self.rng.random(self._n_dims)
                r2 = self.rng.random(self._n_dims)
                velocities[i] = (
                    w * velocities[i]
                    + self.c1 * r1 * (pbest[i] - positions[i])
                    + self.c2 * r2 * (gbest - positions[i])
                )
                positions[i] = np.clip(positions[i] + velocities[i], lo, hi)

            history.append({
                "iteration": t,
                "best_ndcg": float(-gbest_fitness),
                "n_evals": n_evals,
            })

            if early_stop_active:
                if current_best_ndcg > best_ndcg_seen + self.early_stop_min_delta:
                    best_ndcg_seen = current_best_ndcg
                    no_improve = 0
                else:
                    no_improve += 1
                if (
                    no_improve >= self.early_stop_patience
                    and (t + 1) >= self.early_stop_min_iters
                ):
                    break

        return OptimizeResult(
            best_params=gbest.tolist() if gbest is not None else lo.tolist(),
            best_fitness=gbest_fitness,
            history=history,
        )
