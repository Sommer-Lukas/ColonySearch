"""
Quantum Particle Swarm Optimisation (QPSO).

Reference: Sun et al., "Quantum-Behaved Particle Swarm Optimization: Analysis
of Individual Particle Behavior and Parameter Selection", 2012.

BOUNDS = [(0.0, 2.0), (8, 128), (0.0, 1.0)]  — [link_bias, svd_dims, alpha]
"""

import numpy as np

from . import BOUNDS, OptimizeResult


class QPSO:
    def __init__(
        self,
        bounds: list[tuple] = BOUNDS,
        n_particles: int = 20,
        max_iter: int = 30,
        seed: int = 42,
    ):
        self.bounds = bounds
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.rng = np.random.default_rng(seed)
        self._n_dims = len(bounds)

    def optimize(self, objective) -> OptimizeResult:
        """
        Minimise objective(params).  objective should return a scalar
        (use negated NDCG so lower = better).
        """
        lo = np.array([b[0] for b in self.bounds], dtype=float)
        hi = np.array([b[1] for b in self.bounds], dtype=float)

        # Initialise particles uniformly in bounds
        particles = self.rng.uniform(lo, hi, size=(self.n_particles, self._n_dims))
        pbest = particles.copy()
        pbest_fitness = np.full(self.n_particles, np.inf)
        gbest: np.ndarray | None = None
        gbest_fitness = np.inf

        history: list[dict] = []
        n_evals = 0

        for t in range(self.max_iter):
            # Evaluate each particle
            for i in range(self.n_particles):
                fit = objective(particles[i])
                n_evals += 1
                if fit < pbest_fitness[i]:
                    pbest_fitness[i] = fit
                    pbest[i] = particles[i].copy()
                if fit < gbest_fitness:
                    gbest_fitness = fit
                    gbest = particles[i].copy()

            beta = 1.0 - 0.5 * (t / self.max_iter)
            mbest = pbest.mean(axis=0)

            # Update positions
            for i in range(self.n_particles):
                phi = self.rng.random(self._n_dims)
                p = phi * pbest[i] + (1 - phi) * gbest
                u = self.rng.random(self._n_dims)
                # Avoid log(0): clip u away from zero
                u = np.clip(u, 1e-10, 1.0)
                L = beta * np.abs(mbest - particles[i])
                sign = np.where(self.rng.random(self._n_dims) > 0.5, 1.0, -1.0)
                particles[i] = p + sign * L * np.log(1.0 / u)
                particles[i] = np.clip(particles[i], lo, hi)

            history.append({
                "iteration": t,
                "best_ndcg": float(-gbest_fitness),
                "n_evals": n_evals,
            })

        return OptimizeResult(
            best_params=gbest.tolist() if gbest is not None else lo.tolist(),
            best_fitness=gbest_fitness,
            history=history,
        )
