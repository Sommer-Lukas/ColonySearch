from dataclasses import dataclass, field

# ── Parameter bounds ──────────────────────────────────────────────────────────
# Single source of truth for all three optimisation algorithms.
# Each entry is (min, max) for one parameter in this order:
#
#   [0] link_bias  float  weight of link-graph features relative to text embeddings
#                         0 = pure text, 2 = link features twice as strong as text
#   [1] svd_dims   int    TruncatedSVD dimensionality of the link-graph features
#                         (always rounded to int before use)
#   [2] alpha      float  BM25 vs KNN blend in final scoring
#                         0 = pure KNN, 1 = pure BM25
BOUNDS: list[tuple] = [
    (0.0, 2.0),   # link_bias
    (8,   128),   # svd_dims
    (0.0, 1.0),   # alpha
]


@dataclass
class OptimizeResult:
    best_params: list       # [link_bias, svd_dims, alpha]
    best_fitness: float     # lowest value seen (negated NDCG)
    history: list[dict]     # [{iteration, best_ndcg, n_evals}, ...]
    n_evals: int = field(init=False)

    def __post_init__(self):
        self.n_evals = self.history[-1]["n_evals"] if self.history else 0
