#!/usr/bin/env python3
"""
Parameter tuner CLI for ColonySearch local node search.

Evaluates across ALL nodes under --data-dir to avoid overfitting to one topic.

Usage:
  python tuner.py --algo PSO    [--data-dir ../../data]
  python tuner.py --algo QPSO   [--data-dir ../../data]
  python tuner.py --algo GRID   [--data-dir ../../data]
  python tuner.py --algo RANDOM [--data-dir ../../data]

Output:
  {algo}_results.json     — best params + full history
    {algo}_convergence.png  — n_evals vs best NDCG@k
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Allow importing project modules from repo root
_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(_ROOT))

from data.tuning.ground_truth import load_all_ground_truth
from data.tuning.metrics import evaluate
from data.tuning.local_search import resolve_svd_dims
from data.tuning.algos import BOUNDS, OptimizeResult
from data.tuning.algos.qpso import QPSO
from data.tuning.algos.pso import PSO
from data.tuning.algos.grid_search import GridSearch
from data.tuning.algos.random_search import RandomSearch


def main():
    p = argparse.ArgumentParser(description="Tune local node search hyperparameters.")
    p.add_argument("--algo", required=True, choices=["PSO", "QPSO", "GRID", "RANDOM"],
                   help="Optimisation algorithm")
    p.add_argument("--data-dir", default=str(_ROOT / "data"),
                   help="Data directory containing dbs/ (all nodes used)")
    p.add_argument("--output", default=".",
                   help="Directory for output files (default: current dir)")
    p.add_argument("--k", type=int, default=3, help="@k for NDCG evaluation")
    p.add_argument("--particles", type=int, default=20, help="QPSO: number of particles")
    p.add_argument("--iterations", type=int, default=20, help="QPSO: max iterations")
    p.add_argument("--n-evals", type=int, default=200, help="RANDOM: evaluation budget")
    p.add_argument("--n-per-dim", type=int, default=5, help="GRID: points per dimension")
    args = p.parse_args()

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load ground truth across all nodes
    print("Loading ground truth across all nodes …")
    test_cases = load_all_ground_truth(data_dir)
    unique_clusters = len({frozenset(tc["relevant_urls"]) for tc in test_cases})
    print(f"Loaded {len(test_cases)} test queries across {unique_clusters} clusters")

    if not test_cases:
        print("ERROR: no test cases generated — check that index.db files have topic labels.")
        sys.exit(1)

    # 2. Objective function
    def objective(params):
        result = evaluate(params, test_cases, k=args.k)
        return -result["mean_ndcg"]   # minimise → negate NDCG

    # 3. Select and run algorithm
    algo_name = args.algo
    t0 = time.time()

    if algo_name == "PSO":
        algo = PSO(n_particles=args.particles, max_iter=args.iterations)
    elif algo_name == "QPSO":
        algo = QPSO(n_particles=args.particles, max_iter=args.iterations)
    elif algo_name == "GRID":
        algo = GridSearch(n_per_dim=args.n_per_dim)
    else:
        algo = RandomSearch(n_evaluations=args.n_evals)

    print(f"Running {algo_name} …")
    result: OptimizeResult = algo.optimize(objective)
    elapsed = time.time() - t0

    # 4. Print results
    lb, svd, alpha = result.best_params
    best_ndcg = -result.best_fitness
    print()
    print(f"Best link_bias: {lb:.3f}")
    print(f"Best svd_dims:  {resolve_svd_dims(svd)}")
    print(f"Best alpha:     {alpha:.3f}")
    print(f"Best NDCG@{args.k}:   {best_ndcg:.4f}")
    print(f"Total evals:    {result.n_evals}")
    print(f"Time elapsed:   {elapsed:.1f}s")

    # 5. Save JSON results
    slug = algo_name.lower()
    json_path = output_dir / f"{slug}_results.json"
    json_path.write_text(json.dumps({
        "algo": algo_name,
        "best_params": {"link_bias": lb, "svd_dims": resolve_svd_dims(svd), "alpha": alpha},
        "best_ndcg": best_ndcg,
        "n_evals": result.n_evals,
        "elapsed_s": elapsed,
        "history": result.history,
    }, indent=2))
    print(f"\nSaved results → {json_path}")

    # 6. Convergence plot
    png_path = output_dir / f"{slug}_convergence.png"
    n_evals_list = [h["n_evals"] for h in result.history]
    ndcg_list = [h["best_ndcg"] for h in result.history]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(n_evals_list, ndcg_list, linewidth=2)
    ax.set_xlabel("Evaluations")
    ax.set_ylabel(f"Best NDCG@{args.k}")
    ax.set_title(f"{algo_name} Convergence (NDCG@{args.k})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f"Saved plot    → {png_path}")


if __name__ == "__main__":
    main()
