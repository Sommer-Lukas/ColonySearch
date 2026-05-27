"""
Retrieval metrics: NDCG@k, Precision@k, Recall@k.

evaluate(params, test_cases, k=10) → dict with aggregated metrics.

test_cases must include a 'node_dir' field (use load_all_ground_truth).
The index is rebuilt once per unique node per evaluate() call.
"""

from math import log2

from .local_search import build_index, search


# ── Core metric functions ─────────────────────────────────────────────────────

def dcg_at_k(ranked_urls: list[str], relevant_urls: set, k: int) -> float:
    total = 0.0
    for i, url in enumerate(ranked_urls[:k], start=1):
        if url in relevant_urls:
            total += 1.0 / log2(i + 1)
    return total


def ndcg_at_k(ranked_urls: list[str], relevant_urls: set, k: int) -> float:
    if not relevant_urls:
        return 0.0
    actual = dcg_at_k(ranked_urls, relevant_urls, k)
    # Ideal: place all relevant docs first
    ideal_len = min(len(relevant_urls), k)
    ideal = sum(1.0 / log2(i + 1) for i in range(1, ideal_len + 1))
    if ideal == 0.0:
        return 0.0
    return actual / ideal


def precision_at_k(ranked_urls: list[str], relevant_urls: set, k: int) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for url in ranked_urls[:k] if url in relevant_urls)
    return hits / k


def recall_at_k(ranked_urls: list[str], relevant_urls: set, k: int) -> float:
    if not relevant_urls:
        return 0.0
    hits = sum(1 for url in ranked_urls[:k] if url in relevant_urls)
    return hits / len(relevant_urls)


# ── Evaluation harness ────────────────────────────────────────────────────────

def evaluate(params, test_cases: list[dict], k: int = 10) -> dict:
    """
    Evaluate search quality for the given parameter set across all test cases.

    params     — [link_bias, svd_dims, alpha]  (same order as BOUNDS)
    test_cases — must include 'node_dir' field (from load_all_ground_truth)
    Returns dict: {mean_ndcg, mean_precision, mean_recall, per_query_ndcg}
    """
    from pathlib import Path

    link_bias = float(params[0])
    svd_dims  = int(round(params[1]))
    alpha     = float(params[2])

    # Rebuild index once per unique node for this parameter set
    seen: set[Path] = set()
    for case in test_cases:
        nd = Path(case["node_dir"])
        if nd not in seen:
            build_index(nd, link_bias, svd_dims)
            seen.add(nd)

    ndcg_scores: list[float] = []
    prec_scores: list[float] = []
    rec_scores:  list[float] = []

    for case in test_cases:
        rel = case["relevant_urls"]
        if not rel:
            continue
        results = search(case["node_dir"], case["query_text"], case["query_vector"], alpha)
        ranked_urls = [r["url"] for r in results]
        ndcg_scores.append(ndcg_at_k(ranked_urls, rel, k))
        prec_scores.append(precision_at_k(ranked_urls, rel, k))
        rec_scores.append(recall_at_k(ranked_urls, rel, k))

    if not ndcg_scores:
        return {"mean_ndcg": 0.0, "mean_precision": 0.0, "mean_recall": 0.0, "per_query_ndcg": []}

    return {
        "mean_ndcg":       float(sum(ndcg_scores) / len(ndcg_scores)),
        "mean_precision":  float(sum(prec_scores)  / len(prec_scores)),
        "mean_recall":     float(sum(rec_scores)   / len(rec_scores)),
        "per_query_ndcg":  ndcg_scores,
    }
