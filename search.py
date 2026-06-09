"""
Local hybrid search for a single ColonySearch node.

Combines FTS5 BM25 (keyword relevance) with KNN cosine similarity
(semantic relevance).  Both signals are independently min-max normalised
to [0, 1] before blending so neither can dominate purely because of scale.

Call from a node's Flask handler:
    from search import search, SearchResult
    results = search(node_dir, query_text, query_vector)
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from joblib import load


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    url:   str
    title: str
    body:  str
    score: float   # combined normalised score [0.0 – 1.0]

    def to_dict(self) -> dict:
        return asdict(self)


# ── Internal helpers ──────────────────────────────────────────────────────────

# FTS5 special characters that break the MATCH expression.
_FTS5_SPECIAL = re.compile(r'["\'\(\)\*\+\-\^:,\.!]')

# Common English question/stop words that add noise to BM25 but carry no
# topical signal.  "What is a large language model" → "large language model".
_STOP_WORDS = {
    "WHAT", "IS", "ARE", "WAS", "WERE", "BE", "BEEN", "BEING",
    "A", "AN", "THE", "OF", "IN", "ON", "AT", "TO", "FOR",
    "HOW", "WHY", "WHO", "WHEN", "WHERE", "WHICH", "DOES", "DO",
    "AND", "OR", "NOT",
}


@lru_cache(maxsize=32)
def _cached_knn(knn_path_str: str):
    """Load KNN model once per node — avoids reloading on every search call."""
    return load(knn_path_str)


@lru_cache(maxsize=32)
def _cached_meta_urls(meta_path_str: str) -> list:
    return json.loads(Path(meta_path_str).read_text())["urls"]


@lru_cache(maxsize=32)
def _cached_url_rowid(db_path_str: str) -> dict:
    with sqlite3.connect(db_path_str) as con:
        return {url: rid for rid, url in con.execute("SELECT rowid, url FROM documents").fetchall()}


def _sanitize_fts5(query: str) -> str:
    """Strip FTS5 operator chars and stop words, preserve topical tokens."""
    cleaned = _FTS5_SPECIAL.sub(" ", query)
    tokens = [t for t in cleaned.split() if t.upper() not in _STOP_WORDS]
    return " ".join(tokens)


def _minmax(scores: dict[int, float]) -> dict[int, float]:
    """Normalise a rowid→score map to [0, 1].  Flat maps become all-1.0."""
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi == lo:
        return {k: 1.0 for k in scores}
    span = hi - lo
    return {k: (v - lo) / span for k, v in scores.items()}


# ── Public API ────────────────────────────────────────────────────────────────

def search(
    node_dir: str | Path,
    query_text: str,
    query_vector: np.ndarray | None,
    *,
    alpha: float = 0.51,
    top_k: int = 10,
    bm25_candidates: int =30000,
) -> list[SearchResult]:
    """
    Hybrid BM25 + KNN search over one node's local index.

    Parameters
    ----------
    node_dir        : path to the node folder (contains index.db + knn_model.joblib)
    query_text      : raw user query string — used for FTS5 BM25 matching
    query_vector    : 1-D sentence embedding of the query (shape (D,));
                      zero-padding for link-graph dims is applied internally.
                      Pass None to skip the KNN signal entirely.
    alpha           : blend weight for BM25.  1.0 = pure BM25, 0.0 = pure KNN.
    top_k           : number of results to return
    bm25_candidates : maximum BM25 rows fetched before normalisation

    Returns
    -------
    List of SearchResult (url, title, body, score) sorted by score descending.
    score is the blended normalised value in [0.0, 1.0].
    """
    node_dir = Path(node_dir)
    db_path  = node_dir / "index.db"
    knn_path = node_dir / "knn_model.joblib"

    # ── BM25 via FTS5 ─────────────────────────────────────────────────────────
    # FTS5 bm25() returns negative values; negate so higher = better.
    bm25_raw: dict[int, float] = {}

    use_bm25 = bool(query_text and query_text.strip()) and alpha > 0.0
    if use_bm25:
        safe_query = _sanitize_fts5(query_text)
        if safe_query:
            try:
                with sqlite3.connect(db_path) as con:
                    rows = con.execute(
                        "SELECT rowid, bm25(documents) "
                        "FROM documents "
                        "WHERE documents MATCH ? "
                        "ORDER BY rank "
                        "LIMIT ?",
                        (safe_query, bm25_candidates),
                    ).fetchall()
                for rowid, score in rows:
                    bm25_raw[rowid] = -float(score)
            except sqlite3.OperationalError:
                # Malformed query — fall back to no BM25 signal.
                pass

    # ── KNN cosine similarity ─────────────────────────────────────────────────
    # KNN is fitted on X_node which may include link-graph feature columns
    # appended after the text embedding.  We pad with zeros — the query has no
    # link-graph position, and zero is the neutral value (documented in
    # node_pipeline_setup.py :: build_feature_matrix).
    knn_raw: dict[int, float] = {}

    use_knn = query_vector is not None and alpha < 1.0 and knn_path.exists()
    if use_knn:
        nn  = _cached_knn(str(knn_path))
        vec = np.array(query_vector, dtype=np.float32).reshape(1, -1)

        pad = nn.n_features_in_ - vec.shape[1]
        if pad > 0:
            vec = np.hstack([vec, np.zeros((1, pad), dtype=np.float32)])

        distances, indices = nn.kneighbors(vec)

        meta_path = knn_path.parent / "knn_metadata.json"
        meta_urls: list[str] = _cached_meta_urls(str(meta_path)) if meta_path.exists() else []
        if meta_urls:
            url_to_rowid = _cached_url_rowid(str(db_path))
            for dist, idx in zip(distances[0], indices[0]):
                url    = meta_urls[int(idx)]
                rowid  = url_to_rowid.get(url)
                if rowid is not None:
                    knn_raw[rowid] = 1.0 - float(dist)   # cosine distance → similarity
        else:
            # Fallback for nodes without metadata (assumes no filtering was done).
            for dist, idx in zip(distances[0], indices[0]):
                knn_raw[int(idx) + 1] = 1.0 - float(dist)

    # ── Nothing matched ───────────────────────────────────────────────────────
    all_rowids = set(bm25_raw) | set(knn_raw)
    if not all_rowids:
        return []

    # ── Normalise BM25 only — KNN cosine similarity is already in [0, 1] ────────
    # Applying min-max to KNN would destroy the absolute similarity signal:
    # a node full of irrelevant docs would still produce a top score of 1.0,
    # making its results indistinguishable from a relevant node when merging
    # across nodes.  Cosine similarity needs no rescaling.
    bm25_norm = _minmax(bm25_raw)

    # ── Blend ─────────────────────────────────────────────────────────────────
    combined: dict[int, float] = {
        rid: alpha * bm25_norm.get(rid, 0.0) + (1.0 - alpha) * knn_raw.get(rid, 0.0)
        for rid in all_rowids
    }

    top_rowids = sorted(combined, key=combined.__getitem__, reverse=True)[:top_k]

    # ── Fetch url / title / body for top results ──────────────────────────────
    placeholders = ",".join("?" * len(top_rowids))
    with sqlite3.connect(db_path) as con:
        rows = con.execute(
            f"SELECT rowid, url, title, body "
            f"FROM documents "
            f"WHERE rowid IN ({placeholders})",
            top_rowids,
        ).fetchall()

    row_map = {r[0]: r for r in rows}

    results: list[SearchResult] = []
    for rid in top_rowids:
        if rid not in row_map:
            continue
        _, url, title, body = row_map[rid]
        results.append(SearchResult(
            url=url,
            title=title,
            body=body,
            score=round(combined[rid], 6),
        ))

    return results
