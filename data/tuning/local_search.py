"""
Self-contained search stub for parameter tuning.

Provides the same interface as the real node.py will eventually expose:

    build_index(node_dir, link_bias, svd_dims)
    search(node_dir, query_text, query_vector, alpha) → list of {url, title, score}

Works directly on data/dbs/Node_X/ before the production node split is wired up.
The real node.py can later replace this file without changing any tuning code.
"""

import json
import re
import sqlite3
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import numpy as np
import scipy.sparse as sp
from joblib import dump, load
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from .algos import BOUNDS

_MIN_BODY_LEN = 200
_RANDOM_STATE = 42
_KNN_NEIGHBOURS = 20
_TOP_K = 50

_SVD_MIN = int(BOUNDS[1][0])
_SVD_MAX = int(BOUNDS[1][1])

# In-memory caches to avoid repeated disk I/O + expensive rebuilds
_INDEX_CACHE: dict[tuple[str, float, int], dict[str, object]] = {}
_ACTIVE_INDEX: dict[str, tuple[str, float, int]] = {}
_EMBEDDINGS_CACHE: dict[str, np.ndarray] = {}
_DOCS_CACHE: dict[str, list[dict]] = {}
_TEXT_EMB_CACHE: dict[str, np.ndarray] = {}
_LINK_FEATS_CACHE: dict[tuple[str, int], np.ndarray | None] = {}


def resolve_svd_dims(value: float) -> int:
    """Round to int with half-up behavior and clip to bounds."""
    rounded = int(np.floor(float(value) + 0.5))
    return int(np.clip(rounded, _SVD_MIN, _SVD_MAX))


def clear_index_cache() -> None:
    """Clear in-memory index caches (useful after data changes)."""
    _INDEX_CACHE.clear()
    _ACTIVE_INDEX.clear()
    _EMBEDDINGS_CACHE.clear()
    _DOCS_CACHE.clear()
    _TEXT_EMB_CACHE.clear()
    _LINK_FEATS_CACHE.clear()


# ── URL normalisation (matches node_pipeline_setup.py) ───────────────────────

def _norm_url(u: str) -> str:
    if not u:
        return ""
    u = u.strip().rstrip(").,;:!?\"'")
    if u.startswith("www."):
        u = "http://" + u
    try:
        parsed = urlparse(u)
    except Exception:
        return ""
    if not parsed.netloc:
        return ""
    scheme = "https" if parsed.scheme in ("http", "https") else "https"
    return urlunparse((scheme, parsed.netloc.lower(), parsed.path.rstrip("/"), "", "", ""))


# ── Corpus index: URL → embedding-cache row ──────────────────────────────────

@lru_cache(maxsize=4)
def _url_to_cache_index(corpus_dir: str) -> dict[str, int]:
    """
    Replay the sorted-file pass from node_pipeline_setup.py to recover
    the URL → row-index mapping for .embeddings_cache.npy.
    """
    mapping: dict[str, int] = {}
    idx = 0
    for path in sorted(Path(corpus_dir).glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        url = raw.get("url", "").strip()
        body = raw.get("body", "")
        title = raw.get("title", "").strip()
        if len(body) < _MIN_BODY_LEN or not url or not title:
            continue
        mapping[url] = idx
        idx += 1
    return mapping


# ── Link features ─────────────────────────────────────────────────────────────

def _link_features(docs: list[dict], svd_dims: int) -> np.ndarray | None:
    n = len(docs)
    url_to_i = {d["url"]: i for i, d in enumerate(docs)}
    edges = []
    for i, doc in enumerate(docs):
        for raw in doc.get("links", []):
            target = _norm_url(raw if isinstance(raw, str) else "")
            if target and target in url_to_i and url_to_i[target] != i:
                edges.append((i, url_to_i[target]))
    if not edges:
        return None
    rows, cols = zip(*edges)
    adj = sp.coo_matrix(
        (np.ones(len(edges), dtype=np.float32), (rows, cols)), shape=(n, n)
    )
    adj = (adj + adj.T).tocsr()
    dims = min(int(svd_dims), max(1, n - 1))
    feats = TruncatedSVD(n_components=dims, random_state=_RANDOM_STATE).fit_transform(adj)
    return normalize(feats)


# ── Public API ────────────────────────────────────────────────────────────────

def build_index(node_dir, link_bias: float, svd_dims: int) -> None:
    """Rebuild the KNN model with the given hyperparameters."""
    node_dir = Path(node_dir)
    svd_dims = resolve_svd_dims(svd_dims)
    node_key = str(node_dir)
    cache_key = (node_key, float(link_bias), svd_dims)
    _ACTIVE_INDEX[node_key] = cache_key
    if cache_key in _INDEX_CACHE:
        return
    data_dir = node_dir.parent.parent

    embeddings_all = _EMBEDDINGS_CACHE.get(str(data_dir))
    if embeddings_all is None:
        embeddings_all = np.load(data_dir / ".embeddings_cache.npy")
        _EMBEDDINGS_CACHE[str(data_dir)] = embeddings_all

    docs = _DOCS_CACHE.get(node_key)
    if docs is None:
        url_idx = _url_to_cache_index(str(data_dir / "corpus"))
        conn = sqlite3.connect(node_dir / "index.db")
        rows = conn.execute("SELECT url, title, links FROM documents").fetchall()
        conn.close()

        docs = []
        for url, title, links_json in rows:
            if url not in url_idx:
                continue
            docs.append({
                "url": url,
                "title": title,
                "links": json.loads(links_json) if links_json else [],
                "_row": url_idx[url],
            })
        _DOCS_CACHE[node_key] = docs

    if not docs:
        raise ValueError(f"No indexable docs in {node_dir}")

    text_emb = _TEXT_EMB_CACHE.get(node_key)
    if text_emb is None:
        text_emb = normalize(embeddings_all[[d["_row"] for d in docs]])
        _TEXT_EMB_CACHE[node_key] = text_emb

    link_key = (node_key, svd_dims)
    if link_key in _LINK_FEATS_CACHE:
        link_feats = _LINK_FEATS_CACHE[link_key]
    else:
        link_feats = _link_features(docs, svd_dims)
        _LINK_FEATS_CACHE[link_key] = link_feats

    X = np.hstack([text_emb, link_bias * link_feats]) if (link_feats is not None and link_bias > 0) else text_emb

    k = min(_KNN_NEIGHBOURS, len(docs))
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
    nn.fit(X)
    urls = [d["url"] for d in docs]
    titles = [d["title"] for d in docs]

    _INDEX_CACHE[cache_key] = {"nn": nn, "urls": urls, "titles": titles}

    dump(nn, node_dir / "knn_model.joblib")
    (node_dir / "knn_metadata.json").write_text(
        json.dumps({"urls": urls, "titles": titles})
    )


def search(node_dir, query_text: str, query_vector, alpha: float) -> list[dict]:
    """Return up to _TOP_K results; score = alpha*bm25_norm + (1-alpha)*knn_norm."""
    node_dir = Path(node_dir)
    qv = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)

    bm25 = _bm25(node_dir, query_text)
    knn = _knn(node_dir, qv)

    url_title: dict[str, str] = {}
    b_scores: dict[str, float] = {}
    k_scores: dict[str, float] = {}
    for r in bm25:
        url_title[r["url"]] = r["title"]
        b_scores[r["url"]] = r["score"]
    for r in knn:
        url_title[r["url"]] = r["title"]
        k_scores[r["url"]] = r["score"]

    all_urls = set(b_scores) | set(k_scores)
    if not all_urls:
        return []

    bn = _minmax(b_scores)
    kn = _minmax(k_scores)
    results = [
        {"url": u, "title": url_title[u], "score": alpha * bn.get(u, 0.0) + (1 - alpha) * kn.get(u, 0.0)}
        for u in all_urls
    ]
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:_TOP_K]


# ── Internals ─────────────────────────────────────────────────────────────────

def _bm25(node_dir: Path, query_text: str) -> list[dict]:
    conn = sqlite3.connect(node_dir / "index.db")
    try:
        rows = conn.execute(
            "SELECT url, title, -rank FROM documents WHERE documents MATCH ? ORDER BY rank LIMIT ?",
            (_fts_escape(query_text), _TOP_K),
        ).fetchall()
        return [{"url": r[0], "title": r[1], "score": float(r[2])} for r in rows]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def _knn(node_dir: Path, qv: np.ndarray) -> list[dict]:
    node_key = str(node_dir)
    active_key = _ACTIVE_INDEX.get(node_key)
    if active_key is not None:
        entry = _INDEX_CACHE.get(active_key)
        if entry is not None:
            nn = entry["nn"]
            urls = entry["urls"]
            titles = entry["titles"]
            if qv.shape[1] < nn.n_features_in_:
                qv = np.hstack([qv, np.zeros((1, nn.n_features_in_ - qv.shape[1]), dtype=np.float32)])
            dists, idxs = nn.kneighbors(qv, n_neighbors=min(_TOP_K, nn.n_samples_fit_))
            return [
                {"url": urls[i], "title": titles[i], "score": float(1.0 - d)}
                for d, i in zip(dists[0], idxs[0])
            ]

    model_path = node_dir / "knn_model.joblib"
    meta_path = node_dir / "knn_metadata.json"
    if not model_path.exists() or not meta_path.exists():
        return []
    nn = load(model_path)
    meta = json.loads(meta_path.read_text())
    if qv.shape[1] < nn.n_features_in_:
        qv = np.hstack([qv, np.zeros((1, nn.n_features_in_ - qv.shape[1]), dtype=np.float32)])
    dists, idxs = nn.kneighbors(qv, n_neighbors=min(_TOP_K, nn.n_samples_fit_))
    return [
        {"url": meta["urls"][i], "title": meta["titles"][i], "score": float(1.0 - d)}
        for d, i in zip(dists[0], idxs[0])
    ]


def _minmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    mn, mx = min(scores.values()), max(scores.values())
    if mx == mn:
        return {k: 1.0 for k in scores}
    return {k: (v - mn) / (mx - mn) for k, v in scores.items()}


def _fts_escape(q: str) -> str:
    tokens = q.split()
    return " ".join(f'"{t}"' for t in tokens) if tokens else '""'
