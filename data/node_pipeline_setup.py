#!/usr/bin/env python3
"""
Build a per-node search pipeline from a JSON corpus.

Each node gets its own sub-folder under dbs/:
  dbs/Node_1/
    index.db   — SQLite FTS5 full-text index
    knn.npz    — precomputed KNN neighbours (indices + distances, cross-node)

Pipeline (mirrors data_preperation.ipynb):
  1. Load all corpus JSON files
  2. Clean & normalise text (mojibake, HTML, boilerplate)
  3. Filter short / broken documents (MIN_BODY_LEN)
  4. Encode with sentence-transformers (cached in .embeddings_cache.npy)
  5. Build link-graph features via truncated SVD on the document adjacency matrix
  6. Combine: X = [normalize(embeddings) | LINK_BIAS_WEIGHT * link_features]
  7. KMeans(k=N) to assign documents to nodes
  8. Auto-select KNN-k by scanning KNN_K_RANGE and maximising neighbour purity
     (fraction of k-nearest-neighbours that share the same cluster label).
     Override with --knn-k to skip the scan, exactly like FORCE_K in the notebook.
  9. Per node: write FTS5 SQLite + precomputed KNN (neighbours may span other nodes)

Node assignment is content-based (KMeans on embeddings), not random and not by
filename prefix.  Documents with similar semantics end up on the same node, which
typically correlates with topic but can merge overlapping topics or split noisy ones.

Usage:
  python data/node_pipeline_setup.py --nodes 5
  python data/node_pipeline_setup.py --nodes 5 --knn-k 12   # skip auto-select
  python data/node_pipeline_setup.py --nodes 5 --dry-run
  python data/node_pipeline_setup.py --nodes 5 --embed-model all-MiniLM-L6-v2
"""

import argparse
import html
import json
import re
import sqlite3
import sys
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

# ── Defaults (mirror data_preperation.ipynb config cell) ─────────────────────
MIN_BODY_LEN     = 150
LINK_BIAS_WEIGHT = 0.5
LINK_SVD_DIMS    = 61
EMBED_MODEL      = "all-mpnet-base-v2"
RANDOM_STATE     = 42

# Scan this range to auto-select KNN-k (mirrors K_RANGE / silhouette scan in notebook).
# Pass --knn-k on the CLI to skip the scan and lock a specific value (like FORCE_K).
KNN_K_RANGE = (3, 20)   # (min_k, max_k) inclusive

# ── Text cleaning (identical to notebook cell 2) ─────────────────────────────

_MOJIBAKE = {
    "Â¶": "¶",
    "â€™": "'",
    "â€œ": '"',
    "â€": '"',
    "â€“": "–",
    "â€”": "—",
    "â€¦": "…",
    "Â©": "©",
    "Â®": "®",
    "Â·": "·",
    "Â ": " ",
}

_BOILERPLATE = re.compile(
    r"Loading component\.{0,3}"
    r"|Past Events\s+View Upcoming"
    r"|This content is now at"
    r"|Please enable JavaScript"
    r"|To view this (video|page) please enable"
    r"|Cookie\s+(Policy|Settings|Notice)",
    re.IGNORECASE,
)

_LINK_PATTERN = re.compile(r"https?://[^\s)>\"]+")


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    for bad, good in _MOJIBAKE.items():
        text = text.replace(bad, good)
    text = re.sub(r"<[^>]{1,200}>", " ", text)
    text = _BOILERPLATE.sub(" ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalize_labels(raw_labels) -> list[str]:
    if raw_labels is None:
        return []
    if isinstance(raw_labels, list):
        labels = raw_labels
    elif isinstance(raw_labels, str):
        labels = [raw_labels]
    else:
        labels = [str(raw_labels)]

    seen: set[str] = set()
    normalized: list[str] = []
    for label in labels:
        if label is None:
            continue
        text = str(label).strip().lower()
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return normalized


def _normalize_url(u: str) -> str:
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
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/")
    return urlunparse((scheme, netloc, path, "", "", ""))


# ── Corpus loading & cleaning ─────────────────────────────────────────────────

def load_corpus(corpus_dir: Path) -> list[dict]:
    """Load, clean, and filter corpus JSON files. Returns list of doc dicts."""
    docs = []
    for path in sorted(corpus_dir.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  [warn] skipping {path.name}: {exc}", file=sys.stderr)
            continue

        url   = raw.get("url", "").strip()
        title = raw.get("title", "").strip()
        body  = raw.get("body", "")
        raw_labels = raw.get("ai_labels")
        labels = _normalize_labels(raw_labels)
        if not labels:
            labels = _normalize_labels(raw.get("topic", ""))
        topic = json.dumps(labels) if labels else ""
        links = raw.get("links")

        title_clean = clean_text(title)
        body_clean  = clean_text(body)

        if len(body_clean) < MIN_BODY_LEN:
            continue
        if not url or not title_clean:
            continue

        docs.append({
            "file":        path.name,
            "url":         url,
            "title":       title_clean,
            "body":        body_clean,
            "topic":       topic,
            "links":       links,
        })

    return docs


# ── Embeddings ────────────────────────────────────────────────────────────────

def build_embeddings(docs: list[dict], cache_path: Path, model_name: str) -> np.ndarray:
    """Return (N, D) sentence embeddings, loading from cache when valid."""
    texts = [d["title"] + " " + d["body"] for d in docs]

    if cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape[0] == len(texts):
            print(f"Loaded cached embeddings: {cached.shape}  [{cache_path}]")
            return cached
        print(f"Cache size mismatch ({cached.shape[0]} vs {len(texts)}) — re-encoding.")

    from sentence_transformers import SentenceTransformer
    print(f"Encoding {len(texts)} documents with '{model_name}' …")
    model      = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=16)
    np.save(cache_path, embeddings)
    print(f"Saved embeddings -> {cache_path}  shape={embeddings.shape}")
    return embeddings


# ── Link-graph features (mirrors notebook cell 4) ────────────────────────────

def build_link_features(docs: list[dict]) -> np.ndarray | None:
    """Build SVD-reduced link-graph features. Returns None if no edges found."""
    n = len(docs)

    url_to_idx: dict[str, int] = {}
    for i, doc in enumerate(docs):
        norm = _normalize_url(doc["url"])
        if norm:
            url_to_idx[norm] = i

    edges: list[tuple[int, int]] = []
    for i, doc in enumerate(docs):
        for raw in _LINK_PATTERN.findall(doc["body"]):
            norm = _normalize_url(raw)
            if norm and norm in url_to_idx:
                j = url_to_idx[norm]
                if j != i:
                    edges.append((i, j))

    if not edges:
        return None

    rows, cols = zip(*edges)
    data = np.ones(len(edges), dtype=np.float32)
    adj  = sp.coo_matrix((data, (rows, cols)), shape=(n, n))
    adj  = (adj + adj.T).tocsr()

    n_components = min(LINK_SVD_DIMS, max(1, n - 1))
    svd          = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    feats        = svd.fit_transform(adj)
    feats        = normalize(feats)

    print(f"Link features: {feats.shape}  ({len(edges)} edges)")
    return feats


# ── Combined feature matrix ───────────────────────────────────────────────────

def build_feature_matrix(embeddings: np.ndarray, link_features: np.ndarray | None) -> np.ndarray:
    """
    Build the combined feature matrix used for both KMeans and KNN.

    Text embeddings + weighted link-graph features are concatenated so that
    link-connected documents are pulled closer together in the index.

    At query time the caller pads the query vector with zeros for the link
    dimensions — a query has no link-graph position, so zero is the neutral
    value.  Because the text component (768 dims) dominates over the link
    component (32 dims × 0.3 weight), text similarity still drives results;
    the link features act as a soft structural bias between documents.
    The node can compute the required padding from nn.n_features_in_ at runtime.
    """
    text_emb = normalize(embeddings)
    if link_features is not None and LINK_BIAS_WEIGHT > 0:
        X = np.hstack([text_emb, LINK_BIAS_WEIGHT * link_features])
        print(f"Feature matrix (text + link bias={LINK_BIAS_WEIGHT}): {X.shape}  "
              f"(query padding at runtime: {X.shape[1] - text_emb.shape[1]} zeros)")
    else:
        X = text_emb
        print(f"Feature matrix (text only): {X.shape}")
    return X


# ── SQLite FTS5 ───────────────────────────────────────────────────────────────

_CREATE_FTS = """
    CREATE VIRTUAL TABLE documents USING fts5(
        url   UNINDEXED,
        title,
        body,
        topic UNINDEXED,
        links UNINDEXED,
        tokenize = 'porter unicode61'
    )
"""
_INSERT = "INSERT INTO documents VALUES (?, ?, ?, ?, ?)"


def write_fts_db(db_path: Path, node_docs: list[dict]) -> int:
    """Write FTS5 SQLite database for a node. Returns number of inserted docs."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    try:
        con.execute("DROP TABLE IF EXISTS documents")
        con.execute(_CREATE_FTS)
        for doc in node_docs:
            raw_links = doc.get("links")
            links     = json.dumps(raw_links) if raw_links else None
            con.execute(_INSERT, (doc["url"], doc["title"], doc["body"], doc["topic"], links))
        con.commit()
    finally:
        con.close()
    return len(node_docs)


# ── KNN (local per node) ──────────────────────────────────────────────────────

def select_local_knn_k(X_node: np.ndarray, node_name: str) -> int:
    """
    Auto-select KNN-k for one node using an elbow on the k-distance curve.

    For each candidate k, compute the mean cosine distance to the k-th nearest
    neighbour within this node's documents.  The curve rises slowly at first then
    accelerates once k reaches dissimilar docs — the elbow (maximum second
    derivative) marks the natural cut-off.

    Mirrors the inertia-elbow / silhouette scan the notebook uses for KMeans k,
    but applied locally so nodes with different sizes can get different k values.
    """
    k_min, k_max = KNN_K_RANGE
    # Need at least k_max+1 docs to query k_max neighbours + self.
    max_k = min(k_max, len(X_node) - 1)

    if max_k < k_min:
        # Node too small to scan — fall back to the minimum.
        print(f"  [{node_name}] too few docs for k-scan, using k={k_min}")
        return k_min

    # One fit, slice distances per k to avoid repeated fitting.
    nn = NearestNeighbors(n_neighbors=max_k + 1, metric="cosine", algorithm="brute")
    nn.fit(X_node)
    distances, _ = nn.kneighbors(X_node)   # shape (n, max_k+1); col 0 = self (dist=0)

    ks         = list(range(k_min, max_k + 1))
    mean_dists = [distances[:, k].mean() for k in ks]   # mean dist to k-th neighbour

    # Find elbow via maximum second derivative (highest acceleration in the curve).
    if len(mean_dists) >= 3:
        diffs2  = np.diff(mean_dists, n=2)
        elbow_i = int(np.argmax(diffs2)) + 1   # +1 accounts for the two diff offsets
        best_k  = ks[min(elbow_i, len(ks) - 1)]
    else:
        best_k = ks[0]

    print(f"  [{node_name}] auto-selected KNN-k={best_k}  "
          f"(k-distance elbow, range {k_min}..{max_k})")
    return best_k


def build_and_save_knn(node_dir: Path, X_node: np.ndarray, k: int) -> None:
    """
    Fit and save the local KNN for one node.

    X_node — combined embeddings for this node (text + link features).
             At query time, pad the query vector with zeros to match X_node's
             column count:  np.hstack([query_vec, np.zeros((1, nn.n_features_in_ - query_vec.shape[1]))])
    """
    from joblib import dump

    nn = NearestNeighbors(n_neighbors=min(k, len(X_node)), metric="cosine", algorithm="brute")
    nn.fit(X_node)
    dump(nn, node_dir / "knn_model.joblib")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Build per-node FTS5 + KNN search pipeline from a JSON corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python data/node_pipeline_setup.py --nodes 5
  python data/node_pipeline_setup.py --nodes 5 --knn-k 12   # skip auto-select
  python data/node_pipeline_setup.py --nodes 5 --dry-run
  python data/node_pipeline_setup.py --nodes 5 --embed-model all-MiniLM-L6-v2
""",
    )
    p.add_argument("--nodes",       type=int, required=True,
                   help="Number of nodes to create (Node_1 … Node_N)")
    p.add_argument("--knn-k",       type=int, default=None,
                   help=(
                       f"Fix KNN neighbours per document and skip auto-selection "
                       f"(scans {KNN_K_RANGE[0]}..{KNN_K_RANGE[1]} by default, "
                       f"like FORCE_K in the notebook)"
                   ))
    p.add_argument("--corpus-dir",  default=None,
                   help="Path to corpus directory (default: <script_dir>/corpus)")
    p.add_argument("--db-dir",      default=None,
                   help="Output directory for node sub-folders (default: <script_dir>/dbs)")
    p.add_argument("--embed-cache", default=None,
                   help="Path to embedding cache .npy (default: <script_dir>/.embeddings_cache.npy)")
    p.add_argument("--embed-model", default=EMBED_MODEL,
                   help=f"sentence-transformers model name (default: {EMBED_MODEL})")
    p.add_argument("--dry-run",     action="store_true",
                   help="Show plan without writing any files")
    args = p.parse_args()

    script_dir  = Path(__file__).parent
    corpus_dir  = Path(args.corpus_dir)  if args.corpus_dir  else script_dir / "corpus"
    db_dir      = Path(args.db_dir)      if args.db_dir      else script_dir / "dbs"
    embed_cache = Path(args.embed_cache) if args.embed_cache else script_dir / ".embeddings_cache.npy"

    if not corpus_dir.exists():
        print(f"error: corpus directory not found: {corpus_dir}", file=sys.stderr)
        sys.exit(1)

    n_nodes   = args.nodes
    force_knn = args.knn_k  # None = auto-select

    # ── 1. Load & clean corpus ────────────────────────────────────────────────
    print(f"Loading corpus from {corpus_dir} …")
    docs = load_corpus(corpus_dir)
    print(f"Clean documents: {len(docs)}")

    if len(docs) < n_nodes:
        print(f"error: only {len(docs)} documents but --nodes={n_nodes}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        knn_desc = f"k={force_knn} (fixed)" if force_knn else f"auto per node (elbow on k-distance, range {KNN_K_RANGE[0]}..{KNN_K_RANGE[1]})"
        print(f"\n[dry-run] Would create {n_nodes} node(s) in {db_dir}:")
        for i in range(1, n_nodes + 1):
            node_dir = db_dir / f"Node_{i}"
            print(f"  {node_dir}/")
            print(f"    index.db         — FTS5 SQLite (local documents only)")
            print(f"    knn_model.joblib — local NearestNeighbors for query-time search ({knn_desc})")
        print(f"\nCorpus: {len(docs)} clean documents, whole corpus split across {n_nodes} node(s).")
        print("Split: KMeans on sentence embeddings + link-graph features (content-based, deterministic).")
        return

    # ── 2. Embeddings ─────────────────────────────────────────────────────────
    embeddings = build_embeddings(docs, embed_cache, args.embed_model)

    # ── 3. Link-graph features ────────────────────────────────────────────────
    link_features = build_link_features(docs)

    # ── 4. Feature matrix — used for both KMeans and KNN ─────────────────────
    X = build_feature_matrix(embeddings, link_features)

    # ── 5. KMeans assignment — whole corpus, each doc goes to exactly one node ─
    print(f"Clustering {len(docs)} documents into {n_nodes} nodes …")
    km     = KMeans(n_clusters=n_nodes, random_state=RANDOM_STATE, n_init="auto")
    labels = km.fit_predict(X)

    node_to_indices: dict[int, list[int]] = {i: [] for i in range(n_nodes)}
    for doc_idx, cluster_id in enumerate(labels):
        node_to_indices[cluster_id].append(doc_idx)

    # ── 6. Per-node output ────────────────────────────────────────────────────
    for cluster_id in range(n_nodes):
        node_name        = f"Node_{cluster_id + 1}"
        node_dir         = db_dir / node_name
        node_indices  = node_to_indices[cluster_id]
        node_docs     = [docs[i] for i in node_indices]
        X_node        = X[node_indices]

        node_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{node_name}  ({len(node_docs)} docs)  →  {node_dir}")

        inserted = write_fts_db(node_dir / "index.db", node_docs)
        print(f"  index.db         — {inserted} documents indexed")

        if force_knn is not None:
            knn_k = force_knn
        else:
            knn_k = select_local_knn_k(X_node, node_name)

        build_and_save_knn(node_dir, X_node, knn_k)
        print(f"  knn_model.joblib — fitted on {len(node_docs)} local docs, k={knn_k}")

    print(f"\nDone — {n_nodes} node(s) written to {db_dir}")


if __name__ == "__main__":
    main()
