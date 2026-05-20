"""
Ground truth generation for parameter tuning.

load_ground_truth(node_dir) → test cases for one node:
    [{query_text, query_vector, relevant_urls, node_dir}, ...]

load_all_ground_truth(data_dir) → test cases across every node in data_dir/dbs/:
    Same structure; averages metrics over all topics and nodes so the
    optimiser does not overfit to a single node or topic cluster.

Relevance is defined by topic label: all docs in the same topic cluster
on the same node are considered relevant to one another (excluding the
query doc itself).
"""

import json
import sqlite3
from pathlib import Path

import numpy as np

N_QUERIES_PER_CLUSTER = 3
MIN_BODY_LEN = 200

# Shared-query generation defaults
DEFAULT_QUERIES_PER_TOPIC = 2
DEFAULT_QUERY_STYLE = "mixed"


def _corpus_url_to_index(data_dir: Path) -> dict[str, int]:
    """Rebuild URL → embedding-cache row mapping by replaying pipeline sort order."""
    url_to_idx: dict[str, int] = {}
    idx = 0
    corpus_dir = data_dir / "corpus"
    for path in sorted(corpus_dir.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        url = raw.get("url", "").strip()
        title = raw.get("title", "").strip()
        body = raw.get("body", "")
        if len(body) < MIN_BODY_LEN or not url or not title:
            continue
        url_to_idx[url] = idx
        idx += 1
    return url_to_idx


def _load_node_docs(node_dir: Path, url_to_idx: dict[str, int]) -> list[dict]:
    """Load index docs for one node and attach embedding cache indices."""
    db = node_dir / "index.db"
    conn = sqlite3.connect(db)
    c = conn.cursor()
    c.execute("SELECT url, title, body, topic FROM documents")
    rows = c.fetchall()
    conn.close()

    docs = []
    for url, title, body, topic in rows:
        if url not in url_to_idx:
            continue
        docs.append({
            "url": url,
            "title": title,
            "body": body,
            "topic": topic,
            "emb_idx": url_to_idx[url],
        })
    return docs


def extract_topics(data_dir: Path) -> list[str]:
    """Collect all topic labels across every node under data_dir/dbs/."""
    data_dir = Path(data_dir)
    dbs_dir = data_dir / "dbs"
    node_dirs = sorted(p for p in dbs_dir.iterdir() if (p / "index.db").exists())

    topics: set[str] = set()
    for node_dir in node_dirs:
        conn = sqlite3.connect(node_dir / "index.db")
        rows = conn.execute("SELECT DISTINCT topic FROM documents").fetchall()
        conn.close()
        topics.update(r[0] for r in rows if r and r[0])
    return sorted(topics)


def generate_topic_queries(topics: list[str], n_per_topic: int = DEFAULT_QUERIES_PER_TOPIC, style: str = DEFAULT_QUERY_STYLE, seed: int = 42) -> list[dict]:
    """
    Create synthetic but topic-grounded queries.

    Returns list of dicts: {query_text, topic}
    """
    rng = np.random.default_rng(seed)

    short_tpl = [
        "{topic}",
        "{topic} overview",
        "{topic} basics",
        "{topic} guide",
        "{topic} examples",
    ]
    question_tpl = [
        "what is {topic}",
        "how to use {topic}",
        "why {topic} matters",
        "best practices for {topic}",
        "{topic} explained",
    ]

    queries: list[dict] = []
    for topic in topics:
        if not topic:
            continue
        tpl_pool = short_tpl + question_tpl if style == "mixed" else (question_tpl if style == "question" else short_tpl)
        picks = rng.choice(tpl_pool, size=min(n_per_topic, len(tpl_pool)), replace=False)
        for tpl in picks:
            queries.append({"query_text": tpl.format(topic=topic), "topic": topic})
    return queries


def load_ground_truth(node_dir) -> list[dict]:
    """
    Build test cases for the given node directory.

    Each test case:
        query_text    — title + first 200 chars of body
        query_vector  — pre-cached 768-dim sentence embedding (np.ndarray)
        relevant_urls — set of URLs sharing the same topic (excl. query doc)
    """
    node_dir = Path(node_dir)
    data_dir = node_dir.parent.parent

    embeddings_all = np.load(data_dir / ".embeddings_cache.npy")
    url_to_idx = _corpus_url_to_index(data_dir)
    docs = _load_node_docs(node_dir, url_to_idx)

    # Group by topic
    clusters: dict[str, list[dict]] = {}
    for d in docs:
        clusters.setdefault(d["topic"], []).append(d)

    rng = np.random.default_rng(42)
    test_cases: list[dict] = []

    for topic, cluster_docs in clusters.items():
        if len(cluster_docs) < 2:
            continue

        # Deterministic shuffle then take up to N_QUERIES_PER_CLUSTER
        order = rng.permutation(len(cluster_docs))
        query_count = min(N_QUERIES_PER_CLUSTER, len(cluster_docs))
        query_indices = order[:query_count]

        relevant_urls = {d["url"] for d in cluster_docs}

        for qi in query_indices:
            q = cluster_docs[qi]
            query_text = q["title"] + " " + q["body"][:200]
            query_vector = embeddings_all[q["emb_idx"]]
            rel = relevant_urls - {q["url"]}
            test_cases.append({
                "query_text": query_text,
                "query_vector": query_vector,
                "relevant_urls": rel,
                "node_dir": node_dir,   # needed by multi-node evaluate()
            })

    return test_cases


def load_all_ground_truth(data_dir) -> list[dict]:
    """
    Discover every node under data_dir/dbs/ and aggregate their test cases.

    Use this instead of load_ground_truth() for optimisation so that the
    objective function averages over all topics and nodes rather than
    overfitting to one.
    """
    data_dir = Path(data_dir)
    dbs_dir = data_dir / "dbs"
    node_dirs = sorted(p for p in dbs_dir.iterdir() if (p / "index.db").exists())

    all_cases: list[dict] = []
    for node_dir in node_dirs:
        cases = load_ground_truth(node_dir)
        all_cases.extend(cases)
        print(f"  {node_dir.name}: {len(cases)} test cases")

    return all_cases


def load_shared_ground_truth(data_dir, queries: list[dict]) -> list[dict]:
    """
    Build test cases for all nodes using a shared query list.

    Each query item must include: {query_text, topic}.
    Query vectors are approximated as the topic centroid embedding per node.
    """
    data_dir = Path(data_dir)
    dbs_dir = data_dir / "dbs"
    node_dirs = sorted(p for p in dbs_dir.iterdir() if (p / "index.db").exists())

    embeddings_all = np.load(data_dir / ".embeddings_cache.npy")
    url_to_idx = _corpus_url_to_index(data_dir)
    zero_vec = np.zeros((embeddings_all.shape[1],), dtype=np.float32)

    all_cases: list[dict] = []
    for node_dir in node_dirs:
        docs = _load_node_docs(node_dir, url_to_idx)
        if not docs:
            continue

        topic_to_urls: dict[str, set[str]] = {}
        topic_to_embs: dict[str, list[np.ndarray]] = {}
        for d in docs:
            topic_to_urls.setdefault(d["topic"], set()).add(d["url"])
            topic_to_embs.setdefault(d["topic"], []).append(embeddings_all[d["emb_idx"]])

        topic_to_centroid: dict[str, np.ndarray] = {}
        for topic, embs in topic_to_embs.items():
            if not embs:
                continue
            topic_to_centroid[topic] = np.mean(np.vstack(embs), axis=0).astype(np.float32)

        cases = 0
        for q in queries:
            topic = q.get("topic")
            rel = topic_to_urls.get(topic, set())
            qv = topic_to_centroid.get(topic, zero_vec)
            all_cases.append({
                "query_text": q.get("query_text", ""),
                "query_vector": qv,
                "relevant_urls": rel,
                "node_dir": node_dir,
            })
            cases += 1

        print(f"  {node_dir.name}: {cases} shared test cases")

    return all_cases
