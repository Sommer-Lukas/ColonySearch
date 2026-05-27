"""
Ground truth generation for parameter tuning.

load_ground_truth(node_dir) → test cases for one node:
    [{query_text, query_vector, relevant_urls, node_dir}, ...]

load_all_ground_truth(data_dir) → test cases across every node in data_dir/dbs/:
    Same structure; averages metrics over all topics and nodes so the
    optimiser does not overfit to a single node or topic cluster.

Relevance is defined by label overlap: docs that share at least one
ai_labels value on the same node are considered relevant to one another
(excluding the query doc itself).
"""

import json
import random
import re
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

N_QUERIES_PER_CLUSTER = 3
MIN_BODY_LEN = 200

# Shared-query generation defaults
DEFAULT_QUERIES_PER_TOPIC = 2
DEFAULT_QUERY_STYLE = "mixed"

# LLM query generation defaults
DEFAULT_LLM_QUERY_PROJECT = "project-a13aa98d-4768-4700-bc5"
DEFAULT_LLM_QUERY_LOCATION = "us-central1"
DEFAULT_LLM_QUERY_MODEL = "gemini-2.5-flash"
DEFAULT_LLM_QUERY_CACHE = "llm_queries.json"
LLM_QUERY_BODY_PREVIEW_LEN = 800
LLM_QUERY_RETRY_ATTEMPTS = 3
LLM_QUERY_RETRY_DELAY = 5.0

LLM_QUERY_SYSTEM_PROMPT = """\
You are a search user.
Given a document title and excerpt, write a single search query a human would type
to find that document.

Rules:
- 3 to 12 words
- no quotes, no bullet points, no numbering
- output ONLY the query text
"""


def _parse_labels(raw) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        labels = raw
    else:
        text = str(raw).strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = text
        labels = parsed if isinstance(parsed, list) else [parsed]

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
        labels = _parse_labels(topic)
        docs.append({
            "url": url,
            "title": title,
            "body": body,
            "labels": labels,
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
        rows = conn.execute("SELECT topic FROM documents").fetchall()
        conn.close()
        for row in rows:
            if not row:
                continue
            for label in _parse_labels(row[0]):
                topics.add(label)
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


def _clean_query_text(raw: str) -> str:
    if not raw:
        return ""
    line = raw.strip().splitlines()[0]
    line = re.sub(r"^[\s\-\*\d\.\)\(]+", "", line)
    line = line.strip("\"'`")
    line = re.sub(r"\s+", " ", line).strip()
    return line[:160]


def _load_corpus_docs_for_llm(data_dir: Path) -> list[dict]:
    corpus_dir = Path(data_dir) / "corpus"
    docs: list[dict] = []
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
        labels = _parse_labels(raw.get("ai_labels"))
        if not labels:
            labels = _parse_labels(raw.get("topic"))
        if not labels:
            continue
        docs.append({"url": url, "title": title, "body": body, "labels": labels})
    return docs


def _generate_llm_query(client, model: str, title: str, body_preview: str) -> str:
    from google.genai import types  # noqa: PLC0415

    prompt = f"Title: {title}\n\nExcerpt: {body_preview}"
    for attempt in range(1, LLM_QUERY_RETRY_ATTEMPTS + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=LLM_QUERY_SYSTEM_PROMPT,
                    max_output_tokens=64,
                    temperature=0.6,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            query = _clean_query_text(response.text or "")
            if query:
                return query
            raise ValueError("empty query")
        except Exception as exc:
            if attempt < LLM_QUERY_RETRY_ATTEMPTS:
                print(f"  [retry {attempt}/{LLM_QUERY_RETRY_ATTEMPTS}] {exc}", file=sys.stderr)
                time.sleep(LLM_QUERY_RETRY_DELAY)
            else:
                raise


def _write_llm_cache(cache_path: Path, cache_by_url: dict[str, dict]) -> None:
    payload = sorted(cache_by_url.values(), key=lambda x: x.get("url", ""))
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def generate_llm_doc_queries(
    data_dir: Path,
    n_queries: int = 50,
    cache_path: Path | None = None,
    seed: int | None = None,
    model: str = DEFAULT_LLM_QUERY_MODEL,
    project: str = DEFAULT_LLM_QUERY_PROJECT,
    location: str = DEFAULT_LLM_QUERY_LOCATION,
    allow_llm_fill: bool = False,
) -> list[dict]:
    """
    Create human-like queries from random documents and cache them on disk.

    Returns list of dicts: {query_text, topic}
    By default, uses only cached queries; set allow_llm_fill=True to call the LLM
    for missing entries.
    """
    data_dir = Path(data_dir)
    docs = _load_corpus_docs_for_llm(data_dir)
    if not docs:
        return []

    cache_path = Path(cache_path) if cache_path else data_dir / "tuning" / DEFAULT_LLM_QUERY_CACHE
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cache_by_url: dict[str, dict] = {}
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            for entry in cached if isinstance(cached, list) else []:
                url = entry.get("url") if isinstance(entry, dict) else None
                if url:
                    cache_by_url[url] = entry
        except Exception:
            cache_by_url = {}

    rng = random.Random(seed) if seed is not None else random.SystemRandom()
    doc_by_url = {d["url"]: d for d in docs}

    cached_entries = [e for e in cache_by_url.values() if e.get("query_text")]
    selected_entries: list[dict] = []
    if cached_entries:
        take = min(len(cached_entries), n_queries)
        selected_entries = rng.sample(cached_entries, k=take)
        print(f"Using {take} cached LLM queries")

    if not allow_llm_fill or len(cached_entries) >= n_queries:
        if not cached_entries:
            print("No cached LLM queries available. Set allow_llm_fill=True to generate.")
        elif len(cached_entries) < n_queries:
            print(f"Cache has {len(cached_entries)} queries; returning cached subset only.")
        queries: list[dict] = []
        for entry in selected_entries:
            topic = entry.get("topic")
            if not topic:
                doc = doc_by_url.get(entry.get("url", ""))
                if doc and doc.get("labels"):
                    topic = rng.choice(doc["labels"])
            if not topic:
                continue
            queries.append({"query_text": entry.get("query_text", ""), "topic": topic})
        return queries

    needed = n_queries - len(selected_entries)
    candidates = [d for d in docs if not cache_by_url.get(d["url"], {}).get("query_text")]
    missing_docs = rng.sample(candidates, k=min(needed, len(candidates))) if candidates else []

    updated = False
    if missing_docs:
        from google import genai  # noqa: PLC0415

        client = genai.Client(vertexai=True, project=project, location=location)
        for doc in missing_docs:
            topic = rng.choice(doc["labels"])
            query = _generate_llm_query(
                client,
                model,
                doc["title"],
                doc["body"][:LLM_QUERY_BODY_PREVIEW_LEN].strip(),
            )
            if not query:
                query = doc["title"][:160]
            cache_by_url[doc["url"]] = {
                "url": doc["url"],
                "topic": topic,
                "query_text": query,
                "title": doc["title"],
            }
            selected_entries.append(cache_by_url[doc["url"]])
            updated = True

    queries: list[dict] = []
    for entry in selected_entries:
        topic = entry.get("topic")
        if not topic:
            doc = doc_by_url.get(entry.get("url", ""))
            if doc and doc.get("labels"):
                topic = rng.choice(doc["labels"])
                entry["topic"] = topic
                updated = True
        if not topic:
            continue
        queries.append({"query_text": entry.get("query_text", ""), "topic": topic})

    if updated:
        _write_llm_cache(cache_path, cache_by_url)
        print(f"Saved LLM query cache -> {cache_path}")

    return queries


def load_ground_truth(node_dir) -> list[dict]:
    """
    Build test cases for the given node directory.

    Each test case:
        query_text    — title + first 200 chars of body
        query_vector  — pre-cached 768-dim sentence embedding (np.ndarray)
        relevant_urls — set of URLs sharing at least one label (excl. query doc)
    """
    node_dir = Path(node_dir)
    data_dir = node_dir.parent.parent

    embeddings_all = np.load(data_dir / ".embeddings_cache.npy")
    url_to_idx = _corpus_url_to_index(data_dir)
    docs = _load_node_docs(node_dir, url_to_idx)

    # Group by label
    label_to_docs: dict[str, list[dict]] = {}
    for d in docs:
        for label in d["labels"]:
            label_to_docs.setdefault(label, []).append(d)

    rng = np.random.default_rng(42)
    test_cases: list[dict] = []

    for label, cluster_docs in label_to_docs.items():
        if len(cluster_docs) < 2:
            continue

        # Deterministic shuffle then take up to N_QUERIES_PER_CLUSTER
        order = rng.permutation(len(cluster_docs))
        query_count = min(N_QUERIES_PER_CLUSTER, len(cluster_docs))
        query_indices = order[:query_count]

        for qi in query_indices:
            q = cluster_docs[qi]
            if not q["labels"]:
                continue
            relevant_urls: set[str] = set()
            for q_label in q["labels"]:
                for d in label_to_docs.get(q_label, []):
                    relevant_urls.add(d["url"])
            rel = relevant_urls - {q["url"]}
            if not rel:
                continue
            query_text = q["title"] + " " + q["body"][:200]
            query_vector = embeddings_all[q["emb_idx"]]
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

        label_to_urls: dict[str, set[str]] = {}
        label_to_embs: dict[str, list[np.ndarray]] = {}
        for d in docs:
            for label in d["labels"]:
                label_to_urls.setdefault(label, set()).add(d["url"])
                label_to_embs.setdefault(label, []).append(embeddings_all[d["emb_idx"]])

        label_to_centroid: dict[str, np.ndarray] = {}
        for label, embs in label_to_embs.items():
            if not embs:
                continue
            label_to_centroid[label] = np.mean(np.vstack(embs), axis=0).astype(np.float32)

        cases = 0
        for q in queries:
            topic = q.get("topic")
            rel = label_to_urls.get(topic, set())
            qv = label_to_centroid.get(topic, zero_vec)
            all_cases.append({
                "query_text": q.get("query_text", ""),
                "query_vector": qv,
                "relevant_urls": rel,
                "node_dir": node_dir,
            })
            cases += 1

        print(f"  {node_dir.name}: {cases} shared test cases")

    return all_cases
