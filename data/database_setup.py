#!/usr/bin/env python3
"""
Build one SQLite FTS5 database per node from a JSON corpus.

Each node is defined in corpus_nodes.toml.  Selection rules per node:

  topics  — match files whose filename starts with "{topic}__"
  include — additional fnmatch glob patterns against the filename
  exclude — fnmatch patterns to remove from the combined selection

Columns: url, title, body, topic, links (JSON array or NULL).

Usage:
  python data/database_setup.py                        # build all nodes
  python data/database_setup.py --nodes climate        # one node
  python data/database_setup.py --nodes climate,tech   # several nodes
  python data/database_setup.py --dry-run              # show plan only
  python data/database_setup.py --config other.toml    # alternate config
"""

import argparse
import fnmatch
import json
import sqlite3
import sys
import tomllib
from pathlib import Path


# ── Config loading ────────────────────────────────────────────────────────────

def load_config(config_path: Path) -> dict:
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def resolve_path(raw: str, anchor: Path) -> Path:
    """Resolve a path that may be relative to the config file's directory."""
    p = Path(raw)
    return p if p.is_absolute() else anchor / p


# ── File selection ────────────────────────────────────────────────────────────

def _topic_prefix(filename: str) -> str:
    """
    Extract the topic prefix from a corpus filename.
    'climate_change__arxiv_org__abs_123.json' → 'climate_change'
    Returns '' for files that don't follow the convention.
    """
    if "__" in filename:
        return filename.split("__", 1)[0]
    return ""


def select_files(corpus_dir: Path, node: dict) -> list[Path]:
    """Return sorted list of corpus JSON files that belong to this node."""
    all_files = list(corpus_dir.glob("*.json"))

    topics           = set(node.get("topics", []))
    include_patterns = node.get("include", [])
    exclude_patterns = node.get("exclude", [])

    selected: set[Path] = set()

    # Match by filename topic-prefix — fast, no JSON reads needed.
    if topics:
        for f in all_files:
            if _topic_prefix(f.name) in topics:
                selected.add(f)

    # Additional glob patterns against the filename.
    for pattern in include_patterns:
        for f in all_files:
            if fnmatch.fnmatch(f.name, pattern):
                selected.add(f)

    # Remove excluded files.
    for pattern in exclude_patterns:
        selected = {f for f in selected if not fnmatch.fnmatch(f.name, pattern)}

    return sorted(selected)


# ── Database building ─────────────────────────────────────────────────────────

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


def build_db(db_path: Path, files: list[Path]) -> tuple[int, int]:
    """
    (Re)build the FTS5 database at db_path from the given JSON files.
    Returns (inserted, skipped) counts.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(db_path)
    try:
        con.execute("DROP TABLE IF EXISTS documents")
        con.execute(_CREATE_FTS)

        inserted = skipped = 0
        for f in files:
            try:
                doc = json.loads(f.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                print(f"  [warn] skipping {f.name}: {exc}")
                skipped += 1
                continue

            url   = doc.get("url", "").strip()
            title = doc.get("title", "").strip()
            body  = doc.get("body", "").strip()
            topic = doc.get("topic", "")

            # url and title are the minimum viable document.
            if not url or not title:
                skipped += 1
                continue

            # links is a list or absent — store as a JSON array string, or NULL.
            raw_links = doc.get("links")
            links = json.dumps(raw_links) if raw_links else None

            con.execute(_INSERT, (url, title, body, topic, links))
            inserted += 1

        con.commit()
        return inserted, skipped
    finally:
        con.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Build SQLite FTS5 databases from a JSON corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python data/database_setup.py                         # build all nodes
  python data/database_setup.py --nodes climate         # single node
  python data/database_setup.py --nodes climate,tech    # multiple nodes
  python data/database_setup.py --dry-run               # show plan without writing
  python data/database_setup.py --config custom.toml    # alternate config
""",
    )
    p.add_argument(
        "--config",
        default="corpus_nodes.toml",
        help="TOML config file (default: corpus_nodes.toml, resolved relative to this script)",
    )
    p.add_argument(
        "--nodes",
        help="Comma-separated node names to build (default: all nodes in config)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print which files would be indexed without writing any database",
    )
    args = p.parse_args()

    # Resolve config path relative to this script so the CLI works from any cwd.
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path

    if not config_path.exists():
        print(f"error: config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config    = load_config(config_path)
    base_dir  = config_path.parent
    corpus_dir = resolve_path(config.get("corpus", {}).get("dir", "corpus"), base_dir)
    db_dir     = resolve_path(config.get("databases", {}).get("dir", "dbs"),  base_dir)
    nodes      = config.get("node", [])

    if not nodes:
        print("error: no [[node]] entries found in config.", file=sys.stderr)
        sys.exit(1)

    if not corpus_dir.exists():
        print(f"error: corpus directory not found: {corpus_dir}", file=sys.stderr)
        sys.exit(1)

    # Optional filter: --nodes climate,technology
    wanted: set[str] | None = None
    if args.nodes:
        wanted = {n.strip() for n in args.nodes.split(",")}
        unknown = wanted - {n.get("name", "") for n in nodes}
        if unknown:
            known_names = ", ".join(sorted(n.get("name", "") for n in nodes))
            print(
                f"error: unknown node(s): {', '.join(sorted(unknown))}. "
                f"Available: {known_names}",
                file=sys.stderr,
            )
            sys.exit(1)

    total_inserted = 0
    for node in nodes:
        name = node.get("name", "").strip()
        if not name:
            print("[warn] skipping unnamed [[node]] entry", file=sys.stderr)
            continue
        if wanted and name not in wanted:
            continue

        files   = select_files(corpus_dir, node)
        db_path = db_dir / f"{name}.db"

        if args.dry_run:
            print(f"\n[dry-run] node={name!r}  files={len(files)}  → {db_path}")
            for f in files:
                print(f"  {f.name}")
            continue

        print(f"Building {name}.db  ({len(files)} file(s)) …", end=" ", flush=True)
        inserted, skipped = build_db(db_path, files)
        total_inserted += inserted
        note = f", {skipped} skipped" if skipped else ""
        print(f"{inserted} documents indexed{note}  [{db_path}]")

    if not args.dry_run:
        print(f"\nDone — {total_inserted} documents total across all built nodes.")


if __name__ == "__main__":
    main()
