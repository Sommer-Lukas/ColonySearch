#!/usr/bin/env python3
"""
Quick search demo.

    python tests/demo_search.py --node Node_2 --query "carbon capture CO2"
    python tests/demo_search.py --node Node_2 --query "carbon capture CO2" --output 25
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sentence_transformers import SentenceTransformer
from search import search

parser = argparse.ArgumentParser()
parser.add_argument("--node",  required=True, help="e.g. Node_2")
parser.add_argument("--query", required=True)
parser.add_argument("--output", type=int, default=10, help="number of results to show")
parser.add_argument("--alpha", type=float, default=0.5)
args = parser.parse_args()

node_dir = ROOT / "data" / "dbs" / args.node
if not node_dir.exists():
    sys.exit(f"Node not found: {node_dir}")

print("Encoding query …")
model = SentenceTransformer("all-mpnet-base-v2")
vec   = model.encode(args.query, show_progress_bar=False)

results = search(node_dir, args.query, vec, alpha=args.alpha, top_k=args.output)

print(f"\nResults for: \"{args.query}\"  [{args.node}, alpha={args.alpha}]\n")
for i, r in enumerate(results, 1):
    print(f"{i:2}. [{r.score:.4f}] {r.title}")
    print(f"    {r.url}")
    print(f"    {r.body[:200].replace(chr(10), ' ').strip()} …")
    print()
