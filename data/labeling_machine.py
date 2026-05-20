#!/usr/bin/env python3
"""
labeling_machine.py — Add AI-generated fine-grained labels to corpus JSON files.

Reads every *.json in the corpus directory, asks Gemini to classify the document,
and appends the result to a new `ai_labels` array in each file.  The existing
`topic` field (and any other fields) are never touched.  Running the script
multiple times with different models accumulates labels in the array rather than
overwriting them.

Only the document title and body excerpt are sent to Gemini — no filename, URL,
or existing topic, so the model classifies purely on content.

Usage:
  python data/labeling_machine.py
  python data/labeling_machine.py --corpus-dir data/corpus
  python data/labeling_machine.py --dry-run       # preview a random sample
  python data/labeling_machine.py --overwrite     # clear ai_labels and re-run

Vertex AI Application Default Credentials are used automatically (gcloud auth
application-default login).  No API key needed.
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

# ── Config defaults ────────────────────────────────────────────────────────────
DEFAULT_PROJECT  = "project-a13aa98d-4768-4700-bc5"
DEFAULT_LOCATION = "us-central1"
DEFAULT_MODEL    = "gemini-2.5-flash"
BODY_PREVIEW_LEN = 800   # chars of body sent to Gemini
DRY_RUN_SAMPLE   = 10    # random files shown in --dry-run preview
RETRY_ATTEMPTS   = 3
RETRY_DELAY      = 5.0   # seconds between retries

SYSTEM_PROMPT = """\
You are a document tagger for a research corpus.
Given a document title and a short excerpt, output 3 to 8 topic tags.

Always include at least one broad domain tag such as:
  medicine, programming, machine learning, climate change, physics, biology,
  chemistry, economics, history, politics, space, energy, law, mathematics,
  engineering, environment, agriculture, finance, psychology, religion

Then add more specific tags that describe the document's exact content.
Prefer simple, common words that will match other documents on the same topic —
reusing the same tag across documents is intentional and good.
Tags are lowercase, 1–3 words, spaces allowed (no hyphens).

Output ONLY a comma-separated list of tags — no numbering, no explanation, no quotes.
"""


def _parse_labels(raw: str) -> list[str]:
    labels = []
    for part in raw.split(","):
        tag = part.strip().lower()
        tag = re.sub(r"\s+", " ", tag)          # collapse whitespace
        tag = re.sub(r"[^a-z0-9 ]", "", tag).strip()  # keep letters, digits, spaces
        if tag:
            labels.append(tag)
    return labels or ["unlabeled"]


def classify(client, model: str, title: str, body_preview: str) -> list[str]:
    from google.genai import types  # noqa: PLC0415

    # Only content — no filename, URL, or existing topic — to keep labels unbiased.
    prompt = f"Title: {title}\n\nExcerpt: {body_preview}"
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=192,
                    temperature=0.2,
                    # Disable thinking so response.text is always populated
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            return _parse_labels(response.text)
        except Exception as exc:
            if attempt < RETRY_ATTEMPTS:
                print(f"  [retry {attempt}/{RETRY_ATTEMPTS}] {exc}", file=sys.stderr)
                time.sleep(RETRY_DELAY)
            else:
                raise


def main() -> None:
    p = argparse.ArgumentParser(
        description="Append a Gemini-generated label to the ai_labels array in each corpus JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python data/labeling_machine.py
  python data/labeling_machine.py --dry-run
  python data/labeling_machine.py --model gemini-2.5-pro --overwrite
""",
    )
    p.add_argument("--corpus-dir", default=None,
                   help="Path to corpus directory (default: <script_dir>/corpus)")
    p.add_argument("--model",      default=DEFAULT_MODEL,
                   help=f"Gemini model ID (default: {DEFAULT_MODEL})")
    p.add_argument("--project",    default=DEFAULT_PROJECT,
                   help=f"GCP project ID (default: {DEFAULT_PROJECT})")
    p.add_argument("--location",   default=DEFAULT_LOCATION,
                   help=f"Vertex AI region (default: {DEFAULT_LOCATION})")
    p.add_argument("--dry-run",    action="store_true",
                   help="Call the API on a random sample and show the real labels, without writing files")
    p.add_argument("--overwrite",  action="store_true",
                   help="Clear existing ai_labels and re-label every file from scratch")
    args = p.parse_args()

    script_dir = Path(__file__).parent
    corpus_dir = Path(args.corpus_dir) if args.corpus_dir else script_dir / "corpus"

    if not corpus_dir.exists():
        print(f"error: corpus directory not found: {corpus_dir}", file=sys.stderr)
        sys.exit(1)

    files = sorted(corpus_dir.glob("*.json"))
    if not files:
        print("No JSON files found.", file=sys.stderr)
        sys.exit(1)

    todo, skip = [], []
    for f in files:
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            print(f"  [warn] skipping unreadable file: {f.name}", file=sys.stderr)
            continue
        # Already has labels → skip unless --overwrite clears the list
        if not args.overwrite and doc.get("ai_labels", []):
            skip.append(f)
        else:
            todo.append((f, doc))

    print(f"Corpus: {len(files)} files — {len(todo)} to label, {len(skip)} already done")

    if args.dry_run:
        sample_size = min(DRY_RUN_SAMPLE, len(todo))
        sample = random.sample(todo, sample_size)
        print(f"\n[dry-run] Calling API on {sample_size} random files (no writes):\n")

        from google import genai  # noqa: PLC0415
        client = genai.Client(vertexai=True, project=args.project, location=args.location)

        for f, doc in sample:
            title        = doc.get("title", "").strip()
            body_preview = doc.get("body", "")[:BODY_PREVIEW_LEN].strip()
            try:
                labels = classify(client, args.model, title, body_preview)
            except Exception as exc:
                labels = [f"ERROR: {exc}"]
            print(f"  {title[:70]}")
            print(f"  → {labels}\n")
        return

    if not todo:
        print("Nothing to do.")
        return

    from google import genai  # noqa: PLC0415

    client = genai.Client(
        vertexai=True,
        project=args.project,
        location=args.location,
    )

    ok, failed = 0, 0
    for i, (path, doc) in enumerate(todo, start=1):
        title        = doc.get("title", "").strip()
        body_preview = doc.get("body", "")[:BODY_PREVIEW_LEN].strip()

        print(f"[{i}/{len(todo)}] {title[:70]}", end="  ", flush=True)
        try:
            label = classify(client, args.model, title, body_preview)

            # Extend the array; clear first if --overwrite
            if args.overwrite:
                doc["ai_labels"] = label
            else:
                existing = doc.get("ai_labels", [])
                doc["ai_labels"] = existing + [t for t in label if t not in existing]

            path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"→ {label}")
            ok += 1
        except Exception as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            failed += 1

    print(f"\nDone — {ok} labeled, {failed} failed, {len(skip)} skipped")


if __name__ == "__main__":
    main()
