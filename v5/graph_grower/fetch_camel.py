"""Fetch camel-ai science Q->solution pairs -> extractor input jsonl (targeted).

Companion to ``fetch_cot`` (which is OpenThoughts-only). OpenThoughts' Science
domain is sparse + slow to stream-filter into physics/chem/bio, so for those
question-bank domains we pull from the dedicated camel-ai datasets instead:

  physics -> camel-ai/physics      chem -> camel-ai/chemistry      bio -> camel-ai/biology   (CC-BY-NC)

Each row is a (question, worked solution) pair; the solution is a step-by-step
derivation, i.e. a CoT trace. We emit uniform ``{id, text, domain, mode:"cot"}``
rows that ``extract.py`` consumes -- identical schema to fetch_cot, so the same
extract -> stitch -> hub-wire -> apply pipeline grows the graph.

camel-ai columns: role_1, 'topic;', sub_topic, message_1 (question), message_2
(worked solution). HF dep is lazy so the transform stays import-clean + testable.
"""
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

# question-bank domain -> camel-ai dataset
DATASET_MAP = {
    "physics": "camel-ai/physics",
    "chem": "camel-ai/chemistry",
    "chemistry": "camel-ai/chemistry",
    "bio": "camel-ai/biology",
    "biology": "camel-ai/biology",
}

MIN_SOLUTION_CHARS = 200


def row_to_doc(
    row: Mapping[str, Any],
    idx: int,
    *,
    domain: str,
    keywords: Optional[Sequence[str]] = None,
    min_solution_chars: int = MIN_SOLUTION_CHARS,
) -> Optional[Dict[str, Any]]:
    """Map one camel-ai row -> an extractor doc, or None if filtered."""
    question = str(row.get("message_1") or "").strip()
    solution = str(row.get("message_2") or "").strip()
    sub_topic = str(row.get("sub_topic") or "").strip()
    if len(solution) < min_solution_chars:
        return None

    haystack = f"{question}\n{solution}\n{sub_topic}".lower()
    if keywords and not any(k.lower() in haystack for k in keywords):
        return None

    # Problem context + worked solution; solution carries the reasoning to atomize.
    text = f"Problem: {question}\n\nSolution: {solution}" if question else solution
    return {
        "id": f"camel_{domain}_{idx:06d}",
        "text": text,
        "domain": domain,
        "mode": "cot",
        "meta": {"sub_topic": sub_topic, "source": DATASET_MAP[domain]},
    }


def stream_camel(
    domain: str,
    *,
    keywords: Optional[Sequence[str]] = None,
    limit: int = 0,
    min_solution_chars: int = MIN_SOLUTION_CHARS,
    per_subtopic: int = 2,
    scan_limit: int = 6000,
) -> Iterator[Dict[str, Any]]:
    """Stream + filter a camel-ai science dataset into DIVERSE extractor docs.

    camel-ai is ordered by topic, so naive first-N streaming returns one cluster
    (e.g. 12x "Schrodinger equation") that matches almost no question-bank Qs.
    Instead: scan a pool (up to ``scan_limit`` rows), bucket by ``sub_topic``, then
    round-robin <= ``per_subtopic`` docs per sub_topic so the batch spans many
    topics. Stops scanning early once enough distinct sub_topics are collected.
    """
    from datasets import load_dataset  # lazy: only needed when actually fetching

    dataset = DATASET_MAP.get(domain.lower())
    if dataset is None:
        raise ValueError(f"unknown camel domain {domain!r}; choose from {sorted(DATASET_MAP)}")
    # NON-streaming: camel-ai is ordered by topic (one giant block per sub_topic),
    # so streaming first-N / scan gives a single cluster. Load the (cached) table
    # once and bucket all rows in memory -> true round-robin diversity, fast.
    ds = load_dataset(dataset, split="train")

    buckets: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    for idx, row in enumerate(ds):
        doc = row_to_doc(row, idx, domain=_canon(domain), keywords=keywords,
                         min_solution_chars=min_solution_chars)
        if doc is None:
            continue
        b = buckets.setdefault(doc["meta"].get("sub_topic") or "?", [])
        if len(b) < per_subtopic:
            b.append(doc)

    # round-robin: one per sub_topic per tier, so diversity comes first
    emitted = 0
    for tier in range(per_subtopic):
        for docs in buckets.values():
            if tier < len(docs):
                yield docs[tier]
                emitted += 1
                if limit and emitted >= limit:
                    return


def _canon(domain: str) -> str:
    """Normalize to the question-bank label (physics/chem/bio)."""
    d = domain.lower()
    return {"chemistry": "chem", "biology": "bio"}.get(d, d)


def write_docs(docs: Iterable[Mapping[str, Any]], path: str | Path) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as h:
        for d in docs:
            h.write(json.dumps(d, ensure_ascii=False) + "\n")
            n += 1
    return n


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch camel-ai science CoT -> extractor input jsonl (targeted).")
    parser.add_argument("--domain", required=True, choices=sorted(DATASET_MAP),
                        help="science domain: physics / chem(istry) / bio(logy)")
    parser.add_argument("--out", default=None, help="default: data/external_kb/camel_<domain>.jsonl")
    parser.add_argument("--keywords", default="",
                        help="comma-sep keywords; keep only rows mentioning one")
    parser.add_argument("--limit", type=int, default=12, help="max kept rows (0 = all)")
    parser.add_argument("--per-subtopic", type=int, default=2, help="max docs per sub_topic (diversity)")
    parser.add_argument("--scan-limit", type=int, default=6000, help="max rows to stream while pooling")
    parser.add_argument("--min-solution-chars", type=int, default=MIN_SOLUTION_CHARS)
    args = parser.parse_args(argv)

    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
    out = args.out or f"data/external_kb/camel_{_canon(args.domain)}.jsonl"
    docs = list(stream_camel(args.domain, keywords=keywords or None,
                             limit=args.limit, min_solution_chars=args.min_solution_chars,
                             per_subtopic=args.per_subtopic, scan_limit=args.scan_limit))
    n = write_docs(docs, out)
    print("Fetch camel-ai science CoT")
    print(f"  domain: {args.domain} -> {_canon(args.domain)}  keywords: {keywords or '(none)'}  limit: {args.limit}")
    print(f"  wrote {n} docs -> {out}")
    print(f"  next: python -m v5.graph_grower.extract --docs {out} --backend codex --link-graph --collect-sft "
          f"--out artifacts/graph_growth/camel_{_canon(args.domain)}_candidates.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
