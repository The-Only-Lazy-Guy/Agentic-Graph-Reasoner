"""Fetch OpenThoughts-114k CoT traces -> extractor input jsonl (targeted).

Adapter for the V5 graph grower's Source-B `cot` mode. Streams the
`open-thoughts/OpenThoughts-114k` `metadata` config (so we never download the
full 3.5 GB), filters to the question-bank domains + optional keywords, and
emits uniform `{id, text, domain, mode:"cot"}` rows that `extract.py` consumes.

The `metadata` config columns we use:
  * problem            — the question
  * deepseek_reasoning — the chain-of-thought trace (the meat)
  * deepseek_solution  — final answer
  * domain             — Math / Science / Code / Puzzles  (filter key)
  * source             — provenance

Domain mapping to the question bank (cs/physics/algo/math/sysdesign/logic/chem/bio):
  Math->math, Code->algo/cs, Science->physics/chem/bio, Puzzles->logic.
Science is not pre-split into physics/chem/bio upstream — use --keywords to
target a science sub-domain. There is no sysdesign coverage here (synthesize
sysdesign CoT separately).

The HF dependency is imported lazily (only in `stream_openthoughts`) so the
transform logic stays import-clean and unit-testable offline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

DATASET = "open-thoughts/OpenThoughts-114k"
CONFIG = "metadata"

# OpenThoughts `domain` value -> question-bank-aligned label
DOMAIN_MAP = {
    "math": "math",
    "code": "algo",
    "science": "science",   # split further via --keywords (physics/chem/bio)
    "puzzle": "logic",
    "puzzles": "logic",
}

MIN_REASONING_CHARS = 200


def row_to_doc(
    row: Mapping[str, Any],
    idx: int,
    *,
    keywords: Optional[Sequence[str]] = None,
    ot_domains: Optional[Sequence[str]] = None,
    min_reasoning_chars: int = MIN_REASONING_CHARS,
) -> Optional[Dict[str, Any]]:
    """Map one OpenThoughts metadata row -> an extractor doc, or None if filtered.

    Filters: OpenThoughts domain in ``ot_domains`` (case-insensitive) and, if
    ``keywords`` given, at least one keyword present in problem+reasoning.
    """
    ot_domain = str(row.get("domain") or "").strip().lower()
    if ot_domains and ot_domain not in {d.lower() for d in ot_domains}:
        return None

    problem = str(row.get("problem") or "").strip()
    reasoning = str(row.get("deepseek_reasoning") or "").strip()
    solution = str(row.get("deepseek_solution") or "").strip()
    if len(reasoning) < min_reasoning_chars:
        return None

    haystack = f"{problem}\n{reasoning}".lower()
    if keywords and not any(k.lower() in haystack for k in keywords):
        return None

    # Context (problem) + the reasoning trace; solution appended so the final
    # answer is available for solved_subgoal extraction.
    parts = [p for p in (problem, reasoning, solution) if p]
    text = "\n\n".join(parts)
    return {
        "id": f"ot114k_{idx:06d}",
        "text": text,
        "domain": DOMAIN_MAP.get(ot_domain, ot_domain or "external_kb"),
        "mode": "cot",
        "meta": {"ot_domain": ot_domain, "source": row.get("source")},
    }


def stream_openthoughts(
    *,
    ot_domains: Optional[Sequence[str]] = None,
    keywords: Optional[Sequence[str]] = None,
    limit: int = 0,
    min_reasoning_chars: int = MIN_REASONING_CHARS,
) -> Iterator[Dict[str, Any]]:
    """Stream + filter OpenThoughts-114k metadata into extractor docs."""
    from datasets import load_dataset  # lazy: only needed when actually fetching

    ds = load_dataset(DATASET, CONFIG, split="train", streaming=True)
    kept = 0
    for idx, row in enumerate(ds):
        doc = row_to_doc(row, idx, keywords=keywords, ot_domains=ot_domains,
                         min_reasoning_chars=min_reasoning_chars)
        if doc is None:
            continue
        yield doc
        kept += 1
        if limit and kept >= limit:
            break


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
    parser = argparse.ArgumentParser(description="Fetch OpenThoughts-114k CoT -> extractor input jsonl (targeted).")
    parser.add_argument("--out", default="data/external_kb/cot_openthoughts.jsonl")
    parser.add_argument("--domains", default="math,code,science,puzzle",
                        help="OpenThoughts domains to keep (comma-sep): math,code,science,puzzle")
    parser.add_argument("--keywords", default="",
                        help="comma-sep keywords; keep only rows mentioning one (e.g. for a science sub-domain)")
    parser.add_argument("--limit", type=int, default=200, help="max kept rows (0 = all)")
    parser.add_argument("--min-reasoning-chars", type=int, default=MIN_REASONING_CHARS)
    args = parser.parse_args(argv)

    ot_domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
    docs = stream_openthoughts(ot_domains=ot_domains, keywords=keywords or None,
                               limit=args.limit, min_reasoning_chars=args.min_reasoning_chars)
    n = write_docs(docs, args.out)
    print("Fetch OpenThoughts-114k CoT")
    print(f"  domains: {','.join(ot_domains)}  keywords: {keywords or '(none)'}  limit: {args.limit}")
    print(f"  wrote {n} docs -> {args.out}")
    print(f"  next: python -m v5.graph_grower.extract --docs {args.out} --out artifacts/graph_growth/cot_candidates.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
