"""Fetch nvidia/Open-SWE-Traces -> extractor input jsonl (targeted).

Adapter for the V5 graph grower's Source-B `cot` mode, mirroring fetch_cot.py's pattern for a real,
long-horizon, AGENTIC domain (multi-step tool-use bug fixes) instead of math CoT.

Real inspection of this dataset (not assumed) found a severe signal-to-noise problem: every trajectory's
`role: system` turn is an 8049-char agent-role/instructions block, BYTE-IDENTICAL across completely
different repos (confirmed: python-attrs/attrs and altair-viz/altair both produced exactly 8049 chars) --
pure boilerplate, zero per-example learning signal, but the single largest token cost in the raw trace.
The `tools` field (JSON schema of available functions) is the same story -- fixed across every row. Feeding
either into a small on-device model's limited context/graph-learning pipeline per example would waste most
of the budget on text that never varies and teaches nothing.

The REAL per-example signal lives elsewhere:
  * the first `user` turn's <issue_description> -- the actual bug being fixed (real, varies per example)
  * each `assistant` turn's `reasoning_content` -- genuine per-step CoT (content is EMPTY on assistant
    turns in this dataset; everything routes through reasoning_content + tool_calls instead)
  * each `assistant` turn's `tool_calls` -- the concrete action taken (function name + JSON arguments,
    structured, not free text)
  * `metadata.reference_patch` -- the real fix as a unified diff (verifiable ground truth)
  * `resolved` -- the dataset's own real success label (1/0/-1 unknown)

row_to_doc keeps only these, dropping the static system prompt and tools schema entirely.

Columns used (config="openhands"|"sweagent", split="minimax_m25"|"qwen35_122b"):
  * trajectory       -- list of {role, content, reasoning_content, think, tool_calls}
  * metadata          -- {category, reference_patch, model_patch}
  * resolved          -- 1 / 0 / -1
  * instance_id, repo, language, license

The HF dependency is imported lazily (only in `stream_swe_traces`) so the transform logic stays
import-clean and unit-testable offline.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

DATASET = "nvidia/Open-SWE-Traces"

MIN_STEPS = 1


def _extract_problem_text(trajectory: Sequence[Mapping[str, Any]]) -> Optional[str]:
    for turn in trajectory:
        if turn.get("role") == "user":
            content = str(turn.get("content") or "")
            m = re.search(r"<issue_description>\s*(.*?)\s*</issue_description>", content, re.S)
            text = m.group(1).strip() if m else content.strip()
            return text or None
    return None


def _extract_steps(trajectory: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Real per-step (reasoning, action) pairs -- the actual chain of thought + what it did about it.
    Skips assistant turns with neither (shouldn't happen in practice, but no reason to force one)."""
    steps = []
    for turn in trajectory:
        if turn.get("role") != "assistant":
            continue
        reasoning = str(turn.get("reasoning_content") or "").strip()
        calls = turn.get("tool_calls") or []
        if calls:
            for c in calls:
                fn = (c or {}).get("function") or {}
                steps.append({"reasoning": reasoning, "tool": fn.get("name"), "args": fn.get("arguments")})
        elif reasoning:
            steps.append({"reasoning": reasoning, "tool": None, "args": None})
    return steps


def _steps_to_text(steps: Sequence[Mapping[str, Any]]) -> str:
    """Render steps as numbered markers -- matches extract.py's cot-mode chunker, which splits on
    numbered/marker-separated reasoning steps, same shape as OpenThoughts' deepseek_reasoning already
    produces naturally as prose (this dataset's reasoning is per-step instead, so make the steps explicit)."""
    lines = []
    for i, s in enumerate(steps, 1):
        action = f" [action: {s['tool']}({s['args']})]" if s.get("tool") else ""
        if s.get("reasoning"):
            lines.append(f"Step {i}: {s['reasoning']}{action}")
        elif action:
            lines.append(f"Step {i}:{action}")
    return "\n\n".join(lines)


def row_to_doc(
    row: Mapping[str, Any],
    idx: int,
    *,
    resolved_only: bool = True,
    min_steps: int = MIN_STEPS,
) -> Optional[Dict[str, Any]]:
    """Map one Open-SWE-Traces row -> an extractor doc, or None if filtered.

    resolved_only=True (default): only keep trajectories the dataset itself labels resolved==1 -- same
    anti-poison principle as everything else in this codebase (gold is a REAL, independently-recorded
    outcome, never the model's own unverified attempt) -- an unresolved trajectory's reasoning may look
    plausible but led nowhere, not a safe thing to teach from as if it were a successful pattern."""
    if resolved_only and row.get("resolved") != 1:
        return None

    trajectory = row.get("trajectory") or []
    problem_text = _extract_problem_text(trajectory)
    if not problem_text:
        return None

    steps = _extract_steps(trajectory)
    if len(steps) < min_steps:
        return None

    # Tool OUTPUTS (raw environment feedback -- file views, command results) are deliberately left out of
    # `text` for now, not silently dropped: real signal is mixed in there (e.g. the actual buggy code) but
    # so is a lot of mechanical bulk (directory tree dumps routinely run into the thousands of chars).
    # reasoning_content + tool_calls alone already capture the genuine per-step CoT + real actions taken --
    # a real, honest first cut; only the COUNT is kept here as a signal of how much was left on the table.
    n_tool_outputs = sum(
        1 for t in trajectory if t.get("role") == "tool" and str(t.get("content") or "").strip()
    )

    steps_text = _steps_to_text(steps)
    text = f"{problem_text}\n\n{steps_text}" if steps_text else problem_text

    meta = row.get("metadata") or {}
    return {
        "id": f"oswe_{idx:06d}",
        "text": text,
        "domain": "code",
        "mode": "cot",
        "meta": {
            "instance_id": row.get("instance_id"),
            "repo": row.get("repo"),
            "language": row.get("language"),
            "resolved": row.get("resolved"),
            "reference_patch": meta.get("reference_patch"),
            "n_steps": len(steps),
            "n_tool_outputs": n_tool_outputs,
        },
    }


def stream_swe_traces(
    *,
    config: str = "openhands",
    split: str = "minimax_m25",
    resolved_only: bool = True,
    min_steps: int = MIN_STEPS,
    limit: int = 0,
) -> Iterator[Dict[str, Any]]:
    """Stream + filter Open-SWE-Traces into extractor docs."""
    from datasets import load_dataset  # lazy: only needed when actually fetching

    ds = load_dataset(DATASET, config, split=split, streaming=True)
    kept = 0
    for idx, row in enumerate(ds):
        doc = row_to_doc(row, idx, resolved_only=resolved_only, min_steps=min_steps)
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
    parser = argparse.ArgumentParser(description="Fetch nvidia/Open-SWE-Traces -> extractor input jsonl (targeted).")
    parser.add_argument("--out", default="data/external_kb/swe_traces.jsonl")
    parser.add_argument("--config", default="openhands", choices=["openhands", "sweagent"])
    parser.add_argument("--split", default="minimax_m25", choices=["minimax_m25", "qwen35_122b"])
    parser.add_argument("--resolved-only", action="store_true", default=True)
    parser.add_argument("--include-unresolved", dest="resolved_only", action="store_false")
    parser.add_argument("--min-steps", type=int, default=MIN_STEPS)
    parser.add_argument("--limit", type=int, default=200, help="max kept rows (0 = all)")
    args = parser.parse_args(argv)

    docs = stream_swe_traces(config=args.config, split=args.split, resolved_only=args.resolved_only,
                             min_steps=args.min_steps, limit=args.limit)
    n = write_docs(docs, args.out)
    print("Fetch nvidia/Open-SWE-Traces")
    print(f"  config: {args.config}  split: {args.split}  resolved_only: {args.resolved_only}  limit: {args.limit}")
    print(f"  wrote {n} docs -> {args.out}")
    print(f"  next: python -m v5.graph_grower.extract --docs {args.out} --out artifacts/graph_growth/swe_candidates.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
