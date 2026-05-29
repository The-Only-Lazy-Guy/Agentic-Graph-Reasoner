"""Format the distillation corpus into SFT training data.

Converts sessions.jsonl into conversation-format JSONL suitable for
fine-tuning with torchtune / HuggingFace TRL / axolotl.

Each session becomes one training example:
  system: graph-only reasoner prompt
  user:   question + seed nodes
  assistant: plan + tool calls + reasoning + answer (multi-turn)

Usage:
    python scripts/format_sft_data.py
    python scripts/format_sft_data.py --min-grounding 0.3  # filter poorly grounded
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CORPUS_PATH = Path("data/distillation_corpus/sessions.jsonl")
OUTPUT_PATH = Path("data/sft_dataset/train.jsonl")

# Minimal system prompt for the distilled model (no tool documentation —
# the model learns the tools from the traces themselves).
SYSTEM_PROMPT = """\
You are a graph-reasoning agent. Navigate the knowledge graph using tools, \
cite every claim to a node you read, and synthesize your answer only from \
graph content. Your general knowledge helps you search efficiently, but \
the answer must come from the graph."""


def session_to_sft_messages(row: Dict[str, Any]) -> List[Dict[str, str]]:
    """Convert one corpus row into a multi-turn conversation."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # User turn: question + seed anchors
    question = row.get("input", {}).get("question", "")
    anchors = row.get("input", {}).get("anchors", [])
    user_parts = [f"Question: {question}"]
    if anchors:
        user_parts.append("\nSeed nodes:")
        for a in anchors[:5]:
            aid = a.get("id", "?")
            atype = a.get("node_type", "?")
            atext = a.get("text", "")[:120]
            user_parts.append(f"  `{aid}` [{atype}] {atext}")
    messages.append({"role": "user", "content": "\n".join(user_parts)})

    # Assistant turns: full CoT trace + raw answer (with citations preserved).
    # IMPORTANT: use answer_raw, NOT answer_polished. Polish strips node
    # citations which are the critical training signal for graph grounding.
    cot_log = row.get("trace", {}).get("cot_log", [])
    if cot_log:
        combined_cot = "\n\n".join(
            f"[Step {i}]\n{step}" for i, step in enumerate(cot_log)
        )
        messages.append({"role": "assistant", "content": combined_cot})
    else:
        # Fallback: use raw answer (with citations)
        answer = row.get("outputs", {}).get("answer_raw", "")
        if not answer:
            answer = row.get("outputs", {}).get("answer_polished", "")
        if answer:
            messages.append({"role": "assistant", "content": answer})

    return messages


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default=str(CORPUS_PATH))
    ap.add_argument("--output", default=str(OUTPUT_PATH))
    ap.add_argument("--min-grounding", type=float, default=0.0,
                    help="minimum grounding_ratio to include (0=all)")
    ap.add_argument("--finalized-only", action="store_true", default=True,
                    help="only include finalized sessions")
    args = ap.parse_args()

    corpus = Path(args.corpus)
    if not corpus.exists():
        print(f"Corpus not found: {corpus}")
        print("Run some v4 sessions first (scripts/run_batch.py)")
        sys.exit(1)

    rows = []
    with corpus.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    print(f"Corpus: {len(rows)} sessions")

    # Filter
    filtered = []
    for row in rows:
        quality = row.get("quality", {})
        if args.finalized_only and not quality.get("finalized"):
            continue
        # Grounding filter (if grounding metrics available)
        grounding = row.get("grounding", {})
        if args.min_grounding > 0 and grounding:
            ratio = grounding.get("grounding_ratio", 0)
            if ratio < args.min_grounding:
                continue
        filtered.append(row)

    print(f"After filtering: {len(filtered)} sessions")

    # Convert to SFT format
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    total_tokens_est = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in filtered:
            messages = session_to_sft_messages(row)
            if len(messages) < 2:
                continue
            sft_row = {"messages": messages}
            # Add metadata for the trainer to use as weights
            sft_row["metadata"] = {
                "session_id": row.get("session_id", ""),
                "domain": row.get("input", {}).get("question", "")[:20],
                "steps": row.get("metrics", {}).get("steps", 0),
                "coverage_pct": row.get("metrics", {}).get("coverage_addressed_pct", 0),
                "complexity": row.get("quality", {}).get("complexity_proxy_score", 0),
            }
            f.write(json.dumps(sft_row, ensure_ascii=False) + "\n")
            count += 1
            # Rough token estimate
            total_chars = sum(len(m["content"]) for m in messages)
            total_tokens_est += total_chars // 4

    print(f"\nSFT dataset: {count} examples")
    print(f"Estimated tokens: {total_tokens_est:,}")
    print(f"Output: {out_path}")

    # Stats
    if filtered:
        steps = [r.get("metrics", {}).get("steps", 0) for r in filtered]
        print(f"\nStep distribution: min={min(steps)} mean={sum(steps)/len(steps):.1f} max={max(steps)}")


if __name__ == "__main__":
    main()
