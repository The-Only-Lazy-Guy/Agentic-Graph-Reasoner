from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from train_pred_v1 import read_jsonl


SYNTHESIS_TASKS = {"mixed_add_link", "multi_region_attach"}
COPYISH_CATEGORIES = {"exact_span", "substring_of_span", "contains_span", "concat_two_spans"}
MAX_EXAMPLES_PER_CATEGORY = 5


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def ordered_unique_spans(spans: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for span in spans:
        text = str(span.get("text", ""))
        norm = normalize_text(text)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(
            {
                "id": str(span.get("id", "")),
                "text": text,
                "norm": norm,
            }
        )
    return out


def find_concat_of_two_spans(gold_norm: str, spans: Sequence[Mapping[str, Any]]) -> list[str] | None:
    unique_spans = ordered_unique_spans(spans)
    for i, span_a in enumerate(unique_spans):
        for j, span_b in enumerate(unique_spans):
            joined = normalize_text(f"{span_a['text']} {span_b['text']}")
            if joined == gold_norm:
                return [span_a["id"], span_b["id"]]
            if i != j:
                joined_nosep = normalize_text(f"{span_a['text']}{span_b['text']}")
                if joined_nosep == gold_norm:
                    return [span_a["id"], span_b["id"]]
    return None


def classify_gold_text(
    gold_text: str,
    spans: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    gold_norm = normalize_text(gold_text)
    if not gold_norm:
        return {
            "category": "empty",
            "exact_span_ids": [],
            "substring_span_ids": [],
            "contains_span_ids": [],
            "concat_span_ids": None,
        }

    exact_span_ids: List[str] = []
    substring_span_ids: List[str] = []
    contains_span_ids: List[str] = []
    unique_spans = ordered_unique_spans(spans)
    for span in unique_spans:
        span_norm = span["norm"]
        if span_norm == gold_norm:
            exact_span_ids.append(span["id"])
        elif gold_norm in span_norm:
            substring_span_ids.append(span["id"])
        elif span_norm and span_norm in gold_norm:
            contains_span_ids.append(span["id"])

    concat_span_ids = find_concat_of_two_spans(gold_norm, spans)

    if exact_span_ids:
        category = "exact_span"
    elif substring_span_ids:
        category = "substring_of_span"
    elif contains_span_ids:
        category = "contains_span"
    elif concat_span_ids is not None:
        category = "concat_two_spans"
    else:
        category = "none_above"

    return {
        "category": category,
        "exact_span_ids": exact_span_ids,
        "substring_span_ids": substring_span_ids,
        "contains_span_ids": contains_span_ids,
        "concat_span_ids": concat_span_ids,
    }


def node_needs_synthesis(row: Mapping[str, Any], node: Mapping[str, Any]) -> bool:
    task_type = str(row.get("task_type", ""))
    name = str(node.get("name", ""))
    if task_type == "mixed_add_link":
        return name == "new_note"
    if task_type == "multi_region_attach":
        return name == "bridge"
    return False


def summarize_counter(counter: Counter[str]) -> Dict[str, Dict[str, float]]:
    total = sum(counter.values())
    summary: Dict[str, Dict[str, float]] = {}
    for key in sorted(counter):
        count = counter[key]
        summary[key] = {
            "count": count,
            "rate": count / max(total, 1),
        }
    summary["_total"] = {"count": total, "rate": 1.0 if total else 0.0}
    return summary


def append_example(
    bucket: Dict[str, List[Dict[str, Any]]],
    category: str,
    example: Dict[str, Any],
) -> None:
    examples = bucket.setdefault(category, [])
    if len(examples) < MAX_EXAMPLES_PER_CATEGORY:
        examples.append(example)


def analyze_rows(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    by_task: Dict[str, Counter[str]] = defaultdict(Counter)
    by_task_synth_only: Dict[str, Counter[str]] = defaultdict(Counter)
    by_name: Dict[str, Counter[str]] = defaultdict(Counter)
    examples_by_task: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(dict)
    examples_synth_only_by_task: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(dict)

    synth_total = 0
    synth_copyish = 0

    for row in rows:
        task_type = str(row.get("task_type", ""))
        if task_type not in SYNTHESIS_TASKS:
            continue
        spans = row.get("spans", []) or []
        for node in (row.get("goal", {}) or {}).get("session_nodes", []) or []:
            name = str(node.get("name", ""))
            gold_text = str(node.get("span_text", ""))
            result = classify_gold_text(gold_text, spans)
            category = str(result["category"])
            is_synth = node_needs_synthesis(row, node)

            by_task[task_type][category] += 1
            by_name[f"{task_type}:{name}"][category] += 1
            append_example(
                examples_by_task[task_type],
                category,
                {
                    "row_id": str(row.get("id", "")),
                    "session_name": name,
                    "node_type": str(node.get("node_type", "")),
                    "gold_text": gold_text,
                    "exact_span_ids": result["exact_span_ids"],
                    "substring_span_ids": result["substring_span_ids"],
                    "contains_span_ids": result["contains_span_ids"],
                    "concat_span_ids": result["concat_span_ids"],
                    "span_texts_preview": [
                        {
                            "id": str(span.get("id", "")),
                            "text": str(span.get("text", "")),
                        }
                        for span in (spans[:5] if len(spans) > 5 else spans)
                    ],
                },
            )

            if is_synth:
                synth_total += 1
                if category in COPYISH_CATEGORIES:
                    synth_copyish += 1
                by_task_synth_only[task_type][category] += 1
                append_example(
                    examples_synth_only_by_task[task_type],
                    category,
                    {
                        "row_id": str(row.get("id", "")),
                        "session_name": name,
                        "gold_text": gold_text,
                        "exact_span_ids": result["exact_span_ids"],
                        "substring_span_ids": result["substring_span_ids"],
                        "contains_span_ids": result["contains_span_ids"],
                        "concat_span_ids": result["concat_span_ids"],
                        "all_spans": [
                            {
                                "id": str(span.get("id", "")),
                                "text": str(span.get("text", "")),
                                "span_kind": str(span.get("span_kind", "")),
                            }
                            for span in spans
                        ],
                    },
                )

    return {
        "by_task_all_nodes": {task: summarize_counter(counter) for task, counter in sorted(by_task.items())},
        "by_task_synthesis_nodes_only": {
            task: summarize_counter(counter) for task, counter in sorted(by_task_synth_only.items())
        },
        "by_task_and_name": {name: summarize_counter(counter) for name, counter in sorted(by_name.items())},
        "synthesis_copyish_rate": synth_copyish / max(synth_total, 1),
        "synthesis_copyish_counts": {
            "copyish": synth_copyish,
            "total_synthesis_nodes": synth_total,
        },
        "examples_by_task_all_nodes": examples_by_task,
        "examples_by_task_synthesis_nodes_only": examples_synth_only_by_task,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze synthesis text patterns in pred_v1 rows")
    ap.add_argument(
        "--jsonl",
        nargs="+",
        required=True,
        help="One or more pred_v1 jsonl files",
    )
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    rows: List[Mapping[str, Any]] = []
    for path in args.jsonl:
        rows.extend(read_jsonl(path))

    report = analyze_rows(rows)
    print(json.dumps(report["by_task_synthesis_nodes_only"], indent=2, ensure_ascii=False))
    print(json.dumps(report["synthesis_copyish_counts"], indent=2, ensure_ascii=False))
    print(json.dumps({"synthesis_copyish_rate": report["synthesis_copyish_rate"]}, indent=2, ensure_ascii=False))

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"report written to {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
