from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from graph_core import MemoryGraph
from ngr_v1_tasks import clean
from train_pred_v1 import read_jsonl


SYNTHESIS_TASKS = {"mixed_add_link", "multi_region_attach"}
MAX_EXAMPLES = 5


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").split())


def exact_match(a: str, b: str) -> bool:
    return normalize_text(a) == normalize_text(b)


def graph_for_path(cache: Dict[str, MemoryGraph], path: str) -> MemoryGraph:
    if path not in cache:
        cache[path] = MemoryGraph.load_json(path)
    return cache[path]


def text_by_id(graph: MemoryGraph, node_id: str) -> str:
    node = graph.nodes.get(str(node_id))
    return str(node.text) if node is not None else ""


def node_by_name(goal: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    for node in (goal.get("session_nodes", []) or []):
        if str(node.get("name", "")) == name:
            return node
    return None


def bridge_attachment_ids(goal: Mapping[str, Any]) -> List[str]:
    return [
        str(att.get("memory_id", ""))
        for att in (goal.get("memory_attachments", []) or [])
        if str(att.get("session", "")) == "bridge" and str(att.get("memory_id", ""))
    ]


def new_note_attachment_id(goal: Mapping[str, Any]) -> str | None:
    for att in (goal.get("memory_attachments", []) or []):
        if str(att.get("session", "")) == "new_note":
            mem = str(att.get("memory_id", ""))
            return mem or None
    return None


def best_matching_memory_id(memory_ids: Sequence[str], support_text: str, graph: MemoryGraph) -> str | None:
    support_clean = clean(support_text, 120)
    for mem_id in memory_ids:
        if clean(text_by_id(graph, mem_id), 120) == support_clean:
            return mem_id

    best_id = None
    best_score = -1
    support_tokens = set(normalize_text(support_text).lower().split())
    for mem_id in memory_ids:
        mem_tokens = set(normalize_text(text_by_id(graph, mem_id)).lower().split())
        score = len(support_tokens & mem_tokens)
        if score > best_score:
            best_score = score
            best_id = mem_id
    return best_id


def build_mixed_oracle(row: Mapping[str, Any], graph: MemoryGraph) -> str | None:
    goal = row.get("goal", {}) or {}
    source = node_by_name(goal, "source_note")
    mem_id = new_note_attachment_id(goal)
    if source is None or mem_id is None:
        return None
    source_text = str(source.get("span_text", ""))
    dst_text = clean(text_by_id(graph, mem_id), 110)
    return clean(f"{source_text} This supports a new note related to {dst_text}.", 220)


def build_mixed_heuristic(row: Mapping[str, Any], graph: MemoryGraph) -> str | None:
    goal = row.get("goal", {}) or {}
    source = node_by_name(goal, "source_note")
    if source is None:
        return None
    memory_ids = [str(x) for x in (row.get("initial_memory_node_ids", []) or []) if str(x)]
    if not memory_ids:
        return None
    source_text = str(source.get("span_text", ""))
    dst_text = clean(text_by_id(graph, memory_ids[0]), 110)
    return clean(f"{source_text} This supports a new note related to {dst_text}.", 220)


def build_bridge_oracle(row: Mapping[str, Any], graph: MemoryGraph) -> str | None:
    goal = row.get("goal", {}) or {}
    support = node_by_name(goal, "support_note")
    mem_ids = bridge_attachment_ids(goal)
    if support is None or len(mem_ids) < 2:
        return None
    support_text = str(support.get("span_text", ""))
    support_mem = best_matching_memory_id(mem_ids, support_text, graph)
    if support_mem is None:
        return None
    other = next((m for m in mem_ids if m != support_mem), None)
    if other is None:
        return None
    text_a = clean(text_by_id(graph, support_mem), 90)
    text_b = clean(text_by_id(graph, other), 90)
    return clean(f"{text_a} and {text_b} are connected by a shared bridge concept.", 180)


def build_bridge_heuristic(row: Mapping[str, Any], graph: MemoryGraph) -> str | None:
    goal = row.get("goal", {}) or {}
    support = node_by_name(goal, "support_note")
    memory_ids = [str(x) for x in (row.get("initial_memory_node_ids", []) or []) if str(x)]
    if support is None or len(memory_ids) < 2:
        return None
    support_text = str(support.get("span_text", ""))
    support_mem = best_matching_memory_id(memory_ids, support_text, graph)
    if support_mem is None:
        return None
    other = next((m for m in memory_ids if m != support_mem), None)
    if other is None:
        return None
    text_a = clean(text_by_id(graph, support_mem), 90)
    text_b = clean(text_by_id(graph, other), 90)
    return clean(f"{text_a} and {text_b} are connected by a shared bridge concept.", 180)


def verify_rows(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    graph_cache: Dict[str, MemoryGraph] = {}
    by_task = defaultdict(lambda: {"oracle": Counter(), "heuristic": Counter(), "oracle_examples": [], "heuristic_examples": []})

    for row in rows:
        task_type = str(row.get("task_type", ""))
        if task_type not in SYNTHESIS_TASKS:
            continue
        graph = graph_for_path(graph_cache, str(row.get("graph_path", "")))
        goal = row.get("goal", {}) or {}

        if task_type == "mixed_add_link":
            gold_node = node_by_name(goal, "new_note")
            gold_text = str((gold_node or {}).get("span_text", ""))
            oracle_text = build_mixed_oracle(row, graph)
            heuristic_text = build_mixed_heuristic(row, graph)
        else:
            gold_node = node_by_name(goal, "bridge")
            gold_text = str((gold_node or {}).get("span_text", ""))
            oracle_text = build_bridge_oracle(row, graph)
            heuristic_text = build_bridge_heuristic(row, graph)

        oracle_ok = oracle_text is not None and exact_match(oracle_text, gold_text)
        heuristic_ok = heuristic_text is not None and exact_match(heuristic_text, gold_text)
        by_task[task_type]["oracle"]["match" if oracle_ok else "mismatch"] += 1
        by_task[task_type]["heuristic"]["match" if heuristic_ok else "mismatch"] += 1

        if not oracle_ok and len(by_task[task_type]["oracle_examples"]) < MAX_EXAMPLES:
            by_task[task_type]["oracle_examples"].append(
                {
                    "row_id": str(row.get("id", "")),
                    "gold_text": gold_text,
                    "reconstructed_text": oracle_text,
                    "initial_memory_node_ids": row.get("initial_memory_node_ids", []),
                    "memory_attachments": goal.get("memory_attachments", []),
                    "session_nodes": goal.get("session_nodes", []),
                }
            )
        if not heuristic_ok and len(by_task[task_type]["heuristic_examples"]) < MAX_EXAMPLES:
            by_task[task_type]["heuristic_examples"].append(
                {
                    "row_id": str(row.get("id", "")),
                    "gold_text": gold_text,
                    "reconstructed_text": heuristic_text,
                    "initial_memory_node_ids": row.get("initial_memory_node_ids", []),
                    "memory_attachments": goal.get("memory_attachments", []),
                    "session_nodes": goal.get("session_nodes", []),
                }
            )

    summary: Dict[str, Any] = {}
    for task_type, stats in sorted(by_task.items()):
        oracle_total = sum(stats["oracle"].values())
        heuristic_total = sum(stats["heuristic"].values())
        summary[task_type] = {
            "oracle_exact_match": {
                "count": stats["oracle"]["match"],
                "total": oracle_total,
                "rate": stats["oracle"]["match"] / max(oracle_total, 1),
            },
            "heuristic_exact_match": {
                "count": stats["heuristic"]["match"],
                "total": heuristic_total,
                "rate": stats["heuristic"]["match"] / max(heuristic_total, 1),
            },
            "oracle_mismatch_examples": stats["oracle_examples"],
            "heuristic_mismatch_examples": stats["heuristic_examples"],
        }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify deterministic synthesis templates against pred_v1 rows")
    ap.add_argument("--jsonl", nargs="+", required=True)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    rows: List[Mapping[str, Any]] = []
    for path in args.jsonl:
        rows.extend(read_jsonl(path))

    report = verify_rows(rows)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"report written to {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
