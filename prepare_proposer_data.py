from __future__ import annotations

"""
prepare_proposer_data.py

Build PRED-v3 proposer supervision from existing pred_v1 rows.

Input rows are goal-conditioned aligner rows that already contain:
  - spans
  - goal.session_nodes
  - span_oracle

This script converts them into fixed-slot proposer targets with canonical slot
ordering derived from oracle span anchors.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _slot_order_key(
    goal_index: int,
    oracle_entry: Mapping[str, Any],
    span_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[int, int, int]:
    span_id = oracle_entry.get("best_span_id")
    span = span_by_id.get(str(span_id), {})
    start = int(span.get("start", 10**9))
    end = int(span.get("end", 10**9))
    return (start, goal_index, end)


def canonicalize_target_slots(
    row: Mapping[str, Any],
    *,
    k_max: int,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    goal = row.get("goal", {}) or {}
    session_nodes = list(goal.get("session_nodes", []) or [])
    oracle_entries = list(row.get("span_oracle", []) or [])
    span_by_id = {str(span.get("id", "")): span for span in row.get("spans", []) or []}
    oracle_by_name = {str(entry.get("session_name", "")): entry for entry in oracle_entries}

    ordered_items: List[tuple[tuple[int, int, int], Dict[str, Any]]] = []
    for goal_index, spec in enumerate(session_nodes):
        name = str(spec.get("name", f"s{goal_index}"))
        node_type = str(spec.get("node_type", "unknown"))
        spec_text = str(spec.get("span_text", ""))
        oracle_entry = oracle_by_name.get(name, {})
        span_id = oracle_entry.get("best_span_id")
        span = span_by_id.get(str(span_id), {})
        order_key = _slot_order_key(goal_index, oracle_entry, span_by_id)
        ordered_items.append(
            (
                order_key,
                {
                    "use": True,
                    "session_name": name,
                    "node_type": node_type,
                    "span_id": span_id,
                    "span_text": spec_text,
                    "source_goal_index": goal_index,
                    "anchor_start": None if not span else int(span.get("start", -1)),
                    "anchor_end": None if not span else int(span.get("end", -1)),
                    "oracle_best_score": float(oracle_entry.get("best_score", 0.0)),
                },
            )
        )

    ordered_items.sort(key=lambda item: item[0])
    kept = [slot for _key, slot in ordered_items[:k_max]]
    dropped = [slot for _key, slot in ordered_items[k_max:]]

    padded = kept + [
        {
            "use": False,
            "session_name": None,
            "node_type": None,
            "span_id": None,
            "span_text": None,
            "source_goal_index": None,
            "anchor_start": None,
            "anchor_end": None,
            "oracle_best_score": 0.0,
        }
        for _ in range(max(k_max - len(kept), 0))
    ]

    shared_anchor = False
    anchors = [slot["anchor_start"] for slot in kept if slot["anchor_start"] is not None]
    if len(anchors) != len(set(anchors)):
        shared_anchor = True

    meta = {
        "num_gold_session_nodes": len(session_nodes),
        "num_kept_slots": len(kept),
        "num_dropped_slots": len(dropped),
        "dropped_session_names": [slot["session_name"] for slot in dropped],
        "shared_anchor_start": shared_anchor,
    }
    return padded, meta


def process_row(row: Mapping[str, Any], *, k_max: int) -> Dict[str, Any]:
    target_slots, slot_meta = canonicalize_target_slots(row, k_max=k_max)
    return {
        "id": row.get("id", ""),
        "task_type": row.get("task_type", ""),
        "signal": row.get("signal", ""),
        "graph_path": row.get("graph_path", ""),
        "initial_memory_node_ids": list(row.get("initial_memory_node_ids", []) or []),
        "spans": list(row.get("spans", []) or []),
        "K_max": k_max,
        "target_slots": target_slots,
        "_oracle_goal": row.get("goal", {}) or {},
        "_slot_meta": slot_meta,
    }


def infer_k_max(rows: Sequence[Mapping[str, Any]]) -> int:
    return max(len((row.get("goal", {}) or {}).get("session_nodes", []) or []) for row in rows)


def compute_stats(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    task_counts: Dict[str, int] = Counter()
    slot_use_hist: Dict[int, int] = Counter()
    shared_anchor_rows = 0
    dropped_rows = 0
    dropped_slots = 0
    node_type_counts: Dict[str, int] = Counter()

    for row in rows:
        task_counts[str(row.get("task_type", "unknown"))] += 1
        used = sum(1 for slot in row.get("target_slots", []) if slot.get("use"))
        slot_use_hist[used] += 1
        meta = row.get("_slot_meta", {}) or {}
        if bool(meta.get("shared_anchor_start")):
            shared_anchor_rows += 1
        dropped = int(meta.get("num_dropped_slots", 0))
        if dropped > 0:
            dropped_rows += 1
            dropped_slots += dropped
        for slot in row.get("target_slots", []):
            if slot.get("use") and slot.get("node_type"):
                node_type_counts[str(slot["node_type"])] += 1

    return {
        "total_rows": len(rows),
        "task_counts": dict(sorted(task_counts.items())),
        "slot_use_histogram": dict(sorted(slot_use_hist.items())),
        "shared_anchor_rows": shared_anchor_rows,
        "rows_with_dropped_slots": dropped_rows,
        "total_dropped_slots": dropped_slots,
        "node_type_counts": dict(sorted(node_type_counts.items())),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build fixed-slot proposer targets from pred_v1 rows")
    ap.add_argument("--input-dir", default="artifacts/pred_v1_20260511_fix8", help="Directory with pred_{train,val}.jsonl")
    ap.add_argument("--output-dir", default="artifacts/proposer_v1_20260512", help="Directory for proposer_{train,val}.jsonl")
    ap.add_argument("--splits", nargs="+", default=["train", "val"], help="Splits to convert")
    ap.add_argument("--k-max", type=int, default=0, help="Fixed slot count; 0 means infer from input data")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "splits": list(args.splits),
        "k_max": int(args.k_max),
        "split_stats": {},
    }

    for split in args.splits:
        in_path = input_dir / f"pred_{split}.jsonl"
        if not in_path.exists():
            print(f"[skip] {in_path} not found")
            continue
        rows = read_jsonl(in_path)
        split_k = int(args.k_max) if int(args.k_max) > 0 else infer_k_max(rows)
        out_rows = [process_row(row, k_max=split_k) for row in rows]
        out_path = output_dir / f"proposer_{split}.jsonl"
        write_jsonl(out_path, out_rows)
        stats = compute_stats(out_rows)
        stats["k_max"] = split_k
        manifest["split_stats"][split] = stats
        print(json.dumps({"split": split, **stats}, ensure_ascii=False))

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
