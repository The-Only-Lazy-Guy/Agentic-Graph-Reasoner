from __future__ import annotations

"""
pred_tasks.py

PRED-v1: Build predictor training data from executor task rows.

Reads ngr_v1_{train,val}.jsonl produced by the executor task generator
(artifacts/tasks_trv_executor_20260511/) and writes supervised training pairs
to artifacts/pred_v1_20260511/.

Each output row keeps the full input (signal, spans, graph_path,
initial_memory_node_ids) and adds:

  "goal"         -- effective goal spec (session_nodes always populated;
                    covered tasks get pseudo_cover_goal applied)
  "span_oracle"  -- list of per-node span assignment labels derived from
                    lexical overlap, one entry per goal session_node
  "is_pseudo_goal" -- true when the goal was synthesized from covered_mappings
  "meta"         -- counts: num_nodes, num_edges, num_attachments, num_covered

span_oracle[i] = {
  "session_name": str,
  "spec_text": str,
  "node_type": str,
  "best_span_id": str | null,
  "best_score": float,
  "span_scores": [{"span_id": str, "score": float}, ...]
}
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

from graph_core import lexical_overlap


# ---------------------------------------------------------------------------
# Shared utilities (duplicated from traverse_threshold_draft_edit to keep
# this file self-contained and importable without the full executor deps)
# ---------------------------------------------------------------------------

def _span_overlap(text: str, spec_text: str) -> float:
    return float(lexical_overlap(str(text or ""), str(spec_text or "")))


def _pseudo_cover_goal(row: Mapping[str, Any]) -> Dict[str, Any]:
    goal = dict(row.get("goal", {}) or {})
    if goal.get("session_nodes"):
        return goal
    covs = goal.get("covered_mappings", []) or []
    goal["session_nodes"] = [
        {
            "name": f"covered_{i}",
            "span_text": str(cov.get("span_text", row.get("signal", ""))),
            "node_type": "concept",
        }
        for i, cov in enumerate(covs)
    ]
    return goal


def _goal_for_row(row: Mapping[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Returns (effective_goal, is_pseudo) where is_pseudo=True for covered tasks."""
    goal = row.get("goal", {}) or {}
    if goal.get("session_nodes"):
        return dict(goal), False
    return _pseudo_cover_goal(row), True


def _build_span_oracle(
    row: Mapping[str, Any],
    goal: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Compute per-node span assignment labels using lexical overlap."""
    spans: List[Mapping[str, Any]] = row.get("spans", []) or []
    used: Set[str] = set()
    oracle: List[Dict[str, Any]] = []

    for spec in goal.get("session_nodes", []) or []:
        session_name = str(spec.get("name", ""))
        spec_text = str(spec.get("span_text", ""))
        node_type = str(spec.get("node_type", "concept"))

        # Score every span
        scored: List[Tuple[float, int, str]] = []  # (score, len, span_id)
        for span in spans:
            sid = str(span.get("id", ""))
            text = str(span.get("text", ""))
            if not (sid and text):
                continue
            score = _span_overlap(text, spec_text)
            scored.append((score, len(text), sid))

        # All scores (descending) for training signal
        span_scores = sorted(
            [{"span_id": sid, "score": round(sc, 6)} for sc, _ln, sid in scored],
            key=lambda x: (-x["score"], x["span_id"]),
        )

        # Best span: highest overlap, shorter-text tiebreak; exclude used first.
        # If exclusivity would force a null but some previously used span still
        # has positive overlap, allow reuse instead of emitting a fake null.
        best_span_id: Optional[str] = None
        best_score = -1.0
        best_len = 10 ** 9
        for sc, ln, sid in scored:
            if sid in used:
                continue
            if sc > best_score or (sc == best_score and ln < best_len):
                best_span_id = sid
                best_score = sc
                best_len = ln

        reused = False
        if best_score <= 0.0:
            reuse_span_id: Optional[str] = None
            reuse_score = -1.0
            reuse_len = 10 ** 9
            for sc, ln, sid in scored:
                if sc > reuse_score or (sc == reuse_score and ln < reuse_len):
                    reuse_span_id = sid
                    reuse_score = sc
                    reuse_len = ln
            if reuse_span_id is not None and reuse_score > 0.0:
                best_span_id = reuse_span_id
                best_score = reuse_score
                best_len = reuse_len
                reused = True

        if best_span_id is not None and best_score > 0.0 and not reused:
            used.add(best_span_id)

        oracle.append(
            {
                "session_name": session_name,
                "spec_text": spec_text,
                "node_type": node_type,
                "best_span_id": best_span_id if best_score > 0.0 else None,
                "best_score": round(max(best_score, 0.0), 6),
                "span_scores": span_scores,
            }
        )

    return oracle


# ---------------------------------------------------------------------------
# Row processing
# ---------------------------------------------------------------------------

def process_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    goal, is_pseudo = _goal_for_row(row)
    span_oracle = _build_span_oracle(row, goal)

    out = {
        "id": row.get("id", ""),
        "task_type": row.get("task_type", ""),
        "graph_path": row.get("graph_path", ""),
        "signal": row.get("signal", ""),
        "initial_memory_node_ids": list(row.get("initial_memory_node_ids", []) or []),
        "spans": list(row.get("spans", []) or []),
        "goal": goal,
        "span_oracle": span_oracle,
        "is_pseudo_goal": is_pseudo,
        "meta": {
            "num_nodes": len(goal.get("session_nodes", []) or []),
            "num_edges": len(goal.get("session_edges", []) or []),
            "num_attachments": len(goal.get("memory_attachments", []) or []),
            "num_covered": len(goal.get("covered_mappings", []) or []),
        },
    }
    return out


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def compute_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    task_counts: Dict[str, int] = {}
    total_nodes = 0
    total_edges = 0
    total_attachments = 0
    total_covered = 0
    nodes_with_span = 0
    total_nodes_for_span = 0
    score_buckets = {"0.0": 0, "0.01-0.3": 0, "0.3-0.6": 0, "0.6-0.8": 0, "0.8+": 0}

    for row in rows:
        tt = row.get("task_type", "unknown")
        task_counts[tt] = task_counts.get(tt, 0) + 1
        meta = row.get("meta", {})
        total_nodes += meta.get("num_nodes", 0)
        total_edges += meta.get("num_edges", 0)
        total_attachments += meta.get("num_attachments", 0)
        total_covered += meta.get("num_covered", 0)
        for entry in row.get("span_oracle", []):
            total_nodes_for_span += 1
            sc = entry.get("best_score", 0.0)
            if sc > 0.0:
                nodes_with_span += 1
            if sc == 0.0:
                score_buckets["0.0"] += 1
            elif sc < 0.3:
                score_buckets["0.01-0.3"] += 1
            elif sc < 0.6:
                score_buckets["0.3-0.6"] += 1
            elif sc < 0.8:
                score_buckets["0.6-0.8"] += 1
            else:
                score_buckets["0.8+"] += 1

    span_coverage = nodes_with_span / max(total_nodes_for_span, 1)
    return {
        "total": len(rows),
        "task_counts": task_counts,
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "total_attachments": total_attachments,
        "total_covered": total_covered,
        "span_coverage": round(span_coverage, 4),
        "score_distribution": score_buckets,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="PRED-v1: build predictor training data")
    parser.add_argument("--input-dir", default="artifacts/tasks_trv_executor_20260511",
                        help="Directory with ngr_v1_{train,val}.jsonl")
    parser.add_argument("--output-dir", default="artifacts/pred_v1_20260511",
                        help="Output directory for pred_{train,val}.jsonl")
    parser.add_argument("--splits", nargs="+", default=["train", "val"],
                        help="Which splits to process")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        in_path = input_dir / f"ngr_v1_{split}.jsonl"
        if not in_path.exists():
            print(f"[skip] {in_path} not found")
            continue

        rows: List[Dict[str, Any]] = []
        with in_path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))

        out_rows = [process_row(r) for r in rows]

        out_path = output_dir / f"pred_{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for row in out_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        stats = compute_stats(out_rows)
        stats_path = output_dir / f"pred_{split}_stats.json"
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        print(f"[{split}] {stats['total']} rows -> {out_path}")
        print(f"        task_counts: {stats['task_counts']}")
        print(f"        span_coverage: {stats['span_coverage']:.4f} "
              f"({stats['total_nodes']} nodes, {stats['total_edges']} edges, "
              f"{stats['total_attachments']} attachments, {stats['total_covered']} covered)")
        print(f"        score dist: {stats['score_distribution']}")

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "splits": args.splits,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nManifest written to {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
