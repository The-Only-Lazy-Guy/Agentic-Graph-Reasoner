from __future__ import annotations

"""
traverse_threshold_eval.py

Traversal-only ablation for graph navigation.

This path does not build session nodes or emit edit actions. It:
1. scores anchor nodes from the signal
2. expands graph neighbors step by step
3. traverses only adjacency candidates above a score threshold
4. records a full trace for inspection

Primary use:
- debug whether simple thresholded traversal can recover the relevant graph
  region without the current edit-program controller
"""

import argparse
import heapq
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from graph_core import MemoryGraph, canonical_relation, lexical_overlap


@dataclass
class TraversalConfig:
    anchor_threshold: float = 0.10
    traverse_threshold: float = 0.18
    top_k_anchors: int = 8
    max_steps: int = 24
    max_visited: int = 64
    node_overlap_weight: float = 0.65
    importance_weight: float = 0.20
    confidence_weight: float = 0.10
    edge_strength_weight: float = 0.20
    relation_bonus_weight: float = 0.05


RELATION_BONUS = {
    "support": 1.0,
    "refine": 0.8,
    "related": 0.6,
    "part_of": 0.6,
    "example_of": 0.5,
    "depend": 0.4,
    "cause": 0.4,
    "contradict": 0.2,
    "conflict": 0.2,
    "refute": 0.1,
}


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / max(len(xs), 1)


def target_memory_ids_from_row(row: Mapping[str, Any]) -> List[str]:
    goal = row.get("goal", {}) or {}
    out: List[str] = []
    for cov in goal.get("covered_mappings", []) or []:
        mem = str(cov.get("memory_id", ""))
        if mem:
            out.append(mem)
    for att in goal.get("memory_attachments", []) or []:
        mem = str(att.get("memory_id", ""))
        if mem:
            out.append(mem)
    seen = set()
    deduped = []
    for mem in out:
        if mem not in seen:
            seen.add(mem)
            deduped.append(mem)
    return deduped


def node_score(signal: str, nid: str, graph: MemoryGraph, cfg: TraversalConfig) -> float:
    node = graph.nodes[nid]
    overlap = float(lexical_overlap(signal, f"{nid.replace('_', ' ')} {node.text}"))
    return (
        cfg.node_overlap_weight * overlap
        + cfg.importance_weight * float(getattr(node, "importance", 0.5))
        + cfg.confidence_weight * float(getattr(node, "confidence", 0.5))
    )


def transition_score(
    signal: str,
    src_id: str,
    dst_id: str,
    graph: MemoryGraph,
    cfg: TraversalConfig,
) -> Tuple[float, Dict[str, float]]:
    edge = graph.edge_between(src_id, dst_id)
    rel = canonical_relation(edge.relation if edge else "related")
    edge_strength = float(edge.strength if edge else 0.5)
    dst_score = node_score(signal, dst_id, graph, cfg)
    src_score = node_score(signal, src_id, graph, cfg)
    rel_bonus = RELATION_BONUS.get(rel, 0.3)
    score = dst_score + cfg.edge_strength_weight * edge_strength + cfg.relation_bonus_weight * rel_bonus + 0.05 * src_score
    return score, {
        "dst_score": dst_score,
        "src_score": src_score,
        "edge_strength": edge_strength,
        "relation_bonus": rel_bonus,
    }


def select_anchors(signal: str, graph: MemoryGraph, cfg: TraversalConfig) -> List[Tuple[float, str]]:
    scored: List[Tuple[float, str]] = []
    for nid in graph.nodes:
        sc = node_score(signal, nid, graph, cfg)
        if sc >= cfg.anchor_threshold:
            scored.append((sc, nid))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored[: cfg.top_k_anchors]


def traverse_graph(signal: str, graph: MemoryGraph, cfg: TraversalConfig) -> Dict[str, Any]:
    anchors = select_anchors(signal, graph, cfg)
    visited: Dict[str, float] = {}
    frontier: List[Tuple[float, str, Optional[str]]] = []
    expanded: set[str] = set()
    trace: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    for score, nid in anchors:
        if nid not in visited:
            visited[nid] = score
            heapq.heappush(frontier, (-score, nid, None))

    steps = 0
    while frontier and steps < cfg.max_steps and len(visited) < cfg.max_visited:
        neg_score, current, parent = heapq.heappop(frontier)
        current_score = -neg_score
        if current in expanded:
            continue
        if current_score + 1e-9 < visited.get(current, current_score):
            continue
        expanded.add(current)
        trace.append({
            "step": steps,
            "node_id": current,
            "from_node_id": parent,
            "node_score": current_score,
            "node_text": graph.nodes[current].text,
        })
        steps += 1

        for neighbor in graph._adj.get(current, []):
            score, parts = transition_score(signal, current, neighbor, graph, cfg)
            edge = graph.edge_between(current, neighbor)
            rel = canonical_relation(edge.relation if edge else "related")
            rec = {
                "step": steps - 1,
                "from_node_id": current,
                "to_node_id": neighbor,
                "relation": rel,
                "score": score,
                "threshold": cfg.traverse_threshold,
                **parts,
            }
            if score < cfg.traverse_threshold:
                rejected.append(rec)
                continue
            if neighbor not in visited or score > visited[neighbor]:
                visited[neighbor] = score
                heapq.heappush(frontier, (-score, neighbor, current))

    visited_nodes = [
        {
            "id": nid,
            "score": score,
            "text": graph.nodes[nid].text,
            "node_type": graph.nodes[nid].node_type,
        }
        for nid, score in sorted(visited.items(), key=lambda x: (-x[1], x[0]))
    ]
    return {
        "anchors": [{"id": nid, "score": score, "text": graph.nodes[nid].text} for score, nid in anchors],
        "visited_nodes": visited_nodes,
        "trace": trace,
        "rejected_edges": rejected[:256],
        "steps": steps,
    }


def evaluate_row(row: Mapping[str, Any], cfg: TraversalConfig) -> Dict[str, Any]:
    graph = MemoryGraph.load_json(str(row["graph_path"]))
    signal = str(row.get("signal", ""))
    traversal = traverse_graph(signal, graph, cfg)
    visited_ids = {str(x["id"]) for x in traversal["visited_nodes"]}
    target_memory_ids = target_memory_ids_from_row(row)
    has_target_memory = bool(target_memory_ids)
    target_hit = sum(1 for nid in target_memory_ids if nid in visited_ids)
    recall = (target_hit / len(target_memory_ids)) if target_memory_ids else None
    all_hit = bool(target_memory_ids) and target_hit == len(target_memory_ids)
    return {
        "id": row.get("id"),
        "task_type": row.get("task_type", "unknown"),
        "has_target_memory": has_target_memory,
        "target_memory_ids": target_memory_ids,
        "target_memory_hit_count": target_hit,
        "target_memory_recall": recall,
        "all_target_memory_hit": all_hit,
        **traversal,
    }


def aggregate(results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_task: Dict[str, List[Mapping[str, Any]]] = {}
    for r in results:
        by_task.setdefault(str(r.get("task_type", "unknown")), []).append(r)

    def agg(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        target_rows = [r for r in rows if bool(r.get("has_target_memory", False))]
        return {
            "n": len(rows),
            "n_target_memory_rows": len(target_rows),
            "avg_steps": mean([float(r.get("steps", 0)) for r in rows]),
            "target_memory_recall": mean([float(r["target_memory_recall"]) for r in target_rows]) if target_rows else None,
            "all_target_memory_hit_rate": mean([1.0 if bool(r.get("all_target_memory_hit", False)) else 0.0 for r in target_rows]) if target_rows else None,
            "avg_visited_nodes": mean([float(len(r.get("visited_nodes", []) or [])) for r in rows]),
        }

    return {
        "overall": agg(results),
        "by_task": {k: agg(v) for k, v in sorted(by_task.items())},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-jsonl", default="")
    ap.add_argument("--graph", default="")
    ap.add_argument("--signal", default="")
    ap.add_argument("--max-rows", type=int, default=20)
    ap.add_argument("--anchor-threshold", type=float, default=0.10)
    ap.add_argument("--traverse-threshold", type=float, default=0.18)
    ap.add_argument("--top-k-anchors", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=24)
    ap.add_argument("--max-visited", type=int, default=64)
    ap.add_argument("--save-json", default="")
    args = ap.parse_args()

    cfg = TraversalConfig(
        anchor_threshold=args.anchor_threshold,
        traverse_threshold=args.traverse_threshold,
        top_k_anchors=args.top_k_anchors,
        max_steps=args.max_steps,
        max_visited=args.max_visited,
    )

    if args.task_jsonl:
        rows = read_jsonl(args.task_jsonl)[: args.max_rows]
        results = [evaluate_row(row, cfg) for row in rows]
        out = {
            "config": vars(args),
            "summary": aggregate(results),
            "results": results,
        }
    else:
        if not args.graph or not args.signal:
            raise SystemExit("Provide either --task-jsonl or both --graph and --signal.")
        graph = MemoryGraph.load_json(args.graph)
        out = {
            "config": vars(args),
            "result": traverse_graph(args.signal, graph, cfg),
        }

    text = json.dumps(out, ensure_ascii=False, indent=2)
    if args.save_json:
        Path(args.save_json).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
