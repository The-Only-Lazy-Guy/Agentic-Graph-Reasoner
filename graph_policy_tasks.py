from __future__ import annotations

"""
graph_policy_tasks.py

NGR-v0 task generator.

Coverage/novelty patch:
- Keeps shuffled candidate positions.
- Adds stronger duplicate/no_op examples without saying "already covered".
- Filters ambiguous node_mask/add_node tasks when candidates already cover the signal too well.
- Keeps all labels from controlled graph structure/corruption, not fallback winners.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from graph_core import CANONICAL_RELATIONS, MemoryGraph, canonical_relation, lexical_overlap


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def graph_files(graphs_dir: Path) -> List[Path]:
    return sorted(p for p in graphs_dir.glob("*.json") if p.is_file())


def clean_text(text: Any, max_len: int = 360) -> str:
    s = " ".join(str(text or "").split())
    return s[:max_len].rstrip()


def relation_phrase(rel: str) -> str:
    rel = canonical_relation(rel)
    return {
        "support": "supports",
        "contradict": "contradicts",
        "conflict": "conflicts with",
        "refute": "refutes",
        "refine": "refines",
        "depend": "depends on",
        "cause": "causes",
        "part_of": "is part of",
        "example_of": "is an example of",
        "related": "is related to",
        "imply": "implies",
    }.get(rel, "is related to")


def add_unique(out: List[str], seen: Set[str], nid: str, graph: MemoryGraph) -> None:
    nid = str(nid)
    if nid in graph.nodes and nid not in seen:
        seen.add(nid)
        out.append(nid)


def candidate_ids(
    graph: MemoryGraph,
    required: Sequence[str],
    rng: random.Random,
    *,
    max_candidates: int = 64,
    local_hops: int = 1,
) -> List[str]:
    required_clean: List[str] = []
    seen_required: Set[str] = set()
    for nid in required:
        add_unique(required_clean, seen_required, str(nid), graph)

    pool: List[str] = []
    seen_pool: Set[str] = set(seen_required)

    for nid in required_clean:
        for x in graph.local_neighborhood([nid], max_hops=local_hops, max_nodes=24):
            add_unique(pool, seen_pool, x, graph)

    all_ids = list(graph.nodes.keys())
    rng.shuffle(all_ids)
    for nid in all_ids:
        add_unique(pool, seen_pool, nid, graph)

    rng.shuffle(pool)

    final = list(required_clean)
    final.extend(pool[: max(0, max_candidates - len(final))])
    rng.shuffle(final)
    return final[:max_candidates]


def best_candidate_overlap(graph: MemoryGraph, signal: str, cids: Sequence[str]) -> float:
    best = 0.0
    for nid in cids:
        if nid not in graph.nodes:
            continue
        n = graph.nodes[nid]
        score = float(lexical_overlap(signal, f"{nid.replace('_', ' ')} {n.text}"))
        best = max(best, score)
    return best


def row_base(
    *,
    row_id: str,
    graph_path: Path,
    task_type: str,
    signal: str,
    candidate_node_ids: List[str],
    gold: Mapping[str, Any],
    meta: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "id": row_id,
        "task_type": task_type,
        "graph_path": str(graph_path),
        "signal": clean_text(signal),
        "candidate_node_ids": candidate_node_ids,
        "gold": dict(gold),
        "metadata": dict(meta or {}),
    }


def is_false_like(nid: str, text: Any, node_type: Any, metadata: Mapping[str, Any]) -> bool:
    raw = " ".join([
        str(nid).lower(),
        str(text).lower(),
        str(node_type).lower(),
        json.dumps(metadata or {}, ensure_ascii=False).lower(),
    ])
    return any(x in raw for x in [
        "false", "wrong", "incorrect", "misconception", "contradict",
        "contradiction", "conflict", "not true", "invalid"
    ])


def make_edge_mask_tasks(graph: MemoryGraph, graph_path: Path, rng: random.Random, limit: int, max_candidates: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    edges = [e for e in graph.edges if e.src in graph.nodes and e.dst in graph.nodes and e.src != e.dst]
    rng.shuffle(edges)
    for e in edges:
        if len(rows) >= limit:
            break
        rel = canonical_relation(e.relation)
        if rel not in CANONICAL_RELATIONS:
            continue
        src_node = graph.nodes[e.src]
        dst_node = graph.nodes[e.dst]
        signal = f"{clean_text(src_node.text, 140)} {relation_phrase(rel)} {clean_text(dst_node.text, 140)}"
        cands = candidate_ids(graph, [e.src, e.dst], rng, max_candidates=max_candidates)
        rows.append(row_base(
            row_id=f"{graph_path.stem}::edge_mask::{len(rows):06d}",
            graph_path=graph_path,
            task_type="edge_mask",
            signal=signal,
            candidate_node_ids=cands,
            gold={"edit_type": "link_nodes", "src_id": e.src, "dst_id": e.dst, "relation": rel},
            meta={"masked_edge": {"src": e.src, "dst": e.dst, "relation": rel}},
        ))
    return rows


def make_relation_corrupt_tasks(graph: MemoryGraph, graph_path: Path, rng: random.Random, limit: int, max_candidates: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    edges = [e for e in graph.edges if e.src in graph.nodes and e.dst in graph.nodes and e.src != e.dst]
    rng.shuffle(edges)
    for e in edges:
        if len(rows) >= limit:
            break
        rel = canonical_relation(e.relation)
        if rel not in CANONICAL_RELATIONS:
            continue
        alternatives = [r for r in CANONICAL_RELATIONS if r != rel]
        corrupt = rng.choice(alternatives)
        src_node = graph.nodes[e.src]
        dst_node = graph.nodes[e.dst]
        signal = (
            f"The relation between {e.src} and {e.dst} should be {rel}, not {corrupt}, "
            f"because {clean_text(src_node.text, 110)} {relation_phrase(rel)} {clean_text(dst_node.text, 110)}"
        )
        cands = candidate_ids(graph, [e.src, e.dst], rng, max_candidates=max_candidates)
        rows.append(row_base(
            row_id=f"{graph_path.stem}::relation_corrupt::{len(rows):06d}",
            graph_path=graph_path,
            task_type="relation_corrupt",
            signal=signal,
            candidate_node_ids=cands,
            gold={"edit_type": "link_nodes", "src_id": e.src, "dst_id": e.dst, "relation": rel},
            meta={"corrupted_relation": corrupt, "correct_relation": rel},
        ))
    return rows


_DUP_TEMPLATES = [
    "{text}",
    "{text}",
    "The graph should know that {text}",
    "Important fact: {text}",
    "Concept detail: {text}",
]


def make_duplicate_noop_tasks(graph: MemoryGraph, graph_path: Path, rng: random.Random, limit: int, max_candidates: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    ids = list(graph.nodes.keys())
    rng.shuffle(ids)
    cursor = 0
    while len(rows) < limit and cursor < len(ids) * 3:
        nid = ids[cursor % len(ids)]
        cursor += 1
        node = graph.nodes[nid]
        base = clean_text(node.text)
        if len(base) < 15:
            continue
        signal = rng.choice(_DUP_TEMPLATES).format(text=base)
        cands = candidate_ids(graph, [nid], rng, max_candidates=max_candidates)
        rows.append(row_base(
            row_id=f"{graph_path.stem}::duplicate_signal::{len(rows):06d}",
            graph_path=graph_path,
            task_type="duplicate_signal",
            signal=signal,
            candidate_node_ids=cands,
            gold={"edit_type": "no_op", "target_id": nid},
            meta={"covered_by": nid, "coverage_overlap": best_candidate_overlap(graph, signal, cands)},
        ))
    return rows


_FALSE_SIGNAL_TEMPLATES = [
    "Correct this false or conflicting graph claim: {text}",
    "The graph contains a wrong claim that should be resolved: {text}",
    "This node appears false or contradicted by the domain signal and should be repaired: {text}",
    "Resolve the misconception stored in this node: {text}",
    "A contradiction is present around this claim; resolve it instead of adding a duplicate: {text}",
]


def make_false_claim_tasks(
    graph: MemoryGraph,
    graph_path: Path,
    rng: random.Random,
    limit: int,
    max_candidates: int,
    *,
    multiplier: int = 6,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    false_ids = [
        nid for nid, n in graph.nodes.items()
        if is_false_like(nid, n.text, n.node_type, n.metadata or {})
    ]
    rng.shuffle(false_ids)

    expanded: List[str] = []
    for nid in false_ids:
        expanded.extend([nid] * max(1, multiplier))
    rng.shuffle(expanded)

    for nid in expanded:
        if len(rows) >= limit:
            break
        n = graph.nodes[nid]
        signal = rng.choice(_FALSE_SIGNAL_TEMPLATES).format(text=clean_text(n.text))
        cands = candidate_ids(graph, [nid], rng, max_candidates=max_candidates)
        rows.append(row_base(
            row_id=f"{graph_path.stem}::false_claim::{len(rows):06d}",
            graph_path=graph_path,
            task_type="false_claim",
            signal=signal,
            candidate_node_ids=cands,
            gold={"edit_type": "resolve_conflict", "target_id": nid},
            meta={"false_target": nid, "oversampled": True},
        ))
    return rows


def make_update_detail_tasks(graph: MemoryGraph, graph_path: Path, rng: random.Random, limit: int, max_candidates: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    ids = [
        nid for nid, n in graph.nodes.items()
        if str(n.node_type).lower() not in {"summary", "hub", "overview"}
        and len(str(n.text)) >= 30
        and not is_false_like(nid, n.text, n.node_type, n.metadata or {})
    ]
    rng.shuffle(ids)
    for nid in ids:
        if len(rows) >= limit:
            break
        n = graph.nodes[nid]
        signal = f"Refine the existing node {nid} with this detail: {clean_text(n.text)}"
        cands = candidate_ids(graph, [nid], rng, max_candidates=max_candidates)
        rows.append(row_base(
            row_id=f"{graph_path.stem}::update_existing::{len(rows):06d}",
            graph_path=graph_path,
            task_type="update_existing",
            signal=signal,
            candidate_node_ids=cands,
            gold={"edit_type": "update_node", "target_id": nid},
            meta={"update_target": nid},
        ))
    return rows


def make_node_mask_tasks(
    graph: MemoryGraph,
    graph_path: Path,
    rng: random.Random,
    limit: int,
    max_candidates: int,
    *,
    max_candidate_overlap: float = 0.86,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    ids = [
        nid for nid, n in graph.nodes.items()
        if str(n.node_type).lower() not in {"summary", "hub", "overview"}
        and len(str(n.text)) >= 30
        and not is_false_like(nid, n.text, n.node_type, n.metadata or {})
    ]
    rng.shuffle(ids)
    for nid in ids:
        if len(rows) >= limit:
            break
        node = graph.nodes[nid]
        local_edges = [e for e in graph.edges if e.src == nid or e.dst == nid]
        neighbor_ids = []
        for e in local_edges:
            other = e.dst if e.src == nid else e.src
            if other in graph.nodes:
                neighbor_ids.append(other)

        seen = set()
        neighbor_ids = [x for x in neighbor_ids if not (x in seen or seen.add(x))]
        if not neighbor_ids:
            continue

        required = neighbor_ids[:2]
        signal = clean_text(node.text)
        cands = candidate_ids(graph, required, rng, max_candidates=max_candidates)

        # Important: avoid ambiguous add_node examples where an existing candidate
        # already almost fully covers the masked signal.
        best_ov = best_candidate_overlap(graph, signal, cands)
        if best_ov >= max_candidate_overlap:
            continue

        rows.append(row_base(
            row_id=f"{graph_path.stem}::node_mask::{len(rows):06d}",
            graph_path=graph_path,
            task_type="node_mask",
            signal=signal,
            candidate_node_ids=cands,
            gold={
                "edit_type": "add_node",
                "masked_node_id": nid,
                "node_type": node.node_type,
                "evidence_ids": required,
                "relation": canonical_relation(local_edges[0].relation) if local_edges else "related",
            },
            meta={"masked_node_id": nid, "best_candidate_overlap": best_ov},
        ))
    return rows


def generate_for_graph(
    graph_path: Path,
    rng: random.Random,
    *,
    per_type: int,
    max_candidates: int,
    false_claim_multiplier: int,
    node_mask_max_candidate_overlap: float,
) -> List[Dict[str, Any]]:
    graph = MemoryGraph.load_json(graph_path)
    rows: List[Dict[str, Any]] = []
    rows += make_edge_mask_tasks(graph, graph_path, rng, per_type, max_candidates)
    rows += make_relation_corrupt_tasks(graph, graph_path, rng, per_type, max_candidates)
    rows += make_duplicate_noop_tasks(graph, graph_path, rng, per_type, max_candidates)
    rows += make_false_claim_tasks(graph, graph_path, rng, per_type, max_candidates, multiplier=false_claim_multiplier)
    rows += make_update_detail_tasks(graph, graph_path, rng, per_type, max_candidates)
    rows += make_node_mask_tasks(
        graph, graph_path, rng, per_type, max_candidates,
        max_candidate_overlap=node_mask_max_candidate_overlap,
    )
    return rows


def balanced_trim(rows: List[Dict[str, Any]], rng: random.Random, max_tasks: int) -> List[Dict[str, Any]]:
    if len(rows) <= max_tasks:
        rng.shuffle(rows)
        return rows

    by_edit: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_edit.setdefault(r["gold"]["edit_type"], []).append(r)

    for bucket in by_edit.values():
        rng.shuffle(bucket)

    edits = sorted(by_edit)
    base_quota = max_tasks // max(len(edits), 1)
    selected: List[Dict[str, Any]] = []
    leftovers: List[Dict[str, Any]] = []

    for edit in edits:
        bucket = by_edit[edit]
        take = min(base_quota, len(bucket))
        selected.extend(bucket[:take])
        leftovers.extend(bucket[take:])

    rng.shuffle(leftovers)
    selected.extend(leftovers[: max_tasks - len(selected)])
    rng.shuffle(selected)
    return selected[:max_tasks]


def split_rows(rows: List[Dict[str, Any]], rng: random.Random, val_ratio: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows = list(rows)
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * val_ratio)) if rows else 0
    return rows[n_val:], rows[:n_val]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs-dir", default="graphs")
    ap.add_argument("--out-dir", default="artifacts/tasks")
    ap.add_argument("--max-tasks", type=int, default=3000)
    ap.add_argument("--per-type-per-graph", type=int, default=180)
    ap.add_argument("--max-candidates", type=int, default=64)
    ap.add_argument("--false-claim-multiplier", type=int, default=6)
    ap.add_argument("--node-mask-max-candidate-overlap", type=float, default=0.86)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-balanced-trim", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    graphs = graph_files(Path(args.graphs_dir))
    if not graphs:
        raise SystemExit(f"No .json graphs found in {args.graphs_dir}")

    rows: List[Dict[str, Any]] = []
    for gp in graphs:
        try:
            rows.extend(generate_for_graph(
                gp,
                rng,
                per_type=args.per_type_per_graph,
                max_candidates=args.max_candidates,
                false_claim_multiplier=args.false_claim_multiplier,
                node_mask_max_candidate_overlap=args.node_mask_max_candidate_overlap,
            ))
        except Exception as exc:
            print(f"[warn] failed {gp}: {exc}")

    if args.no_balanced_trim:
        rng.shuffle(rows)
        rows = rows[: args.max_tasks]
    else:
        rows = balanced_trim(rows, rng, args.max_tasks)

    train, val = split_rows(rows, rng, args.val_ratio)

    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / "graph_policy_train.jsonl", train)
    write_jsonl(out_dir / "graph_policy_val.jsonl", val)

    summary = {
        "graphs": [str(p) for p in graphs],
        "total": len(rows),
        "train": len(train),
        "val": len(val),
        "task_counts": dict(Counter(r["task_type"] for r in rows)),
        "edit_counts": dict(Counter(r["gold"]["edit_type"] for r in rows)),
        "candidate_position_leakage_fix": "required nodes preserved but final candidate order shuffled",
        "coverage_novelty_patch": True,
        "false_claim_multiplier": args.false_claim_multiplier,
        "node_mask_max_candidate_overlap": args.node_mask_max_candidate_overlap,
        "balanced_trim": not args.no_balanced_trim,
    }

    (out_dir / "graph_policy_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
