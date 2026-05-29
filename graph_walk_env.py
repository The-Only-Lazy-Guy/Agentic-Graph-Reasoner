from __future__ import annotations

import math
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from graph_core import (
    CONTRADICTION_RELATIONS,
    Edge,
    MemoryGraph,
    Node,
    canonical_relation,
    lexical_overlap,
    relation_family,
)


@dataclass
class SessionGraph:
    session_id: str
    signal: str
    active_regions: List[str] = field(default_factory=list)
    anchor_nodes: List[str] = field(default_factory=list)
    visited_nodes: List[str] = field(default_factory=list)
    frontier_nodes: List[str] = field(default_factory=list)
    candidate_paths: List[List[str]] = field(default_factory=list)
    candidate_edits: List[Dict[str, Any]] = field(default_factory=list)
    tool_trace: List[Dict[str, Any]] = field(default_factory=list)
    step_budget: int = 6
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


REGION_NODE_TYPES = {"summary", "hub", "overview", "bridge"}
RELATION_STRENGTH = {
    "support": 1.00,
    "refine": 0.92,
    "depend": 0.88,
    "cause": 0.86,
    "part_of": 0.84,
    "example_of": 0.76,
    "related": 0.52,
    "contradict": 0.96,
}
COMPOSE_TABLE: Dict[Tuple[str, str], str] = {
    ("support", "support"): "support",
    ("support", "refine"): "support",
    ("refine", "support"): "support",
    ("example_of", "part_of"): "weak_support",
    ("part_of", "support"): "weak_support",
    ("support", "contradict"): "contradict",
    ("contradict", "support"): "contradict",
    ("contradict", "contradict"): "weak_support",
    ("related", "support"): "weak_related",
    ("support", "related"): "weak_related",
    ("related", "related"): "weak_related",
}


def _safe_metadata(node: Node) -> Mapping[str, Any]:
    return node.metadata if isinstance(node.metadata, Mapping) else {}


def _node_entry(graph: MemoryGraph, nid: str, score: float = 0.0, **extra: Any) -> Dict[str, Any]:
    node = graph.nodes[nid]
    out: Dict[str, Any] = {
        "id": nid,
        "score": float(score),
        "text": str(node.text),
        "node_type": str(node.node_type),
        "confidence": float(node.confidence),
        "importance": float(node.importance),
    }
    out.update(extra)
    return out


def _is_region_node(node: Node) -> bool:
    nid = str(node.id).lower()
    if str(node.node_type).lower() in REGION_NODE_TYPES:
        return True
    return any(tok in nid for tok in ("hub", "summary", "overview", "bridge"))


def _lexical_rank(graph: MemoryGraph, query: str, *, top_k: int) -> List[Dict[str, Any]]:
    scored: List[Tuple[float, str]] = []
    for nid, node in graph.nodes.items():
        score = float(lexical_overlap(query, node.text))
        if score <= 0.0:
            continue
        scored.append((score, nid))
    scored.sort(key=lambda x: (x[0], graph.nodes[x[1]].importance, graph.nodes[x[1]].confidence), reverse=True)
    return [_node_entry(graph, nid, score, source="lexical") for score, nid in scored[:top_k]]


def _compose_relations(relations: Sequence[str]) -> str:
    if not relations:
        return "unknown"
    current = canonical_relation(relations[0])
    for rel in relations[1:]:
        nxt = canonical_relation(rel)
        current = COMPOSE_TABLE.get((current, nxt), "weak_related" if "related" in {current, nxt} else nxt)
    return current


def _path_structural_score(graph: MemoryGraph, path: Sequence[str]) -> float:
    if len(path) < 2:
        return 0.0
    vals: List[float] = []
    for a, b in zip(path, path[1:]):
        edge = graph.edge_between(a, b)
        if edge is None:
            continue
        rel = canonical_relation(edge.relation)
        vals.append(float(edge.strength) * RELATION_STRENGTH.get(rel, 0.50))
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def route_regions(graph: MemoryGraph, signal: str, *, k: int = 4) -> List[Dict[str, Any]]:
    anchors = _lexical_rank(graph, signal, top_k=max(8, k * 3))
    anchor_ids = {str(item["id"]) for item in anchors}
    regions: List[Tuple[float, str, int, float]] = []
    for nid, node in graph.nodes.items():
        if not _is_region_node(node):
            continue
        own = float(lexical_overlap(signal, node.text))
        local = graph.local_neighborhood([nid], max_hops=1, max_nodes=24)
        anchor_hits = sum(1 for x in local if x in anchor_ids)
        anchor_boost = 0.0
        if local:
            anchor_boost = sum(float(lexical_overlap(signal, graph.nodes[x].text)) for x in local[:12]) / max(len(local[:12]), 1)
        score = own + 0.18 * float(node.importance) + 0.10 * float(node.confidence) + 0.10 * min(anchor_hits, 3) + 0.15 * anchor_boost
        regions.append((score, nid, anchor_hits, anchor_boost))
    regions.sort(key=lambda x: (x[0], x[2], graph.nodes[x[1]].importance), reverse=True)
    out: List[Dict[str, Any]] = []
    for score, nid, anchor_hits, anchor_boost in regions[:k]:
        out.append(_node_entry(graph, nid, score, source="route_regions", anchor_hits=int(anchor_hits), anchor_boost=float(anchor_boost)))
    return out


def choose_anchor_nodes(
    graph: MemoryGraph,
    signal: str,
    regions: Sequence[str],
    *,
    top_k: int = 6,
) -> List[Dict[str, Any]]:
    candidates: Set[str] = set()
    for rid in regions:
        if rid not in graph.nodes:
            continue
        candidates.update(graph.local_neighborhood([rid], max_hops=1, max_nodes=24))
        candidates.add(rid)
    if not candidates:
        return _lexical_rank(graph, signal, top_k=top_k)
    ranked: List[Tuple[float, str]] = []
    for nid in candidates:
        node = graph.nodes[nid]
        score = float(lexical_overlap(signal, node.text)) + 0.10 * float(node.importance) + 0.06 * float(node.confidence)
        if _is_region_node(node):
            score -= 0.03
        ranked.append((score, nid))
    ranked.sort(key=lambda x: (x[0], graph.nodes[x[1]].importance), reverse=True)
    return [_node_entry(graph, nid, score, source="anchor") for score, nid in ranked[:top_k] if score > 0.0]


def get_frontier(
    graph: MemoryGraph,
    active_nodes: Sequence[str],
    *,
    relation_filter: Optional[Sequence[str]] = None,
    depth: int = 1,
) -> List[Dict[str, Any]]:
    rel_filter = {canonical_relation(x) for x in (relation_filter or []) if str(x).strip()}
    q = deque((nid, 0) for nid in active_nodes if nid in graph.nodes)
    seen: Set[str] = {nid for nid in active_nodes if nid in graph.nodes}
    out: List[Dict[str, Any]] = []
    while q:
        nid, dist = q.popleft()
        if dist >= max(1, int(depth)):
            continue
        for nxt in sorted(graph._adj.get(nid, []), key=lambda x: (-graph.nodes[x].importance, x)):
            if nxt in seen:
                continue
            edge = graph.edge_between(nid, nxt)
            if edge is None:
                continue
            rel = canonical_relation(edge.relation)
            if rel_filter and rel not in rel_filter and relation_family(rel) not in rel_filter:
                continue
            seen.add(nxt)
            q.append((nxt, dist + 1))
            score = RELATION_STRENGTH.get(rel, 0.50) + 0.08 * float(graph.nodes[nxt].importance) - 0.06 * dist
            out.append(_node_entry(graph, nxt, score, source="frontier", via_from=nid, via_relation=rel, depth=dist + 1))
    out.sort(key=lambda x: (-float(x.get("score", 0.0)), str(x.get("id", ""))))
    return out


def score_path(
    graph: MemoryGraph,
    signal: str,
    path: Sequence[str],
    *,
    deterministic_only: bool = True,
) -> Dict[str, Any]:
    if not path:
        return {
            "path_nodes": [],
            "path_edges": [],
            "relation_composition": "unknown",
            "semantic_score": 0.0,
            "structural_score": 0.0,
            "confidence": 0.0,
            "is_conflict_path": False,
        }
    edges: List[Dict[str, Any]] = []
    relations: List[str] = []
    semantic_parts: List[float] = []
    node_conf_parts: List[float] = []
    for nid in path:
        if nid not in graph.nodes:
            continue
        node = graph.nodes[nid]
        semantic_parts.append(float(lexical_overlap(signal, node.text)))
        node_conf_parts.append((float(node.confidence) + float(node.importance)) / 2.0)
    for a, b in zip(path, path[1:]):
        edge = graph.edge_between(a, b)
        if edge is None:
            continue
        rel = canonical_relation(edge.relation)
        relations.append(rel)
        edges.append({"src": a, "dst": b, "relation": rel, "strength": float(edge.strength)})
    composition = _compose_relations(relations)
    semantic_score = sum(semantic_parts) / max(len(semantic_parts), 1)
    structural_score = _path_structural_score(graph, path)
    confidence = 0.45 * semantic_score + 0.40 * structural_score + 0.15 * (sum(node_conf_parts) / max(len(node_conf_parts), 1))
    confidence -= 0.06 * max(0, len(path) - 2)

    try:
        from scorers.learned_scorers import score_path_learned
        learned = score_path_learned(signal, {
            "structural_score": structural_score,
            "semantic_score": semantic_score,
            "path_nodes": path
        })
        if learned > 0.0:
            confidence = learned
    except (ImportError, Exception):
        pass
    is_conflict = any(rel in CONTRADICTION_RELATIONS for rel in relations) or composition == "contradict"
    return {
        "path_nodes": list(path),
        "path_edges": edges,
        "relation_composition": composition,
        "semantic_score": float(semantic_score),
        "structural_score": float(structural_score),
        "confidence": float(max(0.0, min(1.0, confidence))),
        "is_conflict_path": bool(is_conflict),
        "deterministic_only": bool(deterministic_only),
    }


def find_paths(
    graph: MemoryGraph,
    src_ids: Sequence[str],
    dst_ids: Optional[Sequence[str]] = None,
    *,
    max_len: int = 3,
    beam_size: int = 6,
    relation_filter: Optional[Sequence[str]] = None,
    signal: str = "",
) -> List[Dict[str, Any]]:
    sources = [nid for nid in src_ids if nid in graph.nodes]
    targets = {nid for nid in (dst_ids or []) if nid in graph.nodes}
    rel_filter = {canonical_relation(x) for x in (relation_filter or []) if str(x).strip()}
    frontier: List[List[str]] = [[nid] for nid in sources]
    complete: List[List[str]] = []
    for _ in range(max(1, int(max_len))):
        next_frontier: List[Tuple[float, List[str]]] = []
        for path in frontier:
            tail = path[-1]
            for nxt in graph._adj.get(tail, []):
                if nxt in path:
                    continue
                edge = graph.edge_between(tail, nxt)
                if edge is None:
                    continue
                rel = canonical_relation(edge.relation)
                if rel_filter and rel not in rel_filter and relation_family(rel) not in rel_filter:
                    continue
                new_path = path + [nxt]
                if targets and nxt in targets:
                    complete.append(new_path)
                partial_score = _path_structural_score(graph, new_path) + 0.25 * (float(lexical_overlap(signal, graph.nodes[nxt].text)) if signal else 0.0)
                next_frontier.append((partial_score, new_path))
        next_frontier.sort(key=lambda x: x[0], reverse=True)
        frontier = [path for _, path in next_frontier[: max(beam_size, 1)]]
        if targets and complete:
            break
    if not complete:
        complete = frontier[: max(beam_size, 1)]
    scored = [score_path(graph, signal, path, deterministic_only=True) for path in complete]
    scored.sort(key=lambda x: (float(x.get("confidence", 0.0)), float(x.get("structural_score", 0.0))), reverse=True)
    return scored[: max(beam_size, 1)]


def find_conflicting_paths(
    graph: MemoryGraph,
    node_ids: Sequence[str],
    *,
    max_len: int = 3,
) -> List[Dict[str, Any]]:
    seeds = [nid for nid in node_ids if nid in graph.nodes]
    if not seeds:
        return []
    paths: List[Dict[str, Any]] = []
    local = graph.local_neighborhood(seeds, max_hops=max_len, max_nodes=48)
    for src in seeds[:4]:
        for dst in local[:24]:
            if src == dst:
                continue
            for info in find_paths(graph, [src], [dst], max_len=max_len, beam_size=3):
                if bool(info.get("is_conflict_path")):
                    paths.append(info)
    paths.sort(key=lambda x: (float(x.get("confidence", 0.0)), -len(x.get("path_nodes", []))), reverse=True)
    dedup: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, ...]] = set()
    for item in paths:
        key = tuple(item.get("path_nodes", []) or [])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(item)
        if len(dedup) >= 6:
            break
    return dedup


def create_session_graph(
    graph: MemoryGraph,
    signal: str,
    *,
    region_k: int = 4,
    step_budget: int = 6,
) -> SessionGraph:
    regions = route_regions(graph, signal, k=region_k)
    region_ids = [str(x["id"]) for x in regions]
    anchors = choose_anchor_nodes(graph, signal, region_ids, top_k=6)
    anchor_ids = [str(x["id"]) for x in anchors]
    frontier = get_frontier(graph, anchor_ids, relation_filter=None, depth=1)
    frontier_ids = [str(x["id"]) for x in frontier[:8]]
    candidate_paths: List[List[str]] = []
    if anchor_ids:
        path_infos = find_paths(graph, anchor_ids[:2], frontier_ids[:4], max_len=3, beam_size=4, signal=signal)
        candidate_paths = [list(x.get("path_nodes", []) or []) for x in path_infos]
    return SessionGraph(
        session_id=f"sg_{uuid.uuid4().hex[:10]}",
        signal=str(signal),
        active_regions=region_ids,
        anchor_nodes=anchor_ids,
        visited_nodes=list(dict.fromkeys(region_ids + anchor_ids)),
        frontier_nodes=frontier_ids,
        candidate_paths=candidate_paths,
        candidate_edits=[],
        tool_trace=[],
        step_budget=int(step_budget),
        metadata={"created_at": time.time()},
    )


def beam_walk(
    graph: MemoryGraph,
    session: SessionGraph,
    *,
    max_steps: int = 4,
    beam_size: int = 4,
) -> Dict[str, Any]:
    visited: Set[str] = set(session.visited_nodes)
    active = list(session.anchor_nodes or session.active_regions)
    frontier_trace: List[Dict[str, Any]] = []
    best_paths: List[Dict[str, Any]] = []
    for step in range(max(1, int(max_steps))):
        frontier = get_frontier(graph, active, relation_filter=None, depth=1)
        frontier = [x for x in frontier if str(x.get("id", "")) not in visited]
        if not frontier:
            break
        frontier = frontier[: max(beam_size * 2, 4)]
        frontier_trace.append({"step": step + 1, "frontier": frontier})
        frontier_ids = [str(x["id"]) for x in frontier[:beam_size]]
        paths = find_paths(
            graph,
            session.anchor_nodes[:2] if session.anchor_nodes else active[:2],
            frontier_ids,
            max_len=3,
            beam_size=beam_size,
            signal=session.signal,
        )
        if paths:
            best_paths = paths
        active = frontier_ids
        visited.update(active)
    session.visited_nodes = list(dict.fromkeys(list(visited)))
    session.frontier_nodes = [str(x.get("id", "")) for x in (frontier_trace[-1]["frontier"] if frontier_trace else []) if str(x.get("id", ""))]
    session.candidate_paths = [list(x.get("path_nodes", []) or []) for x in best_paths]
    return {
        "session_graph": session.to_dict(),
        "candidate_paths": best_paths,
        "frontier_trace": frontier_trace,
    }


def graph_walk_context_for_signal(
    graph: MemoryGraph,
    signal: str,
    *,
    top_k: int = 8,
    region_k: int = 4,
    beam_size: int = 4,
) -> Dict[str, Any]:
    session = create_session_graph(graph, signal, region_k=region_k, step_budget=6)
    walk = beam_walk(graph, session, max_steps=3, beam_size=beam_size)
    merged: Dict[str, Dict[str, Any]] = {}
    for rid in session.active_regions:
        if rid in graph.nodes:
            merged[rid] = _node_entry(graph, rid, 0.28, source="region")
    for idx, nid in enumerate(session.anchor_nodes):
        if nid in graph.nodes:
            merged[nid] = _node_entry(graph, nid, 0.70 - 0.04 * idx, source="anchor")
    for idx, nid in enumerate(session.frontier_nodes):
        if nid in graph.nodes and nid not in merged:
            merged[nid] = _node_entry(graph, nid, 0.48 - 0.03 * idx, source="frontier")
    for path_info in walk.get("candidate_paths", []) or []:
        for nid in path_info.get("path_nodes", []) or []:
            if nid in graph.nodes:
                prev = merged.get(nid, {})
                merged[nid] = _node_entry(
                    graph,
                    nid,
                    max(float(prev.get("score", 0.0) or 0.0), float(path_info.get("confidence", 0.0))),
                    source="path",
                )
    nodes = sorted(merged.values(), key=lambda x: (-float(x.get("score", 0.0)), str(x.get("id", ""))))[:top_k]
    return {
        "nodes": nodes,
        "session_graph": session.to_dict(),
        "candidate_paths": walk.get("candidate_paths", []),
        "frontier_trace": walk.get("frontier_trace", []),
        "regions": route_regions(graph, signal, k=region_k),
    }


def hybrid_context_for_signal(
    graph: MemoryGraph,
    signal: str,
    *,
    top_k: int = 8,
    region_k: int = 4,
    beam_size: int = 4,
) -> Dict[str, Any]:
    # 1. Run graph_dc_context_for_signal for routing
    from graph_dc_env import graph_dc_context_for_signal
    dc_packet = graph_dc_context_for_signal(
        graph, signal, top_k=top_k, region_k=region_k, max_depth=3, beam_size=beam_size
    )

    # 2. Take top selected regions / candidate edit zones
    edit_zones = dc_packet.get("candidate_edit_zones", [])
    if not edit_zones:
        # Fallback if D&C returns nothing
        edit_zones = [n["id"] for n in dc_packet.get("nodes", [])[:2] if "id" in n]

    # 3. Feed those as start regions/anchors into graph_walk
    session = create_session_graph(graph, signal, region_k=region_k, step_budget=6)
    session.anchor_nodes = edit_zones[:4]
    session.active_regions = edit_zones
    session.visited_nodes = list(dict.fromkeys(edit_zones[:4]))

    walk = beam_walk(graph, session, max_steps=3, beam_size=beam_size)

    # Merge nodes
    merged: Dict[str, Dict[str, Any]] = {}
    for n in dc_packet.get("nodes", []):
        nid = str(n.get("id", ""))
        if nid:
            merged[nid] = n

    for path_info in walk.get("candidate_paths", []) or []:
        for nid in path_info.get("path_nodes", []) or []:
            if nid in graph.nodes:
                prev = merged.get(nid, {})
                # Use actual lexical overlap instead of path confidence to avoid irrelevant nodes bypassing the guard
                score = float(lexical_overlap(signal, graph.nodes[nid].text))
                merged[nid] = _node_entry(
                    graph,
                    nid,
                    max(float(prev.get("score", 0.0) or 0.0), score),
                    source="hybrid_path",
                )
    
    for idx, nid in enumerate(session.frontier_nodes):
        if nid in graph.nodes and nid not in merged:
            score = float(lexical_overlap(signal, graph.nodes[nid].text))
            merged[nid] = _node_entry(graph, nid, score, source="hybrid_frontier")

    nodes = sorted(merged.values(), key=lambda x: (-float(x.get("score", 0.0)), str(x.get("id", ""))))[:top_k]

    if nodes and float(nodes[0].get("score", 0.0)) < 0.22:
        nodes = []
        edit_zones = []

    # 4. Build merged context packet
    return {
        "nodes": nodes,
        "session_graph": session.to_dict(),
        "global_summary_candidates": dc_packet.get("global_summary_candidates", []),
        "selected_region_path": dc_packet.get("selected_region_path", []),
        "candidate_paths": walk.get("candidate_paths", []),
        "candidate_edit_zones": edit_zones,
        "frontier_trace": walk.get("frontier_trace", []),
        "frontier": dc_packet.get("frontier", []),
    }

