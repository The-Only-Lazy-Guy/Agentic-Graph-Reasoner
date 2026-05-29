"""
graph_dc_env.py — DeepSeek-inspired Graph Divide-and-Conquer Memory Features (Option B)

Three-band memory hierarchy adapted from DeepSeek V4's long-context architecture:

  Band 1 (HCA-like): dense scoring over heavily-compressed global region summaries
    → cheap global awareness; routes signal to the likely graph region
    → implemented: route_global_regions()

  Band 2 (CSA-like): sparse selection over compressed child regions / path sketches
    → narrows focus to the best subregion without expanding all nodes
    → implemented: rank_child_regions()

  Band 3 (exact): graph-walk evidence after D&C zoom completes
    → delegates to graph_walk_env.beam_walk for exact node/path evidence
    → implemented: zoom_region() bottom-level leaf case

D&C zoom replaces sliding-window attention:
  tokens are linear → recency/locality matters → sliding window
  graphs are relational → branch relevance matters → hierarchical zoom

Hybrid use (recommended):
  graph_dc_env  → routes + compresses → selects promising region
  graph_walk_env → walks exact nodes → collects evidence paths
  LLM planner   → writes final graph edit JSON

CLI:
  python graph_dc_env.py --graph graphs/cs4.json --signal "..." --beam-size 4 --max-depth 3
  python graph_dc_env.py --graph graphs/cs4.json --signal "..." --out-json artifacts/tmp/dc_test.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from graph_core import (
    CONTRADICTION_RELATIONS,
    MemoryGraph,
    Node,
    canonical_relation,
    lexical_overlap,
)
from graph_walk_env import (
    RELATION_STRENGTH,
    SessionGraph,
    _is_region_node,
    _node_entry,
    _compose_relations,
    _path_structural_score,
    beam_walk,
    create_session_graph,
    find_conflicting_paths,
    find_paths,
    get_frontier,
    score_path,
)

# ── constants ─────────────────────────────────────────────────────────────────

_SMALL_REGION_SIZE = 6    # regions with ≤ this many nodes are treated as leaves
_MAX_COMPRESS_TEXTS = 8   # max neighbor texts included in compressed summary

_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "in", "on", "at", "to", "for", "of", "and", "or", "but",
    "with", "as", "it", "its", "this", "that", "these", "those", "from",
    "by", "not", "can", "may", "might", "both", "each", "when", "which",
    "so", "also", "then", "than", "they", "their", "there", "what", "how",
    "all", "any", "if", "only", "into", "over", "such", "while",
})


# ── dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class RegionSummary:
    """Compressed representation of a graph region (Band 1 / Band 2 memory)."""
    region_id: str
    summary_text: str
    keywords: List[str]
    node_count: int
    child_region_ids: List[str]
    centroid_score: float       # region node importance
    depth: int                  # 0 = top-level; incremented by zoom
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PathSummary:
    """Compressed representation of a path with composed relation (Band 2 memory)."""
    path_id: str
    nodes: List[str]
    relations: List[str]
    composed_relation: str
    compressed_text: str
    path_strength: float
    is_conflict_path: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── private helpers ───────────────────────────────────────────────────────────

def _extract_keywords(texts: Sequence[str], *, max_k: int = 20) -> List[str]:
    freq: Counter = Counter()
    for text in texts:
        for word in re.findall(r"[a-z][a-z0-9_-]*", str(text).lower()):
            if len(word) >= 4 and word not in _STOPWORDS:
                freq[word] += 1
    return [w for w, _ in freq.most_common(max_k)]


def _build_region_summaries(graph: MemoryGraph) -> Dict[str, RegionSummary]:
    """Scan graph once, compress every region node into a RegionSummary."""
    result: Dict[str, RegionSummary] = {}
    for nid, node in graph.nodes.items():
        if _is_region_node(node):
            rs = compress_region(graph, nid)
            if rs is not None:
                result[nid] = rs
    return result


# ── public: compression ───────────────────────────────────────────────────────

def compress_region(
    graph: MemoryGraph,
    region_id: str,
    *,
    max_nodes: int = 24,
) -> Optional[RegionSummary]:
    """Build a RegionSummary for one region node.

    Combines the region node text with top-importance neighbor texts to form a
    compact description that can be lexically scored against a signal cheaply.
    """
    if region_id not in graph.nodes:
        return None
    region_node = graph.nodes[region_id]
    local_ids = graph.local_neighborhood([region_id], max_hops=1, max_nodes=max_nodes)

    texts: List[str] = [str(region_node.text)]
    child_region_ids: List[str] = []
    ranked_neighbors: List[Tuple[float, str]] = []

    for nid in local_ids:
        if nid == region_id:
            continue
        node = graph.nodes.get(nid)
        if node is None:
            continue
        if _is_region_node(node):
            child_region_ids.append(nid)
        else:
            ranked_neighbors.append((float(node.importance), str(node.text)))

    ranked_neighbors.sort(reverse=True)
    for _, txt in ranked_neighbors[:_MAX_COMPRESS_TEXTS]:
        texts.append(txt)

    summary_text = " ".join(texts)
    keywords = _extract_keywords(texts, max_k=20)

    return RegionSummary(
        region_id=region_id,
        summary_text=summary_text,
        keywords=keywords,
        node_count=len(local_ids),
        child_region_ids=child_region_ids,
        centroid_score=float(region_node.importance),
        depth=0,
        metadata={"compressed_from": region_id},
    )


def compress_path(
    graph: MemoryGraph,
    path_nodes: Sequence[str],
    *,
    signal: str = "",
) -> PathSummary:
    """Build a PathSummary with composed relation and evidence text."""
    nodes = [nid for nid in path_nodes if nid in graph.nodes]
    relations: List[str] = []
    text_parts: List[str] = []

    for nid in nodes:
        text_parts.append(str(graph.nodes[nid].text))

    for a, b in zip(nodes, nodes[1:]):
        edge = graph.edge_between(a, b)
        if edge is not None:
            relations.append(canonical_relation(edge.relation))

    composed = _compose_relations(relations) if relations else "unknown"
    structural = _path_structural_score(graph, nodes)

    path_scored = score_path(graph, signal, nodes) if signal else {}
    is_conflict = bool(path_scored.get("is_conflict_path", False)) or (
        any(r in CONTRADICTION_RELATIONS for r in relations)
    )

    compressed_text = " → ".join(text_parts[:4]) if text_parts else ""

    return PathSummary(
        path_id=f"ps_{abs(hash(tuple(nodes)))%100000:05d}",
        nodes=list(nodes),
        relations=relations,
        composed_relation=composed,
        compressed_text=compressed_text[:300],
        path_strength=float(structural),
        is_conflict_path=is_conflict,
    )


# ── public: scoring ───────────────────────────────────────────────────────────

def score_region_summary(signal: str, region: RegionSummary) -> float:
    """Score a compressed region summary against a signal.

    Attempts to use learned MLP scorer; falls back to heuristic if model not found.
    """
    try:
        from scorers.learned_scorers import score_region_learned
        learned = score_region_learned(signal, region.to_dict())
        if learned > 0.0:
            return learned
    except (ImportError, Exception):
        pass

    text_score = float(lexical_overlap(signal, region.summary_text))
    keyword_score = 0.0
    if region.keywords:
        signal_lower = signal.lower()
        hits = sum(1 for kw in region.keywords if kw in signal_lower)
        keyword_score = hits / len(region.keywords)
    # Prefer larger, more important regions slightly
    size_boost = min(0.08, math.log1p(max(region.node_count, 1)) * 0.015)
    centroid_boost = 0.04 * float(region.centroid_score)
    return text_score + 0.25 * keyword_score + size_boost + centroid_boost


# ── public: routing ───────────────────────────────────────────────────────────

def route_global_regions(
    graph: MemoryGraph,
    signal: str,
    *,
    k: int = 8,
    _cache: Optional[Dict[str, RegionSummary]] = None,
) -> List[RegionSummary]:
    """Band 1 (HCA-like): dense scoring over all compressed region summaries.

    Cheap because region nodes are few (typically 5-30 per graph). Returns top-k
    regions sorted by relevance. This gives global routing before any expansion.
    """
    summaries = _cache if _cache is not None else _build_region_summaries(graph)
    scored: List[Tuple[float, RegionSummary]] = [
        (score_region_summary(signal, rs), rs) for rs in summaries.values()
    ]
    scored.sort(key=lambda x: (x[0], x[1].centroid_score), reverse=True)
    return [rs for score, rs in scored[:k] if score > 0.0]


def rank_child_regions(
    graph: MemoryGraph,
    signal: str,
    parent_region_id: str,
    *,
    k: int = 4,
    _cache: Optional[Dict[str, RegionSummary]] = None,
) -> List[RegionSummary]:
    """Band 2 (CSA-like): sparse selection over child regions of a parent.

    Identifies child regions within the parent, scores them, and returns top-k.
    If the parent has no child regions, returns an empty list (caller should use
    beam_walk directly in that case).
    """
    summaries = _cache if _cache is not None else _build_region_summaries(graph)
    parent = summaries.get(parent_region_id)
    if parent is None:
        parent = compress_region(graph, parent_region_id)
    if parent is None:
        return []

    children: List[RegionSummary] = []
    for child_id in parent.child_region_ids:
        rs = summaries.get(child_id) or compress_region(graph, child_id)
        if rs is not None:
            children.append(rs)

    scored: List[Tuple[float, RegionSummary]] = [
        (score_region_summary(signal, rs), rs) for rs in children
    ]
    scored.sort(key=lambda x: (x[0], x[1].centroid_score), reverse=True)
    return [rs for _, rs in scored[:k]]


# ── public: D&C zoom ──────────────────────────────────────────────────────────

def zoom_region(
    graph: MemoryGraph,
    signal: str,
    region_id: str,
    *,
    depth: int = 3,
    beam_size: int = 4,
    _cache: Optional[Dict[str, RegionSummary]] = None,
) -> Dict[str, Any]:
    """D&C zoom: replace sliding window with hierarchical graph zoom.

    Recursively enters the most relevant branch:
      depth > 0 and children exist → rank_child_regions → recurse
      depth == 0 or no children    → beam_walk (exact evidence leaf)

    Returns: {region_id, depth_reached, selected_children, nodes, paths}
    """
    summaries = _cache if _cache is not None else _build_region_summaries(graph)
    region = summaries.get(region_id) or compress_region(graph, region_id)
    if region is None:
        return {"region_id": region_id, "depth_reached": 0, "nodes": [], "paths": []}

    is_leaf = (
        depth <= 0
        or region.node_count <= _SMALL_REGION_SIZE
        or not region.child_region_ids
    )

    if is_leaf:
        # Exact evidence: use graph-walk beam inside this region
        anchor_ids: List[str] = [region_id]
        for nid in sorted(
            graph._adj.get(region_id, []),
            key=lambda x: -graph.nodes[x].importance if x in graph.nodes else 0,
        )[:4]:
            if nid in graph.nodes and not _is_region_node(graph.nodes[nid]):
                anchor_ids.append(nid)

        session = SessionGraph(
            session_id=f"dc_leaf_{region_id[:12]}",
            signal=signal,
            active_regions=[region_id],
            anchor_nodes=anchor_ids[:4],
            visited_nodes=list(dict.fromkeys([region_id] + anchor_ids[:4])),
            frontier_nodes=[],
            candidate_paths=[],
            step_budget=3,
            metadata={"source": "graph_dc_zoom"},
        )
        walk_result = beam_walk(graph, session, max_steps=2, beam_size=beam_size)

        node_entries = [
            _node_entry(graph, nid, float(lexical_overlap(signal, graph.nodes[nid].text)))
            for nid in session.visited_nodes
            if nid in graph.nodes
        ]
        return {
            "region_id": region_id,
            "depth_reached": 0,
            "selected_children": [],
            "nodes": sorted(node_entries, key=lambda n: -float(n.get("score", 0.0)))[:8],
            "paths": walk_result.get("candidate_paths", [])[:beam_size],
            "session_graph": walk_result.get("session_graph", {}),
        }

    # Recursive case: rank children, zoom into top-beam_size
    children = rank_child_regions(graph, signal, region_id, k=beam_size, _cache=summaries)

    all_nodes: Dict[str, Any] = {}
    all_paths: List[Dict[str, Any]] = []

    for child in children[:beam_size]:
        child_result = zoom_region(
            graph, signal, child.region_id,
            depth=depth - 1, beam_size=beam_size, _cache=summaries,
        )
        for n in child_result.get("nodes", []):
            nid = str(n.get("id", ""))
            if nid and nid not in all_nodes:
                all_nodes[nid] = n
        all_paths.extend(child_result.get("paths", []))

    # Also include anchor nodes directly under this region (not just its children)
    direct_anchors = [
        nid for nid in graph._adj.get(region_id, [])
        if nid in graph.nodes and not _is_region_node(graph.nodes[nid])
    ][:4]
    for nid in direct_anchors:
        if nid not in all_nodes:
            score = float(lexical_overlap(signal, graph.nodes[nid].text))
            all_nodes[nid] = _node_entry(graph, nid, score, source="dc_direct")

    ranked_nodes = sorted(
        all_nodes.values(),
        key=lambda n: -(float(n.get("score", 0.0)) + float(lexical_overlap(signal, n.get("text", "")))),
    )
    ranked_paths = sorted(all_paths, key=lambda p: -float(p.get("confidence", 0.0)))

    return {
        "region_id": region_id,
        "depth_reached": depth,
        "selected_children": [c.region_id for c in children],
        "nodes": ranked_nodes[:10],
        "paths": ranked_paths[:beam_size],
    }


# ── public: main entry ────────────────────────────────────────────────────────

def graph_dc_context_for_signal(
    graph: MemoryGraph,
    signal: str,
    *,
    top_k: int = 8,
    region_k: int = 4,
    max_depth: int = 3,
    beam_size: int = 4,
) -> Dict[str, Any]:
    """Full D&C pipeline for one signal.

    Returns a structured context packet that the planner and consistency_regret_loop
    can consume. The 'nodes' field has the same format as graph_walk_context_for_signal
    so it is a drop-in replacement for the retrieval step.

    Packet fields:
      nodes                    — top-k scored nodes (same format as graph_walk mode)
      global_summary_candidates — Band 1 region scores (HCA-like)
      selected_region_path     — regions selected by D&C zoom (CSA-like)
      candidate_paths          — top scored paths from exact evidence
      candidate_edit_zones     — suggested region IDs for graph edits
      frontier                 — 1-hop frontier from top anchors
      dc_trace                 — full D&C zoom trace (for logging/training data)
    """
    region_cache = _build_region_summaries(graph)

    # Band 1: dense global routing over all compressed region summaries
    global_regions = route_global_regions(graph, signal, k=max(8, region_k * 2), _cache=region_cache)
    top_regions = global_regions[:region_k]

    # Band 2+3: D&C zoom into each selected region
    all_nodes: Dict[str, Any] = {}
    all_paths: List[Dict[str, Any]] = []
    dc_trace: List[Dict[str, Any]] = []

    for rs in top_regions:
        zoom_result = zoom_region(
            graph, signal, rs.region_id,
            depth=max_depth, beam_size=beam_size, _cache=region_cache,
        )
        dc_trace.append({
            "region_id": rs.region_id,
            "region_score": score_region_summary(signal, rs),
            "depth_reached": zoom_result.get("depth_reached", 0),
            "selected_children": zoom_result.get("selected_children", []),
            "node_count": len(zoom_result.get("nodes", [])),
        })
        for n in zoom_result.get("nodes", []):
            nid = str(n.get("id", ""))
            if nid and nid not in all_nodes:
                all_nodes[nid] = n
        # Deduplicate paths by node sequence before extending
        seen_path_keys: Set[Tuple[str, ...]] = {
            tuple(p.get("path_nodes", []) or []) for p in all_paths
        }
        for p in zoom_result.get("paths", []):
            key = tuple(p.get("path_nodes", []) or [])
            if key and key not in seen_path_keys:
                seen_path_keys.add(key)
                all_paths.append(p)

    # Supplement with direct lexical hits for nodes that D&C routing may have missed.
    # Must happen BEFORE final ranking so high-scoring fallback nodes can displace
    # low-scoring D&C nodes when top_k is already full.
    _direct_threshold = 0.22
    for nid, node in graph.nodes.items():
        if nid in all_nodes:
            continue
        score = float(lexical_overlap(signal, node.text))
        if score >= _direct_threshold:
            all_nodes[nid] = _node_entry(graph, nid, score, source="dc_lexical_fallback")

    # Re-rank everything (including fallback nodes) by signal relevance
    ranked_nodes = sorted(
        all_nodes.values(),
        key=lambda n: -(
            float(n.get("score", 0.0))
            + float(lexical_overlap(signal, n.get("text", "")))
        ),
    )[:top_k]

    ranked_paths = sorted(
        all_paths,
        key=lambda p: -float(p.get("confidence", 0.0)),
    )[:beam_size]

    # Candidate edit zones: the regions with the highest signal overlap
    candidate_edit_zones = [rs.region_id for rs in top_regions[:2]]

    # Frontier from top anchors (local expansion after D&C routing)
    anchor_ids = [str(n["id"]) for n in ranked_nodes[:3] if "id" in n]
    frontier = get_frontier(graph, anchor_ids, relation_filter=None, depth=1)[:6]

    return {
        "nodes": ranked_nodes,
        "global_summary_candidates": [
            {
                "region_id": rs.region_id,
                "score": round(score_region_summary(signal, rs), 4),
                "summary": rs.summary_text[:200],
                "node_count": rs.node_count,
                "keywords": rs.keywords[:10],
                "children": rs.child_region_ids,
            }
            for rs in global_regions[:8]
        ],
        "selected_region_path": [rs.region_id for rs in top_regions],
        "candidate_paths": ranked_paths,
        "candidate_edit_zones": candidate_edit_zones,
        "frontier": frontier,
        "dc_trace": dc_trace,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Graph D&C context for a signal — test Option B routing."
    )
    p.add_argument("--graph", required=True, help="Path to graph JSON")
    p.add_argument("--signal", required=True, help="Signal text to process")
    p.add_argument("--beam-size", type=int, default=4)
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--region-k", type=int, default=4)
    p.add_argument("--out-json", default="", help="Optional path to write full packet JSON")
    args = p.parse_args()

    graph = MemoryGraph.load_json(args.graph)
    print(f"Graph: {args.graph}  nodes={len(graph.nodes)}  edges={len(graph.edges)}")
    print(f"Signal: {args.signal}\n")

    result = graph_dc_context_for_signal(
        graph,
        args.signal,
        top_k=args.top_k,
        region_k=args.region_k,
        max_depth=args.max_depth,
        beam_size=args.beam_size,
    )

    SEP = "-" * 66
    print(SEP)
    print("Band 1: global region scores (HCA-like)")
    print(SEP)
    for r in result["global_summary_candidates"][:6]:
        print(f"  {r['region_id']:<52} score={r['score']:.3f}  nodes={r['node_count']}")
        kws = ", ".join(r["keywords"][:6])
        print(f"    keywords: {kws}")

    print(f"\n{SEP}")
    print("D&C selected region path")
    print(SEP)
    print(f"  {result['selected_region_path']}")
    for trace in result["dc_trace"]:
        print(f"  {trace['region_id']}: depth_reached={trace['depth_reached']}  "
              f"children={trace['selected_children']}  nodes={trace['node_count']}")

    print(f"\n{SEP}")
    print("Top nodes (exact evidence, Band 3)")
    print(SEP)
    for n in result["nodes"][:6]:
        print(f"  [{n['node_type']:<12}] {n['id']:<50} score={n.get('score',0):.3f}")
        print(f"    {str(n.get('text',''))[:90]}")

    print(f"\n{SEP}")
    print("Candidate edit zones")
    print(SEP)
    print(f"  {result['candidate_edit_zones']}")

    if result["candidate_paths"]:
        print(f"\n{SEP}")
        print("Top paths")
        print(SEP)
        for path in result["candidate_paths"][:3]:
            nodes_str = " -> ".join(path.get("path_nodes", []))
            print(f"  conf={path.get('confidence',0):.3f}  {nodes_str}")
            print(f"    relation: {path.get('relation_composition','?')}  "
                  f"conflict={path.get('is_conflict_path', False)}")

    if result["frontier"]:
        print(f"\n{SEP}")
        print("Frontier (1-hop from anchors)")
        print(SEP)
        for f in result["frontier"][:4]:
            print(f"  {f['id']:<50} score={f.get('score',0):.3f}  via={f.get('via_relation','?')}")

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nFull packet written to {args.out_json}")
