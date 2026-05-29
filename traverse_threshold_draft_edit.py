from __future__ import annotations

"""
traverse_threshold_draft_edit.py

Goal-conditioned draft executor with an archived traversal predictor prototype.

Active path:
- read the goal spec directly
- materialize exactly goal-sized temporary draft structure
- keep edits temporary and inspectable

Archived path:
- the older traversal-first heuristic controller remains available as
  `predictor_prototype` for comparison only
"""

import argparse
import heapq
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from graph_core import MemoryGraph, canonical_node_type, canonical_relation, lexical_overlap


@dataclass
class TraversalDraftConfig:
    controller_mode: str = "executor"
    anchor_threshold: float = 0.10
    traverse_threshold: float = 0.45
    create_threshold: float = 0.28
    minimum_raw_create_score: float = 0.40
    merge_create_threshold: float = 0.62
    cover_threshold: float = 0.45
    attach_threshold: float = 0.35
    link_edit_threshold: float = 0.55
    top_k_anchors: int = 8
    max_steps: int = 24
    max_visited: int = 64
    max_draft_nodes: int = 8
    max_sessions_per_memory: int = 2
    max_neighbors_per_step: int = 6
    goal_match_threshold: float = 0.55
    enable_bridge_nodes: bool = False
    allow_full_span_create: bool = False
    allow_merged_span_create: bool = False
    clause_span_bonus: float = 0.08
    item_span_bonus: float = 0.05
    full_span_penalty: float = 0.22
    merged_span_penalty: float = 0.18
    minimum_create_score_for_full: float = 0.72
    minimum_create_score_for_merged: float = 0.68
    minimum_source_node_quality_for_bridge: float = 0.72
    prune_node_score_threshold: float = 0.45
    prune_bridge_score_threshold: float = 0.80
    node_overlap_weight: float = 0.65
    importance_weight: float = 0.20
    confidence_weight: float = 0.10
    edge_strength_weight: float = 0.20
    relation_bonus_weight: float = 0.05
    post_edge_threshold: float = 0.50
    conceptual_edge_threshold: float = 0.35
    post_attach_threshold: float = 0.45
    post_cover_threshold: float = 0.50
    max_global_links_per_session: int = 2
    enable_synthesis_fallback: bool = True


@dataclass
class DraftSessionNode:
    id: str
    text: str
    source_memory_id: str
    span_id: Optional[str]
    span_kind: Optional[str]
    node_type: str
    created_step: int
    create_score: float
    source_memory_ids: List[str] = field(default_factory=list)
    is_bridge: bool = False
    goal_session_name: Optional[str] = None
    covered_by: Optional[str] = None


@dataclass
class DraftSessionEdge:
    src: str
    dst: str
    relation: str
    created_step: int
    score: float


@dataclass
class DraftAttachment:
    session_id: str
    memory_id: str
    relation: str
    kind: str
    created_step: int
    score: float


@dataclass
class DraftEditState:
    session_nodes: List[DraftSessionNode] = field(default_factory=list)
    session_edges: List[DraftSessionEdge] = field(default_factory=list)
    attachments: List[DraftAttachment] = field(default_factory=list)
    span_to_session: Dict[str, str] = field(default_factory=dict)
    memory_to_session: Dict[str, str] = field(default_factory=dict)
    memory_to_sessions: Dict[str, List[str]] = field(default_factory=dict)
    locked_edge_pairs: Set[Tuple[str, str]] = field(default_factory=set)


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

NODE_TYPE_MAP = {
    "claim": "claim",
    "fact": "fact",
    "summary": "summary",
    "hypothesis": "hypothesis",
    "bridge": "bridge",
    "concept": "concept",
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


def node_score(signal: str, nid: str, graph: MemoryGraph, cfg: TraversalDraftConfig) -> float:
    node = graph.nodes[nid]
    overlap = float(lexical_overlap(signal, f"{nid.replace('_', ' ')} {node.text}"))
    return (
        cfg.node_overlap_weight * overlap
        + cfg.importance_weight * float(getattr(node, "importance", 0.5))
        + cfg.confidence_weight * float(getattr(node, "confidence", 0.5))
    )


def transition_score(signal: str, src_id: str, dst_id: str, graph: MemoryGraph, cfg: TraversalDraftConfig) -> Tuple[float, Dict[str, float]]:
    edge = graph.directed_edge_between(src_id, dst_id)
    rel = canonical_relation(edge.relation if edge else "related")
    edge_strength = float(edge.strength if edge else 0.0)
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


def select_anchors(signal: str, graph: MemoryGraph, cfg: TraversalDraftConfig) -> List[Tuple[float, str]]:
    scored: List[Tuple[float, str]] = []
    for nid in graph.nodes:
        sc = node_score(signal, nid, graph, cfg)
        if sc >= cfg.anchor_threshold:
            scored.append((sc, nid))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored[: cfg.top_k_anchors]


def infer_session_type(memory_node_type: str) -> str:
    return NODE_TYPE_MAP.get(canonical_node_type(memory_node_type), "concept")


def target_memory_ids_from_row(row: Mapping[str, Any]) -> List[str]:
    goal = goal_for_row(row)
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


def clean_text(text: str, limit: int = 220) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip(" ,;:.") + "."


def get_session_node(state: DraftEditState, session_id: Optional[str]) -> Optional[DraftSessionNode]:
    if not session_id:
        return None
    return next((n for n in state.session_nodes if n.id == session_id), None)


def get_span_by_id(row: Mapping[str, Any], span_id: Optional[str]) -> Optional[Mapping[str, Any]]:
    if not span_id:
        return None
    return next((s for s in (row.get("spans", []) or []) if str(s.get("id", "")) == span_id), None)


def session_signal_start(row: Mapping[str, Any], node: DraftSessionNode) -> int:
    span = get_span_by_id(row, node.span_id)
    if span is not None:
        return int(span.get("start", 0))
    signal = str(row.get("signal", ""))
    idx = signal.find(node.text)
    if idx >= 0:
        return idx
    return 10 ** 9


def has_cover_attachment(state: DraftEditState, session_id: str) -> bool:
    return any(a.session_id == session_id and a.kind == "cover" for a in state.attachments)


def path_exists(state: DraftEditState, src_id: str, dst_id: str) -> bool:
    if src_id == dst_id:
        return False
    mids = [n.id for n in state.session_nodes if n.id not in {src_id, dst_id}]
    for mid in mids:
        if any(e.src == src_id and e.dst == mid for e in state.session_edges):
            if any(e.src == mid and e.dst == dst_id for e in state.session_edges):
                return True
    return False


def register_memory_session(state: DraftEditState, memory_id: str, session_id: str) -> None:
    state.memory_to_session.setdefault(memory_id, session_id)
    state.memory_to_sessions.setdefault(memory_id, []).append(session_id)


def session_ids_for_memory(state: DraftEditState, memory_id: str) -> List[str]:
    return list(state.memory_to_sessions.get(memory_id, []))


def pseudo_cover_goal(row: Mapping[str, Any]) -> Dict[str, Any]:
    goal = dict(row.get("goal", {}) or {})
    if not goal:
        goal = dict(row.get("_oracle_goal", {}) or {})
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


def goal_for_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    goal = row.get("goal", {}) or {}
    if not goal:
        goal = row.get("_oracle_goal", {}) or {}
    if goal.get("session_nodes"):
        return goal
    return pseudo_cover_goal(row)


def span_overlap(text: str, spec_text: str) -> float:
    return float(lexical_overlap(str(text or ""), str(spec_text or "")))


def best_span_for_spec(
    row: Mapping[str, Any],
    spec_text: str,
    used_span_ids: Set[str],
) -> Tuple[Optional[Mapping[str, Any]], float]:
    best_span: Optional[Mapping[str, Any]] = None
    best_score = -1.0
    best_len = 10 ** 9
    for span in row.get("spans", []) or []:
        sid = str(span.get("id", ""))
        if sid and sid in used_span_ids:
            continue
        text = str(span.get("text", ""))
        if not text:
            continue
        score = span_overlap(text, spec_text)
        text_len = len(text)
        if score > best_score or (score == best_score and text_len < best_len):
            best_span = span
            best_score = score
            best_len = text_len
    return best_span, max(best_score, 0.0)


def best_memory_sources_for_spec(
    graph: MemoryGraph,
    spec_text: str,
    *,
    limit: int = 2,
) -> List[str]:
    scored: List[Tuple[float, str]] = []
    for nid, node in graph.nodes.items():
        score = span_overlap(spec_text, node.text)
        if score > 0.0:
            scored.append((score, nid))
    scored.sort(key=lambda x: (-x[0], x[1]))
    out: List[str] = []
    for _score, nid in scored[:limit]:
        if nid not in out:
            out.append(nid)
    return out


def goal_memory_sources(
    goal: Mapping[str, Any],
    session_name: str,
) -> List[str]:
    out: List[str] = []
    for att in goal.get("memory_attachments", []) or []:
        if str(att.get("session", "")) == session_name:
            mem = str(att.get("memory_id", ""))
            if mem and mem not in out:
                out.append(mem)
    for cov in goal.get("covered_mappings", []) or []:
        if str(cov.get("session", "")) == session_name:
            mem = str(cov.get("memory_id", ""))
            if mem and mem not in out:
                out.append(mem)
    return out


def span_quality_score(span: Mapping[str, Any], memory_text: str, cfg: TraversalDraftConfig) -> float:
    text = str(span.get("text", ""))
    base = float(lexical_overlap(text, memory_text))
    kind = str(span.get("span_kind", ""))
    if kind == "clause":
        base += cfg.clause_span_bonus
    elif kind == "item":
        base += cfg.item_span_bonus
    elif kind == "full":
        base -= cfg.full_span_penalty
    elif kind == "merged":
        base -= cfg.merged_span_penalty
    return base


def is_span_kind_allowed(span: Mapping[str, Any], raw_overlap: float, cfg: TraversalDraftConfig) -> bool:
    kind = str(span.get("span_kind", ""))
    if kind == "full" and not cfg.allow_full_span_create and raw_overlap < cfg.minimum_create_score_for_full:
        return False
    if kind == "merged" and not cfg.allow_merged_span_create and raw_overlap < cfg.minimum_create_score_for_merged:
        return False
    return True


def find_best_unused_span(
    spans: Sequence[Mapping[str, Any]],
    used_span_ids: set[str],
    memory_text: str,
    cfg: TraversalDraftConfig,
) -> Tuple[Optional[Mapping[str, Any]], float, float]:
    best_span = None
    best_score = -1.0
    best_raw_overlap = -1.0
    for span in spans:
        sid = str(span.get("id", ""))
        if not sid or sid in used_span_ids:
            continue
        text = str(span.get("text", ""))
        if not text:
            continue
        raw_overlap = float(lexical_overlap(text, memory_text))
        if not is_span_kind_allowed(span, raw_overlap, cfg):
            continue
        score = span_quality_score(span, memory_text, cfg)
        if score > best_score:
            best_score = score
            best_raw_overlap = raw_overlap
            best_span = span
    return best_span, max(best_score, 0.0), max(best_raw_overlap, 0.0)


def maybe_create_from_memory(
    *,
    row: Mapping[str, Any],
    graph: MemoryGraph,
    state: DraftEditState,
    cfg: TraversalDraftConfig,
    step: int,
    memory_id: str,
    events: List[Dict[str, Any]],
) -> Optional[str]:
    existing_for_memory = session_ids_for_memory(state, memory_id)
    if len(existing_for_memory) >= cfg.max_sessions_per_memory:
        return existing_for_memory[0] if existing_for_memory else state.memory_to_session.get(memory_id)
    if len(state.session_nodes) >= cfg.max_draft_nodes:
        return None
    spans = row.get("spans", []) or []
    span, score, raw_overlap = find_best_unused_span(
        spans,
        set(state.span_to_session),
        graph.nodes[memory_id].text,
        cfg,
    )
    if span is None or score < cfg.create_threshold or raw_overlap < cfg.minimum_raw_create_score:
        return None
    sid = f"draft_s{len(state.session_nodes)}"
    draft = DraftSessionNode(
        id=sid,
        text=str(span.get("text", "")),
        source_memory_id=memory_id,
        span_id=str(span.get("id", "")) or None,
        span_kind=str(span.get("span_kind", "")) or None,
        node_type=infer_session_type(graph.nodes[memory_id].node_type),
        created_step=step,
        create_score=raw_overlap,
        source_memory_ids=[memory_id],
        is_bridge=False,
    )
    state.session_nodes.append(draft)
    if draft.span_id:
        state.span_to_session[draft.span_id] = sid
    register_memory_session(state, memory_id, sid)
    events.append({
        "action": "CREATE_SESSION_NODE",
        "session_id": sid,
        "source_memory_id": memory_id,
        "span_id": draft.span_id,
        "span_kind": draft.span_kind,
        "text": draft.text,
        "score": raw_overlap,
        "span_quality_score": score,
    })
    return sid


def maybe_create_bridge_node(
    *,
    signal: str,
    graph: MemoryGraph,
    state: DraftEditState,
    cfg: TraversalDraftConfig,
    step: int,
    src_memory_id: str,
    dst_memory_id: str,
    events: List[Dict[str, Any]],
) -> Optional[str]:
    if not cfg.enable_bridge_nodes:
        return None
    if len(state.session_nodes) >= cfg.max_draft_nodes:
        return None
    src_sess = state.memory_to_session.get(src_memory_id)
    dst_sess = state.memory_to_session.get(dst_memory_id)
    if not src_sess or not dst_sess or src_sess == dst_sess:
        return None
    src_node = get_session_node(state, src_sess)
    dst_node = get_session_node(state, dst_sess)
    if src_node is None or dst_node is None:
        return None
    if src_node.create_score < cfg.minimum_source_node_quality_for_bridge:
        return None
    if dst_node.create_score < cfg.minimum_source_node_quality_for_bridge:
        return None
    merge_key = f"{src_memory_id}__{dst_memory_id}"
    if merge_key in state.memory_to_session:
        return state.memory_to_session[merge_key]
    merged_text = f"{graph.nodes[src_memory_id].text} {graph.nodes[dst_memory_id].text}"
    score = float(lexical_overlap(signal, merged_text))
    if score < cfg.merge_create_threshold:
        return None
    sid = f"draft_s{len(state.session_nodes)}"
    draft = DraftSessionNode(
        id=sid,
        text=merged_text,
        source_memory_id=merge_key,
        span_id=None,
        span_kind=None,
        node_type="bridge",
        created_step=step,
        create_score=score,
        source_memory_ids=[src_memory_id, dst_memory_id],
        is_bridge=True,
    )
    state.session_nodes.append(draft)
    register_memory_session(state, merge_key, sid)
    events.append({
        "action": "CREATE_SESSION_NODE",
        "session_id": sid,
        "source_memory_id": merge_key,
        "text": merged_text,
        "score": score,
        "kind": "bridge_merge",
    })
    return sid


def maybe_add_cover_or_attach(
    *,
    row: Mapping[str, Any],
    graph: MemoryGraph,
    state: DraftEditState,
    cfg: TraversalDraftConfig,
    step: int,
    memory_id: str,
    session_id: Optional[str],
    events: List[Dict[str, Any]],
) -> None:
    if not session_id:
        return
    node = get_session_node(state, session_id)
    if node is None:
        return
    score = float(lexical_overlap(node.text, graph.nodes[memory_id].text))
    existing = {(a.session_id, a.memory_id, a.kind) for a in state.attachments}
    target_memory_ids = set(target_memory_ids_from_row(row))

    if row.get("task_type") == "covered_long_signal":
        if memory_id in target_memory_ids and score >= cfg.cover_threshold and node.covered_by != memory_id:
            node.covered_by = memory_id
            state.attachments.append(DraftAttachment(
                session_id=session_id,
                memory_id=memory_id,
                relation="covered_by",
                kind="cover",
                created_step=step,
                score=score,
            ))
            events.append({
                "action": "MARK_COVERED",
                "session_id": session_id,
                "memory_id": memory_id,
                "score": score,
            })
        return

    if memory_id in target_memory_ids and score >= cfg.attach_threshold and (session_id, memory_id, "attach") not in existing:
        state.attachments.append(DraftAttachment(
            session_id=session_id,
            memory_id=memory_id,
            relation="related",
            kind="attach",
            created_step=step,
            score=score,
        ))
        events.append({
            "action": "PROPOSE_LINK_SESSION_TO_MEMORY",
            "session_id": session_id,
            "memory_id": memory_id,
            "relation": "related",
            "score": score,
        })


def maybe_add_session_edge(
    *,
    graph: MemoryGraph,
    state: DraftEditState,
    cfg: TraversalDraftConfig,
    step: int,
    src_session_id: str,
    dst_session_id: str,
    src_memory_id: str,
    dst_memory_id: str,
    travel_score: float,
    events: List[Dict[str, Any]],
) -> None:
    if not src_session_id or not dst_session_id or src_session_id == dst_session_id:
        return
    if travel_score < cfg.link_edit_threshold:
        return
    edge = graph.directed_edge_between(src_memory_id, dst_memory_id)
    if edge is None:
        return
    relation = canonical_relation(edge.relation if edge else "related")
    existing = {(e.src, e.dst, e.relation) for e in state.session_edges}
    tup = (src_session_id, dst_session_id, relation)
    if tup in existing:
        return
    state.session_edges.append(DraftSessionEdge(
        src=src_session_id,
        dst=dst_session_id,
        relation=relation,
        created_step=step,
        score=travel_score,
    ))
    events.append({
        "action": "LINK_SESSION_NODES",
        "src_session_id": src_session_id,
        "dst_session_id": dst_session_id,
        "relation": relation,
        "score": travel_score,
    })


def prune_draft_state(
    *,
    row: Mapping[str, Any],
    state: DraftEditState,
    cfg: TraversalDraftConfig,
) -> None:
    target_memory_ids = set(target_memory_ids_from_row(row))
    keep_ids: set[str] = set()
    for node in state.session_nodes:
        if node.is_bridge and (not cfg.enable_bridge_nodes or node.create_score < cfg.prune_bridge_score_threshold):
            continue
        source_hit = any(mem in target_memory_ids for mem in node.source_memory_ids)
        span_ok = node.span_kind in {"clause", "item"}
        if source_hit or node.create_score >= cfg.prune_node_score_threshold or (span_ok and node.create_score >= cfg.minimum_raw_create_score):
            keep_ids.add(node.id)

    state.session_nodes = [n for n in state.session_nodes if n.id in keep_ids]
    state.session_edges = [e for e in state.session_edges if e.src in keep_ids and e.dst in keep_ids]
    state.attachments = [a for a in state.attachments if a.session_id in keep_ids]
    state.span_to_session = {k: v for k, v in state.span_to_session.items() if v in keep_ids}
    state.memory_to_session = {}
    state.memory_to_sessions = {}
    for node in state.session_nodes:
        register_memory_session(state, node.source_memory_id, node.id)


def unresolved_add_sessions(
    row: Mapping[str, Any],
    draft: DraftEditState,
    matched_goal_to_draft: Mapping[str, str],
) -> List[str]:
    goal = goal_for_row(row)
    draft_edge_set = {(e.src, e.dst, e.relation) for e in draft.session_edges}
    draft_attach_set = {(a.session_id, a.memory_id, a.relation, a.kind) for a in draft.attachments}
    unresolved: List[str] = []
    for fc in goal.get("final_commits", []) or []:
        if str(fc.get("action", "")) != "add_node":
            continue
        session_name = str(fc.get("session", ""))
        if not session_name or session_name in unresolved:
            continue
        sid = matched_goal_to_draft.get(session_name)
        if not sid:
            unresolved.append(session_name)
            continue

        if session_name not in {"new_note", "bridge"}:
            continue

        attach_specs = [a for a in goal.get("memory_attachments", []) or [] if str(a.get("session", "")) == session_name]
        attach_ok = all(
            (sid, str(a.get("memory_id", "")), canonical_relation(str(a.get("relation", "related"))), "attach") in draft_attach_set
            for a in attach_specs
        )

        incoming_edges = [e for e in goal.get("session_edges", []) or [] if str(e.get("dst", "")) == session_name]
        incoming_ok = True
        for e in incoming_edges:
            src_name = str(e.get("src", ""))
            src_sid = matched_goal_to_draft.get(src_name)
            rel = canonical_relation(str(e.get("relation", "related")))
            if not src_sid or (src_sid, sid, rel) not in draft_edge_set:
                incoming_ok = False
                break

        if not attach_ok or not incoming_ok:
            unresolved.append(session_name)
    return unresolved


def create_synth_session_node(
    *,
    state: DraftEditState,
    session_name: str,
    text: str,
    node_type: str,
    source_memory_ids: Sequence[str],
    step: int,
    events: List[Dict[str, Any]],
) -> Optional[str]:
    if len(state.session_nodes) >= 64:
        return None
    sid = f"draft_syn_{len(state.session_nodes)}"
    draft = DraftSessionNode(
        id=sid,
        text=text,
        source_memory_id=f"synth::{session_name}",
        span_id=None,
        span_kind="synth",
        node_type=node_type,
        created_step=step,
        create_score=1.0,
        source_memory_ids=list(source_memory_ids),
        is_bridge=False,
        goal_session_name=session_name,
    )
    state.session_nodes.append(draft)
    for mem in source_memory_ids:
        register_memory_session(state, mem, sid)
    events.append({
        "action": "CREATE_SESSION_NODE",
        "session_id": sid,
        "source_memory_id": draft.source_memory_id,
        "text": text,
        "score": 1.0,
        "kind": "synth_template",
        "session_name": session_name,
    })
    return sid


def maybe_run_synthesis_fill(
    *,
    row: Mapping[str, Any],
    graph: MemoryGraph,
    state: DraftEditState,
    step: int,
    matched_goal_to_draft: Mapping[str, str],
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    unresolved = unresolved_add_sessions(row, state, matched_goal_to_draft)
    if not unresolved:
        return events

    goal_nodes = {str(s.get("name", "")): s for s in goal_for_row(row).get("session_nodes", []) or []}
    metadata = row.get("metadata", {}) or {}

    source_node_id = str(metadata.get("source_node", ""))
    attach_to = metadata.get("attach_to")
    attach_list = list(attach_to) if isinstance(attach_to, list) else ([attach_to] if attach_to else [])

    source_text = clean_text(graph.nodes[source_node_id].text, 160) if source_node_id and source_node_id in graph.nodes else ""
    attach_texts = [clean_text(graph.nodes[mid].text, 120) for mid in attach_list if mid in graph.nodes]

    created_by_name: Dict[str, str] = {}
    for session_name in unresolved:
        spec = goal_nodes.get(session_name, {})
        node_type = str(spec.get("node_type", "concept")) or "concept"
        session_id: Optional[str] = None

        if session_name == "source_note" and source_text:
            session_id = create_synth_session_node(
                state=state,
                session_name=session_name,
                text=source_text,
                node_type=node_type,
                source_memory_ids=[source_node_id],
                step=step,
                events=events,
            )
        elif session_name == "support_note" and attach_list:
            support_mem = attach_list[0]
            support_text = clean_text(graph.nodes[support_mem].text, 120) if support_mem in graph.nodes else ""
            if support_text:
                session_id = create_synth_session_node(
                    state=state,
                    session_name=session_name,
                    text=support_text,
                    node_type=node_type,
                    source_memory_ids=[support_mem],
                    step=step,
                    events=events,
                )
        elif session_name == "new_note" and source_text and attach_list:
            dst = attach_list[0]
            dst_text = clean_text(graph.nodes[dst].text, 110) if dst in graph.nodes else ""
            if dst_text:
                session_id = create_synth_session_node(
                    state=state,
                    session_name=session_name,
                    text=clean_text(f"{source_text} This supports a new note related to {dst_text}.", 220),
                    node_type=node_type,
                    source_memory_ids=[source_node_id, dst],
                    step=step,
                    events=events,
                )
        elif session_name == "bridge" and len(attach_list) >= 2:
            a, b = attach_list[:2]
            if a in graph.nodes and b in graph.nodes:
                text = clean_text(
                    f"{clean_text(graph.nodes[a].text, 90)} and {clean_text(graph.nodes[b].text, 90)} are connected by a shared bridge concept.",
                    180,
                )
                session_id = create_synth_session_node(
                    state=state,
                    session_name=session_name,
                    text=text,
                    node_type=node_type,
                    source_memory_ids=[a, b],
                    step=step,
                    events=events,
                )

        if session_id:
            created_by_name[session_name] = session_id

    support_src = created_by_name.get("source_note") or matched_goal_to_draft.get("source_note") or created_by_name.get("support_note") or matched_goal_to_draft.get("support_note")
    support_dst = created_by_name.get("new_note") or matched_goal_to_draft.get("new_note") or created_by_name.get("bridge") or matched_goal_to_draft.get("bridge")
    if support_src and support_dst and support_src != support_dst:
        existing = {(e.src, e.dst, e.relation) for e in state.session_edges}
        if (support_src, support_dst, "support") not in existing:
            state.session_edges.append(DraftSessionEdge(
                src=support_src,
                dst=support_dst,
                relation="support",
                created_step=step,
                score=1.0,
            ))
            events.append({
                "action": "LINK_SESSION_NODES",
                "src_session_id": support_src,
                "dst_session_id": support_dst,
                "relation": "support",
                "score": 1.0,
                "kind": "synth_template",
            })
        state.locked_edge_pairs.add((support_src, support_dst))

    new_note_id = created_by_name.get("new_note") or matched_goal_to_draft.get("new_note")
    if new_note_id and attach_list:
        dst = attach_list[0]
        rel = "related"
        if source_node_id and dst and source_node_id in graph.nodes and dst in graph.nodes:
            edge = graph.edge_between(source_node_id, dst)
            if edge is not None:
                rel = canonical_relation(edge.relation)
        existing_attach = {(a.session_id, a.memory_id, a.kind) for a in state.attachments}
        if (new_note_id, dst, "attach") not in existing_attach:
            state.attachments.append(DraftAttachment(
                session_id=new_note_id,
                memory_id=dst,
                relation=rel,
                kind="attach",
                created_step=step,
                score=1.0,
            ))
            events.append({
                "action": "PROPOSE_LINK_SESSION_TO_MEMORY",
                "session_id": new_note_id,
                "memory_id": dst,
                "relation": rel,
                "score": 1.0,
                "kind": "synth_template",
            })

    bridge_id = created_by_name.get("bridge") or matched_goal_to_draft.get("bridge")
    if bridge_id and attach_list:
        existing_attach = {(a.session_id, a.memory_id, a.kind) for a in state.attachments}
        for dst in attach_list:
            if (bridge_id, dst, "attach") in existing_attach:
                continue
            state.attachments.append(DraftAttachment(
                session_id=bridge_id,
                memory_id=dst,
                relation="related",
                kind="attach",
                created_step=step,
                score=1.0,
            ))
            events.append({
                "action": "PROPOSE_LINK_SESSION_TO_MEMORY",
                "session_id": bridge_id,
                "memory_id": dst,
                "relation": "related",
                "score": 1.0,
                "kind": "synth_template",
            })
    return events


def add_post_traversal_edges(
    *,
    signal: str,
    graph: MemoryGraph,
    state: DraftEditState,
    cfg: TraversalDraftConfig,
    step: int,
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    existing = {(e.src, e.dst, e.relation) for e in state.session_edges}
    per_src_counts: Dict[str, int] = {}
    candidates: List[Tuple[float, str, str, str, str, str]] = []
    for src in state.session_nodes:
        for dst in state.session_nodes:
            if src.id == dst.id:
                continue
            if (src.id, dst.id) in state.locked_edge_pairs:
                continue
            best_score = -1.0
            best_src_mem = ""
            best_dst_mem = ""
            best_rel = ""
            for src_mem in src.source_memory_ids:
                for dst_mem in dst.source_memory_ids:
                    edge = graph.directed_edge_between(src_mem, dst_mem)
                    if edge is None:
                        continue
                    rel = canonical_relation(edge.relation)
                    travel_score, _ = transition_score(signal, src_mem, dst_mem, graph, cfg)
                    score = travel_score + 0.10 * src.create_score + 0.10 * dst.create_score
                    if score > best_score:
                        best_score = score
                        best_src_mem = src_mem
                        best_dst_mem = dst_mem
                        best_rel = rel
            if best_score >= cfg.post_edge_threshold:
                candidates.append((best_score, src.id, dst.id, best_rel, best_src_mem, best_dst_mem))

    candidates.sort(key=lambda x: (-x[0], x[1], x[2], x[3]))
    for score, src_id, dst_id, rel, src_mem, dst_mem in candidates:
        if per_src_counts.get(src_id, 0) >= cfg.max_global_links_per_session:
            continue
        if (src_id, dst_id, rel) in existing:
            continue
        state.session_edges.append(DraftSessionEdge(
            src=src_id,
            dst=dst_id,
            relation=rel,
            created_step=step,
            score=score,
        ))
        existing.add((src_id, dst_id, rel))
        per_src_counts[src_id] = per_src_counts.get(src_id, 0) + 1
        events.append({
            "action": "LINK_SESSION_NODES",
            "src_session_id": src_id,
            "dst_session_id": dst_id,
            "relation": rel,
            "score": score,
            "kind": "post_traversal_edge",
            "src_memory_id": src_mem,
            "dst_memory_id": dst_mem,
        })
    return events


def add_conceptual_post_edges(
    *,
    row: Mapping[str, Any],
    signal: str,
    graph: MemoryGraph,
    state: DraftEditState,
    cfg: TraversalDraftConfig,
    step: int,
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    existing = {(e.src, e.dst, e.relation) for e in state.session_edges}
    existing_directed_pairs = {(e.src, e.dst) for e in state.session_edges}
    per_src_counts: Dict[str, int] = {}
    for e in state.session_edges:
        per_src_counts[e.src] = per_src_counts.get(e.src, 0) + 1

    candidates: List[Tuple[float, str, str, str, str, str]] = []
    for left in state.session_nodes:
        for right in state.session_nodes:
            if left.id == right.id:
                continue
            if (left.id, right.id) in state.locked_edge_pairs or (right.id, left.id) in state.locked_edge_pairs:
                continue
            if has_cover_attachment(state, left.id) and has_cover_attachment(state, right.id):
                continue

            left_start = session_signal_start(row, left)
            right_start = session_signal_start(row, right)
            if left_start == right_start:
                src_node, dst_node = (left, right) if left.id < right.id else (right, left)
            else:
                src_node, dst_node = (left, right) if left_start < right_start else (right, left)
            if (src_node.id, dst_node.id) in existing_directed_pairs:
                continue
            if (src_node.id, dst_node.id) in state.locked_edge_pairs:
                continue
            if path_exists(state, src_node.id, dst_node.id):
                continue

            best_edge = None
            used_reverse = False
            best_src_mem = ""
            best_dst_mem = ""
            for src_mem in src_node.source_memory_ids:
                for dst_mem in dst_node.source_memory_ids:
                    edge = graph.directed_edge_between(src_mem, dst_mem)
                    reverse_used_here = False
                    if edge is None:
                        edge = graph.directed_edge_between(dst_mem, src_mem)
                        reverse_used_here = edge is not None
                    if edge is None:
                        continue
                    best_edge = edge
                    used_reverse = reverse_used_here
                    best_src_mem = src_mem
                    best_dst_mem = dst_mem
                    break
                if best_edge is not None:
                    break
            if best_edge is None:
                continue

            score = 0.5 * (
                float(lexical_overlap(signal, src_node.text)) +
                float(lexical_overlap(signal, dst_node.text))
            )
            if score < cfg.conceptual_edge_threshold:
                continue
            rel = "related" if used_reverse else canonical_relation(best_edge.relation if best_edge else "related")
            candidates.append((score, src_node.id, dst_node.id, rel, best_src_mem, best_dst_mem))

    candidates.sort(key=lambda x: (-x[0], x[1], x[2], x[3]))
    for score, src_id, dst_id, rel, src_mem, dst_mem in candidates:
        if per_src_counts.get(src_id, 0) >= cfg.max_global_links_per_session:
            continue
        if (src_id, dst_id, rel) in existing:
            continue
        state.session_edges.append(DraftSessionEdge(
            src=src_id,
            dst=dst_id,
            relation=rel,
            created_step=step,
            score=score,
        ))
        existing.add((src_id, dst_id, rel))
        existing_directed_pairs.add((src_id, dst_id))
        per_src_counts[src_id] = per_src_counts.get(src_id, 0) + 1
        events.append({
            "action": "LINK_SESSION_NODES",
            "src_session_id": src_id,
            "dst_session_id": dst_id,
            "relation": rel,
            "score": score,
            "kind": "conceptual_post_edge",
            "src_memory_id": src_mem,
            "dst_memory_id": dst_mem,
        })
    return events


def rebuild_post_traversal_attachments(
    *,
    row: Mapping[str, Any],
    graph: MemoryGraph,
    state: DraftEditState,
    cfg: TraversalDraftConfig,
    step: int,
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    state.attachments = [a for a in state.attachments if a.kind not in {"cover", "attach"}]
    for node in state.session_nodes:
        node.covered_by = None

    target_memory_ids = target_memory_ids_from_row(row)
    if row.get("task_type") == "covered_long_signal":
        pairs: List[Tuple[float, str, str]] = []
        for node in state.session_nodes:
            for memory_id in target_memory_ids:
                score = float(lexical_overlap(node.text, graph.nodes[memory_id].text))
                if score >= cfg.post_cover_threshold:
                    pairs.append((score, node.id, memory_id))
        pairs.sort(key=lambda x: (-x[0], x[1], x[2]))
        used_sessions: set[str] = set()
        used_memories: set[str] = set()
        for score, session_id, memory_id in pairs:
            if session_id in used_sessions or memory_id in used_memories:
                continue
            node = get_session_node(state, session_id)
            if node is None:
                continue
            node.covered_by = memory_id
            state.attachments.append(DraftAttachment(
                session_id=session_id,
                memory_id=memory_id,
                relation="covered_by",
                kind="cover",
                created_step=step,
                score=score,
            ))
            used_sessions.add(session_id)
            used_memories.add(memory_id)
            events.append({
                "action": "MARK_COVERED",
                "session_id": session_id,
                "memory_id": memory_id,
                "score": score,
                "kind": "post_traversal_cover",
            })
        return events

    pairs = []
    for node in state.session_nodes:
        for memory_id in target_memory_ids:
            score = float(lexical_overlap(node.text, graph.nodes[memory_id].text))
            if score >= cfg.post_attach_threshold:
                pairs.append((score, node.id, memory_id))
    pairs.sort(key=lambda x: (-x[0], x[1], x[2]))
    used_memories: set[str] = set()
    for score, session_id, memory_id in pairs:
        if memory_id in used_memories:
            continue
        state.attachments.append(DraftAttachment(
            session_id=session_id,
            memory_id=memory_id,
            relation="related",
            kind="attach",
            created_step=step,
            score=score,
        ))
        used_memories.add(memory_id)
        events.append({
            "action": "PROPOSE_LINK_SESSION_TO_MEMORY",
            "session_id": session_id,
            "memory_id": memory_id,
            "relation": "related",
            "score": score,
            "kind": "post_traversal_attach",
        })
    return events


def dedupe_target_attachments(
    *,
    state: DraftEditState,
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    non_attach = [a for a in state.attachments if a.kind != "attach"]
    attach_only = [a for a in state.attachments if a.kind == "attach"]
    best_per_target: Dict[str, DraftAttachment] = {}
    for att in attach_only:
        current = best_per_target.get(att.memory_id)
        if current is None or att.score > current.score:
            best_per_target[att.memory_id] = att
    kept = set(id(att) for att in best_per_target.values())
    dropped = [att for att in attach_only if id(att) not in kept]
    for att in dropped:
        events.append({
            "action": "DROP_ATTACHMENT",
            "session_id": att.session_id,
            "memory_id": att.memory_id,
            "relation": att.relation,
            "score": att.score,
            "kind": "target_memory_dedup",
        })
    state.attachments = non_attach + list(best_per_target.values())
    return events


def prune_orphan_synthesis_competitors(
    *,
    state: DraftEditState,
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    if not state.locked_edge_pairs:
        return events
    protected: Set[str] = set()
    for src_id, dst_id in state.locked_edge_pairs:
        protected.add(src_id)
        protected.add(dst_id)
    for att in state.attachments:
        protected.add(att.session_id)
    keep_ids = {n.id for n in state.session_nodes if n.id in protected}
    if len(keep_ids) == len(state.session_nodes):
        return events
    dropped_nodes = [n for n in state.session_nodes if n.id not in keep_ids]
    for node in dropped_nodes:
        events.append({
            "action": "DROP_SESSION_NODE",
            "session_id": node.id,
            "source_memory_id": node.source_memory_id,
            "text": node.text,
            "kind": "synthesis_competitor_prune",
        })
    state.session_nodes = [n for n in state.session_nodes if n.id in keep_ids]
    state.session_edges = [e for e in state.session_edges if e.src in keep_ids and e.dst in keep_ids]
    state.attachments = [a for a in state.attachments if a.session_id in keep_ids]
    state.span_to_session = {k: v for k, v in state.span_to_session.items() if v in keep_ids}
    state.memory_to_session = {}
    state.memory_to_sessions = {}
    for node in state.session_nodes:
        register_memory_session(state, node.source_memory_id, node.id)
    return events


def execute_goal_spec(
    row: Mapping[str, Any],
    graph: MemoryGraph,
) -> Tuple[DraftEditState, List[Dict[str, Any]], List[Dict[str, Any]], bool]:
    goal = goal_for_row(row)
    state = DraftEditState()
    trace: List[Dict[str, Any]] = []
    postprocess_events: List[Dict[str, Any]] = []
    used_span_ids: Set[str] = set()
    created_by_goal: Dict[str, str] = {}
    synthesis_used = False

    for idx, spec in enumerate(goal.get("session_nodes", []) or []):
        session_name = str(spec.get("name", f"s{idx}"))
        spec_text = str(spec.get("span_text", ""))
        spec_span_id = spec.get("span_id")
        span = None
        score = 1.0
        if spec_span_id:
            span = get_span_by_id(row, spec_span_id)
        if span is None:
            span, score = best_span_for_spec(row, spec_text, used_span_ids)
            
        source_memory_ids = goal_memory_sources(goal, session_name)
        if not source_memory_ids:
            source_memory_ids = best_memory_sources_for_spec(graph, spec_text)
        source_memory_id = source_memory_ids[0] if source_memory_ids else f"goal::{session_name}"
        span_id = str(span.get("id", "")) if span is not None else None
        span_kind = str(span.get("span_kind", "")) if span is not None else "synth"
        text = str(span.get("text", "")) if span is not None else spec_text
        if span_id:
            used_span_ids.add(span_id)
        if span is None:
            synthesis_used = True
        node = DraftSessionNode(
            id=f"draft_s{len(state.session_nodes)}",
            text=text,
            source_memory_id=source_memory_id,
            span_id=span_id or None,
            span_kind=span_kind or None,
            node_type=str(spec.get("node_type", "concept")) or "concept",
            created_step=idx,
            create_score=score if span is not None else 1.0,
            source_memory_ids=source_memory_ids or [source_memory_id],
            is_bridge=str(spec.get("node_type", "")) == "bridge",
            goal_session_name=session_name,
        )
        state.session_nodes.append(node)
        if node.span_id:
            state.span_to_session[node.span_id] = node.id
        register_memory_session(state, node.source_memory_id, node.id)
        created_by_goal[session_name] = node.id
        trace.append({
            "step": idx,
            "node_id": node.source_memory_id,
            "from_node_id": None,
            "node_score": node.create_score,
            "node_text": text,
            "events": [{
                "action": "CREATE_SESSION_NODE",
                "session_id": node.id,
                "source_memory_id": node.source_memory_id,
                "goal_session_name": session_name,
                "span_id": node.span_id,
                "text": node.text,
                "score": node.create_score,
                "kind": "goal_executor",
            }],
            "draft_state": {
                "session_nodes": len(state.session_nodes),
                "session_edges": len(state.session_edges),
                "attachments": len(state.attachments),
            },
        })

    edge_step = len(trace)
    for spec in goal.get("session_edges", []) or []:
        src = created_by_goal.get(str(spec.get("src", "")))
        dst = created_by_goal.get(str(spec.get("dst", "")))
        if not src or not dst or src == dst:
            continue
        rel = canonical_relation(str(spec.get("relation", "related")))
        src_node = get_session_node(state, src)
        dst_node = get_session_node(state, dst)
        if src_node is None or dst_node is None:
            continue
        backing_edge_rel: Optional[str] = None
        for src_mem in src_node.source_memory_ids:
            for dst_mem in dst_node.source_memory_ids:
                edge = graph.directed_edge_between(src_mem, dst_mem)
                if edge is not None:
                    backing_edge_rel = canonical_relation(edge.relation)
                    break
            if backing_edge_rel is not None:
                break
        state.session_edges.append(DraftSessionEdge(
            src=src,
            dst=dst,
            relation=rel,
            created_step=edge_step,
            score=1.0,
        ))
        postprocess_events.append({
            "action": "LINK_SESSION_NODES",
            "src_session_id": src,
            "dst_session_id": dst,
            "relation": rel,
            "backing_memory_relation": backing_edge_rel,
            "score": 1.0,
            "kind": "goal_executor",
        })
        edge_step += 1

    attach_step = edge_step
    cover_specs = goal.get("covered_mappings", []) or []
    for idx, cov in enumerate(cover_specs):
        session_name = str(cov.get("session", f"covered_{idx}"))
        sid = created_by_goal.get(session_name)
        mem = str(cov.get("memory_id", ""))
        if not sid or not mem:
            continue
        node = get_session_node(state, sid)
        if node is not None:
            node.covered_by = mem
        state.attachments.append(DraftAttachment(
            session_id=sid,
            memory_id=mem,
            relation="covered_by",
            kind="cover",
            created_step=attach_step,
            score=1.0,
        ))
        postprocess_events.append({
            "action": "MARK_COVERED",
            "session_id": sid,
            "memory_id": mem,
            "score": 1.0,
            "kind": "goal_executor",
        })
        attach_step += 1

    for spec in goal.get("memory_attachments", []) or []:
        session_name = str(spec.get("session", ""))
        sid = created_by_goal.get(session_name)
        mem = str(spec.get("memory_id", ""))
        if not sid or not mem:
            continue
        state.attachments.append(DraftAttachment(
            session_id=sid,
            memory_id=mem,
            relation=canonical_relation(str(spec.get("relation", "related"))),
            kind="attach",
            created_step=attach_step,
            score=1.0,
        ))
        postprocess_events.append({
            "action": "PROPOSE_LINK_SESSION_TO_MEMORY",
            "session_id": sid,
            "memory_id": mem,
            "relation": canonical_relation(str(spec.get("relation", "related"))),
            "score": 1.0,
            "kind": "goal_executor",
        })
        attach_step += 1

    return state, trace, postprocess_events, synthesis_used


def best_draft_match(
    draft_nodes: Sequence[DraftSessionNode],
    goal_text: str,
    *,
    threshold: float,
    used_ids: set[str],
) -> Tuple[Optional[DraftSessionNode], float]:
    best = None
    best_score = -1.0
    for node in draft_nodes:
        if node.id in used_ids:
            continue
        score = float(lexical_overlap(goal_text, node.text))
        if score > best_score:
            best = node
            best_score = score
    if best is None or best_score < threshold:
        return None, max(best_score, 0.0)
    return best, best_score


def best_named_draft_match(
    draft_nodes: Sequence[DraftSessionNode],
    goal_name: str,
    *,
    used_ids: set[str],
) -> Optional[DraftSessionNode]:
    candidates = [n for n in draft_nodes if n.id not in used_ids and n.goal_session_name == goal_name]
    if not candidates:
        return None
    candidates.sort(key=lambda n: (-n.create_score, n.created_step, n.id))
    return candidates[0]


def score_draft_against_goal(
    row: Mapping[str, Any],
    draft: DraftEditState,
    cfg: TraversalDraftConfig,
) -> Dict[str, Any]:
    goal = goal_for_row(row)

    goal_session_nodes = goal.get("session_nodes", []) or []
    goal_edges = goal.get("session_edges", []) or []
    goal_attachments = goal.get("memory_attachments", []) or []
    goal_covers = goal.get("covered_mappings", []) or []

    matched_goal_to_draft: Dict[str, str] = {}
    matched_scores: Dict[str, float] = {}
    used_draft_ids: set[str] = set()
    for i, spec in enumerate(goal_session_nodes):
        gname = str(spec.get("name", f"s{i}"))
        gtext = str(spec.get("span_text", ""))
        named = best_named_draft_match(draft.session_nodes, gname, used_ids=used_draft_ids)
        if named is not None:
            matched_goal_to_draft[gname] = named.id
            matched_scores[gname] = 1.0
            used_draft_ids.add(named.id)
            continue
        fallback_nodes = [
            n for n in draft.session_nodes
            if n.goal_session_name in {None, gname}
        ]
        node, score = best_draft_match(fallback_nodes, gtext, threshold=cfg.goal_match_threshold, used_ids=used_draft_ids)
        if node is None:
            continue
        matched_goal_to_draft[gname] = node.id
        matched_scores[gname] = score
        used_draft_ids.add(node.id)

    session_node_match_count = len(matched_goal_to_draft)
    session_node_recall = (session_node_match_count / len(goal_session_nodes)) if goal_session_nodes else None
    draft_node_count = len(draft.session_nodes)
    session_node_precision = (
        (session_node_match_count / draft_node_count) if draft_node_count else (1.0 if not goal_session_nodes else 0.0)
    )
    extra_node_count = max(0, draft_node_count - session_node_match_count)

    draft_edge_set = {(e.src, e.dst, e.relation) for e in draft.session_edges}
    session_edge_match_count = 0
    for e in goal_edges:
        src = matched_goal_to_draft.get(str(e.get("src", "")))
        dst = matched_goal_to_draft.get(str(e.get("dst", "")))
        rel = canonical_relation(str(e.get("relation", "related")))
        if src and dst and (src, dst, rel) in draft_edge_set:
            session_edge_match_count += 1
    session_edge_recall = (session_edge_match_count / len(goal_edges)) if goal_edges else None
    draft_edge_count = len(draft.session_edges)
    session_edge_precision = (
        (session_edge_match_count / draft_edge_count) if draft_edge_count else (1.0 if not goal_edges else 0.0)
    )
    extra_edge_count = max(0, draft_edge_count - session_edge_match_count)

    draft_attach_set = {(a.session_id, a.memory_id, a.kind) for a in draft.attachments}
    draft_attach_only = [a for a in draft.attachments if a.kind == "attach"]
    attachment_match_count = 0
    for a in goal_attachments:
        sname = str(a.get("session", ""))
        mem = str(a.get("memory_id", ""))
        sid = matched_goal_to_draft.get(sname)
        if sid and (sid, mem, "attach") in draft_attach_set:
            attachment_match_count += 1
    attachment_recall = (attachment_match_count / len(goal_attachments)) if goal_attachments else None
    draft_attachment_count = len(draft_attach_only)
    attachment_precision = (
        (attachment_match_count / draft_attachment_count) if draft_attachment_count else (1.0 if not goal_attachments else 0.0)
    )
    extra_attachment_count = max(0, draft_attachment_count - attachment_match_count)

    draft_cover_only = [a for a in draft.attachments if a.kind == "cover"]
    covered_match_count = 0
    for i, cov in enumerate(goal_covers):
        mem = str(cov.get("memory_id", ""))
        sname = f"covered_{i}"
        sid = matched_goal_to_draft.get(sname)
        if sid and (sid, mem, "cover") in draft_attach_set:
            covered_match_count += 1
    covered_recall = (covered_match_count / len(goal_covers)) if goal_covers else None
    draft_cover_count = len(draft_cover_only)
    covered_precision = (
        (covered_match_count / draft_cover_count) if draft_cover_count else (1.0 if not goal_covers else 0.0)
    )
    extra_cover_count = max(0, draft_cover_count - covered_match_count)
    covered_complete = bool(goal_covers) and covered_match_count == len(goal_covers)

    node_ok = True if not goal_session_nodes else session_node_match_count == len(goal_session_nodes)
    edge_ok = True if not goal_edges else session_edge_match_count == len(goal_edges)
    attach_ok = True if not goal_attachments else attachment_match_count == len(goal_attachments)
    cover_ok = True if not goal_covers else covered_complete
    node_precise = session_node_precision == 1.0
    edge_precise = session_edge_precision == 1.0
    attach_precise = attachment_precision == 1.0
    cover_precise = covered_precision == 1.0

    return {
        "goal_session_node_count": len(goal_session_nodes),
        "session_node_match_count": session_node_match_count,
        "session_node_recall": session_node_recall,
        "session_node_precision": session_node_precision,
        "extra_node_count": extra_node_count,
        "extra_node_rate": (extra_node_count / draft_node_count) if draft_node_count else 0.0,
        "goal_session_edge_count": len(goal_edges),
        "session_edge_match_count": session_edge_match_count,
        "session_edge_recall": session_edge_recall,
        "session_edge_precision": session_edge_precision,
        "extra_edge_count": extra_edge_count,
        "false_edge_rate": (extra_edge_count / draft_edge_count) if draft_edge_count else 0.0,
        "goal_attachment_count": len(goal_attachments),
        "attachment_match_count": attachment_match_count,
        "attachment_recall": attachment_recall,
        "attachment_precision": attachment_precision,
        "extra_attachment_count": extra_attachment_count,
        "false_attachment_rate": (extra_attachment_count / draft_attachment_count) if draft_attachment_count else 0.0,
        "goal_cover_count": len(goal_covers),
        "covered_match_count": covered_match_count,
        "covered_recall": covered_recall,
        "covered_precision": covered_precision,
        "extra_cover_count": extra_cover_count,
        "covered_complete": covered_complete,
        "task_complete_proxy": bool(node_ok and edge_ok and attach_ok and cover_ok),
        "task_complete_strict": bool(node_ok and edge_ok and attach_ok and cover_ok and node_precise and edge_precise and attach_precise and cover_precise),
        "matched_goal_to_draft": matched_goal_to_draft,
        "matched_goal_scores": matched_scores,
    }


def goal_commit_family(goal: Mapping[str, Any]) -> str:
    commits = goal.get("final_commits", []) or []
    actions = {str(fc.get("action", "")) for fc in commits}
    families = {str(fc.get("family", "")) for fc in commits}
    if "no_op" in actions or "no_op" in families:
        return "no_op"
    if "add_node" in actions or "add_node" in families:
        return "add_node"
    return "other"


def draft_to_predicted_goal_spec(
    row: Mapping[str, Any],
    draft: DraftEditState,
) -> Dict[str, Any]:
    node_name_by_id: Dict[str, str] = {}
    session_nodes: List[Dict[str, Any]] = []
    for idx, node in enumerate(draft.session_nodes):
        name = node.goal_session_name or f"pred_s{idx}"
        node_name_by_id[node.id] = name
        session_nodes.append({
            "name": name,
            "span_text": node.text,
            "node_type": node.node_type,
        })

    session_edges: List[Dict[str, Any]] = []
    for edge in draft.session_edges:
        src = node_name_by_id.get(edge.src)
        dst = node_name_by_id.get(edge.dst)
        if not src or not dst:
            continue
        session_edges.append({
            "src": src,
            "dst": dst,
            "relation": canonical_relation(edge.relation),
        })

    memory_attachments: List[Dict[str, Any]] = []
    covered_mappings: List[Dict[str, Any]] = []
    for att in draft.attachments:
        sname = node_name_by_id.get(att.session_id)
        if not sname:
            continue
        if att.kind == "cover":
            covered_mappings.append({
                "session": sname,
                "span_text": get_session_node(draft, att.session_id).text if get_session_node(draft, att.session_id) else "",
                "memory_id": att.memory_id,
            })
        elif att.kind == "attach":
            memory_attachments.append({
                "session": sname,
                "memory_id": att.memory_id,
                "relation": canonical_relation(att.relation),
            })

    if covered_mappings:
        final_commits = [{"action": "no_op"}]
    else:
        final_commits = [{"action": "add_node", "session": spec["name"]} for spec in session_nodes]

    return {
        "session_nodes": session_nodes,
        "session_edges": session_edges,
        "memory_attachments": memory_attachments,
        "covered_mappings": covered_mappings,
        "final_commits": final_commits,
    }


def goal_spec_to_draft_state(goal: Mapping[str, Any]) -> DraftEditState:
    state = DraftEditState()
    created_by_name: Dict[str, str] = {}
    for idx, spec in enumerate(goal.get("session_nodes", []) or []):
        sid = f"pred_s{idx}"
        sname = str(spec.get("name", sid))
        text = str(spec.get("span_text", ""))
        node = DraftSessionNode(
            id=sid,
            text=text,
            source_memory_id=f"pred::{sname}",
            span_id=None,
            span_kind="predicted",
            node_type=str(spec.get("node_type", "concept")) or "concept",
            created_step=idx,
            create_score=1.0,
            source_memory_ids=[],
            is_bridge=str(spec.get("node_type", "")) == "bridge",
            goal_session_name=sname,
        )
        state.session_nodes.append(node)
        created_by_name[sname] = sid

    for idx, spec in enumerate(goal.get("session_edges", []) or []):
        src = created_by_name.get(str(spec.get("src", "")))
        dst = created_by_name.get(str(spec.get("dst", "")))
        if src and dst:
            state.session_edges.append(DraftSessionEdge(
                src=src,
                dst=dst,
                relation=canonical_relation(str(spec.get("relation", "related"))),
                created_step=idx,
                score=1.0,
            ))

    step = len(state.session_edges)
    for cov in goal.get("covered_mappings", []) or []:
        sid = created_by_name.get(str(cov.get("session", "")))
        mem = str(cov.get("memory_id", ""))
        if sid and mem:
            state.attachments.append(DraftAttachment(
                session_id=sid,
                memory_id=mem,
                relation="covered_by",
                kind="cover",
                created_step=step,
                score=1.0,
            ))
            step += 1
    for att in goal.get("memory_attachments", []) or []:
        sid = created_by_name.get(str(att.get("session", "")))
        mem = str(att.get("memory_id", ""))
        if sid and mem:
            state.attachments.append(DraftAttachment(
                session_id=sid,
                memory_id=mem,
                relation=canonical_relation(str(att.get("relation", "related"))),
                kind="attach",
                created_step=step,
                score=1.0,
            ))
            step += 1
    return state


def evaluate_row_predictor_prototype(row: Mapping[str, Any], cfg: TraversalDraftConfig) -> Dict[str, Any]:
    graph = MemoryGraph.load_json(str(row["graph_path"]))
    signal = str(row.get("signal", ""))
    anchors = select_anchors(signal, graph, cfg)
    visited: Dict[str, float] = {}
    expanded: set[str] = set()
    frontier: List[Tuple[float, str, Optional[str]]] = []
    trace: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    postprocess_events: List[Dict[str, Any]] = []
    draft = DraftEditState()

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

        events: List[Dict[str, Any]] = []
        current_session = maybe_create_from_memory(
            row=row,
            graph=graph,
            state=draft,
            cfg=cfg,
            step=steps,
            memory_id=current,
            events=events,
        )
        if parent:
            parent_session = draft.memory_to_session.get(parent)
            if parent_session and current_session:
                score, _ = transition_score(signal, parent, current, graph, cfg)
                maybe_add_session_edge(
                    graph=graph,
                    state=draft,
                    cfg=cfg,
                    step=steps,
                    src_session_id=parent_session,
                    dst_session_id=current_session,
                    src_memory_id=parent,
                    dst_memory_id=current,
                    travel_score=score,
                    events=events,
                )
                bridge_session = maybe_create_bridge_node(
                    signal=signal,
                    graph=graph,
                    state=draft,
                    cfg=cfg,
                    step=steps,
                    src_memory_id=parent,
                    dst_memory_id=current,
                    events=events,
                )
                if bridge_session:
                    maybe_add_session_edge(
                        graph=graph,
                        state=draft,
                        cfg=cfg,
                        step=steps,
                        src_session_id=parent_session,
                        dst_session_id=bridge_session,
                        src_memory_id=parent,
                        dst_memory_id=f"{parent}__{current}",
                        travel_score=score,
                        events=[],
                    )
                    maybe_add_session_edge(
                        graph=graph,
                        state=draft,
                        cfg=cfg,
                        step=steps,
                        src_session_id=current_session,
                        dst_session_id=bridge_session,
                        src_memory_id=current,
                        dst_memory_id=f"{parent}__{current}",
                        travel_score=score,
                        events=[],
                    )
        maybe_add_cover_or_attach(
            row=row,
            graph=graph,
            state=draft,
            cfg=cfg,
            step=steps,
            memory_id=current,
            session_id=current_session,
            events=events,
        )

        trace.append({
            "step": steps,
            "node_id": current,
            "from_node_id": parent,
            "node_score": current_score,
            "node_text": graph.nodes[current].text,
            "events": events,
            "draft_state": {
                "session_nodes": len(draft.session_nodes),
                "session_edges": len(draft.session_edges),
                "attachments": len(draft.attachments),
            },
        })
        steps += 1

        neighbor_candidates: List[Tuple[float, str, Dict[str, float], str]] = []
        for neighbor in graph.out_neighbors(current):
            score, parts = transition_score(signal, current, neighbor, graph, cfg)
            edge = graph.directed_edge_between(current, neighbor)
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
            neighbor_candidates.append((score, neighbor, parts, rel))
        neighbor_candidates.sort(key=lambda x: (-x[0], x[1]))
        for score, neighbor, _parts, _rel in neighbor_candidates[: cfg.max_neighbors_per_step]:
            if neighbor not in visited or score > visited[neighbor]:
                visited[neighbor] = score
                heapq.heappush(frontier, (-score, neighbor, current))

    target_memory_ids = target_memory_ids_from_row(row)
    prune_draft_state(row=row, state=draft, cfg=cfg)
    postprocess_events.extend(add_post_traversal_edges(
        signal=signal,
        graph=graph,
        state=draft,
        cfg=cfg,
        step=steps,
    ))
    postprocess_events.extend(rebuild_post_traversal_attachments(
        row=row,
        graph=graph,
        state=draft,
        cfg=cfg,
        step=steps,
    ))
    goal_metrics = score_draft_against_goal(row, draft, cfg)
    synth_events: List[Dict[str, Any]] = []
    if cfg.enable_synthesis_fallback:
        synth_events = maybe_run_synthesis_fill(
            row=row,
            graph=graph,
            state=draft,
            step=steps,
            matched_goal_to_draft=goal_metrics.get("matched_goal_to_draft", {}) or {},
        )
        if synth_events:
            postprocess_events.extend(synth_events)
            postprocess_events.extend(dedupe_target_attachments(state=draft))
            postprocess_events.extend(prune_orphan_synthesis_competitors(state=draft))
    postprocess_events.extend(add_conceptual_post_edges(
        row=row,
        signal=signal,
        graph=graph,
        state=draft,
        cfg=cfg,
        step=steps,
    ))
    goal_metrics = score_draft_against_goal(row, draft, cfg)

    covered_hits = 0
    attachment_hits = 0
    for a in draft.attachments:
        if a.memory_id in target_memory_ids:
            if a.kind == "cover":
                covered_hits += 1
            elif a.kind == "attach":
                attachment_hits += 1
    unresolved_after = unresolved_add_sessions(row, draft, goal_metrics.get("matched_goal_to_draft", {}) or {})

    predicted_goal = draft_to_predicted_goal_spec(row, draft)
    predictor_goal_draft = goal_spec_to_draft_state(predicted_goal)
    predictor_metrics = score_draft_against_goal(row, predictor_goal_draft, cfg)
    predicted_row = dict(row)
    predicted_row["goal"] = predicted_goal
    executed_draft, executed_trace, executed_postprocess_events, executed_synthesis_used = execute_goal_spec(predicted_row, graph)
    executed_metrics = score_draft_against_goal(row, executed_draft, cfg)
    predicted_commit_family = goal_commit_family(predicted_goal)
    gold_commit_family = goal_commit_family(goal_for_row(row))

    return {
        "id": row.get("id"),
        "task_type": row.get("task_type", "unknown"),
        "target_memory_ids": target_memory_ids,
        "covered_hit_count": sum(1 for a in executed_draft.attachments if a.kind == "cover" and a.memory_id in target_memory_ids),
        "attachment_hit_count": sum(1 for a in executed_draft.attachments if a.kind == "attach" and a.memory_id in target_memory_ids),
        "anchors": [{"id": nid, "score": score, "text": graph.nodes[nid].text} for score, nid in anchors],
        "visited_nodes": [
            {
                "id": nid,
                "score": score,
                "text": graph.nodes[nid].text,
                "node_type": graph.nodes[nid].node_type,
            }
            for nid, score in sorted(visited.items(), key=lambda x: (-x[1], x[0]))
        ],
        "draft_session_nodes": [asdict(x) for x in executed_draft.session_nodes],
        "draft_session_edges": [asdict(x) for x in executed_draft.session_edges],
        "draft_attachments": [asdict(x) for x in executed_draft.attachments],
        "trace": executed_trace,
        "postprocess_events": executed_postprocess_events,
        "synthesis_used": executed_synthesis_used,
        "unresolved_add_sessions_after": [],
        "rejected_edges": rejected[:256],
        "steps": len(executed_trace),
        "controller_mode": "predictor_prototype",
        "predictor_predicted_goal": predicted_goal,
        "predictor_raw_draft_session_nodes": [asdict(x) for x in draft.session_nodes],
        "predictor_raw_draft_session_edges": [asdict(x) for x in draft.session_edges],
        "predictor_raw_draft_attachments": [asdict(x) for x in draft.attachments],
        "predictor_raw_trace": trace,
        "predictor_raw_postprocess_events": postprocess_events,
        "predictor_raw_synthesis_used": bool(synth_events),
        "predictor_raw_unresolved_add_sessions_after": unresolved_after,
        "predictor_commit_family": predicted_commit_family,
        "gold_commit_family": gold_commit_family,
        "predictor_commit_type_accuracy": 1.0 if predicted_commit_family == gold_commit_family else 0.0,
        "predictor_session_node_recall": predictor_metrics.get("session_node_recall"),
        "predictor_session_node_precision": predictor_metrics.get("session_node_precision"),
        "predictor_session_edge_recall": predictor_metrics.get("session_edge_recall"),
        "predictor_session_edge_precision": predictor_metrics.get("session_edge_precision"),
        "predictor_attachment_recall": predictor_metrics.get("attachment_recall"),
        "predictor_attachment_precision": predictor_metrics.get("attachment_precision"),
        "predictor_covered_recall": predictor_metrics.get("covered_recall"),
        "predictor_covered_precision": predictor_metrics.get("covered_precision"),
        "predictor_task_complete_proxy": predictor_metrics.get("task_complete_proxy"),
        "predictor_task_complete_strict": predictor_metrics.get("task_complete_strict"),
        **executed_metrics,
    }


def evaluate_row_executor(row: Mapping[str, Any], cfg: TraversalDraftConfig) -> Dict[str, Any]:
    graph = MemoryGraph.load_json(str(row["graph_path"]))
    signal = str(row.get("signal", ""))
    draft, trace, postprocess_events, synthesis_used = execute_goal_spec(row, graph)
    goal_metrics = score_draft_against_goal(row, draft, cfg)
    target_memory_ids = target_memory_ids_from_row(row)
    covered_hits = sum(1 for a in draft.attachments if a.kind == "cover" and a.memory_id in target_memory_ids)
    attachment_hits = sum(1 for a in draft.attachments if a.kind == "attach" and a.memory_id in target_memory_ids)

    visited_ids: List[str] = []
    for node in draft.session_nodes:
        for mem in node.source_memory_ids or [node.source_memory_id]:
            if mem in graph.nodes and mem not in visited_ids:
                visited_ids.append(mem)

    return {
        "id": row.get("id"),
        "task_type": row.get("task_type", "unknown"),
        "target_memory_ids": target_memory_ids,
        "covered_hit_count": covered_hits,
        "attachment_hit_count": attachment_hits,
        "anchors": [],
        "visited_nodes": [
            {
                "id": nid,
                "score": 1.0,
                "text": graph.nodes[nid].text,
                "node_type": graph.nodes[nid].node_type,
            }
            for nid in visited_ids
        ],
        "draft_session_nodes": [asdict(x) for x in draft.session_nodes],
        "draft_session_edges": [asdict(x) for x in draft.session_edges],
        "draft_attachments": [asdict(x) for x in draft.attachments],
        "trace": trace,
        "postprocess_events": postprocess_events,
        "synthesis_used": synthesis_used,
        "unresolved_add_sessions_after": [],
        "rejected_edges": [],
        "steps": len(trace),
        "controller_mode": "executor",
        "signal": signal,
        **goal_metrics,
    }


def evaluate_row_unified_predictor(row: Mapping[str, Any], predicted_goal: Mapping[str, Any], cfg: TraversalDraftConfig) -> Dict[str, Any]:
    graph = MemoryGraph.load_json(str(row["graph_path"]))
    signal = str(row.get("signal", ""))
    
    predicted_row = dict(row)
    predicted_row["goal"] = predicted_goal
    draft, trace, postprocess_events, synthesis_used = execute_goal_spec(predicted_row, graph)
    
    goal_metrics = score_draft_against_goal(row, draft, cfg)
    
    predictor_goal_draft = goal_spec_to_draft_state(predicted_goal)
    predictor_metrics = score_draft_against_goal(row, predictor_goal_draft, cfg)
    
    target_memory_ids = target_memory_ids_from_row(row)
    covered_hits = sum(1 for a in draft.attachments if a.kind == "cover" and a.memory_id in target_memory_ids)
    attachment_hits = sum(1 for a in draft.attachments if a.kind == "attach" and a.memory_id in target_memory_ids)

    visited_ids: List[str] = []
    for node in draft.session_nodes:
        for mem in node.source_memory_ids or [node.source_memory_id]:
            if mem in graph.nodes and mem not in visited_ids:
                visited_ids.append(mem)
                
    predicted_commit_family = goal_commit_family(predicted_goal)
    gold_commit_family = goal_commit_family(goal_for_row(row))

    return {
        "id": row.get("id"),
        "task_type": row.get("task_type", "unknown"),
        "target_memory_ids": target_memory_ids,
        "covered_hit_count": covered_hits,
        "attachment_hit_count": attachment_hits,
        "anchors": [],
        "visited_nodes": [
            {
                "id": nid,
                "score": 1.0,
                "text": graph.nodes[nid].text,
                "node_type": graph.nodes[nid].node_type,
            }
            for nid in visited_ids
        ],
        "draft_session_nodes": [asdict(x) for x in draft.session_nodes],
        "draft_session_edges": [asdict(x) for x in draft.session_edges],
        "draft_attachments": [asdict(x) for x in draft.attachments],
        "trace": trace,
        "postprocess_events": postprocess_events,
        "synthesis_used": synthesis_used,
        "unresolved_add_sessions_after": [],
        "rejected_edges": [],
        "steps": len(trace),
        "controller_mode": "unified_predictor",
        "signal": signal,
        "predictor_commit_type_accuracy": 1.0 if predicted_commit_family == gold_commit_family else 0.0,
        "predictor_session_node_recall": predictor_metrics.get("session_node_recall"),
        "predictor_session_node_precision": predictor_metrics.get("session_node_precision"),
        "predictor_session_edge_recall": predictor_metrics.get("session_edge_recall"),
        "predictor_session_edge_precision": predictor_metrics.get("session_edge_precision"),
        "predictor_attachment_recall": predictor_metrics.get("attachment_recall"),
        "predictor_attachment_precision": predictor_metrics.get("attachment_precision"),
        "predictor_covered_recall": predictor_metrics.get("covered_recall"),
        "predictor_covered_precision": predictor_metrics.get("covered_precision"),
        "predictor_task_complete_proxy": predictor_metrics.get("task_complete_proxy"),
        "predictor_task_complete_strict": predictor_metrics.get("task_complete_strict"),
        **goal_metrics,
    }


def evaluate_row(row: Mapping[str, Any], cfg: TraversalDraftConfig, predicted_goals: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    if cfg.controller_mode == "predictor_prototype":
        return evaluate_row_predictor_prototype(row, cfg)
    elif cfg.controller_mode == "unified_predictor":
        return evaluate_row_unified_predictor(row, (predicted_goals or {})[row["id"]], cfg)
    return evaluate_row_executor(row, cfg)


def aggregate(results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_task: Dict[str, List[Mapping[str, Any]]] = {}
    for r in results:
        by_task.setdefault(str(r.get("task_type", "unknown")), []).append(r)

    def agg(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        target_rows = [r for r in rows if r.get("target_memory_ids")]
        return {
            "n": len(rows),
            "n_target_rows": len(target_rows),
            "avg_steps": mean([float(r.get("steps", 0)) for r in rows]),
            "avg_visited_nodes": mean([float(len(r.get("visited_nodes", []) or [])) for r in rows]),
            "avg_goal_nodes": mean([float(r.get("goal_session_node_count", 0)) for r in rows]),
            "avg_draft_nodes": mean([float(len(r.get("draft_session_nodes", []) or [])) for r in rows]),
            "avg_extra_nodes": mean([float(r.get("extra_node_count", 0)) for r in rows]),
            "avg_draft_edges": mean([float(len(r.get("draft_session_edges", []) or [])) for r in rows]),
            "avg_goal_edges": mean([float(r.get("goal_session_edge_count", 0)) for r in rows]),
            "avg_extra_edges": mean([float(r.get("extra_edge_count", 0)) for r in rows]),
            "avg_draft_attachments": mean([float(len(r.get("draft_attachments", []) or [])) for r in rows]),
            "avg_goal_attachments": mean([float(r.get("goal_attachment_count", 0) + r.get("goal_cover_count", 0)) for r in rows]),
            "avg_extra_attachments": mean([float(r.get("extra_attachment_count", 0) + r.get("extra_cover_count", 0)) for r in rows]),
            "synthesis_used_rate": mean([1.0 if bool(r.get("synthesis_used", False)) else 0.0 for r in rows]),
            "session_node_recall": mean([float(r["session_node_recall"]) for r in rows if r.get("session_node_recall") is not None]) if any(r.get("session_node_recall") is not None for r in rows) else None,
            "session_node_precision": mean([float(r["session_node_precision"]) for r in rows]) if rows else None,
            "session_edge_recall": mean([float(r["session_edge_recall"]) for r in rows if r.get("session_edge_recall") is not None]) if any(r.get("session_edge_recall") is not None for r in rows) else None,
            "session_edge_precision": mean([float(r["session_edge_precision"]) for r in rows]) if rows else None,
            "attachment_recall": mean([float(r["attachment_recall"]) for r in rows if r.get("attachment_recall") is not None]) if any(r.get("attachment_recall") is not None for r in rows) else None,
            "attachment_precision": mean([float(r["attachment_precision"]) for r in rows]) if rows else None,
            "covered_recall": mean([float(r["covered_recall"]) for r in rows if r.get("covered_recall") is not None]) if any(r.get("covered_recall") is not None for r in rows) else None,
            "covered_precision": mean([float(r["covered_precision"]) for r in rows]) if rows else None,
            "covered_complete_rate": mean([1.0 if bool(r.get("covered_complete", False)) else 0.0 for r in rows if r.get("covered_recall") is not None]) if any(r.get("covered_recall") is not None for r in rows) else None,
            "predictor_session_node_recall": mean([float(r["predictor_session_node_recall"]) for r in rows if r.get("predictor_session_node_recall") is not None]) if any(r.get("predictor_session_node_recall") is not None for r in rows) else None,
            "predictor_session_node_precision": mean([float(r["predictor_session_node_precision"]) for r in rows if r.get("predictor_session_node_precision") is not None]) if any(r.get("predictor_session_node_precision") is not None for r in rows) else None,
            "predictor_session_edge_recall": mean([float(r["predictor_session_edge_recall"]) for r in rows if r.get("predictor_session_edge_recall") is not None]) if any(r.get("predictor_session_edge_recall") is not None for r in rows) else None,
            "predictor_session_edge_precision": mean([float(r["predictor_session_edge_precision"]) for r in rows if r.get("predictor_session_edge_precision") is not None]) if any(r.get("predictor_session_edge_precision") is not None for r in rows) else None,
            "predictor_attachment_recall": mean([float(r["predictor_attachment_recall"]) for r in rows if r.get("predictor_attachment_recall") is not None]) if any(r.get("predictor_attachment_recall") is not None for r in rows) else None,
            "predictor_attachment_precision": mean([float(r["predictor_attachment_precision"]) for r in rows if r.get("predictor_attachment_precision") is not None]) if any(r.get("predictor_attachment_precision") is not None for r in rows) else None,
            "predictor_covered_recall": mean([float(r["predictor_covered_recall"]) for r in rows if r.get("predictor_covered_recall") is not None]) if any(r.get("predictor_covered_recall") is not None for r in rows) else None,
            "predictor_covered_precision": mean([float(r["predictor_covered_precision"]) for r in rows if r.get("predictor_covered_precision") is not None]) if any(r.get("predictor_covered_precision") is not None for r in rows) else None,
            "predictor_task_complete_proxy_rate": mean([1.0 if bool(r.get("predictor_task_complete_proxy", False)) else 0.0 for r in rows]),
            "predictor_task_complete_strict_rate": mean([1.0 if bool(r.get("predictor_task_complete_strict", False)) else 0.0 for r in rows]),
            "predictor_commit_type_accuracy": mean([float(r.get("predictor_commit_type_accuracy", 0.0)) for r in rows]) if rows else None,
            "task_complete_proxy_rate": mean([1.0 if bool(r.get("task_complete_proxy", False)) else 0.0 for r in rows]),
            "task_complete_strict_rate": mean([1.0 if bool(r.get("task_complete_strict", False)) else 0.0 for r in rows]),
            "extra_node_rate": mean([float(r.get("extra_node_rate", 0.0)) for r in rows]) if rows else None,
            "false_edge_rate": mean([float(r.get("false_edge_rate", 0.0)) for r in rows]) if rows else None,
            "false_attachment_rate": mean([float(r.get("false_attachment_rate", 0.0)) for r in rows]) if rows else None,
            "cover_hit_rate": mean([
                1.0 if float(r.get("covered_hit_count", 0)) > 0 else 0.0
                for r in target_rows
            ]) if target_rows else None,
            "attachment_hit_rate": mean([
                1.0 if float(r.get("attachment_hit_count", 0)) > 0 else 0.0
                for r in target_rows
            ]) if target_rows else None,
        }

    return {
        "overall": agg(results),
        "by_task": {k: agg(v) for k, v in sorted(by_task.items())},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-jsonl", default="")
    ap.add_argument("--max-rows", type=int, default=10)
    ap.add_argument("--controller-mode", choices=["executor", "predictor_prototype", "unified_predictor"], default="executor")
    ap.add_argument("--unified-checkpoint", default="")
    ap.add_argument("--cand-emb-cache", default="")
    ap.add_argument("--mem-emb-cache", default="")
    ap.add_argument("--anchor-threshold", type=float, default=0.10)
    ap.add_argument("--traverse-threshold", type=float, default=0.45)
    ap.add_argument("--create-threshold", type=float, default=0.28)
    ap.add_argument("--merge-create-threshold", type=float, default=0.62)
    ap.add_argument("--cover-threshold", type=float, default=0.45)
    ap.add_argument("--attach-threshold", type=float, default=0.35)
    ap.add_argument("--link-edit-threshold", type=float, default=0.55)
    ap.add_argument("--top-k-anchors", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=24)
    ap.add_argument("--max-visited", type=int, default=64)
    ap.add_argument("--max-draft-nodes", type=int, default=8)
    ap.add_argument("--save-json", default="")
    args = ap.parse_args()

    if not args.task_jsonl:
        raise SystemExit("Provide --task-jsonl.")

    cfg = TraversalDraftConfig(
        controller_mode=args.controller_mode,
        anchor_threshold=args.anchor_threshold,
        traverse_threshold=args.traverse_threshold,
        create_threshold=args.create_threshold,
        merge_create_threshold=args.merge_create_threshold,
        cover_threshold=args.cover_threshold,
        attach_threshold=args.attach_threshold,
        link_edit_threshold=args.link_edit_threshold,
        top_k_anchors=args.top_k_anchors,
        max_steps=args.max_steps,
        max_visited=args.max_visited,
        max_draft_nodes=args.max_draft_nodes,
    )

    rows = read_jsonl(args.task_jsonl)[: args.max_rows]
    predicted_goals = None

    
    if args.controller_mode == "unified_predictor":
        if not args.unified_checkpoint:
            raise SystemExit("Must provide --unified-checkpoint for unified_predictor mode.")
        import torch
        from eval_unified_roundtrip import predict_unified_goals
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        predicted_goals = predict_unified_goals(
            checkpoint_path=args.unified_checkpoint,
            rows=rows,
            cand_emb_cache=args.cand_emb_cache,
            mem_emb_cache=args.mem_emb_cache,
            device=device,
        )

    results = [evaluate_row(row, cfg, predicted_goals) for row in rows]
    out = {
        "config": vars(args),
        "summary": aggregate(results),
        "results": results,
    }
    if args.save_json:
        text = json.dumps(out, ensure_ascii=False, indent=2)
        Path(args.save_json).write_text(text, encoding="utf-8")
    
    out_print = {
        "config": vars(args),
        "summary": out["summary"],
    }
    print(json.dumps(out_print, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
