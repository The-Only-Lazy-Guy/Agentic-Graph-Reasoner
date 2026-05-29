from __future__ import annotations

"""
answerer_v1.py

Answerer-v1: dynamic graph reasoning system.

Architecture:
  SessionGraph + frontier-based reasoning loop + PRED-as-local-operator + path reasoning.

  The answerer does NOT follow a fixed chain. It repeatedly chooses a frontier item,
  expands or edits the session graph, scores reasoning paths, and stops when the
  session graph contains enough verified structure to answer.

Key components:
  - SessionGraph: stores reasoning state (nodes, edges, paths, frontier)
  - Path reasoning: relation composition, path search, path scoring
  - PRED integration: calls PRED as a local graph operator
  - Answer composition: template-based answer from best evidence paths
"""

import heapq
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import torch

from graph_core import MemoryGraph, canonical_relation, lexical_overlap, lexical_tokens


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CANONICAL_RELATIONS = ["support", "contradict", "refine", "depend", "cause", "part_of", "example_of", "related"]

# Relation composition table (r1, r2) -> (composed_relation, confidence)
RELATION_COMPOSITION: Dict[Tuple[str, str], Tuple[str, float]] = {
    # Support chains
    ("support", "support"): ("support", 1.0),
    ("support", "refine"): ("support", 0.9),
    ("refine", "support"): ("support", 0.9),
    ("refine", "refine"): ("refine", 1.0),
    # Contradiction chains
    ("support", "contradict"): ("contradict", 0.9),
    ("contradict", "support"): ("contradict", 0.9),
    ("contradict", "contradict"): ("support", 0.7),
    ("refine", "contradict"): ("contradict", 0.8),
    ("contradict", "refine"): ("contradict", 0.8),
    # Dependency chains
    ("depend", "depend"): ("depend", 1.0),
    ("depend", "cause"): ("depend", 0.9),
    ("cause", "depend"): ("depend", 0.9),
    ("cause", "cause"): ("cause", 1.0),
    # Part-whole chains
    ("part_of", "part_of"): ("part_of", 1.0),
    ("part_of", "example_of"): ("part_of", 0.8),
    ("example_of", "part_of"): ("example_of", 0.8),
    ("example_of", "example_of"): ("example_of", 1.0),
    # Cross-family (conservative)
    ("support", "depend"): ("support", 0.5),
    ("depend", "support"): ("support", 0.5),
    ("support", "cause"): ("support", 0.6),
    ("cause", "support"): ("support", 0.6),
    ("support", "part_of"): ("support", 0.6),
    ("part_of", "support"): ("support", 0.6),
    ("depend", "part_of"): ("depend", 0.5),
    ("part_of", "depend"): ("depend", 0.5),
    ("cause", "part_of"): ("cause", 0.5),
    ("part_of", "cause"): ("cause", 0.5),
    # Example links
    ("example_of", "support"): ("support", 0.7),
    ("support", "example_of"): ("support", 0.7),
    ("contradict", "example_of"): ("contradict", 0.6),
    # Unknown bridge
    ("related", "support"): ("support", 0.3),
    ("related", "contradict"): ("contradict", 0.3),
    ("support", "related"): ("support", 0.3),
    ("related", "related"): ("related", 0.2),
}

# Node type family for session nodes
SESSION_NODE_TYPES = {"evidence", "hypothesis", "question", "bridge", "conflict", "conclusion"}

# Frontier item types
FRONTIER_TYPES = {
    "expand_node",
    "extend_path",
    "verify_edge",
    "retrieve_neighbors",
    "resolve_contradiction",
    "summarize_cluster",
}

# Default config
MAX_STEPS = 24
MAX_PATHS = 50
MAX_FRONTIER = 100
PATH_DEPTH_LIMIT = 6
ANSWER_CONFIDENCE_THRESHOLD = 0.65


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class SessionNode:
    id: str
    text: str
    node_type: str = "evidence"
    source_memory_id: Optional[str] = None
    confidence: float = 0.5
    relevance: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_step: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "node_type": self.node_type,
            "source_memory_id": self.source_memory_id,
            "confidence": self.confidence,
            "relevance": self.relevance,
            "metadata": dict(self.metadata),
            "created_step": self.created_step,
        }


@dataclass
class SessionEdge:
    src: str
    dst: str
    relation: str = "related"
    status: str = "draft"
    confidence: float = 0.5
    created_step: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "src": self.src,
            "dst": self.dst,
            "relation": self.relation,
            "status": self.status,
            "confidence": self.confidence,
            "created_step": self.created_step,
            "metadata": dict(self.metadata),
        }


@dataclass
class ReasoningPath:
    path_id: str
    node_ids: List[str]
    edges: List[Tuple[str, str, str]]
    path_type: str = "support_chain"
    confidence: float = 0.5
    evidence_texts: List[str] = field(default_factory=list)
    status: str = "active"
    score: float = 0.0
    created_step: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path_id": self.path_id,
            "node_ids": list(self.node_ids),
            "edges": [(s, d, r) for s, d, r in self.edges],
            "path_type": self.path_type,
            "confidence": self.confidence,
            "evidence_texts": list(self.evidence_texts),
            "status": self.status,
            "score": self.score,
            "created_step": self.created_step,
        }


@dataclass
class FrontierItem:
    item_id: str
    item_type: str = "expand_node"
    node_id: Optional[str] = None
    path_id: Optional[str] = None
    edge_key: Optional[Tuple[str, str]] = None
    score: float = 0.0
    priority: float = 0.0
    reason: str = ""
    created_step: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: "FrontierItem") -> bool:
        return self.priority > other.priority


@dataclass
class AnswerPacket:
    question: str
    answer: str
    confidence: float
    path_count: int
    evidence_paths: List[Dict[str, Any]]
    contradictions_found: int
    steps_taken: int
    answer_type: str = "explanatory"


# ---------------------------------------------------------------------------
# SessionGraph
# ---------------------------------------------------------------------------

class SessionGraph:
    def __init__(self, question: str) -> None:
        self.nodes: Dict[str, SessionNode] = {}
        self.edges: List[SessionEdge] = []
        self.paths: Dict[str, ReasoningPath] = {}
        self.frontier: List[FrontierItem] = []
        self.question = question
        self.step = 0
        self._path_counter = 0
        self._frontier_counter = 0
        self._adj: Dict[str, List[str]] = {}
        self._adj_out: Dict[str, List[str]] = {}
        self._edge_by_pair: Dict[Tuple[str, str], SessionEdge] = {}
        self._rev_edge_by_pair: Dict[Tuple[str, str], SessionEdge] = {}
        self._add_question_node()

    def _add_question_node(self) -> None:
        qid = "Q0"
        self.nodes[qid] = SessionNode(
            id=qid,
            text=self.question,
            node_type="question",
            confidence=1.0,
            relevance=1.0,
        )

    def _rebuild_index(self) -> None:
        self._adj = {nid: [] for nid in self.nodes}
        self._adj_out = {nid: [] for nid in self.nodes}
        self._edge_by_pair = {}
        self._rev_edge_by_pair = {}
        for e in self.edges:
            if e.src not in self.nodes or e.dst not in self.nodes:
                continue
            self._adj[e.src].append(e.dst)
            self._adj[e.dst].append(e.src)
            self._adj_out[e.src].append(e.dst)
            self._edge_by_pair[(e.src, e.dst)] = e

    def add_node(self, node: SessionNode) -> str:
        assert node.id not in self.nodes, f"Node {node.id} already exists"
        self.nodes[node.id] = node
        self._adj[node.id] = []
        self._adj_out[node.id] = []
        return node.id

    def add_edge(self, src: str, dst: str, relation: str, status: str = "draft", confidence: float = 0.5) -> None:
        assert src in self.nodes, f"Source {src} not in session graph"
        assert dst in self.nodes, f"Dest {dst} not in session graph"
        rel = canonical_relation(relation)
        edge = SessionEdge(
            src=src, dst=dst, relation=rel, status=status,
            confidence=confidence, created_step=self.step,
        )
        self.edges.append(edge)
        self._adj.setdefault(src, []).append(dst)
        self._adj.setdefault(dst, []).append(src)
        self._adj_out.setdefault(src, []).append(dst)
        self._edge_by_pair[(src, dst)] = edge

    def edge_between(self, src: str, dst: str) -> Optional[SessionEdge]:
        return self._edge_by_pair.get((src, dst))

    def out_neighbors(self, node_id: str) -> List[str]:
        return list(self._adj_out.get(node_id, []))

    def all_neighbors(self, node_id: str) -> List[str]:
        return list(self._adj.get(node_id, []))

    def import_memory_nodes(self, memory_ids: Sequence[str], graph: MemoryGraph) -> List[str]:
        imported: List[str] = []
        for mem_id in memory_ids:
            mem_id = str(mem_id)
            if mem_id in self.nodes:
                continue
            if mem_id not in graph.nodes:
                continue
            gnode = graph.nodes[mem_id]
            node = SessionNode(
                id=mem_id,
                text=gnode.text,
                node_type="evidence",
                source_memory_id=mem_id,
                confidence=gnode.confidence,
                relevance=self._compute_relevance(gnode.text),
                metadata={"node_type": gnode.node_type, "importance": gnode.importance},
                created_step=self.step,
            )
            self.nodes[mem_id] = node
            self._adj[mem_id] = []
            self._adj_out[mem_id] = []
            imported.append(mem_id)
        for nid in imported:
            self.add_edge("Q0", nid, "related", status="draft", confidence=0.5)
        return imported

    def _compute_relevance(self, text: str) -> float:
        return float(lexical_overlap(self.question, text))

    def add_path(self, path: ReasoningPath) -> None:
        self.paths[path.path_id] = path

    def add_frontier_item(self, item: FrontierItem) -> None:
        if len(self.frontier) >= MAX_FRONTIER * 2:
            self.frontier.sort(key=lambda x: -x.priority)
            self.frontier = self.frontier[:MAX_FRONTIER]
        self.frontier.append(item)

    def pop_best_frontier(self) -> Optional[FrontierItem]:
        if not self.frontier:
            return None
        best = max(self.frontier, key=lambda x: x.priority)
        self.frontier.remove(best)
        return best

    def has_node(self, node_id: str) -> bool:
        return node_id in self.nodes

    def node_text(self, node_id: str) -> str:
        node = self.nodes.get(node_id)
        return node.text if node else ""

    def get_evidence_nodes(self) -> List[SessionNode]:
        return [n for n in self.nodes.values() if n.node_type == "evidence"]

    def get_hypothesis_nodes(self) -> List[SessionNode]:
        return [n for n in self.nodes.values() if n.node_type == "hypothesis"]

    def get_edges_by_status(self, status: str) -> List[SessionEdge]:
        return [e for e in self.edges if e.status == status]

    def update_edge_status(self, src: str, dst: str, status: str, confidence: Optional[float] = None) -> bool:
        edge = self._edge_by_pair.get((src, dst))
        if edge is None:
            return False
        edge.status = status
        if confidence is not None:
            edge.confidence = confidence
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "step": self.step,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "edges": [e.to_dict() for e in self.edges],
            "paths": {k: v.to_dict() for k, v in self.paths.items()},
            "frontier": [(f.item_id, f.item_type, f.priority) for f in self.frontier],
        }


# ---------------------------------------------------------------------------
# Path Reasoning
# ---------------------------------------------------------------------------

def compose_relations(r1: str, r2: str) -> Tuple[str, float]:
    key = (canonical_relation(r1), canonical_relation(r2))
    if key in RELATION_COMPOSITION:
        return RELATION_COMPOSITION[key]
    rev_key = (canonical_relation(r2), canonical_relation(r1))
    if rev_key in RELATION_COMPOSITION:
        result = RELATION_COMPOSITION[rev_key]
        return result
    return ("related", 0.2)


def compose_path_relations(edges: List[Tuple[str, str, str]]) -> Tuple[str, float]:
    if not edges:
        return ("related", 0.0)
    current_rel = canonical_relation(edges[0][2])
    current_conf = 1.0
    for i in range(1, len(edges)):
        next_rel = canonical_relation(edges[i][2])
        composed, conf = compose_relations(current_rel, next_rel)
        current_rel = composed
        current_conf = current_conf * conf * 0.95
    return current_rel, current_conf


def classify_path_type(edges: List[Tuple[str, str, str]]) -> str:
    if not edges:
        return "unknown"
    relations = [canonical_relation(e[2]) for e in edges]
    has_contradiction = any(r == "contradict" for r in relations)
    has_depend = any(r in ("depend", "cause") for r in relations)
    has_part = any(r in ("part_of", "example_of") for r in relations)
    has_support = any(r in ("support", "refine") for r in relations)
    all_related = all(r == "related" for r in relations)

    if all_related:
        return "direct"
    if has_contradiction:
        return "contradiction"
    if has_depend:
        return "dependency"
    if has_part:
        return "part_whole"
    if has_support:
        return "support_chain"
    return "direct"


def find_paths_bfs(
    session: SessionGraph,
    start_id: str,
    end_id: Optional[str] = None,
    max_depth: int = 5,
    max_paths: int = 20,
) -> List[ReasoningPath]:
    found: List[ReasoningPath] = []
    if start_id not in session.nodes:
        return found
    visited_paths: Set[str] = set()
    queue: List[Tuple[str, List[str], List[Tuple[str, str, str]]]] = [
        (start_id, [start_id], [])
    ]

    while queue and len(found) < max_paths:
        current, node_path, edge_path = queue.pop(0)

        if end_id is not None and current == end_id and len(node_path) > 1:
            path_key = "->".join(node_path)
            if path_key not in visited_paths:
                visited_paths.add(path_key)
                composed_rel, composed_conf = compose_path_relations(edge_path)
                ptype = classify_path_type(edge_path)
                evidence_texts = [session.node_text(nid) for nid in node_path]
                path = ReasoningPath(
                    path_id=f"p{len(found)}",
                    node_ids=list(node_path),
                    edges=list(edge_path),
                    path_type=ptype,
                    confidence=composed_conf,
                    evidence_texts=evidence_texts,
                    score=composed_conf * len(edge_path),
                    created_step=session.step,
                )
                found.append(path)
            if len(found) >= max_paths:
                break
            continue

        if len(node_path) >= max_depth:
            continue

        for neighbor in session.all_neighbors(current):
            if neighbor in node_path:
                continue
            edge = session.edge_between(current, neighbor)
            if edge is None:
                edge = session.edge_between(neighbor, current)
            if edge is None:
                continue
            queue.append(
                (neighbor, node_path + [neighbor], edge_path + [(current, neighbor, edge.relation)])
            )

    return found


def find_all_paths_from_question(
    session: SessionGraph,
    max_depth: int = 5,
    max_paths: int = 30,
) -> List[ReasoningPath]:
    all_paths: List[ReasoningPath] = []
    question_nodes = [nid for nid, n in session.nodes.items() if n.node_type == "question"]
    evidence_nodes = [nid for nid, n in session.nodes.items() if n.node_type == "evidence"]
    hypothesis_nodes = [nid for nid, n in session.nodes.items() if n.node_type == "hypothesis"]
    target_nodes = evidence_nodes + hypothesis_nodes

    for qid in question_nodes:
        for target in target_nodes:
            paths = find_paths_bfs(session, qid, target, max_depth=max_depth, max_paths=max_paths // max(len(question_nodes), 1))
            all_paths.extend(paths)

    path_set: Dict[str, ReasoningPath] = {}
    for p in all_paths:
        key = "->".join(p.node_ids)
        if key not in path_set or p.confidence > path_set[key].confidence:
            path_set[key] = p
    return list(path_set.values())[:max_paths]


def score_path(path: ReasoningPath, question: str, session: SessionGraph) -> float:
    relevance_score = 0.0
    for nid in path.node_ids:
        node = session.nodes.get(nid)
        if node is not None:
            relevance_score += float(lexical_overlap(question, node.text))
    relevance_score /= max(len(path.node_ids), 1)

    length_penalty = math.exp(-0.15 * (len(path.node_ids) - 1))
    confidence_score = path.confidence

    contradiction_bonus = 0.2 if path.path_type == "contradiction" else 0.0
    dependency_bonus = 0.1 if path.path_type == "dependency" else 0.0

    score = (
        0.3 * relevance_score
        + 0.3 * confidence_score
        + 0.2 * length_penalty
        + 0.1 * contradiction_bonus
        + 0.1 * dependency_bonus
    )
    return score


def rank_paths(paths: List[ReasoningPath], question: str, session: SessionGraph) -> List[ReasoningPath]:
    for p in paths:
        p.score = score_path(p, question, session)
    paths.sort(key=lambda x: -x.score)
    return paths


def find_contradictions(session: SessionGraph) -> List[List[str]]:
    contradictions: List[List[str]] = []
    visited: Set[str] = set()
    for e in session.edges:
        if e.relation != "contradict":
            continue
        pair = tuple(sorted([e.src, e.dst]))
        if pair in visited:
            continue
        visited.add(pair)
        contradictions.append(list(pair))
    return contradictions


def find_contradictions_on_paths(session: SessionGraph, paths: List[ReasoningPath]) -> List[List[str]]:
    relevant_node_ids: Set[str] = set()
    for p in paths:
        for nid in p.node_ids:
            relevant_node_ids.add(nid)
    contradictions: List[List[str]] = []
    visited: Set[str] = set()
    for e in session.edges:
        if e.relation != "contradict":
            continue
        if e.src not in relevant_node_ids and e.dst not in relevant_node_ids:
            continue
        pair = tuple(sorted([e.src, e.dst]))
        if pair in visited:
            continue
        visited.add(pair)
        contradictions.append(list(pair))
    return contradictions


# ---------------------------------------------------------------------------
# Expansion Strategies
# ---------------------------------------------------------------------------

def retrieve_anchors(question: str, graph: MemoryGraph, k: int = 8) -> List[str]:
    scored: List[Tuple[float, str]] = []
    for nid, node in graph.nodes.items():
        score = lexical_overlap(question, node.text)
        importance_bonus = node.importance * 0.2
        scored.append((score + importance_bonus, nid))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [nid for _score, nid in scored[:k]]


def retrieve_neighbors(node_ids: Sequence[str], graph: MemoryGraph, max_per_node: int = 4) -> List[str]:
    gathered: List[str] = []
    seen: Set[str] = set(node_ids)
    for nid in node_ids:
        if nid not in graph.nodes:
            continue
        neighbors = graph.out_neighbors(nid)
        scored = []
        for neighbor in neighbors:
            if neighbor in seen:
                continue
            seen.add(neighbor)
            node = graph.nodes.get(neighbor)
            score = node.importance if node else 0.0
            scored.append((score, neighbor))
        scored.sort(key=lambda x: (-x[0], x[1]))
        gathered.extend(nid for _score, nid in scored[:max_per_node])
    return gathered


def expand_node(session: SessionGraph, node_id: str, graph: MemoryGraph) -> int:
    if node_id not in graph.nodes:
        return 0
    neighbors = graph.out_neighbors(node_id)
    imported = session.import_memory_nodes(neighbors, graph)
    for nid in neighbors:
        if not session.has_node(nid):
            continue
        if session.edge_between(node_id, nid) is not None:
            continue
        edge = graph.directed_edge_between(node_id, nid)
        if edge is not None:
            session.add_edge(node_id, nid, canonical_relation(edge.relation), status="draft", confidence=edge.strength)
    return len(imported)


def init_frontier_from_anchors(session: SessionGraph, anchor_ids: List[str]) -> None:
    for nid in anchor_ids:
        if not session.has_node(nid):
            continue
        relevance = session.nodes[nid].relevance if nid in session.nodes else 0.5
        item = FrontierItem(
            item_id=f"f_expand_{nid}",
            item_type="expand_node",
            node_id=nid,
            score=relevance,
            priority=relevance * 0.8 + 0.2,
            reason="initial_anchor",
        )
        session.add_frontier_item(item)


def build_frontier_from_paths(session: SessionGraph, paths: List[ReasoningPath]) -> None:
    frontier_thread_ids: Set[str] = set()
    for path in paths:
        if path.status != "active":
            continue
        for nid in path.node_ids:
            if nid in frontier_thread_ids:
                continue
            if session.nodes[nid].node_type == "evidence":
                frontier_thread_ids.add(nid)
                score = path.score * 0.7
                item = FrontierItem(
                    item_id=f"f_expand_{nid}",
                    item_type="expand_node",
                    node_id=nid,
                    path_id=path.path_id,
                    score=score,
                    priority=score,
                    reason="path_extension",
                )
                session.add_frontier_item(item)

    contradictions = find_contradictions(session)
    for pair in contradictions:
        cid = f"f_contradict_{pair[0]}_{pair[1]}"
        if cid not in frontier_thread_ids:
            frontier_thread_ids.add(cid)
            item = FrontierItem(
                item_id=cid,
                item_type="resolve_contradiction",
                node_id=pair[0],
                edge_key=(pair[0], pair[1]),
                score=0.9,
                priority=0.9,
                reason=f"contradiction_{pair[0]}_vs_{pair[1]}",
            )
            session.add_frontier_item(item)


# ---------------------------------------------------------------------------
# Answer Composition (100% offline, template-based, no LLM)
# ---------------------------------------------------------------------------

def classify_question_type(question: str) -> str:
    q = question.lower()
    if any(w in q for w in ("why", "reason", "cause", "because", "explain")):
        return "explanatory"
    if any(w in q for w in ("how", "process", "way", "method")):
        return "procedural"
    if any(w in q for w in ("what", "define", "meaning")):
        return "definitional"
    if any(w in q for w in ("true", "false", "correct", "valid", "verify", "confirm")):
        return "verification"
    if any(w in q for w in ("compare", "difference", "versus", "vs")):
        return "comparison"
    if any(w in q for w in ("example", "instance", "case")):
        return "example"
    return "general"


def compose_answer(
    question: str,
    session: SessionGraph,
    max_evidence_paths: int = 3,
) -> AnswerPacket:
    all_paths = find_all_paths_from_question(session, max_depth=PATH_DEPTH_LIMIT)
    all_paths = rank_paths(all_paths, question, session)
    best_paths = all_paths[:max_evidence_paths]
    contradictions = find_contradictions_on_paths(session, best_paths)
    qtype = classify_question_type(question)

    evidence_data: List[Dict[str, Any]] = []
    for p in best_paths:
        evidence_data.append({
            "path_id": p.path_id,
            "type": p.path_type,
            "confidence": p.confidence,
            "score": p.score,
            "nodes": p.node_ids,
            "edges": [(s, d, r) for s, d, r in p.edges],
            "texts": p.evidence_texts,
        })

    if not best_paths:
        answer = _compose_no_evidence_answer(question, qtype)
    elif contradictions:
        answer = _compose_contradiction_answer(question, best_paths, contradictions, qtype, session)
    elif qtype == "explanatory":
        answer = _compose_explanatory_answer(question, best_paths, session)
    elif qtype == "procedural":
        answer = _compose_procedural_answer(question, best_paths, session)
    elif qtype == "verification":
        answer = _compose_verification_answer(question, best_paths, session)
    elif qtype == "comparison":
        answer = _compose_comparison_answer(question, best_paths, session)
    else:
        answer = _compose_general_answer(question, best_paths, session)

    confidence = max((p.confidence for p in best_paths), default=0.0)

    return AnswerPacket(
        question=question,
        answer=answer,
        confidence=confidence,
        path_count=len(best_paths),
        evidence_paths=evidence_data,
        contradictions_found=len(contradictions),
        steps_taken=session.step,
        answer_type=qtype,
    )


def _compose_no_evidence_answer(question: str, qtype: str) -> str:
    return (
        f'Based on the available information, I cannot find sufficient evidence to answer "{question}". '
        f"The graph does not contain enough connected structure to form a reliable reasoning path."
    )


def _compose_contradiction_answer(
    question: str,
    paths: List[ReasoningPath],
    contradictions: List[List[str]],
    qtype: str,
    session: Optional[SessionGraph] = None,
) -> str:
    lines: List[str] = []
    for p in paths[:2]:
        line = _format_path_text(p, session)
        lines.append(f"  - {line}")
    c_lines = []
    for pair in contradictions[:2]:
        c_lines.append(f'  - Contradiction between nodes "{pair[0]}" and "{pair[1]}"')

    if c_lines:
        contradiction_section = "\n".join(c_lines)
    else:
        contradiction_section = "  - Contradiction detected in reasoning structure."

    best = paths[0] if paths else None
    if best and best.confidence > 0.5:
        conclusion = _extract_conclusion_from_path(best)
        return (
            f'The analysis reveals conflicting evidence regarding "{question}".\n\n'
            f"Evidence paths:\n"
            f"{chr(10).join(lines)}\n\n"
            f"Contradictions found:\n"
            f"{contradiction_section}\n\n"
            f"Conclusion: {conclusion} "
            f"The evidence is divided, suggesting this is an open question requiring further investigation."
        )
    return (
        f'The analysis reveals conflicting evidence regarding "{question}".\n\n'
        f"Evidence paths:\n"
        f"{chr(10).join(lines)}\n\n"
        f"Contradictions found:\n"
        f"{contradiction_section}\n\n"
        f"Due to unresolved contradictions, a definitive answer cannot be given."
    )


def _format_path_text(p: ReasoningPath, session: Optional[SessionGraph] = None) -> str:
    parts: List[str] = []
    skip_question = session is not None
    for i, nid in enumerate(p.node_ids):
        if skip_question and session.nodes.get(nid, None) and session.nodes[nid].node_type == "question":
            continue
        text = p.evidence_texts[i] if i < len(p.evidence_texts) else nid
        text = text[:80].replace("\n", " ")
        parts.append(f'"{text}"')
    flat_parts = [p for p in parts if p]
    if len(flat_parts) <= 1:
        return flat_parts[0] if flat_parts else "(empty)"

    evidence_edge_map: Dict[int, str] = {}
    edge_idx = 0
    for i in range(len(p.node_ids)):
        if skip_question and session and session.nodes.get(p.node_ids[i], None) and session.nodes[p.node_ids[i]].node_type == "question":
            continue
        if edge_idx < len(p.edges) and i < len(p.node_ids) - 1:
            nxt = p.node_ids[i + 1]
            if skip_question and session and session.nodes.get(nxt, None) and session.nodes[nxt].node_type == "question":
                edge_idx += 1
                continue
            rel = canonical_relation(p.edges[edge_idx][2])
            if rel not in ("related", "unknown"):
                evidence_edge_map[i] = rel
            edge_idx += 1

    result: List[str] = []
    for i, part in enumerate(flat_parts):
        result.append(part)
        if i in evidence_edge_map:
            result.append(f"==[{evidence_edge_map[i]}]==")
        else:
            result.append("->")
    return " ".join(result[:-1])


def _compose_explanatory_answer(question: str, paths: List[ReasoningPath], session: Optional[SessionGraph] = None) -> str:
    lines: List[str] = []
    for p in paths[:3]:
        line = _format_path_text(p, session)
        lines.append(f"  - {line} (conf={p.confidence:.2f})")
    best = paths[0] if paths else None
    conclusion = _extract_conclusion_from_path(best) if best else "Cannot determine."
    prev_count = len(lines)
    if prev_count == 1:
        return (
            f'Explanation for "{question}":\n\n'
            f"Evidence chain:\n"
            f"{chr(10).join(lines)}\n\n"
            f"Conclusion: {conclusion}"
        )
    return (
        f'Explanation for "{question}":\n\n'
        f"Evidence chains ({prev_count} paths):\n"
        f"{chr(10).join(lines)}\n\n"
        f"Conclusion: {conclusion}"
    )


def _compose_verification_answer(question: str, paths: List[ReasoningPath], session: Optional[SessionGraph] = None) -> str:
    best = paths[0] if paths else None
    if best is None:
        return f'Insufficient evidence to verify the claim in "{question}".'
    has_contradiction = any(r == "contradict" for _, _, r in best.edges)
    has_support = any(r in ("support", "refine") for _, _, r in best.edges)

    if has_contradiction and not has_support:
        verdict = "likely false"
    elif has_support and not has_contradiction:
        verdict = "likely true"
    else:
        verdict = "uncertain (mixed evidence)"

    evidence_line = _format_path_text(best, session)

    return (
        f'Verification of "{question}":\n'
        f"Verdict: {verdict}\n"
        f"Evidence: {evidence_line}\n"
        f"Confidence: {best.confidence:.2f}"
    )


def _compose_comparison_answer(question: str, paths: List[ReasoningPath], session: Optional[SessionGraph] = None) -> str:
    lines: List[str] = []
    for i, p in enumerate(paths[:2]):
        line = _format_path_text(p, session)
        lines.append(f"  Path {i + 1} ({p.path_type}, conf={p.confidence:.2f}): {line}")
    return (
        f'Comparison analysis for "{question}":\n'
        f"{chr(10).join(lines)}\n\n"
        f"Based on {len(paths)} evidence paths identified in the graph."
    )


def _compose_general_answer(question: str, paths: List[ReasoningPath], session: Optional[SessionGraph] = None) -> str:
    lines: List[str] = []
    has_direct = False
    for p in paths:
        line = _format_path_text(p, session)
        if p.path_type == "direct":
            has_direct = True
        lines.append(f"  - {line}")
    summary = f"Found {len(paths)} relevant evidence paths."
    if has_direct:
        summary += " Some nodes are directly relevant to the question but not yet connected through intermediate reasoning."
    return (
        f'Regarding "{question}":\n'
        f"{chr(10).join(lines)}\n\n"
        f"{summary}"
    )


def _compose_procedural_answer(question: str, paths: List[ReasoningPath], session: Optional[SessionGraph] = None) -> str:
    dep_paths = [p for p in paths if p.path_type == "dependency"]
    support_paths = [p for p in paths if p.path_type in ("support_chain", "direct")]
    lines: List[str] = []
    for p in (dep_paths + support_paths)[:3]:
        lines.append(f"  - {_format_path_text(p, session)}")
    if not lines:
        return _compose_general_answer(question, paths, session)
    return (
        f'Procedure for "{question}":\n'
        f"{chr(10).join(lines)}\n\n"
        f"The evidence shows {len(dep_paths)} dependency chain(s) and {len(support_paths)} supporting evidence path(s)."
    )


def _extract_conclusion_from_path(path: ReasoningPath) -> str:
    if path.evidence_texts:
        return path.evidence_texts[-1]
    return "unknown"


# ---------------------------------------------------------------------------
# PRED Integration
# ---------------------------------------------------------------------------

def run_pred_and_apply(
    session: SessionGraph,
    pred_model: Any,
    signal: str,
    memory_ids: List[str],
    graph: MemoryGraph,
    device: torch.device,
    hash_dim: int = 512,
) -> int:
    if pred_model is None:
        return 0
    spans = _build_spans_from_memory(memory_ids, graph)
    row = _build_pseudo_row(signal, spans, memory_ids, graph)

    batch = _build_pred_batch(row, hash_dim, device)
    if batch is None:
        return 0

    pred_model.eval()
    with torch.no_grad():
        out = pred_model(batch)

    nodes_added = _apply_pred_output(session, out, row, memory_ids, graph, pred_model)
    return nodes_added


def _build_spans_from_memory(
    memory_ids: List[str],
    graph: MemoryGraph,
    max_spans: int = 16,
) -> List[Dict[str, Any]]:
    spans: List[Dict[str, Any]] = []
    for i, mid in enumerate(memory_ids[:max_spans]):
        if mid not in graph.nodes:
            continue
        node = graph.nodes[mid]
        spans.append({
            "id": f"span_{mid}",
            "text": node.text,
            "span_kind": "clause",
            "start": 0.0,
            "end": min(len(node.text), 512),
        })
    return spans


def _build_pseudo_row(
    signal: str,
    spans: List[Dict[str, Any]],
    memory_ids: List[str],
    graph: MemoryGraph,
) -> Dict[str, Any]:
    return {
        "signal": signal,
        "spans": spans,
        "initial_memory_node_ids": memory_ids,
        "graph_path": "",
        "task_type": "answerer_v1",
    }


def _build_pred_batch(
    row: Dict[str, Any],
    hash_dim: int,
    device: torch.device,
) -> Any:
    from graph_policy_model import bow_hash
    from train_unified_v1 import UnifiedDataset, collate, to_device

    dummy_rows = [row]
    dataset = UnifiedDataset(dummy_rows, hash_dim=hash_dim)
    if len(dataset) == 0:
        return None
    sample = dataset[0]
    dummy_batch = collate([sample])
    return to_device(dummy_batch, device)


def _apply_pred_output(
    session: SessionGraph,
    out: Dict[str, torch.Tensor],
    row: Dict[str, Any],
    memory_ids: List[str],
    graph: MemoryGraph,
    pred_model: Any,
) -> int:
    from eval_unified_roundtrip import decode_unified_prediction_to_goal
    from train_unified_v1 import build_predicted_session_nodes, derive_gold_targets, decode_edge_predictions, decode_mem_kind_predictions
    from pred_model import REL_WITH_NONE, MEM_LINK_KIND_TO_ID, COMMIT_FAMILIES

    B = 1
    use_pred = (torch.sigmoid(out["use_logits"]) >= 0.5)
    bridge_pred = (torch.sigmoid(out["type_logits"]) >= 0.5) & use_pred
    span_pred = out["span_logits"].argmax(dim=-1)
    commit_pred = out["commit_logits"].argmax(dim=-1)
    pred_edge_mask = use_pred[:, :, None] & use_pred[:, None, :]
    diag = torch.eye(use_pred.size(1), device=out["use_logits"].device, dtype=torch.bool)[None, :, :]
    edge_exist_pred = decode_edge_predictions(out["edge_exist_logits"], pred_edge_mask & ~diag)
    edge_logits = out["verifier_logits"] if pred_model.use_verifier else out["edge_rel_logits"]
    edge_rel_pred = edge_logits.argmax(dim=-1)
    mem_kind_pred = decode_mem_kind_predictions(out["mem_kind_logits"], torch.ones(B, memory_ids, device=out["use_logits"].device))
    mem_rel_pred = out["mem_rel_logits"].argmax(dim=-1)

    pred_nodes = build_predicted_session_nodes(
        row,
        use_pred=use_pred[0].cpu(),
        span_pred=span_pred[0].cpu(),
        bridge_pred=bridge_pred[0].cpu(),
        mixed_dst_pred=out["mixed_dst_mem_logits"].argmax(dim=-1)[0].cpu(),
        bridge_a_pred=out["bridge_mem_a_logits"].argmax(dim=-1)[0].cpu(),
        bridge_b_pred=out["bridge_mem_b_logits"].argmax(dim=-1)[0].cpu(),
        memory_ids=memory_ids,
        graph=graph,
    )
    pred_edges_set: Set[Tuple[str, str, str]] = set()
    pred_names = [n.get("name", "") for n in pred_nodes]
    for i, src_name in enumerate(pred_names):
        for j, dst_name in enumerate(pred_names):
            if i == j:
                continue
            if bool(edge_exist_pred[0, i, j].item()):
                rel_id = int(edge_rel_pred[0, i, j].item())
                rel = REL_WITH_NONE[rel_id] if rel_id < len(REL_WITH_NONE) else "none"
                if rel != "none":
                    pred_edges_set.add((src_name, dst_name, rel))

    pred_attach: Set[Tuple[str, str, str]] = set()
    pred_cover: Set[Tuple[str, str]] = set()
    for i, name in enumerate(pred_names):
        for j, mid in enumerate(memory_ids):
            if j >= mem_kind_pred.size(2):
                continue
            kind_id = int(mem_kind_pred[0, i, j].item())
            if kind_id == MEM_LINK_KIND_TO_ID.get("attach", -1):
                rel_id = int(mem_rel_pred[0, i, j].item())
                rel = REL_WITH_NONE[rel_id] if rel_id < len(REL_WITH_NONE) else "related"
                pred_attach.add((name, str(mid), rel))
            elif kind_id == MEM_LINK_KIND_TO_ID.get("cover", -1):
                pred_cover.add((name, str(mid)))

    commit_name = COMMIT_FAMILIES[int(commit_pred[0].item())] if int(commit_pred[0].item()) < len(COMMIT_FAMILIES) else "other"
    predicted_goal = decode_unified_prediction_to_goal(
        row, pred_nodes, pred_edges_set, pred_attach, pred_cover, commit_name,
    )

    added = 0
    for spec in predicted_goal.get("session_nodes", []):
        nid = f'pred_{str(spec.get("name", ""))}'
        if nid not in session.nodes:
            node = SessionNode(
                id=nid,
                text=str(spec.get("span_text", "")),
                node_type="evidence",
                confidence=0.7,
                relevance=float(lexical_overlap(session.question, str(spec.get("span_text", "")))),
                created_step=session.step,
            )
            session.add_node(node)
            added += 1

    for spec in predicted_goal.get("session_edges", []):
        src = f'pred_{str(spec.get("src", ""))}'
        dst = f'pred_{str(spec.get("dst", ""))}'
        rel = str(spec.get("relation", "related"))
        if src in session.nodes and dst in session.nodes:
            session.add_edge(src, dst, rel, status="draft", confidence=0.6)

    for att in predicted_goal.get("memory_attachments", []):
        session_name = str(att.get("session", ""))
        session_nid = f"pred_{session_name}"
        mem_id = str(att.get("memory_id", ""))
        if mem_id not in session.nodes and mem_id in graph.nodes:
            session.import_memory_nodes([mem_id], graph)
        if session_nid in session.nodes and mem_id in session.nodes:
            session.add_edge(session_nid, mem_id, str(att.get("relation", "related")), status="verified", confidence=0.7)

    return added


# ---------------------------------------------------------------------------
# Main Answerer Loop
# ---------------------------------------------------------------------------

def answer_query(
    question: str,
    graph: MemoryGraph,
    pred_model: Any = None,
    pred_device: Optional[torch.device] = None,
    max_steps: int = MAX_STEPS,
    k_anchors: int = 8,
    path_depth: int = PATH_DEPTH_LIMIT,
    expansion_budget: int = 6,
) -> AnswerPacket:
    session = SessionGraph(question)
    anchors = retrieve_anchors(question, graph, k=k_anchors)
    session.import_memory_nodes(anchors, graph)
    init_frontier_from_anchors(session, anchors)

    if pred_model is not None and pred_device is not None:
        memory_ids = [nid for nid in session.get_evidence_nodes()]
        memory_ids = [n.source_memory_id for n in session.get_evidence_nodes() if n.source_memory_id]
        run_pred_and_apply(session, pred_model, question, memory_ids, graph, pred_device)

    for step in range(max_steps):
        session.step = step
        item = session.pop_best_frontier()
        if item is None:
            break

        if item.item_type == "expand_node" and item.node_id is not None:
            n_imported = expand_node(session, item.node_id, graph)
            if n_imported > 0:
                new_paths = find_paths_bfs(session, "Q0", max_depth=path_depth)
                for p in new_paths:
                    if p.path_id not in session.paths:
                        session.add_path(p)
                build_frontier_from_paths(session, new_paths)

        elif item.item_type == "resolve_contradiction" and item.edge_key is not None:
            src, dst = item.edge_key
            if session.edge_between(src, dst):
                expand_node(session, src, graph)
                expand_node(session, dst, graph)

        elif item.item_type == "extend_path" and item.path_id is not None:
            path = session.paths.get(item.path_id)
            if path is not None and path.node_ids:
                last_node = path.node_ids[-1]
                expand_node(session, last_node, graph)

        elif item.item_type == "verify_edge" and item.edge_key is not None:
            src, dst = item.edge_key
            session.update_edge_status(src, dst, "verified", confidence=0.8)

    all_paths = find_all_paths_from_question(session, max_depth=path_depth)
    for p in all_paths:
        if p.path_id not in session.paths:
            session.add_path(p)
    all_paths = rank_paths(all_paths, question, session)

    packet = compose_answer(question, session, max_evidence_paths=3)
    packet.steps_taken = session.step
    return packet


# ---------------------------------------------------------------------------
# CLI entry point for testing
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Answerer-v1: dynamic graph reasoning")
    parser.add_argument("--question", type=str, default="Explain the dependency in this optimization.")
    parser.add_argument("--graph", type=str, required=True, help="Path to graph JSON")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--k-anchors", type=int, default=8)
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    graph = MemoryGraph.load_json(args.graph)
    print(f"Loaded graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    print(f"Question: {args.question}")

    packet = answer_query(
        question=args.question,
        graph=graph,
        max_steps=args.max_steps,
        k_anchors=args.k_anchors,
    )

    print("\n" + "=" * 60)
    print(f"Answer (type={packet.answer_type}, conf={packet.confidence:.3f})")
    print("=" * 60)
    print(packet.answer)
    print("=" * 60)
    print(f"Steps: {packet.steps_taken}, Paths: {packet.path_count}, Contradictions: {packet.contradictions_found}")

    if args.output:
        output = {
            "question": args.question,
            "answer": packet.answer,
            "confidence": packet.confidence,
            "answer_type": packet.answer_type,
            "steps_taken": packet.steps_taken,
            "path_count": packet.path_count,
            "contradictions_found": packet.contradictions_found,
            "evidence_paths": packet.evidence_paths,
        }
        Path(args.output).write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nOutput saved to {args.output}")


if __name__ == "__main__":
    main()
