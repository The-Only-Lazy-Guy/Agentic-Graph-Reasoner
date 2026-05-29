"""Post-session consolidation: reasoning results → graph mutations.

After each reasoning session (or batch of sessions), extract what was learned
and update the persistent MemoryGraph:

  - Create a "session_memory" node for each session
  - Link it to activated signals and relevant existing nodes
  - Reinforce/punish edge strengths based on outcome correctness

All graph mutations are batched, then applied atomically.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from graph_core import Edge, MemoryGraph, Node
from reasoning.outcome_scorer import SubstrateOutcomeRow
from reasoning.reasoning_loop import ReasoningResult


# ── Data structures ──────────────────────────────────────────────────────


@dataclass
class SessionMemory:
    """Compact record of one reasoning session for graph consolidation."""

    task_id: str
    question: str
    answer: str
    activated_signal_ids: List[str]
    activated_signal_source_ids: List[Optional[str]]
    outcome_correct: Optional[bool]
    outcome_score: float
    outcome_source: str
    timestamp: str

    @property
    def memory_id(self) -> str:
        raw = f"{self.task_id}_{self.timestamp}"
        return f"session_memory_{hashlib.sha256(raw.encode()).hexdigest()[:12]}"

    def summary_text(self, max_question: int = 80, max_answer: int = 120) -> str:
        q = self.question[:max_question]
        a = self.answer.replace("\n", " ")[:max_answer]
        score = f"{self.outcome_score}" if self.outcome_correct is not None else "?"
        src = self.outcome_source
        return f"Task: {q} | Answer: {a} | Score: {score} | Source: {src}"


@dataclass
class GraphUpdate:
    """Batch of graph mutations to apply atomically."""

    new_nodes: List[Node] = field(default_factory=list)
    new_edges: List[Edge] = field(default_factory=list)
    edge_strength_updates: List[Tuple[str, str, float]] = field(default_factory=list)


# ── Extraction ───────────────────────────────────────────────────────────


def extract_session_memory(
    result: ReasoningResult,
    task: Dict[str, Any],
    *,
    outcome_row: Optional[SubstrateOutcomeRow] = None,
) -> SessionMemory:
    """Extract a SessionMemory from one completed reasoning session."""
    audit = result.audit_summary or {}
    debug_dump: List[Dict[str, Any]] = audit.get("debug_signal_dump", []) or []

    signal_ids: List[str] = []
    source_ids: List[Optional[str]] = []
    seen: set = set()
    for row in debug_dump:
        rid = row.get("id", "")
        if rid and rid not in seen:
            seen.add(rid)
            signal_ids.append(rid)
            source_ids.append(row.get("source_node_id"))

    correct: Optional[bool] = None
    score = 0.0
    source = "unknown"
    if outcome_row is not None:
        correct = outcome_row.outcome_correct
        score = outcome_row.outcome_score
        source = outcome_row.outcome_source

    return SessionMemory(
        task_id=str(task.get("id", "")),
        question=str(task.get("question", "")),
        answer=result.answer,
        activated_signal_ids=signal_ids,
        activated_signal_source_ids=source_ids,
        outcome_correct=correct,
        outcome_score=score,
        outcome_source=source,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


def extract_session_memories(
    results: Sequence[Tuple[ReasoningResult, Dict[str, Any]]],
    outcome_rows: Optional[Sequence[Optional[SubstrateOutcomeRow]]] = None,
) -> List[SessionMemory]:
    """Extract memories from a batch of completed sessions."""
    memories: List[SessionMemory] = []
    for i, (result, task) in enumerate(results):
        row = outcome_rows[i] if outcome_rows and i < len(outcome_rows) else None
        memories.append(extract_session_memory(result, task, outcome_row=row))
    return memories


# ── Graph mutation construction ──────────────────────────────────────────


def _clean_id(raw: str) -> str:
    return raw.strip().replace(" ", "_").replace("/", "_")[:64]


def _find_relevant_node_ids(
    text: str,
    graph: MemoryGraph,
    *,
    max_nodes: int = 5,
    min_text_len: int = 10,
) -> List[str]:
    """Find graph nodes whose text lexically overlaps with *text*."""
    low = text.lower()
    tokens = set(low.split())
    if len(tokens) < 3:
        return []

    scored: List[Tuple[float, str]] = []
    for nid, node in graph.nodes.items():
        nlow = node.text.lower()
        overlap = len(tokens.intersection(nlow.split()))
        if overlap >= 2:
            scored.append((overlap + node.importance, nid))

    scored.sort(key=lambda x: -x[0])
    return [nid for _, nid in scored[:max_nodes]]


def _make_memory_node(memory: SessionMemory) -> Node:
    return Node(
        id=memory.memory_id,
        text=memory.summary_text(),
        node_type="session_memory",
        confidence=memory.outcome_score if memory.outcome_correct is not None else 0.5,
        importance=0.6 if memory.outcome_correct else 0.3,
        metadata={
            "task_id": memory.task_id,
            "outcome_correct": memory.outcome_correct,
            "outcome_score": memory.outcome_score,
            "outcome_source": memory.outcome_source,
            "timestamp": memory.timestamp,
        },
    )


def build_graph_updates(
    memories: List[SessionMemory],
    graph: MemoryGraph,
) -> GraphUpdate:
    """Produce all graph mutations for a batch of session memories."""
    updates = GraphUpdate()

    for memory in memories:
        mem_node = _make_memory_node(memory)
        updates.new_nodes.append(mem_node)
        mem_id = memory.memory_id

        # Link to activated signals (via source_node_id if available)
        linked_signal_ids: set = set()
        for sig_id, src_id in zip(
            memory.activated_signal_ids, memory.activated_signal_source_ids
        ):
            target = _clean_id(src_id) if src_id else None
            if target and target in graph.nodes and target not in linked_signal_ids:
                updates.new_edges.append(Edge(
                    src=mem_id,
                    dst=target,
                    relation="used_signal",
                    strength=0.7,
                    directed=True,
                    metadata={"signal_id": sig_id, "session_memory": mem_id},
                ))
                linked_signal_ids.add(target)

        # Link to related graph nodes by lexical overlap with question + answer
        hay = f"{memory.question} {memory.answer}"
        related = _find_relevant_node_ids(hay, graph)
        for rnid in related:
            if rnid in linked_signal_ids:
                continue
            updates.new_edges.append(Edge(
                src=mem_id,
                dst=rnid,
                relation="related",
                strength=0.5,
                directed=True,
                metadata={"session_memory": mem_id},
            ))

    return updates


# ── Application ──────────────────────────────────────────────────────────


def apply_graph_updates(
    graph: MemoryGraph,
    updates: GraphUpdate,
) -> int:
    """Apply batched mutations to an in-memory graph and return new node count."""
    added = 0
    for node in updates.new_nodes:
        if node.id not in graph.nodes:
            graph.nodes[node.id] = node
            added += 1

    for edge in updates.new_edges:
        if edge.src not in graph.nodes or edge.dst not in graph.nodes:
            continue
        existing = graph.edge_between(edge.src, edge.dst)
        if existing is None:
            graph.edges.append(edge)
        else:
            existing.strength = max(0.0, min(1.0, existing.strength + 0.05))

    for src, dst, delta in updates.edge_strength_updates:
        edge = graph.edge_between(src, dst)
        if edge is not None:
            edge.strength = max(0.0, min(1.0, edge.strength + delta))

    graph._rebuild_index()
    return added


_BACKUP_DIR = Path("data/graph_backups")


def _backup_graph(path: Path) -> Optional[Path]:
    """Copy current graph to a timestamped backup before overwriting."""
    if not path.exists():
        return None
    _BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup = _BACKUP_DIR / f"graph_{stamp}.json"
    import shutil
    shutil.copy2(path, backup)
    return backup


def save_graph_atomically(graph: MemoryGraph, path: str | Path, *, backup: bool = True) -> Optional[Path]:
    """Write graph to disk atomically (tmp + rename).

    Returns the backup path if a backup was made, else None.
    """
    path = Path(path)
    backup_path: Optional[Path] = None
    if backup:
        backup_path = _backup_graph(path)
    tmp = path.with_suffix(".json.tmp")
    graph.save_json(tmp)
    tmp.replace(path)
    return backup_path


# ── Orchestration ────────────────────────────────────────────────────────


def batch_consolidate(
    session_results: Sequence[Tuple[ReasoningResult, Dict[str, Any]]],
    graph_path: str | Path,
    *,
    outcome_rows: Optional[Sequence[Optional[SubstrateOutcomeRow]]] = None,
    dry_run: bool = False,
) -> int:
    """Load graph, extract memories, build + apply updates, save back.

    Args:
        session_results: List of (ReasoningResult, task_dict) pairs.
        graph_path: Path to the graph JSON file.
        outcome_rows: Optional parallel list of SubstrateOutcomeRow.
        dry_run: If True, print what would be done but do not write.

    Returns:
        Number of new session_memory nodes that would be / were added.
    """
    if not session_results:
        return 0

    path = Path(graph_path)
    if not path.exists():
        return 0

    graph = MemoryGraph.load_json(path)
    memories = extract_session_memories(session_results, outcome_rows=outcome_rows)
    updates = build_graph_updates(memories, graph)
    added = apply_graph_updates(graph, updates)

    if dry_run:
        print(f"[dry-run] Would add {added} session_memory nodes to {path}")
        print(f"[dry-run] Would add {len(updates.new_edges)} edges")
        if updates.edge_strength_updates:
            print(f"[dry-run] Would update {len(updates.edge_strength_updates)} edge strengths")
    else:
        backup_path = save_graph_atomically(graph, path)
        if backup_path:
            print(f"Backed up previous graph to {backup_path}")
        print(f"Saved consolidated graph ({len(graph.nodes)} nodes, {len(graph.edges)} edges) to {path}")

    return added
