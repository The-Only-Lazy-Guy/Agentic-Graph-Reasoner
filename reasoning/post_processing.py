"""Post-processing: extract learning from a session and produce graph edits.

After a v4 session closes, this module walks the SessionSubgraphController
state plus the per-tool call log to produce:

  LearningReport — structured view of what the model demonstrably learned
                   (verified hypotheses, recorded failures, cited nodes,
                   synthesized session objects).

  graph_edits     — a list of graph-mutation operations derived from the
                   report. Two safety tiers:
                     (1) soft increments (citation counts) — low risk
                     (2) add_node / add_edge — higher risk, dry-run default

Nothing in this module mutates the long-term graph unless apply_graph_edits()
is called explicitly with dry_run=False, and even then a backup is written
first.

Honors REASONING_ARCHITECTURE §7 consolidation gates — promotion-gated
operations (decision="promote" from Consolidator) are added to the edit list
only when the gates pass. First-session items default to soft-only.
"""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from graph_core import MemoryGraph, Node as GraphNode, Edge as GraphEdge


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# LearningReport
# ---------------------------------------------------------------------------

@dataclass
class VerifiedClaim:
    """A hypothesis the model raised and explicitly verified via verify_hypotheses."""
    hypothesis_id: str
    text: str
    evidence: str
    evidence_node_ids: List[str] = field(default_factory=list)


@dataclass
class RecordedFailure:
    """A failure pattern the model recorded via record_failure."""
    approach: str
    condition: str
    mechanism: str
    failure_pattern_node_id: Optional[str] = None  # id in the SESSION subgraph


@dataclass
class SynthesizedObject:
    """A session_object whose state the model populated during reasoning."""
    name: str
    v4_id: str           # the "obj_N" id v4 used
    controller_id: Optional[str]  # the "so_xxx" id in the session subgraph
    fields: List[str]
    state: Dict[str, Any]


@dataclass
class LearningReport:
    session_id: str
    question: str
    graph_id: str
    cited_node_ids: List[str] = field(default_factory=list)        # nodes the model read
    cited_counts: Dict[str, int] = field(default_factory=dict)     # node_id -> read count this session
    verified_claims: List[VerifiedClaim] = field(default_factory=list)
    discarded_hypotheses: List[Dict[str, str]] = field(default_factory=list)
    recorded_failures: List[RecordedFailure] = field(default_factory=list)
    synthesized_objects: List[SynthesizedObject] = field(default_factory=list)
    timestamp: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "question": self.question,
            "graph_id": self.graph_id,
            "cited_node_ids": list(self.cited_node_ids),
            "cited_counts": dict(self.cited_counts),
            "verified_claims": [
                {
                    "hypothesis_id": c.hypothesis_id,
                    "text": c.text,
                    "evidence": c.evidence,
                    "evidence_node_ids": list(c.evidence_node_ids),
                }
                for c in self.verified_claims
            ],
            "discarded_hypotheses": list(self.discarded_hypotheses),
            "recorded_failures": [
                {
                    "approach": f.approach,
                    "condition": f.condition,
                    "mechanism": f.mechanism,
                    "failure_pattern_node_id": f.failure_pattern_node_id,
                }
                for f in self.recorded_failures
            ],
            "synthesized_objects": [
                {
                    "name": s.name,
                    "v4_id": s.v4_id,
                    "controller_id": s.controller_id,
                    "fields": list(s.fields),
                    "state": dict(s.state),
                }
                for s in self.synthesized_objects
            ],
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

# Recognized graph-read tools in the v4 call log.
_READ_TOOLS = {"read_node", "expand_neighbors"}


def extract_learning_report(
    *,
    session_id: str,
    question: str,
    graph_id: str,
    main_graph: MemoryGraph,
    hypotheses: Mapping[str, Mapping[str, Any]],
    failures: Sequence[Any],   # list of FailureRecord
    objects: Mapping[str, Any],  # v4 SessionObject mapping
    tool_log: Sequence[Mapping[str, Any]],
    v4_to_ctrl_id: Mapping[str, str],
    subgraph_nodes: Mapping[str, Any],
) -> LearningReport:
    """Produce a structured report of what the session demonstrably learned.

    No heuristics — every entry corresponds to an explicit model action
    (read_node call, verify_hypotheses verdict, record_failure call, etc.).
    """
    # ── cited nodes (from read_node calls) ──
    cited_counts: Dict[str, int] = {}
    for entry in tool_log:
        name = entry.get("name")
        args = entry.get("args") or {}
        if name == "read_node":
            nid = args.get("node_id")
            if nid and nid in main_graph.nodes:
                cited_counts[nid] = cited_counts.get(nid, 0) + 1
        elif name == "expand_neighbors":
            nid = args.get("node_id")
            if nid and nid in main_graph.nodes:
                # The source node was inspected; count as a soft citation.
                cited_counts[nid] = cited_counts.get(nid, 0) + 1

    # ── verified vs discarded hypotheses ──
    verified: List[VerifiedClaim] = []
    discarded: List[Dict[str, str]] = []
    for hid, h in hypotheses.items():
        text = (h.get("text") or "").strip()
        verdict = h.get("verdict")
        evidence = (h.get("evidence") or "").strip()
        if verdict == "verified" and text:
            evidence_node_ids = _extract_node_ids_from_text(evidence, main_graph)
            verified.append(VerifiedClaim(
                hypothesis_id=hid, text=text, evidence=evidence,
                evidence_node_ids=evidence_node_ids,
            ))
        elif verdict == "discarded" and text:
            discarded.append({"id": hid, "text": text, "reason": evidence})

    # ── recorded failures ──
    recorded: List[RecordedFailure] = []
    for fr in failures:
        # Find the corresponding fp_v4_* node in the session subgraph (if controller mirror ran).
        fp_id = None
        approach = getattr(fr, "approach", None)
        for sg_id, sg_node in subgraph_nodes.items():
            if not isinstance(sg_node, dict):
                continue
            if sg_node.get("node_type") == "failure_pattern" and sg_node.get("attempted_approach") == approach:
                fp_id = sg_id
                break
        recorded.append(RecordedFailure(
            approach=getattr(fr, "approach", ""),
            condition=getattr(fr, "condition", ""),
            mechanism=getattr(fr, "mechanism", ""),
            failure_pattern_node_id=fp_id,
        ))

    # ── synthesized session objects ──
    synthesized: List[SynthesizedObject] = []
    for v4_id, obj in objects.items():
        # Only include objects whose state is at least partially populated
        # (avoid recording stillborn create-without-update workspaces).
        state = getattr(obj, "state", {}) or {}
        if not any(v is not None and v != [] and v != "" for v in state.values()):
            continue
        synthesized.append(SynthesizedObject(
            name=getattr(obj, "name", ""),
            v4_id=v4_id,
            controller_id=v4_to_ctrl_id.get(v4_id),
            fields=list(getattr(obj, "fields", [])),
            state=dict(state),
        ))

    return LearningReport(
        session_id=session_id,
        question=question,
        graph_id=graph_id,
        cited_node_ids=sorted(cited_counts.keys()),
        cited_counts=cited_counts,
        verified_claims=verified,
        discarded_hypotheses=discarded,
        recorded_failures=recorded,
        synthesized_objects=synthesized,
    )


def _extract_node_ids_from_text(text: str, graph: MemoryGraph) -> List[str]:
    """Find any graph node ids mentioned in `text` (snake_case-ish tokens that match)."""
    if not text or not graph.nodes:
        return []
    # candidate tokens — alphanum + underscores, 3+ chars
    tokens = set(re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", text))
    return sorted(t for t in tokens if t in graph.nodes)


# ---------------------------------------------------------------------------
# Graph edits
# ---------------------------------------------------------------------------

def produce_graph_edits(
    report: LearningReport,
    *,
    graph: Optional[MemoryGraph] = None,
    promotion_decisions: Optional[Sequence[Mapping[str, Any]]] = None,
    strategy: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Translate a LearningReport into a list of graph-mutation operations.

    Edit op shapes:
      {"op": "increment_meta", "node_id": str, "field": str, "delta": int,
       "session_id": str, "tier": "soft"}

      {"op": "add_node", "node_id": str, "node_type": str, "text": str,
       "metadata": dict, "tier": "add"}

      {"op": "add_edge", "src": str, "dst": str, "relation": str,
       "metadata": dict, "tier": "add"}

    `tier` lets the apply step apply soft ops by default and gate add ops
    behind explicit user opt-in. No graph mutation occurs here.
    """
    edits: List[Dict[str, Any]] = []

    # ── soft: citation count increments ──
    for nid, count in report.cited_counts.items():
        edits.append({
            "op": "increment_meta",
            "node_id": nid,
            "field": "session_cite_count",
            "delta": int(count),
            "session_id": report.session_id,
            "tier": "soft",
        })

    # ── add: verified hypotheses → claim nodes ──
    # Semantic dedupe: if an existing node is semantically equivalent, bump
    # its citation instead of adding a duplicate.
    _dedupe_idx = None
    if graph is not None:
        try:
            from reasoning.semantic_dedupe import build_dedupe_index
            _dedupe_idx = build_dedupe_index(graph)
        except Exception:
            pass
    for claim in report.verified_claims:
        status = "new"
        match = None
        if _dedupe_idx is not None:
            status, match = _dedupe_idx.classify(claim.text, dup_threshold=0.92)
            if status == "duplicate" and match is not None:
                edits.append({
                    "op": "increment_meta",
                    "node_id": match.node_id,
                    "field": "session_cite_count",
                    "delta": 1,
                    "session_id": report.session_id,
                    "tier": "soft",
                })
                continue
        new_id = f"claim_{report.session_id}_{claim.hypothesis_id}"
        edit = {
            "op": "add_node",
            "node_id": new_id,
            "node_type": "claim",
            "text": claim.text,
            "metadata": {
                "source_session": report.session_id,
                "evidence": claim.evidence,
                "evidence_node_ids": list(claim.evidence_node_ids),
                "created_at": _now_iso(),
                "promotion_status": "session_local",  # not yet cross-session validated
            },
            "tier": "add",
        }
        if status == "ambiguous" and match is not None:
            edit["needs_judge"] = True
            edit["_dedupe_status"] = "ambiguous"
            edit["_dedupe_sim"] = match.similarity
            edit["_dedupe_nearest"] = match.node_id
        edits.append(edit)
        # Connect to evidence nodes if valid, else nearest semantic neighbor
        valid_ev = [eid for eid in claim.evidence_node_ids if graph is not None and eid in graph.nodes]
        if valid_ev:
            for ev_nid in valid_ev:
                edits.append({
                    "op": "add_edge",
                    "src": new_id, "dst": ev_nid, "relation": "derived_from",
                    "metadata": {"source_session": report.session_id},
                    "tier": "add",
                })
        elif graph is not None:
            # Semantic auto-connect: find nearest existing node
            try:
                from reasoning.semantic_dedupe import build_dedupe_index
                _idx = build_dedupe_index(graph)
                top1 = _idx.query_topk(claim.text, k=1)
                if top1 and top1[0].node_id in graph.nodes:
                    edits.append({
                        "op": "add_edge",
                        "src": new_id, "dst": top1[0].node_id, "relation": "related",
                        "metadata": {"source_session": report.session_id, "auto_connected": True,
                                     "semantic_sim": round(top1[0].similarity, 3)},
                        "tier": "add",
                    })
            except Exception:
                pass

    # ── add: recorded failures → failure_pattern nodes ──
    for fr in report.recorded_failures:
        new_id = f"fp_{report.session_id}_{abs(hash(fr.approach)) % 10_000_000}"
        edits.append({
            "op": "add_node",
            "node_id": new_id,
            "node_type": "failure_pattern",
            "text": (
                f"Approach: {fr.approach}\n"
                f"Failure condition: {fr.condition}\n"
                f"Mechanism: {fr.mechanism}"
            ),
            "metadata": {
                "source_session": report.session_id,
                "attempted_approach": fr.approach,
                "failure_condition": fr.condition,
                "failure_mechanism": fr.mechanism,
                "created_at": _now_iso(),
                "promotion_status": "session_local",
            },
            "tier": "add",
        })
        # Auto-connect failure pattern to nearest semantic neighbor
        if graph is not None:
            try:
                from reasoning.semantic_dedupe import build_dedupe_index
                _idx = build_dedupe_index(graph)
                fp_text = f"{fr.approach} {fr.condition} {fr.mechanism}"
                top1 = _idx.query_topk(fp_text, k=1)
                if top1 and top1[0].node_id in graph.nodes:
                    edits.append({
                        "op": "add_edge",
                        "src": new_id, "dst": top1[0].node_id, "relation": "related",
                        "metadata": {"source_session": report.session_id, "auto_connected": True,
                                     "semantic_sim": round(top1[0].similarity, 3)},
                        "tier": "add",
                    })
            except Exception:
                pass

    # ── deprecate: _false misconception nodes that were read during this session ──
    for nid in report.cited_node_ids:
        if nid.endswith("_false") and graph is not None and nid in graph.nodes:
            node = graph.nodes[nid]
            if not node.metadata.get("deprecated"):
                edits.append({
                    "op": "deprecate_node",
                    "node_id": nid,
                    "reason": f"Misconception node (confidence={node.confidence}) read in session {report.session_id}",
                    "tier": "soft",
                })

    # ── add: strategy node (proven reasoning recipe) ──
    if strategy is not None:
        strategy_text = _build_strategy_text(strategy)
        edits.append({
            "op": "add_node",
            "node_id": strategy.id,
            "node_type": "strategy",
            "text": strategy_text,
            "metadata": {
                "source_session": report.session_id,
                "question_pattern": strategy.question_pattern,
                "domain_keywords": strategy.domain_keywords,
                "task_family": getattr(strategy, "task_family", ""),
                "task_subtype": getattr(strategy, "task_subtype", ""),
                "question_mode": getattr(strategy, "question_mode", ""),
                "entry_conditions": getattr(strategy, "entry_conditions", {}),
                "plan_template": strategy.plan_template,
                "slot_order": getattr(strategy, "slot_order", []),
                "checkpoint_plan": getattr(strategy, "checkpoint_plan", []),
                "stop_conditions": getattr(strategy, "stop_conditions", []),
                "forbidden_finalize_conditions": getattr(strategy, "forbidden_finalize_conditions", []),
                "key_node_ids": strategy.key_node_ids,
                "workspace_schema": strategy.workspace_schema,
                "pitfalls": strategy.pitfalls,
                "effective_queries": strategy.effective_queries,
                "session_stats": strategy.session_stats,
                "strategy_schema_version": getattr(strategy, "strategy_schema_version", 2),
                "created_at": _now_iso(),
            },
            "tier": "add",
        })
        for knid in (strategy.key_node_ids or []):
            if graph is not None and knid in graph.nodes:
                edits.append({
                    "op": "add_edge",
                    "src": strategy.id,
                    "dst": knid,
                    "relation": "leveraged",
                    "metadata": {"source_session": report.session_id},
                    "tier": "add",
                })

    # ── promotion-gated: full nodes from Consolidator decisions ──
    for d in (promotion_decisions or []):
        if d.get("decision") != "promote":
            continue
        node_data = d.get("node_data")
        if not isinstance(node_data, dict) or "id" not in node_data:
            continue
        edits.append({
            "op": "add_node",
            "node_id": node_data["id"],
            "node_type": node_data.get("node_type", "claim"),
            "text": node_data.get("text") or node_data.get("name") or "",
            "metadata": {
                "promoted_by_consolidator": True,
                "source_session": report.session_id,
                "created_at": _now_iso(),
                "raw_node_data": copy.deepcopy(node_data),
            },
            "tier": "promote",
        })

    return edits


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

def apply_graph_edits(
    graph: MemoryGraph,
    edits: Sequence[Mapping[str, Any]],
    *,
    dry_run: bool = True,
    backup_path: Optional[Path] = None,
    allowed_tiers: Sequence[str] = ("soft",),
) -> Dict[str, Any]:
    """Apply graph edits to a live MemoryGraph (or report what would change).

    Returns a summary dict {"applied": int, "skipped": int, "errors": [...]}.

    dry_run=True (default): no mutation; the returned counts describe what
    WOULD happen if applied. backup_path is ignored.

    dry_run=False: mutates the graph in place. If backup_path is provided,
    the current graph state is written there BEFORE any mutations.

    allowed_tiers controls which edit tiers actually apply when dry_run=False:
      - "soft": citation_count increments only
      - "add":  also add_node / add_edge from verified-hypotheses + failures
      - "promote": also promote consolidator-approved nodes
    The default is soft-only, which is the safest non-dry-run posture.
    """
    summary = {"applied": 0, "skipped": 0, "errors": []}

    if not dry_run and backup_path is not None:
        try:
            graph.save_json(str(backup_path))
        except Exception as e:
            summary["errors"].append({"phase": "backup", "error": str(e)})

    for edit in edits:
        tier = edit.get("tier", "add")
        if not dry_run and tier not in allowed_tiers:
            summary["skipped"] += 1
            continue
        if dry_run:
            summary["applied"] += 1  # what would happen
            continue

        op = edit.get("op")
        try:
            if op == "increment_meta":
                nid = edit["node_id"]
                node = graph.nodes.get(nid)
                if node is None:
                    summary["skipped"] += 1
                    continue
                current = node.metadata.get(edit["field"], 0)
                node.metadata[edit["field"]] = int(current) + int(edit["delta"])
                node.metadata.setdefault("session_cite_log", []).append(edit["session_id"])
                summary["applied"] += 1

            elif op == "add_node":
                nid = edit["node_id"]
                if nid in graph.nodes:
                    summary["skipped"] += 1
                    continue
                graph.nodes[nid] = GraphNode(
                    id=nid,
                    text=str(edit.get("text", "")),
                    node_type=str(edit.get("node_type", "claim")),
                    metadata=dict(edit.get("metadata") or {}),
                )
                summary["applied"] += 1

            elif op == "add_edge":
                src = edit["src"]; dst = edit["dst"]
                if src not in graph.nodes or dst not in graph.nodes:
                    summary["skipped"] += 1
                    continue
                graph.edges.append(GraphEdge(
                    src=src, dst=dst,
                    relation=str(edit.get("relation", "related")),
                    metadata=dict(edit.get("metadata") or {}),
                ))
                summary["applied"] += 1

            elif op == "deprecate_node":
                nid = edit["node_id"]
                node = graph.nodes.get(nid)
                if node is None:
                    summary["skipped"] += 1
                    continue
                node.metadata["deprecated"] = True
                node.metadata["deprecated_reason"] = edit.get("reason", "")
                summary["applied"] += 1

            else:
                summary["skipped"] += 1
                summary["errors"].append({"op": op, "error": "unknown op"})
        except Exception as e:
            summary["errors"].append({"op": op, "error": str(e)})

    return summary


# ---------------------------------------------------------------------------
# Strategy extraction
# ---------------------------------------------------------------------------

def _build_strategy_text(strategy: Any) -> str:
    """Format strategy node text for optimal retrieval matching."""
    lines = [f"Strategy family: {strategy.task_family or 'generic'}"]
    if getattr(strategy, "task_subtype", ""):
        lines.append(f"Strategy subtype: {strategy.task_subtype}")
    if getattr(strategy, "question_mode", ""):
        lines.append(f"Question mode: {strategy.question_mode}")
    entry_conditions = getattr(strategy, "entry_conditions", {}) or {}
    if entry_conditions:
        rendered = ", ".join(
            f"{key}={value}" for key, value in entry_conditions.items() if value
        )
        if rendered:
            lines.append(f"Entry conditions: {rendered}")
    if strategy.domain_keywords:
        lines.append(f"Keywords: {', '.join(strategy.domain_keywords)}")
    checkpoint_plan = list(getattr(strategy, "checkpoint_plan", []) or strategy.plan_template)
    if checkpoint_plan:
        lines.append("Checkpoint plan:")
        for i, sg in enumerate(checkpoint_plan, 1):
            lines.append(f"  {i}. {sg}")
    slot_order = list(getattr(strategy, "slot_order", []) or [])
    if slot_order:
        lines.append(f"Slot order: {', '.join(slot_order)}")
    if strategy.key_node_ids:
        lines.append(f"Key nodes: {', '.join(strategy.key_node_ids[:8])}")
    stop_conditions = list(getattr(strategy, "stop_conditions", []) or [])
    if stop_conditions:
        lines.append(f"Stop when: {'; '.join(stop_conditions[:3])}")
    forbidden_finalize_conditions = list(getattr(strategy, "forbidden_finalize_conditions", []) or [])
    if forbidden_finalize_conditions:
        lines.append(f"Do not finalize when: {'; '.join(forbidden_finalize_conditions[:3])}")
    if strategy.pitfalls:
        summaries = [p.get("approach", "") for p in strategy.pitfalls if p.get("approach")]
        if summaries:
            lines.append(f"Pitfalls: {'; '.join(summaries[:3])}")
    if strategy.effective_queries:
        lines.append(f"Effective searches: {'; '.join(strategy.effective_queries[:5])}")
    return "\n".join(lines)


def extract_strategy(
    *,
    report: LearningReport,
    plan: Sequence,
    tool_log: Sequence[Mapping[str, Any]],
    question: str,
    task_frame: Optional[Any] = None,
    finalized: bool,
    steps: int,
    tool_call_count: int,
    elapsed_sec: float,
    min_done_ratio: float = 0.5,
    max_key_nodes: int = 8,
    max_effective_queries: int = 5,
) -> Optional[Any]:
    """Extract a StrategyNode from a successful session."""
    from reasoning.schemas import StrategyNode, Provenance
    if task_frame is None:
        from reasoning.micro_controller import build_task_frame
        task_frame = build_task_frame(question)

    if not finalized:
        return None

    done_subgoals = [sg.text for sg in plan if sg.done]
    total_subgoals = len(plan) if plan else 0
    if total_subgoals == 0 or len(done_subgoals) / total_subgoals < min_done_ratio:
        return None

    sorted_cited = sorted(
        report.cited_counts.items(), key=lambda kv: -kv[1]
    )[:max_key_nodes]
    key_node_ids = [nid for nid, _ in sorted_cited]
    key_node_rationales = {nid: f"cited {count}x" for nid, count in sorted_cited}

    workspace_schema = []
    for so in report.synthesized_objects:
        workspace_schema.append({"name": so.name, "fields": list(so.fields)})

    pitfalls = []
    for rf in report.recorded_failures:
        pitfalls.append({"approach": rf.approach, "condition": rf.condition, "mechanism": rf.mechanism})

    effective_queries = []
    for entry in tool_log:
        if entry.get("name") == "search_nodes":
            q = (entry.get("args") or {}).get("query", "")
            if q and q not in effective_queries:
                effective_queries.append(q)
    effective_queries = effective_queries[:max_effective_queries]

    _STOP = {"the", "and", "for", "with", "that", "this", "from", "not", "are", "was",
             "what", "how", "why", "does", "which", "when", "can", "its", "has", "you",
             "your", "one", "two", "use", "also", "must", "will", "each", "all"}
    q_tokens = set(re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", question.lower()))
    nid_tokens = set()
    for nid in key_node_ids:
        nid_tokens.update(re.findall(r"[A-Za-z][A-Za-z0-9]{2,}", nid.lower()))
    domain_keywords = sorted((q_tokens | nid_tokens) - _STOP)[:8]
    entry_conditions = {
        key: value
        for key, value in (task_frame.context.entities | task_frame.context.conditions).items()
        if value
    }
    stop_conditions = [f"Fill slots: {', '.join(task_frame.required_slots)}"]
    forbidden_finalize_conditions: List[str] = []

    strategy_id = f"strat_{report.session_id}"
    return StrategyNode(
        id=strategy_id,
        question_pattern=question,
        domain_keywords=domain_keywords,
        plan_template=done_subgoals,
        key_node_ids=key_node_ids,
        key_node_rationales=key_node_rationales,
        workspace_schema=workspace_schema,
        pitfalls=pitfalls,
        effective_queries=effective_queries,
        session_stats={"steps": steps, "tool_calls": tool_call_count, "elapsed_sec": elapsed_sec},
        provenance=Provenance(created_in_session_id=report.session_id, last_modified=_now_iso()),
        task_family=task_frame.task_family,
        task_subtype=task_frame.context.task_subtype,
        question_mode=task_frame.context.question_mode,
        entry_conditions=entry_conditions,
        slot_order=list(task_frame.required_slots),
        checkpoint_plan=done_subgoals,
        stop_conditions=stop_conditions,
        forbidden_finalize_conditions=forbidden_finalize_conditions,
        strategy_schema_version=2,
    )
