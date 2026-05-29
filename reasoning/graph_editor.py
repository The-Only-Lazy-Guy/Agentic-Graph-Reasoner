"""Graph editor — turns a ReflectionResult into graph edits and applies them.

This is the deterministic half of the new post-processing pipeline. The
model articulates what it learned (`reasoning/reflection.py`); this module
validates the candidate edits and applies them with proper provenance and
backup. The model never directly mutates the graph.

Two layers of validation before any add_node / add_edge applies:
  1. Evidence node IDs must exist in the live graph (else drop the edit).
  2. add_edge src/dst must both exist in the live graph (else drop).

Soft increments (citation count on existing nodes) bypass these checks
because they target nodes the validator already verified exist.

Provenance: every new node and edge carries `source_session` and
`created_at` in metadata, so we can trace any future regression back to
the originating session.
"""
from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from graph_core import MemoryGraph, Node as GraphNode, Edge as GraphEdge
from reasoning.reflection import ReflectionResult
from reasoning.graph_health import GraphHealthReport, HealthDelta, compute_health, compare_health


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Edit derivation
# ---------------------------------------------------------------------------

def edits_from_reflection(
    reflection: ReflectionResult,
    *,
    graph: MemoryGraph,
) -> List[Dict[str, Any]]:
    """Translate a parsed reflection into graph-mutation operations.

    Validates against the LIVE graph (drops edits referencing non-existent
    nodes). Tags every op with tier and provenance.
    """
    edits: List[Dict[str, Any]] = []
    sid = reflection.session_id

    # ── reinforced existing nodes → soft citation bump ──
    for r in reflection.reinforced_nodes:
        if r.node_id not in graph.nodes:
            continue
        edits.append({
            "op": "increment_meta",
            "node_id": r.node_id,
            "field": "reflection_reinforced_count",
            "delta": 1,
            "session_id": sid,
            "rationale": r.rationale,
            "tier": "soft",
        })

    # ── new facts → claim nodes with derived_from edges to evidence ──
    # Semantic dedupe: replaced legacy text-match ([:100] lowercase) with
    # embedding cosine similarity. Falls back to no-dedupe if index unavailable.
    try:
        from reasoning.semantic_dedupe import build_dedupe_index as _build_idx
        _v1_dedupe = _build_idx(graph)
    except Exception:
        _v1_dedupe = None
    for idx, fact in enumerate(reflection.new_facts):
        if _v1_dedupe is not None:
            _status, _match = _v1_dedupe.classify(fact.text, dup_threshold=0.92)
            if _status == "duplicate" and _match is not None:
                edits.append({
                    "op": "increment_meta",
                    "node_id": _match.node_id,
                    "field": "reflection_reinforced_count",
                    "delta": 1,
                    "session_id": sid,
                    "tier": "soft",
                })
                continue
        valid_evidence = [eid for eid in fact.evidence_node_ids if eid in graph.nodes]
        new_id = f"reflect_claim_{sid}_{idx}"
        edits.append({
            "op": "add_node",
            "node_id": new_id,
            "node_type": "claim",
            "text": fact.text,
            "metadata": {
                "source_session": sid,
                "source": "reflection",
                "confidence": fact.confidence,
                "evidence_node_ids": valid_evidence,
                "evidence_dropped": [
                    eid for eid in fact.evidence_node_ids if eid not in graph.nodes
                ],
                "created_at": _now_iso(),
                "promotion_status": "session_local",
            },
            "tier": "add",
        })
        for eid in valid_evidence:
            edits.append({
                "op": "add_edge",
                "src": new_id,
                "dst": eid,
                "relation": "derived_from",
                "metadata": {"source_session": sid, "source": "reflection"},
                "tier": "add",
            })

    # ── new relationships → add_edge between existing nodes ──
    for rel in reflection.new_relationships:
        if rel.src not in graph.nodes or rel.dst not in graph.nodes:
            continue
        edits.append({
            "op": "add_edge",
            "src": rel.src,
            "dst": rel.dst,
            "relation": rel.relation,
            "metadata": {
                "source_session": sid,
                "source": "reflection",
                "rationale": rel.rationale,
                "created_at": _now_iso(),
            },
            "tier": "add",
        })

    # ── failed approaches → failure_pattern nodes ──
    for idx, fa in enumerate(reflection.failed_approaches):
        new_id = f"reflect_fp_{sid}_{idx}"
        edits.append({
            "op": "add_node",
            "node_id": new_id,
            "node_type": "failure_pattern",
            "text": (
                f"Approach: {fa.approach}\n"
                f"Failure condition: {fa.condition}\n"
                f"Mechanism: {fa.mechanism}"
            ),
            "metadata": {
                "source_session": sid,
                "source": "reflection",
                "attempted_approach": fa.approach,
                "failure_condition": fa.condition,
                "failure_mechanism": fa.mechanism,
                "replacement_suggestion": fa.replacement_suggestion,
                "created_at": _now_iso(),
                "promotion_status": "session_local",
            },
            "tier": "add",
        })
        # If the model suggested a replacement that's an existing node, link it.
        if fa.replacement_suggestion and fa.replacement_suggestion in graph.nodes:
            edits.append({
                "op": "add_edge",
                "src": new_id,
                "dst": fa.replacement_suggestion,
                "relation": "replacement_for",
                "metadata": {"source_session": sid, "source": "reflection"},
                "tier": "add",
            })

    return edits


# Rename old version for backward compat
edits_from_reflection_v1 = edits_from_reflection


def edits_from_reflection_v2(
    reflection: ReflectionResult,
    *,
    graph: MemoryGraph,
    dedupe_index: Any,  # semantic_dedupe.DedupeIndex
) -> List[Dict[str, Any]]:
    """V2: uses semantic dedupe + supports richer edit types.

    Replaces heuristic text-match with embedding cosine similarity.
    Adds update_node, deprecate_node, implementation, worked_example ops.
    Flags ambiguous edits with needs_judge=True for the LLM judge.
    """
    edits: List[Dict[str, Any]] = []
    sid = reflection.session_id

    def _auto_connect(new_id: str, text: str, evidence_ids: List[str]) -> None:
        """Ensure every new node has at least one edge to an existing node.

        First tries explicit evidence IDs. If none are valid (e.g., because
        anonymization made them session-scoped), falls back to the semantically
        nearest existing node from the dedupe index. This prevents orphans.
        """
        valid_ev = [eid for eid in evidence_ids if eid in graph.nodes]
        if valid_ev:
            for eid in valid_ev:
                edits.append({
                    "op": "add_edge",
                    "src": new_id, "dst": eid, "relation": "derived_from",
                    "metadata": {"source_session": sid, "auto_connected": False},
                    "tier": "add",
                })
            return
        # No valid evidence — connect to nearest semantic neighbor
        top1 = dedupe_index.query_topk(text, k=1)
        if top1 and top1[0].node_id in graph.nodes:
            edits.append({
                "op": "add_edge",
                "src": new_id, "dst": top1[0].node_id, "relation": "related",
                "metadata": {
                    "source_session": sid,
                    "auto_connected": True,
                    "semantic_sim": round(top1[0].similarity, 3),
                },
                "tier": "add",
            })

    # ── reinforced existing nodes → soft citation bump (same as v1) ──
    for r in reflection.reinforced_nodes:
        if r.node_id not in graph.nodes:
            continue
        edits.append({
            "op": "increment_meta",
            "node_id": r.node_id,
            "field": "reflection_reinforced_count",
            "delta": 1,
            "session_id": sid,
            "rationale": r.rationale,
            "tier": "soft",
        })

    # ── new facts → semantic dedupe then add_node ──
    for idx, fact in enumerate(reflection.new_facts):
        status, match = dedupe_index.classify(fact.text)
        if status == "duplicate" and match is not None:
            edits.append({
                "op": "increment_meta",
                "node_id": match.node_id,
                "field": "session_cite_count",
                "delta": 1,
                "session_id": sid,
                "tier": "soft",
                "_dedupe_status": "duplicate",
                "_dedupe_sim": match.similarity,
            })
            continue
        new_id = f"reflect_claim_{sid}_{idx}"
        edit = {
            "op": "add_node",
            "node_id": new_id,
            "node_type": "claim",
            "text": fact.text,
            "metadata": {
                "source_session": sid,
                "source": "reflection",
                "confidence": fact.confidence,
                "evidence_node_ids": list(fact.evidence_node_ids),
                "created_at": _now_iso(),
            },
            "tier": "add",
        }
        if status == "ambiguous" and match is not None:
            edit["needs_judge"] = True
            edit["_dedupe_status"] = "ambiguous"
            edit["_dedupe_sim"] = match.similarity
            edit["_dedupe_nearest"] = match.node_id
        edits.append(edit)
        _auto_connect(new_id, fact.text, fact.evidence_node_ids)

    # ── implementations → add_node with type=implementation ──
    for idx, impl in enumerate(reflection.implementations):
        status, match = dedupe_index.classify(impl.text)
        if status == "duplicate" and match is not None:
            continue
        new_id = f"reflect_impl_{sid}_{idx}"
        edit = {
            "op": "add_node",
            "node_id": new_id,
            "node_type": "implementation",
            "text": impl.text,
            "metadata": {
                "source_session": sid, "source": "reflection",
                "language": impl.language,
                "evidence_node_ids": list(impl.evidence_node_ids),
                "created_at": _now_iso(),
            },
            "tier": "add",
        }
        if status == "ambiguous" and match is not None:
            edit["needs_judge"] = True
            edit["_dedupe_nearest"] = match.node_id
        edits.append(edit)
        _auto_connect(new_id, impl.text, impl.evidence_node_ids)

    # ── worked examples → add_node with type=worked_example ──
    for idx, ex in enumerate(reflection.worked_examples):
        status, match = dedupe_index.classify(ex.text)
        if status == "duplicate" and match is not None:
            continue
        new_id = f"reflect_example_{sid}_{idx}"
        edit = {
            "op": "add_node",
            "node_id": new_id,
            "node_type": "worked_example",
            "text": ex.text,
            "metadata": {
                "source_session": sid, "source": "reflection",
                "problem": ex.problem,
                "evidence_node_ids": list(ex.evidence_node_ids),
                "created_at": _now_iso(),
            },
            "tier": "add",
        }
        if status == "ambiguous" and match is not None:
            edit["needs_judge"] = True
            edit["_dedupe_nearest"] = match.node_id
        edits.append(edit)
        _auto_connect(new_id, ex.text, ex.evidence_node_ids)

    # ── node updates → update_node ops ──
    for upd in reflection.updates:
        if upd.node_id not in graph.nodes:
            continue
        edits.append({
            "op": "update_node",
            "node_id": upd.node_id,
            "mode": upd.mode,
            "append_text": upd.text if upd.mode == "append" else "",
            "replace_text": upd.text if upd.mode == "replace" else "",
            "metadata_patch": {
                "last_updated_session": sid,
                "update_source": "reflection",
            },
            "tier": "mutate",
            "needs_judge": True,
        })

    # ── deprecations → deprecate_node ops ──
    for dep in reflection.deprecations:
        if dep.node_id not in graph.nodes:
            continue
        edits.append({
            "op": "deprecate_node",
            "node_id": dep.node_id,
            "reason": dep.reason,
            "successor_id": dep.successor_id,
            "metadata_patch": {
                "deprecated_by_session": sid,
            },
            "tier": "mutate",
            "needs_judge": True,
        })

    # ── failed approaches → failure_pattern nodes (same as v1, with semantic dedupe) ──
    for idx, fa in enumerate(reflection.failed_approaches):
        fa_text = f"Approach: {fa.approach}\nFailure condition: {fa.condition}\nMechanism: {fa.mechanism}"
        status, match = dedupe_index.classify(fa_text)
        if status == "duplicate" and match is not None:
            continue
        new_id = f"reflect_fp_{sid}_{idx}"
        edits.append({
            "op": "add_node",
            "node_id": new_id,
            "node_type": "failure_pattern",
            "text": fa_text,
            "metadata": {
                "source_session": sid, "source": "reflection",
                "attempted_approach": fa.approach,
                "failure_condition": fa.condition,
                "failure_mechanism": fa.mechanism,
                "replacement_suggestion": fa.replacement_suggestion,
                "created_at": _now_iso(),
            },
            "tier": "add",
        })
        _auto_connect(new_id, fa_text, [])

    # ── new relationships (same as v1) ──
    for rel in reflection.new_relationships:
        if rel.src not in graph.nodes or rel.dst not in graph.nodes:
            continue
        edits.append({
            "op": "add_edge",
            "src": rel.src, "dst": rel.dst, "relation": rel.relation,
            "metadata": {"source_session": sid, "source": "reflection", "rationale": rel.rationale},
            "tier": "add",
        })

    return edits


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

def apply_edits(
    graph: MemoryGraph,
    edits: Sequence[Dict[str, Any]],
    *,
    dry_run: bool = True,
    backup_path: Optional[Path] = None,
    allowed_tiers: Sequence[str] = ("soft",),
) -> Dict[str, Any]:
    """Apply edits to the live graph (or report what would happen).

    Same safety contract as Phase-11 apply_graph_edits: dry_run by default,
    backup written when applying, allowed_tiers gates the riskier ops.
    """
    summary: Dict[str, Any] = {"applied": 0, "skipped": 0, "errors": []}

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
            summary["applied"] += 1
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
                node.metadata.setdefault("reflection_session_log", []).append(edit["session_id"])
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

            elif op == "update_node":
                nid = edit["node_id"]
                node = graph.nodes.get(nid)
                if node is None:
                    summary["skipped"] += 1
                    continue
                append_text = edit.get("append_text", "")
                replace_text = edit.get("replace_text", "")
                if replace_text:
                    node.text = replace_text
                elif append_text:
                    node.text = node.text.rstrip() + "\n" + append_text
                for mk, mv in (edit.get("metadata_patch") or {}).items():
                    node.metadata[mk] = mv
                node.metadata.setdefault("edit_log", []).append({
                    "session": edit.get("metadata_patch", {}).get("last_updated_session", ""),
                    "op": "update_node",
                })
                summary["applied"] += 1

            elif op == "deprecate_node":
                nid = edit["node_id"]
                node = graph.nodes.get(nid)
                if node is None:
                    summary["skipped"] += 1
                    continue
                node.metadata["deprecated"] = True
                node.metadata["deprecated_reason"] = edit.get("reason", "")
                for mk, mv in (edit.get("metadata_patch") or {}).items():
                    node.metadata[mk] = mv
                successor = edit.get("successor_id")
                if successor and successor in graph.nodes:
                    node.metadata["successor_id"] = successor
                    graph.edges.append(GraphEdge(
                        src=nid, dst=successor,
                        relation="superseded_by",
                        metadata={"source": "reflection"},
                    ))
                summary["applied"] += 1

            else:
                summary["skipped"] += 1
                summary["errors"].append({"op": op, "error": "unknown op"})
        except Exception as e:
            summary["errors"].append({"op": op, "error": str(e)})

    return summary


# ---------------------------------------------------------------------------
# Inline editor — applies edits with health reward/punishment
# ---------------------------------------------------------------------------

@dataclass
class EditResult:
    """Result of applying one edit with health gating."""
    edit: Dict[str, Any]
    applied: bool
    reward: float = 0.0       # positive = improved health, negative = degraded
    health_delta: Optional[Dict[str, Any]] = None
    reason: str = ""


def apply_edits_inline(
    graph: MemoryGraph,
    edits: Sequence[Dict[str, Any]],
    *,
    allowed_tiers: Sequence[str] = ("soft", "add", "mutate"),
    degradation_threshold: float = -0.02,
    backup_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Inline editor: apply each edit individually, check health after each.

    Reward/punishment per edit:
      - If health_score improves: reward = +delta (positive)
      - If health_score stays neutral: reward = 0
      - If health_score degrades past threshold: REJECT the edit + reward = delta (negative)

    Returns summary with per-edit results and aggregate reward.
    """
    if backup_path is not None:
        try:
            graph.save_json(str(backup_path))
        except Exception:
            pass

    health_before = compute_health(graph)
    results: List[EditResult] = []
    total_reward = 0.0
    applied_count = 0
    rejected_count = 0

    for edit in edits:
        tier = edit.get("tier", "add")
        if tier not in allowed_tiers:
            results.append(EditResult(
                edit=edit, applied=False, reason=f"tier {tier!r} not in allowed_tiers",
            ))
            continue

        # Soft ops (increment_meta) are always safe — skip health check
        if edit.get("op") == "increment_meta":
            one_edit_summary = apply_edits(graph, [edit], dry_run=False, allowed_tiers=allowed_tiers)
            results.append(EditResult(
                edit=edit, applied=True, reward=0.0, reason="soft op, always accepted",
            ))
            applied_count += 1
            continue

        # For non-soft ops: apply tentatively, check health, revert if degraded
        # Snapshot the node/edge state BEFORE this edit
        snap_nodes = dict(graph.nodes)
        snap_edges = list(graph.edges)

        apply_edits(graph, [edit], dry_run=False, allowed_tiers=allowed_tiers)
        health_after = compute_health(graph)
        delta = compare_health(health_before, health_after, degradation_threshold=degradation_threshold)

        if delta.verdict == "degraded":
            # Revert: restore snapshot
            graph.nodes.clear()
            graph.nodes.update(snap_nodes)
            graph.edges.clear()
            graph.edges.extend(snap_edges)
            reward = delta.score_delta  # negative
            results.append(EditResult(
                edit=edit, applied=False, reward=reward,
                health_delta=delta.to_dict(),
                reason=f"rejected: health degraded by {delta.score_delta:.4f}",
            ))
            rejected_count += 1
        else:
            reward = max(0.0, delta.score_delta)
            results.append(EditResult(
                edit=edit, applied=True, reward=reward,
                health_delta=delta.to_dict(),
                reason=f"accepted: {delta.verdict} (score delta={delta.score_delta:.4f})",
            ))
            health_before = health_after  # update baseline for next edit
            applied_count += 1

        total_reward += reward

    return {
        "applied": applied_count,
        "rejected": rejected_count,
        "total_reward": round(total_reward, 4),
        "final_health_score": round(compute_health(graph).health_score, 4),
        "results": [
            {
                "op": r.edit.get("op"),
                "node_id": r.edit.get("node_id", r.edit.get("node_type", "")),
                "applied": r.applied,
                "reward": round(r.reward, 4),
                "reason": r.reason,
            }
            for r in results
        ],
    }


# ---------------------------------------------------------------------------
# Offline editor — batch review with health report
# ---------------------------------------------------------------------------

def apply_edits_offline(
    graph: MemoryGraph,
    edits: Sequence[Dict[str, Any]],
    *,
    allowed_tiers: Sequence[str] = ("soft", "add", "mutate"),
    degradation_threshold: float = -0.02,
    backup_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Offline editor: compute health before/after the FULL batch of edits.

    Unlike inline (per-edit gating), offline applies ALL edits at once,
    then reports the aggregate health impact. If the batch as a whole
    degrades health, it returns the full report but does NOT auto-revert
    (the caller decides).

    Use this for `scripts/process_session.py --apply` where a human
    reviews the health report before committing.
    """
    health_before = compute_health(graph)

    if backup_path is not None:
        try:
            graph.save_json(str(backup_path))
        except Exception:
            pass

    apply_summary = apply_edits(
        graph, edits, dry_run=False, allowed_tiers=allowed_tiers,
    )
    health_after = compute_health(graph)
    delta = compare_health(health_before, health_after, degradation_threshold=degradation_threshold)

    reward = delta.score_delta
    if delta.verdict == "degraded":
        reward_label = "PUNISHMENT"
    elif delta.verdict == "healthy":
        reward_label = "REWARD"
    else:
        reward_label = "NEUTRAL"

    return {
        "apply_summary": apply_summary,
        "health_before": health_before.to_dict(),
        "health_after": health_after.to_dict(),
        "health_delta": delta.to_dict(),
        "reward": round(reward, 4),
        "reward_label": reward_label,
    }
