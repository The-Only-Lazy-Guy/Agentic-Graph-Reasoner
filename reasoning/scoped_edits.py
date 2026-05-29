"""Scoped graph-edit patches and validation.

The raw Phase-11 edit list says *what* could mutate. This module adds the
control layer needed for research-grade graph learning: scope, evidence,
risk, and a deterministic validation verdict before anything is trusted.

No function here mutates the graph.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

from graph_core import MemoryGraph
from reasoning.lexical_matching import content_tokens, lexical_overlap, normalize_text


VALIDATION_ACCEPT = "accept"
VALIDATION_SOFT_ONLY = "soft_only"
VALIDATION_NEEDS_REVIEW = "needs_review"
VALIDATION_REJECT = "reject"
APPLYABLE_STATUSES = frozenset({VALIDATION_ACCEPT, VALIDATION_SOFT_ONLY})

_NEGATION_TERMS = {
    "cannot",
    "can't",
    "not",
    "never",
    "invalid",
    "incorrect",
    "unsafe",
    "false",
    "fails",
    "failure",
}
_AFFIRMATION_TERMS = {
    "can",
    "valid",
    "correct",
    "safe",
    "true",
    "works",
    "suitable",
    "guaranteed",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stable_patch_id(edit: Mapping[str, Any], index: int) -> str:
    op = str(edit.get("op", "edit"))
    target = (
        edit.get("node_id")
        or f"{edit.get('src', '')}->{edit.get('dst', '')}"
        or str(index)
    )
    return f"patch_{index:04d}_{_slug(op)}_{_slug(str(target))[:80]}"


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", str(text or "").strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "unknown"


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return sorted(value)
    return [value]


def _unique_strings(values: Iterable[Any]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _risk_for(edit: Mapping[str, Any], patch_type: str) -> str:
    tier = str(edit.get("tier", "") or "").lower()
    node_type = str(edit.get("node_type", "") or "").lower()
    if patch_type == "reinforce_existing":
        return "soft"
    if patch_type in {"deprecate_fact", "add_control_rule"}:
        return "high"
    if patch_type == "add_relation":
        return "low" if tier == "soft" else "medium"
    if node_type in {"strategy", "solved_subgoal", "reasoning_atom", "claim"}:
        return "medium"
    if node_type == "failure_pattern":
        return "low"
    if tier == "promote":
        return "medium"
    return "medium"


def _patch_type_for(edit: Mapping[str, Any]) -> str:
    op = str(edit.get("op", "") or "")
    node_type = str(edit.get("node_type", "") or "")
    if op == "increment_meta":
        return "reinforce_existing"
    if op == "add_edge":
        return "add_relation"
    if op == "deprecate_node":
        return "deprecate_fact"
    if op == "add_node":
        return {
            "claim": "add_fact",
            "fact": "add_fact",
            "failure_pattern": "add_failure_pattern",
            "strategy": "add_strategy",
            "solved_subgoal": "add_solved_subgoal",
            "reasoning_atom": "add_reasoning_atom",
            "control_rule": "add_control_rule",
        }.get(node_type, f"add_{node_type or 'node'}")
    return f"unknown_{op or 'edit'}"


def _scope_from_edit(
    edit: Mapping[str, Any],
    *,
    question: str = "",
    learning_report: Optional[Mapping[str, Any]] = None,
    task_frame: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    metadata = edit.get("metadata") if isinstance(edit.get("metadata"), Mapping) else {}
    scope: Dict[str, Any] = {
        "question": question or (learning_report or {}).get("question", ""),
        "task_family": metadata.get("task_family", ""),
        "task_subtype": metadata.get("task_subtype", ""),
        "question_mode": metadata.get("question_mode", ""),
        "input_conditions": dict(metadata.get("input_conditions") or metadata.get("entry_conditions") or {}),
        "required_slots": list(metadata.get("required_slots") or metadata.get("slot_order") or []),
    }
    if isinstance(task_frame, Mapping):
        scope["task_family"] = scope["task_family"] or str(task_frame.get("task_family", "") or "")
        context = task_frame.get("context") if isinstance(task_frame.get("context"), Mapping) else {}
        scope["task_subtype"] = scope["task_subtype"] or str(context.get("task_subtype", "") or "")
        scope["question_mode"] = scope["question_mode"] or str(context.get("question_mode", "") or "")
    return {k: v for k, v in scope.items() if v not in ("", [], {}, None)}


def _valid_when_for(edit: Mapping[str, Any], scope: Mapping[str, Any]) -> List[str]:
    metadata = edit.get("metadata") if isinstance(edit.get("metadata"), Mapping) else {}
    explicit = _unique_strings(metadata.get("valid_when") or [])
    if explicit:
        return explicit

    out: List[str] = []
    task_family = scope.get("task_family")
    task_subtype = scope.get("task_subtype")
    question_mode = scope.get("question_mode")
    if task_family:
        out.append(f"task_family == {task_family}")
    if task_subtype:
        out.append(f"task_subtype == {task_subtype}")
    if question_mode:
        out.append(f"question_mode == {question_mode}")
    if metadata.get("question_pattern"):
        out.append("question matches the stored strategy pattern closely")
    if not out:
        out.append("same semantic task and compatible input conditions")
    return out


def _invalid_when_for(edit: Mapping[str, Any]) -> List[str]:
    metadata = edit.get("metadata") if isinstance(edit.get("metadata"), Mapping) else {}
    explicit = _unique_strings(metadata.get("invalid_when") or [])
    if explicit:
        return explicit

    node_type = str(edit.get("node_type", "") or "")
    if node_type == "strategy":
        return [
            "required output slots differ from the stored slot order",
            "question constraints conflict with strategy entry conditions",
            "key evidence nodes are low relevance or deprecated",
        ]
    if node_type == "solved_subgoal":
        return [
            "asking about a specific instance excluded by the stored capsule",
            "input conditions differ from the solved subgoal conditions",
            "requested proof depth exceeds the stored output slots",
        ]
    if node_type == "control_rule":
        return [
            "task family or required slot policy changed",
            "rule would suppress needed verification for a high-risk task",
        ]
    if node_type == "reasoning_atom":
        return [
            "dependencies are absent or incompatible",
            "the atom only supports a nearby but not equivalent subgoal",
        ]
    return ["new evidence contradicts this edit or its support nodes"]


def _evidence_ids_for(edit: Mapping[str, Any]) -> List[str]:
    metadata = edit.get("metadata") if isinstance(edit.get("metadata"), Mapping) else {}
    values: List[Any] = []
    values.extend(_as_list(metadata.get("evidence_node_ids")))
    values.extend(_as_list(metadata.get("supporting_node_ids")))
    values.extend(_as_list(metadata.get("key_node_ids")))
    if edit.get("op") == "add_edge":
        values.extend([edit.get("src"), edit.get("dst")])
    if edit.get("op") == "increment_meta":
        values.append(edit.get("node_id"))
    return _unique_strings(values)


def _affected_ids_for(edit: Mapping[str, Any]) -> List[str]:
    values: List[Any] = []
    if edit.get("node_id"):
        values.append(edit.get("node_id"))
    if edit.get("src"):
        values.append(edit.get("src"))
    if edit.get("dst"):
        values.append(edit.get("dst"))
    metadata = edit.get("metadata") if isinstance(edit.get("metadata"), Mapping) else {}
    values.extend(_as_list(metadata.get("affected_node_ids")))
    return _unique_strings(values)


def _source_session_for(edit: Mapping[str, Any], learning_report: Optional[Mapping[str, Any]]) -> str:
    metadata = edit.get("metadata") if isinstance(edit.get("metadata"), Mapping) else {}
    return str(
        metadata.get("source_session")
        or edit.get("session_id")
        or (learning_report or {}).get("session_id")
        or ""
    )


@dataclass
class PatchValidationResult:
    status: str
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    support_score: float = 0.0
    duplicate_of: Optional[str] = None
    conflicts_with: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "support_score": self.support_score,
            "duplicate_of": self.duplicate_of,
            "conflicts_with": list(self.conflicts_with),
        }


@dataclass
class GraphEditPatch:
    patch_id: str
    patch_type: str
    source_op: str
    target_id: str
    text: str
    scope: Dict[str, Any] = field(default_factory=dict)
    valid_when: List[str] = field(default_factory=list)
    invalid_when: List[str] = field(default_factory=list)
    evidence_node_ids: List[str] = field(default_factory=list)
    affected_node_ids: List[str] = field(default_factory=list)
    source_session: str = ""
    confidence: float = 0.5
    risk_level: str = "medium"
    payload: Dict[str, Any] = field(default_factory=dict)
    raw_edit: Dict[str, Any] = field(default_factory=dict)
    validation: PatchValidationResult = field(
        default_factory=lambda: PatchValidationResult(status=VALIDATION_NEEDS_REVIEW)
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patch_id": self.patch_id,
            "patch_type": self.patch_type,
            "source_op": self.source_op,
            "target_id": self.target_id,
            "text": self.text,
            "scope": dict(self.scope),
            "valid_when": list(self.valid_when),
            "invalid_when": list(self.invalid_when),
            "evidence_node_ids": list(self.evidence_node_ids),
            "affected_node_ids": list(self.affected_node_ids),
            "source_session": self.source_session,
            "confidence": self.confidence,
            "risk_level": self.risk_level,
            "payload": dict(self.payload),
            "raw_edit": dict(self.raw_edit),
            "validation": self.validation.to_dict(),
        }


def patches_from_graph_edits(
    edits: Sequence[Mapping[str, Any]],
    *,
    graph: Optional[MemoryGraph] = None,
    learning_report: Optional[Mapping[str, Any]] = None,
    question: str = "",
    task_frame: Optional[Mapping[str, Any]] = None,
) -> List[GraphEditPatch]:
    """Annotate raw graph edits with scope, evidence, and risk metadata."""
    patches: List[GraphEditPatch] = []
    for index, edit in enumerate(edits):
        if not isinstance(edit, Mapping):
            continue
        op = str(edit.get("op", "") or "")
        patch_type = _patch_type_for(edit)
        metadata = edit.get("metadata") if isinstance(edit.get("metadata"), Mapping) else {}
        target_id = str(
            edit.get("node_id")
            or f"{edit.get('src', '')}->{edit.get('dst', '')}"
            or _stable_patch_id(edit, index)
        )
        text = str(edit.get("text") or metadata.get("summary") or metadata.get("claim") or "")
        if not text and op == "increment_meta" and graph is not None:
            node = graph.nodes.get(str(edit.get("node_id", "")))
            text = node.text if node is not None else ""
        if not text and op == "add_edge":
            text = f"{edit.get('src', '')} --{edit.get('relation', 'related')}--> {edit.get('dst', '')}"
        scope = _scope_from_edit(edit, question=question, learning_report=learning_report, task_frame=task_frame)
        confidence = float(metadata.get("confidence", edit.get("confidence", 0.5)) or 0.5)
        patch = GraphEditPatch(
            patch_id=_stable_patch_id(edit, index),
            patch_type=patch_type,
            source_op=op,
            target_id=target_id,
            text=text,
            scope=scope,
            valid_when=_valid_when_for(edit, scope),
            invalid_when=_invalid_when_for(edit),
            evidence_node_ids=_evidence_ids_for(edit),
            affected_node_ids=_affected_ids_for(edit),
            source_session=_source_session_for(edit, learning_report),
            confidence=max(0.0, min(1.0, confidence)),
            risk_level=_risk_for(edit, patch_type),
            payload={
                "node_type": edit.get("node_type", ""),
                "relation": edit.get("relation", ""),
                "tier": edit.get("tier", ""),
                "field": edit.get("field", ""),
                "delta": edit.get("delta", None),
                "metadata": dict(metadata),
            },
            raw_edit=dict(edit),
        )
        patches.append(patch)
    return patches


def validate_patches(patches: Sequence[GraphEditPatch], graph: MemoryGraph) -> List[GraphEditPatch]:
    """Validate all patches and return the same patch objects with verdicts set."""
    proposed_node_ids = {
        patch.target_id
        for patch in patches
        if patch.source_op == "add_node" and patch.target_id
    }
    available_node_ids = set(graph.nodes) | proposed_node_ids
    for patch in patches:
        patch.validation = validate_patch(patch, graph, available_node_ids=available_node_ids)
    _inherit_parent_validation(patches)
    return list(patches)


def approved_raw_edits_from_patches(
    patches: Sequence[GraphEditPatch],
    *,
    allowed_statuses: Iterable[str] = APPLYABLE_STATUSES,
) -> List[Dict[str, Any]]:
    """Return raw edits whose scoped patch status is safe to apply."""
    allowed = set(allowed_statuses)
    return [
        dict(patch.raw_edit)
        for patch in patches
        if patch.validation.status in allowed
    ]


def validate_patch(
    patch: GraphEditPatch,
    graph: MemoryGraph,
    *,
    available_node_ids: Optional[Set[str]] = None,
) -> PatchValidationResult:
    """Deterministic safety check for one scoped edit patch."""
    reasons: List[str] = []
    warnings: List[str] = []
    conflicts: List[str] = []
    duplicate_of: Optional[str] = None
    status = VALIDATION_ACCEPT
    known_ids = available_node_ids if available_node_ids is not None else set(graph.nodes)

    op = patch.source_op
    raw = patch.raw_edit

    if op == "increment_meta":
        if patch.target_id not in graph.nodes:
            return PatchValidationResult(
                status=VALIDATION_REJECT,
                reasons=["target node does not exist for soft reinforcement"],
            )
        relevance_score, relevance_warning = _reinforcement_relevance(patch, graph)
        if relevance_warning:
            return PatchValidationResult(
                status=VALIDATION_NEEDS_REVIEW,
                reasons=["soft reinforcement target is not relevant enough to the current question"],
                warnings=[relevance_warning],
                support_score=round(relevance_score, 3),
            )
        return PatchValidationResult(
            status=VALIDATION_SOFT_ONLY,
            reasons=["soft reinforcement only; no semantic graph change proposed"],
            support_score=round(relevance_score if relevance_score > 0 else 1.0, 3),
        )

    if op == "add_edge":
        src = str(raw.get("src", "") or "")
        dst = str(raw.get("dst", "") or "")
        if not src or not dst:
            return PatchValidationResult(
                status=VALIDATION_REJECT,
                reasons=["edge edit missing src or dst"],
            )
        missing = [nid for nid in (src, dst) if nid not in known_ids]
        if missing:
            return PatchValidationResult(
                status=VALIDATION_REJECT,
                reasons=[f"edge endpoint missing from graph/pre-edit batch: {', '.join(missing)}"],
            )
        if src in graph.nodes and dst in graph.nodes and graph.directed_edge_between(src, dst) is not None:
            duplicate_of = f"{src}->{dst}"
            status = VALIDATION_SOFT_ONLY
            reasons.append("relation already exists")
        return PatchValidationResult(
            status=status,
            reasons=reasons,
            warnings=warnings,
            support_score=1.0 if status != VALIDATION_REJECT else 0.0,
            duplicate_of=duplicate_of,
            conflicts_with=conflicts,
        )

    if op == "deprecate_node":
        if patch.target_id not in graph.nodes:
            return PatchValidationResult(
                status=VALIDATION_REJECT,
                reasons=["cannot deprecate missing node"],
            )
        reasons.append("deprecation requires review before persistent mutation")
        return PatchValidationResult(
            status=VALIDATION_NEEDS_REVIEW,
            reasons=reasons,
            support_score=0.5,
        )

    if op != "add_node":
        return PatchValidationResult(
            status=VALIDATION_NEEDS_REVIEW,
            reasons=[f"unknown edit op: {op or 'missing'}"],
        )

    if patch.target_id in graph.nodes:
        duplicate_of = patch.target_id
        return PatchValidationResult(
            status=VALIDATION_SOFT_ONLY,
            reasons=["node id already exists; treat as reinforcement/update candidate"],
            support_score=1.0,
            duplicate_of=duplicate_of,
        )

    support_score, support_warnings = _support_score(patch, graph)
    warnings.extend(support_warnings)

    if patch.risk_level in {"medium", "high"} and not patch.evidence_node_ids:
        status = VALIDATION_NEEDS_REVIEW
        reasons.append("medium/high-risk add has no explicit evidence nodes")

    if patch.evidence_node_ids and support_score < _min_support_score(patch):
        status = VALIDATION_NEEDS_REVIEW
        reasons.append(f"evidence support score too low: {support_score:.2f}")

    if patch.patch_type == "add_fact":
        unsupported_slots = _unsupported_claim_slots(patch, graph)
        if unsupported_slots:
            status = VALIDATION_NEEDS_REVIEW
            reasons.append("claim contains slots unsupported by evidence: " + ", ".join(unsupported_slots))

    if patch.patch_type == "add_strategy":
        noisy = _low_relevance_evidence_ids(patch, graph)
        if noisy:
            warnings.append("low_relevance_evidence:" + ",".join(noisy[:8]))
            status = VALIDATION_NEEDS_REVIEW
            reasons.append("strategy has low-relevance key evidence")
        key_count = len(patch.evidence_node_ids)
        if key_count and len(noisy) / max(1, key_count) >= 0.25:
            status = VALIDATION_NEEDS_REVIEW
            reasons.append("strategy key nodes include too much low-relevance evidence")
        if not patch.payload.get("metadata", {}).get("slot_order"):
            warnings.append("strategy has no slot_order; shortcut may be underspecified")

    if patch.patch_type == "add_solved_subgoal":
        metadata = patch.payload.get("metadata", {})
        output_slots = metadata.get("output_slots") if isinstance(metadata, Mapping) else {}
        if not output_slots:
            status = VALIDATION_NEEDS_REVIEW
            reasons.append("solved subgoal has no output_slots")
        if not metadata.get("valid_when"):
            warnings.append("solved subgoal has no explicit valid_when")

    if patch.patch_type == "add_control_rule":
        status = VALIDATION_NEEDS_REVIEW if status == VALIDATION_ACCEPT else status
        reasons.append("control rules are broad and require offline review")

    conflicts.extend(_possible_conflicts(patch, graph))
    if conflicts:
        status = VALIDATION_NEEDS_REVIEW
        reasons.append("possible polarity conflict with existing graph node")

    if not reasons and status == VALIDATION_ACCEPT:
        reasons.append("patch has explicit scope and sufficient deterministic support")

    return PatchValidationResult(
        status=status,
        reasons=reasons,
        warnings=warnings,
        support_score=round(support_score, 3),
        duplicate_of=duplicate_of,
        conflicts_with=conflicts,
    )


def _inherit_parent_validation(patches: Sequence[GraphEditPatch]) -> None:
    """If a node patch is unsafe, edges attached to it are unsafe too."""
    by_target = {patch.target_id: patch for patch in patches if patch.source_op == "add_node"}
    for patch in patches:
        if patch.source_op != "add_edge":
            continue
        src = str(patch.raw_edit.get("src", "") or "")
        dst = str(patch.raw_edit.get("dst", "") or "")
        parents = [by_target[nid] for nid in (src, dst) if nid in by_target]
        bad_parent = next(
            (
                parent for parent in parents
                if parent.validation.status in {VALIDATION_NEEDS_REVIEW, VALIDATION_REJECT}
            ),
            None,
        )
        if bad_parent is None:
            continue
        inherited_status = (
            VALIDATION_REJECT
            if bad_parent.validation.status == VALIDATION_REJECT
            else VALIDATION_NEEDS_REVIEW
        )
        patch.validation.status = inherited_status
        reason = f"inherits {bad_parent.validation.status} from parent patch {bad_parent.patch_id}"
        if reason not in patch.validation.reasons:
            patch.validation.reasons.append(reason)
        warning = f"parent_patch_{bad_parent.validation.status}:{bad_parent.target_id}"
        if warning not in patch.validation.warnings:
            patch.validation.warnings.append(warning)


def _min_support_score(patch: GraphEditPatch) -> float:
    if patch.patch_type in {"add_strategy", "add_control_rule"}:
        return 0.08
    if patch.patch_type in {"add_solved_subgoal", "add_reasoning_atom"}:
        return 0.10
    if patch.patch_type == "add_fact":
        return 0.12
    return 0.05


def _reinforcement_relevance(patch: GraphEditPatch, graph: MemoryGraph) -> tuple[float, str]:
    """Score whether a soft citation increment should actually reinforce a node."""
    question = str(patch.scope.get("question", "") or "").strip()
    if not question:
        return (1.0, "")
    node = graph.nodes.get(patch.target_id)
    if node is None:
        return (0.0, "missing_reinforcement_target")
    node_basis = f"{node.id} {node.node_type} {node.text}"
    score = lexical_overlap(question, node_basis)
    score = max(score, _conceptual_relevance_bonus(question, node_basis))
    if _topic_mismatch(question, node_basis):
        return (score, f"low_relevance_reinforcement:{patch.target_id}")
    if score < 0.035:
        return (score, f"low_relevance_reinforcement:{patch.target_id}")
    return (score, "")


def _conceptual_relevance_bonus(question: str, node_basis: str) -> float:
    q = normalize_text(question).lower()
    n = normalize_text(node_basis).lower()
    if ("prism" in q or "bend light" in q or ("light" in q and "bend" in q)) and "refraction" in n:
        return 0.12
    if ("sound" in q and "space" in q) and ("medium" in n or "vacuum" in n):
        return 0.12
    if ("rank" in q or "leaderboard" in q) and ("fenwick" in n or "prefix" in n):
        return 0.10
    return 0.0


def _topic_mismatch(question: str, node_basis: str) -> bool:
    q_tokens = content_tokens(question, min_chars=4)
    n_tokens = content_tokens(node_basis, min_chars=4)
    if not q_tokens or not n_tokens:
        return False
    topic_groups = [
        {"shortest", "path", "grid", "bfs", "dijkstra", "bellman"},
        {"leaderboard", "rank", "score", "pagination", "concurrent"},
        {"prism", "refraction", "frequency", "light"},
        {"sound", "vacuum", "medium", "electromagnetic"},
    ]
    q_groups = [group for group in topic_groups if q_tokens & group]
    n_groups = [group for group in topic_groups if n_tokens & group]
    if not q_groups or not n_groups:
        return False
    return not any(q_group is n_group for q_group in q_groups for n_group in n_groups)


def _unsupported_claim_slots(patch: GraphEditPatch, graph: MemoryGraph) -> List[str]:
    if patch.patch_type != "add_fact":
        return []
    claim = normalize_text(patch.text).lower()
    if not claim:
        return []
    evidence_parts: List[str] = []
    for nid in patch.evidence_node_ids:
        node = graph.nodes.get(nid)
        if node is not None:
            evidence_parts.append(f"{node.id} {node.node_type} {node.text}")
    evidence = normalize_text(" ".join(evidence_parts)).lower()
    if not evidence:
        return []

    slot_specs = [
        (
            "find_kth_or_select",
            ["find_kth", "find kth", "kth", "k-th", "pagination", "ranks 1000", "range pagination"],
            ["find_kth", "find kth", "kth", "k-th", "select", "order statistic", "binary lifting", "lower_bound"],
        ),
        (
            "tie_policy",
            ["tie", "ties", "tied", "tiebreaker", "timestamp"],
            ["tie", "ties", "tied", "tiebreaker", "timestamp", "sorted set", "balanced tree"],
        ),
        (
            "latency_or_pubsub",
            ["100ms", "websocket", "pub/sub", "snapshot", "propagate"],
            ["100ms", "websocket", "pub/sub", "snapshot", "latency", "propagate"],
        ),
        (
            "score_bucket_assumption",
            ["score bucket", "score buckets", "bucket"],
            ["score bucket", "score buckets", "bucket", "coordinate compression", "bounded score"],
        ),
        (
            "balanced_set_structure",
            ["balanced bst", "balanced tree", "sorted set"],
            ["balanced bst", "balanced tree", "sorted set", "order statistic", "tree"],
        ),
    ]
    unsupported: List[str] = []
    for slot, triggers, support_terms in slot_specs:
        if not _contains_any(claim, triggers):
            continue
        if not _contains_any(evidence, support_terms):
            unsupported.append(slot)
    return unsupported


def _contains_any(text: str, terms: Sequence[str]) -> bool:
    return any(term in text for term in terms)


def _support_score(patch: GraphEditPatch, graph: MemoryGraph) -> tuple[float, List[str]]:
    warnings: List[str] = []
    if not patch.evidence_node_ids:
        return (0.0, warnings)
    evidence_texts = []
    missing = []
    for nid in patch.evidence_node_ids:
        node = graph.nodes.get(nid)
        if node is None:
            missing.append(nid)
            continue
        evidence_texts.append(f"{node.id} {node.node_type} {node.text}")
    if missing:
        warnings.append("missing_evidence_nodes:" + ",".join(missing[:8]))
    if not evidence_texts:
        return (0.0, warnings)

    patch_basis = " ".join([
        patch.text,
        str(patch.scope.get("question", "")),
        " ".join(str(x) for x in patch.scope.get("required_slots", [])),
    ])
    scores = [lexical_overlap(patch_basis, text) for text in evidence_texts]
    if not scores:
        return (0.0, warnings)
    # Reward multiple weak supports a little, but keep the score bounded.
    best = max(scores)
    avg = sum(scores) / len(scores)
    score = min(1.0, best * 0.75 + avg * 0.25 + math.log1p(len(scores)) * 0.03)
    return (score, warnings)


def _low_relevance_evidence_ids(patch: GraphEditPatch, graph: MemoryGraph) -> List[str]:
    basis = " ".join([
        patch.text,
        str(patch.scope.get("question", "")),
        " ".join(str(x) for x in patch.scope.get("required_slots", [])),
    ])
    out: List[str] = []
    for nid in patch.evidence_node_ids:
        node = graph.nodes.get(nid)
        if node is None:
            out.append(nid)
            continue
        node_basis = f"{node.id} {node.text}"
        if lexical_overlap(basis, node_basis) < 0.085:
            out.append(nid)
    return out


def _possible_conflicts(patch: GraphEditPatch, graph: MemoryGraph) -> List[str]:
    patch_tokens = content_tokens(patch.text, min_chars=4)
    if len(patch_tokens) < 3:
        return []
    polarity = _polarity(patch.text)
    if polarity == 0:
        return []
    conflicts: List[str] = []
    for nid, node in graph.nodes.items():
        if nid in patch.evidence_node_ids:
            continue
        node_tokens = content_tokens(node.text, min_chars=4)
        if len(patch_tokens & node_tokens) < 3:
            continue
        if lexical_overlap(patch.text, node.text) < 0.18:
            continue
        if _condition_contrast(patch.text, node.text):
            continue
        node_polarity = _polarity(node.text)
        if node_polarity and node_polarity != polarity:
            conflicts.append(nid)
            if len(conflicts) >= 5:
                break
    return conflicts


def _polarity(text: str) -> int:
    lower = normalize_text(text).lower()
    toks = set(re.findall(r"[a-z']+", lower))
    neg = bool(toks & _NEGATION_TERMS)
    pos = bool(toks & _AFFIRMATION_TERMS)
    if neg and not pos:
        return -1
    if pos and not neg:
        return 1
    if "not guaranteed" in lower or "not generally" in lower:
        return -1
    return 0


def _condition_contrast(a: str, b: str) -> bool:
    a_low = normalize_text(a).lower()
    b_low = normalize_text(b).lower()
    a_negative = "negative edge" in a_low or "negative weight" in a_low
    b_negative = "negative edge" in b_low or "negative weight" in b_low
    a_nonnegative = "nonnegative" in a_low or "non-negative" in a_low
    b_nonnegative = "nonnegative" in b_low or "non-negative" in b_low
    return (a_negative and b_nonnegative) or (b_negative and a_nonnegative)


def summarize_patches(patches: Sequence[GraphEditPatch]) -> Dict[str, Any]:
    by_status: Dict[str, int] = {}
    by_type: Dict[str, int] = {}
    by_risk: Dict[str, int] = {}
    warnings: List[str] = []
    review_patch_ids: List[str] = []
    for patch in patches:
        status = patch.validation.status
        by_status[status] = by_status.get(status, 0) + 1
        by_type[patch.patch_type] = by_type.get(patch.patch_type, 0) + 1
        by_risk[patch.risk_level] = by_risk.get(patch.risk_level, 0) + 1
        if status in {VALIDATION_NEEDS_REVIEW, VALIDATION_REJECT}:
            review_patch_ids.append(patch.patch_id)
        for warning in patch.validation.warnings:
            if warning not in warnings:
                warnings.append(warning)
    return {
        "generated_at": _now_iso(),
        "patch_count": len(patches),
        "by_status": by_status,
        "by_type": by_type,
        "by_risk": by_risk,
        "needs_attention_count": len(review_patch_ids),
        "needs_attention_patch_ids": review_patch_ids,
        "warnings": warnings[:50],
    }


def patches_to_dicts(patches: Sequence[GraphEditPatch]) -> List[Dict[str, Any]]:
    return [patch.to_dict() for patch in patches]


def write_scoped_patch_artifacts(out_dir: str | Path, patches: Sequence[GraphEditPatch]) -> Dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = summarize_patches(patches)
    (out / "scoped_patches.json").write_text(
        json.dumps(patches_to_dicts(patches), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out / "scoped_patch_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out / "scoped_patch_report.md").write_text(
        render_patch_report(patches, title="Scoped Patch Report"),
        encoding="utf-8",
    )
    return summary


def render_patch_report(patches: Sequence[GraphEditPatch], *, title: str = "Scoped Edit Lab Report") -> str:
    summary = summarize_patches(patches)
    lines: List[str] = [
        f"# {title}",
        "",
        f"- generated_at: {summary['generated_at']}",
        f"- patch_count: {summary['patch_count']}",
        f"- by_status: `{json.dumps(summary['by_status'], sort_keys=True)}`",
        f"- by_type: `{json.dumps(summary['by_type'], sort_keys=True)}`",
        f"- by_risk: `{json.dumps(summary['by_risk'], sort_keys=True)}`",
        "",
        "## Patches",
    ]
    for patch in patches:
        lines.extend([
            "",
            f"### {patch.patch_id}",
            "",
            f"- type: `{patch.patch_type}`",
            f"- target: `{patch.target_id}`",
            f"- risk: `{patch.risk_level}`",
            f"- status: `{patch.validation.status}`",
            f"- support_score: `{patch.validation.support_score}`",
            f"- evidence: `{', '.join(patch.evidence_node_ids) or '(none)'}`",
            f"- reasons: {'; '.join(patch.validation.reasons) or '(none)'}",
            f"- warnings: {'; '.join(patch.validation.warnings) or '(none)'}",
            "",
            "```text",
            patch.text.strip()[:2000],
            "```",
        ])
    return "\n".join(lines).rstrip() + "\n"
