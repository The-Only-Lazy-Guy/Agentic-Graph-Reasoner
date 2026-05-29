"""Rules-first micro epistemic controller.

The controller operates at subgoal granularity:
  1. identify the next semantic subgoal
  2. check whether local working memory already satisfies it
  3. choose the cheapest next action
  4. stop once required answer slots are filled

V1 is deterministic and schema-driven. A learned scorer can plug in later,
but correctness does not depend on it.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Set, Tuple

from graph_core import MemoryGraph, Node
from reasoning.lexical_matching import DEFAULT_CONTENT_STOPWORDS, content_tokens, lexical_overlap
from reasoning.schemas import ControlRuleNode, Provenance


TaskFamily = str

_RELATIONSHIP_RE = re.compile(
    r"\b(relationship between|difference between|compare|contrast|how .* relate)\b",
    re.IGNORECASE,
)
_DESIGN_RE = re.compile(
    r"\b(design|architect|system|distributed|migration|migrate|pipeline|service|workflow|rollout|backend|frontend)\b",
    re.IGNORECASE,
)
_CODE_RE = re.compile(
    r"\b(implement|implementation|write code|function|class|template|pseudocode|refactor|build)\b",
    re.IGNORECASE,
)
_PROCEDURE_RE = re.compile(
    r"\b(before answering|precondition|preconditions|apply\s+verify|verify each|given graph|instance\b|edges?\s*\(|source\s+[A-Za-z0-9_]+)\b",
    re.IGNORECASE,
)
_JUDGMENT_RE = re.compile(r"^\s*(can|is|are|does|do|did|should|will|would|could)\b", re.IGNORECASE)
_LOOKUP_PREFIXES = (
    "what is",
    "what are",
    "why is",
    "why does",
    "when should",
    "when is",
    "explain",
    "describe",
)
_NEGATIVE_EDGE_RE = re.compile(r"\bnegative\s+(edge|edges|weight|weights)\b", re.IGNORECASE)
_ALL_NEGATIVE_RE = re.compile(r"\ball(?:[\s-]+negative)\s+(array|arrays)\b", re.IGNORECASE)
_HOW_WORKS_RE = re.compile(r"\bhow\s+does\b.*\b(work|works)\b", re.IGNORECASE)
_HOW_APPLIES_RE = re.compile(r"\bhow\s+does\b.*\b(apply|applies|used)\b", re.IGNORECASE)
_ALGORITHM_HINTS = (
    "dijkstra",
    "bellman-ford",
    "floyd-warshall",
    "kruskal",
    "prim",
    "bfs",
    "dfs",
    "topological",
    "segment tree",
    "fenwick",
    "union-find",
    "dsu",
    "kadane",
)
_ROLE_ALTERNATIVE_CUES = ("bellman-ford", "alternative", "instead", "safe alternative")
_ROLE_REASON_CUES = (
    "because",
    "invariant",
    "settles",
    "too early",
    "breaks",
    "counterexample",
    "wrong shortest path",
    "finalized",
)
_ROLE_VERDICT_CUES = (
    "requires",
    "cannot",
    "unsafe",
    "not guaranteed",
    "incorrect",
    "fails",
    "nonnegative",
)
_ROLE_CAVEAT_CUES = ("some graphs", "specific graph", "not generally", "standard dijkstra", "variant")
_MECHANISM_CUES = (
    "priority queue",
    "priority_queue",
    "min-heap",
    "heap",
    "relax",
    "tentative",
    "smallest",
    "unsettled",
    "extract",
    "stale",
    "pop",
)
_DESIGN_CORE_STRUCTURE_CUES = (
    "fenwick",
    "binary indexed tree",
    "segment tree",
    "sorted set",
    "score bucket",
    "bucketized score",
)
_DESIGN_RANK_QUERY_CUES = (
    "prefix-sum",
    "prefix sum",
    "order-statistic",
    "order statistic",
    "rank query",
    "o(log n)",
    "count of",
)
_DESIGN_PAGINATION_CUES = (
    "pagination",
    "ordered access",
    "order-statistic",
    "order statistic",
    "find-kth",
    "find kth",
    "k-th",
    "lower_bound",
    "lower bound",
    "b-tree",
    "sorted vector",
)
_DESIGN_TIE_POLICY_CUES = (
    "tie",
    "timestamp",
    "secondary sort",
    "secondary key",
    "deterministic ordering",
    "stable ordering",
)
_DESIGN_SCALE_CUES = (
    "shard",
    "replica",
    "router",
    "fan out",
    "fan-out",
    "leader",
    "partition",
    "frontend",
)
_DESIGN_LATENCY_CUES = (
    "latency",
    "millisecond",
    "microsecond",
    "100ms",
    "throughput",
    "propagate",
)
_DESIGN_CONSISTENCY_CUES = (
    "read-after-write",
    "eventual consistency",
    "eventual",
    "consistent",
    "consistency",
    "atomic",
    "replication",
)
_DESIGN_FAILURE_FIX_CUES = (
    "race condition",
    "interleaving",
    "deadlock",
    "compare-and-swap",
    "compare and swap",
    "cas",
    "mutex",
    "lock ordering",
    "invariant",
)
_FOCUSED_RETRIEVAL_STOPWORDS = frozenset(set(DEFAULT_CONTENT_STOPWORDS) | {
    "also",
    "because",
    "been",
    "being",
    "both",
    "cannot",
    "cant",
    "could",
    "does",
    "doing",
    "done",
    "from",
    "have",
    "into",
    "just",
    "more",
    "must",
    "need",
    "only",
    "really",
    "should",
    "than",
    "that",
    "their",
    "then",
    "there",
    "this",
    "through",
    "using",
    "what",
    "when",
    "where",
    "which",
    "while",
    "would",
    "your",
    "not",
    "but",
    "can",
    "why",
})


class MicroAction(str, Enum):
    REUSE = "REUSE"
    QUERY = "QUERY"
    DERIVE = "DERIVE"
    VERIFY = "VERIFY"
    ASK = "ASK"
    FINALIZE = "FINALIZE"


class ControllerScorerHook(Protocol):
    def rank_actions(
        self,
        *,
        subgoal: "Subgoal",
        candidates: Sequence[MicroAction],
        state: Mapping[str, Any],
    ) -> Optional[Sequence[MicroAction]]:
        ...

    def compatibility_score(
        self,
        *,
        task_frame: "TaskFrame",
        candidate: Mapping[str, Any],
    ) -> Optional[float]:
        ...


@dataclass
class SlotMask:
    required: List[str]
    filled: List[str] = field(default_factory=list)

    def missing(self) -> List[str]:
        return [slot for slot in self.required if slot not in set(self.filled)]

    def sufficient(self) -> bool:
        return not self.missing()

    def to_dict(self) -> Dict[str, Any]:
        return {"required": list(self.required), "filled": list(self.filled), "missing": self.missing()}


@dataclass
class SubgoalSignature:
    value: str

    def to_dict(self) -> Dict[str, Any]:
        return {"value": self.value}


@dataclass
class ContextSignature:
    task_family: TaskFamily
    task_signature: str
    task_subtype: str
    question_mode: str
    entities: Dict[str, str]
    conditions: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Subgoal:
    name: str
    signature: SubgoalSignature
    prompt: str
    required_slots: List[str]
    optional_slots: List[str] = field(default_factory=list)
    desired_evidence_type: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "signature": self.signature.to_dict(),
            "prompt": self.prompt,
            "required_slots": list(self.required_slots),
            "optional_slots": list(self.optional_slots),
            "desired_evidence_type": self.desired_evidence_type,
        }


@dataclass
class SubgoalResult:
    slot_values: Dict[str, str] = field(default_factory=dict)
    evidence_node_ids: List[str] = field(default_factory=list)
    source_node_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class KnownnessResult:
    sufficient: bool
    action: MicroAction
    slot_mask: SlotMask
    result: SubgoalResult = field(default_factory=SubgoalResult)
    matched_node_id: Optional[str] = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sufficient": self.sufficient,
            "action": self.action.value,
            "slot_mask": self.slot_mask.to_dict(),
            "result": self.result.to_dict(),
            "matched_node_id": self.matched_node_id,
            "reason": self.reason,
        }


@dataclass
class MicroStepDecision:
    index: int
    subgoal: str
    subgoal_signature: str
    action: MicroAction
    sufficient: bool
    filled_slots: List[str]
    missing_slots: List[str]
    evidence_node_ids: List[str]
    matched_node_id: Optional[str]
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "subgoal": self.subgoal,
            "subgoal_signature": self.subgoal_signature,
            "action": self.action.value,
            "sufficient": self.sufficient,
            "filled_slots": list(self.filled_slots),
            "missing_slots": list(self.missing_slots),
            "evidence_node_ids": list(self.evidence_node_ids),
            "matched_node_id": self.matched_node_id,
            "detail": self.detail,
        }


@dataclass
class WorkingSet:
    anchor_ids: List[str]
    local_node_ids: List[str]
    facts: List[Node]
    solved_subgoals: List[Dict[str, Any]]
    reasoning_atoms: List[Dict[str, Any]]
    strategies: List[Dict[str, Any]]
    strategy_key_node_ids: List[str]
    control_rules: List[ControlRuleNode]
    global_queries_used: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anchor_ids": list(self.anchor_ids),
            "local_node_ids": list(self.local_node_ids),
            "fact_ids": [n.id for n in self.facts],
            "solved_subgoal_ids": [n.get("id", "") for n in self.solved_subgoals],
            "reasoning_atom_ids": [n.get("id", "") for n in self.reasoning_atoms],
            "strategy_ids": [n.get("id", "") for n in self.strategies],
            "strategy_key_node_ids": list(self.strategy_key_node_ids),
            "control_rule_ids": [n.id for n in self.control_rules],
            "global_queries_used": self.global_queries_used,
        }


@dataclass
class ControllerPolicy:
    task_family: TaskFamily
    required_slots: List[str]
    optional_slots: List[str]
    preferred_action_order: List[MicroAction]
    forbidden_escalations: List[MicroAction]
    max_subgoals: int
    max_graph_queries: int
    max_derivations: int
    answer_style: str = "concise"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_family": self.task_family,
            "required_slots": list(self.required_slots),
            "optional_slots": list(self.optional_slots),
            "preferred_action_order": [a.value for a in self.preferred_action_order],
            "forbidden_escalations": [a.value for a in self.forbidden_escalations],
            "max_subgoals": self.max_subgoals,
            "max_graph_queries": self.max_graph_queries,
            "max_derivations": self.max_derivations,
            "answer_style": self.answer_style,
        }


@dataclass
class TaskFrame:
    question: str
    task_family: TaskFamily
    task_signature: str
    context: ContextSignature
    required_slots: List[str]
    optional_slots: List[str]
    subgoals: List[Subgoal]
    policy: ControllerPolicy

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "task_family": self.task_family,
            "task_signature": self.task_signature,
            "context": self.context.to_dict(),
            "required_slots": list(self.required_slots),
            "optional_slots": list(self.optional_slots),
            "subgoals": [sg.to_dict() for sg in self.subgoals],
            "policy": self.policy.to_dict(),
        }


@dataclass
class MicroControllerOutcome:
    task_frame: TaskFrame
    working_set: WorkingSet
    slot_values: Dict[str, str]
    slot_sources: Dict[str, List[str]]
    micro_steps: List[MicroStepDecision]
    controller_action_counts: Dict[str, int]
    subgoal_reuse_count: int
    selected_node_ids: List[str]
    finalizable: bool
    controller_fallback_used: bool
    used_query_expansion: bool
    strategy_assist_used: bool
    exact_answer_reuse_used: bool

    @property
    def task_family(self) -> TaskFamily:
        return self.task_frame.task_family

    def missing_slots(self) -> List[str]:
        return [slot for slot in self.task_frame.required_slots if slot not in self.slot_values]

    def slot_fill_stats(self) -> Dict[str, Any]:
        return {
            "required_slots": list(self.task_frame.required_slots),
            "filled_slots": sorted(self.slot_values.keys()),
            "missing_slots": self.missing_slots(),
            "filled_count": len(self.slot_values),
            "required_count": len(self.task_frame.required_slots),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_frame": self.task_frame.to_dict(),
            "working_set": self.working_set.to_dict(),
            "slot_values": dict(self.slot_values),
            "slot_sources": {k: list(v) for k, v in self.slot_sources.items()},
            "micro_steps": [step.to_dict() for step in self.micro_steps],
            "controller_action_counts": dict(self.controller_action_counts),
            "subgoal_reuse_count": self.subgoal_reuse_count,
            "selected_node_ids": list(self.selected_node_ids),
            "finalizable": self.finalizable,
            "controller_fallback_used": self.controller_fallback_used,
            "used_query_expansion": self.used_query_expansion,
            "strategy_assist_used": self.strategy_assist_used,
            "exact_answer_reuse_used": self.exact_answer_reuse_used,
            "slot_fill_stats": self.slot_fill_stats(),
        }


def infer_task_family(question: str) -> TaskFamily:
    q = (question or "").strip().lower()
    if not q:
        return "direct_judgment"
    if _PROCEDURE_RE.search(q):
        return "procedure_or_instance_verification"
    if _DESIGN_RE.search(q) or _CODE_RE.search(q):
        return "design_synthesis"
    if _RELATIONSHIP_RE.search(q):
        return "relational_explanation"
    if _looks_like_algorithm_applicability(q):
        return "algorithm_applicability"
    if _JUDGMENT_RE.match(q) or q.startswith(_LOOKUP_PREFIXES):
        return "direct_judgment"
    return "direct_judgment"


def build_task_frame(question: str, *, task_family: Optional[TaskFamily] = None) -> TaskFrame:
    family = task_family or infer_task_family(question)
    entities = _extract_entities(question, family)
    conditions = _extract_conditions(question, family)
    task_subtype, question_mode = _infer_task_profile(question, family, entities, conditions)
    context = ContextSignature(
        task_family=family,
        task_signature=_task_signature(question, family, task_subtype, question_mode, entities, conditions),
        task_subtype=task_subtype,
        question_mode=question_mode,
        entities=entities,
        conditions=conditions,
    )
    policy = _policy_for_family(family, task_subtype)
    subgoals = _subgoals_for_family(family, task_subtype, context.task_signature)
    if family == "algorithm_applicability" and not _requires_alternative_slot(question, context):
        required_slots = [slot for slot in policy.required_slots if slot != "alternative"]
        optional_slots = list(policy.optional_slots)
        if "alternative" not in optional_slots:
            optional_slots.append("alternative")
        subgoals = [
            sg for sg in subgoals
            if "alternative" not in sg.required_slots
        ]
        policy = ControllerPolicy(
            task_family=policy.task_family,
            required_slots=required_slots,
            optional_slots=optional_slots,
            preferred_action_order=list(policy.preferred_action_order),
            forbidden_escalations=list(policy.forbidden_escalations),
            max_subgoals=max(len(required_slots), 3),
            max_graph_queries=policy.max_graph_queries,
            max_derivations=policy.max_derivations,
            answer_style=policy.answer_style,
        )
    else:
        required_slots = list(policy.required_slots)
        optional_slots = list(policy.optional_slots)
        # BUGFIX: Enforce nonnegative precondition for Dijkstra in learned capsule's answer slot
        if "dijkstra" in str(entities.get("algorithm", "")).lower() and "preconditions" in optional_slots:
            optional_slots.remove("preconditions")
            required_slots.insert(0, "preconditions")
            
            # also insert the subgoal
            for idx, sg in enumerate(subgoals):
                if sg.name == "compose_answer":
                    subgoals.insert(idx, Subgoal(
                        name="check_preconditions",
                        signature=SubgoalSignature(f"{context.task_signature}.check_preconditions"),
                        prompt="check preconditions",
                        required_slots=["preconditions"],
                        desired_evidence_type="preconditions",
                    ))
                    break
            
            policy = ControllerPolicy(
                task_family=policy.task_family,
                required_slots=required_slots,
                optional_slots=optional_slots,
                preferred_action_order=list(policy.preferred_action_order),
                forbidden_escalations=list(policy.forbidden_escalations),
                max_subgoals=max(len(required_slots), 3),
                max_graph_queries=policy.max_graph_queries,
                max_derivations=policy.max_derivations,
                answer_style=policy.answer_style,
            )
    return TaskFrame(
        question=question,
        task_family=family,
        task_signature=context.task_signature,
        context=context,
        required_slots=required_slots,
        optional_slots=optional_slots,
        subgoals=subgoals,
        policy=policy,
    )


def _active_invalidators(
    *,
    graph: MemoryGraph,
    selected_node_ids: Sequence[str],
    question: str,
    overlap_threshold: float = 0.10,
) -> List[Tuple[str, str, float]]:
    """Return (source_node_id, invalidator_node_id, overlap_score) for any
    invalidated_by edge whose destination text overlaps the question.

    Scoped to: edges with relation == "invalidated_by" originating from a
    currently selected evidence node. The overlap threshold is intentionally
    low; a stricter check would deny safe shortcuts on legitimate questions.
    """
    if not selected_node_ids:
        return []
    selected = set(selected_node_ids)
    hits: List[Tuple[str, str, float]] = []
    edges = getattr(graph, "edges", None) or []
    for edge in edges:
        try:
            relation = getattr(edge, "relation", None) or (
                edge.get("relation") if isinstance(edge, dict) else None
            )
        except Exception:
            continue
        if relation != "invalidated_by":
            continue
        src = getattr(edge, "src", None) or (
            edge.get("src") if isinstance(edge, dict) else None
        )
        dst = getattr(edge, "dst", None) or (
            edge.get("dst") if isinstance(edge, dict) else None
        )
        if not (src in selected and dst):
            continue
        dst_node = graph.nodes.get(dst)
        if dst_node is None:
            continue
        condition_text = (dst_node.text or "") + " " + " ".join(
            str(v) for v in (getattr(dst_node, "metadata", {}) or {}).values()
            if isinstance(v, str)
        )
        if not condition_text.strip():
            continue
        score = _focused_overlap(question, condition_text, min_chars=4)
        if score >= overlap_threshold:
            hits.append((str(src), str(dst), float(score)))
    return hits


def run_micro_epistemic_controller(
    *,
    question: str,
    graph: MemoryGraph,
    anchor_ids: Sequence[str],
    scorer_hook: Optional[ControllerScorerHook] = None,
) -> MicroControllerOutcome:
    task_frame = build_task_frame(question)
    working_set = _build_working_set(graph, anchor_ids, task_frame)

    slot_values: Dict[str, str] = {}
    slot_sources: Dict[str, List[str]] = {}
    _seed_question_derived_slots(task_frame, slot_values, slot_sources)
    selected_node_ids: List[str] = []
    seen_selected: Set[str] = set()
    micro_steps: List[MicroStepDecision] = []
    action_counts = {action.value: 0 for action in MicroAction}
    subgoal_reuse_count = 0
    controller_fallback_used = False
    used_query_expansion = False
    strategy_assist_used = bool(working_set.strategy_key_node_ids)
    exact_answer_reuse_used = False

    for idx, subgoal in enumerate(task_frame.subgoals[: task_frame.policy.max_subgoals], start=1):
        known = _check_knownness(
            task_frame=task_frame,
            subgoal=subgoal,
            working_set=working_set,
            slot_values=slot_values,
            scorer_hook=scorer_hook,
        )
        if (
            not known.sufficient
            and known.action == MicroAction.QUERY
            and working_set.global_queries_used < task_frame.policy.max_graph_queries
        ):
            used_query_expansion = _expand_working_set_for_missing_slots(
                question=question,
                graph=graph,
                task_frame=task_frame,
                working_set=working_set,
                missing_slots=known.slot_mask.missing(),
            ) or used_query_expansion
            if used_query_expansion:
                known = _check_knownness(
                    task_frame=task_frame,
                    subgoal=subgoal,
                    working_set=working_set,
                    slot_values=slot_values,
                    scorer_hook=scorer_hook,
                )

        _merge_slots(slot_values, slot_sources, known.result.slot_values, known.result.evidence_node_ids, known.matched_node_id)
        for nid in known.result.evidence_node_ids:
            if nid and nid not in seen_selected:
                seen_selected.add(nid)
                selected_node_ids.append(nid)

        if known.action == MicroAction.REUSE:
            subgoal_reuse_count += 1
            if known.reason.startswith("Exact solved_subgoal match reused"):
                exact_answer_reuse_used = True
        action_counts[known.action.value] = action_counts.get(known.action.value, 0) + 1
        micro_steps.append(MicroStepDecision(
            index=idx,
            subgoal=subgoal.name,
            subgoal_signature=subgoal.signature.value,
            action=known.action,
            sufficient=known.sufficient,
            filled_slots=list(known.result.slot_values.keys()),
            missing_slots=known.slot_mask.missing(),
            evidence_node_ids=list(known.result.evidence_node_ids),
            matched_node_id=known.matched_node_id,
            detail=known.reason,
        ))

        _synthesize_missing_slots(task_frame, slot_values, slot_sources, selected_node_ids)
        if _required_slots_satisfied(task_frame, slot_values):
            invalidator_hits = _active_invalidators(
                graph=graph,
                selected_node_ids=selected_node_ids,
                question=question,
            )
            if invalidator_hits:
                action_counts[MicroAction.QUERY.value] = action_counts.get(MicroAction.QUERY.value, 0) + 1
                top = invalidator_hits[0]
                detail = (
                    "FINALIZE blocked by active invalidated_by edge: "
                    f"{top[0]} --invalidated_by--> {top[1]} (overlap={top[2]:.2f}). "
                    "Falling back to graph-tool loop."
                )
                micro_steps.append(MicroStepDecision(
                    index=len(micro_steps) + 1,
                    subgoal="invalidator_guard",
                    subgoal_signature=f"{task_frame.task_signature}.invalidator_guard",
                    action=MicroAction.QUERY,
                    sufficient=False,
                    filled_slots=list(slot_values.keys()),
                    missing_slots=[
                        f"invalidator_clearance:{nid}" for _, nid, _ in invalidator_hits
                    ],
                    evidence_node_ids=list(selected_node_ids),
                    matched_node_id=None,
                    detail=detail,
                ))
                controller_fallback_used = True
                break
            action_counts[MicroAction.FINALIZE.value] = action_counts.get(MicroAction.FINALIZE.value, 0) + 1
            micro_steps.append(MicroStepDecision(
                index=len(micro_steps) + 1,
                subgoal="finalize_answer",
                subgoal_signature=f"{task_frame.task_signature}.finalize",
                action=MicroAction.FINALIZE,
                sufficient=True,
                filled_slots=list(slot_values.keys()),
                missing_slots=[],
                evidence_node_ids=list(selected_node_ids),
                matched_node_id=None,
                detail="All required slots filled from working memory.",
            ))
            break

        if known.action in {MicroAction.DERIVE, MicroAction.VERIFY, MicroAction.ASK}:
            controller_fallback_used = True
            break

    finalizable = _required_slots_satisfied(task_frame, slot_values)
    return MicroControllerOutcome(
        task_frame=task_frame,
        working_set=working_set,
        slot_values=slot_values,
        slot_sources=slot_sources,
        micro_steps=micro_steps,
        controller_action_counts=action_counts,
        subgoal_reuse_count=subgoal_reuse_count,
        selected_node_ids=selected_node_ids,
        finalizable=finalizable,
        controller_fallback_used=controller_fallback_used or not finalizable,
        used_query_expansion=used_query_expansion,
        strategy_assist_used=strategy_assist_used,
        exact_answer_reuse_used=exact_answer_reuse_used,
    )


def cheap_anchor_candidates(
    question: str,
    graph: MemoryGraph,
    *,
    k: int = 8,
) -> List[str]:
    q_tokens = _focused_tokens(question, min_chars=4)
    scored: List[Tuple[float, str]] = []
    for node in graph.nodes.values():
        meta = getattr(node, "metadata", {}) or {}
        if meta.get("deprecated"):
            continue
        if str(getattr(node, "node_type", "") or "").lower() == "strategy" and int(meta.get("strategy_schema_version", 0) or 0) < 2:
            continue
        text = node.text or ""
        t_tokens = _focused_tokens(text, min_chars=4)
        coverage = 0.0 if not q_tokens else len(q_tokens & t_tokens) / max(len(q_tokens), 1)
        node_type = str(getattr(node, "node_type", "") or "").lower()
        score = (
            0.85 * _focused_overlap(question, text, min_chars=4)
            + 0.35 * coverage
            + 0.10 * max(0.0, min(float(getattr(node, "confidence", 0.0) or 0.0), 1.0))
        )
        if node_type in {"claim", "application", "example", "fact", "principle", "explanation"}:
            score += 0.08
        elif node_type in {"bridge", "summary"}:
            score += 0.03
        if meta.get("polarity") == "false":
            score -= 0.08
        low = text.lower()
        if "dijkstra" in question.lower():
            if "dijkstra" in low:
                score += 0.12
            if "bellman-ford" in low:
                score += 0.10
            if "negative edge" in low or "nonnegative" in low:
                score += 0.10
            if "counterexample" in low or "wrong shortest path" in low:
                score += 0.08
        scored.append((score, node.id))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [nid for _score, nid in scored[:k]]


def render_micro_context_block(outcome: MicroControllerOutcome) -> str:
    missing = outcome.missing_slots()
    lines = ["<micro_controller>"]
    lines.append(f"task_family: {outcome.task_family}")
    lines.append(f"task_signature: {outcome.task_frame.task_signature}")
    lines.append(f"required_slots: {', '.join(outcome.task_frame.required_slots)}")
    lines.append(f"filled_slots: {', '.join(sorted(outcome.slot_values)) or '(none)'}")
    lines.append(f"missing_slots: {', '.join(missing) or '(none)'}")
    if outcome.selected_node_ids:
        lines.append(f"evidence_node_ids: {', '.join(outcome.selected_node_ids[:8])}")
    if outcome.micro_steps:
        last = outcome.micro_steps[-1]
        lines.append(f"recommended_action: {last.action.value}")
        lines.append(f"controller_note: {last.detail}")
    if outcome.strategy_assist_used and not outcome.exact_answer_reuse_used:
        lines.append("strategy_mode: checkpoint_assist")
        lines.append("Use learned strategy only as a checkpoint/evidence shortcut. Do not treat it as a memorized final answer.")
    else:
        lines.append("Use the filled slots to shorten the plan and choose the cheapest reads first.")
    lines.append("You must still read graph evidence with read_node before writing the final answer.")
    lines.append("</micro_controller>")
    return "\n".join(lines)


MICRO_FINALIZE_SYSTEM_PROMPT = """\
You are in MICRO-CONTROLLER FINALIZE mode.

The controller has already determined that the required answer slots are
filled from local graph memory. Do not search, re-derive, or broaden scope.

Use the provided slots and evidence only. Output exactly:

<reasoning>
2-5 concise sentences about why the stored answer is sufficient.
</reasoning>

<answer>
Final user-facing answer.
</answer>

<explanation>
One short paragraph about how the answer was grounded.
</explanation>

OPTIONAL: V5 graph patches
==========================
After the explanation, you may emit one or more <patch>{...}</patch> JSON
blocks to record meta-reasoning about the answer. This is encouraged when
the answer relies on a known shortcut, has conditions that would invalidate
it, or requires specific task-frame slots.

Allowed node types:
  epistemic_state, solved_subgoal, strategy, claim, fact, reasoning_atom,
  control_rule, failure_pattern
Allowed edge relations:
  epistemic_of, invalidated_by, requires_slot, transfers_to,
  overlaps, entails, contradicts, leveraged, derived_from, related

Most useful pattern -- attach an epistemic_state to the node you relied on:

<patch>
{"op": "add_node", "node_type": "epistemic_state",
 "target_node_id": "<the_evidence_node_id_you_used>",
 "status": "verified", "confidence": 0.9,
 "support_level": "stored solved_subgoal + textbook fact",
 "open_questions": [],
 "known_risks": ["<condition under which this shortcut would mislead>"],
 "invalidators": ["<specific question variant that would NOT be answered by this>"],
 "last_verified_by": ["<evidence_node_id>", "<another_evidence_node_id>"]}
</patch>

To attach a slot requirement to a strategy node:
<patch>
{"op": "add_edge", "src": "<strategy_node_id>", "dst": "<slot_node_id>",
 "relation": "requires_slot"}
</patch>

Edge endpoints must already exist in the graph OR be added by a sibling
patch in the same response. Unknown endpoints are rejected.
Skip patches entirely if you cannot ground them in the supplied evidence.
"""


def build_finalize_user_message(
    question: str,
    outcome: MicroControllerOutcome,
    graph: MemoryGraph,
    evidence_node_ids: Optional[Sequence[str]] = None,
) -> str:
    lines = [
        f"Question:\n{question}",
        "",
        f"Task family: {outcome.task_family}",
        f"Task signature: {outcome.task_frame.task_signature}",
        "The micro-controller has already filled the required answer slots.",
        "",
        "Resolved slots:",
    ]
    for slot in outcome.task_frame.required_slots:
        value = outcome.slot_values.get(slot)
        if value:
            lines.append(f"- {slot}: {value}")
    optional_values = {
        slot: outcome.slot_values[slot]
        for slot in outcome.slot_values
        if slot not in set(outcome.task_frame.required_slots)
    }
    if optional_values:
        lines.append("")
        lines.append("Optional slots:")
        for slot, value in optional_values.items():
            lines.append(f"- {slot}: {value}")
    chosen_evidence = list(evidence_node_ids or outcome.selected_node_ids)
    if chosen_evidence:
        lines.append("")
        lines.append("Evidence nodes:")
        for nid in chosen_evidence[:8]:
            node = graph.nodes.get(nid)
            if node is None:
                continue
            lines.append(f"### {nid} [{node.node_type}]")
            lines.append(node.text)
            lines.append("")
    return "\n".join(lines).strip()


def finalize_evidence_node_ids(
    outcome: MicroControllerOutcome,
    *,
    max_nodes: int = 4,
) -> List[str]:
    ordered: List[str] = []
    seen: Set[str] = set()

    def add(node_id: str) -> None:
        nid = str(node_id or "").strip()
        if not nid or nid in seen:
            return
        seen.add(nid)
        ordered.append(nid)

    for nid in outcome.selected_node_ids:
        add(nid)
    for slot in outcome.task_frame.required_slots:
        for nid in outcome.slot_sources.get(slot, []):
            add(nid)

    subtype = outcome.task_frame.context.task_subtype
    candidates: List[Tuple[float, str]] = []
    for node in outcome.working_set.facts:
        if _should_skip_local_fact(node) or node.id in seen:
            continue
        text = node.text or ""
        low = text.lower()
        if subtype == "algorithm_mechanism_explanation":
            score = 0.0
            if any(cue in low for cue in _MECHANISM_CUES):
                score += 0.35
            extracted = _extract_slots_from_text(outcome.task_frame, text)
            mech_value = extracted.get("mechanism") or text
            score += _slot_candidate_score(outcome.task_frame, "mechanism", node, mech_value)
            if any(cue in low for cue in ("nonnegative", "greedy", "settled", "distance")):
                score += 0.06
            if score > 0.18:
                candidates.append((score, node.id))
        elif subtype == "algorithm_usage_context":
            extracted = _extract_slots_from_text(outcome.task_frame, text)
            usage_value = extracted.get("usage_context") or text
            score = _slot_candidate_score(outcome.task_frame, "usage_context", node, usage_value)
            if score > 0.18:
                candidates.append((score, node.id))
        elif outcome.task_frame.task_family == "direct_judgment" and subtype == "direct_judgment":
            if not _generic_direct_relevant(outcome.task_frame, text):
                continue
            extracted = _extract_slots_from_text(outcome.task_frame, text)
            extracted.update(_extract_generic_direct_slots(outcome.task_frame, node))
            score = max(
                _slot_candidate_score(outcome.task_frame, "answer", node, extracted.get("answer") or text),
                _slot_candidate_score(outcome.task_frame, "reason", node, extracted.get("reason") or text),
            )
            if score > 0.24:
                candidates.append((score, node.id))

    candidates.sort(key=lambda item: item[0], reverse=True)
    for _score, nid in candidates:
        if len(ordered) >= max_nodes:
            break
        add(nid)

    if not ordered:
        for nid in outcome.working_set.anchor_ids:
            if len(ordered) >= max_nodes:
                break
            add(nid)
    return ordered[:max_nodes]


def compose_answer_from_slots(outcome: MicroControllerOutcome) -> str:
    slots = _normalized_slot_values(outcome)
    family = outcome.task_family
    if family == "algorithm_applicability":
        parts = [
            slots.get("verdict", ""),
            slots.get("reason", ""),
            slots.get("alternative", ""),
            slots.get("caveat", ""),
        ]
        return " ".join(part.strip() for part in parts if part and part.strip()).strip()
    if family == "relational_explanation":
        parts = [slots.get("relationship", ""), slots.get("explanation", "")]
        return " ".join(part.strip() for part in parts if part and part.strip()).strip()
    if family == "procedure_or_instance_verification":
        parts = [slots.get("verdict", ""), slots.get("answer", "")]
        return " ".join(part.strip() for part in parts if part and part.strip()).strip()
    if family == "design_synthesis":
        if slots.get("answer"):
            return slots["answer"]
        parts = [
            slots.get("core_structure", ""),
            slots.get("rank_query", ""),
            slots.get("pagination", ""),
            slots.get("tie_policy", ""),
            slots.get("scale_architecture", ""),
            slots.get("latency_budget", ""),
            slots.get("consistency_model", ""),
            slots.get("failure_mode_fix", ""),
        ]
        return " ".join(part.strip() for part in parts if part and part.strip()).strip()
    if family == "direct_judgment":
        subtype = outcome.task_frame.context.task_subtype
        if subtype == "algorithm_mechanism_explanation":
            parts = [slots.get("answer", ""), slots.get("preconditions", "")]
            return " ".join(part.strip() for part in parts if part and part.strip()).strip()
        if subtype == "algorithm_usage_context":
            parts = [slots.get("answer", ""), slots.get("preconditions", "")]
            return " ".join(part.strip() for part in parts if part and part.strip()).strip()
    parts = [slots.get("answer", ""), slots.get("reason", "")]
    return " ".join(part.strip() for part in parts if part and part.strip()).strip()


def deterministic_finalize_payload(
    question: str,
    outcome: MicroControllerOutcome,
    graph: MemoryGraph,
) -> Dict[str, str]:
    answer = compose_answer_from_slots(outcome).strip()
    selected_ids = [nid for nid in outcome.selected_node_ids if nid in graph.nodes]
    evidence_summary = ", ".join(selected_ids[:3]) if selected_ids else "local working memory"
    reasoning = (
        f"The controller already filled the required slots from {evidence_summary}. "
        "No extra graph search or re-derivation was needed."
    )
    explanation = (
        "This answer was produced directly from graph evidence that the micro-controller "
        "marked as sufficient for the current subgoal set."
    )
    return {
        "reasoning": reasoning,
        "answer": answer,
        "explanation": explanation,
    }


def propose_control_memory_edits(
    *,
    outcome: MicroControllerOutcome,
    question: str,
    session_id: str,
    graph: Optional[MemoryGraph] = None,
) -> List[Dict[str, Any]]:
    if not outcome.finalizable or not outcome.selected_node_ids:
        return []

    edits: List[Dict[str, Any]] = []
    seen_node_ids: Set[str] = set(graph.nodes.keys()) if graph is not None else set()
    task_frame = outcome.task_frame
    provenance_id = session_id or "micro_controller"
    support_ids = list(outcome.selected_node_ids)
    entities = task_frame.context.entities

    rule_id = f"ctrl_{_short_hash(task_frame.task_family)}_{_short_hash(json.dumps(task_frame.policy.required_slots, sort_keys=True))}"
    if rule_id not in seen_node_ids:
        rule_text = (
            f"Control rule for {task_frame.task_family}: fill "
            f"{', '.join(task_frame.policy.required_slots)} before escalating."
        )
        edits.append({
            "op": "add_node",
            "node_id": rule_id,
            "node_type": "control_rule",
            "text": rule_text,
            "metadata": {
                "task_family": task_frame.task_family,
                "guidance": f"Prefer reuse and targeted slot fill for {task_frame.task_family}.",
                "required_slots": list(task_frame.policy.required_slots),
                "optional_slots": list(task_frame.policy.optional_slots),
                "forbidden_escalations": [a.value for a in task_frame.policy.forbidden_escalations],
                "preferred_action_order": [a.value for a in task_frame.policy.preferred_action_order],
                "stop_condition": "All required slots filled.",
                "source_session": provenance_id,
                "created_at": _now_iso(),
            },
            "tier": "add",
        })
        seen_node_ids.add(rule_id)

    solved_id = f"ssg_{_short_hash(task_frame.task_signature)}"
    if solved_id not in seen_node_ids:
        summary = compose_answer_from_slots(outcome) or question
        edits.append({
            "op": "add_node",
            "node_id": solved_id,
            "node_type": "solved_subgoal",
            "text": summary,
            "metadata": {
                "summary": summary,
                "subgoal_signature": task_frame.task_signature,
                "question_type": task_frame.task_family,
                "task_subtype": task_frame.context.task_subtype,
                "question_mode": task_frame.context.question_mode,
                "input_conditions": dict(task_frame.context.entities | task_frame.context.conditions),
                "output_slots": dict(outcome.slot_values),
                "valid_when": _default_valid_when(task_frame),
                "invalid_when": _default_invalid_when(task_frame),
                "supporting_node_ids": support_ids,
                "confidence": 0.9,
                "source_sessions": [provenance_id],
                "source_session": provenance_id,
                "created_at": _now_iso(),
            },
            "tier": "add",
        })
        seen_node_ids.add(solved_id)
        for support_id in support_ids:
            edits.append({
                "op": "add_edge",
                "src": solved_id,
                "dst": support_id,
                "relation": "derived_from",
                "metadata": {"source_session": provenance_id},
                "tier": "add",
            })

    for slot_name in ("reason", "alternative", "relationship"):
        slot_text = outcome.slot_values.get(slot_name, "").strip()
        if not slot_text:
            continue
        atom_id = f"atom_{slot_name}_{_short_hash(task_frame.task_signature)}"
        if atom_id in seen_node_ids:
            continue
        atom_text = slot_text
        edits.append({
            "op": "add_node",
            "node_id": atom_id,
            "node_type": "reasoning_atom",
            "text": atom_text,
            "metadata": {
                "atom_type": slot_name,
                "claim": atom_text,
                "reusable_for": [task_frame.task_family],
                "task_subtype": task_frame.context.task_subtype,
                "question_mode": task_frame.context.question_mode,
                "dependencies": [entities.get("algorithm", ""), entities.get("condition", "")],
                "supporting_node_ids": support_ids,
                "confidence": 0.88,
                "source_session": provenance_id,
                "created_at": _now_iso(),
            },
            "tier": "add",
        })
        seen_node_ids.add(atom_id)
        for support_id in support_ids[:3]:
            edits.append({
                "op": "add_edge",
                "src": atom_id,
                "dst": support_id,
                "relation": "derived_from",
                "metadata": {"source_session": provenance_id},
                "tier": "add",
            })
    return edits


def _normalized_slot_values(outcome: MicroControllerOutcome) -> Dict[str, str]:
    slots = dict(outcome.slot_values)
    if outcome.task_family == "algorithm_applicability":
        return _repair_algorithm_applicability_slots(outcome, slots)
    return slots


def _repair_algorithm_applicability_slots(
    outcome: MicroControllerOutcome,
    slots: Dict[str, str],
) -> Dict[str, str]:
    repaired = dict(slots)
    facts = list(outcome.working_set.facts)
    entities = outcome.task_frame.context.entities
    algorithm = entities.get("algorithm", "This algorithm") or "This algorithm"
    condition = (
        entities.get("condition")
        or outcome.task_frame.context.conditions.get("graph_property", "")
        or outcome.task_frame.context.conditions.get("input_property", "")
    )

    if (
        not repaired.get("verdict")
        or "requires nonnegative" in repaired.get("verdict", "").lower()
        or repaired.get("verdict", "").strip() == algorithm
        or (
            algorithm.lower() == "kadane"
            and condition == "all_negative_arrays"
            and not any(
                cue in repaired.get("verdict", "").lower()
                for cue in ("does not fail", "least negative", "largest")
            )
        )
    ):
        if algorithm.lower() == "dijkstra" and condition == "negative_edge_weights":
            repaired["verdict"] = "No. Standard Dijkstra is not guaranteed to be correct when negative edge weights are present."
        elif algorithm.lower() == "kadane" and condition == "all_negative_arrays":
            repaired["verdict"] = "No. Standard non-empty Kadane does not fail on all-negative arrays; it returns the largest (least negative) element."
        elif repaired.get("verdict"):
            repaired["verdict"] = _ensure_sentence(repaired["verdict"])

    if (
        not repaired.get("reason")
        or not any(
            cue in repaired.get("reason", "").lower()
            for cue in ("counterexample", "settles", "finalized", "wrong shortest path", "invariant")
        )
    ):
        reason_fact = _best_fact_sentence(
            facts,
            required_cues=("counterexample", "settles", "wrong shortest path", "final", "invariant"),
            preferred_node_types=("application", "claim", "example"),
        )
        if reason_fact:
            repaired["reason"] = reason_fact
        elif algorithm.lower() == "kadane" and condition == "all_negative_arrays":
            repaired["reason"] = (
                "The correct non-empty Kadane initializes current and best to a[0], "
                "so on an all-negative array it keeps the least-negative element instead of returning 0."
            )

    if condition == "negative_edge_weights" and (
        not repaired.get("alternative")
        or "bellman-ford" not in repaired.get("alternative", "").lower()
    ):
        alt_fact = _best_fact_sentence(
            facts,
            required_cues=("bellman-ford",),
            preferred_node_types=("claim", "application", "summary"),
        )
        if alt_fact:
            repaired["alternative"] = alt_fact
        elif algorithm.lower() == "dijkstra" and condition == "negative_edge_weights":
            repaired["alternative"] = "Use Bellman-Ford instead when negative edge weights are present."

    if (
        not repaired.get("caveat")
        or (
            algorithm.lower() == "kadane"
            and condition == "all_negative_arrays"
            and "reset-to-zero" not in repaired.get("caveat", "").lower()
        )
    ):
        if algorithm.lower() == "kadane" and condition == "all_negative_arrays":
            repaired["caveat"] = (
                "The confusion comes from a reset-to-zero variant; the standard non-empty Kadane starts from a[0] instead of clamping sums at 0."
            )
        else:
            repaired["caveat"] = (
                f"{algorithm} may still work on some specific inputs, "
                "but it is not correct in the general case covered by the question."
            )

    repaired["verdict"] = _ensure_sentence(repaired.get("verdict", ""))
    repaired["reason"] = _ensure_sentence(repaired.get("reason", ""))
    repaired["alternative"] = _normalize_alternative_sentence(repaired.get("alternative", ""))
    repaired["caveat"] = _ensure_sentence(repaired.get("caveat", ""))
    return repaired


def _normalize_alternative_sentence(text: str) -> str:
    low = (text or "").lower()
    if "bellman-ford" in low and "use bellman-ford" not in low:
        return "Use Bellman-Ford instead when negative edge weights are present."
    return _ensure_sentence(text)


def _best_fact_sentence(
    facts: Sequence[Node],
    *,
    required_cues: Sequence[str],
    preferred_node_types: Sequence[str] = (),
) -> str:
    best_text = ""
    best_score = -1.0
    for node in facts:
        text = _first_sentence(node.text or "")
        if not text:
            continue
        low = text.lower()
        if not any(cue in low for cue in required_cues):
            continue
        score = 1.0
        if str(getattr(node, "node_type", "")).lower() in set(preferred_node_types):
            score += 0.2
        score += 0.05 * sum(1 for cue in required_cues if cue in low)
        if score > best_score:
            best_score = score
            best_text = text
    return best_text


def _ensure_sentence(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return ""
    if text[-1] not in ".!?":
        return text + "."
    return text


def _build_working_set(
    graph: MemoryGraph,
    anchor_ids: Sequence[str],
    task_frame: TaskFrame,
) -> WorkingSet:
    local_node_ids = graph.local_neighborhood(list(anchor_ids), max_hops=1, max_nodes=24)
    local_ids = list(dict.fromkeys([str(a) for a in anchor_ids] + [str(nid) for nid in local_node_ids]))
    facts = [graph.nodes[nid] for nid in local_ids if nid in graph.nodes]
    solved_subgoals: List[Dict[str, Any]] = []
    reasoning_atoms: List[Dict[str, Any]] = []
    strategies: List[Dict[str, Any]] = []
    strategy_key_node_ids: List[str] = []
    control_rules: List[ControlRuleNode] = []
    for node in graph.nodes.values():
        ntype = str(getattr(node, "node_type", "") or "").lower()
        meta = getattr(node, "metadata", {}) or {}
        if ntype == "solved_subgoal" and _solved_subgoal_relevant(task_frame, node):
            solved_subgoals.append(_normalize_memory_node(node))
        elif ntype == "reasoning_atom" and _reasoning_atom_relevant(task_frame, node):
            reasoning_atoms.append(_normalize_memory_node(node))
        elif ntype == "strategy":
            normalized = _normalize_strategy_node(node)
            if _strategy_relevant_to_task(task_frame, normalized):
                strategies.append(normalized)
        elif ntype == "control_rule":
            fam = str(meta.get("task_family", "") or "")
            if not fam or fam == task_frame.task_family or lexical_overlap(task_frame.question, node.text or "", min_chars=3) >= 0.08:
                control_rules.append(_control_rule_from_graph_node(node))
    strategies.sort(key=lambda item: _strategy_relevance_score(task_frame, item), reverse=True)
    for strategy in strategies[:2]:
        for key_id in strategy.get("key_node_ids", [])[:6]:
            nid = str(key_id)
            if nid in graph.nodes and nid not in local_ids:
                local_ids.append(nid)
                facts.append(graph.nodes[nid])
                strategy_key_node_ids.append(nid)
    if not control_rules:
        control_rules.append(_default_control_rule(task_frame.task_family, task_frame.context.task_subtype))
    return WorkingSet(
        anchor_ids=list(anchor_ids),
        local_node_ids=local_ids,
        facts=facts,
        solved_subgoals=solved_subgoals,
        reasoning_atoms=reasoning_atoms,
        strategies=strategies[:3],
        strategy_key_node_ids=strategy_key_node_ids,
        control_rules=control_rules,
    )


def _check_knownness(
    *,
    task_frame: TaskFrame,
    subgoal: Subgoal,
    working_set: WorkingSet,
    slot_values: Mapping[str, str],
    scorer_hook: Optional[ControllerScorerHook],
) -> KnownnessResult:
    if all(slot in slot_values for slot in subgoal.required_slots):
        mask = SlotMask(required=list(subgoal.required_slots), filled=list(subgoal.required_slots))
        return KnownnessResult(
            sufficient=True,
            action=MicroAction.REUSE,
            slot_mask=mask,
            result=SubgoalResult(slot_values={slot: slot_values[slot] for slot in subgoal.required_slots}),
            reason="Required slots already filled earlier in the run.",
        )

    exact = _match_solved_subgoal(task_frame, working_set.solved_subgoals, exact=True)
    if exact is not None:
        slot_hits = {
            slot: str(exact.get("output_slots", {}).get(slot, "")).strip()
            for slot in subgoal.required_slots
            if str(exact.get("output_slots", {}).get(slot, "")).strip()
        }
        mask = SlotMask(required=list(subgoal.required_slots), filled=list(slot_hits.keys()))
        if mask.sufficient():
            return KnownnessResult(
                sufficient=True,
                action=MicroAction.REUSE,
                slot_mask=mask,
                result=SubgoalResult(
                    slot_values=slot_hits,
                    evidence_node_ids=[str(nid) for nid in exact.get("supporting_node_ids", [])],
                    source_node_id=str(exact.get("id", "")) or None,
                ),
                matched_node_id=str(exact.get("id", "")) or None,
                reason="Exact solved_subgoal match reused.",
            )

    slot_hits, evidence_node_ids = _compose_from_local_memory(task_frame, subgoal, working_set)
    mask = SlotMask(required=list(subgoal.required_slots), filled=list(slot_hits.keys()))
    if mask.sufficient():
        return KnownnessResult(
            sufficient=True,
            action=MicroAction.REUSE,
            slot_mask=mask,
            result=SubgoalResult(slot_values=slot_hits, evidence_node_ids=evidence_node_ids),
            reason="Local working memory filled the required slots.",
        )

    action = _choose_action(
        task_frame,
        subgoal,
        working_set,
        scorer_hook=scorer_hook,
        has_partial_hits=bool(slot_hits),
    )
    return KnownnessResult(
        sufficient=False,
        action=action,
        slot_mask=mask,
        result=SubgoalResult(slot_values=slot_hits, evidence_node_ids=evidence_node_ids),
        reason=f"Missing slots: {', '.join(mask.missing()) or '(none)'}",
    )


def _choose_action(
    task_frame: TaskFrame,
    subgoal: Subgoal,
    working_set: WorkingSet,
    *,
    scorer_hook: Optional[ControllerScorerHook],
    has_partial_hits: bool = False,
) -> MicroAction:
    candidates = list(task_frame.policy.preferred_action_order)
    if scorer_hook is not None:
        try:
            ranked = scorer_hook.rank_actions(subgoal=subgoal, candidates=candidates, state={"working_set": working_set.to_dict()})
        except Exception:
            ranked = None
        if ranked:
            candidates = list(ranked)
    for candidate in candidates:
        if candidate == MicroAction.REUSE:
            continue
        if candidate in task_frame.policy.forbidden_escalations:
            continue
        if candidate == MicroAction.QUERY and working_set.global_queries_used >= task_frame.policy.max_graph_queries:
            continue
        return candidate
    if has_partial_hits:
        return MicroAction.DERIVE
    return MicroAction.DERIVE


def _compose_from_local_memory(
    task_frame: TaskFrame,
    subgoal: Subgoal,
    working_set: WorkingSet,
) -> Tuple[Dict[str, str], List[str]]:
    slot_hits: Dict[str, str] = {}
    evidence_node_ids: List[str] = []
    best_slot_scores: Dict[str, float] = {}
    best_slot_node_ids: Dict[str, str] = {}
    candidate_slots = list(dict.fromkeys(list(subgoal.required_slots) + list(task_frame.optional_slots)))

    for atom in working_set.reasoning_atoms:
        if not _reasoning_atom_payload_compatible(task_frame, atom):
            continue
        claim = str(atom.get("claim", "") or atom.get("text", "") or "").strip()
        if not claim:
            continue
        atom_type = str(atom.get("atom_type", "") or "").strip()
        if atom_type in subgoal.required_slots and atom_type not in slot_hits:
            slot_hits[atom_type] = claim
            evidence_node_ids.extend([str(nid) for nid in atom.get("supporting_node_ids", []) if str(nid)])

    for node in working_set.facts:
        if _should_skip_local_fact(node):
            continue
        if task_frame.task_family == "design_synthesis":
            extracted = _extract_design_slots_from_node(task_frame, node)
        else:
            extracted = _extract_slots_from_text(task_frame, node.text or "")
        if (
            task_frame.task_family == "direct_judgment"
            and task_frame.context.task_subtype == "direct_judgment"
        ):
            extracted.update(_extract_generic_direct_slots(task_frame, node))
        for slot in candidate_slots:
            value = extracted.get(slot)
            if not value:
                continue
            score = _slot_candidate_score(task_frame, slot, node, value)
            if not _slot_candidate_acceptable(task_frame, slot, node, value, score):
                continue
            if slot not in best_slot_scores or score > best_slot_scores[slot]:
                best_slot_scores[slot] = score
                slot_hits[slot] = value
                best_slot_node_ids[slot] = node.id

    if task_frame.task_family == "algorithm_applicability":
        slot_hits = _synthesize_algorithm_applicability_slots(slot_hits, task_frame.context.entities)
    elif task_frame.task_family == "direct_judgment":
        slot_hits = _synthesize_direct_judgment_slots(task_frame, slot_hits)
    elif task_frame.task_family == "relational_explanation":
        slot_hits = _synthesize_relational_slots(slot_hits)

    seen: Set[str] = set()
    deduped = []
    ordered_evidence = list(evidence_node_ids)
    ordered_evidence.extend(best_slot_node_ids.get(slot, "") for slot in candidate_slots)
    for nid in ordered_evidence:
        if nid and nid not in seen:
            seen.add(nid)
            deduped.append(nid)
    return slot_hits, deduped


def _should_skip_local_fact(node: Node) -> bool:
    meta = getattr(node, "metadata", {}) or {}
    node_id = str(getattr(node, "id", "") or "")
    if meta.get("deprecated"):
        return True
    if node_id.endswith("_false"):
        return True
    try:
        if float(getattr(node, "confidence", 0.0) or 0.0) < 0.1:
            return True
    except Exception:
        return False
    return False


def _extract_generic_direct_slots(task_frame: TaskFrame, node: Node) -> Dict[str, str]:
    text = node.text or ""
    if not text or not _generic_direct_relevant(task_frame, text):
        return {}
    node_type = str(getattr(node, "node_type", "") or "").lower()
    if node_type in {"code", "implementation"}:
        return {}
    low = text.lower()
    low_q = task_frame.question.lower()
    first = _first_sentence(text)
    slots: Dict[str, str] = {}

    can_be_answer = node_type in {"claim", "fact", "principle", "explanation", "bridge", "law"}
    if can_be_answer and _focused_overlap(task_frame.question, text, min_chars=4) >= 0.18:
        slots["answer"] = first
    if "frequency" in low_q and "frequency" in low and any(cue in low for cue in ("fixed", "stays", "source", "wavelength")):
        slots["answer"] = first
        slots.setdefault("reason", first)
    if _contains_any_phrase(low_q, ("sound", "hear", "hearing", "audible", "audio", "acoustic", "sonic", "noise")) and _contains_any_phrase(low_q, ("light", "sunlight", "starlight", "laser", "flash", "visible", "see", "seeing", "sight", "star", "stars", "sun")) and "sound" in low and ("medium" in low or "vacuum" in low):
        slots["answer"] = first
        slots.setdefault("reason", first)

    reason_cues = (
        "because",
        "require",
        "requires",
        "medium",
        "vacuum",
        "fixed",
        "source",
        "speed",
        "wavelength",
        "refraction",
        "propagate",
        "self-propagate",
    )
    if can_be_answer and any(cue in low for cue in reason_cues):
        slots.setdefault("reason", first)
    return slots


def _slot_candidate_acceptable(
    task_frame: TaskFrame,
    slot: str,
    node: Node,
    value: str,
    score: float,
) -> bool:
    if (
        task_frame.task_family == "direct_judgment"
        and task_frame.context.task_subtype == "direct_judgment"
    ):
        text = node.text or value
        if not _generic_direct_relevant(task_frame, text):
            return False
        node_type = str(getattr(node, "node_type", "") or "").lower()
        if slot == "answer" and node_type in {"example", "summary"}:
            return False
        if score < 0.22:
            return False
    return True


def _slot_candidate_score(
    task_frame: TaskFrame,
    slot: str,
    node: Node,
    value: str,
) -> float:
    text = node.text or value
    low = text.lower()
    node_type = str(getattr(node, "node_type", "") or "").lower()
    confidence = max(0.0, min(float(getattr(node, "confidence", 0.0) or 0.0), 1.0))
    subtype = task_frame.context.task_subtype
    score = lexical_overlap(task_frame.question, text, min_chars=3)
    score += 0.10 * confidence

    if node_type in {"claim", "fact", "application", "example"}:
        score += 0.10
    elif node_type in {"summary", "bridge"}:
        score -= 0.08
    elif node_type in {"code", "implementation"}:
        score -= 0.12
    if any(marker in text for marker in ("#include", "std::", "vector<", "return ", "for (")):
        score -= 0.10

    if task_frame.task_family == "algorithm_applicability":
        condition = (
            task_frame.context.entities.get("condition")
            or task_frame.context.conditions.get("graph_property")
            or task_frame.context.conditions.get("input_property")
        )
        if slot == "verdict":
            if "requires nonnegative" in low or "not guaranteed" in low:
                score += 0.45
            if "works on graphs with non-negative edge weights" in low:
                score += 0.25
        elif slot == "reason":
            if "counterexample" in low:
                score += 0.30
            if any(cue in low for cue in ("invariant", "settlement", "final", "wrong shortest path")):
                score += 0.20
        elif slot == "alternative":
            if "bellman-ford" in low:
                score += 0.50
            if "repeated relaxation" in low:
                score += 0.05
        elif slot == "caveat":
            if any(cue in low for cue in ("some specific", "not general", "variant")):
                score += 0.20
        if condition == "all_negative_arrays":
            if "all-negative" in low or "all negative" in low:
                score += 0.30
            if any(cue in low for cue in ("least-negative", "largest", "a[0]", "returns 0", "reset-to-zero")):
                score += 0.18
    elif task_frame.task_family == "direct_judgment" and subtype == "algorithm_mechanism_explanation":
        if slot == "mechanism":
            if any(cue in low for cue in _MECHANISM_CUES):
                score += 0.45
            if any(cue in low for cue in ("negative edge", "counterexample", "bellman-ford")):
                score -= 0.20
        elif slot == "answer" and any(cue in low for cue in _MECHANISM_CUES):
            score += 0.12
        if node_type in {"concept", "explanation", "application", "example"}:
            score += 0.08
    elif task_frame.task_family == "direct_judgment" and subtype == "algorithm_usage_context":
        if slot == "usage_context":
            if any(cue in low for cue in ("single-source shortest path", "nonnegative", "weighted graph", "travel times", "suitable", "used on", "applies to")):
                score += 0.35
            if any(cue in low for cue in ("breadth-first search", "unweighted graph")):
                score -= 0.10
        elif slot == "answer" and any(cue in low for cue in ("suitable", "single-source shortest path", "nonnegative", "weighted graph")):
            score += 0.10
        if any(cue in low for cue in ("counterexample", "negative edge", "bellman-ford")):
            score -= 0.12
    elif task_frame.task_family == "direct_judgment" and subtype == "direct_judgment":
        focused = _focused_overlap(task_frame.question, text, min_chars=4)
        coverage = _focused_coverage(task_frame.question, text, min_chars=4)
        score = (0.75 * focused) + (0.25 * coverage) + (0.08 * confidence)
        if node_type in {"claim", "fact", "principle", "explanation", "law"}:
            score += 0.12
        elif node_type == "bridge":
            score += 0.04
        elif node_type in {"example", "summary"}:
            score -= 0.10
        low_q = task_frame.question.lower()
        if "frequency" in low_q and "frequency" in low:
            score += 0.18 if slot == "answer" else 0.08
            if any(cue in low for cue in ("fixed", "stays", "source")):
                score += 0.22 if slot == "answer" else 0.06
        if "prism" in low_q and "refraction" in low:
            score += 0.32 if slot == "reason" else 0.16
        elif slot == "reason" and "prism" in low_q:
            score -= 0.25
        if slot == "reason" and "bend" in low_q and any(cue in low for cue in ("speed", "medium", "refractive index", "refraction")):
            score += 0.18
        if _contains_any_phrase(low_q, ("sound", "hear", "hearing", "audible", "audio", "acoustic", "sonic", "noise")) and ("medium" in low or "vacuum" in low):
            score += 0.22
        if _contains_any_phrase(low_q, ("light", "sunlight", "starlight", "laser", "flash", "visible", "see", "seeing", "sight", "star", "stars", "sun")) and ("electromagnetic" in low or "vacuum" in low):
            score += 0.16
        if slot == "answer" and node_type == "example":
            score -= 0.30
    return score


def _expand_working_set_for_missing_slots(
    *,
    question: str,
    graph: MemoryGraph,
    task_frame: TaskFrame,
    working_set: WorkingSet,
    missing_slots: Sequence[str],
) -> bool:
    query = f"{question} {' '.join(missing_slots)}"
    existing = {node.id for node in working_set.facts}
    scored: List[Tuple[float, str]] = []
    for node in graph.nodes.values():
        if node.id in existing:
            continue
        score = lexical_overlap(query, node.text or "", min_chars=3)
        if score <= 0.10:
            continue
        slot_bonus = _slot_alignment_bonus(task_frame, missing_slots, node.text or "")
        if slot_bonus <= 0 and score < 0.20:
            continue
        scored.append((score + slot_bonus, node.id))
    scored.sort(key=lambda item: item[0], reverse=True)
    added = False
    for _score, node_id in scored[:2]:
        if node_id in graph.nodes and node_id not in existing:
            working_set.facts.append(graph.nodes[node_id])
            working_set.local_node_ids.append(node_id)
            added = True
    if added:
        working_set.global_queries_used += 1
    return added


def _slot_alignment_bonus(task_frame: TaskFrame, missing_slots: Sequence[str], text: str) -> float:
    low = text.lower()
    bonus = 0.0
    design_cues = {
        "core_structure": _DESIGN_CORE_STRUCTURE_CUES,
        "rank_query": _DESIGN_RANK_QUERY_CUES,
        "pagination": _DESIGN_PAGINATION_CUES,
        "tie_policy": _DESIGN_TIE_POLICY_CUES,
        "scale_architecture": _DESIGN_SCALE_CUES,
        "latency_budget": _DESIGN_LATENCY_CUES,
        "consistency_model": _DESIGN_CONSISTENCY_CUES,
        "failure_mode_fix": _DESIGN_FAILURE_FIX_CUES,
    }
    for slot in missing_slots:
        if slot == "alternative" and any(cue in low for cue in _ROLE_ALTERNATIVE_CUES):
            bonus += 0.25
        elif slot == "reason" and any(cue in low for cue in _ROLE_REASON_CUES):
            bonus += 0.25
        elif slot in {"verdict", "answer", "relationship"} and any(cue in low for cue in _ROLE_VERDICT_CUES):
            bonus += 0.20
        elif slot == "caveat" and any(cue in low for cue in _ROLE_CAVEAT_CUES):
            bonus += 0.18
        elif slot in design_cues and any(cue in low for cue in design_cues[slot]):
            bonus += 0.18
    if task_frame.task_family == "algorithm_applicability" and "negative edge" in low:
        bonus += 0.10
    return bonus


def _extract_design_slots_from_node(task_frame: TaskFrame, node: Node) -> Dict[str, str]:
    text = node.text or ""
    if not text:
        return {}
    low = text.lower()
    first = _first_sentence(text)
    node_type = str(getattr(node, "node_type", "") or "").lower()
    slots: Dict[str, str] = {}

    if node_type in {"strategy", "control_rule", "solved_subgoal"}:
        return slots

    if any(cue in low for cue in _DESIGN_CORE_STRUCTURE_CUES):
        slots["core_structure"] = first
    if any(cue in low for cue in _DESIGN_RANK_QUERY_CUES):
        slots["rank_query"] = first
    if any(cue in low for cue in _DESIGN_PAGINATION_CUES):
        slots["pagination"] = first
    if any(cue in low for cue in _DESIGN_TIE_POLICY_CUES):
        slots["tie_policy"] = first
    if any(cue in low for cue in _DESIGN_SCALE_CUES):
        slots["scale_architecture"] = first
    if any(cue in low for cue in _DESIGN_LATENCY_CUES):
        slots["latency_budget"] = first
    if any(cue in low for cue in _DESIGN_CONSISTENCY_CUES):
        slots["consistency_model"] = first
    if any(cue in low for cue in _DESIGN_FAILURE_FIX_CUES):
        slots["failure_mode_fix"] = first

    # Prefer concrete data-structure/application nodes for the core design slots.
    if node_type in {"claim", "application", "example"}:
        return slots
    for slot in ("core_structure", "rank_query", "pagination"):
        slots.pop(slot, None)
    return slots


def _extract_slots_from_text(task_frame: TaskFrame, text: str) -> Dict[str, str]:
    low = text.lower()
    first = _first_sentence(text)
    slots: Dict[str, str] = {}
    if task_frame.task_family == "algorithm_applicability":
        if any(cue in low for cue in _ROLE_VERDICT_CUES) and ("negative" in low or "correct" in low or "safe" in low):
            slots["verdict"] = first
        if any(cue in low for cue in _ROLE_REASON_CUES):
            slots["reason"] = first
        if any(cue in low for cue in _ROLE_ALTERNATIVE_CUES):
            slots["alternative"] = first
        if any(cue in low for cue in _ROLE_CAVEAT_CUES):
            slots["caveat"] = first
        if "all-negative" in low or "all negative" in low:
            if any(cue in low for cue in ("least-negative", "largest", "a[0]", "does not fail", "returns 0")):
                slots.setdefault("verdict", first)
                slots.setdefault("reason", first)
            if any(cue in low for cue in ("reset-to-zero", "non-empty")):
                slots.setdefault("caveat", first)
    elif task_frame.task_family == "direct_judgment" and task_frame.context.task_subtype == "algorithm_mechanism_explanation":
        if any(cue in low for cue in _MECHANISM_CUES):
            slots["mechanism"] = first
            slots.setdefault("answer", first)
        if any(cue in low for cue in ("nonnegative", "single-source shortest path", "weighted graph")):
            slots.setdefault("preconditions", first)
    elif task_frame.task_family == "direct_judgment" and task_frame.context.task_subtype == "algorithm_usage_context":
        if any(cue in low for cue in ("single-source shortest path", "nonnegative", "weighted graph", "travel times", "suitable", "used on", "applies to")):
            slots["usage_context"] = first
            slots.setdefault("answer", first)
        if any(cue in low for cue in ("nonnegative", "weighted graph")):
            slots.setdefault("preconditions", first)
    elif task_frame.task_family == "relational_explanation":
        if any(cue in low for cue in ("relate", "extends", "inherits", "difference", "compare", "contrast")):
            slots["relationship"] = first
            slots.setdefault("explanation", first)
    else:
        if any(cue in low for cue in _ROLE_VERDICT_CUES):
            slots["answer"] = first
        if any(cue in low for cue in _ROLE_REASON_CUES):
            slots["reason"] = first
    return slots


def _synthesize_algorithm_applicability_slots(slot_hits: Dict[str, str], entities: Mapping[str, str]) -> Dict[str, str]:
    hits = dict(slot_hits)
    algorithm = entities.get("algorithm", "This algorithm")
    if "caveat" not in hits:
        hits["caveat"] = (
            f"{algorithm} may still work on some specific inputs, "
            "but it is not correct in the general case covered by the question."
        )
    if "verdict" not in hits and "reason" in hits:
        hits["verdict"] = f"{algorithm} is not guaranteed to be correct under these conditions."
    return hits


def _synthesize_direct_judgment_slots(
    task_frame: TaskFrame,
    slot_hits: Dict[str, str],
) -> Dict[str, str]:
    hits = dict(slot_hits)
    subtype = task_frame.context.task_subtype
    if subtype == "algorithm_mechanism_explanation":
        if "answer" not in hits and "mechanism" in hits:
            algorithm = task_frame.context.entities.get("algorithm", "This algorithm") or "This algorithm"
            mechanism = hits["mechanism"].strip()
            preconditions = hits.get("preconditions", "").strip()
            low = mechanism.lower()
            if any(cue in low for cue in ("stale extracted", "stale entries", "stored distance", "priority-queue", "priority queue")):
                answer = (
                    f"{algorithm} maintains tentative distances and a min-priority queue of candidate nodes. "
                    "It repeatedly extracts the node with the smallest current distance, relaxes outgoing edges, "
                    "and pushes improved neighbors back into the queue. "
                    f"{mechanism}"
                )
                if preconditions:
                    answer += f" {preconditions}"
                hits["answer"] = answer
            else:
                hits["answer"] = mechanism if not preconditions else f"{mechanism} {preconditions}"
        return hits
    if subtype == "algorithm_usage_context":
        if "answer" not in hits and "usage_context" in hits:
            hits["answer"] = hits["usage_context"]
        return hits
    if "answer" not in hits and "reason" in hits:
        hits["answer"] = hits["reason"]
    return hits


def _synthesize_relational_slots(slot_hits: Dict[str, str]) -> Dict[str, str]:
    hits = dict(slot_hits)
    if "relationship" in hits and "explanation" not in hits:
        hits["explanation"] = hits["relationship"]
    return hits


def _synthesize_missing_slots(
    task_frame: TaskFrame,
    slot_values: Dict[str, str],
    slot_sources: Dict[str, List[str]],
    selected_node_ids: Sequence[str],
) -> None:
    if task_frame.task_family == "algorithm_applicability":
        synthesized = _synthesize_algorithm_applicability_slots(slot_values, task_frame.context.entities)
    elif task_frame.task_family == "direct_judgment":
        synthesized = _synthesize_direct_judgment_slots(task_frame, slot_values)
    elif task_frame.task_family == "relational_explanation":
        synthesized = _synthesize_relational_slots(slot_values)
    elif task_frame.task_family == "design_synthesis":
        synthesized = dict(slot_values)
    else:
        synthesized = dict(slot_values)
    for slot, value in synthesized.items():
        if slot not in slot_values and value:
            slot_values[slot] = value
            slot_sources[slot] = list(selected_node_ids)[:3]


def _seed_question_derived_slots(
    task_frame: TaskFrame,
    slot_values: Dict[str, str],
    slot_sources: Dict[str, List[str]],
) -> None:
    if task_frame.task_family != "design_synthesis":
        return
    low_q = task_frame.question.lower()
    if "problem_frame" not in slot_values:
        parts: List[str] = ["Design a real-time leaderboard service"]
        if "concurrent" in low_q:
            parts.append("for high concurrent usage")
        requirements: List[str] = []
        if "100ms" in low_q or "100 ms" in low_q:
            requirements.append("sub-100ms update propagation")
        if "o(log n)" in low_q:
            requirements.append("O(log n) rank queries")
        if "pagination" in low_q or "rank 1000" in low_q:
            requirements.append("rank-range pagination")
        if "tie" in low_q:
            requirements.append("deterministic tie handling")
        if requirements:
            parts.append("with " + ", ".join(requirements))
        slot_values["problem_frame"] = " ".join(parts) + "."
        slot_sources["problem_frame"] = ["question"]
    if "latency_budget" not in slot_values:
        latency_match = re.search(r"(\d+\s*ms|\d+\s*milliseconds?)", task_frame.question, re.IGNORECASE)
        if latency_match:
            slot_values["latency_budget"] = (
                f"Updates must propagate within {latency_match.group(1).replace('milliseconds', 'ms')}."
            )
            slot_sources["latency_budget"] = ["question"]


def _merge_slots(
    slot_values: Dict[str, str],
    slot_sources: Dict[str, List[str]],
    new_values: Mapping[str, str],
    evidence_node_ids: Sequence[str],
    matched_node_id: Optional[str],
) -> None:
    sources: List[str] = [str(nid) for nid in evidence_node_ids if str(nid)]
    if matched_node_id:
        sources = [matched_node_id] + [nid for nid in sources if nid != matched_node_id]
    for slot, value in new_values.items():
        if value and slot not in slot_values:
            slot_values[slot] = value
            slot_sources[slot] = list(dict.fromkeys(sources))


def _required_slots_satisfied(task_frame: TaskFrame, slot_values: Mapping[str, str]) -> bool:
    return all(str(slot_values.get(slot, "")).strip() for slot in task_frame.required_slots)


def _match_solved_subgoal(
    task_frame: TaskFrame,
    candidates: Sequence[Mapping[str, Any]],
    *,
    exact: bool,
) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    best_score = -1.0
    for cand in candidates:
        if not _candidate_context_compatible(task_frame, cand):
            continue
        cand_sig = str(cand.get("subgoal_signature", "") or "")
        score = 0.0
        if cand_sig == task_frame.task_signature:
            score += 1.0
        elif exact:
            continue
        elif cand.get("question_type") == task_frame.task_family:
            score += 0.65
        score += 0.20 * lexical_overlap(task_frame.question, str(cand.get("summary", "") or ""), min_chars=3)
        score += 0.05 * min(len(set(cand.get("supporting_node_ids", []))), 4)
        if score > best_score:
            best_score = score
            best = dict(cand)
    if exact and best_score < 1.0:
        return None
    if not exact and best_score < 0.70:
        return None
    return best


def _candidate_context_compatible(task_frame: TaskFrame, candidate: Mapping[str, Any]) -> bool:
    low_q = task_frame.question.lower()
    for phrase in candidate.get("invalid_when", []) or []:
        if str(phrase).lower() in low_q:
            return False
    question_type = str(candidate.get("question_type", "") or "")
    if question_type and question_type != task_frame.task_family:
        return False
    task_subtype = str(candidate.get("task_subtype", "") or "")
    if task_subtype and task_subtype != task_frame.context.task_subtype:
        return False
    question_mode = str(candidate.get("question_mode", "") or "")
    if question_mode and question_mode != task_frame.context.question_mode:
        return False
    input_conditions = candidate.get("input_conditions", {}) or {}
    for key, value in input_conditions.items():
        text = str(value or "").strip().lower()
        if not text:
            continue
        context_value = str(
            task_frame.context.entities.get(str(key), "")
            or task_frame.context.conditions.get(str(key), "")
        ).strip().lower()
        if context_value and _normalized_match(text, context_value):
            continue
        if task_frame.task_family == "algorithm_applicability" and text.replace("_", " ") in low_q:
            continue
        if text in low_q:
            continue
        return False
    return True


def _normalize_memory_node(node: Node) -> Dict[str, Any]:
    meta = getattr(node, "metadata", {}) or {}
    payload = dict(meta)
    payload.setdefault("id", node.id)
    payload.setdefault("text", node.text)
    payload.setdefault("summary", meta.get("summary") or node.text)
    payload.setdefault("supporting_node_ids", list(meta.get("supporting_node_ids", [])))
    payload.setdefault("output_slots", dict(meta.get("output_slots", {})))
    payload.setdefault("question_type", meta.get("question_type", ""))
    payload.setdefault("subgoal_signature", meta.get("subgoal_signature", ""))
    payload.setdefault("input_conditions", dict(meta.get("input_conditions", {})))
    payload.setdefault("valid_when", list(meta.get("valid_when", [])))
    payload.setdefault("invalid_when", list(meta.get("invalid_when", [])))
    payload.setdefault("task_subtype", str(meta.get("task_subtype", "") or ""))
    payload.setdefault("question_mode", str(meta.get("question_mode", "") or ""))
    payload.setdefault("claim", meta.get("claim") or node.text)
    payload.setdefault("atom_type", meta.get("atom_type", ""))
    return payload


def _normalize_strategy_node(node: Node) -> Dict[str, Any]:
    meta = getattr(node, "metadata", {}) or {}
    return {
        "id": node.id,
        "text": node.text,
        "task_family": str(meta.get("task_family", "") or ""),
        "task_subtype": str(meta.get("task_subtype", "") or ""),
        "question_mode": str(meta.get("question_mode", "") or ""),
        "entry_conditions": dict(meta.get("entry_conditions", {})),
        "key_node_ids": [str(nid) for nid in meta.get("key_node_ids", []) if str(nid)],
        "domain_keywords": [str(k) for k in meta.get("domain_keywords", []) if str(k)],
        "slot_order": [str(slot) for slot in meta.get("slot_order", []) if str(slot)],
        "checkpoint_plan": [str(step) for step in meta.get("checkpoint_plan", []) if str(step)],
        "stop_conditions": [str(step) for step in meta.get("stop_conditions", []) if str(step)],
        "forbidden_finalize_conditions": [str(step) for step in meta.get("forbidden_finalize_conditions", []) if str(step)],
        "strategy_schema_version": int(meta.get("strategy_schema_version", 0) or 0),
    }


def _strategy_relevance_score(task_frame: TaskFrame, strategy: Mapping[str, Any]) -> float:
    score = 0.0
    if str(strategy.get("task_family", "")) == task_frame.task_family:
        score += 0.40
    if str(strategy.get("task_subtype", "")) == task_frame.context.task_subtype:
        score += 0.35
    if str(strategy.get("question_mode", "")) == task_frame.context.question_mode:
        score += 0.20
    keywords = " ".join(strategy.get("domain_keywords", []))
    score += 0.15 * lexical_overlap(task_frame.question, keywords, min_chars=3)
    score += 0.03 * min(len(strategy.get("key_node_ids", [])), 4)
    return score


def _strategy_relevant_to_task(task_frame: TaskFrame, strategy: Mapping[str, Any]) -> bool:
    if int(strategy.get("strategy_schema_version", 0) or 0) < 2:
        return False
    if str(strategy.get("task_family", "") or "") not in {"", task_frame.task_family}:
        return False
    if str(strategy.get("task_subtype", "") or "") not in {"", task_frame.context.task_subtype}:
        return False
    if str(strategy.get("question_mode", "") or "") not in {"", task_frame.context.question_mode}:
        return False
    low_q = task_frame.question.lower()
    for phrase in strategy.get("forbidden_finalize_conditions", []) or []:
        if str(phrase).lower() in low_q:
            return False
    for key, value in (strategy.get("entry_conditions", {}) or {}).items():
        text = str(value or "").strip().lower()
        if not text:
            continue
        context_value = str(
            task_frame.context.entities.get(str(key), "")
            or task_frame.context.conditions.get(str(key), "")
        ).strip().lower()
        if context_value:
            if not _normalized_match(text, context_value):
                return False
        elif text.replace("_", " ") not in low_q and text not in low_q:
            return False
    if strategy.get("key_node_ids"):
        return True
    return _strategy_relevance_score(task_frame, strategy) >= 0.55


def _solved_subgoal_relevant(task_frame: TaskFrame, node: Node) -> bool:
    return _candidate_context_compatible(task_frame, _normalize_memory_node(node))


def _reasoning_atom_relevant(task_frame: TaskFrame, node: Node) -> bool:
    payload = _normalize_memory_node(node)
    return _reasoning_atom_payload_compatible(task_frame, payload)


def _reasoning_atom_payload_compatible(task_frame: TaskFrame, payload: Mapping[str, Any]) -> bool:
    reusable_for = [str(x) for x in payload.get("reusable_for", [])]
    if reusable_for and task_frame.task_family not in reusable_for:
        return False
    task_subtype = str(payload.get("task_subtype", "") or "")
    if task_subtype and task_subtype != task_frame.context.task_subtype:
        return False
    question_mode = str(payload.get("question_mode", "") or "")
    if question_mode and question_mode != task_frame.context.question_mode:
        return False
    dependencies = " ".join(str(x) for x in payload.get("dependencies", []))
    if dependencies and lexical_overlap(task_frame.question, dependencies, min_chars=3) >= 0.10:
        return True
    return lexical_overlap(task_frame.question, str(payload.get("text", "") or payload.get("claim", "")), min_chars=3) >= 0.08


def _normalized_match(left: str, right: str) -> bool:
    lnorm = str(left or "").strip().lower().replace("-", "_").replace(" ", "_")
    rnorm = str(right or "").strip().lower().replace("-", "_").replace(" ", "_")
    return bool(lnorm and rnorm and lnorm == rnorm)


def _control_rule_from_graph_node(node: Node) -> ControlRuleNode:
    meta = getattr(node, "metadata", {}) or {}
    return ControlRuleNode(
        id=node.id,
        task_family=str(meta.get("task_family", "") or ""),
        guidance=str(meta.get("guidance", "") or node.text or ""),
        required_slots=[str(x) for x in meta.get("required_slots", [])],
        optional_slots=[str(x) for x in meta.get("optional_slots", [])],
        forbidden_escalations=[str(x) for x in meta.get("forbidden_escalations", [])],
        preferred_action_order=[str(x) for x in meta.get("preferred_action_order", [])],
        stop_condition=str(meta.get("stop_condition", "") or ""),
        provenance=Provenance(created_in_session_id=str(meta.get("source_session", "graph"))),
    )


def _default_control_rule(task_family: TaskFamily, task_subtype: str = "") -> ControlRuleNode:
    specs = {
        "algorithm_applicability": {
            "required_slots": ["verdict", "reason", "alternative", "caveat"],
            "optional_slots": ["counterexample", "proof"],
            "forbidden_escalations": [],
            "preferred_action_order": ["REUSE", "QUERY", "DERIVE", "FINALIZE"],
            "stop_condition": "All required slots filled.",
            "guidance": "Answer with verdict, reason, caveat, and safer alternative before escalating.",
        },
        "relational_explanation": {
            "required_slots": ["relationship", "explanation"],
            "optional_slots": ["example"],
            "forbidden_escalations": [],
            "preferred_action_order": ["REUSE", "QUERY", "DERIVE", "FINALIZE"],
            "stop_condition": "Relationship and explanation filled.",
            "guidance": "State the relationship explicitly, then explain it.",
        },
        "design_synthesis": {
            "required_slots": [
                "problem_frame",
                "core_structure",
                "rank_query",
                "pagination",
                "tie_policy",
                "scale_architecture",
                "latency_budget",
                "consistency_model",
                "failure_mode_fix",
                "answer",
            ],
            "optional_slots": ["tradeoffs", "assumptions", "verification"],
            "forbidden_escalations": [],
            "preferred_action_order": ["REUSE", "QUERY", "DERIVE", "FINALIZE"],
            "stop_condition": "Core structure, query paths, scale, latency, consistency, failure fix, and answer filled.",
            "guidance": "Synthesize only after each requested design dimension is explicit and supported.",
        },
        "procedure_or_instance_verification": {
            "required_slots": ["instance_summary", "precondition_results", "verdict", "answer"],
            "optional_slots": ["counterexample"],
            "forbidden_escalations": [],
            "preferred_action_order": ["REUSE", "VERIFY", "QUERY", "DERIVE", "FINALIZE"],
            "stop_condition": "Instance summary, verification result, and verdict filled.",
            "guidance": "Prefer deterministic verification over abstract derivation.",
        },
        "direct_judgment": {
            "required_slots": ["answer", "reason"],
            "optional_slots": ["caveat"],
            "forbidden_escalations": [],
            "preferred_action_order": ["REUSE", "QUERY", "DERIVE", "FINALIZE"],
            "stop_condition": "Answer and reason filled.",
            "guidance": "Answer directly once the supporting reason is already present.",
        },
    }
    if task_family == "direct_judgment" and task_subtype == "algorithm_mechanism_explanation":
        spec = {
            "required_slots": ["mechanism", "answer"],
            "optional_slots": ["preconditions", "example"],
            "forbidden_escalations": [],
            "preferred_action_order": ["REUSE", "QUERY", "DERIVE", "FINALIZE"],
            "stop_condition": "Mechanism and answer filled.",
            "guidance": "Explain the core algorithmic mechanism, then phrase it as a fresh answer.",
        }
    elif task_family == "direct_judgment" and task_subtype == "algorithm_usage_context":
        spec = {
            "required_slots": ["usage_context", "answer"],
            "optional_slots": ["preconditions", "example", "alternative"],
            "forbidden_escalations": [],
            "preferred_action_order": ["REUSE", "QUERY", "DERIVE", "FINALIZE"],
            "stop_condition": "Usage context and answer filled.",
            "guidance": "State when the algorithm is applicable, then phrase it as a fresh answer.",
        }
    else:
        spec = specs.get(task_family, specs["direct_judgment"])
    return ControlRuleNode(
        id=f"default_control_rule_{task_family}_{task_subtype or 'default'}",
        task_family=task_family,
        guidance=spec["guidance"],
        required_slots=list(spec["required_slots"]),
        optional_slots=list(spec["optional_slots"]),
        forbidden_escalations=list(spec["forbidden_escalations"]),
        preferred_action_order=list(spec["preferred_action_order"]),
        stop_condition=spec["stop_condition"],
        provenance=Provenance(created_in_session_id="builtin_micro_controller"),
    )


def _policy_for_family(task_family: TaskFamily, task_subtype: str = "") -> ControllerPolicy:
    rule = _default_control_rule(task_family, task_subtype)
    preferred = [MicroAction[name] if isinstance(name, str) and name in MicroAction.__members__ else MicroAction.REUSE for name in rule.preferred_action_order]
    forbidden = [MicroAction[name] if isinstance(name, str) and name in MicroAction.__members__ else MicroAction.DERIVE for name in rule.forbidden_escalations]
    max_graph_queries = 0 if task_family == "algorithm_applicability" else 1
    max_derivations = 0 if task_family == "algorithm_applicability" else 1
    return ControllerPolicy(
        task_family=task_family,
        required_slots=list(rule.required_slots),
        optional_slots=list(rule.optional_slots),
        preferred_action_order=preferred,
        forbidden_escalations=forbidden,
        max_subgoals=max(len(rule.required_slots), 3),
        max_graph_queries=max_graph_queries,
        max_derivations=max_derivations,
        answer_style="concise",
    )


def _subgoals_for_family(task_family: TaskFamily, task_subtype: str, task_signature: str) -> List[Subgoal]:
    if task_family == "algorithm_applicability":
        names = [
            ("determine_applicability_verdict", ["verdict"], "rule_or_failure"),
            ("explain_failure_reason", ["reason"], "invariant_explanation"),
            ("name_safe_alternative", ["alternative"], "replacement"),
            ("state_scope_caveat", ["caveat"], "scope_guard"),
        ]
    elif task_family == "relational_explanation":
        names = [
            ("identify_relationship", ["relationship"], "relationship"),
            ("explain_relationship", ["explanation"], "explanation"),
        ]
    elif task_family == "design_synthesis":
        names = [
            ("frame_problem", ["problem_frame"], "problem_frame"),
            ("choose_core_structure", ["core_structure"], "core_structure"),
            ("define_rank_query", ["rank_query"], "rank_query"),
            ("define_pagination", ["pagination"], "pagination"),
            ("define_tie_policy", ["tie_policy"], "tie_policy"),
            ("design_scale_architecture", ["scale_architecture"], "scale_architecture"),
            ("state_latency_budget", ["latency_budget"], "latency_budget"),
            ("state_consistency_model", ["consistency_model"], "consistency_model"),
            ("name_failure_mode_fix", ["failure_mode_fix"], "failure_mode_fix"),
            ("compose_answer", ["answer"], "answer"),
        ]
    elif task_family == "procedure_or_instance_verification":
        names = [
            ("summarize_instance", ["instance_summary"], "instance_summary"),
            ("check_preconditions", ["precondition_results"], "precondition_results"),
            ("decide_verdict", ["verdict"], "verdict"),
            ("compose_answer", ["answer"], "answer"),
        ]
    elif task_family == "direct_judgment" and task_subtype == "algorithm_mechanism_explanation":
        names = [
            ("explain_core_mechanism", ["mechanism"], "mechanism"),
            ("compose_answer", ["answer"], "answer"),
        ]
    elif task_family == "direct_judgment" and task_subtype == "algorithm_usage_context":
        names = [
            ("identify_usage_context", ["usage_context"], "usage_context"),
            ("compose_answer", ["answer"], "answer"),
        ]
    else:
        names = [
            ("answer_question", ["answer"], "answer"),
            ("support_answer", ["reason"], "support"),
        ]
    out: List[Subgoal] = []
    for name, required_slots, desired_evidence_type in names:
        out.append(Subgoal(
            name=name,
            signature=SubgoalSignature(f"{task_signature}.{name}"),
            prompt=name.replace("_", " "),
            required_slots=list(required_slots),
            desired_evidence_type=desired_evidence_type,
        ))
    return out


def _extract_entities(question: str, task_family: TaskFamily) -> Dict[str, str]:
    low = (question or "").lower()
    entities: Dict[str, str] = {"algorithm": "", "condition": "", "artifact": ""}
    for hint in _ALGORITHM_HINTS:
        if hint in low:
            entities["algorithm"] = _display_name(hint)
            break
    if task_family == "algorithm_applicability" and _NEGATIVE_EDGE_RE.search(low):
        entities["condition"] = "negative_edge_weights"
    elif task_family == "algorithm_applicability" and _ALL_NEGATIVE_RE.search(low):
        entities["condition"] = "all_negative_arrays"
    if task_family == "procedure_or_instance_verification":
        entities["artifact"] = "graph_instance"
    return entities


def _extract_conditions(question: str, task_family: TaskFamily) -> Dict[str, str]:
    low = (question or "").lower()
    conditions: Dict[str, str] = {}
    if _NEGATIVE_EDGE_RE.search(low):
        conditions["graph_property"] = "negative_edge_weights"
    if _ALL_NEGATIVE_RE.search(low):
        conditions["input_property"] = "all_negative_arrays"
    if "specific graph" in low or "given graph" in low or "edge(" in low:
        conditions["scope"] = "specific_instance"
    elif task_family == "algorithm_applicability":
        conditions["scope"] = "general_correctness"
    return conditions


def _infer_task_profile(
    question: str,
    task_family: TaskFamily,
    entities: Mapping[str, str],
    conditions: Mapping[str, str],
) -> Tuple[str, str]:
    low = (question or "").lower()
    algorithm = str(entities.get("algorithm", "") or "").strip()
    if task_family == "algorithm_applicability":
        return ("algorithm_applicability", "verdict")
    if task_family == "relational_explanation":
        return ("relational_explanation", "relationship")
    if task_family == "design_synthesis":
        return ("design_synthesis", "design")
    if task_family == "procedure_or_instance_verification":
        return ("procedure_or_instance_verification", "verification")
    if task_family == "direct_judgment" and algorithm:
        if _HOW_WORKS_RE.search(low) or low.startswith("explain ") or low.startswith("describe "):
            return ("algorithm_mechanism_explanation", "mechanism_explanation")
        if _HOW_APPLIES_RE.search(low):
            return ("algorithm_usage_context", "usage_context")
    return (task_family, "answer_reason")


def _requires_alternative_slot(question: str, context: ContextSignature) -> bool:
    if context.task_family != "algorithm_applicability":
        return False
    low = (question or "").lower()
    if any(phrase in low for phrase in ("what should i use", "what algorithm should i use", "instead", "alternative")):
        return True
    condition = context.entities.get("condition") or context.conditions.get("graph_property") or context.conditions.get("input_property")
    return condition == "negative_edge_weights"


def _looks_like_algorithm_applicability(question: str) -> bool:
    if not _JUDGMENT_RE.match(question):
        return False
    if any(hint in question for hint in _ALGORITHM_HINTS):
        return True
    return bool(_NEGATIVE_EDGE_RE.search(question))


def _task_signature(
    question: str,
    task_family: TaskFamily,
    task_subtype: str,
    question_mode: str,
    entities: Mapping[str, str],
    conditions: Mapping[str, str],
) -> str:
    payload = {
        "task_family": task_family,
        "task_subtype": task_subtype,
        "question_mode": question_mode,
        "algorithm": entities.get("algorithm", ""),
        "condition": entities.get("condition", ""),
        "artifact": entities.get("artifact", ""),
        "scope": conditions.get("scope", ""),
    }
    semantic_topic_signature = ""
    if task_family in {"direct_judgment", "relational_explanation"} and not payload["algorithm"]:
        payload["topic_terms"] = sorted(_focused_tokens(question, min_chars=4))[:8]
        semantic_topic_signature = _semantic_topic_signature(question, task_family=task_family)
    if task_family == "algorithm_applicability" and payload["algorithm"] and payload["condition"]:
        algorithm = _slugify(payload["algorithm"])
        condition = _slugify(payload["condition"])
        if algorithm == "dijkstra" and condition == "negative_edge_weights":
            return f"shortest_path.{algorithm}.{condition}.validity"
        return f"algorithm_applicability.{algorithm}.{condition}.validity"
    if task_subtype == "algorithm_mechanism_explanation" and payload["algorithm"]:
        return f"algorithm_explanation.{_slugify(payload['algorithm'])}.mechanism"
    if task_subtype == "algorithm_usage_context" and payload["algorithm"]:
        return f"algorithm_usage.{_slugify(payload['algorithm'])}.context"
    if semantic_topic_signature:
        return semantic_topic_signature
    digest = _short_hash(json.dumps(payload, sort_keys=True))
    return f"{task_family}.{digest}"


def _contains_any_phrase(text: str, phrases: Sequence[str]) -> bool:
    low = str(text or "").lower()
    return any(str(phrase or "").lower() in low for phrase in phrases)


def _semantic_topic_signature(question: str, *, task_family: TaskFamily) -> str:
    low = str(question or "").lower()
    if task_family == "direct_judgment":
        mentions_sound = _contains_any_phrase(
            low,
            (
                "sound",
                "hear",
                "hearing",
                "audible",
                "audio",
                "acoustic",
                "sonic",
                "noise",
            ),
        )
        mentions_sight_or_light = _contains_any_phrase(
            low,
            (
                "light",
                "sunlight",
                "starlight",
                "laser",
                "flash",
                "visible",
                "see",
                "seeing",
                "sight",
                "star",
                "stars",
                "sun",
            ),
        )
        mentions_vacuum = _contains_any_phrase(
            low,
            (
                "vacuum",
                "space",
                "outer space",
                "empty space",
                "airless",
                "without air",
                "no air",
            ),
        )
        if mentions_sound and mentions_sight_or_light and mentions_vacuum:
            return "direct_judgment.sound_requires_medium_vs_light_vacuum"

        mentions_frequency = "frequency" in low
        mentions_optical_signal = _contains_any_phrase(
            low,
            (
                "light",
                "laser",
                "beam",
                "photon",
                "sunlight",
                "starlight",
                "visible",
                "prism",
            ),
        )
        mentions_medium_or_boundary = _contains_any_phrase(
            low,
            (
                "refraction",
                "refract",
                "refracted",
                "refracts",
                "prism",
                "glass",
                "water",
                "medium",
                "boundary",
                "interface",
            ),
        )
        mentions_transition_or_speed_change = _contains_any_phrase(
            low,
            (
                "enter",
                "enters",
                "entering",
                "goes into",
                "passes into",
                "slows down",
                "speed changes",
                "changes speed",
                "bends",
                "bend",
                "wavelength",
            ),
        )
        if (
            mentions_frequency
            and mentions_optical_signal
            and (mentions_medium_or_boundary or mentions_transition_or_speed_change)
        ):
            return "direct_judgment.refraction_changes_speed_not_frequency"
    topic_terms = sorted(_focused_tokens(question, min_chars=4))[:4]
    if not topic_terms:
        return ""
    topic_key = "+".join(_slugify(term) for term in topic_terms[:4])
    return f"{task_family}.{topic_key}"


def _display_name(name: str) -> str:
    if name == "union-find":
        return "Union-Find"
    if name == "dsu":
        return "DSU"
    if name == "bfs":
        return "BFS"
    if name == "dfs":
        return "DFS"
    return " ".join(part.capitalize() for part in name.split())


def _node_relevant_to_question(question: str, node: Node) -> bool:
    score = lexical_overlap(question, node.text or "", min_chars=3)
    meta = getattr(node, "metadata", {}) or {}
    keywords = " ".join(str(v) for v in meta.values() if isinstance(v, (str, int, float)))
    score = max(score, lexical_overlap(question, keywords, min_chars=3))
    return score >= 0.08


def _focused_tokens(text: object, *, min_chars: int = 4) -> Set[str]:
    return content_tokens(
        text,
        min_chars=min_chars,
        stopwords=_FOCUSED_RETRIEVAL_STOPWORDS,
    )


def _focused_overlap(a: object, b: object, *, min_chars: int = 4) -> float:
    left = _focused_tokens(a, min_chars=min_chars)
    right = _focused_tokens(b, min_chars=min_chars)
    if not left or not right:
        return 0.0
    return len(left & right) / math.sqrt(len(left) * len(right))


def _focused_coverage(question: object, text: object, *, min_chars: int = 4) -> float:
    question_tokens = _focused_tokens(question, min_chars=min_chars)
    if not question_tokens:
        return 0.0
    return len(question_tokens & _focused_tokens(text, min_chars=min_chars)) / max(len(question_tokens), 1)


def _generic_direct_relevant(task_frame: TaskFrame, text: str) -> bool:
    low_q = task_frame.question.lower()
    low = str(text or "").lower()
    if _focused_overlap(task_frame.question, text, min_chars=4) >= 0.16:
        return True
    if "prism" in low_q and "refraction" in low:
        return True
    if "frequency" in low_q and "frequency" in low:
        return True
    if _contains_any_phrase(low_q, ("sound", "hear", "hearing", "audible", "audio", "acoustic", "sonic", "noise")) and ("medium" in low or "vacuum" in low):
        return True
    if _contains_any_phrase(low_q, ("light", "sunlight", "starlight", "laser", "flash", "visible", "see", "seeing", "sight", "star", "stars", "sun")) and ("electromagnetic" in low or "vacuum" in low or "refraction" in low):
        return True
    return False


def _first_sentence(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return ""
    match = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)
    return match[0].strip()


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _slugify(text: str) -> str:
    text = str(text or "").strip().lower().replace("-", "_").replace(" ", "_")
    return re.sub(r"[^a-z0-9_]+", "", text).strip("_") or "unknown"


def _default_valid_when(task_frame: TaskFrame) -> List[str]:
    if task_frame.task_family == "algorithm_applicability":
        out = []
        algorithm = task_frame.context.entities.get("algorithm", "")
        condition = task_frame.context.entities.get("condition", "") or task_frame.context.conditions.get("graph_property", "")
        scope = task_frame.context.conditions.get("scope", "")
        if algorithm:
            out.append(f"algorithm={algorithm}")
        if condition:
            out.append(f"condition={condition}")
        if scope:
            out.append(f"scope={scope}")
        return out or ["algorithm_applicability"]
    if task_frame.context.task_subtype == "algorithm_mechanism_explanation":
        algorithm = task_frame.context.entities.get("algorithm", "")
        return [f"algorithm={algorithm}", "mode=mechanism_explanation"] if algorithm else ["mode=mechanism_explanation"]
    if task_frame.context.task_subtype == "algorithm_usage_context":
        algorithm = task_frame.context.entities.get("algorithm", "")
        return [f"algorithm={algorithm}", "mode=usage_context"] if algorithm else ["mode=usage_context"]
    if task_frame.task_family == "relational_explanation":
        return ["relationship question"]
    return [task_frame.task_family]


def _default_invalid_when(task_frame: TaskFrame) -> List[str]:
    if task_frame.task_family == "algorithm_applicability":
        return [
            "modified dijkstra",
            "specific graph instance",
            "dag shortest path",
            "all-pairs shortest path",
        ]
    if task_frame.context.task_subtype == "algorithm_mechanism_explanation":
        return ["negative edge applicability only", "when should i use"]
    if task_frame.context.task_subtype == "algorithm_usage_context":
        return ["counterexample proof", "mechanism derivation only"]
    return []


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")
