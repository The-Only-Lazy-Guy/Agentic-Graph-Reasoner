"""Deterministic adaptive plan-tree substrate - Phase 3D.

This module deliberately contains no LLM calls. It gives the reasoning loop a
stable checkpoint tree that later prompt-mode integration can drive.
"""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence

from reasoning.activation import GraphTaskFrame, evaluate_coverage
from reasoning.schemas import SessionEdge
from reasoning.session_subgraph import SessionSubgraphController


PlanMode = Literal["focus", "plan", "execute", "check", "repair", "finalize"]
PlanStatus = Literal["pending", "active", "passed", "failed", "abandoned"]
FailureScope = Literal["local_step", "algorithm_choice", "task_interpretation", "unknown"]

PLAN_EDGE_RELATIONS = {
    "plan_child",
    "plan_revision_of",
    "backtracked_to",
    "checked_by",
    "failed_because",
    "supports_plan",
}


class PlanningBudgetExceeded(RuntimeError):
    """Raised when deterministic plan-tree revision/backtrack caps are hit."""


def _gen_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class PlanNode:
    node_id: str
    parent_id: Optional[str]
    goal: str
    hypothesis: str
    mode: PlanMode
    status: PlanStatus = "pending"
    checkpoint_quality: float = 0.5
    failure_reason: Optional[str] = None
    evidence_ids: List[str] = field(default_factory=list)
    created_step: int = 0
    node_type: str = "plan_node"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "PlanNode":
        return PlanNode(
            node_id=str(d["node_id"]),
            parent_id=str(d["parent_id"]) if d.get("parent_id") is not None else None,
            goal=str(d.get("goal", "")),
            hypothesis=str(d.get("hypothesis", "")),
            mode=d.get("mode", "plan"),  # type: ignore[arg-type]
            status=d.get("status", "pending"),  # type: ignore[arg-type]
            checkpoint_quality=float(d.get("checkpoint_quality", 0.5)),
            failure_reason=d.get("failure_reason"),
            evidence_ids=[str(x) for x in d.get("evidence_ids", [])],
            created_step=int(d.get("created_step", 0)),
            node_type=str(d.get("node_type", "plan_node")),
        )


@dataclass
class PlanCheckResult:
    checked_node_id: str
    passed: bool
    failure_scope: FailureScope
    failed_requirements: List[str] = field(default_factory=list)
    suggested_backtrack_node_id: Optional[str] = None
    reason: str = ""
    check_id: str = field(default_factory=lambda: _gen_id("plan_check"))
    node_type: str = "plan_check"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "PlanCheckResult":
        return PlanCheckResult(
            checked_node_id=str(d["checked_node_id"]),
            passed=bool(d.get("passed", False)),
            failure_scope=d.get("failure_scope", "unknown"),  # type: ignore[arg-type]
            failed_requirements=[str(x) for x in d.get("failed_requirements", [])],
            suggested_backtrack_node_id=(
                str(d["suggested_backtrack_node_id"])
                if d.get("suggested_backtrack_node_id") is not None
                else None
            ),
            reason=str(d.get("reason", "")),
            check_id=str(d.get("check_id") or _gen_id("plan_check")),
            node_type=str(d.get("node_type", "plan_check")),
        )


@dataclass
class PlanState:
    session_id: str
    root_node_id: str
    active_node_id: str
    revision_count: int = 0
    max_revisions: int = 3
    backtrack_count: int = 0
    max_backtracks: int = 3
    max_depth: int = 6
    finalized: bool = False
    last_failure_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "PlanState":
        return PlanState(
            session_id=str(d["session_id"]),
            root_node_id=str(d["root_node_id"]),
            active_node_id=str(d["active_node_id"]),
            revision_count=int(d.get("revision_count", 0)),
            max_revisions=int(d.get("max_revisions", 3)),
            backtrack_count=int(d.get("backtrack_count", 0)),
            max_backtracks=int(d.get("max_backtracks", 3)),
            max_depth=int(d.get("max_depth", 6)),
            finalized=bool(d.get("finalized", False)),
            last_failure_reason=d.get("last_failure_reason"),
        )


@dataclass(frozen=True)
class BacktrackCandidateScore:
    node_id: str
    score: float
    distance_from_failed_node: int
    components: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AdaptivePlanTree:
    """Session-local checkpoint tree with deterministic revision policy."""

    def __init__(
        self,
        session_id: str,
        root_goal: str,
        root_hypothesis: str = "",
        *,
        max_revisions: int = 3,
        max_backtracks: int = 3,
        max_depth: int = 6,
        root_node_id: Optional[str] = None,
    ) -> None:
        root_id = root_node_id or _gen_id("plan")
        root = PlanNode(
            node_id=root_id,
            parent_id=None,
            goal=root_goal,
            hypothesis=root_hypothesis,
            mode="focus",
            status="active",
            checkpoint_quality=0.75,
            created_step=0,
        )
        self.state = PlanState(
            session_id=session_id,
            root_node_id=root_id,
            active_node_id=root_id,
            max_revisions=max_revisions,
            max_backtracks=max_backtracks,
            max_depth=max_depth,
        )
        self.nodes: Dict[str, PlanNode] = {root_id: root}
        self.checks: Dict[str, PlanCheckResult] = {}
        self.edges: List[SessionEdge] = []

    def add_child(
        self,
        parent_id: str,
        *,
        goal: str,
        hypothesis: str,
        mode: PlanMode = "plan",
        checkpoint_quality: float = 0.5,
        evidence_ids: Optional[Sequence[str]] = None,
        activate: bool = True,
        relation: str = "plan_child",
        node_id: Optional[str] = None,
    ) -> str:
        self._require_node(parent_id)
        if self.depth(parent_id) + 1 > self.state.max_depth:
            raise PlanningBudgetExceeded(
                f"Adding a child under {parent_id!r} would exceed max_depth={self.state.max_depth}"
            )
        child_id = node_id or _gen_id("plan")
        if child_id in self.nodes:
            raise ValueError(f"Plan node {child_id!r} already exists")
        node = PlanNode(
            node_id=child_id,
            parent_id=parent_id,
            goal=goal,
            hypothesis=hypothesis,
            mode=mode,
            status="active" if activate else "pending",
            checkpoint_quality=max(0.0, min(1.0, checkpoint_quality)),
            evidence_ids=[str(x) for x in evidence_ids or []],
            created_step=len(self.nodes),
        )
        if activate:
            self._deactivate_current()
            self.state.active_node_id = child_id
        self.nodes[child_id] = node
        self.edges.append(SessionEdge(parent_id, child_id, relation, {"provider": "phase3d-adaptive-planning"}))
        return child_id

    def record_check(self, check: PlanCheckResult) -> str:
        self._require_node(check.checked_node_id)
        self.checks[check.check_id] = check
        self.edges.append(SessionEdge(
            check.checked_node_id,
            check.check_id,
            "checked_by",
            {"provider": "phase3d-adaptive-planning", "passed": check.passed},
        ))
        if not check.passed:
            self.edges.append(SessionEdge(
                check.checked_node_id,
                check.check_id,
                "failed_because",
                {"provider": "phase3d-adaptive-planning", "failure_scope": check.failure_scope},
            ))
        return check.check_id

    def mark_passed(self, node_id: str) -> None:
        self._require_node(node_id)
        node = self.nodes[node_id]
        node.status = "passed"
        node.failure_reason = None

    def mark_failed(self, node_id: str, reason: str, *, abandon: bool = False) -> None:
        self._require_node(node_id)
        node = self.nodes[node_id]
        node.status = "abandoned" if abandon else "failed"
        node.failure_reason = reason
        self.state.last_failure_reason = reason

    def choose_backtrack_node(
        self,
        failed_node_id: str,
        check: PlanCheckResult,
    ) -> BacktrackCandidateScore:
        self._require_node(failed_node_id)
        if check.suggested_backtrack_node_id and check.suggested_backtrack_node_id in self.nodes:
            suggested = check.suggested_backtrack_node_id
            distance = self.distance_to_ancestor(failed_node_id, suggested)
            return BacktrackCandidateScore(
                node_id=suggested,
                score=1.0,
                distance_from_failed_node=distance if distance >= 0 else 0,
                components={"suggested_backtrack": 1.0},
            )

        ancestors = self.ancestors(failed_node_id)
        if not ancestors:
            return BacktrackCandidateScore(
                node_id=self.state.root_node_id,
                score=self.nodes[self.state.root_node_id].checkpoint_quality,
                distance_from_failed_node=0,
                components={"root_fallback": 1.0},
            )

        scores = [self._score_candidate(failed_node_id, candidate_id, check) for candidate_id in ancestors]
        return max(scores, key=lambda s: (s.score, -s.distance_from_failed_node))

    def revise_from_failure(
        self,
        failed_node_id: str,
        check: PlanCheckResult,
        *,
        new_goal: str,
        new_hypothesis: str,
        mode: PlanMode = "execute",
        checkpoint_quality: float = 0.5,
        evidence_ids: Optional[Sequence[str]] = None,
        node_id: Optional[str] = None,
    ) -> str:
        if self.state.revision_count >= self.state.max_revisions:
            raise PlanningBudgetExceeded(f"max_revisions={self.state.max_revisions} exhausted")
        if self.state.backtrack_count >= self.state.max_backtracks:
            raise PlanningBudgetExceeded(f"max_backtracks={self.state.max_backtracks} exhausted")

        self.record_check(check)
        self.mark_failed(failed_node_id, check.reason, abandon=True)
        checkpoint = self.choose_backtrack_node(failed_node_id, check)
        self.edges.append(SessionEdge(
            failed_node_id,
            checkpoint.node_id,
            "backtracked_to",
            {
                "provider": "phase3d-adaptive-planning",
                "score": checkpoint.score,
                "failure_scope": check.failure_scope,
            },
        ))
        self.state.revision_count += 1
        self.state.backtrack_count += 1
        revised_id = self.add_child(
            checkpoint.node_id,
            goal=new_goal,
            hypothesis=new_hypothesis,
            mode=mode,
            checkpoint_quality=checkpoint_quality,
            evidence_ids=evidence_ids,
            activate=True,
            node_id=node_id,
        )
        self.edges.append(SessionEdge(
            failed_node_id,
            revised_id,
            "plan_revision_of",
            {"provider": "phase3d-adaptive-planning"},
        ))
        return revised_id

    def check_answer_coverage(
        self,
        node_id: str,
        frame: GraphTaskFrame,
        answer: str,
        *,
        critical_priority: int = 80,
    ) -> PlanCheckResult:
        self._require_node(node_id)
        coverage = evaluate_coverage(frame, answer)
        items_by_id = {item.item_id: item for item in frame.all_items()}
        missed_critical = [
            items_by_id[item_id]
            for item_id in coverage["missed_item_ids"]
            if item_id in items_by_id and items_by_id[item_id].priority >= critical_priority
        ]
        if not missed_critical:
            return PlanCheckResult(
                checked_node_id=node_id,
                passed=True,
                failure_scope="local_step",
                reason="All critical task-frame items are covered.",
            )
        return PlanCheckResult(
            checked_node_id=node_id,
            passed=False,
            failure_scope="local_step",
            failed_requirements=[item.text for item in missed_critical],
            suggested_backtrack_node_id=self.nodes[node_id].parent_id,
            reason="Final answer missed critical task-frame items.",
        )

    def try_finalize(
        self,
        node_id: str,
        frame: GraphTaskFrame,
        answer: str,
        *,
        critical_priority: int = 80,
    ) -> PlanCheckResult:
        check = self.check_answer_coverage(node_id, frame, answer, critical_priority=critical_priority)
        self.record_check(check)
        if check.passed:
            self.mark_passed(node_id)
            self.state.finalized = True
        else:
            self.mark_failed(node_id, check.reason)
        return check

    def ancestors(self, node_id: str) -> List[str]:
        self._require_node(node_id)
        out: List[str] = []
        current = self.nodes[node_id].parent_id
        while current is not None:
            out.append(current)
            current = self.nodes[current].parent_id
        return out

    def depth(self, node_id: str) -> int:
        return len(self.ancestors(node_id))

    def distance_to_ancestor(self, node_id: str, ancestor_id: str) -> int:
        for distance, candidate in enumerate(self.ancestors(node_id), start=1):
            if candidate == ancestor_id:
                return distance
        return -1

    def children_of(self, node_id: str) -> List[str]:
        self._require_node(node_id)
        return [n.node_id for n in self.nodes.values() if n.parent_id == node_id]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.to_dict(),
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "checks": {check_id: check.to_dict() for check_id, check in self.checks.items()},
            "edges": [edge.to_dict() for edge in self.edges],
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "AdaptivePlanTree":
        state = PlanState.from_dict(d["state"])
        tree = AdaptivePlanTree(
            state.session_id,
            root_goal="",
            root_node_id=state.root_node_id,
            max_revisions=state.max_revisions,
            max_backtracks=state.max_backtracks,
            max_depth=state.max_depth,
        )
        tree.state = state
        tree.nodes = {str(k): PlanNode.from_dict(v) for k, v in dict(d.get("nodes", {})).items()}
        tree.checks = {str(k): PlanCheckResult.from_dict(v) for k, v in dict(d.get("checks", {})).items()}
        tree.edges = [SessionEdge.from_dict(e) for e in d.get("edges", [])]
        return tree

    def _score_candidate(
        self,
        failed_node_id: str,
        candidate_id: str,
        check: PlanCheckResult,
    ) -> BacktrackCandidateScore:
        node = self.nodes[candidate_id]
        distance = self.distance_to_ancestor(failed_node_id, candidate_id)
        evidence_support = min(0.2, 0.05 * len(node.evidence_ids))
        reusable_context_score = 0.0
        if node.status == "passed":
            reusable_context_score += 0.15
        if node.mode in {"focus", "plan"}:
            reusable_context_score += 0.05
        failure_scope_penalty = self._failure_scope_penalty(node, check.failure_scope)
        if check.failure_scope == "local_step" and candidate_id != self.nodes[failed_node_id].parent_id:
            failure_scope_penalty += 0.5
        distance_penalty = max(0, distance) * 0.03
        repeated_failure_penalty = 0.1 * sum(
            1 for child_id in self.children_of(candidate_id)
            if self.nodes[child_id].status in {"failed", "abandoned"}
        )
        score = (
            node.checkpoint_quality
            + evidence_support
            + reusable_context_score
            - failure_scope_penalty
            - distance_penalty
            - repeated_failure_penalty
        )
        return BacktrackCandidateScore(
            node_id=candidate_id,
            score=score,
            distance_from_failed_node=distance,
            components={
                "checkpoint_quality": node.checkpoint_quality,
                "evidence_support": evidence_support,
                "reusable_context_score": reusable_context_score,
                "failure_scope_penalty": failure_scope_penalty,
                "distance_penalty": distance_penalty,
                "repeated_failure_penalty": repeated_failure_penalty,
            },
        )

    @staticmethod
    def _failure_scope_penalty(node: PlanNode, scope: FailureScope) -> float:
        text = f"{node.goal} {node.hypothesis}".lower()
        if scope == "local_step":
            return 0.0
        if scope == "algorithm_choice":
            if node.mode == "plan" or "choose" in text or "select" in text:
                return 0.0
            return 0.6
        if scope == "task_interpretation":
            if node.parent_id is None or node.mode == "focus":
                return 0.0
            return 0.7
        return 0.0

    def _deactivate_current(self) -> None:
        current = self.nodes.get(self.state.active_node_id)
        if current is not None and current.status == "active":
            current.status = "pending"

    def _require_node(self, node_id: str) -> None:
        if node_id not in self.nodes:
            raise KeyError(f"No plan node {node_id!r}")


def attach_plan_tree_to_session(
    session: SessionSubgraphController,
    tree: AdaptivePlanTree,
) -> None:
    """Project Phase 3D plan nodes/checks into the session subgraph.

    Like Phase 3C activation projection, this is trace structure rather than
    CRUD object state, so it intentionally does not add audit-log entries.
    """
    for node in tree.nodes.values():
        d = node.to_dict()
        d["id"] = node.node_id
        d["text"] = node.goal
        d["metadata"] = {
            "provider": "phase3d-adaptive-planning",
            "hypothesis": node.hypothesis,
            "status": node.status,
        }
        session.subgraph.nodes.setdefault(node.node_id, d)

    for check in tree.checks.values():
        d = check.to_dict()
        d["id"] = check.check_id
        d["text"] = check.reason
        d["metadata"] = {
            "provider": "phase3d-adaptive-planning",
            "passed": check.passed,
            "failure_scope": check.failure_scope,
        }
        session.subgraph.nodes.setdefault(check.check_id, d)

    existing = {(edge.src, edge.dst, edge.relation) for edge in session.subgraph.edges}
    for edge in tree.edges:
        key = (edge.src, edge.dst, edge.relation)
        if key not in existing:
            session.subgraph.edges.append(edge)
            existing.add(key)
