"""Consolidation: session subgraph → long-term graph promotion.

End-of-session pipeline. For each procedure / failure_pattern / session
object created during the session, decide one of:

  promote      — meets all three gates; eligible for long-term storage
  keep_in_pool — useful candidate but not yet validated enough; stays
                 in cold-storage session graphs for future citation accrual
  expire       — clearly unwanted; safe to drop

Three gates a node must clear to be promoted (PHASE1_PLAN.md §8):

  1. Citation count (across distinct sessions) >= promotion_threshold (M)
  2. At least one validating worked example
  3. All depends_on nodes are themselves consolidated to long-term

Citation counting across sessions requires the caller to track prior
counts; we pass them in as an optional dict. Until the substrate is
running for real, prior counts default to 0 — meaning ONLY nodes that
already have validating_examples can be promoted from their own
inaugural session if the threshold is 1, otherwise nothing promotes
on the first run. That's intentional.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Set

from reasoning.schemas import SessionSubgraph


Decision = Literal["promote", "keep_in_pool", "expire"]


@dataclass
class ConsolidationDecision:
    node_id: str
    node_type: str
    decision: Decision
    reason: str
    # Optional: full node dict for the caller to copy into long-term storage.
    # None when decision != "promote".
    node_data: Optional[Dict[str, Any]] = None
    # Diagnostic breakdown of why each gate passed/failed
    gate_results: Dict[str, bool] = field(default_factory=dict)


class Consolidator:
    """Decision-producing layer. Does NOT mutate the long-term graph —
    that's the caller's responsibility, given the decision list.
    """

    def __init__(
        self,
        promotion_threshold: int = 3,
        consolidated_node_ids: Optional[Set[str]] = None,
    ):
        """
        promotion_threshold:
            Minimum cross-session citation count to clear gate 1.
            Default 3 (resolved decision).
        consolidated_node_ids:
            IDs of nodes already in long-term storage. Used to check
            depends_on integrity (gate 3). If None, treated as the
            empty set (nothing already consolidated).
        """
        self.promotion_threshold = promotion_threshold
        self.consolidated_node_ids: Set[str] = consolidated_node_ids or set()

    def consolidate(
        self,
        session: SessionSubgraph,
        prior_citation_counts: Optional[Dict[str, int]] = None,
    ) -> List[ConsolidationDecision]:
        """Produce a decision per session-created node.

        prior_citation_counts: map from node_id to count of *previous*
        sessions that cited the node. The CURRENT session counts as
        +1 since the node lived in this session's subgraph.
        """
        prior = dict(prior_citation_counts or {})
        decisions: List[ConsolidationDecision] = []

        for node_id, node_data in session.nodes.items():
            ntype = node_data.get("node_type") or ""
            if ntype not in ("procedure", "failure_pattern", "session_object"):
                continue                            # not a substrate-managed node type

            decisions.append(self._decide(node_id, node_data, prior))

        return decisions

    # ---- gates -------------------------------------------------------- #

    def _decide(
        self,
        node_id: str,
        node_data: Dict[str, Any],
        prior_counts: Dict[str, int],
    ) -> ConsolidationDecision:
        ntype = node_data.get("node_type") or ""

        # session_object nodes are inherently per-session — they are never
        # promoted; they may carry useful state for the procedure but the
        # procedure is the abstraction that gets promoted, not the instance.
        if ntype == "session_object":
            return ConsolidationDecision(
                node_id=node_id, node_type=ntype,
                decision="expire",
                reason="session_object is per-session by design; the underlying procedure is the promotable abstraction",
                gate_results={"session_scoped": True},
            )

        # Gate 1: citation threshold (prior + current session)
        prior_count = prior_counts.get(node_id, 0)
        total_count = prior_count + 1
        gate1_pass = total_count >= self.promotion_threshold

        # Gate 2: validating example
        prov = node_data.get("provenance") or {}
        validating = prov.get("validating_examples") or []
        gate2_pass = len(validating) >= 1

        # Gate 3: all depends_on are themselves consolidated
        deps = prov.get("depends_on") or []
        missing_deps = [d for d in deps if d not in self.consolidated_node_ids]
        gate3_pass = len(missing_deps) == 0

        gate_results = {
            "citation_threshold": gate1_pass,
            "validating_example": gate2_pass,
            "deps_consolidated": gate3_pass,
        }

        if gate1_pass and gate2_pass and gate3_pass:
            return ConsolidationDecision(
                node_id=node_id, node_type=ntype,
                decision="promote",
                reason=(
                    f"all gates pass: citations={total_count}/"
                    f"{self.promotion_threshold}, has validating example, "
                    f"deps consolidated"
                ),
                node_data=node_data,
                gate_results=gate_results,
            )

        # Otherwise: keep in pool for now (never auto-expire — that's a
        # Phase-2 concern once we have decay/deprecation signals)
        missing = [name for name, passed in gate_results.items() if not passed]
        return ConsolidationDecision(
            node_id=node_id, node_type=ntype,
            decision="keep_in_pool",
            reason=f"gates not yet passed: {', '.join(missing)}",
            gate_results=gate_results,
        )
