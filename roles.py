"""
Role detection for Stage 3 evidence ranking.

Two parallel taxonomies:
  plan-step EXPECTS: what reasoning role(s) the focus plan_step is asking for
  node     PROVIDES: what reasoning role(s) a candidate node can play

Both are sets (not single strings) because nodes can play multiple roles
(e.g., a Bellman-Ford evidence node is both truth and condition; a
"long long needed" node is both condition and application).

Active for Stage 3 scoring bonus: truth, misconception, mechanism, condition, bridge.
Defined but currently inactive in scoring: application, hypothesis, hub.
"""

from __future__ import annotations

import re
from typing import Any, Set


# ---------------------------------------------------------------------------
# Plan-step role detection (regex on focus plan_step text)
# ---------------------------------------------------------------------------

_PLAN_REFUTE = re.compile(
    r"\b(refute|misconception|debunk|wrong(ly)?|incorrect|false|fallac|"
    r"is\s+it\s+(true|correct|safe|valid)|"
    r"is\s+(this|that)\s+(true|correct|safe|valid)|"
    r"can\s+\w+\s+be\s+trusted|"
    r"determine\s+whether|validate)\b",
    re.IGNORECASE,
)
_PLAN_MECHANISM = re.compile(
    r"\b(why|explain|because|mechanism|principle|law|theorem|reason|"
    r"how\s+does|cause[ds]?)\b",
    re.IGNORECASE,
)
_PLAN_CONDITION = re.compile(
    r"\b(if|when|given|requires?|condition|prerequisite|precondition|"
    r"necessary|only\s+if|threshold|determines?)\b",
    re.IGNORECASE,
)
_PLAN_COMPARE = re.compile(
    r"\b(compare|contrast|distinguish|distinct|versus|vs)\b",
    re.IGNORECASE,
)


def detect_plan_step_roles(text: str) -> Set[str]:
    """Return the set of evidence roles a plan_step is asking for."""
    roles: Set[str] = set()
    t = text or ""
    if _PLAN_REFUTE.search(t):
        roles.update({"misconception", "truth"})
    if _PLAN_MECHANISM.search(t):
        roles.add("mechanism")
    if _PLAN_CONDITION.search(t):
        roles.add("condition")
    if _PLAN_COMPARE.search(t):
        roles.update({"truth", "misconception"})
    if not roles:
        roles.add("truth")
    return roles


# ---------------------------------------------------------------------------
# Node role detection (id pattern + text pattern; multi-label)
# ---------------------------------------------------------------------------

_NODE_MECHANISM = re.compile(
    r"\b(law|principle|theorem|equation|formula|invariant|"
    r"because|cause[ds]?|due\s+to|leads\s+to|results\s+in|implies)\b",
    re.IGNORECASE,
)
_NODE_CONDITION = re.compile(
    r"\b(if|when|given|requires?|condition|provided\s+that|only\s+if|"
    r"prerequisite|necessary)\b",
    re.IGNORECASE,
)


def detect_node_roles(node: Any) -> Set[str]:
    """Return all roles a candidate node can play. Multi-label by design.

    A node is identified by its id suffix conventions in our graphs:
      _false           -> misconception (NOT truth)
      _hyp             -> hypothesis    (NOT truth — unverified)
      _hub / _summary  -> hub           (NOT truth — too generic)
      _apply           -> application + truth
      _bridge          -> bridge + truth
      (everything else) -> truth        (baseline for real evidence)

    PLUS additive text signals for mechanism / condition that overlay on
    top, so a Newton's-second-law node gets {truth, mechanism} and a
    "long long needed if sums exceed X" node gets {truth, condition,
    application}.

    Critical rule: nodes whose id suffix indicates NOT-truth (_false,
    _hyp, _hub, _summary) do NOT get the truth baseline. Those are the
    only id suffixes that disqualify a node from being treated as direct
    factual evidence. Everything else, including bridges and concrete
    applications, is grounded enough to count as truth.
    """
    roles: Set[str] = set()
    nid = (getattr(node, "id", "") or "").lower()
    txt = (getattr(node, "text", "") or "").lower()

    is_not_truth = False
    if nid.endswith("_false"):
        roles.add("misconception")
        is_not_truth = True
    if nid.endswith("_hyp"):
        roles.add("hypothesis")
        is_not_truth = True
    if nid.endswith("_hub") or nid.endswith("_summary"):
        roles.add("hub")
        is_not_truth = True
    if nid.endswith("_apply"):
        roles.add("application")
    if nid.endswith("_bridge"):
        roles.add("bridge")

    if _NODE_MECHANISM.search(txt):
        roles.add("mechanism")
    if _NODE_CONDITION.search(txt):
        roles.add("condition")

    if not is_not_truth:
        roles.add("truth")
    return roles


# ---------------------------------------------------------------------------
# Active set for Stage 3 scoring bonus.
# ---------------------------------------------------------------------------

ACTIVE_STAGE3_ROLES: Set[str] = {
    "truth", "misconception", "mechanism", "condition", "bridge",
}
