"""ContradictionDetector — flags sibling session_objects that disagree.

Fires on `post_dispatch`. Looks for pairs of session_objects sharing
a `parent_object_id` (siblings in the composition tree) that have at
least one same-named boolean field in their state with opposing
values. Emits ERROR (sticky) so the model can't ignore the contradiction.

CONSERVATIVE PRECONDITIONS (precision over recall):
  - Both objects must be siblings (same parent_object_id)
  - The compared field must be a known-semantic boolean from
    `KNOWN_VERDICT_FIELDS` — comparing arbitrary same-named bool
    fields would risk false positives from unrelated procedures
    that coincidentally share a field name.
  - once_per_session per (object_pair, field) tuple so the same
    contradiction doesn't re-fire.

The intentional gap: contradictions involving non-boolean fields
(e.g., one procedure says "violated: [X]" and another says
"violated: []") are NOT detected here. Those need procedure-pair-
specific reasoning that's out of scope for the generic detector.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from reasoning.meta import MetaContext, MetaProcedure
from reasoning.signals import Signal


# Whitelist of bool field names that genuinely represent a "yes/no"
# semantic verdict across multiple procedures. Adding to this list is
# how new procedure families opt into contradiction detection.
KNOWN_VERDICT_FIELDS = frozenset({
    "safe_to_apply",
    "preconditions_satisfied",
    "has_negative_edge",
    "has_negative_cycle",
    "is_valid",
    "passes",
})


def _extract_bool_verdicts(state: Dict[str, Any]) -> Dict[str, bool]:
    """Pluck out only the known-verdict bool fields from a state dict."""
    out: Dict[str, bool] = {}
    for key, value in state.items():
        if isinstance(value, bool) and key in KNOWN_VERDICT_FIELDS:
            out[key] = value
    return out


def _detect_contradictions(ctx: MetaContext) -> List[Signal]:
    # Group dispatch_outcomes by parent_object_id (only siblings can contradict
    # each other — independent top-level invocations don't count)
    siblings_by_parent: Dict[str, List] = {}
    for outcome in ctx.dispatch_outcomes:
        if outcome.parent_object_id is None or outcome.object_id is None:
            continue
        siblings_by_parent.setdefault(outcome.parent_object_id, []).append(outcome)

    signals: List[Signal] = []
    for parent_id, group in siblings_by_parent.items():
        if len(group) < 2:
            continue

        # Extract verdict fields for each sibling
        siblings_with_verdicts: List[Tuple[str, str, Dict[str, bool]]] = []
        for outcome in group:
            obj = ctx.session.subgraph.nodes.get(outcome.object_id)
            if obj is None:
                continue
            verdicts = _extract_bool_verdicts(obj.get("state", {}) or {})
            if not verdicts:
                continue
            siblings_with_verdicts.append(
                (outcome.object_id, outcome.match.procedure_name, verdicts)
            )

        # Pairwise comparison for opposing values
        for i in range(len(siblings_with_verdicts)):
            for j in range(i + 1, len(siblings_with_verdicts)):
                oid_i, name_i, v_i = siblings_with_verdicts[i]
                oid_j, name_j, v_j = siblings_with_verdicts[j]
                common_keys = set(v_i.keys()) & set(v_j.keys())
                for key in common_keys:
                    if v_i[key] == v_j[key]:
                        continue
                    # Contradiction!
                    sorted_pair = sorted([oid_i, oid_j])
                    signals.append(Signal(
                        id=f"contradiction_{sorted_pair[0]}_{sorted_pair[1]}_{key}",
                        type="contradiction",
                        severity="error",
                        message=(
                            f"{name_i} reports {key}={v_i[key]} but {name_j} "
                            f"reports {key}={v_j[key]} for the same parent "
                            f"procedure. Reconcile these conflicting verdicts "
                            f"before finalizing your answer."
                        ),
                        emitted_at_step=ctx.current_iteration,
                        emitted_by="contradiction_detector",
                        related_node_ids=sorted_pair,
                        metadata={
                            "parent_object_id": parent_id,
                            "field": key,
                            f"{name_i}_value": v_i[key],
                            f"{name_j}_value": v_j[key],
                        },
                        sticky=True,
                        once=True,
                    ))
    return signals


def build_contradiction_detector() -> MetaProcedure:
    return MetaProcedure(
        id="meta_contradiction_detector",
        name="ContradictionDetector",
        purpose=(
            "Flag sibling session_objects with opposing boolean verdicts "
            "(restricted to known semantic fields to keep precision high)."
        ),
        fires_on="post_dispatch",
        predicate=_detect_contradictions,
        once_per_session=True,
        priority=20,
    )
