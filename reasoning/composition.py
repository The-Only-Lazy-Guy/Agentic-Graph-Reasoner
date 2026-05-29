"""Composition primitives for Phase 2A.

Phase 2A introduces edges between procedures (and between session_object
instances) that capture the *call tree* of compositional reasoning:

  calls               procedure -> procedure  : A invokes B as a subroutine
  inherits            procedure -> procedure  : A's behavior generalizes B's
  specializes         procedure -> procedure  : A is a domain-specific variant of B
  replaces            procedure -> procedure  : A supersedes B (version chain)
  sub_invocation_of   session_object -> session_object
                                              : child invocation under a parent procedure run

This module only defines the relation names and a small validator.
Runtime behaviour (recursive invoke, call-tree assembly) lives in
`reasoning/dispatcher.py` and `reasoning/reasoning_loop.py`.

See PHASE2_PLAN.md §3.2.
"""
from __future__ import annotations

from typing import Final, FrozenSet


# Composition relations between abstract procedure nodes
CALLS: Final[str] = "calls"
INHERITS: Final[str] = "inherits"
SPECIALIZES: Final[str] = "specializes"
REPLACES: Final[str] = "replaces"

# Runtime relation between two session_object instances captured during a session
SUB_INVOCATION_OF: Final[str] = "sub_invocation_of"


COMPOSITION_RELATIONS: Final[FrozenSet[str]] = frozenset({
    CALLS,
    INHERITS,
    SPECIALIZES,
    REPLACES,
    SUB_INVOCATION_OF,
})

# Subset that only applies between abstract procedures (NOT session objects).
PROCEDURE_LEVEL_RELATIONS: Final[FrozenSet[str]] = frozenset({
    CALLS,
    INHERITS,
    SPECIALIZES,
    REPLACES,
})

# Subset that only applies between session_object runtime instances.
SESSION_OBJECT_RELATIONS: Final[FrozenSet[str]] = frozenset({
    SUB_INVOCATION_OF,
})


def is_composition_relation(relation: str) -> bool:
    """True iff `relation` is one of the composition / sub-invocation edges."""
    return relation in COMPOSITION_RELATIONS


def is_procedure_level_relation(relation: str) -> bool:
    """True iff `relation` connects two abstract procedures."""
    return relation in PROCEDURE_LEVEL_RELATIONS


def is_session_object_relation(relation: str) -> bool:
    """True iff `relation` connects two runtime session_object instances."""
    return relation in SESSION_OBJECT_RELATIONS
