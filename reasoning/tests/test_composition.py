"""Tests for reasoning/composition.py — relation name constants + validators.

Phase 2A sub-phase 2.2 acceptance.
"""
from __future__ import annotations

import unittest

from reasoning.composition import (
    CALLS,
    COMPOSITION_RELATIONS,
    INHERITS,
    PROCEDURE_LEVEL_RELATIONS,
    REPLACES,
    SESSION_OBJECT_RELATIONS,
    SPECIALIZES,
    SUB_INVOCATION_OF,
    is_composition_relation,
    is_procedure_level_relation,
    is_session_object_relation,
)


class TestConstants(unittest.TestCase):
    def test_relation_names_match_plan(self):
        self.assertEqual(CALLS, "calls")
        self.assertEqual(INHERITS, "inherits")
        self.assertEqual(SPECIALIZES, "specializes")
        self.assertEqual(REPLACES, "replaces")
        self.assertEqual(SUB_INVOCATION_OF, "sub_invocation_of")

    def test_composition_relations_union(self):
        self.assertEqual(
            COMPOSITION_RELATIONS,
            {CALLS, INHERITS, SPECIALIZES, REPLACES, SUB_INVOCATION_OF},
        )

    def test_procedure_level_excludes_sub_invocation_of(self):
        self.assertNotIn(SUB_INVOCATION_OF, PROCEDURE_LEVEL_RELATIONS)
        for r in (CALLS, INHERITS, SPECIALIZES, REPLACES):
            self.assertIn(r, PROCEDURE_LEVEL_RELATIONS)

    def test_session_object_only_has_sub_invocation_of(self):
        self.assertEqual(SESSION_OBJECT_RELATIONS, {SUB_INVOCATION_OF})


class TestValidators(unittest.TestCase):
    def test_is_composition_relation_positives(self):
        for r in (CALLS, INHERITS, SPECIALIZES, REPLACES, SUB_INVOCATION_OF):
            self.assertTrue(is_composition_relation(r), f"{r!r} should be a composition relation")

    def test_is_composition_relation_negatives(self):
        # Existing Phase-1 relations that are NOT composition relations
        for r in ("support", "refine", "contradict", "example_of", "part_of", "related"):
            self.assertFalse(is_composition_relation(r), f"{r!r} should not be a composition relation")

    def test_procedure_level_validator(self):
        self.assertTrue(is_procedure_level_relation(CALLS))
        self.assertTrue(is_procedure_level_relation(REPLACES))
        self.assertFalse(is_procedure_level_relation(SUB_INVOCATION_OF),
                         "sub_invocation_of is between runtime objects, not abstract procedures")
        self.assertFalse(is_procedure_level_relation("support"))

    def test_session_object_validator(self):
        self.assertTrue(is_session_object_relation(SUB_INVOCATION_OF))
        for r in (CALLS, INHERITS, SPECIALIZES, REPLACES):
            self.assertFalse(is_session_object_relation(r),
                             f"{r!r} is a procedure-level relation, not session-object")
        self.assertFalse(is_session_object_relation("support"))

    def test_disjointness_of_subsets(self):
        """An edge type belongs to exactly one of {procedure-level, session-object}."""
        self.assertEqual(
            PROCEDURE_LEVEL_RELATIONS & SESSION_OBJECT_RELATIONS,
            frozenset(),
        )
        self.assertEqual(
            PROCEDURE_LEVEL_RELATIONS | SESSION_OBJECT_RELATIONS,
            COMPOSITION_RELATIONS,
        )


if __name__ == "__main__":
    unittest.main()
