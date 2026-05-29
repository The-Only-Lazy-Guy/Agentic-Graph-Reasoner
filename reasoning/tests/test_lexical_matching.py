"""Tests for shared lexical matching helpers."""
from __future__ import annotations

import unittest

from graph_core import lexical_overlap, lexical_tokens
from reasoning.lexical_matching import (
    constraint_addressed,
    has_token_overlap,
    matches_packet_constraint,
)


class TestLexicalMatching(unittest.TestCase):
    def test_graph_core_lexical_wrappers_preserve_token_contract(self):
        self.assertEqual(lexical_tokens("aa bbb CCC"), {"bbb", "ccc"})
        self.assertGreater(lexical_overlap("alpha beta", "beta gamma"), 0.0)
        self.assertEqual(lexical_overlap("", "beta gamma"), 0.0)

    def test_activation_overlap_uses_content_tokens(self):
        self.assertTrue(has_token_overlap("Dijkstra shortest path", "Which shortest algorithm?"))
        self.assertFalse(has_token_overlap("the and with", "which the with"))

    def test_constraint_matching_preserves_special_cases(self):
        self.assertTrue(constraint_addressed("Use long long for sums.", "store int64_t aggregate values"))
        self.assertTrue(constraint_addressed("Negative edge present; Dijkstra is unsafe.", "Bellman-Ford handles negative edges"))
        self.assertFalse(constraint_addressed("Use long long for sums.", "use int sums"))

    def test_constraint_matching_handles_compound_deep_task_constraints(self):
        self.assertTrue(
            constraint_addressed(
                "Solve the add/remove/connectivity task offline with edge-active intervals over time and a rollback-capable DSU.",
                "Process queries offline with a segment tree over time, an active interval for each edge lifetime, and a rollback DSU.",
            )
        )
        self.assertFalse(
            constraint_addressed(
                "Model the reservation lifecycle explicitly: hold/reserve, confirm, release/expire, and reconcile from the authoritative source of truth.",
                "Use reservation holds, confirmation, and expiration timers in the cache.",
            )
        )
        self.assertTrue(
            constraint_addressed(
                "Model the reservation lifecycle explicitly: hold/reserve, confirm, release/expire, and reconcile from the authoritative source of truth.",
                "Keep a reservation hold/confirm/release lifecycle and reconcile from the authoritative source of truth.",
            )
        )

    def test_packet_constraint_matching_is_symmetric(self):
        hard = ["Use long long for sums."]
        self.assertTrue(matches_packet_constraint("Use int64 for large numeric sums.", hard))
        self.assertFalse(matches_packet_constraint("Answer concisely.", hard))


if __name__ == "__main__":
    unittest.main()
