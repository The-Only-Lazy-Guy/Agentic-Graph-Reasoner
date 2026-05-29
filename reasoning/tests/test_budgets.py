"""Tests for reasoning/budgets.py.

Focus: each budget category enforces its cap, recursion push/pop works,
fan_out resets per step, summary format is consumable.
"""
from __future__ import annotations

import unittest

from reasoning.budgets import BudgetExhausted, BudgetTracker, Budgets


class TestBasicConsume(unittest.TestCase):
    def test_consume_within_cap(self):
        bt = BudgetTracker(Budgets(max_llm_calls=3))
        bt.consume("llm_call")
        bt.consume("llm_call")
        bt.consume("llm_call")
        self.assertEqual(bt.used["llm_call"], 3)

    def test_consume_one_over_raises(self):
        bt = BudgetTracker(Budgets(max_llm_calls=2))
        bt.consume("llm_call")
        bt.consume("llm_call")
        with self.assertRaises(BudgetExhausted) as cm:
            bt.consume("llm_call")
        self.assertEqual(cm.exception.op, "llm_call")
        self.assertEqual(cm.exception.used, 2)
        self.assertEqual(cm.exception.cap, 2)

    def test_check_is_nondestructive(self):
        bt = BudgetTracker(Budgets(max_llm_calls=2))
        bt.consume("llm_call")
        self.assertTrue(bt.check("llm_call"))
        self.assertEqual(bt.used["llm_call"], 1)        # check didn't consume
        self.assertTrue(bt.check("llm_call"))
        self.assertFalse(bt.check("llm_call", amount=2))  # 1 + 2 > 2

    def test_consume_amount_greater_than_one(self):
        bt = BudgetTracker(Budgets(max_total_tokens=100))
        bt.consume("tokens", amount=60)
        bt.consume("tokens", amount=40)
        self.assertEqual(bt.used["tokens"], 100)
        with self.assertRaises(BudgetExhausted):
            bt.consume("tokens", amount=1)


class TestRecursion(unittest.TestCase):
    def test_push_pop(self):
        bt = BudgetTracker(Budgets(max_recursion_depth=3))
        bt.push_recursion()
        bt.push_recursion()
        bt.push_recursion()
        self.assertEqual(bt.recursion_depth, 3)
        with self.assertRaises(BudgetExhausted) as cm:
            bt.push_recursion()
        self.assertEqual(cm.exception.op, "recursion")
        bt.pop_recursion()
        bt.pop_recursion()
        self.assertEqual(bt.recursion_depth, 1)

    def test_pop_below_zero_is_safe(self):
        bt = BudgetTracker(Budgets())
        bt.pop_recursion()
        bt.pop_recursion()
        self.assertEqual(bt.recursion_depth, 0)


class TestFanOutPerStep(unittest.TestCase):
    def test_fan_out_resets_on_step_change(self):
        bt = BudgetTracker(Budgets(max_composition_fan_out=2))
        bt.on_step_change(0)
        bt.consume("fan_out")
        bt.consume("fan_out")
        # next consume in same step exceeds
        with self.assertRaises(BudgetExhausted):
            bt.consume("fan_out")
        # move to next step — fan_out resets
        bt.on_step_change(1)
        self.assertEqual(bt.fan_out_this_step, 0)
        bt.consume("fan_out")
        bt.consume("fan_out")
        self.assertEqual(bt.fan_out_this_step, 2)

    def test_fan_out_doesnt_persist_across_steps(self):
        bt = BudgetTracker(Budgets(max_composition_fan_out=3))
        bt.on_step_change(0)
        bt.consume("fan_out")
        bt.on_step_change(1)
        self.assertEqual(bt.fan_out_this_step, 0)


class TestSummary(unittest.TestCase):
    def test_summary_structure(self):
        bt = BudgetTracker(Budgets(
            max_llm_calls=5, max_hops=2, max_session_subgraph_size=20,
            max_composition_fan_out=4, max_total_tokens=1000, max_recursion_depth=6,
        ))
        bt.consume("llm_call")
        bt.consume("llm_call")
        bt.consume("tokens", amount=300)
        bt.push_recursion()
        bt.on_step_change(0)
        bt.consume("fan_out")

        s = bt.summary()
        self.assertEqual(s["llm_calls"], {"used": 2, "cap": 5})
        self.assertEqual(s["tokens"], {"used": 300, "cap": 1000})
        self.assertEqual(s["recursion_depth_now"], 1)
        self.assertEqual(s["recursion_depth_cap"], 6)
        self.assertEqual(s["fan_out_max_per_step"]["used"], 1)


class TestExceptionMessages(unittest.TestCase):
    def test_message_includes_diagnostics(self):
        bt = BudgetTracker(Budgets(max_hops=1))
        bt.consume("hop")
        try:
            bt.consume("hop")
        except BudgetExhausted as exc:
            msg = str(exc)
            self.assertIn("hop", msg)
            self.assertIn("1/1", msg)


if __name__ == "__main__":
    unittest.main()
