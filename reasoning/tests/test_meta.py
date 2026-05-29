"""Tests for reasoning/meta.py.

Phase 3A sub-phase 3.1 acceptance: MetaProcedure schema instantiates,
MetaPool registers + runs + dedupes correctly, predicate exceptions
are tolerated, action exceptions don't cancel signals.
"""
from __future__ import annotations

import unittest
from typing import List

from reasoning.meta import MetaContext, MetaPool, MetaProcedure
from reasoning.signals import Signal


# ---- minimal context fixture ------------------------------------------ #

class _StubBudget:
    """Bare-minimum BudgetTracker substitute for predicate tests."""
    def __init__(self):
        self.used = {"llm_call": 0, "fan_out": 0, "tokens": 0}
        self.recursion_depth = 0


class _StubSession:
    """Minimal SessionSubgraphController substitute. Only the predicate
    bodies care about session.subgraph.nodes / .edges, so we give those."""
    class _SG:
        def __init__(self):
            self.nodes = {}
            self.edges = []
    def __init__(self):
        self.subgraph = _StubSession._SG()


def _ctx(iteration: int = 0, previous: List[Signal] | None = None) -> MetaContext:
    return MetaContext(
        session=_StubSession(),               # type: ignore[arg-type]
        budget=_StubBudget(),                 # type: ignore[arg-type]
        dispatch_outcomes=[],
        raw_outputs=[],
        anchor_ids=[],
        current_iteration=iteration,
        previous_signals=list(previous or []),
    )


def _signal(stype: str = "test", severity: str = "info",
            related: list[str] | None = None, sticky: bool = False,
            once: bool = False) -> Signal:
    return Signal(
        id=f"sig_{stype}_{','.join(related or [])}",
        type=stype,
        severity=severity,
        message=f"test message for {stype}",
        emitted_at_step=0,
        emitted_by="test_mp",
        related_node_ids=list(related or []),
        sticky=sticky,
        once=once,
    )


# ---- MetaProcedure schema ------------------------------------------- #

class TestMetaProcedureSchema(unittest.TestCase):
    def test_minimal_construction(self):
        mp = MetaProcedure(
            id="mp_001", name="TestMP", purpose="test",
            fires_on="pre_iter",
            predicate=lambda _ctx: [],
        )
        self.assertEqual(mp.fires_on, "pre_iter")
        self.assertIsNone(mp.action)
        self.assertFalse(mp.once_per_session)


# ---- MetaPool basics ----------------------------------------------- #

class TestMetaPoolBasic(unittest.TestCase):
    def test_register_and_filter_by_hook(self):
        pool = MetaPool()
        pre_mp = MetaProcedure("a", "A", "", "pre_iter", lambda _c: [])
        post_mp = MetaProcedure("b", "B", "", "post_dispatch", lambda _c: [])
        end_mp = MetaProcedure("c", "C", "", "end_of_session", lambda _c: [])
        pool.register(pre_mp)
        pool.register(post_mp)
        pool.register(end_mp)
        self.assertEqual(pool.procedures_for_hook("pre_iter"), [pre_mp])
        self.assertEqual(pool.procedures_for_hook("post_dispatch"), [post_mp])
        self.assertEqual(pool.procedures_for_hook("end_of_session"), [end_mp])

    def test_priority_orders_within_hook(self):
        pool = MetaPool()
        late = MetaProcedure("late", "Late", "", "pre_iter", lambda _c: [],
                             priority=200)
        early = MetaProcedure("early", "Early", "", "pre_iter", lambda _c: [],
                              priority=50)
        pool.register(late)
        pool.register(early)
        order = pool.procedures_for_hook("pre_iter")
        self.assertEqual([mp.id for mp in order], ["early", "late"])


class TestMetaPoolRunHook(unittest.TestCase):
    def test_predicate_returning_signals_emits_them(self):
        pool = MetaPool()
        pool.register(MetaProcedure(
            "fire_one", "FireOne", "", "pre_iter",
            predicate=lambda _c: [_signal("alpha")],
        ))
        emitted = pool.run_hook("pre_iter", _ctx())
        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0].type, "alpha")
        self.assertEqual(len(pool.signal_stream), 1)

    def test_predicate_returning_empty_emits_nothing(self):
        pool = MetaPool()
        pool.register(MetaProcedure(
            "fire_none", "FireNone", "", "pre_iter",
            predicate=lambda _c: [],
        ))
        emitted = pool.run_hook("pre_iter", _ctx())
        self.assertEqual(emitted, [])
        self.assertEqual(pool.signal_stream, [])

    def test_action_runs_on_emission(self):
        seen: List[tuple] = []
        pool = MetaPool()
        def predicate(_c): return [_signal("act_check")]
        def action(_c, sigs):
            seen.append(("called", len(sigs)))
        pool.register(MetaProcedure(
            "with_action", "WithAction", "", "pre_iter",
            predicate=predicate, action=action,
        ))
        pool.run_hook("pre_iter", _ctx())
        self.assertEqual(seen, [("called", 1)])

    def test_action_exception_does_not_cancel_signals(self):
        pool = MetaPool()
        def predicate(_c): return [_signal("survives_action_fail")]
        def action(_c, _s): raise RuntimeError("intentional")
        pool.register(MetaProcedure(
            "bad_action", "BadAction", "", "pre_iter",
            predicate=predicate, action=action,
        ))
        emitted = pool.run_hook("pre_iter", _ctx())
        # Signal still emitted despite action raising
        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0].type, "survives_action_fail")

    def test_predicate_exception_skips_meta_procedure(self):
        pool = MetaPool()
        def good_predicate(_c): return [_signal("good")]
        def bad_predicate(_c): raise RuntimeError("predicate boom")
        pool.register(MetaProcedure(
            "bad", "Bad", "", "pre_iter", predicate=bad_predicate,
        ))
        pool.register(MetaProcedure(
            "good", "Good", "", "pre_iter", predicate=good_predicate,
        ))
        emitted = pool.run_hook("pre_iter", _ctx())
        # The good MP fired; the bad MP was skipped silently
        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0].type, "good")


class TestOncePerSessionDedupe(unittest.TestCase):
    def test_same_type_same_related_only_fires_once(self):
        pool = MetaPool()
        def predicate(_c):
            return [_signal("contradiction", related=["so_a", "so_b"])]
        pool.register(MetaProcedure(
            "dedupe_mp", "DedupeMP", "", "post_dispatch",
            predicate=predicate, once_per_session=True,
        ))

        # First call: fires
        emitted1 = pool.run_hook("post_dispatch", _ctx())
        self.assertEqual(len(emitted1), 1)
        # Second call: predicate still wants to fire, but it's deduped
        emitted2 = pool.run_hook("post_dispatch", _ctx(iteration=1))
        self.assertEqual(emitted2, [])
        # Signal stream contains only the first occurrence
        self.assertEqual(len(pool.signal_stream), 1)

    def test_different_related_nodes_fire_independently(self):
        pool = MetaPool()
        calls = {"n": 0}
        def predicate(_c):
            calls["n"] += 1
            related = [f"so_{calls['n']}_a", f"so_{calls['n']}_b"]
            return [_signal("contradiction", related=related)]
        pool.register(MetaProcedure(
            "diverse_mp", "DiverseMP", "", "post_dispatch",
            predicate=predicate, once_per_session=True,
        ))
        emitted1 = pool.run_hook("post_dispatch", _ctx())
        emitted2 = pool.run_hook("post_dispatch", _ctx(iteration=1))
        self.assertEqual(len(emitted1), 1)
        self.assertEqual(len(emitted2), 1)

    def test_per_signal_once_flag_alone(self):
        """A meta-procedure not marked once_per_session can still emit
        signals that individually carry once=True, and those get deduped."""
        pool = MetaPool()
        def predicate(_c):
            return [_signal("event", related=["x"], once=True)]
        pool.register(MetaProcedure(
            "no_session_once", "X", "", "pre_iter", predicate=predicate,
        ))
        emitted1 = pool.run_hook("pre_iter", _ctx())
        emitted2 = pool.run_hook("pre_iter", _ctx(iteration=1))
        self.assertEqual(len(emitted1), 1)
        self.assertEqual(emitted2, [])


if __name__ == "__main__":
    unittest.main()
