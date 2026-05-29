"""Tests for reasoning/signals.py.

Phase 3A sub-phase 3.1 acceptance: Signal dataclass round-trips through
JSON and through the session-subgraph node form, severity ordering
works for prompt rendering, render_signals_block produces the expected
text with cap and overflow.
"""
from __future__ import annotations

import json
import unittest

from reasoning.signals import (
    Signal,
    render_signals_block,
    severity_rank,
    signal_dedupe_key,
)


def _sig(
    sid: str = "sig_001",
    stype: str = "cycle_detected",
    severity: str = "warn",
    message: str = "test",
    step: int = 0,
    by: str = "test_proc",
    related: list[str] | None = None,
    sticky: bool = False,
    once: bool = False,
) -> Signal:
    return Signal(
        id=sid,
        type=stype,
        severity=severity,
        message=message,
        emitted_at_step=step,
        emitted_by=by,
        related_node_ids=list(related or []),
        metadata={},
        sticky=sticky,
        once=once,
    )


class TestRoundtrip(unittest.TestCase):
    def test_to_dict_from_dict(self):
        s = _sig(
            related=["so_a", "so_b"],
            sticky=True, once=True,
        )
        s.metadata["score"] = 0.42
        restored = Signal.from_dict(json.loads(json.dumps(s.to_dict())))
        self.assertEqual(restored.to_dict(), s.to_dict())
        self.assertEqual(restored.metadata["score"], 0.42)

    def test_node_form_roundtrip(self):
        s = _sig(
            sid="sig_node_001",
            stype="contradiction",
            severity="error",
            message="A and B disagree",
            step=3,
            by="contradiction_detector",
            related=["so_x", "so_y"],
            sticky=True,
        )
        node = s.to_node()
        self.assertEqual(node["node_type"], "signal")
        self.assertEqual(node["text"], "A and B disagree")
        self.assertEqual(node["severity"], "error")
        restored = Signal.from_node(node)
        # message and id round-trip cleanly
        self.assertEqual(restored.message, "A and B disagree")
        self.assertEqual(restored.id, "sig_node_001")
        self.assertEqual(restored.related_node_ids, ["so_x", "so_y"])
        self.assertTrue(restored.sticky)


class TestSeverityRank(unittest.TestCase):
    def test_error_is_most_urgent(self):
        self.assertLess(severity_rank("error"), severity_rank("warn"))
        self.assertLess(severity_rank("warn"), severity_rank("info"))

    def test_unknown_severity_is_least_urgent(self):
        self.assertGreater(severity_rank("bogus"), severity_rank("info"))


class TestSignalDedupeKey(unittest.TestCase):
    def test_dedupe_key_ignores_related_order(self):
        a = _sig(related=["x", "y"])
        b = _sig(related=["y", "x"])
        self.assertEqual(signal_dedupe_key(a), signal_dedupe_key(b))

    def test_dedupe_key_differs_on_type(self):
        a = _sig(stype="cycle")
        b = _sig(stype="contradiction")
        self.assertNotEqual(signal_dedupe_key(a), signal_dedupe_key(b))


class TestRenderSignalsBlock(unittest.TestCase):
    def test_empty_returns_empty_string(self):
        self.assertEqual(render_signals_block([]), "")

    def test_single_signal_renders(self):
        s = _sig(stype="budget_at_75pct", severity="info", message="7/10 LLM calls used")
        out = render_signals_block([s])
        self.assertIn("# System signals", out)
        self.assertIn("INFO ", out)
        self.assertIn("budget_at_75pct", out)
        self.assertIn("7/10 LLM calls used", out)

    def test_severity_ordering_errors_first(self):
        info_sig = _sig(sid="i1", severity="info", message="just info")
        warn_sig = _sig(sid="w1", severity="warn", message="warning")
        err_sig = _sig(sid="e1", severity="error", message="error first")
        out = render_signals_block([info_sig, warn_sig, err_sig])
        # Find positions of each severity prefix in the rendered text
        pos_err = out.find("ERROR")
        pos_warn = out.find("WARN")
        pos_info = out.find("INFO")
        self.assertLess(pos_err, pos_warn)
        self.assertLess(pos_warn, pos_info)

    def test_caps_at_max_with_overflow_meta(self):
        """Newest-first within-severity sort: with 8 INFO signals and a
        cap of 3, the 3 most recent (info-5, info-6, info-7) are visible;
        the 5 oldest are suppressed."""
        signals = [
            _sig(sid=f"i{i}", severity="info", message=f"info-{i}", step=i)
            for i in range(8)
        ]
        out = render_signals_block(signals, max_signals=3)
        # 3 most-recent info lines visible
        self.assertIn("info-7", out)
        self.assertIn("info-6", out)
        self.assertIn("info-5", out)
        # Oldest signals NOT visible
        self.assertNotIn("info-0", out)
        self.assertNotIn("info-4", out)
        # Overflow meta-line confirms the suppression count
        self.assertIn("suppressed_overflow", out)
        self.assertIn("5 additional signals", out)

    def test_within_severity_newest_first(self):
        """Within a single severity, newest emitted_at_step ranks higher."""
        older = _sig(sid="w_old", severity="warn", message="OLD warn", step=1)
        newer = _sig(sid="w_new", severity="warn", message="NEW warn", step=10)
        out = render_signals_block([older, newer])
        pos_new = out.find("NEW warn")
        pos_old = out.find("OLD warn")
        self.assertLess(pos_new, pos_old,
                        "Newer signal should render before older one within same severity")

    def test_no_overflow_meta_when_under_cap(self):
        signals = [_sig(sid="i1", severity="info", message="solo")]
        out = render_signals_block(signals, max_signals=5)
        self.assertNotIn("suppressed_overflow", out)


if __name__ == "__main__":
    unittest.main()
