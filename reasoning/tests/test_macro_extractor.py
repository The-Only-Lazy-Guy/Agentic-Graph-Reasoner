"""Tests for Phase 3B-1 macro extraction."""
from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from reasoning.macro_extractor import (
    CandidateBucket,
    MacroCandidate,
    bucket_for_trace,
    budget_profile_for_call_count,
    collect_status,
    extract_macro_candidates,
    load_candidates,
    scan_trace_logs,
)
from reasoning.trace_log import ProcedureCall, SessionTrace, TraceLogger


def _call(name: str, *, call_id: str, parent: str | None = None) -> ProcedureCall:
    return ProcedureCall(
        call_id=call_id,
        procedure_id=f"proc_{name.lower()}_v1",
        procedure_name=name,
        parent_call_id=parent,
        args_text="instance=test",
        mutations_applied=1,
        error=None,
        elapsed_seconds=0.0,
    )


def _trace(
    session_id: str,
    *,
    domain: str | None = "computer_science",
    correct: bool | None = True,
    budget_exhausted: bool = False,
    procedure_errors: int = 0,
    contradiction_signals: int = 0,
    signal_ids: list[str] | None = None,
    calls: list[ProcedureCall] | None = None,
) -> SessionTrace:
    parent = _call("VerifyShortestPath", call_id=f"{session_id}_p")
    child_a = _call("VerifyAlgorithmPreconditions", call_id=f"{session_id}_a", parent=parent.call_id)
    child_b = _call("VerifyNonNegativeEdges", call_id=f"{session_id}_b", parent=parent.call_id)
    return SessionTrace(
        session_id=session_id,
        graph_id="cs4",
        domain=domain,
        question="Is Dijkstra safe?",
        answer="No, use Bellman-Ford.",
        correct=correct,
        budget_exhausted=budget_exhausted,
        procedure_errors=procedure_errors,
        contradiction_signals=contradiction_signals,
        procedure_calls=calls if calls is not None else [parent, child_a, child_b],
        budget_usage={"llm_calls": {"used": 5, "cap": 6}},
        signal_ids_fired=signal_ids or [],
        elapsed_seconds=1.0,
    )


class TestBucketHelpers(unittest.TestCase):
    def test_budget_profiles(self):
        self.assertEqual(budget_profile_for_call_count(1), "low")
        self.assertEqual(budget_profile_for_call_count(2), "low")
        self.assertEqual(budget_profile_for_call_count(3), "mid")
        self.assertEqual(budget_profile_for_call_count(4), "mid")
        self.assertEqual(budget_profile_for_call_count(5), "high")

    def test_bucket_uses_domain_signal_and_budget_profile(self):
        trace = _trace("sess_1", signal_ids=["sig_warn"])
        bucket = bucket_for_trace(trace)
        self.assertIsNotNone(bucket)
        assert bucket is not None
        self.assertEqual(bucket.graph_domain, "computer_science")
        self.assertEqual(bucket.top_level_procedure, "VerifyShortestPath")
        self.assertFalse(bucket.signal_free)
        self.assertEqual(bucket.budget_profile, "mid")

    def test_empty_procedure_trace_has_no_bucket(self):
        self.assertIsNone(bucket_for_trace(_trace("sess_empty", calls=[])))


class TestExtractMacroCandidates(unittest.TestCase):
    def test_three_clean_sessions_same_bucket_yield_candidate(self):
        traces = [_trace(f"sess_{i}") for i in range(3)]
        candidates = extract_macro_candidates(traces)
        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(candidate.session_count, 3)
        self.assertEqual(candidate.precision, 1.0)
        self.assertEqual(candidate.status, "candidate")
        self.assertEqual(candidate.source_session_ids, ["sess_0", "sess_1", "sess_2"])
        self.assertIn("VerifyShortestPath", candidate.fingerprint)

    def test_threshold_applies_within_bucket_not_globally(self):
        traces = [
            _trace("cs_1", domain="computer_science"),
            _trace("cs_2", domain="computer_science"),
            _trace("math_1", domain="math"),
        ]
        self.assertEqual(extract_macro_candidates(traces), [])

    def test_clean_trace_filter_rejects_messy_recovered_runs(self):
        traces = [
            _trace("clean_1"),
            _trace("clean_2"),
            _trace("budget", budget_exhausted=True),
            _trace("proc_error", procedure_errors=1),
            _trace("contradiction", contradiction_signals=1),
            _trace("wrong", correct=False),
            _trace("ungraded", correct=None),
        ]
        self.assertEqual(extract_macro_candidates(traces), [])

    def test_signal_free_is_part_of_bucket(self):
        traces = [
            _trace("clean_1"),
            _trace("clean_2"),
            _trace("signaled", signal_ids=["sig_info"]),
        ]
        self.assertEqual(extract_macro_candidates(traces), [])

    def test_graph_and_domain_filters(self):
        traces = [_trace(f"sess_{i}") for i in range(3)]
        traces.append(_trace("other_domain", domain="math"))
        self.assertEqual(len(extract_macro_candidates(traces, domain="computer_science")), 1)
        self.assertEqual(extract_macro_candidates(traces, graph_id="other_graph"), [])


class TestCandidateSerialization(unittest.TestCase):
    def test_round_trip(self):
        bucket = CandidateBucket(
            fingerprint="A > B",
            graph_domain="computer_science",
            top_level_procedure="A",
            signal_free=True,
            budget_profile="low",
        )
        candidate = MacroCandidate(
            fingerprint="A > B",
            fingerprint_hash="abc123",
            bucket=bucket,
            procedure_names=["A", "B"],
            source_session_ids=["s1", "s2", "s3"],
            session_count=3,
            precision=1.0,
            proposed_name="Macro_A_B",
        )
        restored = MacroCandidate.from_dict(json.loads(json.dumps(candidate.to_dict())))
        self.assertEqual(restored.bucket, bucket)
        self.assertEqual(restored.procedure_names, ["A", "B"])


class TestScanAndStatus(unittest.TestCase):
    def test_scan_writes_candidate_json_and_status_reads_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace_root = root / "trace_logs"
            candidate_root = root / "macro_candidates"
            logger = TraceLogger(trace_root)
            for i in range(3):
                logger.append(_trace(f"sess_{i}"))

            result = scan_trace_logs(trace_root=trace_root, candidate_root=candidate_root)
            self.assertEqual(result.traces_seen, 3)
            self.assertEqual(result.clean_traces_seen, 3)
            self.assertEqual(len(result.candidates), 1)
            self.assertEqual(len(result.written_paths), 1)
            self.assertTrue(result.written_paths[0].exists())

            loaded = load_candidates(candidate_root)
            self.assertEqual(len(loaded), 1)
            status = collect_status(trace_root=trace_root, candidate_root=candidate_root)
            self.assertEqual(status.trace_count, 3)
            self.assertEqual(status.clean_trace_count, 3)
            self.assertEqual(status.candidate_count, 1)
            self.assertEqual(status.source_session_coverage, 3)
            self.assertEqual(status.candidate_status_counts, {"candidate": 1})

    def test_dry_run_does_not_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace_root = root / "trace_logs"
            candidate_root = root / "macro_candidates"
            logger = TraceLogger(trace_root)
            for i in range(3):
                logger.append(_trace(f"sess_{i}"))

            result = scan_trace_logs(
                trace_root=trace_root,
                candidate_root=candidate_root,
                dry_run=True,
            )
            self.assertEqual(len(result.candidates), 1)
            self.assertEqual(result.written_paths, [])
            self.assertFalse(candidate_root.exists())


class TestCli(unittest.TestCase):
    def test_scan_and_status_cli_json(self):
        from run_macro_extraction import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace_root = root / "trace_logs"
            candidate_root = root / "macro_candidates"
            logger = TraceLogger(trace_root)
            for i in range(3):
                logger.append(_trace(f"sess_{i}"))

            out = StringIO()
            with redirect_stdout(out):
                code = main([
                    "--trace-root", str(trace_root),
                    "--candidate-root", str(candidate_root),
                    "scan",
                    "--json",
                ])
            self.assertEqual(code, 0)
            payload = json.loads(out.getvalue())
            self.assertEqual(payload["candidate_count"], 1)

    def test_grade_then_scan_cli_json(self):
        from run_macro_extraction import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace_root = root / "trace_logs"
            candidate_root = root / "macro_candidates"
            grade_file = root / "grades.jsonl"
            logger = TraceLogger(trace_root)
            for i in range(3):
                logger.append(_trace(f"sess_{i}", correct=None))
            grade_file.write_text(
                "".join(
                    json.dumps({"session_id": f"sess_{i}", "correct": True}) + "\n"
                    for i in range(3)
                ),
                encoding="utf-8",
            )

            out = StringIO()
            with redirect_stdout(out):
                code = main([
                    "--trace-root", str(trace_root),
                    "--candidate-root", str(candidate_root),
                    "grade",
                    str(grade_file),
                    "--json",
                ])
            self.assertEqual(code, 0)
            payload = json.loads(out.getvalue())
            self.assertEqual(payload["trace_rows_updated"], 3)
            self.assertEqual(payload["missing_session_ids"], [])

            out = StringIO()
            with redirect_stdout(out):
                code = main([
                    "--trace-root", str(trace_root),
                    "--candidate-root", str(candidate_root),
                    "scan",
                    "--json",
                ])
            self.assertEqual(code, 0)
            payload = json.loads(out.getvalue())
            self.assertEqual(payload["clean_traces_seen"], 3)
            self.assertEqual(payload["candidate_count"], 1)

            out = StringIO()
            with redirect_stdout(out):
                code = main([
                    "--trace-root", str(trace_root),
                    "--candidate-root", str(candidate_root),
                    "status",
                    "--json",
                ])
            self.assertEqual(code, 0)
            payload = json.loads(out.getvalue())
            self.assertEqual(payload["candidate_count"], 1)


if __name__ == "__main__":
    unittest.main()
