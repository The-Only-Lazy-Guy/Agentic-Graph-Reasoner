import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import answerer_v4
import run_repeat_learning_experiment as repeat_exp
from answerer_v4 import V4OpencodeController
from graph_core import MemoryGraph, Node


class _FakeCompletedProcess:
    def __init__(self, *, stdout: str, stderr: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def _fake_packet(raw_trace):
    return SimpleNamespace(
        question="repeat question",
        answer="final answer",
        answer_raw="final answer",
        execution_mode="loop",
        task_type="direct_judgment",
        controller_task_family="algorithm_applicability",
        steps=2,
        max_steps=4,
        tool_call_count=0,
        elapsed_sec=1.2,
        finalized=True,
        anchors=["n1"],
        shortcut_reason="",
        shortcut_anchor_ids=[],
        controller_fallback_used=False,
        subgoal_reuse_count=0,
        slot_fill_stats={"required_slots": ["verdict"]},
        controller_action_counts={"REUSE": 1, "FINALIZE": 1},
        controller_call_count=len(raw_trace),
        controller_total_elapsed_sec=round(sum(float(t.get("elapsed_sec", 0.0) or 0.0) for t in raw_trace), 3),
        controller_nonempty_turns=sum(1 for t in raw_trace if str(t.get("assistant_text", "") or "").strip()),
        controller_raw_trace=raw_trace,
        session_dir=None,
        plan=[],
        tool_log=[],
        cot_log=["turn one", ""],
        learning_report=None,
        graph_edits=[],
        graph_edits_applied=False,
        explanation="",
        polish_applied=False,
    )


class RawTraceCaptureTests(unittest.TestCase):
    def setUp(self) -> None:
        self._orig_exe = answerer_v4._OPENCODE_EXE
        answerer_v4._OPENCODE_EXE = "opencode"

    def tearDown(self) -> None:
        answerer_v4._OPENCODE_EXE = self._orig_exe

    def test_chat_trace_records_prompt_raw_output_and_events(self) -> None:
        stdout = "\n".join([
            json.dumps({"sessionID": "sess-1", "type": "start"}),
            json.dumps({"sessionID": "sess-1", "type": "text", "part": {"text": "Hello"}}),
            json.dumps({"sessionID": "sess-1", "type": "text", "part": {"text": " world"}}),
        ])

        with patch.object(answerer_v4._subprocess, "run", return_value=_FakeCompletedProcess(stdout=stdout)) as run_mock:
            controller = V4OpencodeController(model="opencode/big-pickle", server_url="http://127.0.0.1:4096")
            resp = controller.chat([{"role": "user", "content": "Say hello"}])

        self.assertEqual("Hello world", resp["choices"][0]["message"]["content"])
        self.assertEqual(1, run_mock.call_count)
        trace = controller.get_raw_trace()
        self.assertEqual(1, len(trace))
        entry = trace[0]
        self.assertEqual("chat", entry["mode"])
        self.assertEqual("Say hello", entry["stdin_message"])
        self.assertEqual("sess-1", entry["session_id_after"])
        self.assertEqual("Hello world", entry["assistant_text"])
        self.assertEqual(stdout, entry["raw_stdout"])
        self.assertEqual(3, len(entry["events"]))
        self.assertGreaterEqual(entry["elapsed_sec"], 0.0)

    def test_chat_oneshot_trace_is_merged_into_parent(self) -> None:
        stdout = "\n".join([
            json.dumps({"sessionID": "oneshot-1", "type": "start"}),
            json.dumps({"sessionID": "oneshot-1", "type": "text", "part": {"text": "Polished"}}),
        ])

        with patch.object(answerer_v4._subprocess, "run", return_value=_FakeCompletedProcess(stdout=stdout)):
            controller = V4OpencodeController(model="opencode/big-pickle", server_url="http://127.0.0.1:4096")
            resp = controller.chat_oneshot(
                [
                    {"role": "system", "content": "rewrite"},
                    {"role": "user", "content": "draft"},
                ]
            )

        self.assertEqual("Polished", resp["choices"][0]["message"]["content"])
        trace = controller.get_raw_trace()
        self.assertEqual(1, len(trace))
        self.assertEqual("chat_oneshot", trace[0]["mode"])
        self.assertIn("rewrite", trace[0]["stdin_message"])
        self.assertIn("draft", trace[0]["stdin_message"])

    def test_repeat_experiment_writes_raw_trace_artifacts(self) -> None:
        raw_trace = [
            {
                "call_index": 1,
                "mode": "chat",
                "session_id_before": None,
                "session_id_after": "sess-1",
                "model": "opencode/big-pickle",
                "server_url": "http://127.0.0.1:4096",
                "variant": None,
                "stdin_message": "prompt",
                "raw_stdout": '{"sessionID":"sess-1","type":"text","part":{"text":"answer"}}',
                "raw_stderr": "",
                "events": [{"sessionID": "sess-1", "type": "text", "part": {"text": "answer"}}],
                "assistant_text": "answer",
                "returncode": 0,
                "started_at": "2026-05-27T00:00:00+00:00",
                "ended_at": "2026-05-27T00:00:01+00:00",
                "elapsed_sec": 1.0,
            }
        ]
        graph = MemoryGraph(nodes={"n1": Node(id="n1", node_type="claim", text="one")}, edges=[])

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            with patch.object(repeat_exp, "answer_query_v4", return_value=_fake_packet(raw_trace)):
                result = repeat_exp._run_once(
                    run_index=1,
                    question="repeat question",
                    graph=graph,
                    graph_input_path=Path("graphs/input.json"),
                    graph_id="g",
                    model="opencode/big-pickle",
                    server_url="http://127.0.0.1:4096",
                    timeout=30.0,
                    out_dir=out_dir,
                )

            self.assertEqual(str(Path("graphs/input.json")), result["graph_input_path"])
            raw_trace_json = out_dir / "run_1_raw_trace.json"
            raw_trace_txt = out_dir / "run_1_raw_trace.txt"
            self.assertTrue(raw_trace_json.exists())
            self.assertTrue(raw_trace_txt.exists())
            stored_json = json.loads(raw_trace_json.read_text(encoding="utf-8"))
            stored_txt = raw_trace_txt.read_text(encoding="utf-8")
            self.assertEqual("prompt", stored_json[0]["stdin_message"])
            self.assertIn("--- raw_stdout ---", stored_txt)
            self.assertIn("answer", stored_txt)


if __name__ == "__main__":
    unittest.main()
