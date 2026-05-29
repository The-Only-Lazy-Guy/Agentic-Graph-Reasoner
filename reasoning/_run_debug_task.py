"""Run one hard benchmark task and print full reasoning trace."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load task 2 — hardest in the suite
data = json.loads(open("bench/core_20.json", encoding="utf-8").read())
task = [t for t in data["tasks"] if t["id"] == "alg_dynamic_max_subarray_online"][0]

from reasoning.reasoning_loop import run_reasoning, ReasoningRequest
from reasoning.budgets import Budgets
from local_model_command import _extract_content, _post_chat


def llm_call(prompt: str) -> str:
    return _extract_content(_post_chat(prompt)).strip()


req = ReasoningRequest(
    question=str(task["question"]),
    graph_id="merged_graph",
    graph_path="graphs/merged_graph.json",
    k_anchors=3,
    max_iterations=3,
    session_persist_root=Path(tempfile.mkdtemp(prefix="debug_")),
    budgets=Budgets(max_llm_calls=6, max_total_tokens=16000),
    enable_substrate_v2=True,
    debug_signals=True,
)

print("=" * 70)
print("TASK: " + task["question"])
print("=" * 70)
sys.stdout.flush()

started = time.perf_counter()
result = run_reasoning(req, llm_call)
elapsed = time.perf_counter() - started

print()
print("=" * 70)
print("ANSWER")
print("=" * 70)
print(result.answer)
print()
print(f"Elapsed: {elapsed:.1f}s")

audit = result.audit_summary or {}
print(f"\nStep count: {audit.get('step_count', '?')}")
tokens_per_call = audit.get("tokens_per_call", [])
print(f"LLM call tokens: {tokens_per_call}")
print(f"Total LLM calls: {len(tokens_per_call)}")
print(f"Step timing (seconds): {audit.get('step_timing', [])}")
print(f"Checker breakdown: {audit.get('checker_outcome_breakdown', {})}")
print(f"Repair triggered: {audit.get('repair_triggered')}")
print(f"Repair succeeded: {audit.get('repair_succeeded')}")
print(f"Workspace warm filled: {audit.get('workspace_warm_filled', '?')}")

# Signal dump
sig_dump = audit.get("debug_signal_dump", [])
if sig_dump:
    print(f"\n=== SIGNAL DEBUG ({len(sig_dump)} rows) ===")
    for row in sig_dump:
        keys = list(row.keys())
        rid = row.get("id", row.get("name", "?"))
        pass_val = row.get("pass", row.get("passed", "?"))
        act = row.get("activated", "?")
        conf = row.get("conf", row.get("confidence", "?"))
        note = row.get("note", "")
        print(f"  {str(rid):30s} pass={pass_val}  conf={conf}  act={act}  {note}")

# Deterministic score
from reasoning.deterministic_scorer import score_task_answer

sr = score_task_answer(result.answer, task)
print(f"\n=== DETERMINISTIC SCORE ===")
print(f"Correct: {sr.correct}  Score: {sr.score}  Source: {sr.source}")
for v in sr.violations:
    print(f"  {v}")
print(f"Details: {json.dumps(sr.details, indent=2)}")

# Required terms check
def _contains_any(answer, terms):
    lower = answer.lower()
    return any(str(t).lower() in lower for t in terms)

print()
print("=== REQUIRED TERMS CHECK ===")
for i, group in enumerate(task.get("required_terms", [])):
    ok = _contains_any(result.answer, group)
    label = "OK" if ok else "MISSING"
    print(f"  Group {i}: {group} -> {label}")

print()
print("=== FORBIDDEN TERMS CHECK ===")
for term in task.get("forbidden_terms", []):
    hit = str(term).lower() in result.answer.lower()
    label = "HIT" if hit else "clean"
    print(f"  {term} -> {label}")
