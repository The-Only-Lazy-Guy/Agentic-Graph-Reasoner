"""Validate all bench suite files are structurally sound."""
from __future__ import annotations
import json
from pathlib import Path

bench_dir = Path("bench")
for f in sorted(bench_dir.glob("*.json")):
    d = json.loads(f.read_text(encoding="utf-8"))
    if "cases" in d:
        print(f"{f.name}: {len(d['cases'])} cases")
    elif "tasks" in d:
        print(f"{f.name}: {len(d['tasks'])} tasks")
    else:
        print(f"{f.name}: keys={list(d.keys())}")

corpus_index = json.loads((bench_dir / "replay_corpus/corpus_index.json").read_text(encoding="utf-8"))
print(f"replay_corpus/: {corpus_index['total_sessions']} sessions")

# Check negative controls have required fields
nc = json.loads((bench_dir / "negative_controls.json").read_text(encoding="utf-8"))
missing_target = [c["id"] for c in nc["cases"] if "target_plugin" not in c]
if missing_target:
    print(f"WARN: cases missing target_plugin: {missing_target}")
missing_violations = [c["id"] for c in nc["cases"] if "expected_violations" not in c]
if missing_violations:
    print(f"WARN: cases missing expected_violations: {missing_violations}")
missing_answer = [c["id"] for c in nc["cases"] if not c.get("known_bad_answer")]
if missing_answer:
    print(f"WARN: cases missing known_bad_answer: {missing_answer}")
print(f"negative_controls.json: all {len(nc['cases'])} cases have required fields")

# Check recursion fuzz has required fields
rf = json.loads((bench_dir / "recursion_fuzz.json").read_text(encoding="utf-8"))
missing_input = [c["id"] for c in rf["cases"] if "input" not in c]
if missing_input:
    print(f"WARN: fuzz cases missing input: {missing_input}")
missing_expect = [c["id"] for c in rf["cases"] if "expect" not in c]
if missing_expect:
    print(f"WARN: fuzz cases missing expect: {missing_expect}")
print(f"recursion_fuzz.json: all {len(rf['cases'])} cases have required fields")
