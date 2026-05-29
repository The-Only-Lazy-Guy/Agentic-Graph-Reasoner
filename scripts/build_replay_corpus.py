"""Harvest the most recent session subgraphs from artifacts/ into bench/replay_corpus/.

Selects up to 60 structurally diverse sessions for deterministic replay testing.
Each session is cloned as a read-only directory with its subgraph.json and audit_log.jsonl.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


REPLAY_CORPUS_ROOT = Path("bench/replay_corpus")
MAX_SESSIONS = 60


def _discover_artifact_sessions() -> list[dict]:
    """Return list of (timestamp, task_id, run_kind, sess_dir) sorted newest-first."""
    entries: list[dict] = []
    for bench_dir in sorted(Path("artifacts").glob("phase3e_benchmark_*"), reverse=True):
        for run_dir in sorted(bench_dir.glob("run_*")):
            run_kind = "cold" if "run_1" in run_dir.name else "warm"
            v2_dir = run_dir / "v2"
            if not v2_dir.is_dir():
                continue
            for task_dir in sorted(v2_dir.iterdir()):
                if not task_dir.is_dir():
                    continue
                for sess_dir in sorted(task_dir.glob("sess_*")):
                    if not sess_dir.is_dir():
                        continue
                    subgraph_file = sess_dir / "subgraph.json"
                    audit_file = sess_dir / "audit_log.jsonl"
                    if not subgraph_file.is_file() or not audit_file.is_file():
                        continue
                    entries.append({
                        "bench_dir": bench_dir.name,
                        "task_id": task_dir.name,
                        "run_kind": run_kind,
                        "sess_dir": sess_dir,
                        "sess_id": sess_dir.name,
                        "timestamp": bench_dir.name.split("_")[-1],  # YYYYMMDD_HHMMSS
                    })
    return entries


def _select_sessions(entries: list[dict]) -> list[dict]:
    """Select up to MAX_SESSIONS sessions across diverse tasks and run kinds."""
    task_buckets: dict[str, list[dict]] = {}
    for e in entries:
        task_buckets.setdefault(e["task_id"], []).append(e)

    selected: list[dict] = []
    seen_dirs: set[str] = set()
    tasks = sorted(task_buckets.keys())

    # Round-robin: pick 1 cold + 1 warm per task, newest first
    for _ in range(MAX_SESSIONS):
        for task in tasks:
            if len(selected) >= MAX_SESSIONS:
                break
            bucket = task_buckets[task]
            for kind in ("cold", "warm"):
                candidates = [e for e in bucket if e["run_kind"] == kind and e["sess_dir"].name not in seen_dirs]
                if candidates:
                    chosen = candidates[0]
                    selected.append(chosen)
                    seen_dirs.add(chosen["sess_dir"].name)
                    if len(selected) >= MAX_SESSIONS:
                        break
    return selected


def _copy_session(session: dict, index: int) -> Path:
    target_dir = REPLAY_CORPUS_ROOT / f"sess_{index:04d}_{session['task_id']}_{session['sess_id']}"
    target_dir.mkdir(parents=True, exist_ok=True)

    src_subgraph = session["sess_dir"] / "subgraph.json"
    src_audit = session["sess_dir"] / "audit_log.jsonl"

    shutil.copy2(str(src_subgraph), str(target_dir / "subgraph.json"))
    shutil.copy2(str(src_audit), str(target_dir / "audit_log.jsonl"))

    # Write metadata
    meta = {
        "source_bench": session["bench_dir"],
        "task_id": session["task_id"],
        "run_kind": session["run_kind"],
        "original_session_id": session["sess_id"],
        "corpus_index": index,
    }
    (target_dir / "metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return target_dir


def main() -> int:
    entries = _discover_artifact_sessions()
    if not entries:
        print("ERROR: no session subgraphs found in artifacts/")
        return 1

    print(f"Found {len(entries)} total sessions across artifacts")

    selected = _select_sessions(entries)
    print(f"Selected {len(selected)} sessions for replay corpus")

    if REPLAY_CORPUS_ROOT.exists():
        shutil.rmtree(str(REPLAY_CORPUS_ROOT))
    REPLAY_CORPUS_ROOT.mkdir(parents=True)

    copied: list[dict] = []
    for i, session in enumerate(selected):
        target = _copy_session(session, i)
        copied.append({
            "index": i,
            "target_dir": str(target),
            "task_id": session["task_id"],
            "run_kind": session["run_kind"],
            "source": session["bench_dir"],
        })
        print(f"  [{i:03d}] {session['task_id']} ({session['run_kind']}) -> {target.name}")

    # Write corpus index
    index = {
        "version": "phase3e-replay-corpus-2026-05-23",
        "purpose": "Persisted session journals for deterministic replay testing.",
        "total_sessions": len(copied),
        "sessions": copied,
    }
    (REPLAY_CORPUS_ROOT / "corpus_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nWROTE {REPLAY_CORPUS_ROOT / 'corpus_index.json'}")
    print(f"Total sessions: {len(copied)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
