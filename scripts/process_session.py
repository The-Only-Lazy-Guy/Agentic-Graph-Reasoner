"""Offline post-processing: run reflection + (optionally) apply graph edits.

Reads a persisted v4 session from data/session_subgraphs/<session_id>/,
runs the reflection LLM call against it, parses the result, derives edits,
and (with --apply) mutates the main graph.

Usage:
    # Dry-run: only writes reflection.json + reflection_graph_edits.json
    python scripts/process_session.py v4_abc123

    # Apply: also mutates the graph and writes a backup.
    python scripts/process_session.py v4_abc123 --apply --graph graphs/merged_graph.json

    # Re-process ALL persisted sessions
    python scripts/process_session.py --all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root via `python scripts/process_session.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Force UTF-8 stdout on Windows.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from graph_core import MemoryGraph
from reasoning.reflection import reflect_from_session_dir
from reasoning.graph_editor import edits_from_reflection, apply_edits


def _load_controller(controller_name: str):
    """Construct an LLM controller suitable for the reflection sub-call."""
    from answerer_v4 import (
        V4OpencodeController, V4LlamaServerController, V4GeminiController,
    )
    if controller_name == "opencode":
        return V4OpencodeController(
            model="opencode/big-pickle",
            server_url="http://127.0.0.1:4096",
            config_dir=r"C:\Users\Ace\AppData\Local\Temp\opencode-empty-config",
        )
    elif controller_name == "gemini":
        return V4GeminiController(model="gemini-2.5-flash")
    elif controller_name == "llama":
        return V4LlamaServerController()
    else:
        raise ValueError(f"unknown controller: {controller_name!r}")


def process_one(
    session_dir: Path,
    *,
    controller_name: str,
    apply: bool,
    graph_path: Path,
    allowed_tiers: tuple,
) -> dict:
    print(f"\n=== {session_dir.name} ===")
    if not session_dir.is_dir():
        print(f"  not a directory; skipping")
        return {"status": "skipped"}
    if not (session_dir / "subgraph.json").exists():
        print(f"  missing subgraph.json; skipping")
        return {"status": "skipped"}

    controller = _load_controller(controller_name)

    print("  running reflection LLM call...")
    reflection = reflect_from_session_dir(session_dir, controller)
    if reflection.parse_errors:
        print(f"  parse errors: {reflection.parse_errors}")
    print(f"  reflection: {len(reflection.new_facts)} new_facts, "
          f"{len(reflection.new_relationships)} new_relationships, "
          f"{len(reflection.failed_approaches)} failed_approaches, "
          f"{len(reflection.reinforced_nodes)} reinforced")
    # reflection.json is written by reflect_from_session_dir.

    if not graph_path.exists():
        print(f"  graph file {graph_path} missing; cannot derive/apply edits")
        return {"status": "reflection_only", "reflection": reflection.to_dict()}

    graph = MemoryGraph.load_json(str(graph_path))
    edits = edits_from_reflection(reflection, graph=graph)
    (session_dir / "reflection_graph_edits.json").write_text(
        json.dumps(edits, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"  derived edits: {len(edits)} "
          f"(soft={sum(1 for e in edits if e['tier']=='soft')}, "
          f"add={sum(1 for e in edits if e['tier']=='add')})")

    if not apply:
        print("  --apply not set; graph NOT mutated")
        return {"status": "dry_run", "n_edits": len(edits)}

    backup = session_dir / "graph_backup_pre_reflection_edits.json"
    summary = apply_edits(
        graph, edits,
        dry_run=False,
        backup_path=backup,
        allowed_tiers=allowed_tiers,
    )
    print(f"  applied: {summary['applied']}  skipped: {summary['skipped']}  errors: {len(summary['errors'])}")
    graph.save_json(str(graph_path))
    print(f"  graph saved back to {graph_path}")
    (session_dir / "reflection_apply_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return {"status": "applied", "summary": summary}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("session_id", nargs="?", help="session id (or omit with --all)")
    ap.add_argument("--all", action="store_true", help="process every persisted session")
    ap.add_argument("--apply", action="store_true",
                    help="mutate the main graph (default: dry-run only)")
    ap.add_argument("--graph", default="graphs/merged_graph.json",
                    help="path to the main graph file")
    ap.add_argument("--controller", default="opencode",
                    choices=["opencode", "gemini", "llama"],
                    help="LLM backend for the reflection call")
    ap.add_argument("--tiers", default="soft,add",
                    help='comma-separated tier allowlist when --apply (default: "soft,add")')
    ap.add_argument("--sessions-dir", default="data/session_subgraphs",
                    help="root directory of persisted sessions")
    args = ap.parse_args()

    sessions_root = Path(args.sessions_dir)
    if not sessions_root.exists():
        print(f"sessions dir not found: {sessions_root}")
        sys.exit(1)

    if args.all:
        targets = sorted([d for d in sessions_root.iterdir() if d.is_dir()])
    else:
        if not args.session_id:
            ap.error("provide a session_id or use --all")
        targets = [sessions_root / args.session_id]

    tiers = tuple(t.strip() for t in args.tiers.split(",") if t.strip())
    graph_path = Path(args.graph)

    for d in targets:
        process_one(
            d,
            controller_name=args.controller,
            apply=args.apply,
            graph_path=graph_path,
            allowed_tiers=tiers,
        )


if __name__ == "__main__":
    main()
