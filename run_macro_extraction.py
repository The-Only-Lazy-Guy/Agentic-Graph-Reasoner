"""Phase 3B-1 macro extraction CLI.

Commands in this phase are intentionally observation-only:
  - scan: read SessionTrace JSONL and emit MacroCandidate JSON
  - status: summarize traces and candidate files
  - grade: back-fill reviewed correctness labels in trace JSONL

Validation, installation, promotion, rollback, and proposal parsing belong
to Phase 3B-2 and are deliberately not implemented here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from reasoning.macro_extractor import (
    DEFAULT_CANDIDATE_ROOT,
    MIN_PRECISION,
    MIN_SESSIONS,
    collect_status,
    scan_trace_logs,
)
from reasoning.trace_log import DEFAULT_TRACE_ROOT, apply_manual_grades


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 3B-1 macro observation tools")
    parser.add_argument(
        "--trace-root",
        type=Path,
        default=DEFAULT_TRACE_ROOT,
        help="Directory containing traces_YYYYMMDD.jsonl files",
    )
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=DEFAULT_CANDIDATE_ROOT,
        help="Directory for macro candidate JSON files",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="Scan trace logs and emit macro candidates")
    scan.add_argument("--min-sessions", type=int, default=MIN_SESSIONS)
    scan.add_argument("--min-precision", type=float, default=MIN_PRECISION)
    scan.add_argument("--dry-run", action="store_true")
    scan.add_argument("--graph-id", default=None)
    scan.add_argument("--domain", default=None)
    scan.add_argument("--json", action="store_true", dest="json_output")

    status = sub.add_parser("status", help="Summarize traces and macro candidates")
    status.add_argument("--json", action="store_true", dest="json_output")

    grade = sub.add_parser("grade", help="Back-fill trace correctness from reviewed JSONL")
    grade.add_argument("grade_file", type=Path)
    grade.add_argument("--dry-run", action="store_true")
    grade.add_argument("--json", action="store_true", dest="json_output")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "scan":
        result = scan_trace_logs(
            trace_root=args.trace_root,
            candidate_root=args.candidate_root,
            min_sessions=args.min_sessions,
            min_precision=args.min_precision,
            graph_id=args.graph_id,
            domain=args.domain,
            dry_run=args.dry_run,
        )
        payload = {
            "traces_seen": result.traces_seen,
            "clean_traces_seen": result.clean_traces_seen,
            "candidate_count": len(result.candidates),
            "written_paths": [str(path) for path in result.written_paths],
            "dry_run": bool(args.dry_run),
        }
        if args.json_output:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(
                f"scan: traces={result.traces_seen} clean={result.clean_traces_seen} "
                f"candidates={len(result.candidates)} written={len(result.written_paths)}"
            )
            for candidate in result.candidates:
                print(
                    f"- {candidate.fingerprint_hash} {candidate.proposed_name} "
                    f"sessions={candidate.session_count} precision={candidate.precision:.3f}"
                )
        return 0

    if args.command == "status":
        status = collect_status(
            trace_root=args.trace_root,
            candidate_root=args.candidate_root,
        )
        payload = status.to_dict()
        if args.json_output:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            avg = (
                "n/a"
                if status.average_precision is None
                else f"{status.average_precision:.3f}"
            )
            print(
                f"status: traces={status.trace_count} clean={status.clean_trace_count} "
                f"candidates={status.candidate_count} "
                f"coverage={status.source_session_coverage} avg_precision={avg}"
            )
            if status.candidate_status_counts:
                counts = ", ".join(
                    f"{key}={value}"
                    for key, value in sorted(status.candidate_status_counts.items())
                )
                print(f"candidate_status: {counts}")
        return 0

    if args.command == "grade":
        result = apply_manual_grades(
            trace_root=args.trace_root,
            grade_file=args.grade_file,
            dry_run=args.dry_run,
        )
        payload = result.to_dict()
        payload["dry_run"] = bool(args.dry_run)
        if args.json_output:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(
                f"grade: rows={result.grade_rows_seen} valid={result.valid_grade_rows} "
                f"invalid={result.invalid_grade_rows} updated={result.trace_rows_updated} "
                f"missing={len(result.missing_session_ids)}"
            )
            if result.updated_session_ids:
                print("updated: " + ", ".join(result.updated_session_ids))
            if result.missing_session_ids:
                print("missing: " + ", ".join(result.missing_session_ids))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
