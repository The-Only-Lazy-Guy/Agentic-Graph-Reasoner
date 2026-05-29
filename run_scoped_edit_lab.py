from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from graph_core import MemoryGraph
from reasoning.scoped_edits import (
    GraphEditPatch,
    patches_from_graph_edits,
    patches_to_dicts,
    render_patch_report,
    summarize_patches,
    validate_patches,
    write_scoped_patch_artifacts,
)


DEFAULT_GRAPH = Path("graphs/merged_graph.json")
DEFAULT_OUT_ROOT = Path("artifacts/scoped_edit_lab")


@dataclass
class EditCase:
    case_id: str
    source_path: Path
    question: str
    graph_edits: List[Dict[str, Any]]
    learning_report: Optional[Dict[str, Any]] = None
    task_frame: Optional[Dict[str, Any]] = None


def _now_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", str(text or "").strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug[:90] or "case"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_unique(paths: Iterable[Path]) -> List[Path]:
    seen = set()
    out: List[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(path)
    return out


def discover_cases(
    *,
    artifact_dirs: Sequence[Path],
    session_dirs: Sequence[Path],
    packet_paths: Sequence[Path],
) -> List[EditCase]:
    cases: List[EditCase] = []

    discovered_packets: List[Path] = list(packet_paths)
    for root in artifact_dirs:
        if root.is_file():
            discovered_packets.append(root)
            continue
        discovered_packets.extend(root.rglob("packet.json"))
        discovered_packets.extend(root.rglob("*_packet.json"))
        discovered_packets.extend(root.rglob("graph_packet.json"))

    for packet_path in _iter_unique(discovered_packets):
        try:
            raw = _load_json(packet_path)
        except Exception:
            continue
        if not isinstance(raw, Mapping):
            continue
        edits = raw.get("graph_edits")
        if not isinstance(edits, list):
            continue
        case_id = _slug(f"{packet_path.parent.name}_{packet_path.stem}")
        question = str(raw.get("question") or raw.get("input", {}).get("question") or "")
        cases.append(EditCase(
            case_id=case_id,
            source_path=packet_path,
            question=question,
            graph_edits=[dict(e) for e in edits if isinstance(e, Mapping)],
            learning_report=(
                dict(raw.get("learning_report"))
                if isinstance(raw.get("learning_report"), Mapping) else None
            ),
            task_frame=(
                dict(raw.get("task_frame"))
                if isinstance(raw.get("task_frame"), Mapping) else None
            ),
        ))

    discovered_sessions: List[Path] = []
    for root in session_dirs:
        if (root / "graph_edits.json").exists():
            discovered_sessions.append(root)
        elif root.exists():
            for edit_file in root.rglob("graph_edits.json"):
                discovered_sessions.append(edit_file.parent)

    for session_dir in _iter_unique(discovered_sessions):
        edit_path = session_dir / "graph_edits.json"
        try:
            edits_raw = _load_json(edit_path)
        except Exception:
            continue
        if not isinstance(edits_raw, list):
            continue
        learning_report = None
        lr_path = session_dir / "learning_report.json"
        if lr_path.exists():
            try:
                lr_raw = _load_json(lr_path)
                if isinstance(lr_raw, Mapping):
                    learning_report = dict(lr_raw)
            except Exception:
                learning_report = None
        case_id = _slug(session_dir.name)
        cases.append(EditCase(
            case_id=case_id,
            source_path=edit_path,
            question=str((learning_report or {}).get("question", "")),
            graph_edits=[dict(e) for e in edits_raw if isinstance(e, Mapping)],
            learning_report=learning_report,
            task_frame=None,
        ))

    return cases


def run_case(case: EditCase, graph: MemoryGraph, out_dir: Path) -> Dict[str, Any]:
    case_out = out_dir / case.case_id
    case_out.mkdir(parents=True, exist_ok=True)
    patches = validate_patches(
        patches_from_graph_edits(
            case.graph_edits,
            graph=graph,
            learning_report=case.learning_report,
            question=case.question,
            task_frame=case.task_frame,
        ),
        graph,
    )
    for patch in patches:
        patch.payload["case_id"] = case.case_id
        patch.payload["source_path"] = str(case.source_path)
        patch.patch_id = f"{case.case_id}_{patch.patch_id}"
    summary = write_scoped_patch_artifacts(case_out, patches)
    (case_out / "case_input.json").write_text(
        json.dumps({
            "case_id": case.case_id,
            "source_path": str(case.source_path),
            "question": case.question,
            "graph_edit_count": len(case.graph_edits),
            "learning_report": case.learning_report,
            "task_frame": case.task_frame,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "case_id": case.case_id,
        "source_path": str(case.source_path),
        "question": case.question,
        "graph_edit_count": len(case.graph_edits),
        "patch_summary": summary,
        "patches": patches,
    }


def _write_global_report(out_dir: Path, results: Sequence[Mapping[str, Any]]) -> None:
    all_patches: List[GraphEditPatch] = []
    for result in results:
        all_patches.extend(result.get("patches", []))
    summary = summarize_patches(all_patches)

    serializable_results = []
    for result in results:
        serializable_results.append({
            "case_id": result.get("case_id"),
            "source_path": result.get("source_path"),
            "question": result.get("question"),
            "graph_edit_count": result.get("graph_edit_count"),
            "patch_summary": result.get("patch_summary"),
        })

    (out_dir / "summary.json").write_text(
        json.dumps({
            "generated_at": summary["generated_at"],
            "case_count": len(results),
            "global_patch_summary": summary,
            "cases": serializable_results,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "all_scoped_patches.json").write_text(
        json.dumps(patches_to_dicts(all_patches), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines: List[str] = [
        "# Scoped Edit Lab",
        "",
        f"- case_count: {len(results)}",
        f"- patch_count: {summary['patch_count']}",
        f"- by_status: `{json.dumps(summary['by_status'], sort_keys=True)}`",
        f"- by_type: `{json.dumps(summary['by_type'], sort_keys=True)}`",
        f"- by_risk: `{json.dumps(summary['by_risk'], sort_keys=True)}`",
        f"- needs_attention_count: {summary['needs_attention_count']}",
        "",
        "## Cases",
        "",
        "| case | edits | patches | needs attention | source |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        patch_summary = result.get("patch_summary", {})
        lines.append(
            "| {case} | {edits} | {patches} | {attention} | `{source}` |".format(
                case=result.get("case_id", ""),
                edits=result.get("graph_edit_count", 0),
                patches=patch_summary.get("patch_count", 0),
                attention=patch_summary.get("needs_attention_count", 0),
                source=result.get("source_path", ""),
            )
        )
    lines.extend([
        "",
        "## Combined Patch Details",
        "",
        render_patch_report(all_patches, title="Combined Scoped Patch Report"),
    ])
    (out_dir / "edit_lab_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay graph edit artifacts through the scoped edit validator.")
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH, help="Path to the graph JSON.")
    parser.add_argument("--artifact-dir", type=Path, action="append", default=[], help="Artifact dir containing packet.json files.")
    parser.add_argument("--session-dir", type=Path, action="append", default=[], help="Session dir/root containing graph_edits.json files.")
    parser.add_argument("--packet", type=Path, action="append", default=[], help="Specific packet JSON file.")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT, help="Root directory for lab outputs.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Exact output directory. Defaults to out-root/timestamp.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max cases to process.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph = MemoryGraph.load_json(args.graph)
    out_dir = args.out_dir or (args.out_root / _now_slug())
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = discover_cases(
        artifact_dirs=args.artifact_dir,
        session_dirs=args.session_dir,
        packet_paths=args.packet,
    )
    if args.limit and args.limit > 0:
        cases = cases[:args.limit]
    if not cases:
        raise SystemExit("No graph edit cases found. Pass --artifact-dir, --session-dir, or --packet.")

    results = [run_case(case, graph, out_dir) for case in cases]
    _write_global_report(out_dir, results)
    print(f"Wrote scoped edit lab report: {out_dir / 'edit_lab_report.md'}")


if __name__ == "__main__":
    main()

