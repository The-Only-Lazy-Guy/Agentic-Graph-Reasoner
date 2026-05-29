from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write_text(path: Path, lines: Iterable[str]) -> None:
    path.write_text("\n".join(lines), encoding="utf-8")


def _load_summary(path: str | Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"summary must be a JSON object: {path}")
    return payload


def _index_cases(summary: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    rows = summary.get("cases", [])
    indexed: Dict[str, Dict[str, Any]] = {}
    if not isinstance(rows, list):
        return indexed
    for row in rows:
        if not isinstance(row, dict):
            continue
        case_id = str(row.get("id", "") or "")
        if case_id:
            indexed[case_id] = row
    return indexed


def _case_status(before: Mapping[str, Any], after: Mapping[str, Any]) -> str:
    before_finalized = bool(before.get("finalized"))
    after_finalized = bool(after.get("finalized"))
    before_steps = int(before.get("steps", 0) or 0)
    after_steps = int(after.get("steps", 0) or 0)
    before_tools = int(before.get("tool_call_count", 0) or 0)
    after_tools = int(after.get("tool_call_count", 0) or 0)
    if before_finalized and not after_finalized:
        return "regressed"
    if not before_finalized and after_finalized:
        return "improved"
    if after_steps < before_steps or after_tools < before_tools:
        return "improved"
    if after_steps > before_steps or after_tools > before_tools:
        return "regressed"
    return "same"


def _summarize_pair(case_id: str, before: Mapping[str, Any], after: Mapping[str, Any]) -> Dict[str, Any]:
    before_steps = int(before.get("steps", 0) or 0)
    after_steps = int(after.get("steps", 0) or 0)
    before_tools = int(before.get("tool_call_count", 0) or 0)
    after_tools = int(after.get("tool_call_count", 0) or 0)
    before_elapsed = float(before.get("elapsed_sec", 0.0) or 0.0)
    after_elapsed = float(after.get("elapsed_sec", 0.0) or 0.0)
    after_live_bias = dict(after.get("signature_live_bias") or {})
    return {
        "id": case_id,
        "difficulty": before.get("difficulty") or after.get("difficulty"),
        "question": before.get("question") or after.get("question"),
        "status": _case_status(before, after),
        "execution_mode_before": before.get("execution_mode"),
        "execution_mode_after": after.get("execution_mode"),
        "finalized_before": bool(before.get("finalized")),
        "finalized_after": bool(after.get("finalized")),
        "steps_before": before_steps,
        "steps_after": after_steps,
        "steps_delta": after_steps - before_steps,
        "tool_calls_before": before_tools,
        "tool_calls_after": after_tools,
        "tool_calls_delta": after_tools - before_tools,
        "elapsed_before": round(before_elapsed, 3),
        "elapsed_after": round(after_elapsed, 3),
        "elapsed_delta": round(after_elapsed - before_elapsed, 3),
        "anchors_before": list(before.get("anchors", []) or []),
        "anchors_after": list(after.get("anchors", []) or []),
        "signature_live_bias_enabled": bool(after.get("signature_live_bias_enabled")),
        "signature_live_bias_applied": bool(after.get("signature_live_bias_applied")),
        "signature_live_bias_reason": str(after.get("signature_live_bias_reason", "") or ""),
        "signature_live_bias_family_id": str(after.get("signature_live_bias_family_id", "") or ""),
        "signature_live_bias_anchor_ids": list(after.get("signature_live_bias_anchor_ids", []) or []),
        "signature_live_bias_full": after_live_bias,
    }


def compare_summaries(
    *,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> Dict[str, Any]:
    baseline_cases = _index_cases(baseline)
    candidate_cases = _index_cases(candidate)
    shared_case_ids = sorted(set(baseline_cases) & set(candidate_cases))
    rows = [_summarize_pair(case_id, baseline_cases[case_id], candidate_cases[case_id]) for case_id in shared_case_ids]
    improved = [row for row in rows if row["status"] == "improved"]
    regressed = [row for row in rows if row["status"] == "regressed"]
    same = [row for row in rows if row["status"] == "same"]
    return {
        "baseline_summary": str(baseline.get("out_dir", "") or ""),
        "candidate_summary": str(candidate.get("out_dir", "") or ""),
        "baseline_enable_signature_live_bias": bool(baseline.get("enable_signature_live_bias")),
        "candidate_enable_signature_live_bias": bool(candidate.get("enable_signature_live_bias")),
        "case_count": len(rows),
        "improved_count": len(improved),
        "regressed_count": len(regressed),
        "same_count": len(same),
        "mean_steps_delta": round(sum(row["steps_delta"] for row in rows) / len(rows), 4) if rows else 0.0,
        "mean_tool_calls_delta": round(sum(row["tool_calls_delta"] for row in rows) / len(rows), 4) if rows else 0.0,
        "mean_elapsed_delta": round(sum(row["elapsed_delta"] for row in rows) / len(rows), 4) if rows else 0.0,
        "live_bias_applied_count": sum(1 for row in rows if row["signature_live_bias_applied"]),
        "rows": rows,
    }


def _write_report(path: Path, result: Mapping[str, Any]) -> None:
    lines: List[str] = [
        "# V4 Sweep Comparison",
        "",
        f"- baseline_summary: {result.get('baseline_summary', '')}",
        f"- candidate_summary: {result.get('candidate_summary', '')}",
        f"- baseline_enable_signature_live_bias: {result.get('baseline_enable_signature_live_bias')}",
        f"- candidate_enable_signature_live_bias: {result.get('candidate_enable_signature_live_bias')}",
        f"- case_count: {result.get('case_count')}",
        f"- improved_count: {result.get('improved_count')}",
        f"- regressed_count: {result.get('regressed_count')}",
        f"- same_count: {result.get('same_count')}",
        f"- mean_steps_delta: {result.get('mean_steps_delta')}",
        f"- mean_tool_calls_delta: {result.get('mean_tool_calls_delta')}",
        f"- mean_elapsed_delta: {result.get('mean_elapsed_delta')}",
        f"- live_bias_applied_count: {result.get('live_bias_applied_count')}",
        "",
        "## Cases",
        "",
        "| Case | Status | Finalized | Steps | Tools | Elapsed | Live Bias |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in result.get("rows", []):
        lines.append(
            "| {id} | {status} | {fb}->{fa} | {sb}->{sa} ({sd:+d}) | {tb}->{ta} ({td:+d}) | {eb}->{ea} ({ed:+.3f}) | {lb} |".format(
                id=row.get("id", ""),
                status=row.get("status", ""),
                fb="Y" if row.get("finalized_before") else "N",
                fa="Y" if row.get("finalized_after") else "N",
                sb=row.get("steps_before", 0),
                sa=row.get("steps_after", 0),
                sd=int(row.get("steps_delta", 0) or 0),
                tb=row.get("tool_calls_before", 0),
                ta=row.get("tool_calls_after", 0),
                td=int(row.get("tool_calls_delta", 0) or 0),
                eb=row.get("elapsed_before", 0.0),
                ea=row.get("elapsed_after", 0.0),
                ed=float(row.get("elapsed_delta", 0.0) or 0.0),
                lb=(
                    "applied"
                    if row.get("signature_live_bias_applied")
                    else ("eligible" if row.get("signature_live_bias_enabled") else "off")
                ),
            )
        )
    lines.extend(["", "## Detailed JSON", "", "```json", json.dumps(_jsonable(result), ensure_ascii=False, indent=2), "```"])
    _write_text(path, lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two V4 difficulty sweep summary.json files.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    baseline = _load_summary(args.baseline)
    candidate = _load_summary(args.candidate)
    result = compare_summaries(baseline=baseline, candidate=candidate)

    out_dir = args.out_dir or (args.candidate.parent / "comparison_against_baseline")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "compare.json").write_text(json.dumps(_jsonable(result), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(out_dir / "compare.md", result)
    print(json.dumps(_jsonable({
        "out_dir": out_dir,
        "case_count": result.get("case_count"),
        "improved_count": result.get("improved_count"),
        "regressed_count": result.get("regressed_count"),
        "same_count": result.get("same_count"),
        "live_bias_applied_count": result.get("live_bias_applied_count"),
        "mean_steps_delta": result.get("mean_steps_delta"),
        "mean_tool_calls_delta": result.get("mean_tool_calls_delta"),
        "mean_elapsed_delta": result.get("mean_elapsed_delta"),
    }), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
