from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PACKET_FILENAMES = frozenset({"packet.json", "run_1_packet.json", "run_2_packet.json", "graph_packet.json"})


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _normalize_question(text: Any) -> str:
    raw = str(text or "").strip().lower()
    raw = re.sub(r"\s+", " ", raw)
    return raw


def _stable_list(values: Iterable[Any]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def load_label_cases(path: str | Path) -> List[Dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        cases = payload.get("cases", [])
    else:
        cases = payload
    if not isinstance(cases, list):
        raise ValueError("label file must be a list or an object with a 'cases' list")

    normalized: List[Dict[str, Any]] = []
    for row in cases:
        if not isinstance(row, Mapping):
            continue
        normalized.append({
            "id": str(row.get("id", "") or ""),
            "question": str(row.get("question", "") or ""),
            "task_family": str(row.get("task_family", "") or ""),
            "gold_signature_family_ids": _stable_list(row.get("gold_signature_family_ids", [])),
            "gold_signature_variant_ids": _stable_list(row.get("gold_signature_variant_ids", [])),
            "unsafe_family_ids": _stable_list(row.get("unsafe_family_ids", [])),
            "matching_expectation": dict(row.get("matching_expectation", {})),
            "notes": str(row.get("notes", "") or ""),
        })
    return normalized


def discover_packet_paths(paths: Sequence[str | Path]) -> List[Path]:
    found: List[Path] = []
    seen = set()
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file():
            if path.name in PACKET_FILENAMES and path not in seen:
                seen.add(path)
                found.append(path)
            continue
        if not path.exists():
            continue
        for candidate in path.rglob("*.json"):
            if candidate.name not in PACKET_FILENAMES or candidate in seen:
                continue
            seen.add(candidate)
            found.append(candidate)
    return sorted(found)


def _packet_task_family(packet: Mapping[str, Any]) -> str:
    return str(
        packet.get("controller_task_family")
        or packet.get("task_family")
        or packet.get("task_type")
        or ""
    )


def _match_label_to_packet(label: Mapping[str, Any], packet: Mapping[str, Any]) -> bool:
    if _normalize_question(label.get("question", "")) != _normalize_question(packet.get("question", "")):
        return False
    label_family = str(label.get("task_family", "") or "")
    packet_family = _packet_task_family(packet)
    if label_family and packet_family and label_family != packet_family:
        return False
    return True


def _rank_rows(shadow: Mapping[str, Any], prefix: str) -> Tuple[List[Dict[str, Any]], bool]:
    full_key = f"{prefix}_ranking"
    top_key = f"{prefix}_top_k"
    rows = shadow.get(full_key)
    if isinstance(rows, list) and rows:
        return [dict(row) for row in rows if isinstance(row, Mapping)], bool(shadow.get("ranking_complete", True))
    rows = shadow.get(top_key)
    if isinstance(rows, list) and rows:
        return [dict(row) for row in rows if isinstance(row, Mapping)], False
    return [], False


def _top_k_hit(rows: Sequence[Mapping[str, Any]], gold_family_ids: Sequence[str], k: int) -> Optional[int]:
    if not rows:
        return None
    gold = set(str(x) for x in gold_family_ids if str(x or "").strip())
    if not gold:
        return None
    return int(any(str(row.get("family_id", "") or "") in gold for row in rows[:k]))


def _top_k_variant_hit(rows: Sequence[Mapping[str, Any]], gold_variant_ids: Sequence[str], k: int) -> Optional[int]:
    if not rows:
        return None
    gold = set(str(x) for x in gold_variant_ids if str(x or "").strip())
    if not gold:
        return None
    return int(any(str(row.get("variant_id", "") or "") in gold for row in rows[:k]))


def _best_family_rank(rows: Sequence[Mapping[str, Any]], gold_family_ids: Sequence[str], rank_key: str, ranking_complete: bool) -> Optional[int]:
    if not rows:
        return None
    gold = set(str(x) for x in gold_family_ids if str(x or "").strip())
    if not gold:
        return None
    best: Optional[int] = None
    for i, row in enumerate(rows, start=1):
        family_id = str(row.get("family_id", "") or "")
        if family_id not in gold:
            continue
        rank = int(row.get(rank_key, i) or i)
        if best is None or rank < best:
            best = rank
    if best is not None:
        return best
    if ranking_complete:
        return len(rows) + 1
    return None


def _mrr(rank: Optional[int], row_count: int) -> Optional[float]:
    if rank is None:
        return None
    if rank > row_count:
        return 0.0
    if rank <= 0:
        return None
    return round(1.0 / rank, 6)


def _bool_mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _candidate_match_score(candidate: Mapping[str, Any], expectation: Mapping[str, Any], gold_family_ids: Sequence[str]) -> Optional[Tuple[int, str]]:
    semantic_type = str(expectation.get("semantic_type", "") or "")
    if semantic_type and str(candidate.get("semantic_type", "") or "") != semantic_type:
        return None

    score = 0
    reasons: List[str] = []
    target_family_ids = set(_stable_list(expectation.get("target_family_ids", [])) or _stable_list(gold_family_ids))
    candidate_family_ids = {
        str(candidate.get("family_id", "") or ""),
        str(candidate.get("matched_family_id", "") or ""),
        str(candidate.get("proposed_family_id", "") or ""),
    }
    candidate_family_ids = {x for x in candidate_family_ids if x}
    family_overlap = sorted(target_family_ids & candidate_family_ids)
    if family_overlap:
        score += 10
        reasons.append(f"family_match:{family_overlap[0]}")

    target_source_node_ids = set(_stable_list(expectation.get("target_source_node_ids", [])))
    candidate_source_ids = set(_stable_list(candidate.get("source_node_ids", [])))
    source_overlap = sorted(target_source_node_ids & candidate_source_ids)
    if source_overlap:
        score += 6
        reasons.append(f"source_match:{source_overlap[0]}")

    target_variant_ids = set(_stable_list(expectation.get("target_variant_ids", [])))
    if str(candidate.get("variant_id", "") or "") in target_variant_ids:
        score += 12
        reasons.append("variant_match")

    if semantic_type:
        score += 2
        reasons.append(f"semantic_type:{semantic_type}")

    if not reasons:
        if len(candidate_source_ids) == 1:
            score += 1
            reasons.append("fallback_single_source")
        elif str(candidate.get("semantic_type", "") or ""):
            score += 1
            reasons.append("fallback_any_candidate")

    if score <= 0:
        return None
    return score, ",".join(reasons)


def _select_resolution_candidate(
    candidates: Sequence[Mapping[str, Any]],
    expectation: Mapping[str, Any],
    gold_family_ids: Sequence[str],
) -> Tuple[Optional[Dict[str, Any]], str]:
    scored: List[Tuple[int, str, Dict[str, Any]]] = []
    for candidate in candidates:
        score = _candidate_match_score(candidate, expectation, gold_family_ids)
        if score is None:
            continue
        points, reason = score
        scored.append((points, reason, dict(candidate)))
    if not scored:
        if len(candidates) == 1:
            return dict(candidates[0]), "fallback_only_candidate"
        return None, "no_candidate_match"
    scored.sort(
        key=lambda row: (
            row[0],
            str(row[2].get("semantic_type", "") or ""),
            str(row[2].get("family_id", "") or ""),
            str(row[2].get("variant_id", "") or ""),
        ),
        reverse=True,
    )
    best_points, best_reason, best_candidate = scored[0]
    return best_candidate, f"score={best_points}:{best_reason}"


def evaluate_case_against_packet(label: Mapping[str, Any], packet_path: Path, packet: Mapping[str, Any]) -> Dict[str, Any]:
    question = str(packet.get("question", "") or "")
    task_family = _packet_task_family(packet)
    shadow = packet.get("signature_shadow_report")
    result: Dict[str, Any] = {
        "label_id": str(label.get("id", "") or ""),
        "packet_path": str(packet_path),
        "question": question,
        "task_family": task_family,
        "status": "evaluated",
        "skip_reason": "",
        "notes": str(label.get("notes", "") or ""),
    }
    if not isinstance(shadow, Mapping) or not shadow:
        result["status"] = "skipped"
        result["skip_reason"] = "missing_shadow_report"
        return result

    baseline_rows, baseline_complete = _rank_rows(shadow, "baseline")
    adjusted_rows, adjusted_complete = _rank_rows(shadow, "adjusted")
    if not baseline_rows or not adjusted_rows:
        result["status"] = "skipped"
        result["skip_reason"] = "missing_shadow_rankings"
        return result

    gold_family_ids = _stable_list(label.get("gold_signature_family_ids", []))
    gold_variant_ids = _stable_list(label.get("gold_signature_variant_ids", []))
    unsafe_family_ids = set(_stable_list(label.get("unsafe_family_ids", [])))
    ranking_complete = bool(shadow.get("ranking_complete", False)) and baseline_complete and adjusted_complete
    result["ranking_complete"] = ranking_complete
    result["candidate_count"] = max(len(baseline_rows), len(adjusted_rows))
    result["family_count"] = int(shadow.get("family_count", 0) or 0)

    for k in (1, 3, 5):
        result[f"family_hit_at_{k}_baseline"] = _top_k_hit(baseline_rows, gold_family_ids, k)
        result[f"family_hit_at_{k}_adjusted"] = _top_k_hit(adjusted_rows, gold_family_ids, k)
    if gold_variant_ids:
        for k in (1, 3):
            result[f"variant_hit_at_{k}_adjusted"] = _top_k_variant_hit(adjusted_rows, gold_variant_ids, k)

    baseline_rank = _best_family_rank(baseline_rows, gold_family_ids, "baseline_rank", ranking_complete)
    adjusted_rank = _best_family_rank(adjusted_rows, gold_family_ids, "adjusted_rank", ranking_complete)
    result["family_rank_baseline"] = baseline_rank
    result["family_rank_adjusted"] = adjusted_rank
    if baseline_rank is not None and adjusted_rank is not None:
        result["family_rank_delta"] = baseline_rank - adjusted_rank
    else:
        result["family_rank_delta"] = None
    result["family_mrr_baseline"] = _mrr(baseline_rank, len(baseline_rows))
    result["family_mrr_adjusted"] = _mrr(adjusted_rank, len(adjusted_rows))
    if result["family_mrr_baseline"] is not None and result["family_mrr_adjusted"] is not None:
        result["delta_family_mrr"] = round(result["family_mrr_adjusted"] - result["family_mrr_baseline"], 6)
    else:
        result["delta_family_mrr"] = None

    top1 = adjusted_rows[0] if adjusted_rows else {}
    result["contested_family_top1_flag"] = int(bool(top1.get("family_contested", False))) if top1 else None
    result["audit_only_or_provisional_top1_flag"] = (
        int(
            str(top1.get("retrieval_tier", "") or "") == "audit_only"
            or str(top1.get("epistemic_status", "") or "") == "provisional"
        )
        if top1
        else None
    )
    if unsafe_family_ids:
        result["unsafe_top3_flag"] = int(
            any(str(row.get("family_id", "") or "") in unsafe_family_ids for row in adjusted_rows[:3])
        )
    else:
        result["unsafe_top3_flag"] = None

    candidates = [dict(row) for row in packet.get("signature_candidates", []) if isinstance(row, Mapping)]
    expectation = dict(label.get("matching_expectation", {}))
    selected_candidate: Optional[Dict[str, Any]] = None
    if expectation:
        selected_candidate, selection_reason = _select_resolution_candidate(candidates, expectation, gold_family_ids)
        result["resolution_selection_reason"] = selection_reason
        if selected_candidate is None:
            result["resolution_status"] = "missing_target_candidate"
        else:
            result["resolution_status"] = "evaluated"
            result["selected_candidate"] = {
                "variant_id": str(selected_candidate.get("variant_id", "") or ""),
                "family_id": str(selected_candidate.get("family_id", "") or ""),
                "semantic_type": str(selected_candidate.get("semantic_type", "") or ""),
                "family_resolution": str(selected_candidate.get("family_resolution", "") or ""),
                "variant_resolution": str(selected_candidate.get("variant_resolution", "") or ""),
                "relation_to_match": str(selected_candidate.get("relation_to_match", "") or ""),
            }

            expected_variant_resolution = str(expectation.get("expected_variant_resolution", "") or "")
            expected_family_resolution = str(expectation.get("expected_family_resolution", "") or "")
            should_match_existing_family = bool(expectation.get("should_match_existing_family", False))

            if expected_variant_resolution:
                result["variant_resolution_expected"] = expected_variant_resolution
                result["variant_resolution_correct"] = int(
                    str(selected_candidate.get("variant_resolution", "") or "") == expected_variant_resolution
                )
            if expected_family_resolution:
                result["family_resolution_expected"] = expected_family_resolution
                result["family_resolution_correct"] = int(
                    str(selected_candidate.get("family_resolution", "") or "") == expected_family_resolution
                )
            if should_match_existing_family:
                result["should_match_existing_family"] = True
                result["new_family_false_split_flag"] = int(
                    str(selected_candidate.get("family_resolution", "") or "") == "new_family"
                )
    return result


def evaluate_shadow_labels(artifact_paths: Sequence[str | Path], labels_path: str | Path) -> Dict[str, Any]:
    labels = load_label_cases(labels_path)
    packets: List[Tuple[Path, Dict[str, Any]]] = []
    for packet_path in discover_packet_paths(artifact_paths):
        try:
            payload = json.loads(packet_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, Mapping):
            packets.append((packet_path, dict(payload)))

    case_results: List[Dict[str, Any]] = []
    unmatched_label_ids: List[str] = []
    for label in labels:
        matches = [(path, packet) for path, packet in packets if _match_label_to_packet(label, packet)]
        if not matches:
            unmatched_label_ids.append(str(label.get("id", "") or ""))
            continue
        for packet_path, packet in matches:
            case_results.append(evaluate_case_against_packet(label, packet_path, packet))

    skip_reason_counts = Counter(row["skip_reason"] for row in case_results if row.get("skip_reason"))
    metric_values: Dict[str, List[float]] = defaultdict(list)

    def record(metric: str, value: Any) -> None:
        if value is None:
            return
        metric_values[metric].append(float(value))

    for row in case_results:
        if row.get("status") == "skipped":
            continue
        for k in (1, 3, 5):
            record(f"family_hit_at_{k}_baseline", row.get(f"family_hit_at_{k}_baseline"))
            record(f"family_hit_at_{k}_adjusted", row.get(f"family_hit_at_{k}_adjusted"))
        for k in (1, 3):
            record(f"variant_hit_at_{k}_adjusted", row.get(f"variant_hit_at_{k}_adjusted"))
        record("family_rank_delta", row.get("family_rank_delta"))
        record("family_mrr_baseline", row.get("family_mrr_baseline"))
        record("family_mrr_adjusted", row.get("family_mrr_adjusted"))
        record("delta_family_mrr", row.get("delta_family_mrr"))
        record("contested_family_top1_rate", row.get("contested_family_top1_flag"))
        record("audit_only_or_provisional_top1_rate", row.get("audit_only_or_provisional_top1_flag"))
        record("unsafe_top3_rate", row.get("unsafe_top3_flag"))
        if row.get("variant_resolution_expected") == "equivalent_revision":
            record("equivalent_revision_precision", row.get("variant_resolution_correct"))
        if row.get("variant_resolution_expected") == "sibling_variant":
            record("sibling_variant_precision", row.get("variant_resolution_correct"))
        if row.get("should_match_existing_family"):
            record("new_family_false_split_rate", row.get("new_family_false_split_flag"))

    family_rank_deltas = metric_values.get("family_rank_delta", [])
    summary_metrics: Dict[str, Any] = {}
    for name, values in sorted(metric_values.items()):
        if name == "family_rank_delta":
            continue
        summary_metrics[name] = _bool_mean(values)
    summary_metrics["family_rank_delta_mean"] = _bool_mean(family_rank_deltas)
    summary_metrics["family_rank_delta_median"] = (
        round(float(statistics.median(family_rank_deltas)), 6) if family_rank_deltas else None
    )
    summary_metrics["family_rank_delta_win_rate"] = (
        round(sum(1.0 for x in family_rank_deltas if x > 0) / len(family_rank_deltas), 6)
        if family_rank_deltas
        else None
    )

    return {
        "generated_at": _now_iso(),
        "labels_path": str(labels_path),
        "artifact_paths": [str(Path(p)) for p in artifact_paths],
        "packet_count": len(packets),
        "label_count": len(labels),
        "matched_case_count": len(case_results),
        "evaluated_case_count": sum(1 for row in case_results if row.get("status") != "skipped"),
        "skipped_case_count": sum(1 for row in case_results if row.get("status") == "skipped"),
        "full_ranking_case_count": sum(1 for row in case_results if row.get("ranking_complete")),
        "unmatched_label_ids": unmatched_label_ids,
        "skip_reason_counts": dict(skip_reason_counts),
        "metric_counts": {name: len(values) for name, values in sorted(metric_values.items())},
        "metrics": summary_metrics,
        "cases": case_results,
    }


def render_markdown_report(report: Mapping[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Signature Shadow Eval")
    lines.append("")
    lines.append(f"- Generated: `{report.get('generated_at', '')}`")
    lines.append(f"- Packets scanned: `{report.get('packet_count', 0)}`")
    lines.append(f"- Labels: `{report.get('label_count', 0)}`")
    lines.append(f"- Matched cases: `{report.get('matched_case_count', 0)}`")
    lines.append(f"- Evaluated cases: `{report.get('evaluated_case_count', 0)}`")
    lines.append(f"- Skipped cases: `{report.get('skipped_case_count', 0)}`")
    lines.append(f"- Full-ranking cases: `{report.get('full_ranking_case_count', 0)}`")
    unmatched = list(report.get("unmatched_label_ids", []) or [])
    if unmatched:
        lines.append(f"- Unmatched labels: `{len(unmatched)}`")
    lines.append("")

    metrics = dict(report.get("metrics", {}))
    if metrics:
        lines.append("## Metrics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("| --- | ---: |")
        for key in sorted(metrics):
            value = metrics[key]
            rendered = "-" if value is None else f"{value:.6f}" if isinstance(value, float) else str(value)
            lines.append(f"| `{key}` | {rendered} |")
        lines.append("")

    skip_reason_counts = dict(report.get("skip_reason_counts", {}))
    if skip_reason_counts:
        lines.append("## Skip Reasons")
        lines.append("")
        for reason, count in sorted(skip_reason_counts.items()):
            lines.append(f"- `{reason}`: `{count}`")
        lines.append("")

    cases = list(report.get("cases", []) or [])
    if cases:
        lines.append("## Cases")
        lines.append("")
        for row in cases:
            lines.append(f"### {row.get('label_id', 'unknown')} :: {Path(str(row.get('packet_path', ''))).name}")
            lines.append("")
            lines.append(f"- Question: {row.get('question', '')}")
            lines.append(f"- Task family: `{row.get('task_family', '')}`")
            lines.append(f"- Status: `{row.get('status', '')}`")
            if row.get("skip_reason"):
                lines.append(f"- Skip reason: `{row.get('skip_reason', '')}`")
            else:
                lines.append(f"- family_rank_baseline: `{row.get('family_rank_baseline', '-')}`")
                lines.append(f"- family_rank_adjusted: `{row.get('family_rank_adjusted', '-')}`")
                lines.append(f"- family_rank_delta: `{row.get('family_rank_delta', '-')}`")
                lines.append(f"- delta_family_mrr: `{row.get('delta_family_mrr', '-')}`")
                if "selected_candidate" in row:
                    picked = row["selected_candidate"]
                    lines.append(
                        "- selected_candidate: "
                        f"`{picked.get('semantic_type', '')}` "
                        f"`{picked.get('family_resolution', '')}` / "
                        f"`{picked.get('variant_resolution', '')}`"
                    )
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Layer 2 shadow retrieval artifacts against labeled gold families.")
    parser.add_argument("artifact_paths", nargs="+", help="Artifact directories or packet.json files to scan.")
    parser.add_argument("--labels", required=True, help="JSON label file path.")
    parser.add_argument("--out-json", help="Optional output JSON path.")
    parser.add_argument("--out-md", help="Optional output Markdown report path.")
    args = parser.parse_args()

    report = evaluate_shadow_labels(args.artifact_paths, args.labels)
    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.out_md:
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(render_markdown_report(report), encoding="utf-8")

    print(json.dumps({
        "packet_count": report["packet_count"],
        "matched_case_count": report["matched_case_count"],
        "evaluated_case_count": report["evaluated_case_count"],
        "skipped_case_count": report["skipped_case_count"],
        "metrics": report["metrics"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
