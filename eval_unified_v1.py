from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

from pred_model import COMMIT_FAMILIES, MEM_LINK_KIND_TO_ID, REL_WITH_NONE
from train_pred_v1 import decode_edge_predictions, decode_mem_kind_predictions, read_jsonl
from train_unified_v1 import (
    UnifiedDataset,
    _ensure_target_slots,
    aggregate_binary_metrics,
    build_candidate_memory_ids,
    build_predicted_session_nodes,
    collate,
    derive_gold_targets,
    set_seed,
    to_device,
)
from unified_proposal_aligner_model import UnifiedProposalAlignerNet


def build_model_from_checkpoint(
    checkpoint: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    dataset: UnifiedDataset,
    device: torch.device,
) -> UnifiedProposalAlignerNet:
    args = checkpoint.get("args", {}) or {}
    model = UnifiedProposalAlignerNet(
        hash_dim=int(args.get("hash_dim", 512)),
        hidden_dim=int(args.get("hidden_dim", 256)),
        k_max=3,
        cand_emb_dim=dataset.cand_emb_dim,
        mem_emb_dim=dataset.mem_emb_dim,
        edge_pair_feat_dim=int(args.get("edge_pair_feat_dim", 0)),
        use_verifier=bool(args.get("verifier_weight", 0) > 0),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    return model


def row_debug_payload(
    row: Mapping[str, Any],
    pred_nodes: Sequence[Mapping[str, Any]],
    gold: Mapping[str, Any],
    failure_flags: Mapping[str, bool],
) -> Dict[str, Any]:
    goal = row.get("_oracle_goal") or row.get("goal") or {}
    return {
        "id": row.get("id", ""),
        "task_type": row.get("task_type", ""),
        "signal": row.get("signal", ""),
        "failure_flags": dict(failure_flags),
        "pred_session_nodes": list(pred_nodes),
        "gold_session_nodes": list(goal.get("session_nodes", []) or []),
        "gold_session_edges": list(goal.get("session_edges", []) or []),
        "gold_memory_attachments": list(goal.get("memory_attachments", []) or []),
        "gold_covered_mappings": list(goal.get("covered_mappings", []) or []),
        "gold_commit_family": gold["gold_commit_family"],
    }


def evaluate(
    model: UnifiedProposalAlignerNet,
    loader: DataLoader,
    device: torch.device,
    *,
    sample_limit: int = 5,
) -> Dict[str, Any]:
    span_total = span_ok = 0
    text_total = text_ok = 0
    use_total = use_ok = 0
    commit_total = commit_ok = 0
    edge_tp = edge_fp = edge_fn = 0
    edge_rel_total = edge_rel_ok = 0
    attach_tp = attach_fp = attach_fn = 0
    attach_rel_total = attach_rel_ok = 0
    cover_tp = cover_fp = cover_fn = 0
    row_total = row_ok = 0
    text_row_ok = 0

    per_task: Dict[str, Dict[str, Any]] = {}
    failure_breakdown: Dict[str, int] = {}
    failure_combo_breakdown: Dict[str, int] = {}
    isolated_failure_breakdown: Dict[str, int] = {}
    per_task_failure_breakdown: Dict[str, Dict[str, int]] = {}
    per_task_failure_combo_breakdown: Dict[str, Dict[str, int]] = {}
    per_task_isolated_failure_breakdown: Dict[str, Dict[str, int]] = {}
    slot_span_stats: Dict[str, Dict[str, int]] = {}
    success_samples: List[Dict[str, Any]] = []
    failure_samples: List[Dict[str, Any]] = []

    model.eval()
    with torch.no_grad():
        for batch, rows in loader:
            batch = to_device(batch, device)
            out = model(batch)
            use_pred = (torch.sigmoid(out["use_logits"]) >= 0.5) & batch.slot_mask
            bridge_pred = (torch.sigmoid(out["type_logits"]) >= 0.5) & use_pred
            span_pred = out["span_logits"].argmax(dim=-1)
            commit_pred = out["commit_logits"].argmax(dim=-1)
            pred_edge_mask = use_pred[:, :, None] & use_pred[:, None, :]
            diag = torch.eye(use_pred.size(1), device=device, dtype=torch.bool)[None, :, :]
            pred_edge_mask = pred_edge_mask & ~diag
            edge_exist_pred = decode_edge_predictions(out["edge_exist_logits"], pred_edge_mask)
            edge_rel_pred = out["edge_rel_logits"].argmax(dim=-1)
            mem_kind_pred = decode_mem_kind_predictions(out["mem_kind_logits"], batch.mem_mask)
            mem_rel_pred = out["mem_rel_logits"].argmax(dim=-1)
            mixed_dst_pred = out["mixed_dst_mem_logits"].argmax(dim=-1)
            bridge_a_pred = out["bridge_mem_a_logits"].argmax(dim=-1)
            bridge_b_pred = out["bridge_mem_b_logits"].argmax(dim=-1)

            for b, row in enumerate(rows):
                task_type = str(row.get("task_type", ""))
                task_stats = per_task.setdefault(
                    task_type,
                    {"rows": 0, "row_complete": 0, "text_faithful_row_complete": 0},
                )
                task_stats["rows"] += 1
                task_failure = per_task_failure_breakdown.setdefault(task_type, {})
                task_failure_combo = per_task_failure_combo_breakdown.setdefault(task_type, {})
                task_isolated = per_task_isolated_failure_breakdown.setdefault(task_type, {})

                gold = derive_gold_targets(row)
                graph = loader.dataset.graph(str(row.get("graph_path", "")))  # type: ignore[attr-defined]
                memory_ids = build_candidate_memory_ids(row, graph)

                pred_nodes = build_predicted_session_nodes(
                    row,
                    use_pred=use_pred[b].cpu(),
                    span_pred=span_pred[b].cpu(),
                    bridge_pred=bridge_pred[b].cpu(),
                    mixed_dst_pred=mixed_dst_pred[b].cpu(),
                    bridge_a_pred=bridge_a_pred[b].cpu(),
                    bridge_b_pred=bridge_b_pred[b].cpu(),
                    memory_ids=memory_ids,
                    graph=graph,
                )
                pred_names = [str(node.get("name", "")) for node in pred_nodes]
                pred_name_set = set(pred_names)
                pred_text_by_name = {str(node.get("name", "")): str(node.get("span_text", "")) for node in pred_nodes}
                _eval_slots = _ensure_target_slots(row)
                pred_name_to_slot = {
                    str(_eval_slots[k].get("session_name", f"slot_{k}")): k
                    for k in range(len(_eval_slots))
                }

                gold_names = set(gold["gold_names"])
                use_gold = batch.y_use[b] > 0.5
                use_match = (use_pred[b] == use_gold) & batch.slot_mask[b]
                use_total += int(batch.slot_mask[b].sum().item())
                use_ok += int(use_match.sum().item())

                span_ok_count = 0
                span_total_row = len(gold_names)
                text_ok_count = 0
                for name in gold_names:
                    slot_idx = pred_name_to_slot.get(name)
                    if slot_idx is None or not bool(use_pred[b, slot_idx].item()):
                        continue
                    pred_span_idx = int(span_pred[b, slot_idx].item())
                    pred_span_id = None
                    spans = row.get("spans", []) or []
                    if 0 <= pred_span_idx < len(spans):
                        pred_span_id = str(spans[pred_span_idx].get("id", ""))
                    if pred_span_id == gold["gold_span_by_name"].get(name):
                        span_ok_count += 1
                    if pred_text_by_name.get(name, "") == gold["gold_text_by_name"].get(name, ""):
                        text_ok_count += 1

                span_total += span_total_row
                span_ok += span_ok_count
                text_total += span_total_row
                text_ok += text_ok_count

                target_slots = _ensure_target_slots(row)
                for k, slot in enumerate(target_slots):
                    if not bool(slot.get("use")):
                        continue
                    stat_key = f"slot_{k}"
                    slot_stats = slot_span_stats.setdefault(task_type, {})
                    slot_stats[f"{stat_key}_total"] = slot_stats.get(f"{stat_key}_total", 0) + 1
                    pred_span_idx = int(span_pred[b, k].item())
                    pred_span_id = None
                    spans = row.get("spans", []) or []
                    if 0 <= pred_span_idx < len(spans):
                        pred_span_id = str(spans[pred_span_idx].get("id", ""))
                    if pred_span_id == str(slot.get("span_id", "")):
                        slot_stats[f"{stat_key}_ok"] = slot_stats.get(f"{stat_key}_ok", 0) + 1

                predicted_edges = set()
                for src_name in pred_names:
                    src_slot = pred_name_to_slot.get(src_name)
                    if src_slot is None:
                        continue
                    for dst_name in pred_names:
                        if src_name == dst_name:
                            continue
                        dst_slot = pred_name_to_slot.get(dst_name)
                        if dst_slot is None:
                            continue
                        if bool(edge_exist_pred[b, src_slot, dst_slot].item()):
                            rel_id = int(edge_rel_pred[b, src_slot, dst_slot].item())
                            rel = REL_WITH_NONE[rel_id]
                            predicted_edges.add((src_name, dst_name, rel))

                edge_tp_set = predicted_edges & gold["gold_edges"]
                edge_fp += len(predicted_edges - gold["gold_edges"])
                edge_fn += len(gold["gold_edges"] - predicted_edges)
                edge_tp += len(edge_tp_set)
                edge_rel_total += len(gold["gold_edges"])
                edge_rel_ok += len(edge_tp_set)

                predicted_attach = set()
                predicted_cover = set()
                for name in pred_names:
                    slot_idx = pred_name_to_slot.get(name)
                    if slot_idx is None:
                        continue
                    for m_idx, mem_id in enumerate(memory_ids):
                        kind_id = int(mem_kind_pred[b, slot_idx, m_idx].item())
                        if kind_id == MEM_LINK_KIND_TO_ID["attach"]:
                            rel_id = int(mem_rel_pred[b, slot_idx, m_idx].item())
                            predicted_attach.add((name, str(mem_id), REL_WITH_NONE[rel_id]))
                        elif kind_id == MEM_LINK_KIND_TO_ID["cover"]:
                            predicted_cover.add((name, str(mem_id)))

                attach_tp_set = predicted_attach & gold["gold_attach"]
                attach_tp += len(attach_tp_set)
                attach_fp += len(predicted_attach - gold["gold_attach"])
                attach_fn += len(gold["gold_attach"] - predicted_attach)
                attach_rel_total += len(gold["gold_attach"])
                attach_rel_ok += len(attach_tp_set)

                cover_tp_set = predicted_cover & gold["gold_cover"]
                cover_tp += len(cover_tp_set)
                cover_fp += len(predicted_cover - gold["gold_cover"])
                cover_fn += len(gold["gold_cover"] - predicted_cover)

                commit_name = COMMIT_FAMILIES[int(commit_pred[b].item())]
                commit_ok_row = commit_name == gold["gold_commit_family"]
                commit_total += 1
                commit_ok += int(commit_ok_row)

                row_components = {
                    "node_set": pred_name_set == gold_names,
                    "use": bool(use_match[batch.slot_mask[b]].all().item()),
                    "span": span_ok_count == span_total_row,
                    "text": text_ok_count == span_total_row,
                    "commit": commit_ok_row,
                    "edge": len(predicted_edges - gold["gold_edges"]) == 0 and len(gold["gold_edges"] - predicted_edges) == 0,
                    "attach": len(predicted_attach - gold["gold_attach"]) == 0 and len(gold["gold_attach"] - predicted_attach) == 0,
                    "cover": len(predicted_cover - gold["gold_cover"]) == 0 and len(gold["gold_cover"] - predicted_cover) == 0,
                }
                row_complete = all(
                    row_components[key] for key in ("node_set", "use", "span", "commit", "edge", "attach", "cover")
                )
                text_faithful_row_complete = row_complete and row_components["text"]

                row_total += 1
                row_ok += int(row_complete)
                text_row_ok += int(text_faithful_row_complete)
                task_stats["row_complete"] += int(row_complete)
                task_stats["text_faithful_row_complete"] += int(text_faithful_row_complete)

                if row_complete:
                    if len(success_samples) < sample_limit:
                        success_samples.append(row_debug_payload(row, pred_nodes, gold, row_components))
                else:
                    failing = [name for name, ok in row_components.items() if not ok]
                    for name in failing:
                        failure_breakdown[name] = failure_breakdown.get(name, 0) + 1
                        task_failure[name] = task_failure.get(name, 0) + 1
                    combo_key = "+".join(sorted(failing))
                    failure_combo_breakdown[combo_key] = failure_combo_breakdown.get(combo_key, 0) + 1
                    task_failure_combo[combo_key] = task_failure_combo.get(combo_key, 0) + 1
                    if len(failing) == 1:
                        only = failing[0]
                        isolated_failure_breakdown[only] = isolated_failure_breakdown.get(only, 0) + 1
                        task_isolated[only] = task_isolated.get(only, 0) + 1
                    if len(failure_samples) < sample_limit:
                        failure_samples.append(row_debug_payload(row, pred_nodes, gold, row_components))

    edge_precision, edge_recall, edge_f1 = aggregate_binary_metrics(edge_tp, edge_fp, edge_fn)
    attach_precision, attach_recall, attach_f1 = aggregate_binary_metrics(attach_tp, attach_fp, attach_fn)
    cover_precision, cover_recall, cover_f1 = aggregate_binary_metrics(cover_tp, cover_fp, cover_fn)
    per_task_rates = {
        task: {
            "rows": stats["rows"],
            "row_complete_rate": stats["row_complete"] / max(stats["rows"], 1),
            "text_faithful_row_complete_rate": stats["text_faithful_row_complete"] / max(stats["rows"], 1),
            "slot_span_acc": {
                key.replace("_ok", ""): slot_span_stats.get(task, {}).get(key, 0)
                / max(slot_span_stats.get(task, {}).get(key.replace("_ok", "_total"), 0), 1)
                for key in sorted(slot_span_stats.get(task, {}))
                if key.endswith("_ok")
            },
            "failure_breakdown": dict(
                sorted(per_task_failure_breakdown.get(task, {}).items(), key=lambda kv: (-kv[1], kv[0]))
            ),
            "isolated_failure_breakdown": dict(
                sorted(per_task_isolated_failure_breakdown.get(task, {}).items(), key=lambda kv: (-kv[1], kv[0]))
            ),
            "failure_combo_breakdown_top": dict(
                sorted(
                    per_task_failure_combo_breakdown.get(task, {}).items(),
                    key=lambda kv: (-kv[1], kv[0]),
                )[:10]
            ),
        }
        for task, stats in sorted(per_task.items())
    }
    overall_slot_span_acc = {
        task: {
            key.replace("_ok", ""): slot_span_stats.get(task, {}).get(key, 0)
            / max(slot_span_stats.get(task, {}).get(key.replace("_ok", "_total"), 0), 1)
            for key in sorted(slot_span_stats.get(task, {}))
            if key.endswith("_ok")
        }
        for task in sorted(slot_span_stats)
    }
    return {
        "global": {
            "use_acc": use_ok / max(use_total, 1),
            "span_top1_acc": span_ok / max(span_total, 1),
            "span_top1_acc_nonnull": span_ok / max(span_total, 1),
            "text_faithful_acc": text_ok / max(text_total, 1),
            "commit_acc": commit_ok / max(commit_total, 1),
            "edge_precision": edge_precision,
            "edge_recall": edge_recall,
            "edge_f1": edge_f1,
            "edge_relation_acc_on_gold": edge_rel_ok / max(edge_rel_total, 1),
            "attachment_precision": attach_precision,
            "attachment_recall": attach_recall,
            "attachment_f1": attach_f1,
            "attachment_relation_acc_on_gold": attach_rel_ok / max(attach_rel_total, 1),
            "cover_precision": cover_precision,
            "cover_recall": cover_recall,
            "cover_f1": cover_f1,
            "row_complete_rate": row_ok / max(row_total, 1),
            "text_faithful_row_complete_rate": text_row_ok / max(row_total, 1),
        },
        "per_task": per_task_rates,
        "failure_breakdown": dict(sorted(failure_breakdown.items(), key=lambda kv: (-kv[1], kv[0]))),
        "failure_combo_breakdown": dict(sorted(failure_combo_breakdown.items(), key=lambda kv: (-kv[1], kv[0]))),
        "isolated_failure_breakdown": dict(sorted(isolated_failure_breakdown.items(), key=lambda kv: (-kv[1], kv[0]))),
        "slot_span_acc_by_task": overall_slot_span_acc,
        "samples": {
            "successes": success_samples,
            "failures": failure_samples,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate unified proposer-aligner checkpoint")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val-jsonl", default="artifacts/proposer_v1_20260512/proposer_val.jsonl")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--cand-emb-cache", default=None)
    ap.add_argument("--mem-emb-cache", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--sample-limit", type=int, default=5)
    ap.add_argument("--max-val-rows", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    ckpt_args = checkpoint.get("args", {}) or {}

    val_rows = read_jsonl(args.val_jsonl)
    if args.max_val_rows > 0:
        val_rows = val_rows[: args.max_val_rows]

    val_ds = UnifiedDataset(
        val_rows,
        hash_dim=int(ckpt_args.get("hash_dim", 512)),
        cand_emb_cache=args.cand_emb_cache or ckpt_args.get("cand_emb_cache", "artifacts/spec_emb_cache/pred_v1_fix8_minilm_cand.npz"),
        mem_emb_cache=args.mem_emb_cache or ckpt_args.get("mem_emb_cache", "artifacts/spec_emb_cache/pred_v1_fix8_minilm_mem.npz"),
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    model = build_model_from_checkpoint(checkpoint, val_rows, val_ds, device)

    report = evaluate(model, val_loader, device, sample_limit=args.sample_limit)
    recorded = checkpoint.get("val", {}) or {}
    current = report["global"]
    parity_keys = [
        "use_acc",
        "span_top1_acc",
        "text_faithful_acc",
        "commit_acc",
        "edge_f1",
        "attachment_f1",
        "cover_f1",
        "row_complete_rate",
        "text_faithful_row_complete_rate",
    ]
    parity = {
        key: {
            "checkpoint": float(recorded.get(key, 0.0)),
            "external_eval": float(current.get(key, 0.0)),
            "delta": float(current.get(key, 0.0)) - float(recorded.get(key, 0.0)),
        }
        for key in parity_keys
    }
    report["checkpoint_path"] = str(args.checkpoint)
    report["checkpoint_epoch"] = checkpoint.get("epoch")
    report["checkpoint_score"] = checkpoint.get("score")
    report["trainer_val_metrics"] = recorded
    report["parity_check"] = parity

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["global"], ensure_ascii=False))
    print(json.dumps({"parity_check": parity}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
