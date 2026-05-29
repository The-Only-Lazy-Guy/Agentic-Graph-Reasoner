from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

from graph_core import canonical_relation
from pred_model import (
    MEM_LINK_KIND_TO_ID,
    PredAlignNet,
    infer_cand_emb_dim_from_state,
    infer_edge_rel_pair_feat_dim_from_state,
    infer_mem_emb_dim_from_state,
    infer_spec_emb_dim_from_state,
)
from proposer_model import ProposerNet, infer_proposer_arch_from_state
from synthesize_node_text import apply_template_synthesis, normalize_text
from train_pred_v1 import (
    PredDataset,
    collate as aligner_collate,
    decode_edge_predictions,
    decode_mem_kind_predictions,
    decode_span_predictions,
    goal_commit_family,
    read_jsonl,
    to_device as aligner_to_device,
)
from train_proposer_v1 import (
    IGNORE,
    ProposerDataset,
    collate,
    compute_metrics,
    predicted_slots_for_row,
    to_device,
)


SYNTHESIS_TASKS = {"mixed_add_link", "multi_region_attach"}


def collect_proposer_outputs(
    model: ProposerNet,
    rows: Sequence[Mapping[str, Any]],
    device: torch.device,
    *,
    hash_dim: int,
    batch_size: int,
    cand_emb_cache: str | None,
    mem_emb_cache: str | None,
) -> List[Dict[str, Any]]:
    ds = ProposerDataset(
        rows,
        hash_dim=hash_dim,
        cand_emb_cache=cand_emb_cache,
        mem_emb_cache=mem_emb_cache,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    records: List[Dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for batch, batch_rows in loader:
            batch = to_device(batch, device)
            out = model.predict(batch)
            use_pred = out["use_pred"]
            span_pred = out["span_pred"]
            bridge_pred = out["bridge_pred"]
            topk_k = min(5, out["span_logits"].size(-1))
            topk = out["span_logits"].topk(k=topk_k, dim=-1).indices.cpu()
            use_prob = torch.sigmoid(out["use_logits"]).cpu()
            bridge_prob = torch.sigmoid(out["type_logits"]).cpu()
            for b, row in enumerate(batch_rows):
                records.append(
                    {
                        "row": row,
                        "gold_use": batch.y_use[b].cpu(),
                        "gold_span": batch.y_span[b].cpu(),
                        "gold_bridge": batch.y_is_bridge[b].cpu(),
                        "slot_mask": batch.y_slot_mask[b].cpu(),
                        "use_pred": use_pred[b].cpu(),
                        "span_pred": span_pred[b].cpu(),
                        "bridge_pred": bridge_pred[b].cpu(),
                        "use_prob": use_prob[b],
                        "bridge_prob": bridge_prob[b],
                        "span_topk": topk[b],
                    }
                )
    return records


def per_task_metrics(
    model: ProposerNet,
    rows: List[Mapping[str, Any]],
    device: torch.device,
    *,
    hash_dim: int,
    batch_size: int,
    cand_emb_cache: str | None,
    mem_emb_cache: str | None,
) -> Dict[str, Any]:
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("task_type", "unknown"))].append(row)

    out: Dict[str, Any] = {}
    for task_type in sorted(groups):
        ds = ProposerDataset(
            groups[task_type],
            hash_dim=hash_dim,
            cand_emb_cache=cand_emb_cache,
            mem_emb_cache=mem_emb_cache,
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
        out[task_type] = {"n": len(groups[task_type]), **compute_metrics(model, loader, device)}
    return out


def shared_anchor_analysis(rows: List[Mapping[str, Any]]) -> Dict[str, Any]:
    total = 0
    by_task: Dict[str, int] = defaultdict(int)
    for row in rows:
        meta = row.get("_slot_meta", {}) or {}
        if bool(meta.get("shared_anchor_start")):
            total += 1
            by_task[str(row.get("task_type", "unknown"))] += 1
    return {
        "shared_anchor_rows": total,
        "shared_anchor_rate": total / max(len(rows), 1),
        "by_task": dict(sorted(by_task.items())),
    }


def proposer_diagnostics(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    span_top3_total = span_top3_ok = 0
    span_top5_total = span_top5_ok = 0

    per_slot: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    shared_bucket: Dict[str, Dict[str, int]] = {
        "shared_anchor": defaultdict(int),
        "single_anchor": defaultdict(int),
    }
    per_task: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for rec in records:
        row = rec["row"]
        task_type = str(row.get("task_type", "unknown"))
        shared_key = "shared_anchor" if bool((row.get("_slot_meta", {}) or {}).get("shared_anchor_start")) else "single_anchor"
        gold_use = rec["gold_use"]
        gold_span = rec["gold_span"]
        gold_bridge = rec["gold_bridge"]
        slot_mask = rec["slot_mask"]
        use_pred = rec["use_pred"]
        span_pred = rec["span_pred"]
        bridge_pred = rec["bridge_pred"]
        span_topk = rec["span_topk"]

        K = slot_mask.numel()
        for k in range(K):
            if not bool(slot_mask[k].item()):
                continue
            per_slot[k]["slots"] += 1
            use_match = bool((use_pred[k] == (gold_use[k] > 0.5)).item())
            per_slot[k]["use_ok"] += int(use_match)
            if bool((gold_use[k] > 0.5).item()):
                per_slot[k]["used_gold"] += 1
                span_match = bool((span_pred[k] == gold_span[k]).item())
                bridge_match = bool((bridge_pred[k] == (gold_bridge[k] > 0.5)).item())
                per_slot[k]["span_ok"] += int(span_match)
                per_slot[k]["bridge_ok"] += int(bridge_match)

                in_top3 = int(gold_span[k].item()) in span_topk[k, : min(3, span_topk.size(1))].tolist()
                in_top5 = int(gold_span[k].item()) in span_topk[k, : min(5, span_topk.size(1))].tolist()
                span_top3_total += 1
                span_top3_ok += int(in_top3)
                span_top5_total += 1
                span_top5_ok += int(in_top5)

                shared_bucket[shared_key]["used_gold"] += 1
                shared_bucket[shared_key]["span_ok"] += int(span_match)
                per_task[task_type]["used_gold"] += 1
                per_task[task_type]["span_ok"] += int(span_match)

    per_slot_out: Dict[str, Any] = {}
    for k in sorted(per_slot):
        d = per_slot[k]
        per_slot_out[str(k)] = {
            "slots": d["slots"],
            "use_acc": d["use_ok"] / max(d["slots"], 1),
            "span_acc_on_used": d["span_ok"] / max(d["used_gold"], 1),
            "bridge_acc_on_used": d["bridge_ok"] / max(d["used_gold"], 1),
        }

    shared_out: Dict[str, Any] = {}
    for key, d in shared_bucket.items():
        shared_out[key] = {
            "used_gold": d["used_gold"],
            "span_acc_on_used": d["span_ok"] / max(d["used_gold"], 1),
        }

    per_task_out: Dict[str, Any] = {}
    for task_type, d in sorted(per_task.items()):
        per_task_out[task_type] = {
            "used_gold": d["used_gold"],
            "span_acc_on_used": d["span_ok"] / max(d["used_gold"], 1),
        }

    return {
        "span_top3_recall_on_used": span_top3_ok / max(span_top3_total, 1),
        "span_top5_recall_on_used": span_top5_ok / max(span_top5_total, 1),
        "per_slot": per_slot_out,
        "shared_anchor_vs_single": shared_out,
        "per_task_span": per_task_out,
    }


def reconcile_slot_names(
    row: Mapping[str, Any],
    predicted_slots: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    span_to_gold_names: Dict[str, List[str]] = defaultdict(list)
    for slot in row.get("target_slots", []) or []:
        if not bool(slot.get("use")):
            continue
        span_id = slot.get("span_id")
        if span_id is None:
            continue
        span_to_gold_names[str(span_id)].append(str(slot.get("session_name", "")))

    used_gold_names: set[str] = set()
    reconciled: List[Dict[str, Any]] = []
    for k, slot in enumerate(predicted_slots):
        if not bool(slot.get("use")):
            continue
        span_id = slot.get("span_id")
        candidates = span_to_gold_names.get(str(span_id), [])
        assigned = None
        for name in candidates:
            if name not in used_gold_names:
                assigned = name
                break
        if assigned is None:
            assigned = f"orphan_{k}"
        used_gold_names.add(assigned)
        reconciled.append(
            {
                **slot,
                "session_name": assigned,
                "name": assigned,
            }
        )
    return reconciled


def build_true_aligner_input_row(
    row: Mapping[str, Any],
    predicted_slots: Sequence[Mapping[str, Any]],
    *,
    apply_synthesis: bool = False,
    graph_cache: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    reconciled = reconcile_slot_names(row, predicted_slots)
    if apply_synthesis:
        reconciled = apply_template_synthesis(row, reconciled, graph_cache=graph_cache)
    session_nodes = [
        {
            "name": str(slot.get("session_name", f"s{i}")),
            "span_text": str(slot.get("span_text", "")),
            "node_type": "bridge" if str(slot.get("node_type", "concept")) == "bridge" else "concept",
        }
        for i, slot in enumerate(reconciled)
    ]
    span_oracle = [
        {
            "session_name": str(slot.get("session_name", f"s{i}")),
            "spec_text": str(slot.get("span_text", "")),
            "node_type": "bridge" if str(slot.get("node_type", "concept")) == "bridge" else "concept",
            "best_span_id": slot.get("span_id"),
            "best_score": float(slot.get("oracle_best_score", 0.0)),
            "span_scores": [],
        }
        for i, slot in enumerate(reconciled)
    ]
    return {
        "id": row.get("id", ""),
        "task_type": row.get("task_type", ""),
        "graph_path": row.get("graph_path", ""),
        "signal": row.get("signal", ""),
        "initial_memory_node_ids": list(row.get("initial_memory_node_ids", []) or []),
        "spans": list(row.get("spans", []) or []),
        "goal": {
            "session_nodes": session_nodes,
            "session_edges": [],
            "covered_mappings": [],
            "memory_attachments": [],
            "final_commits": [],
        },
        "span_oracle": span_oracle,
        "_oracle_goal": dict((row.get("_oracle_goal", {}) or {})),
    }


def derive_gold_targets(row: Mapping[str, Any]) -> Dict[str, Any]:
    goal = row.get("_oracle_goal", {}) or {}
    gold_nodes = goal.get("session_nodes", []) or []
    gold_names = [str(node.get("name", "")) for node in gold_nodes]
    gold_text_by_name = {
        str(node.get("name", "")): str(node.get("span_text", ""))
        for node in gold_nodes
    }
    gold_span_by_name = {
        str(slot.get("session_name", "")): str(slot.get("span_id", ""))
        for slot in row.get("target_slots", []) or []
        if bool(slot.get("use")) and slot.get("span_id") is not None
    }
    gold_edges = {
        (
            str(edge.get("src", "")),
            str(edge.get("dst", "")),
            canonical_relation(str(edge.get("relation", "related"))),
        )
        for edge in goal.get("session_edges", []) or []
    }
    gold_attach = {
        (
            str(att.get("session", "")),
            str(att.get("memory_id", "")),
            canonical_relation(str(att.get("relation", "related"))),
        )
        for att in goal.get("memory_attachments", []) or []
    }

    session_text_by_name = {str(node.get("name", "")): str(node.get("span_text", "")) for node in gold_nodes}
    gold_cover = set()
    covered_mappings = goal.get("covered_mappings", []) or []
    for cov in covered_mappings:
        span_text = str(cov.get("span_text", ""))
        memory_id = str(cov.get("memory_id", ""))
        for name, text in session_text_by_name.items():
            if text == span_text:
                gold_cover.add((name, memory_id))
                break

    return {
        "gold_names": set(gold_names),
        "gold_text_by_name": gold_text_by_name,
        "gold_span_by_name": gold_span_by_name,
        "gold_edges": gold_edges,
        "gold_attach": gold_attach,
        "gold_cover": gold_cover,
        "gold_commit_family": goal_commit_family(goal),
    }


def aggregate_binary_metrics(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def summarize_true_records(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    span_total = span_ok = 0
    text_total = text_ok = 0
    commit_total = commit_ok = 0
    edge_tp = edge_fp = edge_fn = 0
    edge_rel_total = edge_rel_ok = 0
    attach_tp = attach_fp = attach_fn = 0
    attach_rel_total = attach_rel_ok = 0
    cover_tp = cover_fp = cover_fn = 0
    row_total = row_ok = 0
    text_row_ok = 0

    for rec in records:
        row_total += 1
        row_ok += int(bool(rec["row_complete"]))
        text_row_ok += int(bool(rec["text_faithful_row_complete"]))
        commit_total += 1
        commit_ok += int(bool(rec["commit_ok"]))
        span_total += int(rec["span_total"])
        span_ok += int(rec["span_ok"])
        text_total += int(rec["text_total"])
        text_ok += int(rec["text_ok"])
        edge_tp += int(rec["edge_tp"])
        edge_fp += int(rec["edge_fp"])
        edge_fn += int(rec["edge_fn"])
        edge_rel_total += int(rec["edge_rel_total"])
        edge_rel_ok += int(rec["edge_rel_ok"])
        attach_tp += int(rec["attach_tp"])
        attach_fp += int(rec["attach_fp"])
        attach_fn += int(rec["attach_fn"])
        attach_rel_total += int(rec["attach_rel_total"])
        attach_rel_ok += int(rec["attach_rel_ok"])
        cover_tp += int(rec["cover_tp"])
        cover_fp += int(rec["cover_fp"])
        cover_fn += int(rec["cover_fn"])

    edge_precision, edge_recall, edge_f1 = aggregate_binary_metrics(edge_tp, edge_fp, edge_fn)
    attach_precision, attach_recall, attach_f1 = aggregate_binary_metrics(attach_tp, attach_fp, attach_fn)
    cover_precision, cover_recall, cover_f1 = aggregate_binary_metrics(cover_tp, cover_fp, cover_fn)
    return {
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
    }


def run_true_end_to_end_eval(
    model: ProposerNet,
    rows: Sequence[Mapping[str, Any]],
    device: torch.device,
    *,
    batch_size: int,
    hash_dim: int,
    cand_emb_cache: str | None,
    mem_emb_cache: str | None,
    aligner_ckpt_path: str,
    aligner_spec_emb_cache: str | None,
    aligner_cand_emb_cache: str | None,
    aligner_mem_emb_cache: str | None,
    apply_synthesis: bool = False,
) -> Dict[str, Any]:
    proposer_records = collect_proposer_outputs(
        model,
        rows,
        device,
        hash_dim=hash_dim,
        batch_size=batch_size,
        cand_emb_cache=cand_emb_cache,
        mem_emb_cache=mem_emb_cache,
    )
    pred_rows = []
    gold_targets = []
    graph_cache: Dict[str, Any] = {}
    for rec in proposer_records:
        row = rec["row"]
        pred_slots = predicted_slots_for_row(row, rec["use_pred"], rec["span_pred"], rec["bridge_pred"])
        pred_rows.append(
            build_true_aligner_input_row(
                row,
                pred_slots,
                apply_synthesis=apply_synthesis,
                graph_cache=graph_cache,
            )
        )
        gold_targets.append(derive_gold_targets(row))

    aligner_device = device
    ckpt = torch.load(aligner_ckpt_path, map_location=aligner_device)
    aligner_args = dict(ckpt.get("args", {}))
    spec_emb_dim = infer_spec_emb_dim_from_state(ckpt["model"])
    cand_emb_dim = infer_cand_emb_dim_from_state(ckpt["model"])
    mem_emb_dim = infer_mem_emb_dim_from_state(ckpt["model"])
    ds = PredDataset(
        pred_rows,
        hash_dim=int(aligner_args.get("hash_dim", 512)),
        spec_emb_cache=aligner_spec_emb_cache,
        spec_emb_dim_override=spec_emb_dim,
        cand_emb_cache=aligner_cand_emb_cache,
        cand_emb_dim_override=cand_emb_dim,
        mem_emb_cache=aligner_mem_emb_cache,
        mem_emb_dim_override=mem_emb_dim,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=aligner_collate)
    aligner_model = PredAlignNet(
        hash_dim=int(aligner_args.get("hash_dim", 512)),
        hidden_dim=int(aligner_args.get("hidden_dim", 256)),
        edge_rel_pair_feat_dim=infer_edge_rel_pair_feat_dim_from_state(
            ckpt["model"], int(aligner_args.get("hidden_dim", 256))
        ),
        spec_emb_dim=spec_emb_dim,
        cand_emb_dim=cand_emb_dim,
        mem_emb_dim=mem_emb_dim,
    ).to(aligner_device)
    aligner_model.load_state_dict(ckpt["model"])

    row_records: List[Dict[str, Any]] = []
    aligner_model.eval()
    offset = 0
    with torch.no_grad():
        for batch in loader:
            batch = aligner_to_device(batch, aligner_device)
            out = aligner_model(batch)
            span_pred = decode_span_predictions(out["span_logits"], batch.spec_mask, batch.cand_mask).cpu()
            commit_pred = out["commit_logits"].argmax(dim=-1).cpu()
            edge_exist_pred = decode_edge_predictions(out["edge_exist_logits"], batch.edge_mask).cpu()
            edge_rel_pred = out["edge_rel_logits"].argmax(dim=-1).cpu()
            mem_kind_pred = decode_mem_kind_predictions(out["mem_kind_logits"], batch.mem_mask).cpu()
            mem_rel_pred = out["mem_rel_logits"].argmax(dim=-1).cpu()

            B = batch.signal_bow.size(0)
            for b in range(B):
                pred_row = pred_rows[offset + b]
                gold = gold_targets[offset + b]
                task_type = str(pred_row.get("task_type", "unknown"))
                predicted_nodes = pred_row.get("goal", {}).get("session_nodes", []) or []
                pred_names = [str(node.get("name", "")) for node in predicted_nodes]
                pred_name_set = set(pred_names)
                name_to_idx = {name: i for i, name in enumerate(pred_names)}
                gold_names = set(gold["gold_names"])
                pred_text_by_name = {
                    str(node.get("name", "")): str(node.get("span_text", ""))
                    for node in predicted_nodes
                }

                span_ok_count = 0
                span_total = len(gold_names)
                text_ok_count = 0
                for gold_name in gold_names:
                    idx = name_to_idx.get(gold_name)
                    if idx is None:
                        continue
                    pred_span_idx = int(span_pred[b, idx].item())
                    pred_span_id = None
                    spans = pred_row.get("spans", []) or []
                    if 0 <= pred_span_idx < len(spans):
                        pred_span_id = str(spans[pred_span_idx].get("id", ""))
                    if pred_span_id == gold["gold_span_by_name"].get(gold_name):
                        span_ok_count += 1
                    if normalize_text(pred_text_by_name.get(gold_name, "")) == normalize_text(gold["gold_text_by_name"].get(gold_name, "")):
                        text_ok_count += 1

                predicted_edges = set()
                edge_fp = 0
                for i, src_name in enumerate(pred_names):
                    for j, dst_name in enumerate(pred_names):
                        if i == j:
                            continue
                        if bool(edge_exist_pred[b, i, j].item()):
                            rel_id = int(edge_rel_pred[b, i, j].item())
                            rel = __import__("pred_model").REL_WITH_NONE[rel_id]
                            triple = (src_name, dst_name, canonical_relation(rel))
                            predicted_edges.add(triple)
                edge_tp_set = predicted_edges & gold["gold_edges"]
                edge_fp = len(predicted_edges - gold["gold_edges"])
                edge_fn = len(gold["gold_edges"] - predicted_edges)
                edge_rel_total = len(gold["gold_edges"])
                edge_rel_ok = len(edge_tp_set)

                memory_ids = list(pred_row.get("initial_memory_node_ids", []) or [])
                predicted_attach = set()
                predicted_cover = set()
                for i, src_name in enumerate(pred_names):
                    for m_idx, mem_id in enumerate(memory_ids):
                        kind_id = int(mem_kind_pred[b, i, m_idx].item())
                        if kind_id == MEM_LINK_KIND_TO_ID["attach"]:
                            rel_id = int(mem_rel_pred[b, i, m_idx].item())
                            rel = __import__("pred_model").REL_WITH_NONE[rel_id]
                            predicted_attach.add((src_name, str(mem_id), canonical_relation(rel)))
                        elif kind_id == MEM_LINK_KIND_TO_ID["cover"]:
                            predicted_cover.add((src_name, str(mem_id)))

                attach_tp_set = predicted_attach & gold["gold_attach"]
                attach_fp = len(predicted_attach - gold["gold_attach"])
                attach_fn = len(gold["gold_attach"] - predicted_attach)
                attach_rel_total = len(gold["gold_attach"])
                attach_rel_ok = len(attach_tp_set)

                cover_tp_set = predicted_cover & gold["gold_cover"]
                cover_fp = len(predicted_cover - gold["gold_cover"])
                cover_fn = len(gold["gold_cover"] - predicted_cover)

                commit_name = __import__("pred_model").COMMIT_FAMILIES[int(commit_pred[b].item())]
                commit_ok = commit_name == gold["gold_commit_family"]

                row_complete = (
                    span_ok_count == span_total
                    and commit_ok
                    and edge_fp == 0
                    and edge_fn == 0
                    and attach_fp == 0
                    and attach_fn == 0
                    and cover_fp == 0
                    and cover_fn == 0
                    and pred_name_set == gold_names
                )
                text_faithful_row_complete = row_complete and text_ok_count == span_total

                row_records.append(
                    {
                        "task_type": task_type,
                        "needs_synthesis": task_type in SYNTHESIS_TASKS,
                        "span_total": span_total,
                        "span_ok": span_ok_count,
                        "text_total": span_total,
                        "text_ok": text_ok_count,
                        "commit_ok": commit_ok,
                        "edge_tp": len(edge_tp_set),
                        "edge_fp": edge_fp,
                        "edge_fn": edge_fn,
                        "edge_rel_total": edge_rel_total,
                        "edge_rel_ok": edge_rel_ok,
                        "attach_tp": len(attach_tp_set),
                        "attach_fp": attach_fp,
                        "attach_fn": attach_fn,
                        "attach_rel_total": attach_rel_total,
                        "attach_rel_ok": attach_rel_ok,
                        "cover_tp": len(cover_tp_set),
                        "cover_fp": cover_fp,
                        "cover_fn": cover_fn,
                        "row_complete": row_complete,
                        "text_faithful_row_complete": text_faithful_row_complete,
                    }
                )
            offset += B

    overall = summarize_true_records(row_records)
    by_synthesis = {
        "needs_synthesis_false": summarize_true_records([r for r in row_records if not r["needs_synthesis"]]),
        "needs_synthesis_true": summarize_true_records([r for r in row_records if r["needs_synthesis"]]),
    }
    return {
        "overall": overall,
        "by_synthesis": by_synthesis,
        "n_rows": len(row_records),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--jsonl", default="artifacts/proposer_v1_20260512/proposer_val.jsonl")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--cand-emb-cache", default="artifacts/spec_emb_cache/pred_v1_fix8_minilm_cand.npz")
    ap.add_argument("--mem-emb-cache", default="artifacts/spec_emb_cache/pred_v1_fix8_minilm_mem.npz")
    ap.add_argument("--aligner-checkpoint", default="out_pred_v1_train_20260512_fix21_mememb_rel_freeze/best_pred_v1.pt")
    ap.add_argument("--aligner-spec-emb-cache", default="artifacts/spec_emb_cache/pred_v1_fix8_minilm_spec.npz")
    ap.add_argument("--aligner-cand-emb-cache", default="artifacts/spec_emb_cache/pred_v1_fix8_minilm_cand.npz")
    ap.add_argument("--aligner-mem-emb-cache", default="artifacts/spec_emb_cache/pred_v1_fix8_minilm_mem.npz")
    ap.add_argument("--apply-template-synthesis", action="store_true")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("args", {})
    rows = read_jsonl(args.jsonl)
    proposer_arch = infer_proposer_arch_from_state(
        ckpt["model"],
        int(cfg.get("hidden_dim", 256)),
    )

    ds = ProposerDataset(
        rows,
        hash_dim=int(cfg.get("hash_dim", 512)),
        cand_emb_cache=args.cand_emb_cache,
        mem_emb_cache=args.mem_emb_cache,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = ProposerNet(
        hash_dim=int(cfg.get("hash_dim", 512)),
        hidden_dim=int(cfg.get("hidden_dim", 256)),
        k_max=max(len((row.get("target_slots", []) or [])) for row in rows),
        cand_emb_dim=ds.cand_emb_dim,
        mem_emb_dim=ds.mem_emb_dim,
        cand_pair_feat_dim=int(proposer_arch.get("cand_pair_feat_dim", 3)),
        use_ar_span_features=bool(proposer_arch.get("use_ar_span_features", False)),
        slot_attention_mode=str(proposer_arch.get("slot_attention_mode", "none")),
        span_scorer_mode=str(proposer_arch.get("span_scorer_mode", "concat_mlp")),
    ).to(device)
    model.load_state_dict(ckpt["model"])

    global_metrics = compute_metrics(model, loader, device)
    per_task = per_task_metrics(
        model,
        rows,
        device,
        hash_dim=int(cfg.get("hash_dim", 512)),
        batch_size=args.batch_size,
        cand_emb_cache=args.cand_emb_cache,
        mem_emb_cache=args.mem_emb_cache,
    )
    shared_anchor = shared_anchor_analysis(rows)
    collected = collect_proposer_outputs(
        model,
        rows,
        device,
        hash_dim=int(cfg.get("hash_dim", 512)),
        batch_size=args.batch_size,
        cand_emb_cache=args.cand_emb_cache,
        mem_emb_cache=args.mem_emb_cache,
    )
    diagnostics = proposer_diagnostics(collected)

    from train_proposer_v1 import run_end_to_end_eval

    leaky_e2e = run_end_to_end_eval(
        model,
        rows,
        device,
        batch_size=args.batch_size,
        hash_dim=int(cfg.get("hash_dim", 512)),
        cand_emb_cache=args.cand_emb_cache,
        cand_emb_dim=ds.cand_emb_dim,
        mem_emb_cache=args.mem_emb_cache,
        mem_emb_dim=ds.mem_emb_dim,
        aligner_ckpt_path=args.aligner_checkpoint,
        aligner_spec_emb_cache=args.aligner_spec_emb_cache,
        aligner_cand_emb_cache=args.aligner_cand_emb_cache,
        aligner_mem_emb_cache=args.aligner_mem_emb_cache,
    )
    true_e2e = run_true_end_to_end_eval(
        model,
        rows,
        device,
        batch_size=args.batch_size,
        hash_dim=int(cfg.get("hash_dim", 512)),
        cand_emb_cache=args.cand_emb_cache,
        mem_emb_cache=args.mem_emb_cache,
        aligner_ckpt_path=args.aligner_checkpoint,
        aligner_spec_emb_cache=args.aligner_spec_emb_cache,
        aligner_cand_emb_cache=args.aligner_cand_emb_cache,
        aligner_mem_emb_cache=args.aligner_mem_emb_cache,
        apply_synthesis=args.apply_template_synthesis,
    )

    report = {
        "checkpoint": args.checkpoint,
        "jsonl": args.jsonl,
        "global": global_metrics,
        "per_task": per_task,
        "shared_anchor": shared_anchor,
        "diagnostics": diagnostics,
        "end_to_end_leaky": leaky_e2e,
        "end_to_end_true": true_e2e,
        "template_synthesis_applied": bool(args.apply_template_synthesis),
    }
    print("=== proposer metrics ===")
    print(json.dumps(global_metrics, indent=2, ensure_ascii=False))
    print("\n=== proposer diagnostics ===")
    print(json.dumps(diagnostics, indent=2, ensure_ascii=False))
    print("\n=== end-to-end aligner metrics (leaky) ===")
    print(json.dumps(leaky_e2e, indent=2, ensure_ascii=False))
    print("\n=== end-to-end aligner metrics (true) ===")
    print(json.dumps(true_e2e, indent=2, ensure_ascii=False))
    print("\n=== shared-anchor analysis ===")
    print(json.dumps(shared_anchor, indent=2, ensure_ascii=False))

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nreport written to {args.out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
