from __future__ import annotations

import argparse
import json
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
from eval_unified_v1 import build_model_from_checkpoint, row_debug_payload
from traverse_threshold_draft_edit import execute_goal_spec


def decode_unified_prediction_to_goal(
    row: Mapping[str, Any],
    pred_nodes: List[Dict[str, Any]],
    predicted_edges: set,
    predicted_attach: set,
    predicted_cover: set,
    commit_name: str,
) -> Dict[str, Any]:
    """Convert decoded structures into a goal spec matching the environment schema."""
    session_edges = []
    for src, dst, rel in predicted_edges:
        session_edges.append({"src": src, "dst": dst, "relation": rel})
        
    memory_attachments = []
    for session, mem_id, rel in predicted_attach:
        memory_attachments.append({"session": session, "memory_id": mem_id, "relation": rel})
        
    covered_mappings = []
    for session, mem_id in predicted_cover:
        span_text = ""
        for n in pred_nodes:
            if n.get("name") == session:
                span_text = n.get("span_text", "")
                break
        covered_mappings.append({"session": session, "memory_id": mem_id, "span_text": span_text})
        
    final_commits = [{"action": "commit", "family": commit_name}]

    return {
        "session_nodes": pred_nodes,
        "session_edges": session_edges,
        "memory_attachments": memory_attachments,
        "covered_mappings": covered_mappings,
        "final_commits": final_commits,
    }


def evaluate_roundtrip(
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
            edge_logits_for_rel = out["verifier_logits"] if model.use_verifier else out["edge_rel_logits"]
            edge_rel_pred = edge_logits_for_rel.argmax(dim=-1)
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

                gold = derive_gold_targets(row)
                graph = loader.dataset.graph(str(row.get("graph_path", "")))  # type: ignore[attr-defined]
                memory_ids = build_candidate_memory_ids(row, graph)

                # 1. Decode predictions normally
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
                _eval_slots = _ensure_target_slots(row)
                pred_name_to_slot = {
                    str(_eval_slots[k].get("session_name", f"slot_{k}")): k
                    for k in range(len(_eval_slots))
                }

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
                            if rel != "none":
                                predicted_edges.add((src_name, dst_name, rel))

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

                commit_name = COMMIT_FAMILIES[int(commit_pred[b].item())]

                # 2. Convert to Goal Spec
                predicted_goal = decode_unified_prediction_to_goal(
                    row, pred_nodes, predicted_edges, predicted_attach, predicted_cover, commit_name
                )
                
                # 3. Trajectory Translation
                predicted_row = dict(row)
                predicted_row["goal"] = predicted_goal
                executed_draft, executed_trace, executed_postprocess, executed_synth = execute_goal_spec(predicted_row, graph)

                # 4. Extract structures from the executed draft state
                gold_names = set(gold["gold_names"])
                
                exec_names = {str(n.goal_session_name) for n in executed_draft.session_nodes if n.goal_session_name}
                exec_name_set = set(exec_names)
                exec_text_by_name = {str(n.goal_session_name): str(n.text) for n in executed_draft.session_nodes if n.goal_session_name}
                exec_span_by_name = {str(n.goal_session_name): str(n.span_id) for n in executed_draft.session_nodes if n.goal_session_name and n.span_id}
                
                # Use matches are from the actual model outputs
                use_gold = batch.y_use[b] > 0.5
                use_match = (use_pred[b] == use_gold) & batch.slot_mask[b]
                use_total += int(batch.slot_mask[b].sum().item())
                use_ok += int(use_match.sum().item())

                span_ok_count = 0
                span_total_row = len(gold_names)
                text_ok_count = 0
                for name in gold_names:
                    pred_span_id = exec_span_by_name.get(name)
                    if pred_span_id and pred_span_id == gold["gold_span_by_name"].get(name):
                        span_ok_count += 1
                    if exec_text_by_name.get(name, "") == gold["gold_text_by_name"].get(name, ""):
                        text_ok_count += 1

                span_total += span_total_row
                span_ok += span_ok_count
                text_total += span_total_row
                text_ok += text_ok_count

                # Extract edges and attachments from draft, mapping back via goal_session_name
                sid_to_name = {n.id: n.goal_session_name for n in executed_draft.session_nodes}
                
                exec_edges = set()
                for e in executed_draft.session_edges:
                    src_name = sid_to_name.get(e.src)
                    dst_name = sid_to_name.get(e.dst)
                    if src_name and dst_name:
                        exec_edges.add((src_name, dst_name, e.relation))

                edge_tp_set = exec_edges & gold["gold_edges"]
                edge_fp += len(exec_edges - gold["gold_edges"])
                edge_fn += len(gold["gold_edges"] - exec_edges)
                edge_tp += len(edge_tp_set)
                edge_rel_total += len(gold["gold_edges"])
                edge_rel_ok += len(edge_tp_set)

                exec_attach = set()
                exec_cover = set()
                for a in executed_draft.attachments:
                    name = sid_to_name.get(a.session_id)
                    if not name:
                        continue
                    if a.kind == "attach":
                        exec_attach.add((name, str(a.memory_id), a.relation))
                    elif a.kind == "cover":
                        exec_cover.add((name, str(a.memory_id)))

                attach_tp_set = exec_attach & gold["gold_attach"]
                attach_tp += len(attach_tp_set)
                attach_fp += len(exec_attach - gold["gold_attach"])
                attach_fn += len(gold["gold_attach"] - exec_attach)
                attach_rel_total += len(gold["gold_attach"])
                attach_rel_ok += len(attach_tp_set)

                cover_tp_set = exec_cover & gold["gold_cover"]
                cover_tp += len(cover_tp_set)
                cover_fp += len(exec_cover - gold["gold_cover"])
                cover_fn += len(gold["gold_cover"] - exec_cover)

                commit_ok_row = commit_name == gold["gold_commit_family"]
                commit_total += 1
                commit_ok += int(commit_ok_row)

                row_components = {
                    "node_set": exec_name_set == gold_names,
                    "use": bool(use_match[batch.slot_mask[b]].all().item()),
                    "span": span_ok_count == span_total_row,
                    "text": text_ok_count == span_total_row,
                    "commit": commit_ok_row,
                    "edge": len(exec_edges - gold["gold_edges"]) == 0 and len(gold["gold_edges"] - exec_edges) == 0,
                    "attach": len(exec_attach - gold["gold_attach"]) == 0 and len(gold["gold_attach"] - exec_attach) == 0,
                    "cover": len(exec_cover - gold["gold_cover"]) == 0 and len(gold["gold_cover"] - exec_cover) == 0,
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
        }
        for task, stats in sorted(per_task.items())
    }

    return {
        "global": {
            "use_acc": use_ok / max(use_total, 1),
            "span_top1_acc": span_ok / max(span_total, 1),
            "text_faithful_acc": text_ok / max(text_total, 1),
            "commit_acc": commit_ok / max(commit_total, 1),
            "edge_f1": edge_f1,
            "attachment_f1": attach_f1,
            "cover_f1": cover_f1,
            "row_complete_rate": row_ok / max(row_total, 1),
            "text_faithful_row_complete_rate": text_row_ok / max(row_total, 1),
        },
        "per_task": per_task_rates,
        "samples": {
            "successes": success_samples,
            "failures": failure_samples,
        },
    }

def predict_unified_goals(
    checkpoint_path: str,
    rows: List[Dict[str, Any]],
    cand_emb_cache: str,
    mem_emb_cache: str,
    device: torch.device,
) -> Dict[str, Dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ckpt_args = checkpoint.get("args", {}) or {}
    
    ds = UnifiedDataset(
        rows,
        hash_dim=int(ckpt_args.get("hash_dim", 512)),
        cand_emb_cache=cand_emb_cache,
        mem_emb_cache=mem_emb_cache,
    )
    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate)
    model = build_model_from_checkpoint(checkpoint, rows, ds, device)
    
    predicted_goals_by_id = {}
    model.eval()
    with torch.no_grad():
        for batch, batch_rows in loader:
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
            edge_logits_for_rel = out["verifier_logits"] if model.use_verifier else out["edge_rel_logits"]
            edge_rel_pred = edge_logits_for_rel.argmax(dim=-1)
            mem_kind_pred = decode_mem_kind_predictions(out["mem_kind_logits"], batch.mem_mask)
            mem_rel_pred = out["mem_rel_logits"].argmax(dim=-1)
            mixed_dst_pred = out["mixed_dst_mem_logits"].argmax(dim=-1)
            bridge_a_pred = out["bridge_mem_a_logits"].argmax(dim=-1)
            bridge_b_pred = out["bridge_mem_b_logits"].argmax(dim=-1)

            for b, row in enumerate(batch_rows):
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
                _eval_slots = _ensure_target_slots(row)
                pred_name_to_slot = {
                    str(_eval_slots[k].get("session_name", f"slot_{k}")): k
                    for k in range(len(_eval_slots))
                }

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
                            if rel != "none":
                                predicted_edges.add((src_name, dst_name, rel))

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

                commit_name = COMMIT_FAMILIES[int(commit_pred[b].item())]

                predicted_goal = decode_unified_prediction_to_goal(
                    row, pred_nodes, predicted_edges, predicted_attach, predicted_cover, commit_name
                )
                predicted_goals_by_id[row["id"]] = predicted_goal
    return predicted_goals_by_id

def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate unified proposer end-to-end trajectory roundtrip")
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

    report = evaluate_roundtrip(model, val_loader, device, sample_limit=args.sample_limit)
    
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["global"], ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
