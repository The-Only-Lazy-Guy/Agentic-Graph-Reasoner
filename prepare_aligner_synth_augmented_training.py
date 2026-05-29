from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

from graph_core import MemoryGraph
from proposer_model import ProposerNet, infer_proposer_arch_from_state
from synthesize_node_text import _best_matching_memory_id, _memory_text, _load_graph, clean
from train_pred_v1 import read_jsonl
from train_proposer_v1 import ProposerDataset, collate, predicted_slots_for_row, to_device


SYNTH_TASKS = {"mixed_add_link", "multi_region_attach"}


def write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _gold_used_slots(row: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [dict(slot) for slot in (row.get("target_slots", []) or []) if bool(slot.get("use"))]


def _predict_slots_for_row(
    row: Mapping[str, Any],
    use_prob: torch.Tensor,
    span_argmax: torch.Tensor,
    bridge_pred: torch.Tensor,
) -> List[Dict[str, Any]]:
    use_hard = use_prob >= 0.5
    slots = predicted_slots_for_row(row, use_hard, span_argmax, bridge_pred)
    spans = row.get("spans", []) or []
    for idx, slot in enumerate(slots):
        slot["slot_idx"] = idx
        slot["use_score"] = float(use_prob[idx].item())
        slot["bridge_score"] = float(bridge_pred[idx].item())
        pred_idx = int(span_argmax[idx].item())
        if slot.get("span_id") is None and 0 <= pred_idx < len(spans):
            span = spans[pred_idx]
            slot["span_id"] = span.get("id")
            slot["span_text"] = span.get("text")
            slot["anchor_start"] = span.get("start")
            slot["anchor_end"] = span.get("end")
            slot["is_bridge"] = bool(bridge_pred[idx].item())
            slot["node_type"] = "bridge" if slot["is_bridge"] else "concept"
        if slot.get("span_text") is None:
            slot["span_text"] = ""
        slot["use"] = True
    return slots


def _valid_predicted_slots(predicted_slots: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    valid = [dict(slot) for slot in predicted_slots if str(slot.get("span_text", "") or "").strip()]
    return sorted(
        valid,
        key=lambda slot: (
            int(slot.get("anchor_start", 10**9) or 10**9),
            int(slot.get("slot_idx", 10**9)),
        ),
    )


def _pick_mixed_source_slot(predicted_slots: Sequence[Mapping[str, Any]]) -> Dict[str, Any] | None:
    valid = _valid_predicted_slots(predicted_slots)
    return valid[0] if valid else None


def _pick_bridge_support_text(predicted_slots: Sequence[Mapping[str, Any]], fallback: str) -> str:
    valid = _valid_predicted_slots(predicted_slots)
    non_bridge = [slot for slot in valid if not bool(slot.get("is_bridge"))]
    if non_bridge:
        return str(non_bridge[0].get("span_text", "") or "")
    return fallback


def _build_pred_style_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    goal = copy.deepcopy((row.get("_oracle_goal", {}) or {}))
    gold_slots = _gold_used_slots(row)
    slot_by_name = {
        str(slot.get("session_name", "")): slot
        for slot in gold_slots
        if slot.get("session_name") is not None
    }
    span_oracle = []
    for node in goal.get("session_nodes", []) or []:
        name = str(node.get("name", ""))
        gold_slot = slot_by_name.get(name, {})
        span_oracle.append(
            {
                "session_name": name,
                "spec_text": str(node.get("span_text", "")),
                "node_type": str(node.get("node_type", "concept")),
                "best_span_id": gold_slot.get("span_id"),
                "best_score": float(gold_slot.get("oracle_best_score", 0.0)),
                "span_scores": [],
            }
        )
    return {
        "id": row.get("id", ""),
        "task_type": row.get("task_type", ""),
        "graph_path": row.get("graph_path", ""),
        "signal": row.get("signal", ""),
        "initial_memory_node_ids": list(row.get("initial_memory_node_ids", []) or []),
        "spans": copy.deepcopy(list(row.get("spans", []) or [])),
        "goal": goal,
        "span_oracle": span_oracle,
        "is_pseudo_goal": False,
        "meta": {
            "num_nodes": len(goal.get("session_nodes", []) or []),
            "num_edges": len(goal.get("session_edges", []) or []),
            "num_attachments": len(goal.get("memory_attachments", []) or []),
            "num_covered": len(goal.get("covered_mappings", []) or []),
        },
    }


def _rewrite_synthesis_texts(
    row: Mapping[str, Any],
    pred_row: Dict[str, Any],
    predicted_slots: Sequence[Mapping[str, Any]],
    *,
    graph_cache: Dict[str, MemoryGraph],
) -> bool:
    task_type = str(row.get("task_type", ""))
    if task_type not in SYNTH_TASKS:
        return False

    graph = _load_graph(graph_cache, str(row.get("graph_path", "")))
    memory_ids = [str(x) for x in (row.get("initial_memory_node_ids", []) or []) if str(x)]
    nodes = pred_row["goal"].get("session_nodes", []) or []
    oracle_by_name = {str(entry.get("session_name", "")): entry for entry in pred_row.get("span_oracle", []) or []}
    changed = False

    if task_type == "mixed_add_link" and memory_ids:
        source_slot = _pick_mixed_source_slot(predicted_slots)
        if source_slot is not None:
            source_text = str(source_slot.get("span_text", "") or "")
            dst_text = _memory_text(graph, memory_ids[0], 110)
            new_text = clean(
                f"{source_text} This supports a new note related to {dst_text}.",
                220,
            )
            for node in nodes:
                if str(node.get("name", "")) == "new_note":
                    if str(node.get("span_text", "")) != new_text:
                        changed = True
                    node["span_text"] = new_text
                    if "new_note" in oracle_by_name:
                        oracle_by_name["new_note"]["spec_text"] = new_text
                    break
        return changed

    if task_type == "multi_region_attach" and len(memory_ids) >= 2:
        fallback_support = ""
        for node in nodes:
            if str(node.get("name", "")) == "support_note":
                fallback_support = str(node.get("span_text", "") or "")
                break
        support_text = _pick_bridge_support_text(predicted_slots, fallback_support)
        support_mem = _best_matching_memory_id(memory_ids, support_text, graph)
        if support_mem is not None:
            other_mem = next((m for m in memory_ids if m != support_mem), None)
            if other_mem is not None:
                text_a = _memory_text(graph, support_mem, 90)
                text_b = _memory_text(graph, other_mem, 90)
                bridge_text = clean(
                    f"{text_a} and {text_b} are connected by a shared bridge concept.",
                    180,
                )
                for node in nodes:
                    if str(node.get("name", "")) == "bridge":
                        if str(node.get("span_text", "")) != bridge_text:
                            changed = True
                        node["span_text"] = bridge_text
                        if "bridge" in oracle_by_name:
                            oracle_by_name["bridge"]["spec_text"] = bridge_text
                        break
        return changed

    return False


def prepare_augmented_rows(
    proposer_rows: Sequence[Mapping[str, Any]],
    *,
    proposer_ckpt_path: str,
    cand_emb_cache: str | None,
    mem_emb_cache: str | None,
    batch_size: int,
    device: torch.device,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    ckpt = torch.load(proposer_ckpt_path, map_location=device)
    args = dict(ckpt.get("args", {}))
    hidden_dim = int(args.get("hidden_dim", 256))
    arch = infer_proposer_arch_from_state(ckpt["model"], hidden_dim)
    model = ProposerNet(
        hash_dim=int(args.get("hash_dim", 512)),
        hidden_dim=hidden_dim,
        k_max=int(args.get("k_max", 3)),
        cand_pair_feat_dim=int(arch.get("cand_pair_feat_dim", 3)),
        cand_emb_dim=int(args.get("cand_emb_dim", 384)),
        mem_emb_dim=int(args.get("mem_emb_dim", 384)),
        use_ar_span_features=bool(arch.get("use_ar_span_features", False)),
        slot_attention_mode=str(arch.get("slot_attention_mode", "none")),
        span_scorer_mode=str(arch.get("span_scorer_mode", "concat_mlp")),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = ProposerDataset(
        proposer_rows,
        hash_dim=int(args.get("hash_dim", 512)),
        cand_emb_cache=cand_emb_cache,
        cand_emb_dim_override=int(args.get("cand_emb_dim", 384)),
        mem_emb_cache=mem_emb_cache,
        mem_emb_dim_override=int(args.get("mem_emb_dim", 384)),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    augmented_rows: List[Dict[str, Any]] = []
    graph_cache: Dict[str, MemoryGraph] = {}
    synthesis_rows = 0
    changed_rows = 0

    with torch.no_grad():
        for batch, rows in loader:
            batch = to_device(batch, device)
            out = model.predict(batch)
            use_prob = torch.sigmoid(out["use_logits"]).cpu()
            span_argmax = out["span_logits"].argmax(dim=-1).cpu()
            bridge_pred = out["bridge_pred"].cpu()
            for b, row in enumerate(rows):
                pred_slots = _predict_slots_for_row(row, use_prob[b], span_argmax[b], bridge_pred[b])
                pred_row = _build_pred_style_row(row)
                changed = _rewrite_synthesis_texts(row, pred_row, pred_slots, graph_cache=graph_cache)
                pred_row["augmentation_meta"] = {
                    "source": "fix5_proposer",
                    "predicted_slots": [
                        {
                            "slot_idx": int(slot.get("slot_idx", -1)),
                            "span_id": slot.get("span_id"),
                            "span_text": slot.get("span_text"),
                            "node_type": slot.get("node_type"),
                            "use_score": float(slot.get("use_score", 0.0)),
                            "bridge_score": float(slot.get("bridge_score", 0.0)),
                        }
                        for slot in pred_slots
                    ],
                    "text_rewrite_applied": changed,
                }
                augmented_rows.append(pred_row)
                if str(row.get("task_type", "")) in SYNTH_TASKS:
                    synthesis_rows += 1
                    changed_rows += int(changed)

    stats = {
        "rows": len(augmented_rows),
        "synthesis_eligible_rows": synthesis_rows,
        "rows_with_changed_session_text": changed_rows,
        "changed_row_rate_overall": changed_rows / max(len(augmented_rows), 1),
        "changed_row_rate_on_synth_rows": changed_rows / max(synthesis_rows, 1),
    }
    return augmented_rows, stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare gold-name-preserving synthesized aligner training rows")
    ap.add_argument("--input-jsonl", required=True, help="Proposer-format jsonl (e.g. proposer_train.jsonl)")
    ap.add_argument("--proposer-checkpoint", required=True)
    ap.add_argument("--cand-emb-cache", default="artifacts/spec_emb_cache/pred_v1_fix8_minilm_cand.npz")
    ap.add_argument("--mem-emb-cache", default="artifacts/spec_emb_cache/pred_v1_fix8_minilm_mem.npz")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--stats-json", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    proposer_rows = read_jsonl(args.input_jsonl)
    augmented_rows, stats = prepare_augmented_rows(
        proposer_rows,
        proposer_ckpt_path=args.proposer_checkpoint,
        cand_emb_cache=args.cand_emb_cache,
        mem_emb_cache=args.mem_emb_cache,
        batch_size=args.batch_size,
        device=device,
    )
    write_jsonl(args.output_jsonl, augmented_rows)
    if args.stats_json:
        Path(args.stats_json).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
