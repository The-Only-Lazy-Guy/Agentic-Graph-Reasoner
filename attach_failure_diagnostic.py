from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

from pred_model import MEM_LINK_KIND_TO_ID, REL_WITH_NONE
from train_pred_v1 import decode_mem_kind_predictions, read_jsonl
from train_unified_v1 import (
    UnifiedDataset,
    build_candidate_memory_ids,
    build_predicted_session_nodes,
    derive_gold_targets,
    set_seed,
    to_device,
    collate,
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
        k_max=max(len((row.get("target_slots", []) or [])) for row in rows),
        cand_emb_dim=dataset.cand_emb_dim,
        mem_emb_dim=dataset.mem_emb_dim,
    ).to(device)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    allowed_missing = {
        "cand_self_attn.in_proj_weight",
        "cand_self_attn.in_proj_bias",
        "cand_self_attn.out_proj.weight",
        "cand_self_attn.out_proj.bias",
        "cand_self_norm.weight",
        "cand_self_norm.bias",
    }
    missing_set = set(missing)
    unexpected_set = set(unexpected)
    if missing_set - allowed_missing or unexpected_set:
        raise RuntimeError(
            "Unexpected checkpoint/model mismatch: "
            f"missing={sorted(missing_set - allowed_missing)}, unexpected={sorted(unexpected_set)}"
        )
    if missing_set:
        model.use_cand_self_attn = False
    model.eval()
    return model


def row_attach_category(
    *,
    gold_attach: tuple[str, str, str] | None,
    predicted_attach: set[tuple[str, str, str]],
    predicted_cover: set[tuple[str, str]],
    pred_name_to_slot: Mapping[str, int],
    memory_ids: Sequence[str],
    mem_kind_pred_row: torch.Tensor,
    mem_rel_pred_row: torch.Tensor,
) -> tuple[str, Dict[str, Any]]:
    if gold_attach is None:
        if predicted_attach:
            return "spurious_attach_no_gold", {"predicted_attach": sorted(predicted_attach)}
        return "no_gold_attach", {}

    gold_slot, gold_mem, gold_rel = gold_attach
    gold_exact = (gold_slot, gold_mem, gold_rel)
    spurious_attach = predicted_attach - {gold_exact}
    if predicted_attach == {gold_exact}:
        return "exact_match", {}
    if gold_exact in predicted_attach and spurious_attach:
        return "spurious_only", {"spurious_attach": sorted(spurious_attach)}

    slot_idx = pred_name_to_slot.get(gold_slot)
    if slot_idx is None:
        return "slot_missing", {"gold_slot": gold_slot}
    try:
        mem_idx = memory_ids.index(gold_mem)
    except ValueError:
        return "gold_memory_missing", {"gold_mem": gold_mem}

    gold_kind_id = int(mem_kind_pred_row[slot_idx, mem_idx].item())
    gold_rel_id = int(mem_rel_pred_row[slot_idx, mem_idx].item())
    slot_pred_attach = sorted(x for x in predicted_attach if x[0] == gold_slot)
    slot_pred_cover = sorted(x for x in predicted_cover if x[0] == gold_slot)

    if any(mem == gold_mem for _, mem in slot_pred_cover):
        return "wrong_kind_cover", {
            "gold_kind_pred": gold_kind_id,
            "slot_pred_cover": slot_pred_cover,
            "slot_pred_attach": slot_pred_attach,
        }
    if any(mem == gold_mem for _, mem, _ in slot_pred_attach):
        return "wrong_relation", {
            "gold_kind_pred": gold_kind_id,
            "gold_rel_pred": REL_WITH_NONE[gold_rel_id],
            "slot_pred_attach": slot_pred_attach,
        }
    if slot_pred_attach:
        return "wrong_memory", {
            "gold_kind_pred": gold_kind_id,
            "slot_pred_attach": slot_pred_attach,
        }
    if gold_kind_id == MEM_LINK_KIND_TO_ID["cover"]:
        return "wrong_kind_cover", {
            "gold_kind_pred": gold_kind_id,
            "slot_pred_cover": slot_pred_cover,
        }
    if gold_kind_id == MEM_LINK_KIND_TO_ID["attach"]:
        return "wrong_relation", {
            "gold_kind_pred": gold_kind_id,
            "gold_rel_pred": REL_WITH_NONE[gold_rel_id],
            "slot_pred_attach": slot_pred_attach,
        }
    if spurious_attach:
        return "spurious_and_missed", {
            "spurious_attach": sorted(spurious_attach),
        }
    return "missed_attach", {
        "gold_kind_pred": gold_kind_id,
        "gold_rel_pred": REL_WITH_NONE[gold_rel_id],
        "slot_pred_cover": slot_pred_cover,
    }


def evaluate_attach_failures(
    model: UnifiedProposalAlignerNet,
    loader: DataLoader,
    device: torch.device,
    *,
    sample_limit_per_category: int = 3,
) -> Dict[str, Any]:
    row_category_hist: Dict[str, int] = {}
    missing_gold_hist: Dict[str, int] = {}
    category_samples: Dict[str, List[Dict[str, Any]]] = {}
    text_match_span_mismatch = 0
    mixed_rows = 0
    attach_failure_rows = 0

    model.eval()
    with torch.no_grad():
        for batch, rows in loader:
            batch = to_device(batch, device)
            out = model(batch)
            use_pred = (torch.sigmoid(out["use_logits"]) >= 0.5) & batch.slot_mask
            bridge_pred = (torch.sigmoid(out["type_logits"]) >= 0.5) & use_pred
            span_pred = out["span_logits"].argmax(dim=-1)
            mem_kind_pred = decode_mem_kind_predictions(out["mem_kind_logits"], batch.mem_mask)
            mem_rel_pred = out["mem_rel_logits"].argmax(dim=-1)
            mixed_dst_pred = out["mixed_dst_mem_logits"].argmax(dim=-1)
            bridge_a_pred = out["bridge_mem_a_logits"].argmax(dim=-1)
            bridge_b_pred = out["bridge_mem_b_logits"].argmax(dim=-1)

            for b, row in enumerate(rows):
                if str(row.get("task_type", "")) != "mixed_add_link":
                    continue
                mixed_rows += 1

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

                pred_text_by_name = {str(node.get("name", "")): str(node.get("span_text", "")) for node in pred_nodes}
                pred_span_by_name = {str(node.get("name", "")): str(node.get("span_id", "")) for node in pred_nodes}
                pred_name_to_slot = {
                    str((row.get("target_slots", []) or [])[k].get("session_name", f"slot_{k}")): k
                    for k in range(len(row.get("target_slots", []) or []))
                }

                gold_attach = next(iter(gold["gold_attach"]), None)
                predicted_attach = set()
                predicted_cover = set()
                for name in pred_text_by_name:
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

                if gold_attach is not None:
                    gold_name = gold_attach[0]
                    pred_text = pred_text_by_name.get(gold_name, "")
                    gold_text = gold["gold_text_by_name"].get(gold_name, "")
                    pred_span = pred_span_by_name.get(gold_name, "")
                    gold_span = gold["gold_span_by_name"].get(gold_name, "")
                    if pred_text == gold_text and pred_span != gold_span:
                        text_match_span_mismatch += 1

                row_cat, details = row_attach_category(
                    gold_attach=gold_attach,
                    predicted_attach=predicted_attach,
                    predicted_cover=predicted_cover,
                    pred_name_to_slot=pred_name_to_slot,
                    memory_ids=memory_ids,
                    mem_kind_pred_row=mem_kind_pred[b].cpu(),
                    mem_rel_pred_row=mem_rel_pred[b].cpu(),
                )
                row_category_hist[row_cat] = row_category_hist.get(row_cat, 0) + 1
                if row_cat != "exact_match":
                    attach_failure_rows += 1

                if gold_attach is not None and row_cat not in {"exact_match", "spurious_only"}:
                    missing_gold_hist[row_cat] = missing_gold_hist.get(row_cat, 0) + 1

                if row_cat != "exact_match":
                    samples = category_samples.setdefault(row_cat, [])
                    if len(samples) < sample_limit_per_category:
                        samples.append(
                            {
                                "id": row.get("id", ""),
                                "gold_attach": gold_attach,
                                "predicted_attach": sorted(predicted_attach),
                                "predicted_cover": sorted(predicted_cover),
                                "pred_session_nodes": pred_nodes,
                                "gold_session_nodes": list((row.get("_oracle_goal", {}) or {}).get("session_nodes", []) or []),
                                "details": details,
                            }
                        )

    return {
        "mixed_add_link_rows": mixed_rows,
        "attach_failure_rows": attach_failure_rows,
        "row_category_histogram": dict(sorted(row_category_hist.items(), key=lambda kv: (-kv[1], kv[0]))),
        "missing_gold_attach_histogram": dict(sorted(missing_gold_hist.items(), key=lambda kv: (-kv[1], kv[0]))),
        "text_match_span_mismatch_count": text_match_span_mismatch,
        "samples": category_samples,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--cand-emb-cache", default="")
    ap.add_argument("--mem-emb-cache", default="")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = read_jsonl(args.jsonl)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    dataset = UnifiedDataset(
        rows,
        cand_emb_cache=args.cand_emb_cache or None,
        mem_emb_cache=args.mem_emb_cache or None,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    model = build_model_from_checkpoint(checkpoint, rows, dataset, device)
    report = evaluate_attach_failures(model, loader, device)

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
