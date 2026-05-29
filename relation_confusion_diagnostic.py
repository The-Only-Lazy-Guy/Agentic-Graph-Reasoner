from __future__ import annotations

"""
relation_confusion_diagnostic.py

For mixed_add_link rows where the model gets the attach (gold_slot, gold_mem)
correct but predicts the wrong relation label, report:

  - gold_rel -> pred_rel confusion matrix
  - concentration (top confusion pair share of all wrong_relation cases)
  - per-confusion-pair sample rows

This directly informs whether fix3 should target a specific class confusion
(focused loss reweighting) or whether the misclassification is spread out
(needs broader representation work).

Mirrors the structure of attach_failure_diagnostic.py:1 so checkpoints that
predate later architecture additions still load via the same compatibility
guard.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

from pred_model import MEM_LINK_KIND_TO_ID, REL_WITH_NONE
from train_pred_v1 import decode_mem_kind_predictions, read_jsonl
from train_unified_v1 import (
    UnifiedDataset,
    build_candidate_memory_ids,
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


def evaluate_relation_confusion(
    model: UnifiedProposalAlignerNet,
    loader: DataLoader,
    device: torch.device,
    *,
    sample_limit_per_pair: int = 3,
) -> Dict[str, Any]:
    confusion: Dict[Tuple[str, str], int] = {}
    pair_samples: Dict[str, List[Dict[str, Any]]] = {}
    wrong_relation_rows = 0
    mixed_rows = 0
    gold_rel_totals: Dict[str, int] = {}
    pred_rel_totals: Dict[str, int] = {}

    model.eval()
    with torch.no_grad():
        for batch, rows in loader:
            batch = to_device(batch, device)
            out = model(batch)
            mem_kind_pred = decode_mem_kind_predictions(out["mem_kind_logits"], batch.mem_mask)
            mem_rel_pred = out["mem_rel_logits"].argmax(dim=-1)

            for b, row in enumerate(rows):
                if str(row.get("task_type", "")) != "mixed_add_link":
                    continue
                mixed_rows += 1

                gold = derive_gold_targets(row)
                gold_attach = next(iter(gold["gold_attach"]), None)
                if gold_attach is None:
                    continue
                gold_slot, gold_mem, gold_rel = gold_attach

                graph = loader.dataset.graph(str(row.get("graph_path", "")))  # type: ignore[attr-defined]
                memory_ids = build_candidate_memory_ids(row, graph)
                pred_name_to_slot = {
                    str((row.get("target_slots", []) or [])[k].get("session_name", f"slot_{k}")): k
                    for k in range(len(row.get("target_slots", []) or []))
                }
                slot_idx = pred_name_to_slot.get(gold_slot)
                if slot_idx is None:
                    continue
                try:
                    mem_idx = memory_ids.index(gold_mem)
                except ValueError:
                    continue

                kind_id = int(mem_kind_pred[b, slot_idx, mem_idx].item())
                if kind_id != MEM_LINK_KIND_TO_ID["attach"]:
                    continue
                rel_id = int(mem_rel_pred[b, slot_idx, mem_idx].item())
                pred_rel = REL_WITH_NONE[rel_id]
                if pred_rel == gold_rel:
                    continue

                wrong_relation_rows += 1
                confusion_key = (gold_rel, pred_rel)
                confusion[confusion_key] = confusion.get(confusion_key, 0) + 1
                gold_rel_totals[gold_rel] = gold_rel_totals.get(gold_rel, 0) + 1
                pred_rel_totals[pred_rel] = pred_rel_totals.get(pred_rel, 0) + 1

                pair_key = f"{gold_rel}->{pred_rel}"
                samples = pair_samples.setdefault(pair_key, [])
                if len(samples) < sample_limit_per_pair:
                    goal = row.get("_oracle_goal", {}) or {}
                    samples.append(
                        {
                            "id": row.get("id", ""),
                            "gold_slot": gold_slot,
                            "gold_mem_id": gold_mem,
                            "gold_relation": gold_rel,
                            "pred_relation": pred_rel,
                            "signal_excerpt": str(row.get("signal", ""))[:240],
                            "gold_session_nodes": list(goal.get("session_nodes", []) or []),
                            "gold_memory_attachments": list(goal.get("memory_attachments", []) or []),
                        }
                    )

    confusion_serialised = {
        f"{g}->{p}": c
        for (g, p), c in sorted(confusion.items(), key=lambda kv: (-kv[1], kv[0]))
    }
    top_pair_count = max(confusion.values()) if confusion else 0
    top_two_count = sum(sorted(confusion.values(), reverse=True)[:2]) if confusion else 0

    return {
        "mixed_add_link_rows": mixed_rows,
        "wrong_relation_rows": wrong_relation_rows,
        "confusion_matrix": confusion_serialised,
        "top_pair_count": top_pair_count,
        "top_pair_share_of_wrong_relation": (
            top_pair_count / wrong_relation_rows if wrong_relation_rows else 0.0
        ),
        "top_two_pair_share_of_wrong_relation": (
            top_two_count / wrong_relation_rows if wrong_relation_rows else 0.0
        ),
        "gold_rel_distribution": dict(
            sorted(gold_rel_totals.items(), key=lambda kv: (-kv[1], kv[0]))
        ),
        "pred_rel_distribution": dict(
            sorted(pred_rel_totals.items(), key=lambda kv: (-kv[1], kv[0]))
        ),
        "samples": pair_samples,
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
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    dataset = UnifiedDataset(
        rows,
        cand_emb_cache=args.cand_emb_cache or None,
        mem_emb_cache=args.mem_emb_cache or None,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    model = build_model_from_checkpoint(checkpoint, rows, dataset, device)
    report = evaluate_relation_confusion(model, loader, device)

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
