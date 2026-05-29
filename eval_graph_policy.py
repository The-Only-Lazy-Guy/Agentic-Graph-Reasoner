from __future__ import annotations

"""
eval_graph_policy.py

Detailed evaluator for NGR-v0.

Reports:
- overall metrics
- per-task edit accuracy
- per-gold-edit accuracy
- precision/recall/F1 per predicted edit type
- edit confusion matrix
- pointer top-1/top-5
"""

import argparse
import json
from collections import Counter, defaultdict
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

from graph_policy_env import EDIT_TYPES
from graph_policy_model import GraphPolicyNet
from train_graph_policy import GraphPolicyDataset, collate, to_device, read_jsonl, IGNORE


def topk_contains(logits: torch.Tensor, target: torch.Tensor, k: int) -> torch.Tensor:
    kk = min(k, logits.size(-1))
    topk = logits.topk(kk, dim=-1).indices
    return (topk == target[:, None]).any(dim=-1)


@torch.no_grad()
def evaluate_detailed(model: GraphPolicyNet, rows: List[Dict[str, Any]], loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    model.eval()

    total = edit_ok = 0
    rel_total = rel_ok = 0
    target_total = target_ok = target_top5 = 0
    src_total = src_ok = src_top5 = 0
    dst_total = dst_ok = dst_top5 = 0

    per_task_total = Counter()
    per_task_ok = Counter()
    per_edit_total = Counter()
    per_edit_ok = Counter()
    pred_edit_total = Counter()
    pred_edit_true = Counter()
    confusion = defaultdict(Counter)

    row_offset = 0

    for batch in loader:
        batch_size = batch.y_edit.numel()
        batch_rows = rows[row_offset: row_offset + batch_size]
        row_offset += batch_size

        batch = to_device(batch, device)
        out = model(batch)

        pred_edit = out["edit_logits"].argmax(-1)
        pred_rel = out["relation_logits"].argmax(-1)
        pred_target = out["target_logits"].argmax(-1)
        pred_src = out["src_logits"].argmax(-1)
        pred_dst = out["dst_logits"].argmax(-1)

        edit_match = pred_edit == batch.y_edit
        total += batch_size
        edit_ok += edit_match.sum().item()

        for i, row in enumerate(batch_rows):
            task = row.get("task_type", "unknown")
            gold_edit = EDIT_TYPES[int(batch.y_edit[i].item())]
            pred_name = EDIT_TYPES[int(pred_edit[i].item())]

            per_task_total[task] += 1
            per_edit_total[gold_edit] += 1
            pred_edit_total[pred_name] += 1
            confusion[gold_edit][pred_name] += 1

            if bool(edit_match[i].item()):
                per_task_ok[task] += 1
                per_edit_ok[gold_edit] += 1
                pred_edit_true[pred_name] += 1

        mask = batch.y_rel != IGNORE
        rel_total += mask.sum().item()
        rel_ok += ((pred_rel == batch.y_rel) & mask).sum().item()

        mask = batch.y_target != IGNORE
        target_total += mask.sum().item()
        target_ok += ((pred_target == batch.y_target) & mask).sum().item()
        target_top5 += (topk_contains(out["target_logits"], batch.y_target.clamp_min(0), 5) & mask).sum().item()

        mask = batch.y_src != IGNORE
        src_total += mask.sum().item()
        src_ok += ((pred_src == batch.y_src) & mask).sum().item()
        src_top5 += (topk_contains(out["src_logits"], batch.y_src.clamp_min(0), 5) & mask).sum().item()

        mask = batch.y_dst != IGNORE
        dst_total += mask.sum().item()
        dst_ok += ((pred_dst == batch.y_dst) & mask).sum().item()
        dst_top5 += (topk_contains(out["dst_logits"], batch.y_dst.clamp_min(0), 5) & mask).sum().item()

    def ratio(a: int, b: int) -> float:
        return float(a) / float(max(b, 1))

    edit_prf = {}
    for name in EDIT_TYPES:
        tp = pred_edit_true[name]
        pred_n = pred_edit_total[name]
        gold_n = per_edit_total[name]
        precision = ratio(tp, pred_n)
        recall = ratio(tp, gold_n)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        edit_prf[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "gold_n": gold_n,
            "pred_n": pred_n,
            "tp": tp,
        }

    return {
        "overall": {
            "edit_acc": ratio(edit_ok, total),
            "relation_acc": ratio(rel_ok, rel_total),
            "target_top1": ratio(target_ok, target_total),
            "target_top5": ratio(target_top5, target_total),
            "src_top1": ratio(src_ok, src_total),
            "src_top5": ratio(src_top5, src_total),
            "dst_top1": ratio(dst_ok, dst_total),
            "dst_top5": ratio(dst_top5, dst_total),
            "total": total,
        },
        "per_task_edit_acc": {
            k: {"acc": ratio(per_task_ok[k], per_task_total[k]), "n": per_task_total[k]}
            for k in sorted(per_task_total)
        },
        "per_gold_edit_acc": {
            k: {"acc": ratio(per_edit_ok[k], per_edit_total[k]), "n": per_edit_total[k]}
            for k in sorted(per_edit_total)
        },
        "edit_precision_recall_f1": edit_prf,
        "edit_confusion": {
            gold: dict(preds)
            for gold, preds in sorted(confusion.items())
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val-jsonl", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("args", {})

    model = GraphPolicyNet(
        hash_dim=int(cfg.get("hash_dim", 512)),
        hidden_dim=int(cfg.get("hidden_dim", 256)),
    ).to(device)
    model.load_state_dict(ckpt["model"])

    rows = read_jsonl(args.val_jsonl)
    ds = GraphPolicyDataset(
        rows,
        hash_dim=int(cfg.get("hash_dim", 512)),
        max_candidates=int(cfg.get("max_candidates", 64)),
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    metrics = evaluate_detailed(model, rows, loader, device)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
