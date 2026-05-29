from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from eval_proposer_roundtrip import reconstruct_pred_row_from_slots
from graph_core import MemoryGraph, lexical_overlap
from graph_policy_model import bow_hash
from pred_model import (
    PredAlignNet,
    SPAN_KIND_TO_ID,
    infer_cand_emb_dim_from_state,
    infer_edge_rel_pair_feat_dim_from_state,
    infer_mem_emb_dim_from_state,
    infer_spec_emb_dim_from_state,
)
from proposer_model import ProposerBatch, ProposerNet
from train_pred_v1 import (
    PredDataset,
    collate as aligner_collate,
    compute_metrics as compute_aligner_metrics,
    load_embedding_cache,
    read_jsonl,
    text_cache_key,
)


IGNORE = -100


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(batch: ProposerBatch, device: torch.device) -> ProposerBatch:
    return ProposerBatch(**{k: getattr(batch, k).to(device) for k in batch.__dataclass_fields__})


class ProposerDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        hash_dim: int = 512,
        cand_emb_cache: str | Path | None = None,
        cand_emb_dim_override: int = 0,
        mem_emb_cache: str | Path | None = None,
        mem_emb_dim_override: int = 0,
    ) -> None:
        self.rows = list(rows)
        self.hash_dim = hash_dim
        self._graph_cache: Dict[str, MemoryGraph] = {}
        self.cand_emb_cache, self.cand_emb_dim = load_embedding_cache(cand_emb_cache)
        if self.cand_emb_dim == 0 and cand_emb_dim_override > 0:
            self.cand_emb_dim = int(cand_emb_dim_override)
        self.mem_emb_cache, self.mem_emb_dim = load_embedding_cache(mem_emb_cache)
        if self.mem_emb_dim == 0 and mem_emb_dim_override > 0:
            self.mem_emb_dim = int(mem_emb_dim_override)

    def __len__(self) -> int:
        return len(self.rows)

    def graph(self, path: str) -> MemoryGraph:
        if path not in self._graph_cache:
            self._graph_cache[path] = MemoryGraph.load_json(path)
        return self._graph_cache[path]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        signal = str(row.get("signal", ""))
        graph = self.graph(str(row.get("graph_path", "")))
        spans = row.get("spans", []) or []
        target_slots = row.get("target_slots", []) or []
        signal_bow = bow_hash(signal, self.hash_dim)

        cand_bow = []
        cand_emb = []
        cand_kind_ids = []
        cand_feat = []
        cand_pair_feat = []
        span_id_to_idx: Dict[str, int] = {}
        for j, span in enumerate(spans):
            sid = str(span.get("id", ""))
            text = str(span.get("text", ""))
            span_id_to_idx[sid] = j
            cand_bow.append(bow_hash(text, self.hash_dim))
            if self.cand_emb_dim > 0:
                emb = self.cand_emb_cache.get(text_cache_key(text))
                if emb is None:
                    cand_emb.append(torch.zeros(self.cand_emb_dim, dtype=torch.float32))
                else:
                    cand_emb.append(torch.from_numpy(emb.astype("float32")))
            cand_kind_ids.append(SPAN_KIND_TO_ID.get(str(span.get("span_kind", "unknown")), SPAN_KIND_TO_ID["unknown"]))
            cand_feat.append(
                torch.tensor(
                    [
                        float(span.get("start", 0)) / max(len(signal), 1),
                        float(span.get("end", 0)) / max(len(signal), 1),
                    ],
                    dtype=torch.float32,
                )
            )
            start = float(span.get("start", 0))
            end = float(span.get("end", 0))
            cand_pair_feat.append(
                torch.tensor(
                    [
                        float(lexical_overlap(signal, text)),
                        start / max(len(signal), 1),
                        max(end - start, 0.0) / max(len(signal), 1),
                    ],
                    dtype=torch.float32,
                )
            )

        mem_ids = [str(x) for x in row.get("initial_memory_node_ids", []) or [] if str(x) in graph.nodes]
        mem_bow = []
        mem_emb = []
        mem_feat = []
        init_set = set(mem_ids)
        for j, mem_id in enumerate(mem_ids):
            text = str(graph.nodes[mem_id].text)
            mem_bow.append(bow_hash(text, self.hash_dim))
            if self.mem_emb_dim > 0:
                emb = self.mem_emb_cache.get(text_cache_key(text))
                if emb is None:
                    mem_emb.append(torch.zeros(self.mem_emb_dim, dtype=torch.float32))
                else:
                    mem_emb.append(torch.from_numpy(emb.astype("float32")))
            mem_feat.append(
                torch.tensor(
                    [
                        float(lexical_overlap(signal, text)),
                        1.0 if mem_id in init_set else 0.0,
                        float(j) / max(len(mem_ids), 1),
                    ],
                    dtype=torch.float32,
                )
            )

        y_use = []
        y_span = []
        y_is_bridge = []
        for slot in target_slots:
            use = bool(slot.get("use"))
            y_use.append(1.0 if use else 0.0)
            if use:
                y_span.append(span_id_to_idx.get(str(slot.get("span_id", "")), IGNORE))
                y_is_bridge.append(1.0 if str(slot.get("node_type", "concept")) == "bridge" else 0.0)
            else:
                y_span.append(IGNORE)
                y_is_bridge.append(0.0)

        return {
            "id": row.get("id", ""),
            "task_type": row.get("task_type", ""),
            "signal_bow": signal_bow,
            "cand_bow": torch.stack(cand_bow, dim=0) if cand_bow else torch.zeros((0, self.hash_dim), dtype=torch.float32),
            "cand_emb": torch.stack(cand_emb, dim=0) if cand_emb else torch.zeros((len(spans), self.cand_emb_dim), dtype=torch.float32),
            "cand_kind_ids": torch.tensor(cand_kind_ids, dtype=torch.long) if cand_kind_ids else torch.zeros((0,), dtype=torch.long),
            "cand_feat": torch.stack(cand_feat, dim=0) if cand_feat else torch.zeros((0, 2), dtype=torch.float32),
            "cand_pair_feat": torch.stack(cand_pair_feat, dim=0) if cand_pair_feat else torch.zeros((0, 3), dtype=torch.float32),
            "mem_bow": torch.stack(mem_bow, dim=0) if mem_bow else torch.zeros((0, self.hash_dim), dtype=torch.float32),
            "mem_emb": torch.stack(mem_emb, dim=0) if mem_emb else torch.zeros((len(mem_ids), self.mem_emb_dim), dtype=torch.float32),
            "mem_feat": torch.stack(mem_feat, dim=0) if mem_feat else torch.zeros((0, 3), dtype=torch.float32),
            "y_use": torch.tensor(y_use, dtype=torch.float32),
            "y_span": torch.tensor(y_span, dtype=torch.long),
            "y_is_bridge": torch.tensor(y_is_bridge, dtype=torch.float32),
            "row": row,
        }


def collate(batch: Sequence[Mapping[str, Any]]) -> tuple[ProposerBatch, List[Mapping[str, Any]]]:
    B = len(batch)
    max_c = max(x["cand_bow"].size(0) for x in batch)
    max_m = max(x["mem_bow"].size(0) for x in batch)
    k_max = max(x["y_use"].size(0) for x in batch)
    hash_dim = batch[0]["signal_bow"].numel()
    cand_emb_dim = batch[0]["cand_emb"].size(1) if batch[0]["cand_emb"].ndim == 2 else 0
    mem_emb_dim = batch[0]["mem_emb"].size(1) if batch[0]["mem_emb"].ndim == 2 else 0

    signal_bow = torch.zeros((B, hash_dim), dtype=torch.float32)
    cand_bow = torch.zeros((B, max_c, hash_dim), dtype=torch.float32)
    cand_emb = torch.zeros((B, max_c, cand_emb_dim), dtype=torch.float32)
    cand_kind_ids = torch.zeros((B, max_c), dtype=torch.long)
    cand_feat = torch.zeros((B, max_c, 2), dtype=torch.float32)
    cand_pair_feat = torch.zeros((B, max_c, 3), dtype=torch.float32)
    cand_mask = torch.zeros((B, max_c), dtype=torch.bool)
    mem_bow = torch.zeros((B, max_m, hash_dim), dtype=torch.float32)
    mem_emb = torch.zeros((B, max_m, mem_emb_dim), dtype=torch.float32)
    mem_feat = torch.zeros((B, max_m, 3), dtype=torch.float32)
    mem_mask = torch.zeros((B, max_m), dtype=torch.bool)
    y_use = torch.zeros((B, k_max), dtype=torch.float32)
    y_span = torch.full((B, k_max), IGNORE, dtype=torch.long)
    y_is_bridge = torch.zeros((B, k_max), dtype=torch.float32)
    y_slot_mask = torch.zeros((B, k_max), dtype=torch.bool)
    rows: List[Mapping[str, Any]] = []

    for b, x in enumerate(batch):
        signal_bow[b] = x["signal_bow"]
        c = x["cand_bow"].size(0)
        m = x["mem_bow"].size(0)
        k = x["y_use"].size(0)
        cand_bow[b, :c] = x["cand_bow"]
        if cand_emb_dim > 0:
            cand_emb[b, :c] = x["cand_emb"]
        cand_kind_ids[b, :c] = x["cand_kind_ids"]
        cand_feat[b, :c] = x["cand_feat"]
        cand_pair_feat[b, :c] = x["cand_pair_feat"]
        cand_mask[b, :c] = True
        mem_bow[b, :m] = x["mem_bow"]
        if mem_emb_dim > 0:
            mem_emb[b, :m] = x["mem_emb"]
        mem_feat[b, :m] = x["mem_feat"]
        mem_mask[b, :m] = True
        y_use[b, :k] = x["y_use"]
        y_span[b, :k] = x["y_span"]
        y_is_bridge[b, :k] = x["y_is_bridge"]
        y_slot_mask[b, :k] = True
        rows.append(x["row"])

    proposer_batch = ProposerBatch(
        signal_bow=signal_bow,
        cand_bow=cand_bow,
        cand_emb=cand_emb,
        cand_kind_ids=cand_kind_ids,
        cand_feat=cand_feat,
        cand_pair_feat=cand_pair_feat,
        cand_mask=cand_mask,
        mem_bow=mem_bow,
        mem_emb=mem_emb,
        mem_feat=mem_feat,
        mem_mask=mem_mask,
        y_use=y_use,
        y_span=y_span,
        y_is_bridge=y_is_bridge,
        y_slot_mask=y_slot_mask,
    )
    return proposer_batch, rows


def decode_slots(
    use_logits: torch.Tensor,
    span_logits: torch.Tensor,
    type_logits: torch.Tensor,
    cand_mask: torch.Tensor,
    *,
    use_threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    use_pred = torch.sigmoid(use_logits) >= use_threshold
    span_pred = span_logits.argmax(dim=-1)
    span_pred = torch.where(use_pred, span_pred, torch.full_like(span_pred, IGNORE))
    valid_any = cand_mask.any(dim=-1, keepdim=True)
    span_pred = torch.where(valid_any, span_pred, torch.full_like(span_pred, IGNORE))
    is_bridge_pred = torch.sigmoid(type_logits) >= 0.5
    is_bridge_pred = is_bridge_pred & use_pred
    return use_pred, span_pred, is_bridge_pred


def compute_metrics(model: ProposerNet, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    model.eval()
    use_total = use_ok = 0
    span_total = span_ok = 0
    type_total = type_ok = 0
    slot_row_total = slot_row_ok = 0
    slot2_total = slot2_ok = 0

    with torch.no_grad():
        for batch, _rows in loader:
            batch = to_device(batch, device)
            out = model.predict(batch)
            use_pred = out["use_pred"]
            span_pred = out["span_pred"]
            bridge_pred = out["bridge_pred"]
            slot_mask = batch.y_slot_mask
            use_gold = batch.y_use > 0.5
            use_match = (use_pred == use_gold) & slot_mask
            use_total += int(slot_mask.sum().item())
            use_ok += int(use_match.sum().item())

            used_gold = use_gold & slot_mask
            span_match = (span_pred == batch.y_span) & used_gold
            span_total += int(used_gold.sum().item())
            span_ok += int(span_match.sum().item())

            bridge_gold = batch.y_is_bridge > 0.5
            bridge_match = (bridge_pred == bridge_gold) & used_gold
            type_total += int(used_gold.sum().item())
            type_ok += int(bridge_match.sum().item())

            if slot_mask.size(1) > 2:
                slot2_total += int(slot_mask[:, 2].sum().item())
                slot2_ok += int(((use_pred[:, 2] == use_gold[:, 2]) & slot_mask[:, 2]).sum().item())

            B = batch.signal_bow.size(0)
            for b in range(B):
                row_use = bool(use_match[b, slot_mask[b]].all().item()) if slot_mask[b].any() else True
                row_span = bool(span_match[b, used_gold[b]].all().item()) if used_gold[b].any() else True
                row_type = bool(bridge_match[b, used_gold[b]].all().item()) if used_gold[b].any() else True
                slot_row_total += 1
                if row_use and row_span and row_type:
                    slot_row_ok += 1

    return {
        "use_acc": use_ok / max(use_total, 1),
        "span_acc_on_used": span_ok / max(span_total, 1),
        "bridge_acc_on_used": type_ok / max(type_total, 1),
        "slot_row_complete_rate": slot_row_ok / max(slot_row_total, 1),
        "slot2_use_acc": slot2_ok / max(slot2_total, 1),
    }


def predicted_slots_for_row(row: Mapping[str, Any], use_pred: torch.Tensor, span_pred: torch.Tensor, bridge_pred: torch.Tensor) -> List[Dict[str, Any]]:
    spans = row.get("spans", []) or []
    target_slots = row.get("target_slots", []) or []
    pred_slots: List[Dict[str, Any]] = []
    for idx, template in enumerate(target_slots):
        slot = dict(template)
        slot["slot_idx"] = idx
        slot["use"] = bool(use_pred[idx].item())
        if slot["use"] and 0 <= int(span_pred[idx].item()) < len(spans):
            span = spans[int(span_pred[idx].item())]
            slot["span_id"] = span.get("id")
            slot["span_text"] = span.get("text")
            slot["anchor_start"] = span.get("start")
            slot["anchor_end"] = span.get("end")
            slot["is_bridge"] = bool(bridge_pred[idx].item())
            slot["node_type"] = "bridge" if slot["is_bridge"] else "concept"
        else:
            slot["span_id"] = None
            slot["span_text"] = None
            slot["anchor_start"] = None
            slot["anchor_end"] = None
            slot["is_bridge"] = False
            slot["node_type"] = None
        pred_slots.append(slot)
    return pred_slots


def run_end_to_end_eval(
    proposer_model: ProposerNet,
    proposer_rows: Sequence[Mapping[str, Any]],
    device: torch.device,
    *,
    batch_size: int,
    hash_dim: int,
    cand_emb_cache: str | Path | None,
    cand_emb_dim: int,
    mem_emb_cache: str | Path | None,
    mem_emb_dim: int,
    aligner_ckpt_path: str,
    aligner_spec_emb_cache: str | Path | None,
    aligner_cand_emb_cache: str | Path | None,
    aligner_mem_emb_cache: str | Path | None,
) -> Dict[str, Any]:
    proposer_ds = ProposerDataset(
        proposer_rows,
        hash_dim=hash_dim,
        cand_emb_cache=cand_emb_cache,
        cand_emb_dim_override=cand_emb_dim,
        mem_emb_cache=mem_emb_cache,
        mem_emb_dim_override=mem_emb_dim,
    )
    proposer_loader = DataLoader(proposer_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    reconstructed_rows: List[Dict[str, Any]] = []
    proposer_model.eval()
    with torch.no_grad():
        for batch, rows in proposer_loader:
            batch = to_device(batch, device)
            out = proposer_model.predict(batch)
            use_pred = out["use_pred"]
            span_pred = out["span_pred"]
            bridge_pred = out["bridge_pred"]
            for b, row in enumerate(rows):
                pred_slots = predicted_slots_for_row(row, use_pred[b], span_pred[b], bridge_pred[b])
                reconstructed_rows.append(reconstruct_pred_row_from_slots(row, pred_slots))

    aligner_device = device
    ckpt = torch.load(aligner_ckpt_path, map_location=aligner_device)
    aligner_args = dict(ckpt.get("args", {}))
    spec_emb_dim = infer_spec_emb_dim_from_state(ckpt["model"])
    cand_align_emb_dim = infer_cand_emb_dim_from_state(ckpt["model"])
    mem_align_emb_dim = infer_mem_emb_dim_from_state(ckpt["model"])
    aligner_ds = PredDataset(
        reconstructed_rows,
        hash_dim=int(aligner_args.get("hash_dim", 512)),
        spec_emb_cache=aligner_spec_emb_cache,
        spec_emb_dim_override=spec_emb_dim,
        cand_emb_cache=aligner_cand_emb_cache,
        cand_emb_dim_override=cand_align_emb_dim,
        mem_emb_cache=aligner_mem_emb_cache,
        mem_emb_dim_override=mem_align_emb_dim,
    )
    aligner_loader = DataLoader(aligner_ds, batch_size=batch_size, shuffle=False, collate_fn=aligner_collate)
    aligner_model = PredAlignNet(
        hash_dim=int(aligner_args.get("hash_dim", 512)),
        hidden_dim=int(aligner_args.get("hidden_dim", 256)),
        edge_rel_pair_feat_dim=infer_edge_rel_pair_feat_dim_from_state(ckpt["model"], int(aligner_args.get("hidden_dim", 256))),
        spec_emb_dim=spec_emb_dim,
        cand_emb_dim=cand_align_emb_dim,
        mem_emb_dim=mem_align_emb_dim,
    ).to(aligner_device)
    aligner_model.load_state_dict(ckpt["model"])
    return compute_aligner_metrics(aligner_model, aligner_loader, aligner_device)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-jsonl", default="artifacts/proposer_v1_20260512/proposer_train.jsonl")
    ap.add_argument("--val-jsonl", default="artifacts/proposer_v1_20260512/proposer_val.jsonl")
    ap.add_argument("--out-dir", default="out_proposer_v1_20260512")
    ap.add_argument("--hash-dim", type=int, default=512)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--cand-emb-cache", default="artifacts/spec_emb_cache/pred_v1_fix8_minilm_cand.npz")
    ap.add_argument("--mem-emb-cache", default="artifacts/spec_emb_cache/pred_v1_fix8_minilm_mem.npz")
    ap.add_argument("--aligner-checkpoint", default="out_pred_v1_train_20260512_fix21_mememb_rel_freeze/best_pred_v1.pt")
    ap.add_argument("--aligner-spec-emb-cache", default="artifacts/spec_emb_cache/pred_v1_fix8_minilm_spec.npz")
    ap.add_argument("--aligner-cand-emb-cache", default="artifacts/spec_emb_cache/pred_v1_fix8_minilm_cand.npz")
    ap.add_argument("--aligner-mem-emb-cache", default="artifacts/spec_emb_cache/pred_v1_fix8_minilm_mem.npz")
    ap.add_argument("--use-loss-weight", type=float, default=1.0)
    ap.add_argument("--span-loss-weight", type=float, default=1.0)
    ap.add_argument("--type-loss-weight", type=float, default=0.25)
    ap.add_argument(
        "--disable-ar-span-features",
        action="store_true",
        help="Use the interaction scorer without autoregressive previous-slot pair features",
    )
    ap.add_argument(
        "--slot-attention-mode",
        choices=["none", "detr"],
        default="none",
        help="Refine slot queries with attention before span scoring",
    )
    ap.add_argument(
        "--span-scorer-mode",
        choices=["concat_mlp", "interaction_mlp", "dot"],
        default="concat_mlp",
        help="Span scoring head to use",
    )
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    train_rows = read_jsonl(args.train_jsonl)
    val_rows = read_jsonl(args.val_jsonl)
    train_ds = ProposerDataset(train_rows, hash_dim=args.hash_dim, cand_emb_cache=args.cand_emb_cache, mem_emb_cache=args.mem_emb_cache)
    val_ds = ProposerDataset(val_rows, hash_dim=args.hash_dim, cand_emb_cache=args.cand_emb_cache, mem_emb_cache=args.mem_emb_cache)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = ProposerNet(
        hash_dim=args.hash_dim,
        hidden_dim=args.hidden_dim,
        k_max=max(len((row.get("target_slots", []) or [])) for row in train_rows),
        cand_emb_dim=train_ds.cand_emb_dim,
        mem_emb_dim=train_ds.mem_emb_dim,
        use_ar_span_features=not args.disable_ar_span_features,
        slot_attention_mode=args.slot_attention_mode,
        span_scorer_mode=args.span_scorer_mode,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_score = float("-inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        batch_count = 0
        for batch, _rows in train_loader:
            batch = to_device(batch, device)
            out = model(batch)
            slot_mask = batch.y_slot_mask.float()
            use_loss = F.binary_cross_entropy_with_logits(out["use_logits"], batch.y_use, reduction="none")
            use_loss = (use_loss * slot_mask).sum() / slot_mask.sum().clamp_min(1.0)

            used_gold = (batch.y_use > 0.5) & batch.y_slot_mask
            if used_gold.any():
                span_loss = F.cross_entropy(out["span_logits"][used_gold], batch.y_span[used_gold])
                type_loss = F.binary_cross_entropy_with_logits(out["type_logits"][used_gold], batch.y_is_bridge[used_gold])
            else:
                span_loss = torch.zeros((), device=device)
                type_loss = torch.zeros((), device=device)
            loss = (
                args.use_loss_weight * use_loss
                + args.span_loss_weight * span_loss
                + args.type_loss_weight * type_loss
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += float(loss.item())
            batch_count += 1

        val_metrics = compute_metrics(model, val_loader, device)
        e2e_metrics = run_end_to_end_eval(
            model,
            val_rows,
            device,
            batch_size=args.batch_size,
            hash_dim=args.hash_dim,
            cand_emb_cache=args.cand_emb_cache,
            cand_emb_dim=train_ds.cand_emb_dim,
            mem_emb_cache=args.mem_emb_cache,
            mem_emb_dim=train_ds.mem_emb_dim,
            aligner_ckpt_path=args.aligner_checkpoint,
            aligner_spec_emb_cache=args.aligner_spec_emb_cache,
            aligner_cand_emb_cache=args.aligner_cand_emb_cache,
            aligner_mem_emb_cache=args.aligner_mem_emb_cache,
        )
        score = (
            e2e_metrics["row_complete_rate"]
            + 0.5 * val_metrics["slot_row_complete_rate"]
            + 0.25 * val_metrics["span_acc_on_used"]
            + 0.25 * val_metrics["use_acc"]
        )
        payload = {
            "epoch": epoch,
            "train_loss": loss_sum / max(batch_count, 1),
            "val": val_metrics,
            "e2e": e2e_metrics,
            "score": score,
        }
        print(json.dumps(payload, ensure_ascii=False))
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "score": score,
                    "val_metrics": val_metrics,
                    "e2e_metrics": e2e_metrics,
                },
                out_dir / "best_proposer_v1.pt",
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
