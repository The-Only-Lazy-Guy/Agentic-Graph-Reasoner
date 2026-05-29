from __future__ import annotations

"""
train_graph_policy.py

NGR-v0 trainer.

Coverage/novelty patch:
- Supports class-weighted edit loss.
- Default edit weights focus on the weak no_op/add_node boundary.
- Keeps safe_cross_entropy to avoid NaN when a head is unused for a batch.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from graph_core import MemoryGraph, canonical_relation
from graph_policy_env import EDIT_TO_ID, REL_TO_ID, EDIT_TYPES, RELATIONS
from graph_policy_model import Batch, GraphPolicyNet, NODE_TYPE_TO_ID, bow_hash


IGNORE = -100


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_edit_weights(raw: str) -> Dict[str, float]:
    out = {name: 1.0 for name in EDIT_TYPES}
    raw = str(raw or "").strip()
    if not raw:
        return out
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Bad edit weight item: {part!r}; expected name=value")
        k, v = part.split("=", 1)
        k = k.strip()
        if k not in out:
            raise ValueError(f"Unknown edit type in weights: {k!r}")
        out[k] = float(v)
    return out


class GraphPolicyDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        hash_dim: int = 512,
        max_candidates: int = 64,
    ) -> None:
        self.rows = list(rows)
        self.hash_dim = hash_dim
        self.max_candidates = max_candidates
        self._graph_cache: Dict[str, MemoryGraph] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def graph(self, path: str) -> MemoryGraph:
        if path not in self._graph_cache:
            self._graph_cache[path] = MemoryGraph.load_json(path)
        return self._graph_cache[path]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        graph = self.graph(str(row["graph_path"]))
        signal = str(row["signal"])

        cids = [
            str(nid)
            for nid in row.get("candidate_node_ids", [])
            if str(nid) in graph.nodes
        ][: self.max_candidates]

        if not cids:
            cids = list(graph.nodes.keys())[: self.max_candidates]

        id_to_idx = {nid: i for i, nid in enumerate(cids)}

        node_bows = []
        node_scalars = []
        node_types = []

        for nid in cids:
            n = graph.nodes[nid]
            text = f"{nid.replace('_', ' ')} {n.text}"
            node_bows.append(bow_hash(text, self.hash_dim))

            try:
                from graph_core import lexical_overlap
                ov = float(lexical_overlap(signal, text))
            except Exception:
                ov = 0.0

            node_scalars.append(torch.tensor([
                float(getattr(n, "confidence", 0.5)),
                float(getattr(n, "importance", 0.5)),
                ov,
                1.0,
            ], dtype=torch.float32))

            node_types.append(
                NODE_TYPE_TO_ID.get(
                    str(getattr(n, "node_type", "unknown")).lower(),
                    NODE_TYPE_TO_ID["unknown"],
                )
            )

        edge_pairs = []
        edge_rels = []
        try:
            local_edges = graph.iter_local_edges(cids)
        except Exception:
            local_edges = []

        for e in local_edges:
            if e.src in id_to_idx and e.dst in id_to_idx:
                edge_pairs.append([id_to_idx[e.src], id_to_idx[e.dst]])
                edge_rels.append(
                    REL_TO_ID.get(canonical_relation(e.relation), REL_TO_ID["related"])
                )

        gold = row["gold"]
        edit_type = str(gold["edit_type"])
        edit = EDIT_TO_ID[edit_type]

        target = IGNORE
        src = IGNORE
        dst = IGNORE
        rel = IGNORE

        if edit_type in {"update_node", "resolve_conflict", "no_op"}:
            gid = gold.get("target_id")
            if gid in id_to_idx:
                target = id_to_idx[gid]

        elif edit_type == "link_nodes":
            if gold.get("src_id") in id_to_idx:
                src = id_to_idx[gold["src_id"]]
            if gold.get("dst_id") in id_to_idx:
                dst = id_to_idx[gold["dst_id"]]
            rel = REL_TO_ID.get(
                canonical_relation(gold.get("relation", "related")),
                REL_TO_ID["related"],
            )

        elif edit_type == "add_node":
            ev = gold.get("evidence_ids") or []
            for eid in ev:
                if eid in id_to_idx:
                    target = id_to_idx[eid]
                    break
            rel = REL_TO_ID.get(
                canonical_relation(gold.get("relation", "related")),
                REL_TO_ID["related"],
            )

        return {
            "id": row["id"],
            "task_type": row.get("task_type", "unknown"),
            "signal": signal,
            "signal_bow": bow_hash(signal, self.hash_dim),
            "node_bow": torch.stack(node_bows, dim=0),
            "node_scalar": torch.stack(node_scalars, dim=0),
            "node_type_ids": torch.tensor(node_types, dtype=torch.long),
            "edge_index": (
                torch.tensor(edge_pairs, dtype=torch.long)
                if edge_pairs else torch.zeros((0, 2), dtype=torch.long)
            ),
            "edge_rel": (
                torch.tensor(edge_rels, dtype=torch.long)
                if edge_rels else torch.zeros((0,), dtype=torch.long)
            ),
            "candidate_node_ids": cids,
            "y_edit": torch.tensor(edit, dtype=torch.long),
            "y_target": torch.tensor(target, dtype=torch.long),
            "y_src": torch.tensor(src, dtype=torch.long),
            "y_dst": torch.tensor(dst, dtype=torch.long),
            "y_rel": torch.tensor(rel, dtype=torch.long),
        }


def collate(items: Sequence[Mapping[str, Any]]) -> Batch:
    B = len(items)
    max_n = max(x["node_bow"].size(0) for x in items)
    max_e = max(x["edge_index"].size(0) for x in items)
    H = items[0]["signal_bow"].numel()
    S = items[0]["node_scalar"].size(-1)

    signal_bow = torch.stack([x["signal_bow"] for x in items], dim=0)
    node_bow = torch.zeros(B, max_n, H)
    node_scalar = torch.zeros(B, max_n, S)
    node_type_ids = torch.zeros(B, max_n, dtype=torch.long)
    node_mask = torch.zeros(B, max_n, dtype=torch.bool)

    edge_index = torch.zeros(B, max(max_e, 1), 2, dtype=torch.long)
    edge_rel = torch.zeros(B, max(max_e, 1), dtype=torch.long)
    edge_mask = torch.zeros(B, max(max_e, 1), dtype=torch.bool)

    for b, x in enumerate(items):
        n = x["node_bow"].size(0)
        node_bow[b, :n] = x["node_bow"]
        node_scalar[b, :n] = x["node_scalar"]
        node_type_ids[b, :n] = x["node_type_ids"]
        node_mask[b, :n] = True

        e = x["edge_index"].size(0)
        if e:
            edge_index[b, :e] = x["edge_index"]
            edge_rel[b, :e] = x["edge_rel"]
            edge_mask[b, :e] = True

    return Batch(
        signal_bow=signal_bow,
        node_bow=node_bow,
        node_scalar=node_scalar,
        node_type_ids=node_type_ids,
        node_mask=node_mask,
        edge_index=edge_index,
        edge_rel=edge_rel,
        edge_mask=edge_mask,
        y_edit=torch.stack([x["y_edit"] for x in items]),
        y_target=torch.stack([x["y_target"] for x in items]),
        y_src=torch.stack([x["y_src"] for x in items]),
        y_dst=torch.stack([x["y_dst"] for x in items]),
        y_rel=torch.stack([x["y_rel"] for x in items]),
    )


def to_device(batch: Batch, device: torch.device) -> Batch:
    return Batch(**{
        k: getattr(batch, k).to(device)
        for k in batch.__dataclass_fields__
    })


def safe_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    ignore_index: int = IGNORE,
) -> torch.Tensor:
    valid = targets != ignore_index
    if valid.sum().item() == 0:
        return logits.sum() * 0.0

    valid_logits = logits[valid]
    valid_targets = targets[valid]

    in_range = (valid_targets >= 0) & (valid_targets < valid_logits.size(-1))
    if in_range.sum().item() == 0:
        return logits.sum() * 0.0

    return F.cross_entropy(valid_logits[in_range], valid_targets[in_range])


def edit_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    edit_weights: Mapping[str, float],
    device: torch.device,
) -> torch.Tensor:
    weights = torch.ones(len(EDIT_TYPES), dtype=logits.dtype, device=device)
    for name, value in edit_weights.items():
        if name in EDIT_TO_ID:
            weights[EDIT_TO_ID[name]] = float(value)
    return F.cross_entropy(logits, targets, weight=weights)


def supervised_loss(
    out: Mapping[str, torch.Tensor],
    batch: Batch,
    *,
    edit_weights: Mapping[str, float],
) -> tuple[torch.Tensor, Dict[str, float]]:
    device = out["edit_logits"].device

    loss_edit = edit_cross_entropy(out["edit_logits"], batch.y_edit, edit_weights, device)
    loss_target = safe_cross_entropy(out["target_logits"], batch.y_target)
    loss_src = safe_cross_entropy(out["src_logits"], batch.y_src)
    loss_dst = safe_cross_entropy(out["dst_logits"], batch.y_dst)
    loss_relation = safe_cross_entropy(out["relation_logits"], batch.y_rel)

    loss = loss_edit + loss_target + loss_src + loss_dst + loss_relation

    return loss, {
        "loss_edit": float(loss_edit.detach().cpu()),
        "loss_target": float(loss_target.detach().cpu()),
        "loss_src": float(loss_src.detach().cpu()),
        "loss_dst": float(loss_dst.detach().cpu()),
        "loss_relation": float(loss_relation.detach().cpu()),
    }


@torch.no_grad()
def evaluate(model: GraphPolicyNet, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total = edit_ok = 0
    rel_total = rel_ok = 0
    target_total = target_ok = 0
    src_total = src_ok = 0
    dst_total = dst_ok = 0

    for batch in loader:
        batch = to_device(batch, device)
        out = model(batch)

        pred_edit = out["edit_logits"].argmax(-1)
        pred_rel = out["relation_logits"].argmax(-1)
        pred_target = out["target_logits"].argmax(-1)
        pred_src = out["src_logits"].argmax(-1)
        pred_dst = out["dst_logits"].argmax(-1)

        total += batch.y_edit.numel()
        edit_ok += (pred_edit == batch.y_edit).sum().item()

        mask = batch.y_rel != IGNORE
        rel_total += mask.sum().item()
        rel_ok += ((pred_rel == batch.y_rel) & mask).sum().item()

        mask = batch.y_target != IGNORE
        target_total += mask.sum().item()
        target_ok += ((pred_target == batch.y_target) & mask).sum().item()

        mask = batch.y_src != IGNORE
        src_total += mask.sum().item()
        src_ok += ((pred_src == batch.y_src) & mask).sum().item()

        mask = batch.y_dst != IGNORE
        dst_total += mask.sum().item()
        dst_ok += ((pred_dst == batch.y_dst) & mask).sum().item()

    return {
        "edit_acc": edit_ok / max(total, 1),
        "relation_acc": rel_ok / max(rel_total, 1),
        "target_acc": target_ok / max(target_total, 1),
        "src_acc": src_ok / max(src_total, 1),
        "dst_acc": dst_ok / max(dst_total, 1),
    }


def train_sft(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    edit_weights = parse_edit_weights(args.edit_weights)

    train_rows = read_jsonl(args.train_jsonl)
    val_rows = read_jsonl(args.val_jsonl)

    train_ds = GraphPolicyDataset(train_rows, hash_dim=args.hash_dim, max_candidates=args.max_candidates)
    val_ds = GraphPolicyDataset(val_rows, hash_dim=args.hash_dim, max_candidates=args.max_candidates)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0)

    model = GraphPolicyNet(hash_dim=args.hash_dim, hidden_dim=args.hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = -1.0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(json.dumps({
        "device": str(device),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "hash_dim": args.hash_dim,
        "hidden_dim": args.hidden_dim,
        "edit_weights": edit_weights,
    }, indent=2))

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_parts = {
            "loss_edit": 0.0,
            "loss_target": 0.0,
            "loss_src": 0.0,
            "loss_dst": 0.0,
            "loss_relation": 0.0,
        }
        steps = 0
        skipped_nonfinite = 0

        for batch in train_loader:
            batch = to_device(batch, device)
            out = model(batch)
            loss, parts = supervised_loss(out, batch, edit_weights=edit_weights)

            if not torch.isfinite(loss):
                skipped_nonfinite += 1
                continue

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            total_loss += float(loss.detach().cpu())
            for k, v in parts.items():
                total_parts[k] += v
            steps += 1

        metrics = evaluate(model, val_loader, device)
        score = (
            metrics["edit_acc"]
            + 0.40 * metrics["relation_acc"]
            + 0.30 * metrics["target_acc"]
            + 0.15 * metrics["src_acc"]
            + 0.15 * metrics["dst_acc"]
        )

        denom = max(steps, 1)
        log = {
            "epoch": epoch,
            "train_loss": total_loss / denom,
            "skipped_nonfinite_batches": skipped_nonfinite,
            **{k: v / denom for k, v in total_parts.items()},
            **metrics,
            "score": score,
        }
        print(json.dumps(log, indent=2))

        if score > best:
            best = score
            torch.save({
                "model": model.state_dict(),
                "args": vars(args),
                "edit_types": EDIT_TYPES,
                "relations": RELATIONS,
                "best_score": best,
                "edit_weights": edit_weights,
            }, out_dir / "best_graph_policy.pt")

    print(f"saved best checkpoint to {out_dir / 'best_graph_policy.pt'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--val-jsonl", required=True)
    ap.add_argument("--out-dir", default="out_graph_policy_v0")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--hash-dim", type=int, default=512)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--max-candidates", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument(
        "--edit-weights",
        default="no_op=2.50,add_node=1.75,link_nodes=1.00,update_node=0.75,resolve_conflict=0.75",
        help="Comma-separated edit class weights, e.g. no_op=2.5,add_node=1.75",
    )
    args = ap.parse_args()
    train_sft(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
