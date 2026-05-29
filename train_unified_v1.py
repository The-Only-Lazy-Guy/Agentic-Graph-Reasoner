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

from graph_core import MemoryGraph, canonical_relation, lexical_overlap, lexical_tokens
from graph_policy_model import bow_hash
from pred_model import (
    COMMIT_FAMILIES,
    COMMIT_TO_ID,
    EDGE_EXIST_THRESHOLD,
    LOGIT_MASK_VALUE,
    MEM_LINK_KIND_TO_ID,
    REL_WITH_NONE,
    REL_WITH_NONE_TO_ID,
    SPAN_KIND_TO_ID,
)
from synthesize_node_text import _best_matching_memory_id, _memory_text, clean, normalize_text
from train_pred_v1 import (
    decode_edge_predictions,
    decode_mem_kind_predictions,
    load_embedding_cache,
    read_jsonl,
    text_cache_key,
)
from unified_proposal_aligner_model import (
    UNIFIED_NODE_TYPE_TO_ID,
    UnifiedBatch,
    UnifiedProposalAlignerNet,
)


IGNORE = -100
SYNTHESIS_TASKS = {"mixed_add_link", "multi_region_attach"}
EDGE_EXIST_THRESHOLD = 0.5


def build_edge_hard_negative_mask(
    y_edge_exist: torch.Tensor,
    edge_mask: torch.Tensor,
    *,
    max_per_row: int,
) -> torch.Tensor:
    gold = (y_edge_exist > EDGE_EXIST_THRESHOLD) & edge_mask
    reverse = gold.transpose(-1, -2) & ~gold & edge_mask
    transitive = ((gold.float() @ gold.float()) > 0) & ~gold & edge_mask
    candidates = reverse | transitive
    if max_per_row <= 0:
        return torch.zeros_like(candidates, dtype=torch.bool)

    B, S, _ = y_edge_exist.shape
    flat_n = S * S
    if max_per_row >= flat_n:
        return candidates

    priority = torch.full((B, S, S), 2, dtype=torch.long, device=y_edge_exist.device)
    priority[transitive] = 1
    priority[reverse] = 0
    linear = torch.arange(flat_n, device=y_edge_exist.device, dtype=torch.long).view(1, S, S).expand(B, -1, -1)
    key = priority.reshape(B, flat_n) * flat_n + linear.reshape(B, flat_n)
    large_value = flat_n * 3
    large = torch.full_like(key, large_value)
    key = torch.where(candidates.reshape(B, flat_n), key, large)
    topk_k = min(max_per_row, flat_n)
    best_key, best_idx = torch.topk(key, k=topk_k, dim=1, largest=False)
    keep = best_key < large_value
    out = torch.zeros((B, flat_n), dtype=torch.bool, device=y_edge_exist.device)
    out.scatter_(1, best_idx, keep)
    return out.reshape(B, S, S)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(batch: UnifiedBatch, device: torch.device) -> UnifiedBatch:
    return UnifiedBatch(**{k: getattr(batch, k).to(device) for k in batch.__dataclass_fields__})


def goal_commit_family(goal: Mapping[str, Any]) -> str:
    actions = {str(fc.get("action", "")) for fc in goal.get("final_commits", []) or []}
    if "no_op" in actions:
        return "no_op"
    if "add_node" in actions:
        return "add_node"
    return "other"


def build_candidate_memory_ids(row: Mapping[str, Any], graph: MemoryGraph) -> List[str]:
    goal = row.get("_oracle_goal", {}) or {}
    candidate_memory_ids: List[str] = []
    for mem in row.get("initial_memory_node_ids", []) or []:
        mem = str(mem)
        if mem and mem in graph.nodes and mem not in candidate_memory_ids:
            candidate_memory_ids.append(mem)
    for att in goal.get("memory_attachments", []) or []:
        mem = str(att.get("memory_id", ""))
        if mem and mem in graph.nodes and mem not in candidate_memory_ids:
            candidate_memory_ids.append(mem)
    for cov in goal.get("covered_mappings", []) or []:
        mem = str(cov.get("memory_id", ""))
        if mem and mem in graph.nodes and mem not in candidate_memory_ids:
            candidate_memory_ids.append(mem)
    return candidate_memory_ids


MAX_SLOTS = 3


def _ensure_target_slots(row: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Return target_slots from the row, synthesizing from span_oracle if needed."""
    slots = row.get("target_slots") or []
    if slots:
        return list(slots)
    span_oracle = row.get("span_oracle") or []
    goal = row.get("_oracle_goal") or row.get("goal") or {}
    session_nodes = goal.get("session_nodes") or []
    result: List[Dict[str, Any]] = []
    for i, so in enumerate(span_oracle[:MAX_SLOTS]):
        sn = session_nodes[i] if i < len(session_nodes) else {}
        result.append({
            "slot_idx": i,
            "use": True,
            "session_name": str(so.get("session_name", sn.get("name", f"s{i}"))),
            "span_id": str(so.get("best_span_id", "")),
            "span_text": str(so.get("spec_text", sn.get("span_text", ""))),
            "node_type": str(so.get("node_type", sn.get("node_type", "concept"))),
            "confidence": float(so.get("best_score", 0.0)),
        })
    while len(result) < MAX_SLOTS:
        result.append({
            "slot_idx": len(result),
            "use": False,
            "session_name": f"pad_{len(result)}",
            "span_id": "",
            "span_text": "",
            "node_type": "concept",
            "confidence": 0.0,
        })
    return result


def canonical_slot_text_by_span_id(row: Mapping[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for slot in _ensure_target_slots(row):
        if not bool(slot.get("use")):
            continue
        span_id = str(slot.get("span_id", ""))
        span_text = slot.get("span_text")
        if span_id and span_text is not None and span_id not in out:
            out[span_id] = str(span_text)
    return out


class UnifiedDataset(Dataset):
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
        goal = row.get("_oracle_goal") or row.get("goal") or {}
        target_slots = _ensure_target_slots(row)
        spans = row.get("spans", []) or []
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
                cand_emb.append(torch.zeros(self.cand_emb_dim, dtype=torch.float32) if emb is None else torch.from_numpy(emb.astype("float32")))
            cand_kind_ids.append(SPAN_KIND_TO_ID.get(str(span.get("span_kind", "unknown")), SPAN_KIND_TO_ID["unknown"]))
            start = float(span.get("start", 0))
            end = float(span.get("end", 0))
            cand_feat.append(
                torch.tensor([start / max(len(signal), 1), end / max(len(signal), 1)], dtype=torch.float32)
            )
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

        candidate_memory_ids = build_candidate_memory_ids(row, graph)

        mem_bow = []
        mem_emb = []
        mem_feat = []
        init_set = {str(x) for x in row.get("initial_memory_node_ids", []) or []}
        for j, mem_id in enumerate(candidate_memory_ids):
            text = str(graph.nodes[mem_id].text)
            mem_bow.append(bow_hash(text, self.hash_dim))
            if self.mem_emb_dim > 0:
                emb = self.mem_emb_cache.get(text_cache_key(text))
                mem_emb.append(torch.zeros(self.mem_emb_dim, dtype=torch.float32) if emb is None else torch.from_numpy(emb.astype("float32")))
            mem_feat.append(
                torch.tensor(
                    [
                        float(lexical_overlap(signal, text)),
                        1.0 if mem_id in init_set else 0.0,
                        float(j) / max(len(candidate_memory_ids), 1),
                    ],
                    dtype=torch.float32,
                )
            )

        K = len(target_slots)
        y_use = []
        y_span = []
        y_is_bridge = []
        gold_name_to_slot: Dict[str, int] = {}
        gold_slot_texts: List[str] = []
        for k, slot in enumerate(target_slots):
            use = bool(slot.get("use"))
            y_use.append(1.0 if use else 0.0)
            if use:
                y_span.append(span_id_to_idx.get(str(slot.get("span_id", "")), IGNORE))
                is_bridge = str(slot.get("node_type", "concept")) == "bridge"
                y_is_bridge.append(1.0 if is_bridge else 0.0)
                gold_name_to_slot[str(slot.get("session_name", ""))] = k
                gold_slot_texts.append(str(slot.get("span_text", "") or ""))
            else:
                y_span.append(IGNORE)
                y_is_bridge.append(0.0)
                gold_slot_texts.append("")

        mem_pair_feat = []
        for slot in target_slots:
            slot_text = str(slot.get("span_text", "") or "") if bool(slot.get("use")) else ""
            overlaps = [float(lexical_overlap(str(graph.nodes[mid].text), slot_text)) for mid in candidate_memory_ids]
            best_val = max(overlaps) if overlaps else 0.0
            per_mem = [
                torch.tensor([ov, ov / best_val if best_val > 0.0 else 0.0], dtype=torch.float32)
                for ov in overlaps
            ]
            mem_pair_feat.append(
                torch.stack(per_mem, dim=0) if per_mem else torch.zeros((0, 2), dtype=torch.float32)
            )

        slot_ids = list(range(K))

        y_edge_exist = torch.zeros((K, K), dtype=torch.float32)
        y_edge_rel = torch.full((K, K), IGNORE, dtype=torch.long)
        for i in range(K):
            y_edge_rel[i, i] = IGNORE
        for edge in goal.get("session_edges", []) or []:
            src = gold_name_to_slot.get(str(edge.get("src", "")))
            dst = gold_name_to_slot.get(str(edge.get("dst", "")))
            if src is None or dst is None or src == dst:
                continue
            y_edge_exist[src, dst] = 1.0
            y_edge_rel[src, dst] = REL_WITH_NONE_TO_ID[canonical_relation(str(edge.get("relation", "related")))]
        for i in range(K):
            for j in range(K):
                if i != j and y_edge_rel[i, j] == IGNORE:
                    y_edge_rel[i, j] = REL_WITH_NONE_TO_ID["none"]

        M = len(candidate_memory_ids)
        y_mem_kind = torch.zeros((K, M), dtype=torch.long)
        y_mem_rel = torch.full((K, M), REL_WITH_NONE_TO_ID["none"], dtype=torch.long)
        for att in goal.get("memory_attachments", []) or []:
            sidx = gold_name_to_slot.get(str(att.get("session", "")))
            try:
                midx = candidate_memory_ids.index(str(att.get("memory_id", "")))
            except ValueError:
                midx = None
            if sidx is None or midx is None:
                continue
            y_mem_kind[sidx, midx] = MEM_LINK_KIND_TO_ID["attach"]
            y_mem_rel[sidx, midx] = REL_WITH_NONE_TO_ID[canonical_relation(str(att.get("relation", "related")))]

        session_text_by_name = {str(node.get("name", "")): str(node.get("span_text", "")) for node in goal.get("session_nodes", []) or []}
        for cov in goal.get("covered_mappings", []) or []:
            memory_id = str(cov.get("memory_id", ""))
            span_text = str(cov.get("span_text", ""))
            sidx = None
            for name, text in session_text_by_name.items():
                if text == span_text:
                    sidx = gold_name_to_slot.get(name)
                    break
            try:
                midx = candidate_memory_ids.index(memory_id)
            except ValueError:
                midx = None
            if sidx is None or midx is None:
                continue
            y_mem_kind[sidx, midx] = MEM_LINK_KIND_TO_ID["cover"]

        y_mixed_dst_mem = torch.full((K,), IGNORE, dtype=torch.long)
        y_bridge_mem_a = torch.full((K,), IGNORE, dtype=torch.long)
        y_bridge_mem_b = torch.full((K,), IGNORE, dtype=torch.long)

        task_type = str(row.get("task_type", ""))
        if task_type == "mixed_add_link" and "new_note" in gold_name_to_slot and candidate_memory_ids:
            y_mixed_dst_mem[gold_name_to_slot["new_note"]] = 0
        elif task_type == "multi_region_attach" and "bridge" in gold_name_to_slot and len(candidate_memory_ids) >= 2:
            support_text = str(session_text_by_name.get("support_note", ""))
            support_mem = _best_matching_memory_id(candidate_memory_ids, support_text, graph)
            if support_mem is not None:
                support_idx = candidate_memory_ids.index(str(support_mem))
                other_idx = next((i for i, mid in enumerate(candidate_memory_ids) if mid != str(support_mem)), None)
                if other_idx is not None:
                    bidx = gold_name_to_slot["bridge"]
                    y_bridge_mem_a[bidx] = support_idx
                    y_bridge_mem_b[bidx] = other_idx

        feat_dim = 5
        edge_pair_feat = torch.zeros((K, K, feat_dim), dtype=torch.float32)
        slot_token_sets = [lexical_tokens(t) for t in gold_slot_texts]
        for i in range(K):
            for j in range(K):
                if i == j:
                    continue
                ti = slot_token_sets[i]
                tj = slot_token_sets[j]
                if not ti or not tj:
                    continue
                inter = len(ti & tj)
                union = len(ti | tj)
                edge_pair_feat[i, j, 0] = inter / union if union > 0 else 0.0
                edge_pair_feat[i, j, 1] = inter / len(ti)
                edge_pair_feat[i, j, 2] = inter / len(tj)
                li = len(gold_slot_texts[i])
                lj = len(gold_slot_texts[j])
                edge_pair_feat[i, j, 3] = li / max(lj, 1)
                edge_pair_feat[i, j, 4] = float(i - j) / max(K, 1)

        return {
            "id": row.get("id", ""),
            "task_type": task_type,
            "row": row,
            "signal_bow": signal_bow,
            "cand_bow": torch.stack(cand_bow, dim=0) if cand_bow else torch.zeros((0, self.hash_dim), dtype=torch.float32),
            "cand_emb": torch.stack(cand_emb, dim=0) if cand_emb else torch.zeros((len(spans), self.cand_emb_dim), dtype=torch.float32),
            "cand_kind_ids": torch.tensor(cand_kind_ids, dtype=torch.long) if cand_kind_ids else torch.zeros((0,), dtype=torch.long),
            "cand_feat": torch.stack(cand_feat, dim=0) if cand_feat else torch.zeros((0, 2), dtype=torch.float32),
            "cand_pair_feat": torch.stack(cand_pair_feat, dim=0) if cand_pair_feat else torch.zeros((0, 3), dtype=torch.float32),
            "mem_bow": torch.stack(mem_bow, dim=0) if mem_bow else torch.zeros((0, self.hash_dim), dtype=torch.float32),
            "mem_emb": torch.stack(mem_emb, dim=0) if mem_emb else torch.zeros((len(candidate_memory_ids), self.mem_emb_dim), dtype=torch.float32),
            "mem_feat": torch.stack(mem_feat, dim=0) if mem_feat else torch.zeros((0, 3), dtype=torch.float32),
            "mem_pair_feat": torch.stack(mem_pair_feat, dim=0) if mem_pair_feat else torch.zeros((0, 0, 2), dtype=torch.float32),
            "slot_ids": torch.tensor(slot_ids, dtype=torch.long),
            "y_use": torch.tensor(y_use, dtype=torch.float32),
            "y_span": torch.tensor(y_span, dtype=torch.long),
            "y_is_bridge": torch.tensor(y_is_bridge, dtype=torch.float32),
            "y_commit": torch.tensor(COMMIT_TO_ID[goal_commit_family(goal)], dtype=torch.long),
            "y_edge_exist": y_edge_exist,
            "y_edge_rel": y_edge_rel,
            "y_mem_kind": y_mem_kind,
            "y_mem_rel": y_mem_rel,
            "y_mixed_dst_mem": y_mixed_dst_mem,
            "y_bridge_mem_a": y_bridge_mem_a,
            "y_bridge_mem_b": y_bridge_mem_b,
            "edge_pair_feat": edge_pair_feat,
            "memory_ids": candidate_memory_ids,
        }


def collate(batch: Sequence[Mapping[str, Any]]) -> tuple[UnifiedBatch, List[Mapping[str, Any]]]:
    B = len(batch)
    H = batch[0]["signal_bow"].numel()
    max_c = max(max(x["cand_bow"].size(0), 1) for x in batch)
    max_m = max(max(x["mem_bow"].size(0), 1) for x in batch)
    K = max(x["y_use"].size(0) for x in batch)
    cand_emb_dim = batch[0]["cand_emb"].size(1) if batch[0]["cand_emb"].ndim == 2 else 0
    mem_emb_dim = batch[0]["mem_emb"].size(1) if batch[0]["mem_emb"].ndim == 2 else 0

    signal_bow = torch.zeros((B, H), dtype=torch.float32)
    cand_bow = torch.zeros((B, max_c, H), dtype=torch.float32)
    cand_emb = torch.zeros((B, max_c, cand_emb_dim), dtype=torch.float32)
    cand_kind_ids = torch.zeros((B, max_c), dtype=torch.long)
    cand_feat = torch.zeros((B, max_c, 2), dtype=torch.float32)
    cand_pair_feat = torch.zeros((B, max_c, 3), dtype=torch.float32)
    cand_mask = torch.zeros((B, max_c), dtype=torch.bool)
    mem_bow = torch.zeros((B, max_m, H), dtype=torch.float32)
    mem_emb = torch.zeros((B, max_m, mem_emb_dim), dtype=torch.float32)
    mem_feat = torch.zeros((B, max_m, 3), dtype=torch.float32)
    mem_mask = torch.zeros((B, max_m), dtype=torch.bool)
    mem_pair_feat = torch.zeros((B, K, max_m, 2), dtype=torch.float32)
    edge_pair_feat = torch.zeros((B, K, K, 5), dtype=torch.float32)
    slot_ids = torch.zeros((B, K), dtype=torch.long)
    slot_mask = torch.zeros((B, K), dtype=torch.bool)
    y_use = torch.zeros((B, K), dtype=torch.float32)
    y_span = torch.full((B, K), IGNORE, dtype=torch.long)
    y_is_bridge = torch.zeros((B, K), dtype=torch.float32)
    y_commit = torch.zeros((B,), dtype=torch.long)
    y_edge_exist = torch.zeros((B, K, K), dtype=torch.float32)
    y_edge_rel = torch.full((B, K, K), IGNORE, dtype=torch.long)
    y_mem_kind = torch.zeros((B, K, max_m), dtype=torch.long)
    y_mem_rel = torch.full((B, K, max_m), REL_WITH_NONE_TO_ID["none"], dtype=torch.long)
    y_mixed_dst_mem = torch.full((B, K), IGNORE, dtype=torch.long)
    y_bridge_mem_a = torch.full((B, K), IGNORE, dtype=torch.long)
    y_bridge_mem_b = torch.full((B, K), IGNORE, dtype=torch.long)
    rows: List[Mapping[str, Any]] = []

    for b, x in enumerate(batch):
        signal_bow[b] = x["signal_bow"]
        c = x["cand_bow"].size(0)
        if c:
            cand_bow[b, :c] = x["cand_bow"]
            if cand_emb_dim:
                cand_emb[b, :c] = x["cand_emb"]
            cand_kind_ids[b, :c] = x["cand_kind_ids"]
            cand_feat[b, :c] = x["cand_feat"]
            cand_pair_feat[b, :c] = x["cand_pair_feat"]
            cand_mask[b, :c] = True
        m = x["mem_bow"].size(0)
        if m:
            mem_bow[b, :m] = x["mem_bow"]
            if mem_emb_dim:
                mem_emb[b, :m] = x["mem_emb"]
            mem_feat[b, :m] = x["mem_feat"]
            mem_mask[b, :m] = True
            mem_pair_feat[b, : x["y_use"].size(0), :m] = x["mem_pair_feat"]
        k = x["y_use"].size(0)
        slot_ids[b, :k] = x["slot_ids"]
        slot_mask[b, :k] = True
        y_use[b, :k] = x["y_use"]
        y_span[b, :k] = x["y_span"]
        y_is_bridge[b, :k] = x["y_is_bridge"]
        y_commit[b] = x["y_commit"]
        y_edge_exist[b, :k, :k] = x["y_edge_exist"]
        y_edge_rel[b, :k, :k] = x["y_edge_rel"]
        edge_pair_feat[b, :k, :k] = x["edge_pair_feat"]
        if m:
            y_mem_kind[b, :k, :m] = x["y_mem_kind"]
            y_mem_rel[b, :k, :m] = x["y_mem_rel"]
        y_mixed_dst_mem[b, :k] = x["y_mixed_dst_mem"]
        y_bridge_mem_a[b, :k] = x["y_bridge_mem_a"]
        y_bridge_mem_b[b, :k] = x["y_bridge_mem_b"]
        rows.append(x["row"])

    unified_batch = UnifiedBatch(
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
        mem_pair_feat=mem_pair_feat,
        slot_ids=slot_ids,
        slot_mask=slot_mask,
        y_use=y_use,
        y_span=y_span,
        y_is_bridge=y_is_bridge,
        y_commit=y_commit,
        y_edge_exist=y_edge_exist,
        y_edge_rel=y_edge_rel,
        y_mem_kind=y_mem_kind,
        y_mem_rel=y_mem_rel,
        y_mixed_dst_mem=y_mixed_dst_mem,
        y_bridge_mem_a=y_bridge_mem_a,
        y_bridge_mem_b=y_bridge_mem_b,
        edge_pair_feat=edge_pair_feat,
    )
    return unified_batch, rows


def loss_on_valid_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    ignore_index: int = IGNORE,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    valid = targets != ignore_index
    if not bool(valid.any().item()):
        return torch.zeros((), device=logits.device)
    return F.cross_entropy(logits, targets, ignore_index=ignore_index, weight=weight)


def compute_mem_rel_class_weights(
    rows: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    min_weight: float,
    max_weight: float,
) -> torch.Tensor:
    """Inverse-frequency weights for memory_attachments relations.

    Counts gold relations from `_oracle_goal.memory_attachments` across the rows
    and produces a per-class weight clamped to [min_weight, max_weight]. The
    "none" class is held at 1.0 (it never participates in attach supervision).

    The capped range is intentionally conservative: PRED-v2 fix13 used min=0.25
    / max=6.0 and over-corrected toward rare classes. Defaults here are tighter
    so common classes are mildly de-emphasised without rare classes being
    blasted upward.
    """
    counts = torch.zeros(len(REL_WITH_NONE), dtype=torch.float32)
    for row in rows:
        goal = row.get("_oracle_goal", {}) or {}
        for att in goal.get("memory_attachments", []) or []:
            rel = canonical_relation(str(att.get("relation", "related")))
            rid = REL_WITH_NONE_TO_ID.get(rel)
            if rid is not None:
                counts[rid] += 1.0

    weights = torch.ones(len(REL_WITH_NONE), dtype=torch.float32)
    present = counts > 0
    if present.any():
        total = counts[present].sum()
        n_present = float(present.sum().item())
        weights[present] = total / (n_present * counts[present])
        weights = weights.clamp(min=float(min_weight), max=float(max_weight))
    weights[REL_WITH_NONE_TO_ID["none"]] = 1.0
    return weights.to(device)


def compute_edge_rel_class_weights(
    rows: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    min_weight: float,
    max_weight: float,
) -> torch.Tensor:
    """Inverse-frequency weights for session-edge relations.

    Up-weights rare relations like `depend`, `part_of`, `contradict`, and
    `example_of` which are under-represented in training vs the dominant
    `support` / `related` classes.  The "none" class is held at 1.0.
    """
    counts = torch.zeros(len(REL_WITH_NONE), dtype=torch.float32)
    for row in rows:
        goal = row.get("_oracle_goal") or row.get("goal") or {}
        for edge in goal.get("session_edges", []) or []:
            rel = canonical_relation(str(edge.get("relation", "related")))
            rid = REL_WITH_NONE_TO_ID.get(rel)
            if rid is not None:
                counts[rid] += 1.0

    weights = torch.ones(len(REL_WITH_NONE), dtype=torch.float32)
    present = counts > 0
    if present.any():
        total = counts[present].sum()
        n_present = float(present.sum().item())
        weights[present] = total / (n_present * counts[present])
        weights = weights.clamp(min=float(min_weight), max=float(max_weight))
    weights[REL_WITH_NONE_TO_ID["none"]] = 1.0
    return weights.to(device)


def compute_loss(
    batch: UnifiedBatch,
    out: Dict[str, torch.Tensor],
    *,
    use_weight: float = 0.5,
    type_weight: float = 0.25,
    commit_weight: float = 0.25,
    edge_exist_weight: float = 1.0,
    edge_rel_weight: float = 0.5,
    mem_kind_weight: float = 0.5,
    mem_rel_weight: float = 0.25,
    mixed_dst_weight: float = 0.5,
    bridge_a_weight: float = 0.5,
    bridge_b_weight: float = 0.5,
    span_weight: float = 1.0,
    mem_rel_class_weight: torch.Tensor | None = None,
    edge_rel_class_weight: torch.Tensor | None = None,
    edge_hard_neg_weight: float = 1.0,
    edge_hard_neg_max_per_row: int = 6,
    verifier_weight: float = 0.0,
    counterfactual_rate: float = 0.0,
) -> tuple[torch.Tensor, Dict[str, float]]:
    slot_valid = batch.slot_mask
    use_loss = F.binary_cross_entropy_with_logits(out["use_logits"][slot_valid], batch.y_use[slot_valid])

    used_gold = (batch.y_use > 0.5) & slot_valid
    type_loss = F.binary_cross_entropy_with_logits(
        out["type_logits"][used_gold],
        batch.y_is_bridge[used_gold],
    ) if used_gold.any() else torch.zeros((), device=out["use_logits"].device)

    span_logits = out["span_logits"].reshape(-1, out["span_logits"].size(-1))
    span_targets = batch.y_span.reshape(-1)
    span_loss = F.cross_entropy(span_logits, span_targets, ignore_index=IGNORE)

    commit_loss = F.cross_entropy(out["commit_logits"], batch.y_commit)

    edge_mask = slot_valid[:, :, None] & slot_valid[:, None, :]
    diag = torch.eye(slot_valid.size(1), device=slot_valid.device, dtype=torch.bool)[None, :, :]
    edge_mask = edge_mask & ~diag
    edge_weight = torch.ones_like(batch.y_edge_exist)
    if edge_hard_neg_weight != 1.0 and edge_hard_neg_max_per_row > 0:
        hard_neg_mask = build_edge_hard_negative_mask(
            batch.y_edge_exist,
            edge_mask,
            max_per_row=edge_hard_neg_max_per_row,
        )
        edge_weight[hard_neg_mask] = float(edge_hard_neg_weight)
    edge_logits = out["edge_exist_logits"][edge_mask]
    edge_targets = batch.y_edge_exist[edge_mask]
    pos = edge_targets.sum().item()
    neg = edge_targets.numel() - pos
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=edge_logits.device)
    edge_exist_loss = F.binary_cross_entropy_with_logits(
        edge_logits, edge_targets,
        weight=edge_weight[edge_mask],
        pos_weight=pos_weight,
    )

    gold_edge = (batch.y_edge_exist > 0.0) & edge_mask
    edge_rel_loss = loss_on_valid_ce(
        out["edge_rel_logits"][gold_edge],
        batch.y_edge_rel[gold_edge],
        weight=edge_rel_class_weight,
    ) if gold_edge.any() else torch.zeros((), device=edge_logits.device)

    verifier_loss = torch.zeros((), device=edge_logits.device)
    if verifier_weight > 0 and "verifier_logits" in out:
        all_pairs = edge_mask
        verifier_targets = batch.y_edge_rel.clone()
        if counterfactual_rate > 0:
            B = verifier_targets.size(0)
            shuffle_mask = torch.rand(B, device=verifier_targets.device) < counterfactual_rate
            if shuffle_mask.any():
                for b in shuffle_mask.nonzero(as_tuple=True)[0]:
                    gold_mask = (batch.y_edge_exist[b] > 0.0) & edge_mask[b]
                    indices = gold_mask.nonzero()
                    if indices.size(0) >= 2:
                        deltas = indices[:, 1] - indices[:, 0]
                        unique_deltas = deltas.unique()
                        for d in unique_deltas:
                            mask = deltas == d
                            if mask.sum() >= 2:
                                d_idx = indices[mask]
                                d_rels = verifier_targets[b, d_idx[:, 0], d_idx[:, 1]]
                                perm = torch.randperm(d_rels.size(0), device=d_rels.device)
                                verifier_targets[b, d_idx[:, 0], d_idx[:, 1]] = d_rels[perm]
        verifier_loss = loss_on_valid_ce(
            out["verifier_logits"][all_pairs].reshape(-1, out["verifier_logits"].size(-1)),
            verifier_targets[all_pairs].reshape(-1),
            weight=edge_rel_class_weight,
        ) if all_pairs.any() else torch.zeros((), device=edge_logits.device)

    mem_valid = batch.mem_mask[:, None, :].expand_as(batch.y_mem_kind)
    mem_kind_logits = out["mem_kind_logits"][mem_valid]
    mem_kind_targets = batch.y_mem_kind[mem_valid]
    if mem_kind_targets.numel() > 0:
        class_weight = torch.tensor([1.0, 4.0, 4.0], device=mem_kind_logits.device)
        mem_kind_loss = F.cross_entropy(mem_kind_logits, mem_kind_targets, weight=class_weight)
    else:
        mem_kind_loss = torch.zeros((), device=edge_logits.device)

    gold_attach = (batch.y_mem_kind == MEM_LINK_KIND_TO_ID["attach"]) & mem_valid
    mem_rel_loss = loss_on_valid_ce(
        out["mem_rel_logits"][gold_attach],
        batch.y_mem_rel[gold_attach],
        weight=mem_rel_class_weight,
    ) if gold_attach.any() else torch.zeros((), device=edge_logits.device)

    mixed_dst_loss = loss_on_valid_ce(
        out["mixed_dst_mem_logits"].reshape(-1, out["mixed_dst_mem_logits"].size(-1)),
        batch.y_mixed_dst_mem.reshape(-1),
    )
    bridge_a_loss = loss_on_valid_ce(
        out["bridge_mem_a_logits"].reshape(-1, out["bridge_mem_a_logits"].size(-1)),
        batch.y_bridge_mem_a.reshape(-1),
    )
    bridge_b_loss = loss_on_valid_ce(
        out["bridge_mem_b_logits"].reshape(-1, out["bridge_mem_b_logits"].size(-1)),
        batch.y_bridge_mem_b.reshape(-1),
    )

    total = (
        span_weight * span_loss
        + use_weight * use_loss
        + type_weight * type_loss
        + commit_weight * commit_loss
        + edge_exist_weight * edge_exist_loss
        + edge_rel_weight * edge_rel_loss
        + mem_kind_weight * mem_kind_loss
        + mem_rel_weight * mem_rel_loss
        + mixed_dst_weight * mixed_dst_loss
        + bridge_a_weight * bridge_a_loss
        + bridge_b_weight * bridge_b_loss
        + verifier_weight * verifier_loss
    )
    parts = {
        "span": float(span_loss.item()),
        "use": float(use_loss.item()),
        "type": float(type_loss.item()),
        "commit": float(commit_loss.item()),
        "edge_exist": float(edge_exist_loss.item()),
        "edge_rel": float(edge_rel_loss.item()),
        "verifier": float(verifier_loss.item()),
        "mem_kind": float(mem_kind_loss.item()),
        "mem_rel": float(mem_rel_loss.item()),
        "mixed_dst": float(mixed_dst_loss.item()),
        "bridge_a": float(bridge_a_loss.item()),
        "bridge_b": float(bridge_b_loss.item()),
    }
    return total, parts


def build_predicted_session_nodes(
    row: Mapping[str, Any],
    *,
    use_pred: torch.Tensor,
    span_pred: torch.Tensor,
    bridge_pred: torch.Tensor,
    mixed_dst_pred: torch.Tensor,
    bridge_a_pred: torch.Tensor,
    bridge_b_pred: torch.Tensor,
    memory_ids: Sequence[str],
    graph: MemoryGraph,
) -> List[Dict[str, Any]]:
    spans = row.get("spans", []) or []
    target_slots = _ensure_target_slots(row)
    pred_nodes: List[Dict[str, Any]] = []
    slot_text_by_span_id = canonical_slot_text_by_span_id(row)
    slot_idx_by_name = {
        str(template.get("session_name", f"slot_{k}")): k
        for k, template in enumerate(target_slots)
        if template.get("session_name") is not None
    }

    base_nodes: Dict[str, Dict[str, Any]] = {}
    for k, template in enumerate(target_slots):
        if not bool(use_pred[k].item()):
            continue
        pred_idx = int(span_pred[k].item())
        span = spans[pred_idx] if 0 <= pred_idx < len(spans) else {"id": None, "text": ""}
        span_id = str(span.get("id", "")) if span.get("id") is not None else ""
        name = str(template.get("session_name", f"slot_{k}"))
        node_type = "bridge" if bool(bridge_pred[k].item()) else "concept"
        base_nodes[name] = {
            "name": name,
            "node_type": node_type,
            "span_id": span.get("id"),
            "span_text": slot_text_by_span_id.get(span_id, str(span.get("text", ""))),
        }

    task_type = str(row.get("task_type", ""))
    if task_type == "mixed_add_link" and "new_note" in base_nodes and memory_ids:
        dst_slot_idx = slot_idx_by_name.get("new_note", 0)
        dst_idx = int(mixed_dst_pred[dst_slot_idx].item())
        dst_idx = min(max(dst_idx, 0), len(memory_ids) - 1)
        dst_text = _memory_text(graph, memory_ids[dst_idx], 110)
        note_text = str(base_nodes["new_note"].get("span_text", ""))
        base_nodes["new_note"]["span_text"] = clean(
            f"{note_text} (related to {dst_text})",
            220,
        )
    elif task_type == "multi_region_attach" and "bridge" in base_nodes and len(memory_ids) >= 2:
        bridge_slot_idx = slot_idx_by_name.get("bridge")
        if bridge_slot_idx is not None:
            a_idx = min(max(int(bridge_a_pred[bridge_slot_idx].item()), 0), len(memory_ids) - 1)
            b_idx = min(max(int(bridge_b_pred[bridge_slot_idx].item()), 0), len(memory_ids) - 1)
            text_a = _memory_text(graph, memory_ids[a_idx], 90)
            text_b = _memory_text(graph, memory_ids[b_idx], 90)
            base_nodes["bridge"]["span_text"] = clean(
                f"{text_a} and {text_b} are connected by a shared bridge concept.",
                180,
            )

    for template in target_slots:
        name = str(template.get("session_name", ""))
        if name in base_nodes:
            pred_nodes.append(base_nodes[name])
    return pred_nodes


def derive_gold_targets(row: Mapping[str, Any]) -> Dict[str, Any]:
    goal = row.get("_oracle_goal") or row.get("goal") or {}
    gold_nodes = goal.get("session_nodes", []) or []
    gold_text_by_name = {str(node.get("name", "")): str(node.get("span_text", "")) for node in gold_nodes}
    _target_slots = row.get("target_slots") or []
    if _target_slots:
        gold_span_by_name = {
            str(slot.get("session_name", "")): str(slot.get("span_id", ""))
            for slot in _target_slots
            if bool(slot.get("use")) and slot.get("span_id") is not None
        }
    else:
        gold_span_by_name = {
            str(so.get("session_name", "")): str(so.get("best_span_id", ""))
            for so in row.get("span_oracle", []) or []
            if so.get("best_span_id") is not None
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
    for cov in goal.get("covered_mappings", []) or []:
        span_text = str(cov.get("span_text", ""))
        memory_id = str(cov.get("memory_id", ""))
        for name, text in session_text_by_name.items():
            if text == span_text:
                gold_cover.add((name, memory_id))
                break
    return {
        "gold_names": {str(node.get("name", "")) for node in gold_nodes},
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


def oracle_prediction_tensors(row: Mapping[str, Any], memory_ids: Sequence[str], graph: MemoryGraph) -> Dict[str, torch.Tensor]:
    target_slots = _ensure_target_slots(row)
    K = len(target_slots)
    use = torch.tensor([bool(slot.get("use")) for slot in target_slots], dtype=torch.bool)
    span = torch.tensor(
        [
            next(
                (i for i, s in enumerate(row.get("spans", []) or []) if str(s.get("id", "")) == str(slot.get("span_id", ""))),
                0,
            )
            for slot in target_slots
        ],
        dtype=torch.long,
    )
    bridge = torch.tensor([str(slot.get("node_type", "concept")) == "bridge" for slot in target_slots], dtype=torch.bool)
    mixed_dst = torch.zeros((K,), dtype=torch.long)
    bridge_a = torch.zeros((K,), dtype=torch.long)
    bridge_b = torch.zeros((K,), dtype=torch.long)
    goal = row.get("_oracle_goal") or row.get("goal") or {}
    goal_nodes = {str(node.get("name", "")): str(node.get("span_text", "")) for node in goal.get("session_nodes", []) or []}
    slot_idx_by_name = {
        str(slot.get("session_name", f"slot_{k}")): k
        for k, slot in enumerate(target_slots)
        if slot.get("session_name") is not None
    }
    if str(row.get("task_type", "")) == "mixed_add_link" and memory_ids and "new_note" in slot_idx_by_name:
        mixed_dst[slot_idx_by_name["new_note"]] = 0
    elif str(row.get("task_type", "")) == "multi_region_attach" and len(memory_ids) >= 2 and "bridge" in slot_idx_by_name:
        support_text = goal_nodes.get("support_note", "")
        support_mem = _best_matching_memory_id(memory_ids, support_text, graph)
        if support_mem is not None:
            support_idx = memory_ids.index(str(support_mem))
            other_idx = next((i for i, mid in enumerate(memory_ids) if mid != str(support_mem)), support_idx)
            bridge_idx = slot_idx_by_name["bridge"]
            bridge_a[bridge_idx] = support_idx
            bridge_b[bridge_idx] = other_idx
    return {
        "use": use,
        "span": span,
        "bridge": bridge,
        "mixed_dst": mixed_dst,
        "bridge_a": bridge_a,
        "bridge_b": bridge_b,
    }


def compute_metrics(model: UnifiedProposalAlignerNet, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    model.eval()
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
                gold = derive_gold_targets(row)
                graph = loader.dataset.graph(str(row.get("graph_path", "")))  # type: ignore[attr-defined]
                candidate_memory_ids = build_candidate_memory_ids(row, graph)

                pred_nodes = build_predicted_session_nodes(
                    row,
                    use_pred=use_pred[b].cpu(),
                    span_pred=span_pred[b].cpu(),
                    bridge_pred=bridge_pred[b].cpu(),
                    mixed_dst_pred=mixed_dst_pred[b].cpu(),
                    bridge_a_pred=bridge_a_pred[b].cpu(),
                    bridge_b_pred=bridge_b_pred[b].cpu(),
                    memory_ids=candidate_memory_ids,
                    graph=graph,
                )
                pred_names = [str(node.get("name", "")) for node in pred_nodes]
                pred_name_set = set(pred_names)
                pred_text_by_name = {str(node.get("name", "")): str(node.get("span_text", "")) for node in pred_nodes}
                _row_slots = _ensure_target_slots(row)
                pred_name_to_slot = {str(_row_slots[k].get("session_name", f"slot_{k}")): k for k in range(len(_row_slots))}

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
                    if normalize_text(pred_text_by_name.get(name, "")) == normalize_text(gold["gold_text_by_name"].get(name, "")):
                        text_ok_count += 1

                span_total += span_total_row
                span_ok += span_ok_count
                text_total += span_total_row
                text_ok += text_ok_count

                predicted_edges = set()
                for i, src_name in enumerate(pred_names):
                    src_slot = pred_name_to_slot.get(src_name)
                    if src_slot is None:
                        continue
                    for j, dst_name in enumerate(pred_names):
                        if i == j:
                            continue
                        dst_slot = pred_name_to_slot.get(dst_name)
                        if dst_slot is None:
                            continue
                        if bool(edge_exist_pred[b, src_slot, dst_slot].item()):
                            rel_id = int(edge_rel_pred[b, src_slot, dst_slot].item())
                            rel = REL_WITH_NONE[rel_id]
                            if rel != "none":
                                predicted_edges.add((src_name, dst_name, canonical_relation(rel)))
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
                    for m_idx, mem_id in enumerate(candidate_memory_ids):
                        kind_id = int(mem_kind_pred[b, slot_idx, m_idx].item())
                        if kind_id == MEM_LINK_KIND_TO_ID["attach"]:
                            rel_id = int(mem_rel_pred[b, slot_idx, m_idx].item())
                            predicted_attach.add((name, str(mem_id), canonical_relation(REL_WITH_NONE[rel_id])))
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

                row_complete = (
                    pred_name_set == gold_names
                    and bool(use_match[batch.slot_mask[b]].all().item())
                    and span_ok_count == span_total_row
                    and commit_ok_row
                    and len(predicted_edges - gold["gold_edges"]) == 0
                    and len(gold["gold_edges"] - predicted_edges) == 0
                    and len(predicted_attach - gold["gold_attach"]) == 0
                    and len(gold["gold_attach"] - predicted_attach) == 0
                    and len(predicted_cover - gold["gold_cover"]) == 0
                    and len(gold["gold_cover"] - predicted_cover) == 0
                )
                text_faithful_row_complete = row_complete and text_ok_count == span_total_row
                row_total += 1
                row_ok += int(row_complete)
                text_row_ok += int(text_faithful_row_complete)

    edge_precision, edge_recall, edge_f1 = aggregate_binary_metrics(edge_tp, edge_fp, edge_fn)
    attach_precision, attach_recall, attach_f1 = aggregate_binary_metrics(attach_tp, attach_fp, attach_fn)
    cover_precision, cover_recall, cover_f1 = aggregate_binary_metrics(cover_tp, cover_fp, cover_fn)
    return {
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
    }


def run_smoke_checks(
    model: UnifiedProposalAlignerNet,
    dataset: UnifiedDataset,
    loader: DataLoader,
    device: torch.device,
    *,
    mem_rel_class_weight: torch.Tensor | None = None,
    edge_rel_class_weight: torch.Tensor | None = None,
    edge_hard_neg_weight: float = 1.0,
    edge_hard_neg_max_per_row: int = 6,
    use_weight: float = 0.5,
    type_weight: float = 0.25,
    commit_weight: float = 0.25,
    edge_exist_weight: float = 1.0,
    edge_rel_weight: float = 0.5,
    span_weight: float = 1.0,
    mem_kind_weight: float = 0.5,
    mem_rel_weight: float = 0.25,
    mixed_dst_weight: float = 0.5,
    bridge_a_weight: float = 0.5,
    bridge_b_weight: float = 0.5,
    verifier_weight: float = 0.0,
    counterfactual_rate: float = 0.0,
) -> Dict[str, Any]:
    batch, rows = next(iter(loader))
    batch = to_device(batch, device)
    out = model(batch)
    finite_ok = all(torch.isfinite(t).all().item() for t in out.values())
    loss, parts = compute_loss(
        batch, out,
        mem_rel_class_weight=mem_rel_class_weight,
        edge_rel_class_weight=edge_rel_class_weight,
        edge_hard_neg_weight=edge_hard_neg_weight,
        edge_hard_neg_max_per_row=edge_hard_neg_max_per_row,
        use_weight=use_weight,
        type_weight=type_weight,
        commit_weight=commit_weight,
        edge_exist_weight=edge_exist_weight,
        edge_rel_weight=edge_rel_weight,
        span_weight=span_weight,
        mem_kind_weight=mem_kind_weight,
        mem_rel_weight=mem_rel_weight,
        mixed_dst_weight=mixed_dst_weight,
        bridge_a_weight=bridge_a_weight,
        bridge_b_weight=bridge_b_weight,
        verifier_weight=verifier_weight,
        counterfactual_rate=counterfactual_rate,
    )

    oracle_ok = 0
    oracle_total = 0
    mismatches: List[Dict[str, Any]] = []
    for row in rows[:5]:
        graph = dataset.graph(str(row.get("graph_path", "")))
        memory_ids = build_candidate_memory_ids(row, graph)
        oracle = oracle_prediction_tensors(row, memory_ids, graph)
        pred_nodes = build_predicted_session_nodes(
            row,
            use_pred=oracle["use"],
            span_pred=oracle["span"],
            bridge_pred=oracle["bridge"],
            mixed_dst_pred=oracle["mixed_dst"],
            bridge_a_pred=oracle["bridge_a"],
            bridge_b_pred=oracle["bridge_b"],
            memory_ids=memory_ids,
            graph=graph,
        )
        pred_text = {str(node.get("name", "")): str(node.get("span_text", "")) for node in pred_nodes}
        gold_text = {
            str(node.get("name", "")): str(node.get("span_text", ""))
            for node in (row.get("_oracle_goal", {}) or {}).get("session_nodes", []) or []
        }
        for name, expected in gold_text.items():
            oracle_total += 1
            actual = pred_text.get(name, "")
            if actual == expected:
                oracle_ok += 1
            else:
                mismatches.append({"row_id": row.get("id", ""), "name": name, "expected": expected, "actual": actual})

    return {
        "finite_ok": bool(finite_ok and torch.isfinite(loss).item()),
        "loss": float(loss.item()),
        "loss_parts": parts,
        "oracle_text_match": {"ok": oracle_ok, "total": oracle_total, "mismatches": mismatches[:5]},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Train unified proposer+aligner model")
    ap.add_argument("--train-jsonl", default="artifacts/proposer_v1_20260512/proposer_train.jsonl")
    ap.add_argument("--val-jsonl", default="artifacts/proposer_v1_20260512/proposer_val.jsonl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--hash-dim", type=int, default=512)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--cand-emb-cache", default="artifacts/spec_emb_cache/pred_v1_fix8_minilm_cand.npz")
    ap.add_argument("--mem-emb-cache", default="artifacts/spec_emb_cache/pred_v1_fix8_minilm_mem.npz")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--max-train-rows", type=int, default=0)
    ap.add_argument("--max-val-rows", type=int, default=0)
    ap.add_argument("--smoke-check-only", action="store_true")
    ap.add_argument(
        "--resume-from",
        default="",
        help="Path to a .pt checkpoint to warm-start the model weights from (model state_dict only). "
             "Useful for fine-tuning from a previously trained checkpoint with new loss settings.",
    )
    ap.add_argument(
        "--mem-rel-class-weight",
        choices=["none", "inverse_freq"],
        default="none",
        help="If 'inverse_freq', up-weights rare memory_attachment relations during the mem_rel CE loss",
    )
    ap.add_argument("--mem-rel-weight-min", type=float, default=0.5)
    ap.add_argument("--mem-rel-weight-max", type=float, default=2.5)
    ap.add_argument(
        "--edge-rel-class-weight",
        choices=["none", "inverse_freq"],
        default="none",
        help="If 'inverse_freq', up-weights rare session-edge relations (depend, part_of, contradict, example_of) during the edge_rel CE loss",
    )
    ap.add_argument("--edge-rel-weight-min", type=float, default=0.5)
    ap.add_argument("--edge-rel-weight-max", type=float, default=4.0)
    ap.add_argument(
        "--edge-exist-hard-neg-weight",
        type=float,
        default=1.5,
        help="Loss multiplier for structural hard-negative edges (reverse-direction, transitive shortcuts). "
             "1.0 = no up-weighting. Default 1.5 targets long_decompose FP patterns.",
    )
    ap.add_argument(
        "--edge-exist-hard-neg-max-per-row",
        type=int,
        default=6,
        help="Max hard-negative edges to up-weight per row. Default 6 (covers K=3 slots = 6 pairs).",
    )
    ap.add_argument(
        "--edge-pair-feat-dim",
        type=int,
        default=0,
        help="Dimension of content-based edge-pair features (0 = disabled, 5 = jaccard+containment+ratio+position)",
    )
    ap.add_argument(
        "--freeze-backbone-for-edgepair",
        action="store_true",
        help="Freeze all parameters except edge_pair_proj, edge_exist_head, edge_rel_head, and verifier_head. "
             "Use with --edge-pair-feat-dim or --verifier-weight for controlled ablation.",
    )
    ap.add_argument(
        "--verifier-weight",
        type=float,
        default=0.0,
        help="Weight for verifier edge relation loss (0 = disabled). Adds a semantically-rich edge verifier head "
             "that uses span embeddings (product/difference) + signal + position. Trained with frozen backbone.",
    )
    ap.add_argument(
        "--counterfactual-rate",
        type=float,
        default=0.0,
        help="Probability of permuting gold edge relation labels within each row during verifier training "
             "(0 = disabled). Uses delta-preserving permutation (only swaps relations among edges with the "
             "same (j-i) delta) to break the position->relation shortcut while preserving position distribution.",
    )
    ap.add_argument("--edge-exist-weight", type=float, default=1.0, help="Weight for edge existence BCE loss")
    ap.add_argument("--edge-rel-weight", type=float, default=0.5, help="Weight for edge relation CE loss")
    ap.add_argument("--use-weight", type=float, default=0.5, help="Weight for slot-use BCE loss")
    ap.add_argument("--type-weight", type=float, default=0.25, help="Weight for bridge-type BCE loss")
    ap.add_argument("--commit-weight", type=float, default=0.25, help="Weight for commit-family CE loss")
    ap.add_argument("--span-weight", type=float, default=1.0, help="Weight for span CE loss")
    ap.add_argument("--mem-kind-weight", type=float, default=0.5, help="Weight for mem-kind CE loss")
    ap.add_argument("--mem-rel-weight", type=float, default=0.25, help="Weight for mem-relation CE loss")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_rows = read_jsonl(args.train_jsonl)
    val_rows = read_jsonl(args.val_jsonl)
    if args.max_train_rows > 0:
        train_rows = train_rows[: args.max_train_rows]
    if args.max_val_rows > 0:
        val_rows = val_rows[: args.max_val_rows]

    train_ds = UnifiedDataset(train_rows, hash_dim=args.hash_dim, cand_emb_cache=args.cand_emb_cache, mem_emb_cache=args.mem_emb_cache)
    val_ds = UnifiedDataset(val_rows, hash_dim=args.hash_dim, cand_emb_cache=args.cand_emb_cache, mem_emb_cache=args.mem_emb_cache)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = UnifiedProposalAlignerNet(
        hash_dim=args.hash_dim,
        hidden_dim=args.hidden_dim,
        k_max=3,
        cand_emb_dim=train_ds.cand_emb_dim,
        mem_emb_dim=train_ds.mem_emb_dim,
        edge_pair_feat_dim=args.edge_pair_feat_dim,
        use_verifier=args.verifier_weight > 0,
    ).to(device)

    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=device)
        state = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(json.dumps({
            "resume_from": args.resume_from,
            "missing_keys": missing,
            "unexpected_keys": unexpected,
        }))

    if args.freeze_backbone_for_edgepair:
        frozen = 0
        trainable = 0
        keep_prefixes = ("edge_pair_proj", "edge_exist_head", "edge_rel_head", "verifier_head")
        for name, param in model.named_parameters():
            keep = any(name.startswith(p) for p in keep_prefixes)
            if keep:
                param.requires_grad = True
                trainable += 1
            else:
                param.requires_grad = False
                frozen += 1
        print(json.dumps({
            "freeze_backbone_for_edgepair": True,
            "trainable_params": trainable,
            "frozen_params": frozen,
        }))

    mem_rel_class_weight: torch.Tensor | None = None
    if args.mem_rel_class_weight == "inverse_freq":
        mem_rel_class_weight = compute_mem_rel_class_weights(
            train_rows,
            device=device,
            min_weight=args.mem_rel_weight_min,
            max_weight=args.mem_rel_weight_max,
        )
        print(json.dumps({
            "mem_rel_class_weight": {
                rel: round(float(mem_rel_class_weight[i].item()), 4)
                for rel, i in REL_WITH_NONE_TO_ID.items()
            },
        }))

    edge_rel_class_weight: torch.Tensor | None = None
    if args.edge_rel_class_weight == "inverse_freq":
        edge_rel_class_weight = compute_edge_rel_class_weights(
            train_rows,
            device=device,
            min_weight=args.edge_rel_weight_min,
            max_weight=args.edge_rel_weight_max,
        )
        print(json.dumps({
            "edge_rel_class_weight": {
                rel: round(float(edge_rel_class_weight[i].item()), 4)
                for rel, i in REL_WITH_NONE_TO_ID.items()
            },
        }))

    smoke = run_smoke_checks(
        model, train_ds, train_loader, device,
        mem_rel_class_weight=mem_rel_class_weight,
        edge_rel_class_weight=edge_rel_class_weight,
        edge_hard_neg_weight=args.edge_exist_hard_neg_weight,
        edge_hard_neg_max_per_row=args.edge_exist_hard_neg_max_per_row,
        use_weight=args.use_weight,
        type_weight=args.type_weight,
        commit_weight=args.commit_weight,
        edge_exist_weight=args.edge_exist_weight,
        edge_rel_weight=args.edge_rel_weight,
        span_weight=args.span_weight,
        mem_kind_weight=args.mem_kind_weight,
        mem_rel_weight=args.mem_rel_weight,
        mixed_dst_weight=0.5,
        bridge_a_weight=0.5,
        bridge_b_weight=0.5,
        verifier_weight=args.verifier_weight,
        counterfactual_rate=args.counterfactual_rate,
    )
    print(json.dumps({"smoke": smoke}, ensure_ascii=False))
    if args.smoke_check_only:
        return 0

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history: List[Dict[str, Any]] = []
    best_score = float("-inf")
    best_cover = float("-inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_total = 0.0
        part_sums: Dict[str, float] = {}
        batches = 0
        for batch, _rows in train_loader:
            batch = to_device(batch, device)
            out = model(batch)
            loss, parts = compute_loss(
                batch, out,
                use_weight=args.use_weight,
                type_weight=args.type_weight,
                commit_weight=args.commit_weight,
                edge_exist_weight=args.edge_exist_weight,
                edge_rel_weight=args.edge_rel_weight,
                span_weight=args.span_weight,
                mem_kind_weight=args.mem_kind_weight,
                mem_rel_weight=args.mem_rel_weight,
                mem_rel_class_weight=mem_rel_class_weight,
                edge_rel_class_weight=edge_rel_class_weight,
                edge_hard_neg_weight=args.edge_exist_hard_neg_weight,
                edge_hard_neg_max_per_row=args.edge_exist_hard_neg_max_per_row,
                verifier_weight=args.verifier_weight,
                counterfactual_rate=args.counterfactual_rate,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total += float(loss.item())
            batches += 1
            for key, val in parts.items():
                part_sums[key] = part_sums.get(key, 0.0) + val

        train_metrics = {"loss": loss_total / max(batches, 1)}
        for key, total in part_sums.items():
            train_metrics[key] = total / max(batches, 1)

        val_metrics = compute_metrics(model, val_loader, device)
        score = (
            0.5 * val_metrics["row_complete_rate"]
            + 1.0 * val_metrics["text_faithful_row_complete_rate"]
            + 0.25 * val_metrics["span_top1_acc"]
            + 0.25 * val_metrics["cover_f1"]
        )
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics, "score": score}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))

        ckpt = {
            "model": model.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "score": score,
            "val": val_metrics,
        }
        if score > best_score:
            best_score = score
            torch.save(ckpt, out_dir / "best_unified_v1.pt")
        cover_score = val_metrics["cover_f1"]
        if cover_score > best_cover:
            best_cover = cover_score
            torch.save(ckpt, out_dir / "best_cover_unified_v1.pt")

    (out_dir / "train_history.json").write_text(
        json.dumps({"history": history, "best_score": best_score, "best_cover": best_cover}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
