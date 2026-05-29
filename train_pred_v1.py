from __future__ import annotations

import argparse
import hashlib
import json
import random
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from graph_core import MemoryGraph, canonical_relation, lexical_overlap
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
    SPEC_TYPE_TO_ID,
    PredAlignNet,
    PredBatch,
    infer_cand_emb_dim_from_state,
    infer_edge_rel_pair_feat_dim_from_state,
    infer_mem_emb_dim_from_state,
    infer_spec_emb_dim_from_state,
)
from synthesize_node_text import apply_template_synthesis


IGNORE = -100


def _tokens(text: str) -> set[str]:
    return {tok for tok in text.lower().split() if tok}


def text_cache_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


@lru_cache(maxsize=None)
def _load_embedding_cache_cached(path_str: str) -> tuple[Dict[str, np.ndarray], int]:
    data = np.load(Path(path_str), allow_pickle=False)
    keys = [str(k) for k in data["keys"].tolist()]
    embeddings = data["embeddings"].astype("float32")
    return {k: embeddings[i] for i, k in enumerate(keys)}, int(embeddings.shape[1])


def load_embedding_cache(path: str | Path | None) -> tuple[Dict[str, np.ndarray], int]:
    if not path:
        return {}, 0
    return _load_embedding_cache_cached(str(Path(path)))


def edge_relation_pair_features(left_text: str, right_text: str, i: int, j: int, n_specs: int) -> torch.Tensor:
    left_tokens = _tokens(left_text)
    right_tokens = _tokens(right_text)
    inter = left_tokens & right_tokens
    union = left_tokens | right_tokens
    left_n = max(len(left_tokens), 1)
    right_n = max(len(right_tokens), 1)
    return torch.tensor([
        len(inter) / max(len(union), 1),
        len(inter) / left_n,
        len(inter) / right_n,
        right_n / left_n,
        (j - i) / max(n_specs - 1, 1),
    ], dtype=torch.float32)


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def goal_commit_family(goal: Mapping[str, Any]) -> str:
    actions = {str(fc.get("action", "")) for fc in goal.get("final_commits", []) or []}
    if "no_op" in actions:
        return "no_op"
    if "add_node" in actions:
        return "add_node"
    return "other"


def synthesize_goal_session_nodes(row: Mapping[str, Any]) -> Dict[str, Any]:
    goal = dict((row.get("goal", {}) or {}))
    session_nodes = [dict(node) for node in (goal.get("session_nodes", []) or [])]
    slots = [
        {
            "slot_idx": i,
            "use": True,
            "session_name": str(node.get("name", f"s{i}")),
            "name": str(node.get("name", f"s{i}")),
            "span_text": str(node.get("span_text", "")),
            "node_type": str(node.get("node_type", "concept")),
            "is_bridge": str(node.get("node_type", "concept")) == "bridge",
        }
        for i, node in enumerate(session_nodes)
    ]
    synthesized = apply_template_synthesis(row, slots)
    text_by_name = {
        str(slot.get("session_name", "")): str(slot.get("span_text", ""))
        for slot in synthesized
    }
    for node in session_nodes:
        name = str(node.get("name", ""))
        if name in text_by_name:
            node["span_text"] = text_by_name[name]
    goal["session_nodes"] = session_nodes
    return goal


def compute_edge_rel_class_weights(
    rows: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    min_weight: float,
    max_weight: float,
) -> torch.Tensor:
    counts = torch.zeros(len(REL_WITH_NONE), dtype=torch.float32)
    for row in rows:
        goal = row.get("goal", {}) or {}
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


def edge_relation_loss_fn(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    mode: str,
    class_weight: torch.Tensor | None,
    focal_gamma: float,
) -> torch.Tensor:
    if mode == "focal":
        ce = F.cross_entropy(logits, targets, reduction="none")
        p_t = torch.exp(-ce)
        return (((1.0 - p_t) ** float(focal_gamma)) * ce).mean()
    return F.cross_entropy(logits, targets, weight=class_weight)


def build_edge_hard_negative_mask(y_edge_exist: torch.Tensor, edge_mask: torch.Tensor, *, max_per_row: int) -> torch.Tensor:
    """
    Mark structural hard negatives for edge existence.

    Categories:
      - reverse direction of each gold edge, when not also gold
      - transitive shortcut i->k when gold has i->j and j->k

    This uses gold edge structure only; it does not inspect predictions.
    """
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


def resolve_freeze_mode(args: argparse.Namespace) -> str:
    legacy_modes = {
        "edge_emb": bool(getattr(args, "freeze_except_edge_emb", False)),
        "cand_emb_span": bool(getattr(args, "freeze_except_cand_emb_span", False)),
        "mem_emb_rel": bool(getattr(args, "freeze_except_mem_emb_rel", False)),
        "edge": bool(getattr(args, "freeze_except_edge", False)),
    }
    active_legacy = [mode for mode, enabled in legacy_modes.items() if enabled]
    if args.freeze_mode != "none" and active_legacy:
        raise ValueError("Use either --freeze-mode or deprecated --freeze-except-* flags, not both")
    if len(active_legacy) > 1:
        raise ValueError(f"Deprecated freeze flags are mutually exclusive; got {active_legacy}")
    if active_legacy:
        return active_legacy[0]
    return str(args.freeze_mode)


def apply_freeze_mode(model: PredAlignNet, freeze_mode: str) -> None:
    trainable_by_mode = {
        "none": None,
        "edge_emb": ("spec_emb_proj", "edge_exist_head", "edge_rel_head"),
        "cand_emb_span": ("cand_emb_proj", "span_scorer", "none_head"),
        "mem_emb_rel": ("mem_emb_proj", "mem_rel_head"),
        "edge": ("edge_exist_head", "edge_rel_head"),
        "synth_finetune": (
            "signal_proj",
            "spec_proj",
            "spec_emb_proj",
            "span_scorer",
            "none_head",
            "commit_head",
            "edge_exist_head",
            "edge_rel_head",
            "mem_rel_head",
        ),
    }
    if freeze_mode not in trainable_by_mode:
        raise ValueError(f"Unknown freeze mode: {freeze_mode}")
    prefixes = trainable_by_mode[freeze_mode]
    if prefixes is None:
        return
    named_params = dict(model.named_parameters())
    missing_prefixes = [prefix for prefix in prefixes if not any(name.startswith(prefix) for name in named_params)]
    if missing_prefixes:
        raise ValueError(f"Freeze mode {freeze_mode} refers to missing prefixes: {missing_prefixes}")
    for name, param in named_params.items():
        param.requires_grad = name.startswith(prefixes)


class PredDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        hash_dim: int = 512,
        spec_emb_cache: str | Path | None = None,
        spec_emb_dim_override: int = 0,
        cand_emb_cache: str | Path | None = None,
        cand_emb_dim_override: int = 0,
        mem_emb_cache: str | Path | None = None,
        mem_emb_dim_override: int = 0,
        synth_swap_prob: float = 0.0,
    ) -> None:
        self.rows = list(rows)
        self.hash_dim = hash_dim
        self.synth_swap_prob = float(synth_swap_prob)
        self._graph_cache: Dict[str, MemoryGraph] = {}
        self.spec_emb_cache, self.spec_emb_dim = load_embedding_cache(spec_emb_cache)
        if self.spec_emb_dim == 0 and spec_emb_dim_override > 0:
            self.spec_emb_dim = int(spec_emb_dim_override)
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
        goal = row.get("goal", {}) or {}
        if self.synth_swap_prob > 0.0 and str(row.get("task_type", "")) in {"mixed_add_link", "multi_region_attach"}:
            if self.synth_swap_prob >= 1.0 or random.random() < self.synth_swap_prob:
                goal = synthesize_goal_session_nodes(row)
        specs = goal.get("session_nodes", []) or []
        spans = row.get("spans", []) or []
        oracle_by_name = {str(x.get("session_name", "")): x for x in row.get("span_oracle", []) or []}
        spec_name_to_idx = {str(spec.get("name", f"s{i}")): i for i, spec in enumerate(specs)}

        signal_bow = bow_hash(signal, self.hash_dim)
        spec_bow = []
        spec_emb = []
        spec_type_ids = []
        y_span = []
        cand_span_id_to_idx: Dict[str, int] = {}
        cand_bow = []
        cand_emb = []
        cand_kind_ids = []
        cand_feat = []
        span_pair_feat = []

        for j, span in enumerate(spans):
            sid = str(span.get("id", ""))
            stext = str(span.get("text", ""))
            kind = str(span.get("span_kind", "unknown"))
            cand_span_id_to_idx[sid] = j
            cand_bow.append(bow_hash(stext, self.hash_dim))
            if self.cand_emb_dim > 0:
                emb = self.cand_emb_cache.get(text_cache_key(stext))
                if emb is None:
                    cand_emb.append(torch.zeros(self.cand_emb_dim, dtype=torch.float32))
                else:
                    cand_emb.append(torch.from_numpy(emb.astype("float32")))
            cand_kind_ids.append(SPAN_KIND_TO_ID.get(kind, SPAN_KIND_TO_ID["unknown"]))
            cand_feat.append(torch.tensor([
                float(span.get("start", 0)) / max(len(signal), 1),
                float(span.get("end", 0)) / max(len(signal), 1),
            ], dtype=torch.float32))

        for i, spec in enumerate(specs):
            sname = str(spec.get("name", f"s{i}"))
            spec_text = str(spec.get("span_text", ""))
            spec_bow.append(bow_hash(spec_text, self.hash_dim))
            if self.spec_emb_dim > 0:
                emb = self.spec_emb_cache.get(text_cache_key(spec_text))
                if emb is None:
                    spec_emb.append(torch.zeros(self.spec_emb_dim, dtype=torch.float32))
                else:
                    spec_emb.append(torch.from_numpy(emb.astype("float32")))
            spec_type_ids.append(SPEC_TYPE_TO_ID.get(str(spec.get("node_type", "concept")), SPEC_TYPE_TO_ID["unknown"]))
            pair_feats_for_spec = []
            for span in spans:
                stext = str(span.get("text", ""))
                pair_feats_for_spec.append(torch.tensor([
                    float(lexical_overlap(stext, spec_text)),
                ], dtype=torch.float32))
            span_pair_feat.append(
                torch.stack(pair_feats_for_spec, dim=0) if pair_feats_for_spec else torch.zeros((0, 1), dtype=torch.float32)
            )

            oracle = oracle_by_name.get(sname, {})
            best_span_id = oracle.get("best_span_id")
            target = IGNORE
            if best_span_id in cand_span_id_to_idx:
                target = cand_span_id_to_idx[str(best_span_id)]
            y_span.append(target)

        S = len(specs)
        spec_texts = [str(spec.get("span_text", "")) for spec in specs]
        edge_rel_pair_feat = torch.zeros((S, S, 5), dtype=torch.float32)
        for i in range(S):
            for j in range(S):
                if i == j:
                    continue
                edge_rel_pair_feat[i, j] = edge_relation_pair_features(spec_texts[i], spec_texts[j], i, j, S)

        y_edge_exist = torch.zeros((S, S), dtype=torch.float32)
        y_edge_rel = torch.full((S, S), IGNORE, dtype=torch.long)
        for i in range(S):
            y_edge_rel[i, i] = IGNORE
        for edge in goal.get("session_edges", []) or []:
            src = spec_name_to_idx.get(str(edge.get("src", "")))
            dst = spec_name_to_idx.get(str(edge.get("dst", "")))
            if src is None or dst is None or src == dst:
                continue
            y_edge_exist[src, dst] = 1.0
            y_edge_rel[src, dst] = REL_WITH_NONE_TO_ID[canonical_relation(str(edge.get("relation", "related")))]
        for i in range(S):
            for j in range(S):
                if i != j and y_edge_rel[i, j] == IGNORE:
                    y_edge_rel[i, j] = REL_WITH_NONE_TO_ID["none"]

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

        mem_bow = []
        mem_emb = []
        mem_feat = []
        mem_id_to_idx: Dict[str, int] = {}
        init_set = {str(x) for x in row.get("initial_memory_node_ids", []) or []}
        for j, mem in enumerate(candidate_memory_ids):
            mem_id_to_idx[mem] = j
            text = str(graph.nodes[mem].text)
            mem_bow.append(bow_hash(text, self.hash_dim))
            if self.mem_emb_dim > 0:
                emb = self.mem_emb_cache.get(text_cache_key(text))
                if emb is None:
                    mem_emb.append(torch.zeros(self.mem_emb_dim, dtype=torch.float32))
                else:
                    mem_emb.append(torch.from_numpy(emb.astype("float32")))
            mem_feat.append(torch.tensor([
                float(lexical_overlap(signal, text)),
                1.0 if mem in init_set else 0.0,
                float(j) / max(len(candidate_memory_ids), 1),
            ], dtype=torch.float32))

        M = len(candidate_memory_ids)
        mem_pair_feat = []
        for spec in specs:
            spec_text = str(spec.get("span_text", ""))
            overlaps = [
                float(lexical_overlap(str(graph.nodes[mid].text), spec_text))
                for mid in candidate_memory_ids
            ]
            best_val = max(overlaps) if overlaps else 0.0
            per_mem = [
                torch.tensor([ov, ov / best_val if best_val > 0.0 else 0.0], dtype=torch.float32)
                for ov in overlaps
            ]
            mem_pair_feat.append(
                torch.stack(per_mem, dim=0) if per_mem else torch.zeros((0, 2), dtype=torch.float32)
            )

        y_mem_kind = torch.zeros((S, M), dtype=torch.long)
        y_mem_rel = torch.full((S, M), REL_WITH_NONE_TO_ID["none"], dtype=torch.long)
        for att in goal.get("memory_attachments", []) or []:
            sidx = spec_name_to_idx.get(str(att.get("session", "")))
            midx = mem_id_to_idx.get(str(att.get("memory_id", "")))
            if sidx is None or midx is None:
                continue
            y_mem_kind[sidx, midx] = MEM_LINK_KIND_TO_ID["attach"]
            y_mem_rel[sidx, midx] = REL_WITH_NONE_TO_ID[canonical_relation(str(att.get("relation", "related")))]
        for idx, cov in enumerate(goal.get("covered_mappings", []) or []):
            sname = str(cov.get("session", f"covered_{idx}"))
            if not sname:
                continue
            sidx = spec_name_to_idx.get(sname)
            midx = mem_id_to_idx.get(str(cov.get("memory_id", "")))
            if sidx is None or midx is None:
                continue
            y_mem_kind[sidx, midx] = MEM_LINK_KIND_TO_ID["cover"]

        return {
            "id": row.get("id", ""),
            "task_type": row.get("task_type", "unknown"),
            "signal_bow": signal_bow,
            "spec_bow": torch.stack(spec_bow, dim=0) if spec_bow else torch.zeros((0, self.hash_dim), dtype=torch.float32),
            "spec_emb": torch.stack(spec_emb, dim=0) if spec_emb else torch.zeros((len(specs), self.spec_emb_dim), dtype=torch.float32),
            "spec_type_ids": torch.tensor(spec_type_ids, dtype=torch.long) if spec_type_ids else torch.zeros((0,), dtype=torch.long),
            "cand_bow": torch.stack(cand_bow, dim=0) if cand_bow else torch.zeros((0, self.hash_dim), dtype=torch.float32),
            "cand_emb": torch.stack(cand_emb, dim=0) if cand_emb else torch.zeros((len(spans), self.cand_emb_dim), dtype=torch.float32),
            "cand_kind_ids": torch.tensor(cand_kind_ids, dtype=torch.long) if cand_kind_ids else torch.zeros((0,), dtype=torch.long),
            "cand_feat": torch.stack(cand_feat, dim=0) if cand_feat else torch.zeros((0, 2), dtype=torch.float32),
            "span_pair_feat": torch.stack(span_pair_feat, dim=0) if span_pair_feat else torch.zeros((0, 0, 1), dtype=torch.float32),
            "mem_bow": torch.stack(mem_bow, dim=0) if mem_bow else torch.zeros((0, self.hash_dim), dtype=torch.float32),
            "mem_emb": torch.stack(mem_emb, dim=0) if mem_emb else torch.zeros((len(candidate_memory_ids), self.mem_emb_dim), dtype=torch.float32),
            "mem_feat": torch.stack(mem_feat, dim=0) if mem_feat else torch.zeros((0, 3), dtype=torch.float32),
            "mem_pair_feat": torch.stack(mem_pair_feat, dim=0) if mem_pair_feat else torch.zeros((0, 0, 2), dtype=torch.float32),
            "edge_rel_pair_feat": edge_rel_pair_feat,
            "y_span": torch.tensor(y_span, dtype=torch.long) if y_span else torch.zeros((0,), dtype=torch.long),
            "y_commit": torch.tensor(COMMIT_TO_ID[goal_commit_family(goal)], dtype=torch.long),
            "y_edge_exist": y_edge_exist,
            "y_edge_rel": y_edge_rel,
            "y_mem_kind": y_mem_kind,
            "y_mem_rel": y_mem_rel,
        }


def collate(items: Sequence[Mapping[str, Any]]) -> PredBatch:
    B = len(items)
    H = items[0]["signal_bow"].numel()
    max_s = max(x["spec_bow"].size(0) for x in items)
    max_c = max(x["cand_bow"].size(0) for x in items)
    max_c = max(max_c, 1)
    max_m = max(x["mem_bow"].size(0) for x in items)
    max_m = max(max_m, 1)

    signal_bow = torch.stack([x["signal_bow"] for x in items], dim=0)
    spec_emb_dim = items[0]["spec_emb"].size(-1)
    cand_emb_dim = items[0]["cand_emb"].size(-1)
    mem_emb_dim = items[0]["mem_emb"].size(-1)
    spec_bow = torch.zeros(B, max_s, H)
    spec_emb = torch.zeros(B, max_s, spec_emb_dim)
    spec_type_ids = torch.zeros(B, max_s, dtype=torch.long)
    spec_mask = torch.zeros(B, max_s, dtype=torch.bool)
    cand_bow = torch.zeros(B, max_c, H)
    cand_emb = torch.zeros(B, max_c, cand_emb_dim)
    cand_kind_ids = torch.zeros(B, max_c, dtype=torch.long)
    cand_feat = torch.zeros(B, max_c, 2)
    span_pair_feat = torch.zeros(B, max_s, max_c, 1)
    cand_mask = torch.zeros(B, max_s, max_c, dtype=torch.bool)
    mem_bow = torch.zeros(B, max_m, H)
    mem_emb = torch.zeros(B, max_m, mem_emb_dim)
    mem_feat = torch.zeros(B, max_m, 3)
    mem_mask = torch.zeros(B, max_m, dtype=torch.bool)
    mem_pair_feat = torch.zeros(B, max_s, max_m, 2)
    edge_rel_pair_feat = torch.zeros(B, max_s, max_s, 5)
    y_span = torch.full((B, max_s), IGNORE, dtype=torch.long)
    y_commit = torch.stack([x["y_commit"] for x in items], dim=0)
    y_edge_exist = torch.zeros(B, max_s, max_s, dtype=torch.float32)
    y_edge_rel = torch.full((B, max_s, max_s), IGNORE, dtype=torch.long)
    y_mem_kind = torch.zeros(B, max_s, max_m, dtype=torch.long)
    y_mem_rel = torch.full((B, max_s, max_m), REL_WITH_NONE_TO_ID["none"], dtype=torch.long)
    edge_mask = torch.zeros(B, max_s, max_s, dtype=torch.bool)

    for b, x in enumerate(items):
        s = x["spec_bow"].size(0)
        spec_bow[b, :s] = x["spec_bow"]
        if spec_emb_dim:
            spec_emb[b, :s] = x["spec_emb"]
        spec_type_ids[b, :s] = x["spec_type_ids"]
        spec_mask[b, :s] = True
        y_span[b, :s] = x["y_span"]
        y_edge_exist[b, :s, :s] = x["y_edge_exist"]
        y_edge_rel[b, :s, :s] = x["y_edge_rel"]
        edge_rel_pair_feat[b, :s, :s] = x["edge_rel_pair_feat"]
        m = x["mem_bow"].size(0)
        if m:
            mem_bow[b, :m] = x["mem_bow"]
            if mem_emb_dim:
                mem_emb[b, :m] = x["mem_emb"]
            mem_feat[b, :m] = x["mem_feat"]
            mem_mask[b, :m] = True
            y_mem_kind[b, :s, :m] = x["y_mem_kind"]
            y_mem_rel[b, :s, :m] = x["y_mem_rel"]
            if s:
                mem_pair_feat[b, :s, :m] = x["mem_pair_feat"]
        c = x["cand_bow"].size(0)
        if c:
            cand_bow[b, :c] = x["cand_bow"]
            if cand_emb_dim:
                cand_emb[b, :c] = x["cand_emb"]
            cand_kind_ids[b, :c] = x["cand_kind_ids"]
            cand_feat[b, :c] = x["cand_feat"]
            span_pair_feat[b, :s, :c] = x["span_pair_feat"]
        for i in range(s):
            edge_mask[b, i, :s] = True
            edge_mask[b, i, i] = False
            if c:
                cand_mask[b, i, :c] = True
            if y_span[b, i] == IGNORE:
                y_span[b, i] = max_c
            if c == 0:
                y_span[b, i] = max_c

    return PredBatch(
        signal_bow=signal_bow,
        spec_bow=spec_bow,
        spec_emb=spec_emb,
        spec_type_ids=spec_type_ids,
        spec_mask=spec_mask,
        cand_bow=cand_bow,
        cand_emb=cand_emb,
        cand_kind_ids=cand_kind_ids,
        cand_feat=cand_feat,
        span_pair_feat=span_pair_feat,
        cand_mask=cand_mask,
        mem_bow=mem_bow,
        mem_emb=mem_emb,
        mem_feat=mem_feat,
        mem_mask=mem_mask,
        mem_pair_feat=mem_pair_feat,
        edge_rel_pair_feat=edge_rel_pair_feat,
        edge_mask=edge_mask,
        y_span=y_span,
        y_commit=y_commit,
        y_edge_exist=y_edge_exist,
        y_edge_rel=y_edge_rel,
        y_mem_kind=y_mem_kind,
        y_mem_rel=y_mem_rel,
    )


def to_device(batch: PredBatch, device: torch.device) -> PredBatch:
    return PredBatch(**{k: getattr(batch, k).to(device) for k in batch.__dataclass_fields__})


def decode_span_predictions(
    span_logits: torch.Tensor,
    spec_mask: torch.Tensor,
    cand_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Greedy exclusive span decoding.

    Each spec gets its best available span, with a dedicated none class as fallback.
    Real spans are assigned exclusively within a row; the none class is reusable.
    """
    pred = span_logits.argmax(dim=-1)
    B, S, total_c = span_logits.shape
    none_index = total_c - 1
    for b in range(B):
        valid_specs = [s for s in range(S) if bool(spec_mask[b, s].item())]
        if not valid_specs:
            continue
        used: set[int] = set()
        best_real: Dict[int, float] = {}
        for s in valid_specs:
            valid_cands = torch.nonzero(cand_mask[b, s], as_tuple=False).flatten().tolist()
            if valid_cands:
                best_real[s] = float(span_logits[b, s, valid_cands].max().item())
            else:
                best_real[s] = float("-inf")
        ordered_specs = sorted(valid_specs, key=lambda s: best_real[s], reverse=True)
        for s in ordered_specs:
            none_score = float(span_logits[b, s, none_index].item())
            valid_cands = torch.nonzero(cand_mask[b, s], as_tuple=False).flatten().tolist()
            available = [c for c in valid_cands if c not in used]
            if not available:
                pred[b, s] = none_index
                continue
            best_c = max(available, key=lambda c: float(span_logits[b, s, c].item()))
            best_score = float(span_logits[b, s, best_c].item())
            if none_score >= best_score:
                pred[b, s] = none_index
            else:
                pred[b, s] = best_c
                used.add(best_c)
    return pred


def decode_mem_kind_predictions(
    mem_kind_logits: torch.Tensor,
    mem_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Per-spec exclusive cover decode.

    The independent argmax can assign cover to multiple memories for the same
    spec. This is wrong: each spec covers at most one memory. For each spec,
    keep only the memory with the highest cover-class logit; demote all other
    cover predictions to none (class 0). Attach predictions are left unchanged.
    """
    pred = mem_kind_logits.argmax(dim=-1)  # [B, S, M]
    cover_id = MEM_LINK_KIND_TO_ID["cover"]
    B, S, M = pred.shape

    cover_scores = mem_kind_logits[..., cover_id].clone()  # [B, S, M]
    cover_scores.masked_fill_(~mem_mask[:, None, :].expand(B, S, M), LOGIT_MASK_VALUE)
    best_m = cover_scores.argmax(dim=-1).unsqueeze(-1)  # [B, S, 1]

    is_cover = pred == cover_id  # [B, S, M]
    is_best = torch.zeros_like(is_cover)
    is_best.scatter_(2, best_m, is_cover.gather(2, best_m))
    pred = pred.clone()
    pred[is_cover & ~is_best] = 0
    return pred


def decode_edge_predictions(
    edge_exist_logits: torch.Tensor,
    edge_mask: torch.Tensor,
    *,
    threshold: float = EDGE_EXIST_THRESHOLD,
) -> torch.Tensor:
    """
    Decode edge existence with two structural constraints:

    1. Anti-symmetry:
       if both (i, j) and (j, i) are predicted, keep the higher-confidence one.
    2. Transitive reduction:
       if (i, k) is predicted and there exists j with (i, j) and (j, k),
       remove (i, k).

    This is applied row-wise on the predicted graph after thresholding.
    """
    scores = torch.sigmoid(edge_exist_logits)
    pred = (scores >= threshold) & edge_mask
    scores_t = scores.transpose(-1, -2)
    pred_t = pred.transpose(-1, -2)
    tie_keep_upper = torch.triu(torch.ones_like(pred, dtype=torch.bool), diagonal=1)
    prefer = (scores > scores_t) | ((scores == scores_t) & tie_keep_upper)
    conflict = pred & pred_t
    pred = pred & (~conflict | prefer)
    two_hop = (pred.float() @ pred.float()) > 0
    pred = pred & ~(two_hop & pred)
    return pred & edge_mask


def compute_metrics(model: PredAlignNet, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    model.eval()
    span_total = span_ok = 0
    span_nonnull_total = span_nonnull_ok = 0
    commit_total = commit_ok = 0
    edge_tp = edge_fp = edge_fn = 0
    rel_total = rel_ok = 0
    mem_attach_tp = mem_attach_fp = mem_attach_fn = 0
    mem_cover_tp = mem_cover_fp = mem_cover_fn = 0
    mem_rel_total = mem_rel_ok = 0
    row_total = row_ok = 0

    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            out = model(batch)
            span_pred = decode_span_predictions(out["span_logits"], batch.spec_mask, batch.cand_mask)
            commit_pred = out["commit_logits"].argmax(dim=-1)
            edge_exist_pred = decode_edge_predictions(out["edge_exist_logits"], batch.edge_mask)
            edge_rel_pred = out["edge_rel_logits"].argmax(dim=-1)
            mem_kind_pred = decode_mem_kind_predictions(out["mem_kind_logits"], batch.mem_mask)
            mem_rel_pred = out["mem_rel_logits"].argmax(dim=-1)

            spec_valid = batch.spec_mask
            span_match = (span_pred == batch.y_span) & spec_valid
            span_total += int(spec_valid.sum().item())
            span_ok += int(span_match.sum().item())

            none_index = out["span_logits"].size(-1) - 1
            nonnull = (batch.y_span != none_index) & spec_valid
            span_nonnull_total += int(nonnull.sum().item())
            span_nonnull_ok += int(((span_pred == batch.y_span) & nonnull).sum().item())

            commit_total += int(batch.y_commit.numel())
            commit_ok += int((commit_pred == batch.y_commit).sum().item())

            gold_edge = (batch.y_edge_exist > 0.0) & batch.edge_mask
            edge_tp += int((edge_exist_pred & gold_edge).sum().item())
            edge_fp += int((edge_exist_pred & ~gold_edge & batch.edge_mask).sum().item())
            edge_fn += int((~edge_exist_pred & gold_edge).sum().item())

            rel_mask = gold_edge
            rel_total += int(rel_mask.sum().item())
            rel_ok += int(((edge_rel_pred == batch.y_edge_rel) & rel_mask).sum().item())

            mem_mask = batch.mem_mask[:, None, :].expand_as(batch.y_mem_kind)
            gold_attach = (batch.y_mem_kind == MEM_LINK_KIND_TO_ID["attach"]) & mem_mask
            gold_cover = (batch.y_mem_kind == MEM_LINK_KIND_TO_ID["cover"]) & mem_mask
            pred_attach = (mem_kind_pred == MEM_LINK_KIND_TO_ID["attach"]) & mem_mask
            pred_cover = (mem_kind_pred == MEM_LINK_KIND_TO_ID["cover"]) & mem_mask
            mem_attach_tp += int((pred_attach & gold_attach).sum().item())
            mem_attach_fp += int((pred_attach & ~gold_attach).sum().item())
            mem_attach_fn += int((~pred_attach & gold_attach).sum().item())
            mem_cover_tp += int((pred_cover & gold_cover).sum().item())
            mem_cover_fp += int((pred_cover & ~gold_cover).sum().item())
            mem_cover_fn += int((~pred_cover & gold_cover).sum().item())

            mem_rel_mask = gold_attach
            mem_rel_total += int(mem_rel_mask.sum().item())
            mem_rel_ok += int(((mem_rel_pred == batch.y_mem_rel) & mem_rel_mask).sum().item())

            B = batch.signal_bow.size(0)
            for b in range(B):
                spec_rows = bool(span_match[b, batch.spec_mask[b]].all().item()) if batch.spec_mask[b].any() else True
                commit_row = bool((commit_pred[b] == batch.y_commit[b]).item())
                edge_row = bool(torch.equal(edge_exist_pred[b][batch.edge_mask[b]], gold_edge[b][batch.edge_mask[b]]))
                rel_row = True
                if rel_mask[b].any():
                    rel_row = bool(torch.equal(edge_rel_pred[b][rel_mask[b]], batch.y_edge_rel[b][rel_mask[b]]))
                mem_row = True
                if mem_mask[b].any():
                    mem_row = bool(torch.equal(mem_kind_pred[b][mem_mask[b]], batch.y_mem_kind[b][mem_mask[b]]))
                mem_rel_row = True
                if mem_rel_mask[b].any():
                    mem_rel_row = bool(torch.equal(mem_rel_pred[b][mem_rel_mask[b]], batch.y_mem_rel[b][mem_rel_mask[b]]))
                row_total += 1
                if spec_rows and commit_row and edge_row and rel_row and mem_row and mem_rel_row:
                    row_ok += 1

    precision = edge_tp / max(edge_tp + edge_fp, 1)
    recall = edge_tp / max(edge_tp + edge_fn, 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    attach_precision = mem_attach_tp / max(mem_attach_tp + mem_attach_fp, 1)
    attach_recall = mem_attach_tp / max(mem_attach_tp + mem_attach_fn, 1)
    attach_f1 = 0.0 if attach_precision + attach_recall == 0 else 2 * attach_precision * attach_recall / (attach_precision + attach_recall)
    cover_precision = mem_cover_tp / max(mem_cover_tp + mem_cover_fp, 1)
    cover_recall = mem_cover_tp / max(mem_cover_tp + mem_cover_fn, 1)
    cover_f1 = 0.0 if cover_precision + cover_recall == 0 else 2 * cover_precision * cover_recall / (cover_precision + cover_recall)
    return {
        "span_top1_acc": span_ok / max(span_total, 1),
        "span_top1_acc_nonnull": span_nonnull_ok / max(span_nonnull_total, 1),
        "commit_acc": commit_ok / max(commit_total, 1),
        "edge_precision": precision,
        "edge_recall": recall,
        "edge_f1": f1,
        "edge_relation_acc_on_gold": rel_ok / max(rel_total, 1),
        "attachment_precision": attach_precision,
        "attachment_recall": attach_recall,
        "attachment_f1": attach_f1,
        "attachment_relation_acc_on_gold": mem_rel_ok / max(mem_rel_total, 1),
        "cover_precision": cover_precision,
        "cover_recall": cover_recall,
        "cover_f1": cover_f1,
        "row_complete_rate": row_ok / max(row_total, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-jsonl", default="artifacts/pred_v1_20260511/pred_train.jsonl")
    ap.add_argument("--val-jsonl", default="artifacts/pred_v1_20260511/pred_val.jsonl")
    ap.add_argument("--out-dir", default="out_pred_v1_20260511")
    ap.add_argument("--hash-dim", type=int, default=512)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--init-checkpoint", default=None, help="Optional PredAlignNet checkpoint to initialize from")
    ap.add_argument("--spec-emb-cache", default=None, help="Optional .npz cache of frozen spec text embeddings")
    ap.add_argument("--cand-emb-cache", default=None, help="Optional .npz cache of frozen candidate span text embeddings")
    ap.add_argument("--mem-emb-cache", default=None, help="Optional .npz cache of frozen memory node text embeddings")
    ap.add_argument("--synth-swap-prob", type=float, default=0.0, help="Probability of rewriting synthesis-eligible gold session texts into template form during training")
    ap.add_argument("--eval-synth-val", action="store_true", help="Also evaluate a validation view with synthesized template text on all eligible rows")
    ap.add_argument(
        "--freeze-mode",
        choices=["none", "edge_emb", "cand_emb_span", "mem_emb_rel", "edge", "synth_finetune"],
        default="none",
        help="Freeze all parameters except the trainable subset for the selected mode",
    )
    ap.add_argument(
        "--freeze-except-edge-emb",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--freeze-except-cand-emb-span",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--freeze-except-mem-emb-rel",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--freeze-except-edge",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    ap.add_argument("--edge-rel-pair-features", action="store_true", help="Add 5 scalar pair features to edge_rel_head")
    ap.add_argument("--edge-exist-weight", type=float, default=0.5)
    ap.add_argument("--hard-negative-weight", type=float, default=1.0)
    ap.add_argument("--hard-negative-max-per-row", type=int, default=8)
    ap.add_argument("--edge-rel-weight", type=float, default=0.25)
    ap.add_argument("--edge-rel-loss", choices=["ce", "focal"], default="ce")
    ap.add_argument("--edge-rel-focal-gamma", type=float, default=2.0)
    ap.add_argument("--edge-rel-class-weight", choices=["none", "inverse_freq"], default="none")
    ap.add_argument("--edge-rel-weight-min", type=float, default=0.25)
    ap.add_argument("--edge-rel-weight-max", type=float, default=6.0)
    ap.add_argument("--mem-kind-weight", type=float, default=0.5)
    ap.add_argument("--mem-rel-weight", type=float, default=0.25)
    ap.add_argument("--commit-weight", type=float, default=0.5)
    ap.add_argument("--commit-noop-weight", type=float, default=2.0)
    ap.add_argument("--commit-add-weight", type=float, default=1.0)
    ap.add_argument("--commit-other-weight", type=float, default=1.0)
    ap.add_argument("--pos-edge-weight", type=float, default=4.0)
    ap.add_argument("--pos-mem-attach-weight", type=float, default=4.0)
    ap.add_argument("--pos-mem-cover-weight", type=float, default=4.0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    train_rows = read_jsonl(args.train_jsonl)
    val_rows = read_jsonl(args.val_jsonl)
    train_ds = PredDataset(
        train_rows,
        hash_dim=args.hash_dim,
        spec_emb_cache=args.spec_emb_cache,
        cand_emb_cache=args.cand_emb_cache,
        mem_emb_cache=args.mem_emb_cache,
        synth_swap_prob=args.synth_swap_prob,
    )
    val_ds = PredDataset(
        val_rows,
        hash_dim=args.hash_dim,
        spec_emb_cache=args.spec_emb_cache,
        cand_emb_cache=args.cand_emb_cache,
        mem_emb_cache=args.mem_emb_cache,
    )
    val_synth_ds = None
    if args.eval_synth_val:
        val_synth_ds = PredDataset(
            val_rows,
            hash_dim=args.hash_dim,
            spec_emb_cache=args.spec_emb_cache,
            cand_emb_cache=args.cand_emb_cache,
            mem_emb_cache=args.mem_emb_cache,
            synth_swap_prob=1.0,
        )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    val_synth_loader = None if val_synth_ds is None else DataLoader(val_synth_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    edge_rel_pair_feat_dim = 5 if args.edge_rel_pair_features else 0
    model = PredAlignNet(
        hash_dim=args.hash_dim,
        hidden_dim=args.hidden_dim,
        edge_rel_pair_feat_dim=edge_rel_pair_feat_dim,
        spec_emb_dim=train_ds.spec_emb_dim,
        cand_emb_dim=train_ds.cand_emb_dim,
        mem_emb_dim=train_ds.mem_emb_dim,
    ).to(device)
    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location=device)
        state = ckpt.get("model", ckpt)
        inferred_dim = infer_edge_rel_pair_feat_dim_from_state(state, args.hidden_dim)
        inferred_spec_emb_dim = infer_spec_emb_dim_from_state(state)
        inferred_cand_emb_dim = infer_cand_emb_dim_from_state(state)
        inferred_mem_emb_dim = infer_mem_emb_dim_from_state(state)
        if inferred_spec_emb_dim == 0 and train_ds.spec_emb_dim > 0:
            inferred_spec_emb_dim = train_ds.spec_emb_dim
        if inferred_cand_emb_dim == 0 and train_ds.cand_emb_dim > 0:
            inferred_cand_emb_dim = train_ds.cand_emb_dim
        if inferred_mem_emb_dim == 0 and train_ds.mem_emb_dim > 0:
            inferred_mem_emb_dim = train_ds.mem_emb_dim
        if (
            inferred_dim != edge_rel_pair_feat_dim
            or inferred_spec_emb_dim != train_ds.spec_emb_dim
            or inferred_cand_emb_dim != train_ds.cand_emb_dim
            or inferred_mem_emb_dim != train_ds.mem_emb_dim
        ):
            model = PredAlignNet(
                hash_dim=args.hash_dim,
                hidden_dim=args.hidden_dim,
                edge_rel_pair_feat_dim=inferred_dim,
                spec_emb_dim=inferred_spec_emb_dim,
                cand_emb_dim=inferred_cand_emb_dim,
                mem_emb_dim=inferred_mem_emb_dim,
            ).to(device)
            args.edge_rel_pair_features = inferred_dim > 0
        missing, unexpected = model.load_state_dict(state, strict=False)
        allowed_missing = set()
        if inferred_spec_emb_dim > 0:
            allowed_missing.update({"spec_emb_proj.weight", "spec_emb_proj.bias"})
        if inferred_cand_emb_dim > 0:
            allowed_missing.update({"cand_emb_proj.weight", "cand_emb_proj.bias"})
        if inferred_mem_emb_dim > 0:
            allowed_missing.update({"mem_emb_proj.weight", "mem_emb_proj.bias"})
        unexpected = list(unexpected)
        disallowed_missing = [x for x in missing if x not in allowed_missing]
        if disallowed_missing or unexpected:
            raise RuntimeError(f"Unexpected checkpoint mismatch: missing={disallowed_missing}, unexpected={unexpected}")
    freeze_mode = resolve_freeze_mode(args)
    apply_freeze_mode(model, freeze_mode)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_param_count = sum(p.numel() for p in model.parameters())
    trainable_param_count = sum(p.numel() for p in trainable_params)
    print(json.dumps({
        "trainable_params": trainable_param_count,
        "total_params": total_param_count,
        "trainable_fraction": round(trainable_param_count / max(total_param_count, 1), 6),
        "freeze_mode": freeze_mode,
    }))
    if not trainable_params:
        raise RuntimeError("No trainable parameters selected")
    opt = torch.optim.AdamW(trainable_params, lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_metric = -1.0
    best_gold_metric = -1.0
    best_cover_metric = -1.0
    history: List[Dict[str, Any]] = []

    pos_weight = torch.tensor(float(args.pos_edge_weight), device=device)
    commit_class_weight = torch.tensor(
        [
            float(args.commit_noop_weight),
            float(args.commit_add_weight),
            float(args.commit_other_weight),
        ],
        device=device,
    )
    edge_rel_class_weight = None
    if args.edge_rel_class_weight == "inverse_freq":
        edge_rel_class_weight = compute_edge_rel_class_weights(
            train_rows,
            device=device,
            min_weight=args.edge_rel_weight_min,
            max_weight=args.edge_rel_weight_max,
        )
        print("edge_rel_class_weight", {
            rel: round(float(edge_rel_class_weight[i].item()), 4)
            for rel, i in REL_WITH_NONE_TO_ID.items()
        })

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = {"loss": 0.0, "span": 0.0, "commit": 0.0, "edge_exist": 0.0, "edge_rel": 0.0, "mem_kind": 0.0, "mem_rel": 0.0}
        batches = 0
        for batch in train_loader:
            batch = to_device(batch, device)
            out = model(batch)

            span_logits = out["span_logits"].reshape(-1, out["span_logits"].size(-1))
            y_span = batch.y_span.reshape(-1)
            spec_valid = batch.spec_mask.reshape(-1)
            valid_logits = span_logits[spec_valid]
            valid_targets = y_span[spec_valid]
            span_loss = F.cross_entropy(valid_logits, valid_targets)

            commit_loss = F.cross_entropy(out["commit_logits"], batch.y_commit, weight=commit_class_weight)

            edge_mask = batch.edge_mask
            edge_weight = torch.ones_like(batch.y_edge_exist)
            hard_negative_mask = build_edge_hard_negative_mask(
                batch.y_edge_exist,
                edge_mask,
                max_per_row=args.hard_negative_max_per_row,
            )
            if args.hard_negative_weight != 1.0:
                edge_weight[hard_negative_mask] = float(args.hard_negative_weight)
            edge_exist_loss = F.binary_cross_entropy_with_logits(
                out["edge_exist_logits"][edge_mask],
                batch.y_edge_exist[edge_mask],
                weight=edge_weight[edge_mask],
                pos_weight=pos_weight,
            )

            gold_edge_mask = (batch.y_edge_exist > 0.0) & edge_mask
            if gold_edge_mask.any():
                edge_rel_loss = edge_relation_loss_fn(
                    out["edge_rel_logits"][gold_edge_mask],
                    batch.y_edge_rel[gold_edge_mask],
                    mode=args.edge_rel_loss,
                    class_weight=edge_rel_class_weight,
                    focal_gamma=args.edge_rel_focal_gamma,
                )
            else:
                edge_rel_loss = out["edge_rel_logits"].sum() * 0.0

            mem_mask = batch.mem_mask[:, None, :].expand_as(batch.y_mem_kind)
            mem_kind_logits = out["mem_kind_logits"][mem_mask]
            mem_kind_targets = batch.y_mem_kind[mem_mask]
            mem_kind_weight = torch.tensor(
                [1.0, float(args.pos_mem_attach_weight), float(args.pos_mem_cover_weight)],
                device=device,
            )
            mem_kind_loss = F.cross_entropy(mem_kind_logits, mem_kind_targets, weight=mem_kind_weight)

            mem_attach_mask = (batch.y_mem_kind == MEM_LINK_KIND_TO_ID["attach"]) & mem_mask
            if mem_attach_mask.any():
                mem_rel_loss = F.cross_entropy(out["mem_rel_logits"][mem_attach_mask], batch.y_mem_rel[mem_attach_mask])
            else:
                mem_rel_loss = out["mem_rel_logits"].sum() * 0.0

            loss = (
                span_loss
                + args.commit_weight * commit_loss
                + args.edge_exist_weight * edge_exist_loss
                + args.edge_rel_weight * edge_rel_loss
                + args.mem_kind_weight * mem_kind_loss
                + args.mem_rel_weight * mem_rel_loss
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running["loss"] += float(loss.item())
            running["span"] += float(span_loss.item())
            running["commit"] += float(commit_loss.item())
            running["edge_exist"] += float(edge_exist_loss.item())
            running["edge_rel"] += float(edge_rel_loss.item())
            running["mem_kind"] += float(mem_kind_loss.item())
            running["mem_rel"] += float(mem_rel_loss.item())
            batches += 1

        train_avg = {k: v / max(batches, 1) for k, v in running.items()}
        val_gold_metrics = compute_metrics(model, val_loader, device)
        val_synth_metrics = compute_metrics(model, val_synth_loader, device) if val_synth_loader is not None else None
        score_target = val_synth_metrics if val_synth_metrics is not None else val_gold_metrics
        score = (
            score_target["row_complete_rate"]
            + 0.5 * score_target["span_top1_acc_nonnull"]
            + 0.25 * score_target["edge_f1"]
            + 0.25 * score_target["cover_f1"]
        )
        gold_score = (
            val_gold_metrics["row_complete_rate"]
            + 0.5 * val_gold_metrics["span_top1_acc_nonnull"]
            + 0.25 * val_gold_metrics["edge_f1"]
            + 0.25 * val_gold_metrics["cover_f1"]
        )
        cover_score = val_gold_metrics["cover_f1"] + val_gold_metrics["cover_recall"]
        record = {
            "epoch": epoch,
            "train": train_avg,
            "val_gold": val_gold_metrics,
            "val_synth": val_synth_metrics,
            "score": score,
            "gold_score": gold_score,
            "cover_score": cover_score,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))

        if score > best_metric:
            best_metric = score
            torch.save({
                "model": model.state_dict(),
                "args": vars(args),
                "best_score": best_metric,
                "history": history,
            }, out_dir / "best_pred_v1.pt")

        if gold_score > best_gold_metric:
            best_gold_metric = gold_score
            torch.save({
                "model": model.state_dict(),
                "args": vars(args),
                "best_gold_score": best_gold_metric,
                "history": history,
            }, out_dir / "best_gold_pred_v1.pt")

        if cover_score > best_cover_metric:
            best_cover_metric = cover_score
            torch.save({
                "model": model.state_dict(),
                "args": vars(args),
                "best_cover_score": best_cover_metric,
                "history": history,
            }, out_dir / "best_cover_pred_v1.pt")

    final = {"best_score": best_metric, "best_gold_score": best_gold_metric, "best_cover_score": best_cover_metric, "history": history}
    (out_dir / "train_history.json").write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
