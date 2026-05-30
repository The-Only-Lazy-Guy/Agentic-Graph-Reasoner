"""Phase 15 corpus → Phase 16 training dataset.

Each sample maps one V4 session to supervision targets for all six aux heads:

  anchor_mask        [N] float  — 1.0 for nodes V4 accessed (NodeHead target)
  slot_fill_target   [NUM_SLOTS] float — which required slots were filled
  epistemic_target   [N] float  — 1.0 if node received add_epistemic_state patch
  invalidator_target [N] float  — 1.0 if node received deprecate_fact patch
  shortcut_valid     scalar     — 1.0 if task finalized and shortcut-eligible

Text embeddings are not stored here; the training loop calls the embedder
(BERT) on node_texts to produce [N, 768] tensors at collation time.
GNN forward + cross-attention are called inside the trainer with frozen Qwen3.

Usage:
    ds = Phase15Dataset("artifacts/phase15/phase15_corpus.jsonl")
    loader = DataLoader(ds, batch_size=1, collate_fn=phase15_collate_fn)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset

from v5.goal_encoder import (
    NUM_SLOTS,
    SLOT_VOCAB,
    TASK_FAMILY_VOCAB,
    QUESTION_MODE_VOCAB,
)

SLOT_ID: Dict[str, int] = {s: i for i, s in enumerate(SLOT_VOCAB)}

# Corpus slot names → canonical SLOT_VOCAB entries
# V4 task frames use task-specific slot names; map them to the nearest canonical slot.
CORPUS_SLOT_ALIAS: Dict[str, str] = {
    "answer": "verdict",
    "explanation": "reason",
    "relationship": "definition",
    "problem_frame": "definition",
    "core_structure": "definition",
    "rank_query": "definition",
    "pagination": "condition",
    "tie_policy": "condition",
    "scale_architecture": "alternative",
    "latency_budget": "complexity",
    "consistency_model": "condition",
    "failure_mode_fix": "alternative",
}

# Shortcut = finalized in ≤ half of max_steps
SHORTCUT_STEP_RATIO = 0.5

# Patch types that indicate epistemic clarification was needed
EPISTEMIC_PATCH_TYPES = frozenset({"add_epistemic_state"})
# Patch types that indicate an invalidator fired on a node
INVALIDATOR_PATCH_TYPES = frozenset({"deprecate_fact"})

# Substrate-node-adding patch types -> the node_type they create. These are the
# reasoning substrate V4 writes; their patch target nodes are the planning/
# evidence-pool nodes a trace engaged (the planning supervision signal).
SUBSTRATE_PATCH_TYPE_TO_NODE = {
    "add_strategy": "strategy",
    "add_failure_pattern": "failure_pattern",
    "add_control_rule": "control_rule",
    "add_reasoning_atom": "reasoning_atom",
    "add_reasoning_chain": "reasoning_chain",
    "add_solved_subgoal": "solved_subgoal",
    "add_epistemic_state": "epistemic_state",
}
SAFE_PATCH_STATUSES = frozenset({"accept", "soft_only"})


@dataclass
class Phase15Sample:
    """One training sample. All list fields are ordered by node_ids index."""
    session_id: str
    question: str
    answer_text: str
    finalized: bool

    node_ids: List[str]
    node_texts: Dict[str, str]    # node_id -> text (for BERT at collation)
    node_types: Dict[str, str]    # node_id -> node_type string

    task_frame: dict              # task_family, question_mode, required_slots

    # Supervision targets (parallel to node_ids)
    anchor_mask: List[float]       # [N] — NodeHead target
    epistemic_target: List[float]  # [N] — EpistemicHead target
    invalidator_target: List[float]  # [N] — InvalidatorHead target

    # Global targets
    slot_fill_target: List[float]  # [NUM_SLOTS] — SlotHead target
    shortcut_valid: float          # ShortcutHead target

    # Reasoning substrate this trace wrote (safe add_* substrate patches):
    # node_id -> {"type": str, "text": str, "status": str}. These are the
    # planning/evidence-pool nodes the trace engaged — the planning supervision
    # the base-graph anchors lack. Empty until V4 writes substrate / patches are
    # applied. Default factory keeps older constructors working.
    substrate_nodes: Dict[str, dict] = field(default_factory=dict)


class Phase15Dataset(Dataset):
    """Loads phase15_corpus.jsonl and exposes Phase15Sample instances.

    Filters out rows where finalized=False and no anchors (no training signal).
    """

    def __init__(self, corpus_path: str | Path):
        self.samples: List[Phase15Sample] = []
        raw = _load_jsonl(corpus_path)
        for row in raw:
            sample = _parse_row(row)
            if sample is not None:
                self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Phase15Sample:
        return self.samples[idx]


def phase15_collate_fn(batch: List[Phase15Sample]):
    """Minimal collation — returns list (variable N per sample).

    The trainer handles per-sample GNN forward + cross-attention;
    true batching requires padded node tensors and is deferred to Phase 17.
    """
    return batch


# ── parsing helpers ─────────────────────────────────────────────────────────


def _load_jsonl(path: str | Path) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _parse_row(row: dict) -> Optional[Phase15Sample]:
    """Convert one corpus row to a Phase15Sample. Returns None to skip."""
    inp = row.get("input", {})
    anchors = inp.get("anchors") or []
    if not anchors:
        return None  # no nodes → no training signal

    session_id = row.get("session_id", "")
    question = inp.get("question", "")
    outputs = row.get("outputs", {})
    answer_text = outputs.get("answer_polished") or outputs.get("answer_raw") or ""
    metrics = row.get("metrics", {})
    finalized = bool(metrics.get("finalized", False))

    # Build node lists
    node_ids, node_texts, node_types = _extract_nodes(anchors)
    N = len(node_ids)
    if N == 0:
        return None

    # Task frame for GoalEncoder
    task_frame = _extract_task_frame(inp, metrics)

    # Patch-based supervision targets
    patches = row.get("trace", {}).get("scoped_patches") or []
    epistemic_nodes = _nodes_with_patch_type(patches, EPISTEMIC_PATCH_TYPES)
    invalidator_nodes = _nodes_with_patch_type(patches, INVALIDATOR_PATCH_TYPES)
    substrate_nodes = _extract_substrate_nodes(patches)

    anchor_set = {n["id"] for n in anchors if isinstance(n, dict) and "id" in n}

    anchor_mask = [1.0 if nid in anchor_set else 0.0 for nid in node_ids]
    epistemic_target = [1.0 if nid in epistemic_nodes else 0.0 for nid in node_ids]
    invalidator_target = [1.0 if nid in invalidator_nodes else 0.0 for nid in node_ids]

    # Slot fill target
    slot_fill_target = _build_slot_fill_target(metrics)

    # Shortcut: finalized AND steps <= SHORTCUT_STEP_RATIO * max_steps
    steps = metrics.get("steps", 99)
    max_steps = metrics.get("max_steps", 4)
    shortcut_valid = 1.0 if (finalized and steps <= SHORTCUT_STEP_RATIO * max_steps) else 0.0

    return Phase15Sample(
        session_id=session_id,
        question=question,
        answer_text=answer_text,
        finalized=finalized,
        node_ids=node_ids,
        node_texts=node_texts,
        node_types=node_types,
        task_frame=task_frame,
        anchor_mask=anchor_mask,
        epistemic_target=epistemic_target,
        invalidator_target=invalidator_target,
        slot_fill_target=slot_fill_target,
        shortcut_valid=shortcut_valid,
        substrate_nodes=substrate_nodes,
    )


def _extract_substrate_nodes(patches: list) -> Dict[str, dict]:
    """Collect safe substrate add_node patches: node_id -> {type, text, status}.

    Only `accept`/`soft_only` patches are taken (needs_review/reject are held
    back, matching V4's apply gate). Fresh epistemic_state nodes default to
    'uncertain' status -> planning pool (an asserted-but-unverified belief).
    """
    out: Dict[str, dict] = {}
    for p in patches:
        if not isinstance(p, dict):
            continue
        ntype = SUBSTRATE_PATCH_TYPE_TO_NODE.get(p.get("patch_type"))
        if ntype is None:
            continue
        status = (p.get("validation") or {}).get("status")
        if status not in SAFE_PATCH_STATUSES:
            continue
        raw = p.get("raw_edit") or {}
        nid = raw.get("node_id") or p.get("target_id")
        if not nid:
            continue
        epi_status = "uncertain" if ntype == "epistemic_state" else "unknown"
        out[nid] = {"type": ntype, "text": p.get("text") or "", "status": epi_status}
    return out


def _extract_nodes(
    anchors: list,
) -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
    """Return (node_ids, node_texts, node_types) from anchor list."""
    node_ids = []
    node_texts = {}
    node_types = {}
    seen = set()
    for a in anchors:
        if not isinstance(a, dict):
            continue
        nid = a.get("id")
        if not nid or nid in seen:
            continue
        seen.add(nid)
        node_ids.append(nid)
        node_texts[nid] = a.get("text") or ""
        node_types[nid] = a.get("node_type") or "unknown"
    return node_ids, node_texts, node_types


def _extract_task_frame(inp: dict, metrics: dict) -> dict:
    """Build minimal task_frame compatible with GoalEncoder.encode_task_frame."""
    slot_stats = metrics.get("slot_fill_stats") or {}
    required_slots = slot_stats.get("required_slots") or []
    filled_slots = slot_stats.get("filled_slots") or []

    task_family = inp.get("controller_task_family") or inp.get("task_type") or "unknown"
    question_mode = inp.get("task_type") or "unknown"

    return {
        "task_family": task_family,
        "question_mode": question_mode,
        "required_slots": required_slots,
        "filled_slots": filled_slots,
    }


def _nodes_with_patch_type(patches: list, patch_types: frozenset) -> set:
    """Return set of node IDs touched by patches of given type(s)."""
    nodes = set()
    for p in patches:
        if not isinstance(p, dict):
            continue
        if p.get("patch_type") in patch_types:
            target = p.get("target_id")
            if target:
                nodes.add(target)
            for nid in (p.get("evidence_node_ids") or []):
                nodes.add(nid)
            for nid in (p.get("affected_node_ids") or []):
                nodes.add(nid)
    return nodes


def _build_slot_fill_target(metrics: dict) -> List[float]:
    """Build [NUM_SLOTS] float target from slot_fill_stats.

    Corpus uses task-specific slot names; maps via CORPUS_SLOT_ALIAS to
    canonical SLOT_VOCAB before indexing.
    """
    slot_stats = metrics.get("slot_fill_stats") or {}
    filled_raw = set(slot_stats.get("filled_slots") or [])

    # Normalize: corpus name -> canonical SLOT_VOCAB name
    filled = set()
    for name in filled_raw:
        canonical = CORPUS_SLOT_ALIAS.get(name, name)
        filled.add(canonical)

    target = [0.0] * NUM_SLOTS
    for slot_name, idx in SLOT_ID.items():
        if slot_name in filled:
            target[idx] = 1.0
    return target
