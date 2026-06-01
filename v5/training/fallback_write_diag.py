"""Fallback/write diagnostics for the projected V5 corpus.

This is the "do not blindly rerun schedule/model yet" harness. It runs the same
integrated Stage 1 -> 2A -> 2B path as corpus_scaling, then decomposes the
held-out failures:

  - label distributions by split and case type
  - fallback trip reasons by case type
  - applicable-only slot/epistemic/planning calibration breakdown
  - applicable failure focus: routing vs confidence calibration
  - direct_judgment routing audit with top-k plan/evidence label checks
  - direct_judgment failure table
  - negative/no-graph safety table
  - one-condition-at-a-time oracle fallback ablations
  - write ratio by case type, fallback state, and block
  - planning miss categories

Usage:
    $env:KMP_DUPLICATE_LIB_OK="TRUE"; python -u -m v5.training.fallback_write_diag --corpus data/corpus_merged_v5proj.jsonl
"""
from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from v5.cross_attention import V5AttentionAdapter
from v5.exit_condition import (
    EPISTEMIC_THRESHOLD,
    SHORTCUT_THRESHOLD,
    SLOT_FILL_THRESHOLD,
    _force_fallback,
    _required_slot_indices,
    _top_k_indices,
)
from v5.gnn_encoder import RGCNEncoder
from v5.goal_encoder import SLOT_VOCAB
from v5.training.bridge import corpus_to_stage1_examples, load_persisted_graph
from v5.training.corpus_scaling import _stratified_split
from v5.training.providers import FrozenQwenHInitProvider, RealEmbedder
from v5.training.stage1 import Stage1Config, Stage1Example, Stage1Trainer
from v5.training.stage1_real import _graph_path
from v5.training.stage2 import GATE_INIT, Stage2Config, Stage2Trainer
from v5.training.stage2b_real import make_real_negatives
from v5.training.substrate import DEFAULT_OUT as SUBSTRATE_OUT
from v5.training.substrate import build_substrate_graph
from v5.subgraph import INVALIDATOR_RELATIONS

DEFAULT_LM = "Qwen/Qwen2.5-0.5B-Instruct"
SUPPORTIVE_RELATIONS = frozenset({
    "support",
    "supports",
    "entails",
    "leveraged",
    "derived_from",
    "epistemic_of",
    "related",
    "overlaps",
    "refine",
    "refines",
    "depend",
    "depends_on",
})
SUBSTRATE_NODE_TYPES = frozenset({
    "strategy",
    "failure_pattern",
    "control_rule",
    "reasoning_atom",
    "reasoning_chain",
    "derived_reasoning",
    "solved_subgoal",
    "epistemic_state",
})


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class EvalRecord:
    ex: Stage1Example
    planning_state: object
    evidence_state: object


def _mean(values: Sequence[float]) -> float:
    return sum(values) / max(1, len(values))


def _fmt_rate(num: int, den: int) -> str:
    return f"{num}/{den} ({num / max(1, den):.2f})"


def _task_family(ex: Stage1Example) -> str:
    return str((ex.task_frame or {}).get("task_family") or "unknown")


def _required_slots(task_frame: Optional[dict]) -> List[int]:
    return _required_slot_indices(task_frame) or []


def _values_at(tensor, indices: Iterable[int]) -> List[float]:
    if tensor is None:
        return []
    flat = tensor.squeeze(0)
    return [float(flat[i].item()) for i in indices]


def _slots_ok(es, ex: Stage1Example, *, gold: bool = False) -> Tuple[bool, List[int]]:
    req = _required_slots(ex.task_frame)
    if not req:
        return True, []
    if gold and ex.slot_target is not None:
        vals = _values_at(ex.slot_target, req)
        threshold = 0.5
    else:
        vals = _values_at(es.slot_state_r, req)
        threshold = SLOT_FILL_THRESHOLD
    missing = [idx for idx, val in zip(req, vals) if val < threshold]
    return len(missing) == 0, missing


def _no_inv_top(es, ex: Stage1Example, top: Sequence[int], *, gold: bool = False) -> bool:
    if not top:
        return True
    if gold and ex.inv_target is not None:
        inv = ex.inv_target.squeeze(0)
    else:
        inv = es.invalidator_flags_r.squeeze(0)
    return not any(float(inv[i].item()) > 0.5 for i in top)


def _epi_primary_ok(es, ex: Stage1Example, primary: Optional[int], *, gold: bool = False) -> bool:
    if primary is None:
        return False
    if gold and ex.epi_target is not None:
        return float(ex.epi_target.squeeze(0)[primary].item()) >= 0.5
    return float(es.epistemic_confidence_r.squeeze(0)[primary].item()) >= EPISTEMIC_THRESHOLD


def _shortcut_ok(es, ex: Stage1Example, *, gold: bool = False) -> bool:
    if gold and ex.shortcut_target is not None:
        return float(ex.shortcut_target.item()) >= 0.5
    return float(es.shortcut_validity_r.item()) >= SHORTCUT_THRESHOLD


def fallback_bits(
    es,
    ex: Stage1Example,
    *,
    gold_slots: bool = False,
    gold_epi: bool = False,
    gold_inv: bool = False,
    gold_shortcut: bool = False,
) -> Dict[str, object]:
    """Return current fallback decision plus individual condition failures.

    Shortcut is reported for calibration, but current fallback_needed() does not
    gate on it. The oracle shortcut variant is therefore expected to leave the
    fallback decision unchanged.
    """
    max_loops = es.exit_reason == "max_loops_reached"
    empty_pool = not bool(ex.graph_kv.evidence_mask.any().item())
    top = _top_k_indices(es.node_scores_r) if es.node_scores_r.shape[-1] else []
    primary = top[0] if top else None
    slots_ok, missing_slots = _slots_ok(es, ex, gold=gold_slots)
    no_inv = _no_inv_top(es, ex, top, gold=gold_inv)
    epi_ok = _epi_primary_ok(es, ex, primary, gold=gold_epi)
    shortcut_ok = _shortcut_ok(es, ex, gold=gold_shortcut)
    forced_fallback = _force_fallback(ex.task_frame)

    fallback = bool(forced_fallback or (max_loops and (empty_pool or not (slots_ok and no_inv and epi_ok))))
    return {
        "fallback": fallback,
        "exit_reason": es.exit_reason,
        "forced_fallback": forced_fallback,
        "max_loops": max_loops,
        "empty_pool": empty_pool,
        "missing_slot": not slots_ok,
        "missing_slot_ids": missing_slots,
        "invalidator_active": not no_inv,
        "low_epistemic": not epi_ok,
        "shortcut_invalid": not shortcut_ok,
        "primary_idx": primary,
        "top_indices": top,
    }


@torch.no_grad()
def _eval_records(adapter: V5AttentionAdapter, examples: Sequence[Stage1Example]) -> List[EvalRecord]:
    adapter.eval()
    records: List[EvalRecord] = []
    for ex in examples:
        _, ps, _ = adapter.run_planning(
            ex.h_init, ex.goal, ex.graph_kv, ex.node_ids, task_frame=ex.task_frame)
        _, es, _ = adapter.run_evidence(
            ps.h_r, ex.goal, ex.graph_kv, ex.node_ids, task_frame=ex.task_frame)
        records.append(EvalRecord(ex=ex, planning_state=ps, evidence_state=es))
    return records


def _label_distribution(name: str, examples: Sequence[Stage1Example]) -> None:
    print(f"\n[{name}] label distribution over {len(examples)} examples")
    print(f"  case_type: {dict(Counter(ex.tag for ex in examples))}")

    def _rows_with(attr: str) -> int:
        return sum(getattr(ex, attr) is not None for ex in examples)

    plan_pos = sum(int((ex.plan_anchor > 0).sum().item()) for ex in examples if ex.plan_anchor is not None)
    evid_pos = sum(int((ex.evid_anchor > 0).sum().item()) for ex in examples if ex.evid_anchor is not None)
    epi_pos = sum(int((ex.epi_target > 0.5).sum().item()) for ex in examples if ex.epi_target is not None)
    epi_total = sum(int(ex.epi_target.numel()) for ex in examples if ex.epi_target is not None)
    inv_pos = sum(int((ex.inv_target > 0.5).sum().item()) for ex in examples if ex.inv_target is not None)
    inv_total = sum(int(ex.inv_target.numel()) for ex in examples if ex.inv_target is not None)
    shortcut_pos = sum(
        int(ex.shortcut_target.item() >= 0.5)
        for ex in examples if ex.shortcut_target is not None
    )
    shortcut_rows = _rows_with("shortcut_target")

    print(f"  plan_anchor rows: {_rows_with('plan_anchor')}  positive nodes: {plan_pos}")
    print(f"  evid_anchor rows: {_rows_with('evid_anchor')}  positive nodes: {evid_pos}")
    print(f"  epi rows: {_rows_with('epi_target')}  positives: {epi_pos}/{max(1, epi_total)}")
    print(f"  inv rows: {_rows_with('inv_target')}  positives: {inv_pos}/{max(1, inv_total)}")
    print(f"  shortcut positives: {shortcut_pos}/{max(1, shortcut_rows)}")

    slot_counts: Counter[str] = Counter()
    slot_rows = 0
    for ex in examples:
        if ex.slot_target is None:
            continue
        slot_rows += 1
        target = ex.slot_target.squeeze(0)
        for idx, val in enumerate(target.tolist()):
            if val >= 0.5:
                slot_counts[SLOT_VOCAB[idx]] += 1
    print(f"  slot rows: {slot_rows}  positive slots: {dict(slot_counts)}")


def _fallback_trip_report(records: Sequence[EvalRecord]) -> None:
    print("\n=== fallback trip reasons (held-out) ===")
    by_tag: Dict[str, List[EvalRecord]] = defaultdict(list)
    for rec in records:
        by_tag[rec.ex.tag].append(rec)

    reason_keys = [
        "forced_fallback",
        "max_loops",
        "empty_pool",
        "missing_slot",
        "low_epistemic",
        "invalidator_active",
        "shortcut_invalid",
    ]
    for tag in sorted(by_tag):
        items = by_tag[tag]
        counts = Counter()
        fb = 0
        missing_slots: Counter[str] = Counter()
        for rec in items:
            bits = fallback_bits(rec.evidence_state, rec.ex)
            fb += int(bits["fallback"])
            for key in reason_keys:
                counts[key] += int(bool(bits[key]))
            for idx in bits["missing_slot_ids"]:
                missing_slots[SLOT_VOCAB[idx]] += 1
        print(f"  {tag:10s} fallback={_fmt_rate(fb, len(items))}")
        for key in reason_keys:
            suffix = " (telemetry; not in current fallback gate)" if key == "shortcut_invalid" else ""
            print(f"    {key:18s} {_fmt_rate(counts[key], len(items))}{suffix}")
        if missing_slots:
            print(f"    missing slots      {dict(missing_slots)}")


def _task_family_report(records: Sequence[EvalRecord]) -> None:
    print("\n=== fallback/write by task_family (held-out) ===")
    buckets = defaultdict(lambda: {
        "n": 0,
        "fb": 0,
        "missing_slot": 0,
        "low_epistemic": 0,
        "invalidator_active": 0,
        "plan_wr": [],
        "evid_wr": [],
    })
    tags_by_family: Dict[str, Counter[str]] = defaultdict(Counter)
    for rec in records:
        family = _task_family(rec.ex)
        bits = fallback_bits(rec.evidence_state, rec.ex)
        bucket = buckets[family]
        bucket["n"] += 1
        bucket["fb"] += int(bits["fallback"])
        bucket["missing_slot"] += int(bits["missing_slot"])
        bucket["low_epistemic"] += int(bits["low_epistemic"])
        bucket["invalidator_active"] += int(bits["invalidator_active"])
        bucket["plan_wr"].extend(rec.planning_state.write_ratios or [])
        bucket["evid_wr"].extend(rec.evidence_state.write_ratios or [])
        tags_by_family[family][rec.ex.tag] += 1

    print(
        f"{'family':28s} {'n':>4s} {'fallback':>8s} {'slot':>8s} "
        f"{'epi':>8s} {'inv':>8s} {'plan_wr':>8s} {'evid_wr':>8s} {'tags':>24s}"
    )
    for family, bucket in sorted(buckets.items(), key=lambda item: (-item[1]["n"], item[0])):
        n = bucket["n"]
        print(
            f"{family[:28]:28s} {n:>4d} "
            f"{bucket['fb'] / max(1, n):>8.2f} "
            f"{bucket['missing_slot'] / max(1, n):>8.2f} "
            f"{bucket['low_epistemic'] / max(1, n):>8.2f} "
            f"{bucket['invalidator_active'] / max(1, n):>8.2f} "
            f"{_mean(bucket['plan_wr']):>8.3f} "
            f"{_mean(bucket['evid_wr']):>8.3f} "
            f"{str(dict(tags_by_family[family])):>24s}"
        )


def _short(text: object, limit: int = 110) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= limit else s[: limit - 3] + "..."


def _node_summary(graph, node_id: str) -> Tuple[str, str]:
    if graph is None or node_id not in graph.nodes:
        return "unknown", ""
    node = graph.nodes[node_id]
    return str(getattr(node, "node_type", "unknown")), _short(getattr(node, "text", ""))


def _edge_context_lines(
    graph,
    node_id: str,
    active_ids: Optional[Sequence[str]] = None,
    limit: int = 8,
) -> List[str]:
    if graph is None:
        return []
    edges = [
        edge for edge in getattr(graph, "edges", []) or []
        if edge.src == node_id or edge.dst == node_id
    ]
    if not edges:
        return ["    edges: none touching persisted graph node"]

    counts = Counter(
        f"{'out' if edge.src == node_id else 'in'}:{edge.relation}"
        for edge in edges
    )
    lines = [f"    edge_counts: {dict(counts.most_common(10))}"]

    interesting = [
        edge for edge in edges
        if edge.relation in INVALIDATOR_RELATIONS or edge.relation in SUPPORTIVE_RELATIONS
    ]
    interesting.sort(key=lambda e: (e.relation not in INVALIDATOR_RELATIONS, e.relation, e.src, e.dst))
    active_set = set(active_ids or [])
    for edge in interesting[:limit]:
        direction = "out" if edge.src == node_id else "in"
        other_id = edge.dst if direction == "out" else edge.src
        other_type, other_text = _node_summary(graph, other_id)
        malformed = " self_edge" if edge.src == edge.dst else ""
        in_scope = " active_peer" if other_id in active_set else ""
        if direction == "out":
            rel = f"{node_id} --{edge.relation}--> {other_id}"
        else:
            rel = f"{other_id} --{edge.relation}--> {node_id}"
        lines.append(f"    {direction}: {rel}{malformed}{in_scope} other_type={other_type} text={other_text}")
    return lines


def _invalidator_case_report(records: Sequence[EvalRecord], limit: int, graph=None) -> None:
    print("\n=== invalidator-active cases for manual inspection ===")
    shown = 0
    for rec in records:
        bits = fallback_bits(rec.evidence_state, rec.ex)
        if not bits["invalidator_active"]:
            continue
        shown += 1
        top = list(bits["top_indices"])[:3]
        missing = [SLOT_VOCAB[idx] for idx in bits["missing_slot_ids"]]
        print(
            f"\ncase {shown}: tag={rec.ex.tag} family={_task_family(rec.ex)} "
            f"fallback={bits['fallback']} missing_slots={missing}"
        )
        pred_inv = rec.evidence_state.invalidator_flags_r.squeeze(0)
        pred_epi = rec.evidence_state.epistemic_confidence_r.squeeze(0)
        gold_inv = rec.ex.inv_target.squeeze(0) if rec.ex.inv_target is not None else None
        gold_epi = rec.ex.epi_target.squeeze(0) if rec.ex.epi_target is not None else None
        attn = rec.evidence_state.attn_history[-1].squeeze(0) if rec.evidence_state.attn_history else None
        for rank, idx in enumerate(top, start=1):
            node_id = rec.ex.node_ids[idx]
            node_type = rec.ex.graph_kv.node_types[idx]
            static_inv = float(rec.ex.graph_kv.invalidator_flags[idx].item())
            inv_gold = float(gold_inv[idx].item()) if gold_inv is not None else float("nan")
            epi_gold = float(gold_epi[idx].item()) if gold_epi is not None else float("nan")
            attn_val = float(attn[idx].item()) if attn is not None else float("nan")
            persisted_type, persisted_text = _node_summary(graph, node_id)
            print(
                f"  top{rank}: id={node_id} type={node_type} attn={attn_val:.3f} "
                f"static_inv={static_inv:.0f} "
                f"pred_inv={float(pred_inv[idx].item()):.3f} gold_inv={inv_gold:.1f} "
                f"pred_epi={float(pred_epi[idx].item()):.3f} gold_epi={epi_gold:.1f}"
            )
            if persisted_text:
                print(f"    persisted_type={persisted_type} text={persisted_text}")
            if float(pred_inv[idx].item()) > 0.5 or static_inv > 0.5:
                for line in _edge_context_lines(graph, node_id, rec.ex.node_ids):
                    print(line)
        if shown >= limit:
            break
    if shown == 0:
        print("  no held-out cases had active invalidators on fallback top-k nodes")


def _slot_rows(rec: EvalRecord) -> List[Tuple[int, str, float, float, float, bool]]:
    req = _required_slots(rec.ex.task_frame)
    pred = rec.evidence_state.slot_state_r.squeeze(0)
    gold = rec.ex.slot_target.squeeze(0) if rec.ex.slot_target is not None else None
    rows = []
    for idx in req:
        pred_val = float(pred[idx].item())
        gold_val = float(gold[idx].item()) if gold is not None else float("nan")
        rows.append((
            idx,
            SLOT_VOCAB[idx],
            pred_val,
            gold_val,
            pred_val - SLOT_FILL_THRESHOLD,
            pred_val < SLOT_FILL_THRESHOLD,
        ))
    return rows


def _gold_indices(tensor: Optional[torch.Tensor]) -> List[int]:
    if tensor is None:
        return []
    return torch.where(tensor.squeeze(0) > 0.0)[0].tolist()


def _state_scores(state) -> Optional[torch.Tensor]:
    scores = getattr(state, "node_scores_r", None)
    if scores is None:
        return None
    flat = scores.squeeze(0)
    if flat.numel() == 0:
        return None
    return flat


def _top_indices(scores: Optional[torch.Tensor], k: int) -> List[int]:
    if scores is None:
        return []
    k = min(k, int(scores.numel()))
    if k <= 0:
        return []
    return [int(idx) for idx in scores.topk(k).indices.tolist()]


def _sorted_gold_indices(tensor: Optional[torch.Tensor], limit: int = 5) -> List[int]:
    indices = _gold_indices(tensor)
    if tensor is None:
        return indices[:limit]
    weights = tensor.squeeze(0)
    indices.sort(key=lambda idx: (-float(weights[idx].item()), idx))
    return indices[:limit]


def _routing_metrics(
    records: Sequence[EvalRecord],
    anchor_attr: str,
    state_attr: str,
    mask_attr: str,
    k: int = 3,
) -> Dict[str, float]:
    total = 0
    p1 = 0
    hit_k = 0
    recall_k = 0.0
    recall_gold = 0.0
    gold_total = 0
    raw_gold_total = 0
    in_pool_gold_total = 0
    for rec in records:
        anchor = getattr(rec.ex, anchor_attr)
        scores = _state_scores(getattr(rec, state_attr))
        if anchor is None or scores is None:
            continue
        raw_gold = anchor.squeeze(0) > 0.0
        if not bool(raw_gold.any().item()):
            continue
        mask = getattr(rec.ex.graph_kv, mask_attr)
        gold = raw_gold & mask.bool()
        raw_gold_total += int(raw_gold.sum().item())
        in_pool_gold_total += int(gold.sum().item())
        if not bool(gold.any().item()):
            continue
        n_gold = int(gold.sum().item())
        top1 = _top_indices(scores, 1)
        topk = _top_indices(scores, k)
        top_gold = _top_indices(scores, n_gold)
        if not top1:
            continue
        total += 1
        gold_total += n_gold
        p1 += int(bool(gold[top1[0]].item()))
        hit_k += int(bool(gold[topk].any().item())) if topk else 0
        recall_k += float(gold[topk].float().sum().item()) / max(1, n_gold) if topk else 0.0
        recall_gold += (
            float(gold[top_gold].float().sum().item()) / max(1, n_gold)
            if top_gold else 0.0
        )
    return {
        "n": float(total),
        "avg_gold": gold_total / max(1, total),
        "pool_cov": in_pool_gold_total / max(1, raw_gold_total),
        "p1": p1 / max(1, total),
        f"hit{k}": hit_k / max(1, total),
        f"r{k}": recall_k / max(1, total),
        "rgold": recall_gold / max(1, total),
    }


def _node_detail_lines(
    rec: EvalRecord,
    indices: Sequence[int],
    *,
    graph=None,
    score_tensor: Optional[torch.Tensor] = None,
    gold_tensor: Optional[torch.Tensor] = None,
    epi_tensor: Optional[torch.Tensor] = None,
    pool_tensor: Optional[torch.Tensor] = None,
    prefix: str = "      ",
) -> List[str]:
    if not indices:
        return [f"{prefix}none"]
    lines = []
    for rank, idx in enumerate(indices, start=1):
        node_id = rec.ex.node_ids[idx]
        node_type = rec.ex.graph_kv.node_types[idx]
        score = (
            float(score_tensor[idx].item())
            if score_tensor is not None and idx < score_tensor.numel()
            else float("nan")
        )
        gold = (
            float(gold_tensor[idx].item())
            if gold_tensor is not None and idx < gold_tensor.numel()
            else float("nan")
        )
        epi = (
            float(epi_tensor[idx].item())
            if epi_tensor is not None and idx < epi_tensor.numel()
            else float("nan")
        )
        pool = (
            bool(pool_tensor[idx].item())
            if pool_tensor is not None and idx < pool_tensor.numel()
            else False
        )
        persisted_type, persisted_text = _node_summary(graph, node_id)
        text = persisted_text or ""
        lines.append(
            f"{prefix}{rank}. id={node_id} type={node_type} "
            f"score={score:.3f} gold={gold:.2f} epi={epi:.3f} pool={int(pool)} "
            f"persisted_type={persisted_type} text={_short(text, 90)}"
        )
    return lines


def _planning_hit_summary(rec: EvalRecord) -> Tuple[str, bool, bool]:
    ex = rec.ex
    ps = rec.planning_state
    if ex.plan_anchor is None or not ps.attn_history:
        return "none", False, False
    attn = ps.attn_history[-1].squeeze(0)
    top = int(attn.argmax().item())
    gold = ex.plan_anchor.squeeze(0) > 0
    top3 = attn.topk(min(3, attn.numel())).indices
    top_id = ex.node_ids[top]
    top_type = ex.graph_kv.node_types[top]
    return f"{top_id} ({top_type})", bool(gold[top].item()), bool(gold[top3].any().item())


def _top_plan_detail(rec: EvalRecord) -> Tuple[str, str, float, bool, bool, str, float]:
    ex = rec.ex
    ps = rec.planning_state
    if ex.plan_anchor is None or not ps.attn_history:
        return "none", "none", float("nan"), False, False, "none", float("nan")
    attn = ps.attn_history[-1].squeeze(0)
    top = int(attn.argmax().item())
    gold = ex.plan_anchor.squeeze(0) > 0
    top3 = attn.topk(min(3, attn.numel())).indices
    gold_idxs = torch.where(gold)[0].tolist()
    if gold_idxs:
        best_gold = max(gold_idxs, key=lambda idx: float(attn[idx].item()))
        gold_desc = f"{ex.node_ids[best_gold]} ({ex.graph_kv.node_types[best_gold]})"
        gold_score = float(attn[best_gold].item())
    else:
        gold_desc = "none"
        gold_score = float("nan")
    return (
        ex.node_ids[top],
        ex.graph_kv.node_types[top],
        float(attn[top].item()),
        bool(gold[top].item()),
        bool(gold[top3].any().item()),
        gold_desc,
        gold_score,
    )


def _epi_support_bucket(rec: EvalRecord, primary: Optional[int]) -> str:
    if primary is None:
        return "no_primary"
    pred_epi = rec.evidence_state.epistemic_confidence_r.squeeze(0)
    gold_evidence = _gold_indices(rec.ex.evid_anchor)
    if not gold_evidence:
        return "no_gold_evidence"
    primary_is_gold = primary in set(gold_evidence)
    best_gold_epi = max(float(pred_epi[idx].item()) for idx in gold_evidence)
    primary_epi = float(pred_epi[primary].item())
    if primary_is_gold and primary_epi < EPISTEMIC_THRESHOLD:
        return "low_epi_on_gold_evidence"
    if (not primary_is_gold) and best_gold_epi >= EPISTEMIC_THRESHOLD:
        return "wrong_primary_gold_evidence_ok"
    if best_gold_epi < EPISTEMIC_THRESHOLD:
        return "all_gold_evidence_low"
    return "other"


def _evidence_focus(rec: EvalRecord, primary: Optional[int]) -> Tuple[str, str, float, bool, str, float, str]:
    primary_id, primary_type, primary_epi, _primary_gold_epi, primary_gold_evid = _primary_summary(
        rec, primary)
    gold_evidence = _gold_indices(rec.ex.evid_anchor)
    best_gold_id, best_gold_epi, _mean_gold_epi, _gold_n = _gold_evidence_summary(rec)
    if not gold_evidence:
        bucket = "no_gold_evidence"
    elif primary_gold_evid and primary_epi < EPISTEMIC_THRESHOLD:
        bucket = "right_evidence_low_epi"
    elif primary_gold_evid:
        bucket = "right_evidence_ok"
    elif best_gold_epi >= EPISTEMIC_THRESHOLD:
        bucket = "wrong_evidence_selected"
    else:
        bucket = "gold_evidence_low_epi"
    return primary_id, primary_type, primary_epi, primary_gold_evid, best_gold_id, best_gold_epi, bucket


def _slot_failure_names(rec: EvalRecord) -> List[str]:
    return [name for _idx, name, _pred, _gold, _margin, failed in _slot_rows(rec) if failed]


def _format_slot_rows(rows: Sequence[Tuple[int, str, float, float, float, bool]]) -> str:
    parts = []
    for _idx, name, pred, gold, margin, failed in rows:
        mark = "FAIL" if failed else "ok"
        parts.append(f"{name}:pred={pred:.3f} gold={gold:.0f} margin={margin:+.3f} {mark}")
    return "; ".join(parts) if parts else "none"


def _gold_evidence_summary(rec: EvalRecord) -> Tuple[str, float, float, int]:
    pred_epi = rec.evidence_state.epistemic_confidence_r.squeeze(0)
    gold_evidence = _gold_indices(rec.ex.evid_anchor)
    if not gold_evidence:
        return "none", float("nan"), float("nan"), 0
    best_idx = max(gold_evidence, key=lambda idx: float(pred_epi[idx].item()))
    best = float(pred_epi[best_idx].item())
    mean = _mean([float(pred_epi[idx].item()) for idx in gold_evidence])
    return f"{rec.ex.node_ids[best_idx]} ({rec.ex.graph_kv.node_types[best_idx]})", best, mean, len(gold_evidence)


def _failure_names(bits: Dict[str, object], include_shortcut: bool = False) -> List[str]:
    keys = ["forced_fallback", "empty_pool", "missing_slot", "low_epistemic", "invalidator_active"]
    if include_shortcut:
        keys.append("shortcut_invalid")
    return [name for name in keys if bool(bits[name])]


def _primary_summary(rec: EvalRecord, primary: Optional[int]) -> Tuple[str, str, float, float, bool]:
    if primary is None:
        return "none", "none", float("nan"), float("nan"), False
    pred_epi = rec.evidence_state.epistemic_confidence_r.squeeze(0)
    gold_epi = rec.ex.epi_target.squeeze(0) if rec.ex.epi_target is not None else None
    gold_evid = rec.ex.evid_anchor.squeeze(0) if rec.ex.evid_anchor is not None else None
    return (
        rec.ex.node_ids[primary],
        rec.ex.graph_kv.node_types[primary],
        float(pred_epi[primary].item()),
        float(gold_epi[primary].item()) if gold_epi is not None else float("nan"),
        bool(gold_evid[primary].item() > 0.0) if gold_evid is not None else False,
    )


def _node_type_counts(rec: EvalRecord) -> Counter[str]:
    return Counter(str(t) for t in rec.ex.graph_kv.node_types)


def _substrate_type_counts(rec: EvalRecord) -> Dict[str, int]:
    counts = _node_type_counts(rec)
    return {name: counts[name] for name in sorted(SUBSTRATE_NODE_TYPES) if counts[name]}


def _applicable_calibration_report(
    records: Sequence[EvalRecord],
    graph=None,
    limit: int = 18,
) -> None:
    print("\n=== applicable fallback calibration (held-out only) ===")
    applicable = [rec for rec in records if rec.ex.tag == "applicable"]
    if not applicable:
        print("  no applicable held-out records")
        return

    combo_counts: Counter[str] = Counter()
    shortcut_counts: Counter[str] = Counter()
    slot_fail_counts: Counter[str] = Counter()
    epi_buckets: Counter[str] = Counter()
    family = defaultdict(lambda: {
        "n": 0,
        "fb": 0,
        "slot": 0,
        "epi": 0,
        "shortcut": 0,
        "primary_epi": [],
        "best_gold_epi": [],
        "failed_slot_margin": [],
        "plan_hit": 0,
        "plan_top3": 0,
        "plan_n": 0,
    })

    for rec in applicable:
        bits = fallback_bits(rec.evidence_state, rec.ex)
        failures = [
            name for name in ("empty_pool", "missing_slot", "low_epistemic", "invalidator_active")
            if bool(bits[name])
        ]
        combo = "no_fallback" if not bits["fallback"] else ("+".join(failures) or "max_loops_only")
        combo_counts[combo] += 1
        shortcut_counts["shortcut_invalid" if bits["shortcut_invalid"] else "shortcut_ok"] += 1
        for idx in bits["missing_slot_ids"]:
            slot_fail_counts[SLOT_VOCAB[idx]] += 1
        if bits["low_epistemic"]:
            epi_buckets[_epi_support_bucket(rec, bits["primary_idx"])] += 1

        fam = family[_task_family(rec.ex)]
        fam["n"] += 1
        fam["fb"] += int(bits["fallback"])
        fam["slot"] += int(bits["missing_slot"])
        fam["epi"] += int(bits["low_epistemic"])
        fam["shortcut"] += int(bits["shortcut_invalid"])
        primary = bits["primary_idx"]
        pred_epi = rec.evidence_state.epistemic_confidence_r.squeeze(0)
        if primary is not None:
            fam["primary_epi"].append(float(pred_epi[primary].item()))
        gold_evidence = _gold_indices(rec.ex.evid_anchor)
        if gold_evidence:
            fam["best_gold_epi"].append(max(float(pred_epi[idx].item()) for idx in gold_evidence))
        for _idx, _name, _pred, _gold, margin, failed in _slot_rows(rec):
            if failed:
                fam["failed_slot_margin"].append(margin)
        _plan_top, plan_hit, plan_top3 = _planning_hit_summary(rec)
        if rec.ex.plan_anchor is not None:
            fam["plan_n"] += 1
            fam["plan_hit"] += int(plan_hit)
            fam["plan_top3"] += int(plan_top3)

    print(f"  applicable n={len(applicable)}")
    print(f"  gate-failure combos: {dict(combo_counts.most_common())}")
    print(f"  shortcut telemetry : {dict(shortcut_counts.most_common())}")
    print(f"  failed slots       : {dict(slot_fail_counts.most_common())}")
    print(f"  low-epi buckets    : {dict(epi_buckets.most_common())}")
    print(
        f"{'family':28s} {'n':>3s} {'fb':>5s} {'slot':>5s} {'epi':>5s} "
        f"{'sc':>5s} {'prim_epi':>8s} {'gold_epi':>8s} {'fail_margin':>11s} "
        f"{'plan@1':>7s} {'plan@3':>7s}"
    )
    for fam_name, vals in sorted(family.items(), key=lambda item: (-item[1]["n"], item[0])):
        n = vals["n"]
        plan_n = max(1, vals["plan_n"])
        print(
            f"{fam_name[:28]:28s} {n:>3d} "
            f"{vals['fb'] / max(1, n):>5.2f} "
            f"{vals['slot'] / max(1, n):>5.2f} "
            f"{vals['epi'] / max(1, n):>5.2f} "
            f"{vals['shortcut'] / max(1, n):>5.2f} "
            f"{_mean(vals['primary_epi']):>8.3f} "
            f"{_mean(vals['best_gold_epi']):>8.3f} "
            f"{_mean(vals['failed_slot_margin']):>11.3f} "
            f"{vals['plan_hit'] / plan_n:>7.2f} "
            f"{vals['plan_top3'] / plan_n:>7.2f}"
        )

    print("\n  applicable fallback cases:")
    shown = 0
    for rec_idx, rec in enumerate(applicable, start=1):
        bits = fallback_bits(rec.evidence_state, rec.ex)
        if not bits["fallback"]:
            continue
        shown += 1
        case_id = rec.ex.case_id or f"heldout_applicable_{rec_idx}"
        failures = [
            name for name in ("empty_pool", "missing_slot", "low_epistemic", "invalidator_active")
            if bool(bits[name])
        ]
        primary = bits["primary_idx"]
        pred_epi = rec.evidence_state.epistemic_confidence_r.squeeze(0)
        pred_sc = float(rec.evidence_state.shortcut_validity_r.item())
        primary_id = rec.ex.node_ids[primary] if primary is not None else "none"
        primary_type = rec.ex.graph_kv.node_types[primary] if primary is not None else "none"
        primary_epi = float(pred_epi[primary].item()) if primary is not None else float("nan")
        primary_gold_epi = (
            float(rec.ex.epi_target.squeeze(0)[primary].item())
            if primary is not None and rec.ex.epi_target is not None
            else float("nan")
        )
        primary_gold_evid = (
            bool(rec.ex.evid_anchor.squeeze(0)[primary].item() > 0.0)
            if primary is not None and rec.ex.evid_anchor is not None
            else False
        )
        gold_evidence = _gold_indices(rec.ex.evid_anchor)
        best_gold_epi = (
            max(float(pred_epi[idx].item()) for idx in gold_evidence)
            if gold_evidence else float("nan")
        )
        plan_top, plan_hit, plan_top3 = _planning_hit_summary(rec)
        print(
            f"\n  case {shown}: id={case_id} family={_task_family(rec.ex)} "
            f"failures={failures} shortcut={pred_sc:.3f}/{SHORTCUT_THRESHOLD:.2f}"
        )
        print(f"    required_slots: {(rec.ex.task_frame or {}).get('required_slots') or []}")
        print(f"    slots: {_format_slot_rows(_slot_rows(rec))}")
        print(
            f"    primary_evidence: id={primary_id} type={primary_type} "
            f"pred_epi={primary_epi:.3f}/{EPISTEMIC_THRESHOLD:.2f} "
            f"gold_epi={primary_gold_epi:.0f} gold_evidence={primary_gold_evid} "
            f"best_gold_evidence_epi={best_gold_epi:.3f}"
        )
        persisted_type, persisted_text = _node_summary(graph, primary_id)
        if persisted_text:
            print(f"    primary_text: type={persisted_type} text={persisted_text}")
        print(f"    planning_top: {plan_top} hit@1={plan_hit} hit@3={plan_top3}")
        if rec.ex.question:
            print(f"    question: {_short(rec.ex.question, 140)}")
        if shown >= limit:
            remaining = sum(
                1 for rec2 in applicable
                if fallback_bits(rec2.evidence_state, rec2.ex)["fallback"]
            ) - shown
            if remaining > 0:
                print(f"\n  ... {remaining} more applicable fallback cases omitted")
            break


def _applicable_failure_focus_report(
    records: Sequence[EvalRecord],
    graph=None,
    limit: int = 24,
) -> None:
    print("\n=== applicable failure focus: routing vs calibration ===")
    failures = [
        rec for rec in records
        if rec.ex.tag == "applicable" and fallback_bits(rec.evidence_state, rec.ex)["fallback"]
    ]
    if not failures:
        print("  no applicable fallback failures")
        return

    family = defaultdict(lambda: {
        "n": 0,
        "slot_fail": 0,
        "low_epi": 0,
        "plan_hit": 0,
        "plan_top3": 0,
        "wrong_evidence": 0,
        "right_evidence_low_epi": 0,
        "gold_evidence_low_epi": 0,
        "pred_epi": [],
        "gold_epi": [],
        "plan_wr": [],
        "evid_wr": [],
    })
    evidence_buckets: Counter[str] = Counter()
    plan_buckets: Counter[str] = Counter()
    slot_fail_counts: Counter[str] = Counter()

    rows = []
    for rec in failures:
        bits = fallback_bits(rec.evidence_state, rec.ex)
        fam = family[_task_family(rec.ex)]
        fam["n"] += 1
        fam["slot_fail"] += int(bits["missing_slot"])
        fam["low_epi"] += int(bits["low_epistemic"])

        pred_plan_id, pred_plan_type, pred_plan_score, plan_hit, plan_top3, gold_plan_desc, gold_plan_score = _top_plan_detail(rec)
        if rec.ex.plan_anchor is None:
            plan_bucket = "no_plan_label"
        elif plan_hit:
            plan_bucket = "plan_hit"
        elif plan_top3:
            plan_bucket = "plan_top3_near"
        else:
            plan_bucket = "plan_miss"
        plan_buckets[plan_bucket] += 1
        fam["plan_hit"] += int(plan_hit)
        fam["plan_top3"] += int(plan_top3)

        pred_evid_id, pred_evid_type, pred_epi, pred_evid_gold, best_gold_id, best_gold_epi, evid_bucket = _evidence_focus(
            rec, bits["primary_idx"])
        evidence_buckets[evid_bucket] += 1
        fam["wrong_evidence"] += int(evid_bucket == "wrong_evidence_selected")
        fam["right_evidence_low_epi"] += int(evid_bucket == "right_evidence_low_epi")
        fam["gold_evidence_low_epi"] += int(evid_bucket == "gold_evidence_low_epi")
        fam["pred_epi"].append(pred_epi)
        fam["gold_epi"].append(best_gold_epi)
        fam["plan_wr"].extend(rec.planning_state.write_ratios or [])
        fam["evid_wr"].extend(rec.evidence_state.write_ratios or [])

        failed_slots = _slot_failure_names(rec)
        for slot in failed_slots:
            slot_fail_counts[slot] += 1
        rows.append({
            "rec": rec,
            "bits": bits,
            "failures": _failure_names(bits, include_shortcut=True),
            "failed_slots": failed_slots,
            "pred_plan": f"{pred_plan_id} ({pred_plan_type})",
            "pred_plan_score": pred_plan_score,
            "plan_hit": plan_hit,
            "plan_top3": plan_top3,
            "gold_plan": gold_plan_desc,
            "gold_plan_score": gold_plan_score,
            "pred_evid": f"{pred_evid_id} ({pred_evid_type})",
            "pred_epi": pred_epi,
            "pred_evid_gold": pred_evid_gold,
            "best_gold_evid": best_gold_id,
            "best_gold_epi": best_gold_epi,
            "evid_bucket": evid_bucket,
            "shortcut": float(rec.evidence_state.shortcut_validity_r.item()),
            "plan_wr": _mean(list(rec.planning_state.write_ratios or [])),
            "evid_wr": _mean(list(rec.evidence_state.write_ratios or [])),
        })

    print(f"  applicable fallback failures n={len(failures)}")
    print(f"  evidence buckets: {dict(evidence_buckets.most_common())}")
    print(f"  planning buckets: {dict(plan_buckets.most_common())}")
    print(f"  failed slots    : {dict(slot_fail_counts.most_common())}")
    print(
        f"{'family':28s} {'n':>3s} {'slot':>5s} {'low_epi':>7s} "
        f"{'plan@1':>7s} {'plan@3':>7s} {'wrong_ev':>8s} "
        f"{'right_low':>9s} {'gold_low':>8s} {'pred_epi':>8s} "
        f"{'gold_epi':>8s} {'wr':>6s}"
    )
    for fam_name, vals in sorted(family.items(), key=lambda item: (-item[1]["n"], item[0])):
        n = vals["n"]
        wr_vals = vals["plan_wr"] + vals["evid_wr"]
        print(
            f"{fam_name[:28]:28s} {n:>3d} "
            f"{vals['slot_fail'] / max(1, n):>5.2f} "
            f"{vals['low_epi'] / max(1, n):>7.2f} "
            f"{vals['plan_hit'] / max(1, n):>7.2f} "
            f"{vals['plan_top3'] / max(1, n):>7.2f} "
            f"{vals['wrong_evidence'] / max(1, n):>8.2f} "
            f"{vals['right_evidence_low_epi'] / max(1, n):>9.2f} "
            f"{vals['gold_evidence_low_epi'] / max(1, n):>8.2f} "
            f"{_mean(vals['pred_epi']):>8.3f} "
            f"{_mean(vals['gold_epi']):>8.3f} "
            f"{_mean(wr_vals):>6.3f}"
        )

    print(
        f"{'case_id':34s} {'family':22s} {'failures':32s} {'slots':20s} "
        f"{'plan':18s} {'evidence':24s} {'epi':>13s} {'sc':>5s} {'wr':>5s}"
    )
    for row in rows[:limit]:
        rec = row["rec"]
        case_id = (rec.ex.case_id or "unknown")[-34:]
        slots = ",".join(row["failed_slots"]) if row["failed_slots"] else "ok"
        plan = "hit" if row["plan_hit"] else ("top3" if row["plan_top3"] else "miss")
        evidence = row["evid_bucket"]
        epi = f"{row['pred_epi']:.2f}/{row['best_gold_epi']:.2f}"
        wr = _mean([row["plan_wr"], row["evid_wr"]])
        print(
            f"{case_id:34s} {_task_family(rec.ex)[:22]:22s} "
            f"{'+'.join(row['failures'])[:32]:32s} {slots[:20]:20s} "
            f"{plan:18s} {evidence[:24]:24s} {epi:>13s} "
            f"{row['shortcut']:>5.2f} {wr:>5.3f}"
        )

    print("\n  applicable failure details:")
    for idx, row in enumerate(rows[:limit], start=1):
        rec = row["rec"]
        print(
            f"\n  case {idx}: id={rec.ex.case_id or 'unknown'} "
            f"family={_task_family(rec.ex)} failures={row['failures']}"
        )
        if rec.ex.question:
            print(f"    question: {_short(rec.ex.question, 150)}")
        print(f"    required_slots: {(rec.ex.task_frame or {}).get('required_slots') or []}")
        print(f"    slot_scores: {_format_slot_rows(_slot_rows(rec))}")
        print(
            f"    plan_pred: {row['pred_plan']} score={row['pred_plan_score']:.3f} "
            f"hit@1={row['plan_hit']} hit@3={row['plan_top3']}"
        )
        print(f"    plan_gold_best: {row['gold_plan']} score={row['gold_plan_score']:.3f}")
        print(
            f"    evidence_pred: {row['pred_evid']} pred_epi={row['pred_epi']:.3f} "
            f"gold_evidence={row['pred_evid_gold']}"
        )
        ptype, ptext = _node_summary(graph, row["pred_evid"].split(" (", 1)[0])
        if ptext:
            print(f"    evidence_pred_text: type={ptype} text={ptext}")
        print(
            f"    evidence_gold_best: {row['best_gold_evid']} "
            f"pred_epi={row['best_gold_epi']:.3f} bucket={row['evid_bucket']}"
        )
        print(
            f"    shortcut={row['shortcut']:.3f} "
            f"write_plan={row['plan_wr']:.3f} write_evid={row['evid_wr']:.3f}"
        )
    if len(rows) > limit:
        print(f"\n  ... {len(rows) - limit} more applicable failure rows omitted")


def _direct_judgment_routing_audit(
    records: Sequence[EvalRecord],
    graph=None,
    limit: int = 12,
) -> None:
    print("\n=== direct_judgment routing audit (held-out) ===")
    items = [rec for rec in records if _task_family(rec.ex) == "direct_judgment"]
    if not items:
        print("  no direct_judgment held-out records")
        return

    subsets = [
        ("all_direct", items),
        ("applicable", [rec for rec in items if rec.ex.tag == "applicable"]),
        (
            "applicable_fb",
            [
                rec for rec in items
                if rec.ex.tag == "applicable"
                and fallback_bits(rec.evidence_state, rec.ex)["fallback"]
            ],
        ),
        ("blocked", [rec for rec in items if rec.ex.tag == "blocked"]),
        ("negative", [rec for rec in items if rec.ex.tag == "negative"]),
    ]

    print("  metrics use final node_scores_r over in-pool gold anchors, matching trainable/fallback top-k.")
    print(
        f"{'subset':16s} {'n':>4s} "
        f"{'plan_n':>6s} {'planCov':>7s} {'planP@1':>8s} {'planH@3':>8s} "
        f"{'planR@3':>8s} {'planR@g':>8s} "
        f"{'evid_n':>6s} {'evidCov':>7s} {'evidP@1':>8s} {'evidH@3':>8s} "
        f"{'evidR@3':>8s} {'evidR@g':>8s}"
    )
    for name, subset in subsets:
        if not subset:
            continue
        plan = _routing_metrics(subset, "plan_anchor", "planning_state", "planning_mask", k=3)
        evid = _routing_metrics(subset, "evid_anchor", "evidence_state", "evidence_mask", k=3)
        print(
            f"{name[:16]:16s} {len(subset):>4d} "
            f"{int(plan['n']):>6d} {plan['pool_cov']:>7.2f} "
            f"{plan['p1']:>8.2f} {plan['hit3']:>8.2f} "
            f"{plan['r3']:>8.2f} {plan['rgold']:>8.2f} "
            f"{int(evid['n']):>6d} {evid['pool_cov']:>7.2f} "
            f"{evid['p1']:>8.2f} {evid['hit3']:>8.2f} "
            f"{evid['r3']:>8.2f} {evid['rgold']:>8.2f}"
        )

    failure_items = [
        rec for rec in items
        if rec.ex.tag == "applicable" and fallback_bits(rec.evidence_state, rec.ex)["fallback"]
    ]
    if not failure_items:
        print("  no applicable direct_judgment fallback failures to audit")
        return

    plan_gold_types: Counter[str] = Counter()
    plan_oop_types: Counter[str] = Counter()
    plan_pred_types: Counter[str] = Counter()
    evid_gold_types: Counter[str] = Counter()
    evid_oop_types: Counter[str] = Counter()
    evid_pred_types: Counter[str] = Counter()
    for rec in failure_items:
        plan_scores = _state_scores(rec.planning_state)
        evid_scores = _state_scores(rec.evidence_state)
        for idx in _gold_indices(rec.ex.plan_anchor):
            plan_gold_types[rec.ex.graph_kv.node_types[idx]] += 1
            if not bool(rec.ex.graph_kv.planning_mask[idx].item()):
                plan_oop_types[rec.ex.graph_kv.node_types[idx]] += 1
        for idx in _gold_indices(rec.ex.evid_anchor):
            evid_gold_types[rec.ex.graph_kv.node_types[idx]] += 1
            if not bool(rec.ex.graph_kv.evidence_mask[idx].item()):
                evid_oop_types[rec.ex.graph_kv.node_types[idx]] += 1
        top_plan = _top_indices(plan_scores, 1)
        top_evid = _top_indices(evid_scores, 1)
        if top_plan:
            plan_pred_types[rec.ex.graph_kv.node_types[top_plan[0]]] += 1
        if top_evid:
            evid_pred_types[rec.ex.graph_kv.node_types[top_evid[0]]] += 1

    print(f"  applicable direct_judgment fallback failures n={len(failure_items)}")
    print(f"  gold plan types   : {dict(plan_gold_types.most_common(8))}")
    print(f"  gold plan pool=0 : {dict(plan_oop_types.most_common(8))}")
    print(f"  pred plan@1 types : {dict(plan_pred_types.most_common(8))}")
    print(f"  gold evid types   : {dict(evid_gold_types.most_common(8))}")
    print(f"  gold evid pool=0 : {dict(evid_oop_types.most_common(8))}")
    print(f"  pred evid@1 types : {dict(evid_pred_types.most_common(8))}")

    print("\n  direct_judgment routing label details:")
    for row_idx, rec in enumerate(failure_items[:limit], start=1):
        bits = fallback_bits(rec.evidence_state, rec.ex)
        plan_scores = _state_scores(rec.planning_state)
        evid_scores = _state_scores(rec.evidence_state)
        plan_gold = rec.ex.plan_anchor.squeeze(0) if rec.ex.plan_anchor is not None else None
        evid_gold = rec.ex.evid_anchor.squeeze(0) if rec.ex.evid_anchor is not None else None
        plan_pool = rec.ex.graph_kv.planning_mask
        evid_pool = rec.ex.graph_kv.evidence_mask
        pred_epi = rec.evidence_state.epistemic_confidence_r.squeeze(0)
        gold_epi = rec.ex.epi_target.squeeze(0) if rec.ex.epi_target is not None else None

        plan_top3 = _top_indices(plan_scores, 3)
        evid_top3 = _top_indices(evid_scores, 3)
        gold_plan = _sorted_gold_indices(rec.ex.plan_anchor, limit=5)
        gold_evid = _sorted_gold_indices(rec.ex.evid_anchor, limit=5)
        gold_epi_nodes = _sorted_gold_indices(rec.ex.epi_target, limit=5)

        print(
            f"\n  case {row_idx}: id={rec.ex.case_id or 'unknown'} "
            f"failures={_failure_names(bits, include_shortcut=True)}"
        )
        if rec.ex.question:
            print(f"    question: {_short(rec.ex.question, 160)}")
        print(f"    required_slots: {(rec.ex.task_frame or {}).get('required_slots') or []}")
        print(f"    slots: {_format_slot_rows(_slot_rows(rec))}")
        print("    pred_plan_top3:")
        for line in _node_detail_lines(
            rec, plan_top3, graph=graph, score_tensor=plan_scores,
            gold_tensor=plan_gold, epi_tensor=pred_epi, pool_tensor=plan_pool,
        ):
            print(line)
        print("    gold_plan_anchors:")
        for line in _node_detail_lines(
            rec, gold_plan, graph=graph, score_tensor=plan_scores,
            gold_tensor=plan_gold, epi_tensor=pred_epi, pool_tensor=plan_pool,
        ):
            print(line)
        print("    pred_evidence_top3:")
        for line in _node_detail_lines(
            rec, evid_top3, graph=graph, score_tensor=evid_scores,
            gold_tensor=evid_gold, epi_tensor=pred_epi, pool_tensor=evid_pool,
        ):
            print(line)
        print("    gold_evidence_anchors:")
        for line in _node_detail_lines(
            rec, gold_evid, graph=graph, score_tensor=evid_scores,
            gold_tensor=evid_gold, epi_tensor=pred_epi, pool_tensor=evid_pool,
        ):
            print(line)
        print("    gold_epistemic_nodes:")
        for line in _node_detail_lines(
            rec, gold_epi_nodes, graph=graph, score_tensor=evid_scores,
            gold_tensor=gold_epi, epi_tensor=pred_epi, pool_tensor=evid_pool,
        ):
            print(line)
    if len(failure_items) > limit:
        print(f"\n  ... {len(failure_items) - limit} more direct_judgment routing rows omitted")


def _direct_judgment_report(
    records: Sequence[EvalRecord],
    graph=None,
    limit: int = 18,
) -> None:
    print("\n=== direct_judgment calibration detail (held-out) ===")
    items = [rec for rec in records if _task_family(rec.ex) == "direct_judgment"]
    if not items:
        print("  no direct_judgment held-out records")
        return

    by_tag: Dict[str, List[EvalRecord]] = defaultdict(list)
    combos: Counter[str] = Counter()
    slot_fail_counts: Counter[str] = Counter()
    epi_buckets: Counter[str] = Counter()
    for rec in items:
        bits = fallback_bits(rec.evidence_state, rec.ex)
        by_tag[rec.ex.tag].append(rec)
        combo = "no_fallback" if not bits["fallback"] else ("+".join(_failure_names(bits)) or "max_loops_only")
        combos[combo] += 1
        for idx in bits["missing_slot_ids"]:
            slot_fail_counts[SLOT_VOCAB[idx]] += 1
        if bits["low_epistemic"]:
            epi_buckets[_epi_support_bucket(rec, bits["primary_idx"])] += 1

    print(f"  total n={len(items)} tags={dict((tag, len(vals)) for tag, vals in by_tag.items())}")
    for tag, vals in sorted(by_tag.items()):
        fb = sum(int(fallback_bits(rec.evidence_state, rec.ex)["fallback"]) for rec in vals)
        print(f"  {tag:10s} fallback={_fmt_rate(fb, len(vals))}")
    print(f"  gate combos       : {dict(combos.most_common())}")
    print(f"  failed slots      : {dict(slot_fail_counts.most_common())}")
    print(f"  low-epi buckets   : {dict(epi_buckets.most_common())}")

    print(
        f"{'case_id':34s} {'tag':10s} {'fb':>3s} {'failures':34s} "
        f"{'slots':52s} {'primary_epi':>11s} {'best_gold':>10s} {'plan@1':>7s} {'plan@3':>7s}"
    )
    rows = []
    for rec in items:
        bits = fallback_bits(rec.evidence_state, rec.ex)
        if not bits["fallback"] and rec.ex.tag == "applicable":
            continue
        primary_id, primary_type, primary_epi, _primary_gold_epi, _primary_gold_evid = _primary_summary(
            rec, bits["primary_idx"])
        best_gold_id, best_gold_epi, _mean_gold_epi, _gold_n = _gold_evidence_summary(rec)
        _plan_top, plan_hit, plan_top3 = _planning_hit_summary(rec)
        slot_str = _format_slot_rows(_slot_rows(rec))
        rows.append((rec, bits, primary_id, primary_type, primary_epi, best_gold_id, best_gold_epi, plan_hit, plan_top3, slot_str))

    for rec, bits, _primary_id, _primary_type, primary_epi, _best_gold_id, best_gold_epi, plan_hit, plan_top3, slot_str in rows[:limit]:
        case_id = (rec.ex.case_id or "unknown")[-34:]
        failures = "+".join(_failure_names(bits, include_shortcut=True)) or "none"
        print(
            f"{case_id:34s} {rec.ex.tag:10s} {int(bits['fallback']):>3d} "
            f"{failures[:34]:34s} {slot_str[:52]:52s} "
            f"{primary_epi:>11.3f} {best_gold_epi:>10.3f} "
            f"{str(plan_hit):>7s} {str(plan_top3):>7s}"
        )

    print("\n  direct_judgment failure details:")
    for idx, (rec, bits, primary_id, primary_type, primary_epi, best_gold_id, best_gold_epi, plan_hit, plan_top3, slot_str) in enumerate(rows[:limit], start=1):
        primary_gold_epi = _primary_summary(rec, bits["primary_idx"])[3]
        primary_gold_evid = _primary_summary(rec, bits["primary_idx"])[4]
        print(
            f"\n  case {idx}: id={rec.ex.case_id or 'unknown'} tag={rec.ex.tag} "
            f"finalized={rec.ex.tag == 'applicable'} exit={bits['exit_reason']} "
            f"fallback={bits['fallback']} failures={_failure_names(bits, include_shortcut=True)}"
        )
        print(f"    question: {_short(rec.ex.question, 150)}")
        print(f"    required_slots: {(rec.ex.task_frame or {}).get('required_slots') or []}")
        print(f"    slot_scores: {slot_str}")
        print(
            f"    primary: {primary_id} ({primary_type}) pred_epi={primary_epi:.3f} "
            f"gold_epi={primary_gold_epi:.0f} gold_evidence={primary_gold_evid}"
        )
        ptype, ptext = _node_summary(graph, primary_id)
        if ptext:
            print(f"    primary_text: type={ptype} text={ptext}")
        print(f"    best_gold_evidence: {best_gold_id} pred_epi={best_gold_epi:.3f}")
        print(f"    planning_hit: hit@1={plan_hit} hit@3={plan_top3}")
    if len(rows) > limit:
        print(f"\n  ... {len(rows) - limit} more direct_judgment rows omitted")


def _negative_safety_report(
    records: Sequence[EvalRecord],
    graph=None,
    limit: int = 8,
) -> None:
    print("\n=== negative safety detail (held-out no-graph cases) ===")
    items = [rec for rec in records if rec.ex.tag == "negative"]
    if not items:
        print("  no negative held-out records")
        return

    exit_counts: Counter[str] = Counter()
    combo_counts: Counter[str] = Counter()
    fb = 0
    for rec in items:
        bits = fallback_bits(rec.evidence_state, rec.ex)
        fb += int(bits["fallback"])
        exit_counts[str(bits["exit_reason"])] += 1
        combo = "fallback" if bits["fallback"] else "no_fallback"
        combo_counts[f"{combo}:{'+'.join(_failure_names(bits, include_shortcut=True)) or 'no_failed_gate'}"] += 1
    print(f"  fallback={_fmt_rate(fb, len(items))}")
    print(f"  exit reasons: {dict(exit_counts.most_common())}")
    print(f"  gate combos : {dict(combo_counts.most_common())}")

    for idx, rec in enumerate(items[:limit], start=1):
        bits = fallback_bits(rec.evidence_state, rec.ex)
        primary_id, primary_type, primary_epi, primary_gold_epi, primary_gold_evid = _primary_summary(
            rec, bits["primary_idx"])
        plan_wr = _mean(list(rec.planning_state.write_ratios or []))
        evid_wr = _mean(list(rec.evidence_state.write_ratios or []))
        task_frame = rec.ex.task_frame or {}
        node_counts = _node_type_counts(rec)
        substrate_counts = _substrate_type_counts(rec)
        print(
            f"\n  negative {idx}: fallback={bits['fallback']} exit={bits['exit_reason']} "
            f"failures={_failure_names(bits, include_shortcut=True)} "
            f"plan_wr={plan_wr:.3f} evid_wr={evid_wr:.3f}"
        )
        print(
            "    task_context: "
            f"graph_context={task_frame.get('graph_context', 'graph')} "
            f"force_fallback={bool(task_frame.get('force_fallback'))} "
            f"allow_shortcut_exit={task_frame.get('allow_shortcut_exit', True)}"
        )
        print(f"    required_slots: {(rec.ex.task_frame or {}).get('required_slots') or []}")
        print(f"    slots: {_format_slot_rows(_slot_rows(rec))}")
        print(
            f"    primary: {primary_id} ({primary_type}) pred_epi={primary_epi:.3f} "
            f"gold_epi={primary_gold_epi:.0f} gold_evidence={primary_gold_evid} "
            f"shortcut={float(rec.evidence_state.shortcut_validity_r.item()):.3f}"
        )
        print(
            f"    active_nodes: n={len(rec.ex.node_ids)} "
            f"types={dict(node_counts.most_common(8))} "
            f"substrate={substrate_counts or {}}"
        )
        pred_epi = rec.evidence_state.epistemic_confidence_r.squeeze(0)
        node_scores = rec.evidence_state.node_scores_r.squeeze(0)
        print("    top_evidence_nodes:")
        for rank, node_idx in enumerate(list(bits["top_indices"])[:5], start=1):
            node_id = rec.ex.node_ids[node_idx]
            node_type = rec.ex.graph_kv.node_types[node_idx]
            print(
                f"      {rank}. id={node_id} type={node_type} "
                f"score={float(node_scores[node_idx].item()):.3f} "
                f"epi={float(pred_epi[node_idx].item()):.3f}"
            )
        ptype, ptext = _node_summary(graph, primary_id)
        if ptext:
            print(f"    primary_text: type={ptype} text={ptext}")
        if rec.ex.question:
            print(f"    question: {_short(rec.ex.question, 150)}")


def _oracle_report(records: Sequence[EvalRecord]) -> None:
    print("\n=== oracle fallback ablations (held-out) ===")
    print("Shortcut is diagnostic telemetry in the current formula, so gold_shortcut_only should not move fallback.")
    variants = {
        "predicted_all": {},
        "gold_slots_only": {"gold_slots": True},
        "gold_epi_only": {"gold_epi": True},
        "gold_inv_only": {"gold_inv": True},
        "gold_shortcut_only": {"gold_shortcut": True},
        "gold_all": {"gold_slots": True, "gold_epi": True, "gold_inv": True, "gold_shortcut": True},
    }
    tags = sorted({rec.ex.tag for rec in records})
    print(f"{'variant':22s} " + " ".join(f"{tag:>12s}" for tag in tags))
    for name, kwargs in variants.items():
        cells = []
        for tag in tags:
            items = [rec for rec in records if rec.ex.tag == tag]
            fb = sum(int(fallback_bits(rec.evidence_state, rec.ex, **kwargs)["fallback"]) for rec in items)
            cells.append(f"{fb / max(1, len(items)):>12.2f}")
        print(f"{name:22s} " + " ".join(cells))


def _write_ratio_report(records: Sequence[EvalRecord]) -> None:
    print("\n=== write ratio by case type, fallback state, and block ===")
    buckets = defaultdict(lambda: {"plan": [], "evid": [], "total": []})
    for rec in records:
        bits = fallback_bits(rec.evidence_state, rec.ex)
        fb_state = "fallback" if bits["fallback"] else "no_fallback"
        key = (rec.ex.tag, fb_state)
        plan_wr = list(rec.planning_state.write_ratios or [])
        evid_wr = list(rec.evidence_state.write_ratios or [])
        buckets[key]["plan"].extend(plan_wr)
        buckets[key]["evid"].extend(evid_wr)
        buckets[key]["total"].extend(plan_wr + evid_wr)

    print(f"{'tag':12s} {'state':12s} {'plan_wr':>8s} {'evid_wr':>8s} {'total_wr':>9s} {'n_wr':>5s}")
    for (tag, state), vals in sorted(buckets.items()):
        print(
            f"{tag:12s} {state:12s} "
            f"{_mean(vals['plan']):>8.3f} {_mean(vals['evid']):>8.3f} "
            f"{_mean(vals['total']):>9.3f} {len(vals['total']):>5d}"
        )


def _planning_miss_report(records: Sequence[EvalRecord], diffuse_threshold: float) -> None:
    print("\n=== planning inspection (held-out) ===")
    counts = Counter()
    top_types = Counter()
    target_types = Counter()

    for rec in records:
        ex = rec.ex
        ps = rec.planning_state
        if ex.plan_anchor is None:
            counts["no_plan_label"] += 1
            continue
        if not bool(ex.graph_kv.planning_mask.any().item()):
            counts["empty_plan_pool"] += 1
            continue
        if not ps.attn_history:
            counts["no_attention_history"] += 1
            continue

        attn = ps.attn_history[-1].squeeze(0)
        gold = ex.plan_anchor.squeeze(0) > 0
        top = int(attn.argmax().item())
        top_prob = float(attn[top].item())
        top_types[ex.graph_kv.node_types[top]] += 1
        for idx in torch.where(gold)[0].tolist():
            target_types[ex.graph_kv.node_types[idx]] += 1

        if not bool(ex.graph_kv.planning_mask[top].item()):
            counts["wrong_pool"] += 1
        elif bool(gold[top].item()):
            counts["hit"] += 1
        elif bool(gold[attn.topk(min(3, attn.numel())).indices].any().item()):
            counts["top3_near_miss"] += 1
        elif top_prob < diffuse_threshold:
            counts["diffuse_miss"] += 1
        else:
            counts["confident_miss"] += 1

    total = sum(counts.values())
    for key, val in counts.most_common():
        print(f"  {key:20s} {_fmt_rate(val, total)}")
    print(f"  top predicted planning node types: {dict(top_types.most_common(8))}")
    print(f"  gold planning target types       : {dict(target_types.most_common(8))}")


def _build_and_train(
    corpus_path: str,
    model_name: str,
    device_str: Optional[str],
    eval_frac: float,
    e1: int,
    e2a: int,
    e2b: int,
    seed: int,
):
    _set_seed(seed)
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device}  corpus={corpus_path}  model={model_name}")
    print(f"schedule: e1={e1} e2a={e2a} e2b={e2b} eval_frac={eval_frac} seed={seed}")

    print("\n[1] substrate pass...")
    _, stats = build_substrate_graph(corpus_path=corpus_path, out_path=SUBSTRATE_OUT)
    print(f"    +{stats['substrate_nodes_added']} nodes, +{stats['relations_added']} relations -> {stats['out_path']}")

    print("\n[2] loading real providers...")
    provider = FrozenQwenHInitProvider(model_name, device=device)
    lm_dim = provider.hidden_size
    embedder = RealEmbedder(device)
    gnn = RGCNEncoder().to(device).eval()
    for p in gnn.parameters():
        p.requires_grad_(False)

    gpath = _graph_path()
    graph = load_persisted_graph(gpath)
    pos = corpus_to_stage1_examples(
        corpus_path,
        gnn=gnn,
        embedder=embedder,
        h_init_provider=provider,
        device=device,
        lm_dim=lm_dim,
        graph_path=gpath,
        hops=1,
    )
    negs = make_real_negatives(provider, embedder, gnn, graph, device, lm_dim, pos[0])
    examples = pos + negs
    train, ev = _stratified_split(examples, eval_frac)

    _label_distribution("train", train)
    _label_distribution("held-out", ev)

    print("\n[3] integrated Stage 1 -> 2A -> 2B...")
    adapter = V5AttentionAdapter(r_plan=3, r_evidence=4, lm_hidden_dim=lm_dim, gate_init=GATE_INIT).to(device)
    Stage1Trainer(adapter, Stage1Config(epochs=e1, lr=1e-3)).train([ex for ex in train if ex.tag != "negative"])
    Stage2Trainer(adapter, Stage2Config(sub_stage="2A", epochs=e2a, lr=2e-4)).train(train)
    Stage2Trainer(
        adapter,
        Stage2Config(sub_stage="2B", epochs=e2b, lr=1e-4, lambda_delta=1.0, qkv_lr_scale=0.3),
    ).train(train)

    return adapter, train, ev, graph


def run(
    corpus_path: str,
    model_name: str = DEFAULT_LM,
    device_str: Optional[str] = None,
    eval_frac: float = 0.2,
    e1: int = 30,
    e2a: int = 20,
    e2b: int = 20,
    diffuse_threshold: float = 0.35,
    invalidator_limit: int = 12,
    applicable_limit: int = 18,
    applicable_focus_limit: int = 24,
    direct_routing_limit: int = 12,
    direct_limit: int = 18,
    negative_limit: int = 8,
    seed: int = 7,
) -> None:
    adapter, _train, ev, graph = _build_and_train(
        corpus_path, model_name, device_str, eval_frac, e1, e2a, e2b, seed)

    print("\n[4] evaluating held-out records...")
    records = _eval_records(adapter, ev)
    _fallback_trip_report(records)
    _task_family_report(records)
    _applicable_calibration_report(records, graph, applicable_limit)
    _applicable_failure_focus_report(records, graph, applicable_focus_limit)
    _direct_judgment_routing_audit(records, graph, direct_routing_limit)
    _direct_judgment_report(records, graph, direct_limit)
    _negative_safety_report(records, graph, negative_limit)
    _invalidator_case_report(records, invalidator_limit, graph)
    _oracle_report(records)
    _write_ratio_report(records)
    _planning_miss_report(records, diffuse_threshold)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus_merged_v5proj.jsonl")
    ap.add_argument("--model", default=DEFAULT_LM)
    ap.add_argument("--device", default=None)
    ap.add_argument("--eval-frac", type=float, default=0.2)
    ap.add_argument("--e1", type=int, default=30)
    ap.add_argument("--e2a", type=int, default=20)
    ap.add_argument("--e2b", type=int, default=20)
    ap.add_argument("--diffuse-threshold", type=float, default=0.35)
    ap.add_argument("--invalidator-limit", type=int, default=12)
    ap.add_argument("--applicable-limit", type=int, default=18)
    ap.add_argument("--applicable-focus-limit", type=int, default=24)
    ap.add_argument("--direct-routing-limit", type=int, default=12)
    ap.add_argument("--direct-limit", type=int, default=18)
    ap.add_argument("--negative-limit", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    run(
        args.corpus,
        args.model,
        args.device,
        args.eval_frac,
        args.e1,
        args.e2a,
        args.e2b,
        args.diffuse_threshold,
        args.invalidator_limit,
        args.applicable_limit,
        args.applicable_focus_limit,
        args.direct_routing_limit,
        args.direct_limit,
        args.negative_limit,
        args.seed,
    )
