"""Fallback/write diagnostics for the projected V5 corpus.

This is the "do not blindly rerun schedule/model yet" harness. It runs the same
integrated Stage 1 -> 2A -> 2B path as corpus_scaling, then decomposes the
held-out failures:

  - label distributions by split and case type
  - fallback trip reasons by case type
  - one-condition-at-a-time oracle fallback ablations
  - write ratio by case type, fallback state, and block
  - planning miss categories

Usage:
    $env:KMP_DUPLICATE_LIB_OK="TRUE"; python -u -m v5.training.fallback_write_diag --corpus data/corpus_merged_v5proj.jsonl
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from v5.cross_attention import V5AttentionAdapter
from v5.exit_condition import (
    EPISTEMIC_THRESHOLD,
    SHORTCUT_THRESHOLD,
    SLOT_FILL_THRESHOLD,
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

DEFAULT_LM = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass
class EvalRecord:
    ex: Stage1Example
    planning_state: object
    evidence_state: object


def _mean(values: Sequence[float]) -> float:
    return sum(values) / max(1, len(values))


def _fmt_rate(num: int, den: int) -> str:
    return f"{num}/{den} ({num / max(1, den):.2f})"


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

    fallback = bool(max_loops and (empty_pool or not (slots_ok and no_inv and epi_ok)))
    return {
        "fallback": fallback,
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
):
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device}  corpus={corpus_path}  model={model_name}")
    print(f"schedule: e1={e1} e2a={e2a} e2b={e2b} eval_frac={eval_frac}")

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

    return adapter, train, ev


def run(
    corpus_path: str,
    model_name: str = DEFAULT_LM,
    device_str: Optional[str] = None,
    eval_frac: float = 0.2,
    e1: int = 30,
    e2a: int = 20,
    e2b: int = 20,
    diffuse_threshold: float = 0.35,
) -> None:
    adapter, _train, ev = _build_and_train(corpus_path, model_name, device_str, eval_frac, e1, e2a, e2b)

    print("\n[4] evaluating held-out records...")
    records = _eval_records(adapter, ev)
    _fallback_trip_report(records)
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
    args = ap.parse_args()
    run(args.corpus, args.model, args.device, args.eval_frac, args.e1, args.e2a, args.e2b, args.diffuse_threshold)
