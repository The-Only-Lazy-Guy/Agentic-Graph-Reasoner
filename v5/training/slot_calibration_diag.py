"""Slot-calibration diagnostic (the oracle ruled out support-pointer; slot gate is the blocker).

Trains integrated Stage 2 on the train split, then on held-out:
  1. SLOT-THRESHOLD SWEEP (0.50..0.85): fallback rate by case type + slot
     precision/recall at each threshold. Finds the lowest threshold where
     applicable fallback drops while blocked/negative stay high (calibration,
     not gaming).
  2. GOLD-SLOT ORACLE: fallback with predicted slots vs gold required-slot labels.
     If gold slots drop applicable fallback (blocked/negative staying high),
     the slot-confidence diagnosis is fully confirmed.

Fallback decision mirrors exit_condition.fallback_needed but with a configurable
slot threshold and an optional gold-slot override; epistemic gate uses the
primary (top-1) attended node at the standard 0.70.

    $env:KMP_DUPLICATE_LIB_OK="TRUE"; python -u -m v5.training.slot_calibration_diag --corpus artifacts/phase15_50/corpus50.jsonl
"""
from __future__ import annotations

import argparse

import torch

from v5.cross_attention import V5AttentionAdapter
from v5.exit_condition import EPISTEMIC_THRESHOLD, _required_slot_indices, _top_k_indices
from v5.gnn_encoder import RGCNEncoder
from v5.training.bridge import corpus_to_stage1_examples, load_persisted_graph
from v5.training.providers import RealEmbedder, FrozenQwenHInitProvider
from v5.training.stage1 import Stage1Trainer, Stage1Config
from v5.training.stage1_real import _graph_path
from v5.training.stage2 import Stage2Trainer, Stage2Config, GATE_INIT
from v5.training.stage2b_real import make_real_negatives
from v5.training.substrate import build_substrate_graph, DEFAULT_OUT as SUBSTRATE_OUT
from v5.training.corpus_scaling import _stratified_split

DEFAULT_LM = "Qwen/Qwen2.5-1.5B"
THRESHOLDS = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85]


@torch.no_grad()
def _cache(adapter, ex):
    """Run the loops once; cache the tensors the fallback gate needs."""
    _, ps, _ = adapter.run_planning(ex.h_init, ex.goal, ex.graph_kv, ex.node_ids, task_frame=ex.task_frame)
    _, es, _ = adapter.run_evidence(ps.h_r, ex.goal, ex.graph_kv, ex.node_ids, task_frame=ex.task_frame)
    req = _required_slot_indices(ex.task_frame) or []
    top = _top_k_indices(es.node_scores_r)
    inv = es.invalidator_flags_r.squeeze(0)
    epi = es.epistemic_confidence_r.squeeze(0)
    slot_pred = es.slot_state_r.squeeze(0)
    slot_gold = ex.slot_target.squeeze(0) if ex.slot_target is not None else None
    return {
        "tag": ex.tag, "req": req,
        "slot_pred": [float(slot_pred[i].item()) for i in req],
        "slot_gold": ([float(slot_gold[i].item()) for i in req] if slot_gold is not None else None),
        "no_inv_top": not any(float(inv[i].item()) > 0.5 for i in top),
        "epi_primary_ok": (float(epi[top[0]].item()) >= EPISTEMIC_THRESHOLD) if top else False,
    }


def _fallback(c, slot_thresh, use_gold=False):
    slots = c["slot_gold"] if (use_gold and c["slot_gold"] is not None) else c["slot_pred"]
    slots_ok = (len(slots) > 0) and all(s >= (0.5 if use_gold else slot_thresh) for s in slots)
    if not c["req"]:
        slots_ok = True
    return not (slots_ok and c["no_inv_top"] and c["epi_primary_ok"])


def _rate(caches, tag, slot_thresh, use_gold=False):
    items = [c for c in caches if c["tag"] == tag]
    if not items:
        return float("nan")
    return sum(_fallback(c, slot_thresh, use_gold) for c in items) / len(items)


def run(corpus_path, model_name=DEFAULT_LM, device_str=None, eval_frac=0.2,
        e1=200, e2a=120, e2b=150):
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device}  corpus={corpus_path}")
    build_substrate_graph(corpus_path=corpus_path, out_path=SUBSTRATE_OUT)
    provider = FrozenQwenHInitProvider(model_name, device=device)
    lm_dim = provider.hidden_size
    embedder = RealEmbedder(device)
    gnn = RGCNEncoder().to(device).eval()
    for p in gnn.parameters():
        p.requires_grad_(False)
    gpath = _graph_path()
    graph = load_persisted_graph(gpath)
    pos = corpus_to_stage1_examples(corpus_path, gnn=gnn, embedder=embedder,
                                    h_init_provider=provider, device=device,
                                    lm_dim=lm_dim, graph_path=gpath, hops=1)
    negs = make_real_negatives(provider, embedder, gnn, graph, device, lm_dim, pos[0])
    train, ev = _stratified_split(pos + negs, eval_frac)

    print("training integrated Stage 1 -> 2A -> 2B ...")
    adapter = V5AttentionAdapter(r_plan=3, r_evidence=4, lm_hidden_dim=lm_dim, gate_init=GATE_INIT).to(device)
    Stage1Trainer(adapter, Stage1Config(epochs=e1, lr=1e-3)).train([e for e in train if e.tag != "negative"])
    Stage2Trainer(adapter, Stage2Config(sub_stage="2A", epochs=e2a, lr=2e-4)).train(train)
    Stage2Trainer(adapter, Stage2Config(sub_stage="2B", epochs=e2b, lr=1e-4,
                                        lambda_delta=1.0, qkv_lr_scale=0.3)).train(train)
    adapter.eval()
    caches = [_cache(adapter, ex) for ex in ev]
    napp = sum(c["tag"] == "applicable" for c in caches)
    print(f"\nheld-out: {len(caches)} (applicable={napp})  epi gate fixed at {EPISTEMIC_THRESHOLD}\n")

    print("=== SLOT-THRESHOLD SWEEP (fallback rate by tag) ===")
    print(f"{'thresh':>7s} {'applic':>7s} {'blocked':>8s} {'negative':>9s} {'slot_P':>7s} {'slot_R':>7s}")
    for t in THRESHOLDS:
        # slot precision/recall over required slots (predicted >= t vs gold==1)
        tp = fp = fn = 0
        for c in caches:
            if c["slot_gold"] is None:
                continue
            for p, g in zip(c["slot_pred"], c["slot_gold"]):
                pos_pred = p >= t; pos_gold = g >= 0.5
                tp += int(pos_pred and pos_gold); fp += int(pos_pred and not pos_gold)
                fn += int((not pos_pred) and pos_gold)
        P = tp / max(1, tp + fp); R = tp / max(1, tp + fn)
        print(f"{t:>7.2f} {_rate(caches,'applicable',t):>7.2f} {_rate(caches,'blocked',t):>8.2f} "
              f"{_rate(caches,'negative',t):>9.2f} {P:>7.2f} {R:>7.2f}")

    print("\n=== GOLD-SLOT ORACLE (epi/inv gates unchanged) ===")
    print(f"  predicted slots (@0.85): applicable={_rate(caches,'applicable',0.85):.2f} "
          f"blocked={_rate(caches,'blocked',0.85):.2f} negative={_rate(caches,'negative',0.85):.2f}")
    print(f"  GOLD slots             : applicable={_rate(caches,'applicable',0.85,use_gold=True):.2f} "
          f"blocked={_rate(caches,'blocked',0.85,use_gold=True):.2f} negative={_rate(caches,'negative',0.85,use_gold=True):.2f}")

    # recommend the lowest threshold where applicable drops AND blocked/neg stay high
    print("\n=== RECOMMENDATION ===")
    rec = None
    for t in THRESHOLDS:
        a = _rate(caches, "applicable", t)
        b = _rate(caches, "blocked", t); n = _rate(caches, "negative", t)
        b_ok = (b != b) or b >= 0.8
        n_ok = (n != n) or n >= 0.8
        if a <= 0.5 and b_ok and n_ok:
            rec = t; break
    if rec is not None:
        print(f"  lowest threshold with applicable<=0.5 & blocked/negative>=0.8: {rec:.2f}")
    else:
        print("  NO threshold drops applicable<=0.5 while keeping blocked/negative high.")
        gold_a = _rate(caches, 'applicable', 0.85, use_gold=True)
        if gold_a <= 0.5:
            print(f"  BUT gold slots give applicable={gold_a:.2f} -> slot CONFIDENCE is the issue")
            print("  (train slot head harder / pos-weight / temperature), not the threshold.")
        else:
            print("  Gold slots also do not drop applicable -> look beyond slots (inv/epi/labels).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="artifacts/phase15_50/corpus50.jsonl")
    ap.add_argument("--model", default=DEFAULT_LM)
    ap.add_argument("--device", default=None)
    ap.add_argument("--eval-frac", type=float, default=0.2)
    a = ap.parse_args()
    run(a.corpus, a.model, a.device, a.eval_frac)
