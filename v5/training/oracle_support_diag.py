"""Oracle support-pointer diagnostic (run BEFORE building the head).

Question: is fallback-applicable stuck at 1.00 because of SUPPORT SELECTION
(the fallback gate checks the top-attended node, which is the wrong proxy) or
EPISTEMIC CALIBRATION (even the correct support node isn't predicted supported)?

Decisive test: after training integrated Stage 2 on the 46-trace train split,
on held-out APPLICABLE examples compare the fallback decision under:
  - standard gate: epistemic checked on the TOP-ATTENDED evidence node
  - oracle gate:   epistemic checked on the GOLD support anchor (cited evidence)

Also break down WHICH condition trips standard fallback (slot < 0.85 / epi < 0.70
/ invalidator), and report epi on gold-support vs top-attended.

Read:
  oracle fallback << standard  -> support SELECTION is the problem -> build the
                                  support-pointer head.
  oracle fallback still ~1.00  -> EPISTEMIC calibration/labels is the problem ->
                                  the pointer alone won't fix it.

    $env:KMP_DUPLICATE_LIB_OK="TRUE"; python -u -m v5.training.oracle_support_diag --corpus artifacts/phase15_50/corpus50.jsonl
"""
from __future__ import annotations

import argparse

import torch

from v5.cross_attention import V5AttentionAdapter
from v5.exit_condition import (SLOT_FILL_THRESHOLD, EPISTEMIC_THRESHOLD,
                               _required_slot_indices, _all_slots_filled, _top_k_indices)
from v5.gnn_encoder import RGCNEncoder
from v5.goal_encoder import GoalEncoder
from v5.training.bridge import corpus_to_stage1_examples, load_persisted_graph
from v5.training.providers import RealEmbedder, FrozenQwenHInitProvider
from v5.training.stage1 import Stage1Trainer, Stage1Config
from v5.training.stage1_real import _graph_path
from v5.training.stage2 import Stage2Trainer, Stage2Config, GATE_INIT
from v5.training.stage2b_real import make_real_negatives
from v5.training.substrate import build_substrate_graph, DEFAULT_OUT as SUBSTRATE_OUT
from v5.training.corpus_scaling import _stratified_split

DEFAULT_LM = "Qwen/Qwen2.5-1.5B"


@torch.no_grad()
def _run(adapter, ex):
    _, ps, _ = adapter.run_planning(ex.h_init, ex.goal, ex.graph_kv, ex.node_ids, task_frame=ex.task_frame)
    _, es, _ = adapter.run_evidence(ps.h_r, ex.goal, ex.graph_kv, ex.node_ids, task_frame=ex.task_frame)
    return es


def _conditions(es, ex):
    """Return (slots_ok, no_inv_top, epi_top_ok, epi_oracle_ok, epi_top, epi_oracle)."""
    req = _required_slot_indices(ex.task_frame)
    slots_ok = _all_slots_filled(es.slot_state_r, required_indices=req)
    top = _top_k_indices(es.node_scores_r)
    inv = es.invalidator_flags_r.squeeze(0)
    no_inv_top = not any(float(inv[i].item()) > 0.5 for i in top)
    epi = es.epistemic_confidence_r.squeeze(0)
    epi_top = float(epi[top[0]].item()) if top else 0.0
    epi_top_ok = epi_top >= EPISTEMIC_THRESHOLD
    # oracle: gold support = evidence-pool anchors the trace cited
    if ex.evid_anchor is not None and (ex.evid_anchor > 0.5).any():
        sup = (ex.evid_anchor.squeeze(0) > 0.5)
        epi_oracle = float(epi[sup].max().item())
    else:
        epi_oracle = 0.0
    epi_oracle_ok = epi_oracle >= EPISTEMIC_THRESHOLD
    return slots_ok, no_inv_top, epi_top_ok, epi_oracle_ok, epi_top, epi_oracle


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

    print("training integrated Stage 1 -> 2A -> 2B (train split)...")
    adapter = V5AttentionAdapter(r_plan=3, r_evidence=4, lm_hidden_dim=lm_dim, gate_init=GATE_INIT).to(device)
    Stage1Trainer(adapter, Stage1Config(epochs=e1, lr=1e-3)).train([e for e in train if e.tag != "negative"])
    Stage2Trainer(adapter, Stage2Config(sub_stage="2A", epochs=e2a, lr=2e-4)).train(train)
    Stage2Trainer(adapter, Stage2Config(sub_stage="2B", epochs=e2b, lr=1e-4,
                                        lambda_delta=1.0, qkv_lr_scale=0.3)).train(train)
    adapter.eval()

    appl = [e for e in ev if e.tag == "applicable"]
    print(f"\nheld-out applicable: n={len(appl)}")
    print(f"thresholds: slot>={SLOT_FILL_THRESHOLD}  epi>={EPISTEMIC_THRESHOLD}\n")

    std_fb = ora_fb = 0
    trip = {"slot": 0, "inv": 0, "epi_top": 0}
    epi_tops, epi_oras = [], []
    for ex in appl:
        es = _run(adapter, ex)
        slots_ok, no_inv, epi_top_ok, epi_ora_ok, epi_top, epi_ora = _conditions(es, ex)
        epi_tops.append(epi_top); epi_oras.append(epi_ora)
        # standard fallback fires if any condition fails
        std = not (slots_ok and no_inv and epi_top_ok)
        ora = not (slots_ok and no_inv and epi_ora_ok)
        std_fb += int(std); ora_fb += int(ora)
        if not slots_ok: trip["slot"] += 1
        if not no_inv: trip["inv"] += 1
        if not epi_top_ok: trip["epi_top"] += 1

    n = max(1, len(appl))
    print("=== applicable fallback rate ===")
    print(f"  standard gate (top-attended epi): {std_fb}/{len(appl)} = {std_fb/n:.2f}")
    print(f"  ORACLE gate   (gold-support epi): {ora_fb}/{len(appl)} = {ora_fb/n:.2f}")
    print("\n=== which condition trips standard fallback (applicable) ===")
    for k, v in trip.items():
        print(f"  {k:8s} fails: {v}/{len(appl)}")
    print(f"\n  mean epi on top-attended : {sum(epi_tops)/n:.2f}")
    print(f"  mean epi on gold-support : {sum(epi_oras)/n:.2f}")
    print("\n=== VERDICT ===")
    if ora_fb < std_fb:
        print("  ORACLE support DROPS fallback -> SUPPORT SELECTION is the problem ->")
        print("  build the support-pointer head.")
    else:
        print("  Oracle support does NOT drop fallback -> EPISTEMIC CALIBRATION/labels")
        print("  is the problem -> the pointer alone won't fix it (fix epi labels/threshold).")
    return std_fb / n, ora_fb / n, trip


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="artifacts/phase15_50/corpus50.jsonl")
    ap.add_argument("--model", default=DEFAULT_LM)
    ap.add_argument("--device", default=None)
    ap.add_argument("--eval-frac", type=float, default=0.2)
    a = ap.parse_args()
    run(a.corpus, a.model, a.device, a.eval_frac)
