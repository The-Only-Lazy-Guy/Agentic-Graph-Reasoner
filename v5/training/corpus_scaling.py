"""Corpus-scaling harness: substrate -> coverage -> split -> integrated Stage 2 -> held-out metrics.

The single orchestrator for scaling the V4 corpus and measuring whether fallback
CALIBRATION improves with data (not overfitting 20 examples). Target first
100-300 substrate-rich traces — enough to measure calibration, not train a big model.

Pipeline (this file):
  corpus.jsonl
    -> substrate pass (apply safe scoped patches -> enriched graph)
    -> bridge coverage report (plan/evid/slot/epi/inv/shortcut)
    -> stratified train/eval split (held-out by trace)
    -> integrated Stage 1 -> 2A -> 2B on TRAIN
    -> HELD-OUT metrics on EVAL

(Upstream of this: questions -> V4 run -> traces+scoped_patches. Use the data-gen
environment / run_phase15_corpus.py to produce larger corpora, then point --corpus here.)

Held-out report:
  node precision/recall (plan, evidence) · slot/epi/inv/shortcut accuracy ·
  fallback applicable/blocked/negative rate · write ratio by case type.
The key line: fallback applicable should DROP while blocked/negative stay high.

    $env:KMP_DUPLICATE_LIB_OK="TRUE"; python -u -m v5.training.corpus_scaling --corpus artifacts/phase15/phase15_corpus.jsonl
"""
from __future__ import annotations

import argparse
import os
import random
from collections import defaultdict

import torch

from v5.cross_attention import V5AttentionAdapter
from v5.exit_condition import fallback_needed
from v5.gnn_encoder import RGCNEncoder
from v5.goal_encoder import GoalEncoder
from v5.training.bridge import corpus_to_stage1_examples, load_persisted_graph
from v5.training.providers import RealEmbedder, FrozenQwenHInitProvider
from v5.training.stage1 import Stage1Trainer, Stage1Config, _required_slot_idx
from v5.training.stage1_real import _graph_path
from v5.training.stage2 import Stage2Trainer, Stage2Config, GATE_INIT
from v5.training.stage2b_real import make_real_negatives
from v5.training.substrate import build_substrate_graph, DEFAULT_OUT as SUBSTRATE_OUT

DEFAULT_LM = "Qwen/Qwen2.5-1.5B"


# ── held-out metrics ─────────────────────────────────────────────────────────

def _node_pr(adapter, examples, which):
    """Precision@1 + recall@|gold| for plan/evid node attention on held-out."""
    hit1 = tot = rec_num = rec_den = 0
    for ex in examples:
        anchor = ex.plan_anchor if which == "plan" else ex.evid_anchor
        if anchor is None:
            continue
        _, ps, _ = adapter.run_planning(ex.h_init, ex.goal, ex.graph_kv, ex.node_ids, task_frame=ex.task_frame)
        st = ps
        if which == "evid":
            _, st, _ = adapter.run_evidence(ps.h_r, ex.goal, ex.graph_kv, ex.node_ids, task_frame=ex.task_frame)
        if not st.attn_history:
            continue
        attn = st.attn_history[-1].squeeze(0)
        gold = (anchor.squeeze(0) > 0.5)
        n_gold = int(gold.sum().item())
        if n_gold == 0:
            continue
        tot += 1
        hit1 += int(gold[attn.argmax()].item())
        topk = attn.topk(min(n_gold, attn.numel())).indices
        rec_num += int(gold[topk].sum().item()); rec_den += n_gold
    return {"precision@1": hit1 / max(1, tot), "recall@gold": rec_num / max(1, rec_den), "n": tot}


@torch.no_grad()
def heldout_metrics(adapter, examples):
    adapter.eval()
    plan_pr = _node_pr(adapter, examples, "plan")
    evid_pr = _node_pr(adapter, examples, "evid")
    head = defaultdict(lambda: [0, 0])     # name -> [hit, n]  (strict all-node match)
    fb = defaultdict(lambda: [0, 0])       # tag -> [fb, n]
    wr = defaultdict(list)
    epi_node_hit = epi_node_tot = 0        # per-node epistemic accuracy (less strict)
    for ex in examples:
        _, ps, _ = adapter.run_planning(ex.h_init, ex.goal, ex.graph_kv, ex.node_ids, task_frame=ex.task_frame)
        _, es, _ = adapter.run_evidence(ps.h_r, ex.goal, ex.graph_kv, ex.node_ids, task_frame=ex.task_frame)
        if ex.slot_target is not None:
            req = _required_slot_idx(ex.task_frame)
            ok = ((es.slot_state_r[0, req] - ex.slot_target[0, req]).abs() < 0.5).all().item()
            head["slot"][0] += int(ok); head["slot"][1] += 1
        if ex.epi_target is not None:
            pred = (es.epistemic_confidence_r > 0.5).float()
            ok = (pred == ex.epi_target).all().item()
            head["epi"][0] += int(ok); head["epi"][1] += 1
            epi_node_hit += int((pred == ex.epi_target).sum().item()); epi_node_tot += ex.epi_target.numel()
        if ex.shortcut_target is not None:
            ok = (es.shortcut_validity_r.item() > 0.5) == ex.shortcut_target.item()
            head["shortcut"][0] += int(ok); head["shortcut"][1] += 1
        if ex.inv_target is not None and ex.struct_inv_mask is not None and ex.struct_inv_mask.any():
            m = ex.struct_inv_mask
            ok = ((es.invalidator_flags_r[m] > 0.5).float() == ex.inv_target[m]).all().item()
            head["inv"][0] += int(ok); head["inv"][1] += 1
        fb[ex.tag][0] += int(fallback_needed(es, ex.task_frame)); fb[ex.tag][1] += 1
        wr[ex.tag] += (ps.write_ratios or []) + (es.write_ratios or [])
    return {
        "plan_node": plan_pr, "evid_node": evid_pr,
        "head_acc": {k: v[0] / max(1, v[1]) for k, v in head.items()},
        "fallback": {k: v[0] / max(1, v[1]) for k, v in fb.items()},
        "write_ratio": {k: (sum(v) / max(1, len(v))) for k, v in wr.items()},
        "epi_per_node_acc": epi_node_hit / max(1, epi_node_tot),
        "n_eval": len(examples),
    }


def _stratified_split(examples, eval_frac, seed=0):
    rng = random.Random(seed)
    by_tag = defaultdict(list)
    for e in examples:
        by_tag[e.tag].append(e)
    train, ev = [], []
    for tag, items in by_tag.items():
        rng.shuffle(items)
        k = max(1, int(round(len(items) * eval_frac))) if len(items) > 1 else 0
        ev += items[:k]; train += items[k:]
    return train, ev


def run(corpus_path, model_name=DEFAULT_LM, device_str=None, eval_frac=0.2,
        e1=200, e2a=120, e2b=150):
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device}  corpus={corpus_path}  eval_frac={eval_frac}")

    # 1. substrate pass (enriched graph from THIS corpus's patches)
    print("\n[1] substrate pass...")
    _, stats = build_substrate_graph(corpus_path=corpus_path, out_path=SUBSTRATE_OUT)
    print(f"    +{stats['substrate_nodes_added']} nodes, +{stats['relations_added']} relations "
          f"-> {stats['out_path']}")

    # 2. real providers
    print("\n[2] loading real providers (Qwen + mpnet)...")
    provider = FrozenQwenHInitProvider(model_name, device=device)
    model, tok = provider.model, provider.tok
    lm_dim = provider.hidden_size
    embedder = RealEmbedder(device)
    gnn = RGCNEncoder().to(device).eval()
    for p in gnn.parameters():
        p.requires_grad_(False)
    goal_enc = GoalEncoder().to(device).eval()

    gpath = _graph_path()
    graph = load_persisted_graph(gpath)
    pos = corpus_to_stage1_examples(corpus_path, gnn=gnn, embedder=embedder,
                                    h_init_provider=provider, device=device,
                                    lm_dim=lm_dim, graph_path=gpath, hops=1)
    negs = make_real_negatives(provider, embedder, gnn, graph, device, lm_dim, pos[0])
    examples = pos + negs

    # 3. coverage report
    cov = {"plan": 0, "evid": 0, "slot": 0, "epi": 0, "inv": 0, "shortcut": 0}
    for e in pos:
        cov["plan"] += int(e.plan_anchor is not None); cov["evid"] += int(e.evid_anchor is not None)
        cov["slot"] += int(e.slot_target is not None); cov["epi"] += int(e.epi_target is not None)
        cov["inv"] += int(e.inv_target is not None); cov["shortcut"] += int(e.shortcut_target is not None)
    print(f"\n[3] coverage over {len(pos)} positive traces:")
    for k, v in cov.items():
        print(f"    {k:9s} {v}/{len(pos)} ({v/max(1,len(pos)):.0%})")

    # 4. split
    train, ev = _stratified_split(examples, eval_frac)
    print(f"\n[4] split: {len(train)} train / {len(ev)} held-out  "
          f"(eval tags: {dict((t, sum(1 for e in ev if e.tag==t)) for t in set(e.tag for e in ev))})")
    if len(ev) < 5:
        print("    WARNING: held-out set is tiny — metrics are indicative only, not conclusive.")

    # 5. integrated Stage 1 -> 2A -> 2B on TRAIN
    print("\n[5] integrated Stage 1 -> 2A -> 2B (train split)...")
    adapter = V5AttentionAdapter(r_plan=3, r_evidence=4, lm_hidden_dim=lm_dim, gate_init=GATE_INIT).to(device)
    train_pos = [e for e in train if e.tag != "negative"]
    Stage1Trainer(adapter, Stage1Config(epochs=e1, lr=1e-3)).train(train_pos)
    Stage2Trainer(adapter, Stage2Config(sub_stage="2A", epochs=e2a, lr=2e-4)).train(train)
    Stage2Trainer(adapter, Stage2Config(sub_stage="2B", epochs=e2b, lr=1e-4,
                                        lambda_delta=1.0, qkv_lr_scale=0.3)).train(train)

    # 6. held-out metrics
    print("\n[6] HELD-OUT METRICS")
    m = heldout_metrics(adapter, ev)
    print(f"    plan  node  precision@1={m['plan_node']['precision@1']:.2f} "
          f"recall@gold={m['plan_node']['recall@gold']:.2f} (n={m['plan_node']['n']})")
    print(f"    evid  node  precision@1={m['evid_node']['precision@1']:.2f} "
          f"recall@gold={m['evid_node']['recall@gold']:.2f} (n={m['evid_node']['n']})")
    print(f"    head acc (strict all-node): " + "  ".join(f"{k}={v:.2f}" for k, v in m["head_acc"].items()))
    print(f"    epi per-node acc: {m['epi_per_node_acc']:.2f}  (less strict than all-node)")
    print(f"    fallback: " + "  ".join(f"{k}={v:.2f}" for k, v in m["fallback"].items()))
    print(f"    write ratio: " + "  ".join(f"{k}={v:.3f}" for k, v in m["write_ratio"].items()))
    print("\n    KEY: fallback applicable should DROP while blocked/negative stay high.")
    print("    On a tiny held-out set this is indicative; scale to 100-300 traces to conclude.")
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="artifacts/phase15/phase15_corpus.jsonl")
    ap.add_argument("--model", default=DEFAULT_LM)
    ap.add_argument("--device", default=None)
    ap.add_argument("--eval-frac", type=float, default=0.2)
    ap.add_argument("--e1", type=int, default=200)
    ap.add_argument("--e2a", type=int, default=120)
    ap.add_argument("--e2b", type=int, default=150)
    a = ap.parse_args()
    run(a.corpus, a.model, a.device, a.eval_frac, a.e1, a.e2a, a.e2b)
