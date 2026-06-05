"""Stage 3 — answer-loss fine-tuning of the injection adapter (frozen LM).

Stages 1/2 trained the adapter to ATTEND the right nodes (routing precision) and write
a bounded delta. But "attends correctly" != "generates better". Stage 3 closes that:
forward the FROZEN LM on (question + gold answer) WITH injection on, take the CE/NLL on
the answer tokens, and backprop into the ADAPTER. The adapter learns to inject a signal
that makes the frozen LM predict the gold answer better -- grounding improves generation,
with NO LM training (matches the v2 design: frozen LM, the adapter does the grounding).

Metric (no Docker): held-out answer-NLL COLD vs INJECTED. injected < cold = the grounding
makes the gold patch more likely. This is the end-task proxy before real test-pass.

  V5_LM_TRUST_REMOTE_CODE=1 python -m v5.training.stage3_answer \
    --model Qwen/Qwen3.5-4B --corpus data/swe/code_phase15_corpus.jsonl \
    --graph graphs/grown_graph4.json --e1 100 --e2a 80 --e2b 80 --e3 2 --n-train 150 --n-eval 60
"""
from __future__ import annotations

import argparse
import statistics as st
from contextlib import nullcontext

import torch

from v5.adapter import GraphAttentionInjector
from v5.cross_attention import V5AttentionAdapter
from v5.gnn_encoder import RGCNEncoder
from v5.goal_encoder import GoalEncoder
from v5.perturbation_baseline import _stub_graph
from v5.training.bridge import corpus_to_stage1_examples, load_persisted_graph
from v5.training.dataset import Phase15Dataset
from v5.training.providers import RealEmbedder, FrozenQwenHInitProvider
from v5.training.stage1 import Stage1Trainer, Stage1Config
from v5.training.stage1_real import _graph_path
from v5.training.stage2 import Stage2Trainer, Stage2Config, GATE_INIT


def _prep(injector, graph, embedder, s, device):
    """Build the injection session for a sample (stub graph for code nodes not in graph)."""
    if sum(1 for nid in s.node_ids if nid in graph.nodes) >= 2:
        g = graph
        node_ids = [nid for nid in s.node_ids if nid in graph.nodes][:24]
        text_emb = embedder.embed_nodes({nid: (getattr(graph.nodes[nid], "text", "") or "") for nid in node_ids})
    else:
        g = _stub_graph(s)
        node_ids = list(s.node_ids)[:24]
        text_emb = embedder.embed_nodes({nid: s.node_texts.get(nid, "") for nid in node_ids})
    injector.prepare_session(g, node_ids, text_emb, s.task_frame, r_plan=3, r_evidence=4)


def answer_nll(model, tok, injector, s, device, inject: bool):
    """Mean NLL of the gold answer tokens under the (optionally injected) frozen LM."""
    q = (s.question or "")[:1500]
    a = (s.answer_text or "")[:1200]
    if not a.strip():
        return None
    p_ids = tok(q + "\n", return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
    a_ids = tok(a, add_special_tokens=False, return_tensors="pt", truncation=True, max_length=300).input_ids.to(device)
    if a_ids.shape[1] == 0:
        return None
    full = torch.cat([p_ids, a_ids], dim=1)
    ctx = injector.inject(model) if inject else nullcontext()
    with ctx:
        logits = model(full).logits
    plen = p_ids.shape[1]
    lp = torch.log_softmax(logits[0, plen - 1:-1].float(), dim=-1)
    idx = torch.arange(a_ids.shape[1], device=lp.device)
    return -lp[idx, a_ids[0]].mean()


def run(model_name, corpus, graph_path=None, e1=100, e2a=80, e2b=80, e3=2,
        lr3=1e-4, n_train=150, n_eval=60, device_str=None,
        adapter_ckpt="artifacts/stage_cache/adapter_code.pt", retrain=False):
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device}  lm={model_name}  corpus={corpus}")

    provider = FrozenQwenHInitProvider(model_name, device=device)
    model, tok = provider.model, provider.tok
    lm_dim = provider.hidden_size
    embedder = RealEmbedder(device)
    gnn = RGCNEncoder().to(device).eval()
    for p in gnn.parameters():
        p.requires_grad_(False)
    goal_enc = GoalEncoder().to(device).eval()

    import os
    gpath = graph_path or _graph_path()
    graph = load_persisted_graph(gpath)
    adapter = V5AttentionAdapter(r_plan=3, r_evidence=4, lm_hidden_dim=lm_dim, gate_init=GATE_INIT).to(device)

    if adapter_ckpt and os.path.exists(adapter_ckpt) and not retrain:
        adapter.load_state_dict(torch.load(adapter_ckpt, map_location=device))
        print(f"loaded adapter checkpoint {adapter_ckpt}  ->  SKIP stages 1/2 + h_init build")
    else:
        print("\n--- building examples (LM h_init, slow) + Stages 1/2 ---")
        pos = corpus_to_stage1_examples(corpus, gnn=gnn, embedder=embedder, h_init_provider=provider,
                                        device=device, lm_dim=lm_dim, graph_path=gpath, hops=1)
        Stage1Trainer(adapter, Stage1Config(epochs=e1, lr=1e-3)).train(pos)
        Stage2Trainer(adapter, Stage2Config(sub_stage="2A", epochs=e2a, lr=2e-4)).train(pos)
        Stage2Trainer(adapter, Stage2Config(sub_stage="2B", epochs=e2b, lr=1e-4,
                                            lambda_delta=1.0, qkv_lr_scale=0.3)).train(pos)
        if adapter_ckpt:
            os.makedirs(os.path.dirname(adapter_ckpt) or ".", exist_ok=True)
            torch.save(adapter.state_dict(), adapter_ckpt)
            print(f"saved adapter -> {adapter_ckpt}  (reuse with --adapter-ckpt; --retrain to refresh)")

    injector = GraphAttentionInjector(adapter, gnn, goal_enc, device=device)
    injector.inject_all_positions = True   # teacher-forced answer-loss: condition ALL answer tokens
    samples = Phase15Dataset(corpus).samples
    train_s = samples[:n_train]
    eval_s = samples[n_train:n_train + n_eval]
    print(f"\n--- Stage 3: answer-loss on adapter ({len(train_s)} train / {len(eval_s)} eval) ---")

    opt = torch.optim.AdamW([p for p in adapter.parameters() if p.requires_grad], lr=lr3)
    for ep in range(e3):
        adapter.train()
        tot = 0.0; n = 0
        for s in train_s:
            _prep(injector, graph, embedder, s, device)
            nll = answer_nll(model, tok, injector, s, device, inject=True)
            if nll is None or not torch.isfinite(nll):
                continue
            opt.zero_grad(); nll.backward()
            torch.nn.utils.clip_grad_norm_([p for p in adapter.parameters() if p.requires_grad], 1.0)
            opt.step()
            tot += float(nll.item()); n += 1
        print(f"  [S3] epoch {ep+1}  mean answer-NLL(inject) {tot/max(1,n):.4f}  ({n} rows)", flush=True)

    # ── eval: cold vs injected NLL on HELD-OUT (does grounding lower gold NLL?) ──
    adapter.eval()
    cold, inj = [], []
    with torch.no_grad():
        for s in eval_s:
            _prep(injector, graph, embedder, s, device)
            c = answer_nll(model, tok, injector, s, device, inject=False)
            i = answer_nll(model, tok, injector, s, device, inject=True)
            if c is not None and i is not None and torch.isfinite(c) and torch.isfinite(i):
                cold.append(float(c)); inj.append(float(i))
    if not cold:
        print("no eval rows scored"); return
    mc, mi = st.mean(cold), st.mean(inj)
    win = sum(1 for c, i in zip(cold, inj) if i < c)
    print(f"\n=== STAGE 3 held-out answer-NLL (lower=better) ===")
    print(f"  cold {mc:.4f}  ->  injected {mi:.4f}   (delta {mc - mi:+.4f})")
    print(f"  injection BETTER on {win}/{len(cold)} ({100*win/len(cold):.0f}%)")
    print("  injected < cold  =  the grounding makes the gold patch more likely (end-task proxy).")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stage 3: answer-loss adapter fine-tune (frozen LM).")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--corpus", default="data/swe/code_phase15_corpus.jsonl")
    ap.add_argument("--graph", default=None)
    ap.add_argument("--e1", type=int, default=100)
    ap.add_argument("--e2a", type=int, default=80)
    ap.add_argument("--e2b", type=int, default=80)
    ap.add_argument("--e3", type=int, default=2)
    ap.add_argument("--lr3", type=float, default=1e-4)
    ap.add_argument("--n-train", type=int, default=150)
    ap.add_argument("--n-eval", type=int, default=60)
    ap.add_argument("--device", default=None)
    ap.add_argument("--adapter-ckpt", default="artifacts/stage_cache/adapter_code.pt",
                    help="save/load the stages-1/2 adapter here -> skip retraining on reruns")
    ap.add_argument("--retrain", action="store_true", help="force re-run stages 1/2 (refresh checkpoint)")
    a = ap.parse_args(argv)
    run(a.model, a.corpus, a.graph, a.e1, a.e2a, a.e2b, a.e3, a.lr3, a.n_train, a.n_eval, a.device,
        adapter_ckpt=a.adapter_ckpt, retrain=a.retrain)


if __name__ == "__main__":
    raise SystemExit(main())
