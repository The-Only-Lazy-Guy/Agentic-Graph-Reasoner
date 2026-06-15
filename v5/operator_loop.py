"""Operator-Attention REAL LOOP — retrieval over a GROWN graph + op-signed injection on reasoning.

All prior operator tests hand-matched the node to the question. This closes the loop: for each
reasoning query we RETRIEVE the relevant failure_pattern(s) from the grown graph by similarity
(noisy, scaled — 99 INVALIDATE nodes over 12 problems), then compare three injections:

    cold     : no graph
    BLEND    : retrieved fps added (+v)  — plain RAG / array-of-text (op-blind)
    OPERATOR : retrieved fps SUBTRACTED (-v, their op_kind=INVALIDATE)

Each question's WRONG option is the trap the grown fp warns against. INVALIDATE should suppress the
trap; BLEND grounds the wrong-approach text and (per the sign test) tends to HURT. The point: the
operator advantage must SURVIVE real retrieval at graph scale, not just curated pairs.

Embedding = the frozen LM's own mean-pooled hidden state (sentence-transformers segfaults on Windows;
this reuses the already-loaded model). grown fps share heavy lexical cues (AM-GM, incenter, ...) so
crude LM-embedding retrieval surfaces the matching node.

  local 1.5B (L14): V5_LM_TRUST_REMOTE_CODE=1 python -m v5.operator_loop --model Qwen/Qwen2.5-1.5B --layer 14
  4B (L26, A40):    V5_LM_TRUST_REMOTE_CODE=1 python -m v5.operator_loop --layer 26
"""
from __future__ import annotations

import argparse
import json
import os
import torch

from v5.lm_loader import load_frozen_lm
from v5.operator_injector import OperatorInjector

# (query, correct, wrong) — wrong option = the trap a GROWN failure_pattern warns against.
# Grounded in cot_batch1 problems (ot114k_000000/02/06/07/16/17), framed as forced choice on the key step.
PROBES = [
    ("To prove the LOWER bound for Sum 1/(a(1+b)) >= 3/(1+abc), bounding each denominator by AM-GM "
     "(1+b >= 2 sqrt b) is (A) wrong, it gives an upper bound (B) the correct key step. Answer: (", "A", "B"),
    ("Reducing Sum 1/(a(1+b)) >= 3/(1+abc) via Cauchy-Schwarz to proving Sum a(1+b) <= 3(1+abc) is "
     "(A) doomed, a counterexample exists (B) a valid reduction. Answer: (", "A", "B"),
    ("Two triangles that share the same incenter must have the same incircle and inradius: "
     "(A) false, the inradius can differ (B) true. Answer: (", "A", "B"),
    ("When counting triples whose product is divisible by 10, letting ONE entry satisfy both the "
     "factor-5 and the even requirement is (A) wrong, 5 is not even (B) fine. Answer: (", "A", "B"),
    ("Accepting a candidate p just because p is prime, without checking its remainders modulo all "
     "smaller primes, is (A) a mistake (B) sufficient. Answer: (", "A", "B"),
    ("Arguing that the partial sums stay below 1 from the rapid growth of the denominators alone is "
     "(A) insufficient, not a proof (B) a complete proof. Answer: (", "A", "B"),
    ("In modular arithmetic, canceling a common factor from both sides is valid (A) only if that "
     "factor is invertible modulo m (B) always. Answer: (", "A", "B"),
]


@torch.no_grad()
def embed(model, tok, text, dev, layer):
    ids = tok(text, return_tensors="pt", truncation=True, max_length=256).input_ids.to(dev)
    hs = model(ids, output_hidden_states=True).hidden_states[layer]   # [1,T,H]
    v = hs[0].mean(0).float()
    return v / (v.norm() + 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--layer", type=int, default=26)
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--graph", default="graphs/grown_graph6.json")
    ap.add_argument("--topk", type=int, default=1,
                    help="top-k fps to inject; >1 sums op vectors so divide alpha by k or it overshoots "
                         "(topk=1 is the clean single-INVALIDATE regime; topk=2 shows RAG-blend actively harms)")
    ap.add_argument("--emb-layer", type=int, default=-1, help="hidden-state index for retrieval (-1=auto mid)")
    a = ap.parse_args()
    trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
    from transformers import AutoTokenizer
    model = load_frozen_lm(a.model); model.eval()
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=trust)
    inj = OperatorInjector(model, tok, a.layer, a.alpha)
    dev = next(model.parameters()).device
    nlayers = model.config.num_hidden_layers
    emb_layer = a.emb_layer if a.emb_layer >= 0 else nlayers // 2

    g = json.load(open(a.graph, encoding="utf-8"))
    nodes = g.get("nodes"); nodes = list(nodes.values()) if isinstance(nodes, dict) else nodes
    fps = [n for n in nodes if n.get("node_type") == "failure_pattern" and (n.get("text") or "").strip()]
    print(f"loaded | layer {a.layer} | emb_layer {emb_layer} | {len(fps)} INVALIDATE nodes | topk {a.topk}", flush=True)

    # precompute the grown-graph fp embeddings ONCE (real retrieval index)
    fp_emb = torch.stack([embed(model, tok, n["text"], dev, emb_layer) for n in fps])  # [N,H]

    def belief(q, R, W, v):
        lg = inj.answer_logits(q, v)
        return float(lg[tok(R, add_special_tokens=False).input_ids[-1]]
                     - lg[tok(W, add_special_tokens=False).input_ids[-1]])

    rows = []
    for q, R, W in PROBES:
        qv = embed(model, tok, q, dev, emb_layer)
        sims = fp_emb @ qv
        top = torch.topk(sims, a.topk).indices.tolist()
        retrieved = [fps[i]["text"] for i in top]
        inj.alpha = a.alpha / max(1, len(retrieved))    # per-node alpha: summing k nodes else overshoots
        v_op = inj.combine([(t, "INVALIDATE") for t in retrieved], q)   # op_kind: subtract
        v_bl = inj.combine([(t, "ASSERT") for t in retrieved], q)       # RAG blend: add
        inj.alpha = a.alpha
        cold = belief(q, R, W, None)
        oper = belief(q, R, W, v_op)
        blend = belief(q, R, W, v_bl)
        rows.append((cold, blend, oper))
        print(f"  cold {cold:+.2f}  BLEND {blend:+.2f}  OPERATOR {oper:+.2f}   <- retr[0]: {retrieved[0][:70]}", flush=True)

    import statistics as st
    n = len(rows)
    ob = sum(1 for c, b, o in rows if o > b); oc = sum(1 for c, b, o in rows if o > c)
    correct_cold = sum(1 for c, b, o in rows if c > 0)
    correct_op = sum(1 for c, b, o in rows if o > 0)
    mc, mb, mo = (st.mean(r[i] for r in rows) for i in (0, 1, 2))
    print(f"\n=== Operator REAL LOOP (retrieval over {len(fps)} grown fps, layer {a.layer}, alpha {a.alpha}) ===")
    print(f"  belief = logit(correct) - logit(trap)")
    print(f"  accuracy(belief>0): cold {correct_cold}/{n} -> OPERATOR {correct_op}/{n}")
    print(f"  OPERATOR beats BLEND: {ob}/{n} | OPERATOR beats cold: {oc}/{n}")
    print(f"  mean: cold {mc:+.2f} | BLEND {mb:+.2f} | OPERATOR {mo:+.2f}")
    ok = ob >= 0.7 * n and (mo - mb) > 0.3 and mo > mc
    print(f"\n  RESULT: {'PASS — retrieved INVALIDATE beats RAG blend + cold, at graph scale' if ok else 'see numbers (may be narrow / layer-sensitive)'}")


if __name__ == "__main__":
    main()
