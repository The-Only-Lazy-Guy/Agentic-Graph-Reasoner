"""Operator REAL LOOP v2 — EDGE-GATED: retrieve a correct node, hop its `contradicts` edge to the
bare-misconception trap, then op-signed inject. The graph EDGE connects retrieval-friendly content
to injection-friendly content — an array can't do this.

Why v2: v1 retrieved the bare failure_pattern directly, but a bare misconception ("Since 1+b>=2sqrt b,
1/(a(1+b))>=1/(2a sqrt b)") carries no query cues, so query->fp retrieval collapses. The fix is
structural: retrieve the topical CORRECT node, then follow its `contradicts` edge to the trap. We
then compare, with the SAME two nodes present:
    cold     : no graph
    BLEND    : ASSERT(correct) + ASSERT(trap)        (array: both added)
    OPERATOR : ASSERT(correct) + INVALIDATE(trap)    (edge sign: subtract the trap)
The ONLY difference is the trap's sign (the proven subtract). If OPERATOR beats BLEND, suppressing
the edge-linked trap helps where blending it in does not — the edge is load-bearing.

Retrieval is reported two ways to keep it honest (v5 assigns retrieval to a trained ranker):
  --scope all     : real query->correct-node retrieval over ALL refuted correct nodes (noisy).
  --scope problem : retrieval restricted to the probe's source problem (coarse problem-level
                    retrieval assumed) -> isolates the EDGE-HOP + operator from fine retrieval.

  local 1.5B: V5_LM_TRUST_REMOTE_CODE=1 python -m v5.operator_loop_v2 --model Qwen/Qwen2.5-1.5B --layer 14 --scope problem
  4B (A40):  V5_LM_TRUST_REMOTE_CODE=1 python -m v5.operator_loop_v2 --layer 26 --alpha 1.0 --scope problem
"""
from __future__ import annotations

import argparse
import json
import os
import torch

from v5.lm_loader import load_frozen_lm
from v5.operator_injector import OperatorInjector

# (query, correct, wrong, source_doc) — source_doc = the cot_batch1 problem the trap belongs to.
PROBES = [
    ("To prove the LOWER bound for Sum 1/(a(1+b)) >= 3/(1+abc), bounding each denominator by AM-GM "
     "(1+b >= 2 sqrt b) is (A) wrong, it gives an upper bound (B) the correct key step. Answer: (",
     "A", "B", "ot114k_000000"),
    ("Reducing Sum 1/(a(1+b)) >= 3/(1+abc) via Cauchy-Schwarz to proving Sum a(1+b) <= 3(1+abc) is "
     "(A) doomed, a counterexample exists (B) a valid reduction. Answer: (", "A", "B", "ot114k_000000"),
    ("Two triangles that share the same incenter must have the same incircle and inradius: "
     "(A) false, the inradius can differ (B) true. Answer: (", "A", "B", "ot114k_000002"),
    ("When counting triples whose product is divisible by 10, letting ONE entry satisfy both the "
     "factor-5 and the even requirement is (A) wrong, 5 is not even (B) fine. Answer: (", "A", "B", "ot114k_000006"),
    ("Accepting a candidate p just because p is prime, without checking its remainders modulo all "
     "smaller primes, is (A) a mistake (B) sufficient. Answer: (", "A", "B", "ot114k_000007"),
    ("Arguing that the partial sums stay below 1 from the rapid growth of the denominators alone is "
     "(A) insufficient, not a proof (B) a complete proof. Answer: (", "A", "B", "ot114k_000016"),
    ("In modular arithmetic, canceling a common factor from both sides is valid (A) only if that "
     "factor is invertible modulo m (B) always. Answer: (", "A", "B", "ot114k_000017"),
]


def load_candidates(path):
    """-> nodes{id:{text,nt,doc}}, and correct_id->fp_id map from fp<->correct contradicts edges."""
    nodes = {}
    edges = []
    for line in open(path, encoding="utf-8"):
        r = (json.loads(line).get("raw_edit") or {})
        if r.get("op") == "add_node":
            nodes[r["node_id"]] = {"text": r.get("text", ""), "nt": r.get("node_type"),
                                   "doc": (r.get("metadata") or {}).get("kb_source_doc")}
        elif r.get("op") == "add_edge" and r.get("relation") == "contradicts":
            edges.append((r.get("src") or r.get("source"), r.get("dst") or r.get("target")))
    correct2fp = {}
    for s, d in edges:
        ns, nd = nodes.get(s), nodes.get(d)
        if not ns or not nd:
            continue
        if ns["nt"] == "failure_pattern" and nd["nt"] != "failure_pattern":
            correct2fp[d] = s          # correct d refuted by trap s
        elif nd["nt"] == "failure_pattern" and ns["nt"] != "failure_pattern":
            correct2fp[s] = d
    return nodes, correct2fp


@torch.no_grad()
def embed(model, tok, text, dev, layer):
    ids = tok(text, return_tensors="pt", truncation=True, max_length=256).input_ids.to(dev)
    v = model(ids, output_hidden_states=True).hidden_states[layer][0].mean(0).float()
    return v / (v.norm() + 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--layer", type=int, default=26)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--candidates", default="artifacts/graph_growth/fp_bare_candidates.jsonl")
    ap.add_argument("--scope", choices=["all", "problem"], default="problem",
                    help="all: retrieve correct node over every refuted node; problem: scope to the probe's source doc")
    ap.add_argument("--emb-layer", type=int, default=-1)
    a = ap.parse_args()
    trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
    from transformers import AutoTokenizer
    model = load_frozen_lm(a.model); model.eval()
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=trust)
    inj = OperatorInjector(model, tok, a.layer, a.alpha)
    dev = next(model.parameters()).device
    emb_layer = a.emb_layer if a.emb_layer >= 0 else model.config.num_hidden_layers // 2

    nodes, correct2fp = load_candidates(a.candidates)
    correct_ids = list(correct2fp.keys())
    print(f"loaded | layer {a.layer} | scope {a.scope} | {len(correct_ids)} refuted correct nodes "
          f"(each ->contradicts-> a trap) | emb_layer {emb_layer}", flush=True)
    emb_cache = {cid: embed(model, tok, nodes[cid]["text"], dev, emb_layer) for cid in correct_ids}

    def belief(q, R, W, v):
        lg = inj.answer_logits(q, v)
        return float(lg[tok(R, add_special_tokens=False).input_ids[-1]]
                     - lg[tok(W, add_special_tokens=False).input_ids[-1]])

    rows = []
    for q, R, W, doc in PROBES:
        pool = [c for c in correct_ids if nodes[c]["doc"] == doc] if a.scope == "problem" else correct_ids
        if not pool:
            print(f"  [skip] no refuted nodes for {doc}"); continue
        qv = embed(model, tok, q, dev, emb_layer)
        cid = max(pool, key=lambda c: float(emb_cache[c] @ qv))
        fid = correct2fp[cid]
        correct_t, trap_t = nodes[cid]["text"], nodes[fid]["text"]
        cold = belief(q, R, W, None)
        blend = belief(q, R, W, inj.combine([(correct_t, "ASSERT"), (trap_t, "ASSERT")], q, normalize=True))
        oper = belief(q, R, W, inj.combine([(correct_t, "ASSERT"), (trap_t, "INVALIDATE")], q, normalize=True))
        rows.append((cold, blend, oper))
        print(f"  cold {cold:+.2f}  BLEND {blend:+.2f}  OPERATOR {oper:+.2f}", flush=True)
        print(f"      retr correct: {correct_t[:62]}", flush=True)
        print(f"      hop trap    : {trap_t[:62]}", flush=True)

    import statistics as st
    n = len(rows)
    ob = sum(1 for c, b, o in rows if o > b); oc = sum(1 for c, b, o in rows if o > c)
    acc_c = sum(1 for c, b, o in rows if c > 0); acc_o = sum(1 for c, b, o in rows if o > 0)
    mc, mb, mo = (st.mean(r[i] for r in rows) for i in (0, 1, 2))
    print(f"\n=== EDGE-GATED operator loop (scope {a.scope}, layer {a.layer}, alpha {a.alpha}) ===")
    print(f"  accuracy(belief>0): cold {acc_c}/{n} -> OPERATOR {acc_o}/{n}")
    print(f"  OPERATOR beats BLEND: {ob}/{n} | beats cold: {oc}/{n}")
    print(f"  mean: cold {mc:+.2f} | BLEND {mb:+.2f} | OPERATOR {mo:+.2f}")
    ok = ob >= 0.7 * n and (mo - mb) > 0.3 and mo > mc
    print(f"\n  RESULT: {'PASS — edge-hopped INVALIDATE beats blending the same trap' if ok else 'see numbers'}")


if __name__ == "__main__":
    main()
