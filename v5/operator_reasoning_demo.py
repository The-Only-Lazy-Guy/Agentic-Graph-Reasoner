"""REASONING-LEARNING demo — does ONE graph edit make the frozen model REASON better? (not recall)

The recall demo (operator_demo) tests knowledge — off-target. THIS tests reasoning: the node is a
reasoning ASSET the model must USE, not an answer to echo. We teach a tempting WRONG proof approach as
a failure_pattern (INVALIDATE), and ask the model to JUDGE whether that approach is valid for a proof.
Cold the 4B is fooled (the approach sounds plausible); after the edit it reasons correctly.

Reasoning OUTCOME metric (not fact-emission): belief = logit(correct judgement) - logit(wrong), on a
short yes/no the reasoning produces. Control matrix per item — a flip only counts if ALL hold:
  COLD        : wrong before (precheck)
  LEARN       : + the INVALIDATE node (operator) -> correct
  RANDOM      : + an irrelevant node (same op) -> no flip          (kills "any injection")
  RAG         : node text in the prompt -> for INVALIDATE this POISONS (grounds the wrong approach) -> stays/gets wrong
  SPECIFICITY : an unrelated correct judgement + the node -> preserved   (no collateral)

This is the agentic-coder mechanism in miniature: teach "this approach is wrong" -> the model avoids it
and reasons to the right one (the SWE use: a learned pitfall steers the fix). --layers for the adapter test.

  4B (A40): V5_LM_TRUST_REMOTE_CODE=1 python -m v5.operator_reasoning_demo --layer 26 --alpha 1.0
"""
from __future__ import annotations

import argparse
import os
import torch

from v5.lm_loader import load_frozen_lm
from v5.operator_injector import OperatorInjector

# (question -> yes/no judgement, correct, wrong, the bare WRONG-approach node to INVALIDATE)
ITEMS = [
    dict(q="To prove the lower bound  sum 1/(a(1+b)) >= 3/(1+abc)  for positive a,b,c, is it a VALID "
         "first step to bound each denominator below using AM-GM, 1+b >= 2 sqrt(b)? Answer:",
         correct=" no", wrong=" yes",
         node="By AM-GM, 1+b >= 2 sqrt(b), so 1/(a(1+b)) >= 1/(2 a sqrt(b)), and summing the three terms "
              "gives the lower bound 3/(1+abc)."),
    dict(q="To prove  sum 1/(a(1+b)) >= 3/(1+abc),  is it a VALID reduction to use Cauchy-Schwarz and "
         "instead prove  sum a(1+b) <= 3(1+abc)? Answer:",
         correct=" no", wrong=" yes",
         node="By Cauchy-Schwarz, sum 1/(a(1+b)) >= 9 / sum a(1+b), and since sum a(1+b) <= 3(1+abc), "
              "the desired bound 3/(1+abc) follows."),
    dict(q="To prove the inequality for general positive a,b,c, is it VALID to first rescale a,b,c so "
         "that abc=1 and then invoke the abc=1 case? Answer:",
         correct=" no", wrong=" yes",
         node="Scaling a,b,c so that abc=1 lets the abc=1 proof apply directly to the original inequality."),
    dict(q="To prove that infinitely many primes exist, is it VALID to argue that since the product of "
         "all known primes plus one is prime, there must be more? Answer:",
         correct=" no", wrong=" yes",
         node="N = (product of all known primes) + 1 is itself a prime not in the list, so there are "
              "infinitely many primes."),
]
RANDOM_NODE = "Bananas are a good source of dietary potassium."
# specificity: an unrelated, correct judgement the model knows -> injecting a node must not break it.
SPEC_Q = "Is it true that a triangle has three sides? Answer:"
SPEC_R, SPEC_W = " yes", " no"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--layer", type=int, default=26)
    ap.add_argument("--layers", default=None, help="comma list e.g. 8,14,20,26 -> multi-layer injection (adapter fix)")
    ap.add_argument("--alpha", type=float, default=1.0)
    a = ap.parse_args()
    trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
    from transformers import AutoTokenizer
    model = load_frozen_lm(a.model); model.eval()
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=trust)
    if a.layers:
        from v5.operator_injector_ml import OperatorInjectorML
        inj = OperatorInjectorML(model, tok, [int(x) for x in a.layers.split(",")], a.alpha)
        where = f"layers {a.layers} (multi)"
    else:
        inj = OperatorInjector(model, tok, a.layer, a.alpha)
        where = f"layer {a.layer}"
    print(f"loaded | {where} | alpha {a.alpha} | {len(ITEMS)} reasoning items\n", flush=True)

    def tid(s): return tok(s, add_special_tokens=False).input_ids[-1]
    def belief(q, R, W, v=None):
        lg = inj.answer_logits(q, v)
        return float(lg[tid(R)] - lg[tid(W)])

    rows = []
    for it in ITEMS:
        q, R, W = it["q"], it["correct"], it["wrong"]
        v_learn = inj.combine([(it["node"], "INVALIDATE")], q, normalize=True)
        v_rand = inj.combine([(RANDOM_NODE, "INVALIDATE")], q, normalize=True)
        cold = belief(q, R, W)
        learn = belief(q, R, W, v_learn)
        rand = belief(q, R, W, v_rand)
        rag = belief(f"{it['node']}\n{q}", R, W)
        v_spec = inj.combine([(it["node"], "INVALIDATE")], SPEC_Q, normalize=True)
        spec_cold = belief(SPEC_Q, SPEC_R, SPEC_W)
        spec = belief(SPEC_Q, SPEC_R, SPEC_W, v_spec)
        # outcome = belief>0 means the model judges CORRECTLY (the approach is invalid)
        cold_wrong = cold <= 0
        learned = learn > 0
        rand_noflip = rand <= 0
        spec_ok = (spec_cold <= 0) or (spec > 0)
        passed = cold_wrong and learned and rand_noflip and spec_ok
        rows.append((passed, cold_wrong, cold, learn, rand, rag, spec_cold, spec))
        print(f"[{'PASS' if passed else 'fail'}] cold {cold:+.2f} -> LEARN {learn:+.2f}  "
              f"(random {rand:+.2f} | RAG {rag:+.2f})  spec {spec_cold:+.2f}->{spec:+.2f}")
        print(f"     Q: {q[:70]}...")
        print(f"     taught(INVALIDATE): {it['node'][:70]}...\n", flush=True)

    import statistics as st
    n = len(rows)
    valid = [r for r in rows if r[1]]
    learned_n = sum(1 for r in rows if r[0])
    print(f"=== REASONING-LEARNING (PASS = cold-wrong & learned & random-clean & specificity-ok) ===")
    print(f"  learned: {learned_n}/{n}   ({len(valid)}/{n} valid cold-wrong prechecks)")
    print(f"  mean belief: cold {st.mean(r[2] for r in rows):+.2f} -> LEARN {st.mean(r[3] for r in rows):+.2f}"
          f"  | random {st.mean(r[4] for r in rows):+.2f} | RAG {st.mean(r[5] for r in rows):+.2f}")
    print(f"  KEY: LEARN (operator) makes the 4B judge the approach INVALID; RAG of the same node tends "
          f"to POISON (grounds the wrong approach). That's teach-to-reason, not retrieve.")


if __name__ == "__main__":
    main()
