"""HARDER reasoning suite — does the typed-operator graph edit make the 4B REASON better, vs RAG?

Toys failed because a strong 4B isn't fooled by them. This targets the 4B's actual failure zone:
  - competition-math: judge a flawed proof (the grown INVALIDATE traps) — subtle, the 4B can be fooled.
  - code-reasoning: predict buggy-code behaviour / pick the right fix — the agentic-coder ladder.

The node is a reasoning ASSET, not an answer. Metric = belief on the reasoning OUTCOME
(logit(correct) - logit(wrong)). The robust claim (not a toy flip): OPERATOR (typed inject) beats RAG
(same text in context) and beats COLD; for INVALIDATE assets, RAG of the wrong approach POISONS.
Controls: random node (no help), specificity (unrelated correct judgement preserved). Reports which
items genuinely fool the model cold.

  4B (A40): V5_LM_TRUST_REMOTE_CODE=1 python -m v5.reasoning_suite --layer 26 --alpha 1.0
"""
from __future__ import annotations

import argparse
import os
import torch

from v5.lm_loader import load_frozen_lm
from v5.operator_injector import OperatorInjector

# (tag, question -> A/B, correct, wrong, asset node, op). INVALIDATE asset = the bare WRONG approach;
# ASSERT asset = the key insight the model must USE.
ITEMS = [
    ("math", "A student proves  sum 1/(a(1+b)) >= 3/(1+abc)  by writing 1+b >= 2*sqrt(b), hence "
     "1/(a(1+b)) <= 1/(2a*sqrt(b)), then summing the three terms. This proof is (A) correct "
     "(B) flawed, the inequality points the wrong way. Answer: (", "B", "A",
     "Since 1+b >= 2 sqrt(b), 1/(a(1+b)) <= 1/(2 a sqrt(b)), and summing the three terms proves the "
     "lower bound 3/(1+abc).", "INVALIDATE"),
    ("math", "A student argues there are infinitely many primes: 'N = (product of all primes up to p) "
     "+ 1 is itself always prime, giving a brand-new prime.' This argument is (A) correct (B) flawed, "
     "N need not be prime. Answer: (", "B", "A",
     "N = (product of all primes up to p) + 1 is itself a prime not in the list.", "INVALIDATE"),
    ("math", "To prove  sum 1/(a(1+b)) >= 3/(1+abc), a student uses Cauchy-Schwarz to reduce it to "
     "proving  sum a(1+b) <= 3(1+abc). This reduction is (A) valid (B) doomed, a counterexample exists. "
     "Answer: (", "B", "A",
     "By Cauchy-Schwarz, sum 1/(a(1+b)) >= 9 / sum a(1+b), and since sum a(1+b) <= 3(1+abc), the bound "
     "follows.", "INVALIDATE"),

    ("code", "def add(x, lst=[]):\n    lst.append(x)\n    return lst\n\nadd(1) returned [1]. The next "
     "call add(2) returns (A) [2] (B) [1, 2]. Answer: (", "B", "A",
     "In Python a default list argument is created once and shared across all calls, so it accumulates.",
     "ASSERT"),
    ("code", "d = {1: 'a'}. In Python, evaluating d[1.0] gives (A) a KeyError (B) the value 'a'. "
     "Answer: (", "B", "A",
     "In Python 1 and 1.0 compare equal and hash identically, so they index the same dict key.",
     "ASSERT"),
    ("code", "An issue: a function crashes on empty input. The cleaner, more targeted fix is (A) add an "
     "explicit guard 'if not items: return []' at the top (B) wrap the whole body in a broad try/except. "
     "Answer: (", "A", "B",
     "Prefer an explicit guard for a known edge case over broad exception handling, which hides other "
     "errors.", "ASSERT"),
]
RANDOM_NODE = "Bananas are a good source of dietary potassium."
SPEC_Q = "Is the integer 10 an even number? (A) yes (B) no. Answer: ("
SPEC_R, SPEC_W = "A", "B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--layer", type=int, default=26)
    ap.add_argument("--layers", default=None)
    ap.add_argument("--alpha", type=float, default=1.0)
    a = ap.parse_args()
    trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
    from transformers import AutoTokenizer
    model = load_frozen_lm(a.model); model.eval()
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=trust)
    if a.layers:
        from v5.operator_injector_ml import OperatorInjectorML
        inj = OperatorInjectorML(model, tok, [int(x) for x in a.layers.split(",")], a.alpha)
    else:
        inj = OperatorInjector(model, tok, a.layer, a.alpha)
    print(f"loaded | {'layers '+a.layers if a.layers else 'layer '+str(a.layer)} | alpha {a.alpha} | "
          f"{len(ITEMS)} hard items\n", flush=True)

    def tid(s): return tok(s, add_special_tokens=False).input_ids[-1]
    def belief(q, R, W, v=None):
        lg = inj.answer_logits(q, v)
        return float(lg[tid(R)] - lg[tid(W)])

    rows = []
    for tag, q, R, W, node, op in ITEMS:
        v_op = inj.combine([(node, op)], q, normalize=True)
        v_rand = inj.combine([(RANDOM_NODE, op)], q, normalize=True)
        cold = belief(q, R, W)
        oper = belief(q, R, W, v_op)
        rag = belief(f"{node}\n{q}", R, W)
        rand = belief(q, R, W, v_rand)
        v_spec = inj.combine([(node, op)], SPEC_Q, normalize=True)
        spec_cold = belief(SPEC_Q, SPEC_R, SPEC_W)
        spec = belief(SPEC_Q, SPEC_R, SPEC_W, v_spec)
        fooled = cold <= 0                 # model wrong/uncertain cold = the interesting case
        op_beats_rag = oper > rag
        op_beats_cold = oper > cold
        spec_ok = (spec_cold <= 0) or (spec > 0)
        rows.append((tag, cold, oper, rag, rand, fooled, op_beats_rag, op_beats_cold, spec_ok, op))
        print(f"[{tag}|{op[:4]}] cold {cold:+.2f}  OPERATOR {oper:+.2f}  RAG {rag:+.2f}  random {rand:+.2f}"
              f"  {'(fooled cold)' if fooled else ''}{'' if spec_ok else ' SPEC-BROKEN'}")
        print(f"      {q.splitlines()[0][:74]}", flush=True)

    import statistics as st
    n = len(rows)
    for grp in ("math", "code", None):
        rs = [r for r in rows if grp is None or r[0] == grp]
        if not rs: continue
        name = (grp or "ALL").upper()
        print(f"\n=== {name} ({len(rs)}) ===")
        print(f"  cold-fooled (model wrong cold): {sum(1 for r in rs if r[5])}/{len(rs)}")
        print(f"  OPERATOR beats RAG: {sum(1 for r in rs if r[6])}/{len(rs)} | beats cold: {sum(1 for r in rs if r[7])}/{len(rs)}")
        print(f"  mean: cold {st.mean(r[1] for r in rs):+.2f} | OPERATOR {st.mean(r[2] for r in rs):+.2f} | RAG {st.mean(r[3] for r in rs):+.2f}")
    print(f"\n  specificity broken on: {sum(1 for r in rows if not r[8])}/{n}")
    print("  CLAIM: typed-operator inject beats RAG (and RAG of a wrong approach poisons) on hard reasoning.")


if __name__ == "__main__":
    main()
