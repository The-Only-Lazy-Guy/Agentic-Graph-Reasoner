"""CODE-reasoning suite — does a typed graph edit teach the frozen 4B to reason about code? (SWE ladder)

reasoning_suite found the signal: ASSERT a code insight via the operator beats RAG on the 4B. This
sharpens it with the DECISIVE control: correct-insight vs WRONG-insight (same operator).
  - if the CORRECT insight pushes the right answer AND the WRONG insight pushes the WRONG answer,
    the operator carries SPECIFIC code knowledge -> real learning, not generic steering.
  - if both push the same way, it's generic steer (the doubt the banana-random control left open).

Items target the 4B's failure zone: subtle Python bugs (behaviour prediction) + SWE fix-selection.
Metric = belief on the reasoning OUTCOME, logit(correct) - logit(wrong). Prints IMPORTANT FINDINGS.

  4B (A40): V5_LM_TRUST_REMOTE_CODE=1 python -m v5.code_reasoning_suite --layer 26 --alpha 1.0
"""
from __future__ import annotations

import argparse
import os
import torch

from v5.lm_loader import load_frozen_lm
from v5.operator_injector import OperatorInjector

# (tag, question->A/B, correct, wrong, CORRECT insight, WRONG insight). assets injected as ASSERT.
ITEMS = [
    ("bug", "def add(x, lst=[]):\n    lst.append(x)\n    return lst\n\nadd(1) returned [1]. The next "
     "call add(2) returns (A) [2] (B) [1, 2]. Answer: (", "B", "A",
     "A default list argument is created once at definition and shared across all calls, so it accumulates.",
     "Each function call creates a fresh empty list for a default list argument."),
    ("bug", "fns = [lambda: i for i in range(3)]\n\nfns[0]() returns (A) 0 (B) 2. Answer: (", "B", "A",
     "Closures capture the loop variable by reference, so after the loop they all see its final value.",
     "Each lambda captures the current value of the loop variable at the moment it is created."),
    ("bug", "In CPython:  a = 257; b = 257;  the expression (a is b) evaluates to (A) True (B) False. "
     "Answer: (", "B", "A",
     "CPython caches small integers from -5 to 256; 257 is outside that range, so the two are distinct "
     "objects and 'is' is False.",
     "Equal integers are always the same object in Python, so 'is' returns True."),
    ("bug", "import time\ndef f(t=time.time()):\n    return t\n\nCalling f() at different times returns "
     "(A) a different value each call (B) the same value every call. Answer: (", "B", "A",
     "Default argument values are evaluated once when the function is defined, not on each call.",
     "Default arguments are re-evaluated on every call, so time.time() gives a fresh value each time."),
    ("fix", "Issue: a KeyError is raised when an expected config key is missing. The more targeted, "
     "robust fix is (A) use dict.get(key, default) (B) wrap a large block in try/except Exception. "
     "Answer: (", "A", "B",
     "Use dict.get with a default for an expected-missing key; reserve try/except for genuinely "
     "exceptional cases, since broad except hides other bugs.",
     "Wrapping the whole block in try/except Exception is the safest, most robust way to handle a "
     "missing key."),
    ("fix", "Issue: O(n^2) slowness caused by repeated list.index() calls inside a loop. The fix that "
     "addresses the root cause is (A) build a dict/set for O(1) lookups (B) add an lru_cache decorator "
     "to the function. Answer: (", "A", "B",
     "Repeated linear scans (list.index in a loop) are fixed by a dict/set giving O(1) lookup; caching "
     "does not help when the inputs are all distinct.",
     "Adding an lru_cache decorator removes the O(n^2) cost of the repeated list.index calls."),
]
SPEC_Q = "Is the integer 10 an even number? (A) yes (B) no. Answer: ("
SPEC_R, SPEC_W = "A", "B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--layer", type=int, default=26)
    ap.add_argument("--alpha", type=float, default=1.0)
    a = ap.parse_args()
    trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
    from transformers import AutoTokenizer
    model = load_frozen_lm(a.model); model.eval()
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=trust)
    inj = OperatorInjector(model, tok, a.layer, a.alpha)
    print(f"loaded | layer {a.layer} | alpha {a.alpha} | {len(ITEMS)} code items\n", flush=True)

    def tid(s): return tok(s, add_special_tokens=False).input_ids[-1]
    def belief(q, R, W, v=None):
        lg = inj.answer_logits(q, v)
        return float(lg[tid(R)] - lg[tid(W)])

    rows = []
    for tag, q, R, W, good, bad in ITEMS:
        cold = belief(q, R, W)
        op_good = belief(q, R, W, inj.combine([(good, "ASSERT")], q, normalize=True))
        op_bad = belief(q, R, W, inj.combine([(bad, "ASSERT")], q, normalize=True))   # WRONG insight control
        rag = belief(f"{good}\n{q}", R, W)
        v_spec = inj.combine([(good, "ASSERT")], SPEC_Q, normalize=True)
        spec_cold = belief(SPEC_Q, SPEC_R, SPEC_W)
        spec = belief(SPEC_Q, SPEC_R, SPEC_W, v_spec)
        fooled = cold <= 0
        good_helps = op_good > cold
        content_specific = op_good > op_bad           # KEY: right insight beats wrong insight
        beats_rag = op_good > rag
        spec_ok = (spec_cold <= 0) or (spec > 0)
        rows.append((tag, cold, op_good, op_bad, rag, fooled, good_helps, content_specific, beats_rag, spec_ok))
        print(f"[{tag}] cold {cold:+.2f}  OP(correct) {op_good:+.2f}  OP(WRONG) {op_bad:+.2f}  RAG {rag:+.2f}"
              f"   {'FOOLED' if fooled else 'ok-cold'}{'' if spec_ok else ' SPEC-BROKEN'}")
        print(f"     {q.splitlines()[0][:72]}", flush=True)

    import statistics as st
    n = len(rows)
    def cnt(i): return sum(1 for r in rows if r[i])
    def mean(i): return st.mean(r[i] for r in rows)
    print(f"\n========== IMPORTANT FINDINGS ==========")
    print(f"  items: {n}   (bug={sum(1 for r in rows if r[0]=='bug')} fix={sum(1 for r in rows if r[0]=='fix')})")
    print(f"  cold-FOOLED (4B wrong/unsure cold): {cnt(5)}/{n}   <- are these genuinely hard for the 4B?")
    print(f"  correct insight HELPS (OP>cold):    {cnt(6)}/{n}")
    print(f"  CONTENT-SPECIFIC (OP_correct > OP_WRONG): {cnt(7)}/{n}   <- KEY: operator carries the SPECIFIC")
    print(f"       insight, not generic steer (the wrong insight should push the WRONG way)")
    print(f"  OPERATOR beats RAG:                 {cnt(8)}/{n}")
    print(f"  specificity intact:                 {cnt(9)}/{n}")
    print(f"  mean belief: cold {mean(1):+.2f} | OP_correct {mean(2):+.2f} | OP_WRONG {mean(3):+.2f} | RAG {mean(4):+.2f}")
    print(f"\n  DIAGNOSE: if CONTENT-SPECIFIC is high and OP_WRONG <= cold, the graph edit teaches the 4B")
    print(f"  the SPECIFIC code fact (real learning). If OP_correct ~ OP_WRONG, it's generic steer (no learning).")


if __name__ == "__main__":
    main()
