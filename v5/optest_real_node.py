"""Real-node SIGN test: do the GROWN failure_pattern nodes work as INVALIDATE (subtract) or
ASSERT (add)? Decides the scale quality + whether the schema sign is right for this text shape.

The grown failure_patterns read as CORRECTIONS ("approach X is WRONG because Y"), not the bare
wrong-belief the validated operator subtracted. So we must EMPIRICALLY check the sign on real nodes
before scaling 12 docs. For each curated question (where a grown failure_pattern is relevant), the
correct answer = "the approach is wrong"; measure belief(correct - wrong) under:
  cold | ASSERT (+v_fp) | INVALIDATE (-v_fp).
If ASSERT helps & INVALIDATE hurts -> the grown text is a CORRECTION (should be ASSERT, or the
teacher must emit the bare wrong-approach for INVALIDATE to work) -> fix BEFORE scaling.

  python -m v5.optest_real_node --model Qwen/Qwen2.5-1.5B --layer 14
"""
from __future__ import annotations

import argparse
import json
import os
import torch

from v5.lm_loader import load_frozen_lm
from v5.operator_injector import OperatorInjector

# (keyword to find the grown failure_pattern, question, correct, wrong)
PROBES = [
    ("AM-GM", "To prove the LOWER bound sum 1/(a(1+b)) >= 3/(1+abc), bounding each denominator by "
     "AM-GM (1+b >= 2 sqrt(b)) is (A) wrong, it gives an upper bound (B) the correct key step. Answer: (",
     "A", "B"),
    ("Cauchy-Schwarz", "Using Cauchy-Schwarz to reduce the problem to proving sum a(1+b) <= 3(1+abc) "
     "is (A) doomed, a counterexample exists (B) a valid reduction. Answer: (", "A", "B"),
    ("Scaling", "Normalizing by scaling a->a*t to force abc=1 (A) breaks the inequality form (B) is a "
     "clean valid simplification. Answer: (", "A", "B"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--graph", default="graphs/grown_graph5.json")
    a = ap.parse_args()
    trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
    from transformers import AutoTokenizer
    model = load_frozen_lm(a.model); model.eval()
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=trust)
    inj = OperatorInjector(model, tok, a.layer, a.alpha)

    g = json.load(open(a.graph, encoding="utf-8"))
    nodes = g.get("nodes") or g
    if isinstance(nodes, dict):
        nodes = list(nodes.values())
    fps = [n for n in nodes if n.get("node_type") == "failure_pattern"]
    print(f"loaded | layer {a.layer} | {len(fps)} failure_pattern nodes", flush=True)

    def belief(q, R, W, v):
        lg = inj.answer_logits(q, v)
        return float(lg[tok(R, add_special_tokens=False).input_ids[-1]]
                     - lg[tok(W, add_special_tokens=False).input_ids[-1]])

    rows = []
    for kw, q, R, W in PROBES:
        fp = next((n for n in fps if kw.lower() in (n.get("text") or "").lower()), None)
        if fp is None:
            print(f"  [skip] no grown failure_pattern matches '{kw}'"); continue
        v_fp = inj.node_vector(fp["text"], q) * a.alpha       # the grown node's grounding vector
        cold = belief(q, R, W, None)
        asrt = belief(q, R, W, v_fp)                          # ASSERT: add it
        inval = belief(q, R, W, -v_fp)                        # INVALIDATE: subtract it
        rows.append((kw, cold, asrt, inval))
        print(f"  {kw:14} cold {cold:+.2f}  ASSERT(+) {asrt:+.2f}  INVALIDATE(-) {inval:+.2f}", flush=True)

    if not rows:
        print("no probes matched"); return
    import statistics as st
    n = len(rows)
    a_help = sum(1 for _, c, asr, iv in rows if asr > c)
    i_help = sum(1 for _, c, asr, iv in rows if iv > c)
    mc = st.mean(r[1] for r in rows); ma = st.mean(r[2] for r in rows); mi = st.mean(r[3] for r in rows)
    print(f"\n=== grown failure_pattern SIGN test (layer {a.layer}) ===")
    print(f"  belief = logit(correct=avoid-wrong-approach) - logit(wrong)")
    print(f"  ASSERT(+) helps vs cold: {a_help}/{n} | INVALIDATE(-) helps: {i_help}/{n}")
    print(f"  mean: cold {mc:+.2f} | ASSERT(+) {ma:+.2f} | INVALIDATE(-) {mi:+.2f}")
    if ma > mc and ma > mi:
        print("\n  -> the grown failure_patterns are CORRECTIONS: they work as ASSERT(+), NOT INVALIDATE(-).")
        print("     FIX before scaling: map failure_pattern->ASSERT, OR have the teacher emit the BARE")
        print("     wrong-approach (no 'because it is wrong') so INVALIDATE=subtract steers away from it.")
    elif mi > mc and mi > ma:
        print("\n  -> INVALIDATE(-) wins: the schema sign is right; scale as-is.")
    else:
        print("\n  -> inconclusive/noisy; sweep layer/alpha or curate clearer probes.")


if __name__ == "__main__":
    main()
