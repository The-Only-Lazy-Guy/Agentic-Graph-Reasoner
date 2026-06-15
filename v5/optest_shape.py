"""CONTENT-SHAPE discriminator: does INVALIDATE=subtract need a BARE wrong-belief, not a CORRECTION?

The 4B real-loop (operator_loop) showed the operator does NOT beat RAG-blend on the GROWN
failure_pattern nodes, because those read as CORRECTIONS ("approach X is wrong because Y") — a strong
model ASSERTS the correction toward the right answer, so subtracting hurts. Hypothesis: the SAME
operator beats blend if the node grounds the BARE WRONG BELIEF (the misconception stated as if true),
because then +blend pulls toward the trap and only −subtract steers away.

For each probe, compare two node SHAPES under blend(+) vs operator(-):
  CORRECTION (grown text)          : "Applying AM-GM ... gives an upper bound, the WRONG direction."
  BARE misconception (asserted)    : "By AM-GM ... so the lower bound follows."   (wrong, stated as valid)
The decisive read: with BARE nodes, OPERATOR(-) should beat BLEND(+); with CORRECTION nodes it won't.
If so -> fix the grower to emit bare misconceptions (operator content), don't map failure_pattern->ASSERT.

  local 1.5B (L14): V5_LM_TRUST_REMOTE_CODE=1 python -m v5.optest_shape --model Qwen/Qwen2.5-1.5B --layer 14
  4B (L26, A40):    V5_LM_TRUST_REMOTE_CODE=1 python -m v5.optest_shape --layer 26 --alpha 1.0
"""
from __future__ import annotations

import argparse
import os
import torch

from v5.lm_loader import load_frozen_lm
from v5.operator_injector import OperatorInjector

# (query, correct, wrong, CORRECTION node, BARE-misconception node)
PROBES = [
    ("To prove the LOWER bound for Sum 1/(a(1+b)) >= 3/(1+abc), bounding each denominator by AM-GM "
     "(1+b >= 2 sqrt b) is (A) wrong, it gives an upper bound (B) the correct key step. Answer: (", "A", "B",
     "Applying AM-GM 1+b >= 2 sqrt b to each denominator gives an UPPER bound on 1/(a(1+b)), the wrong "
     "direction for proving a lower bound.",
     "By AM-GM, 1+b >= 2 sqrt b, so 1/(a(1+b)) >= 1/(2 a sqrt b); summing the three terms yields the "
     "lower bound 3/(1+abc), which completes the proof."),

    ("Reducing Sum 1/(a(1+b)) >= 3/(1+abc) via Cauchy-Schwarz to proving Sum a(1+b) <= 3(1+abc) is "
     "(A) doomed, a counterexample exists (B) a valid reduction. Answer: (", "A", "B",
     "Using Cauchy-Schwarz to reduce the problem to Sum a(1+b) <= 3(1+abc) FAILS: a counterexample "
     "a=2,b=2,c=0.25 gives LHS 9.25 > RHS 6.",
     "By Cauchy-Schwarz, Sum 1/(a(1+b)) >= 9 / Sum a(1+b), and since Sum a(1+b) <= 3(1+abc), the "
     "desired bound 3/(1+abc) follows directly."),

    ("Two triangles that share the same incenter must have the same incircle and inradius: "
     "(A) false, the inradius can differ (B) true. Answer: (", "A", "B",
     "Assuming two triangles with the same incenter must have the same incircle is WRONG; the inradius "
     "can differ.",
     "Since triangles ABC and EFC share the same incenter, they have the same incircle, so their "
     "inradii are equal and EF = AB."),

    ("When counting triples whose product is divisible by 10, letting ONE entry satisfy both the "
     "factor-5 and the even requirement is (A) wrong, 5 is not even (B) fine. Answer: (", "A", "B",
     "It is WRONG to let the same entry satisfy both the factor-5 and the even requirement, because 5 "
     "is not even.",
     "A single entry equal to 5 satisfies both the factor-5 requirement and the even requirement, so we "
     "count it once for both."),

    ("Accepting a candidate p just because p is prime, without checking its remainders modulo all "
     "smaller primes, is (A) a mistake (B) sufficient. Answer: (", "A", "B",
     "Accepting p merely because p is prime, without checking remainders modulo all smaller primes, is "
     "a common MISTAKE.",
     "Since p is prime, p already satisfies the condition, so we accept p without checking any "
     "remainders modulo smaller primes."),

    ("Arguing that the partial sums stay below 1 from the rapid growth of the denominators alone is "
     "(A) insufficient, not a proof (B) a complete proof. Answer: (", "A", "B",
     "Arguing from the rapid growth of the denominators alone does NOT prove every partial sum is below "
     "1; it is insufficient.",
     "Because the denominators grow very rapidly, every partial sum is clearly below 1, which proves "
     "the bound."),

    ("In modular arithmetic, canceling a common factor from both sides is valid (A) only if that "
     "factor is invertible modulo m (B) always. Answer: (", "A", "B",
     "Canceling a common factor in modular arithmetic is INVALID unless that factor is invertible "
     "modulo m.",
     "We can cancel the common factor from both sides of the congruence, so the simplified equation "
     "holds and the argument goes through."),
]


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
    print(f"loaded | layer {a.layer} | alpha {a.alpha} (normalized) | {len(PROBES)} probes", flush=True)

    def belief(q, R, W, v):
        lg = inj.answer_logits(q, v)
        return float(lg[tok(R, add_special_tokens=False).input_ids[-1]]
                     - lg[tok(W, add_special_tokens=False).input_ids[-1]])

    def run(node_text):
        bl_wins = op_wins = 0
        d_bl = d_op = 0.0
        for q, R, W, corr, bare in PROBES:
            text = corr if node_text == "corr" else bare
            cold = belief(q, R, W, None)
            blend = belief(q, R, W, inj.combine([(text, "ASSERT")], q, normalize=True))
            oper = belief(q, R, W, inj.combine([(text, "INVALIDATE")], q, normalize=True))
            op_wins += oper > blend
            d_bl += blend - cold; d_op += oper - cold
        return op_wins, d_bl / len(PROBES), d_op / len(PROBES)

    print("\nshape         OPERATOR>BLEND   mean dBLEND   mean dOPERATOR")
    for shape, label in (("corr", "CORRECTION"), ("bare", "BARE-miscon")):
        ow, dbl, dop = run(shape)
        print(f"  {label:12} {ow}/{len(PROBES)}            {dbl:+.2f}         {dop:+.2f}", flush=True)
    print("\n  decisive: if BARE makes OPERATOR>BLEND (and dOPERATOR>0) while CORRECTION does not,")
    print("  the fix is content SHAPE -> grower must emit bare misconceptions for INVALIDATE.")


if __name__ == "__main__":
    main()
