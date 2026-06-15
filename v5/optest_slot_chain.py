"""SLOT multi-step — COMPUTE with a slotted value across hops (extends single-hop recall).

optest_slot_write proved single-hop: model writes its computed value to a slot, READS it back (8/8).
Multi-step needs more: at step 2 the model must COMPUTE with the slotted value (s2 = s1 + z), not just
recall it. step B never restates s1, so cold cannot know "the previous result".

  step 1: model generates  "Compute 2 plus 3. The result is" -> "5"   (write slot1 from its own output)
  step 2: "Now add 3 to the previous result. The new result is" + inject(slot1)  -> must output 8 (=5+3)

  cold : step 2 alone                         (can't — no s1)
  text : "The previous result is 5." + step 2 (ceiling — s1 in context)
  slot : inject slot1 (model's own s1, latent) + step 2   (compute s2 from the latent value)
PASS = slot ~ text >> cold  ->  the frozen LM computes with a value held in a latent register, no tokens.

  4B (L26, A40): V5_LM_TRUST_REMOTE_CODE=1 python -m v5.optest_slot_chain --layer 26 --alpha 1.0
"""
from __future__ import annotations

import argparse
import os
import re
import torch

from v5.lm_loader import load_frozen_lm
from v5.operator_injector import OperatorInjector

_S2 = "Now add {z} to the previous result. The new result is"
# (step1_compute, s1, z, s2, distractor)
CHAINS = [
    ("Compute 2 plus 3. The result is", "5", 3, "8", " 4"),
    ("Compute 4 plus 1. The result is", "5", 2, "7", " 3"),
    ("Compute 1 plus 2. The result is", "3", 4, "7", " 2"),
    ("Compute 3 plus 3. The result is", "6", 2, "8", " 5"),
    ("Compute 2 plus 2. The result is", "4", 5, "9", " 3"),
    ("Compute 5 minus 2. The result is", "3", 3, "6", " 9"),
    ("Compute 4 minus 1. The result is", "3", 1, "4", " 8"),
    ("Compute 6 minus 2. The result is", "4", 2, "6", " 1"),
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
    dev = next(model.parameters()).device
    print(f"loaded | layer {a.layer} | alpha {a.alpha} | {len(CHAINS)} 2-hop chains", flush=True)

    @torch.no_grad()
    def gen_answer(prompt):
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        out = model.generate(ids, max_new_tokens=3, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

    def belief(q, R, W, v=None):
        lg = inj.answer_logits(q, v)
        return float(lg[tok(R, add_special_tokens=False).input_ids[-1]]
                     - lg[tok(W, add_special_tokens=False).input_ids[-1]])

    rows = []
    s1_correct = 0
    for step1, s1, z, s2, distr in CHAINS:
        step2 = _S2.format(z=z)
        R, W = f" {s2}", distr
        gen = gen_answer(step1)
        gd = (re.findall(r"-?\d+", gen) or [""])[0]
        ok = (gd == s1); s1_correct += ok
        slot1 = inj.combine([(f"{step1} {gd}", "ASSERT")], step2, normalize=True)   # model's own s1, latent
        cold = belief(step2, R, W)
        text = belief(f"The previous result is {s1}. {step2}", R, W)
        slot = belief(step2, R, W, slot1)
        rows.append((cold, text, slot, ok))
        print(f"  s1='{gd}'({'ok' if ok else 'X'}) z={z} s2={s2}  cold {cold:+.2f}  text {text:+.2f}  slot {slot:+.2f}", flush=True)

    import statistics as st
    n = len(rows)
    def acc(i, sub=None):
        rs = [r for r in rows if sub is None or r[3] == sub]
        return f"{sum(1 for r in rs if r[i] > 0)}/{len(rs)}"
    mc, mt, ms = (st.mean(r[i] for r in rows) for i in range(3))
    print(f"\n=== SLOT multi-step: compute with a latent value (layer {a.layer}, alpha {a.alpha}) ===")
    print(f"  step-1 generation correct: {s1_correct}/{n}")
    print(f"  recall-of-s2 acc(belief>0): cold {acc(0)} | text {acc(1)} | slot {acc(2)}")
    print(f"  on s1-correct subset: text {acc(1, True)} | slot {acc(2, True)}")
    print(f"  mean: cold {mc:+.2f} | text {mt:+.2f} | slot {ms:+.2f}")
    ok = ms > mc + 0.5
    print(f"\n  RESULT: {'PASS — the frozen LM computes s2 from a value held in a latent slot' if ok else 'see numbers'}")


if __name__ == "__main__":
    main()
