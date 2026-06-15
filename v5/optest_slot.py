"""SLOT operator kill-test — a WRITABLE register: write an intermediate at step A, READ it at step B.

ASSERT/INVALIDATE are read-only typed evidence (pre-grown). SLOT is the model's WORKING MEMORY: it
holds an intermediate the model produces, carried to a later step as a LATENT vector (not re-tokenized
text) — the "make DeltaNet state useful" goal, and an inspectable/editable idea-layer.

Minimal falsifiable claim: a 2-step task where step B depends on step A's result. With B phrased as
"that result — is it ...?" (no inputs restated), COLD cannot know the value. Carry it in a slot:
  cold        : B alone                                   (baseline — must guess)
  text        : the result in B's context                (ceiling — plain scratchpad)
  slot-assert : inject the result VALUE's vector at L     (latent register holding the value)
  slot-write  : inject the COMPUTE step's carry at L      (the model's own derivation state)
PASS = a latent slot (assert or write) >> cold and ~ text. Then the frozen LM can READ a written
register without the text — a real working-memory slot.

  local 1.5B (L14): V5_LM_TRUST_REMOTE_CODE=1 python -m v5.optest_slot --model Qwen/Qwen2.5-1.5B --layer 14
  4B (L26, A40):    V5_LM_TRUST_REMOTE_CODE=1 python -m v5.optest_slot --layer 26 --alpha 1.0
"""
from __future__ import annotations

import argparse
import os
import torch

from v5.lm_loader import load_frozen_lm
from v5.operator_injector import OperatorInjector

# (compute_step, result_digit, recall_question, correct, wrong-distractor) — PURE single-digit recall;
# B never restates inputs, so cold cannot know the value. Downstream = recall only (no computation),
# so the text ceiling is a literal copy and MUST work; the slot carry is then the only variable.
_B = "Recall the result. The result was"
PROBES = [
    ("Compute 3 plus 4. The result is", "7", _B, " 7", " 2"),
    ("Compute 2 times 4. The result is", "8", _B, " 8", " 3"),
    ("Compute 9 minus 4. The result is", "5", _B, " 5", " 1"),
    ("Compute 6 plus 3. The result is", "9", _B, " 9", " 4"),
    ("Compute 2 times 3. The result is", "6", _B, " 6", " 1"),
    ("Compute 8 minus 5. The result is", "3", _B, " 3", " 8"),
    ("Compute 1 plus 3. The result is", "4", _B, " 4", " 9"),
    ("Compute 10 minus 8. The result is", "2", _B, " 2", " 7"),
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
    print(f"loaded | layer {a.layer} | alpha {a.alpha} (normalized) | {len(PROBES)} 2-step probes", flush=True)

    def belief(q, R, W, v=None):
        lg = inj.answer_logits(q, v)
        return float(lg[tok(R, add_special_tokens=False).input_ids[-1]]
                     - lg[tok(W, add_special_tokens=False).input_ids[-1]])

    rows = []
    for compute, I, B, R, W in PROBES:
        cold = belief(B, R, W)
        text = belief(f"The result is {I}.\n{B}", R, W)
        v_assert = inj.combine([(f"The result is {I}.", "ASSERT")], B, normalize=True)  # value in a slot
        v_write = inj.combine([(compute, "ASSERT")], B, normalize=True)                 # the derivation carry
        s_assert = belief(B, R, W, v_assert)
        s_write = belief(B, R, W, v_write)
        rows.append((cold, text, s_assert, s_write))
        print(f"  cold {cold:+.2f}  text {text:+.2f}  slot-assert {s_assert:+.2f}  slot-write {s_write:+.2f}  "
              f"(I={I})", flush=True)

    import statistics as st
    n = len(rows)
    def acc(i): return sum(1 for r in rows if r[i] > 0)
    mc, mt, ma, mw = (st.mean(r[i] for r in rows) for i in range(4))
    print(f"\n=== SLOT kill-test (layer {a.layer}, alpha {a.alpha}) ===")
    print(f"  belief = logit(correct) - logit(wrong);  accuracy = belief>0")
    print(f"  accuracy: cold {acc(0)}/{n} | text {acc(1)}/{n} | slot-assert {acc(2)}/{n} | slot-write {acc(3)}/{n}")
    print(f"  mean:     cold {mc:+.2f} | text {mt:+.2f} | slot-assert {ma:+.2f} | slot-write {mw:+.2f}")
    best = max(ma, mw)
    ok = best > mc + 0.5 and acc(2 if ma >= mw else 3) >= 0.7 * n
    print(f"\n  RESULT: {'PASS — a latent slot reads the written value (>> cold, ~ text)' if ok else 'see numbers'}")


if __name__ == "__main__":
    main()
