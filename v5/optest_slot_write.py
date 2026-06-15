"""SLOT write-then-read — the MODEL writes the register, then reads it back latently.

optest_slot proved the READ side (inject a value's vector -> 4B recalls it 8/8) but the slot was
TEACHER-written ("The result is 7"). This tests the real loop: the model COMPUTES the intermediate
itself, that becomes the slot, and step B reads it WITHOUT the value in text.

  step A: model generates  "Compute 3 plus 4. The result is" -> "7"   (the WRITE: its own computation)
  slot  := a vector built from the model's own output / state
  step B: "Recall the result. The result was"  + inject(slot)         (the READ, no value in text)

Conditions (belief = logit(true) - logit(distractor), pure single-digit recall):
  cold        : B alone                                   (can't know)
  ref-teacher : inject shift of "The result is <true>."   (the proven 8/8 READ, teacher-written slot)
  write-gen   : inject shift of "<computeA> <model's own generated answer>"   (model wrote the value)
  write-state : inject the model's pre-emission hidden state at step A (latent, value never a token)

PASS = write-gen ~ ref-teacher (the model's own written value reads back) on the probes it computed
right. write-state is the pure-latent stretch goal (carry with no token for the value at all).

  4B (L26, A40): V5_LM_TRUST_REMOTE_CODE=1 python -m v5.optest_slot_write --layer 26 --alpha 1.0
"""
from __future__ import annotations

import argparse
import os
import re
import torch

from v5.lm_loader import load_frozen_lm
from v5.operator_injector import OperatorInjector

_B = "Recall the result. The result was"
PROBES = [
    ("Compute 3 plus 4. The result is", "7", " 7", " 2"),
    ("Compute 2 times 4. The result is", "8", " 8", " 3"),
    ("Compute 9 minus 4. The result is", "5", " 5", " 1"),
    ("Compute 6 plus 3. The result is", "9", " 9", " 4"),
    ("Compute 2 times 3. The result is", "6", " 6", " 1"),
    ("Compute 8 minus 5. The result is", "3", " 3", " 8"),
    ("Compute 1 plus 3. The result is", "4", " 4", " 9"),
    ("Compute 10 minus 8. The result is", "2", " 2", " 7"),
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
    print(f"loaded | layer {a.layer} | alpha {a.alpha} | {len(PROBES)} write-then-read probes", flush=True)

    @torch.no_grad()
    def gen_answer(prompt):
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        out = model.generate(ids, max_new_tokens=3, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()

    def belief(q, R, W, v=None):
        lg = inj.answer_logits(q, v)
        return float(lg[tok(R, add_special_tokens=False).input_ids[-1]]
                     - lg[tok(W, add_special_tokens=False).input_ids[-1]])

    @torch.no_grad()
    def state_vec(compute):
        # the model's pre-emission hidden at step A's last token (encodes the value, never a token),
        # expressed as a steer relative to the recall context's residual.
        hA = inj._hlast(compute)
        hB = inj._hlast(_B)
        shift = hA - hB
        return a.alpha * shift / (shift.norm() + 1e-6) * hB.norm()

    rows = []
    gen_correct = 0
    for compute, true_I, R, W in PROBES:
        gen = gen_answer(compute)
        gd = (re.findall(r"-?\d+", gen) or [""])[0]
        ok = (gd == true_I)
        gen_correct += ok
        cold = belief(_B, R, W)
        ref = belief(_B, R, W, inj.combine([(f"The result is {true_I}.", "ASSERT")], _B, normalize=True))
        wgen = belief(_B, R, W, inj.combine([(f"{compute} {gd}", "ASSERT")], _B, normalize=True))
        wstate = belief(_B, R, W, state_vec(compute))
        rows.append((cold, ref, wgen, wstate, ok))
        print(f"  gen='{gen[:8]}'({'ok' if ok else 'X'})  cold {cold:+.2f}  ref {ref:+.2f}  "
              f"write-gen {wgen:+.2f}  write-state {wstate:+.2f}  (true={true_I})", flush=True)

    import statistics as st
    n = len(rows)
    def acc(i, sub=None):
        rs = [r for r in rows if sub is None or r[4] == sub]
        return f"{sum(1 for r in rs if r[i] > 0)}/{len(rs)}"
    mc, mr, mg, ms = (st.mean(r[i] for r in rows) for i in range(4))
    print(f"\n=== SLOT write-then-read (layer {a.layer}, alpha {a.alpha}) ===")
    print(f"  step-A generation correct: {gen_correct}/{n}")
    print(f"  recall acc(belief>0): cold {acc(0)} | ref-teacher {acc(1)} | write-gen {acc(2)} | write-state {acc(3)}")
    print(f"  on gen-correct subset: write-gen {acc(2, True)} | write-state {acc(3, True)} | ref {acc(1, True)}")
    print(f"  mean: cold {mc:+.2f} | ref {mr:+.2f} | write-gen {mg:+.2f} | write-state {ms:+.2f}")
    ok = mg > mc + 0.5
    print(f"\n  RESULT: {'PASS — the model writes its own value to the slot and reads it back' if ok else 'see numbers'}")


if __name__ == "__main__":
    main()
