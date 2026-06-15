"""Operator test on REAL reasoning (not the city toy): does INVALIDATE (subtract the trap-belief)
make a frozen LM AVOID a reasoning trap -> answer correctly?

Cognitive-reflection traps: a small model reflexively picks the tempting-WRONG answer W over the
correct R. An INVALIDATE node ("the answer W is wrong") applies its operator = SUBTRACT the W-belief
direction in latent space (edge-gated). If suppressing the trap raises logit(R)-logit(W), the
operator is doing LOGIC plain attention/RAG can't: actively killing a wrong path, in the model's
own computation, reversibly.

  cold        : no steer                 (model leans to the trap W)
  assert-W    : + a*v_W                  (sanity: can be pushed further into the trap)
  INVALIDATE  : - a*v_W                  (suppress W -> should shift toward correct R)   <- the operator
  belief = logit(R) - logit(W)   (higher = more correct).  PASS = INVALIDATE raises belief vs cold.

Local: python -m v5.optest_reasoning --model Qwen/Qwen2.5-1.5B --layer 14 --alpha 4
"""
from __future__ import annotations

import argparse
import os
import statistics as st
import torch

from v5.lm_loader import load_frozen_lm

# (prompt ending at "Answer (", correct_letter, wrong_letter) — wrong = the reflexive trap
TRAPS = [
    ("A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
     "Does the ball cost (A) $0.05 or (B) $0.10? Answer: (", "A", "B"),
    ("If 5 machines take 5 minutes to make 5 widgets, how long do 100 machines take to make "
     "100 widgets? (A) 5 minutes (B) 100 minutes. Answer: (", "A", "B"),
    ("A lily patch doubles in size every day and covers the lake in 48 days. On which day is the "
     "lake half covered? (A) day 47 (B) day 24. Answer: (", "A", "B"),
    ("Emily's father has three daughters: April, May, and ___? (A) Emily (B) June. Answer: (", "A", "B"),
    ("If a red house is made of red bricks and a blue house of blue bricks, what is a greenhouse "
     "made of? (A) glass (B) green bricks. Answer: (", "A", "B"),
    ("A farmer has 17 sheep; all but 9 run away. How many are left? (A) 9 (B) 8. Answer: (", "A", "B"),
    ("How many months have 28 days? (A) all 12 (B) only 1. Answer: (", "A", "B"),
    ("Some months have 31 days, some have 30. How many have 28 days? (A) twelve (B) one. Answer: (",
     "A", "B"),
]


def _layers(model):
    m = model
    for a in ("model", "layers"):
        m = getattr(m, a)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")   # match optest_invalidate default
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--alpha", type=float, default=4.0)
    a = ap.parse_args()
    trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
    from transformers import AutoTokenizer
    model = load_frozen_lm(a.model); model.eval()
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=trust)
    dev = next(model.parameters()).device
    layers = _layers(model); L = a.layer
    print(f"loaded | {len(layers)} layers | layer {L}", flush=True)

    cap = {"h": None}; steer = {"v": None}

    def hook(mod, inp, out):
        is_tup = isinstance(out, tuple); h = out[0] if is_tup else out
        cap["h"] = h.detach()
        if steer["v"] is not None:
            h = h + steer["v"].to(h.dtype)
            return ((h,) + tuple(out[1:])) if is_tup else h
        return out
    layers[L].register_forward_hook(hook)

    @torch.no_grad()
    def hlast(text):
        steer["v"] = None; cap["h"] = None
        ids = tok(text, return_tensors="pt").input_ids.to(dev); model(ids)
        return cap["h"][0, -1].float()

    @torch.no_grad()
    def belief(prompt, R, W, v):
        steer["v"] = v
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        lg = model(ids).logits[0, -1].float(); steer["v"] = None
        ri = tok(R, add_special_tokens=False).input_ids[-1]
        wi = tok(W, add_special_tokens=False).input_ids[-1]
        return float(lg[ri] - lg[wi])

    rows = []
    for prompt, R, W in TRAPS:
        # the W-belief direction = effect of asserting the wrong answer in context
        w_ctx = f"The correct answer is ({W}).\n{prompt}"
        v_w = (hlast(w_ctx) - hlast(prompt)) * a.alpha
        cold = belief(prompt, R, W, None)
        asrt = belief(prompt, R, W, v_w)        # push INTO the trap (sanity)
        inv = belief(prompt, R, W, -v_w)        # INVALIDATE: suppress the trap
        rows.append((R, W, cold, asrt, inv))
        print(f"  {('correct=' + R):11} cold {cold:+.2f}  assert-W {asrt:+.2f}  INVALIDATE {inv:+.2f}", flush=True)

    n = len(rows)
    inv_up = sum(1 for r in rows if r[4] > r[2])      # invalidate raised correctness vs cold
    asr_dn = sum(1 for r in rows if r[3] < r[2])      # assert-W lowered it (sanity: trap deepened)
    mc = st.mean(r[2] for r in rows); mi = st.mean(r[4] for r in rows)
    correct_cold = sum(1 for r in rows if r[2] > 0)
    correct_inv = sum(1 for r in rows if r[4] > 0)
    print(f"\n=== reasoning-trap INVALIDATE test (layer {L}, alpha {a.alpha}) ===")
    print(f"  belief = logit(correct) - logit(trap)  (higher = avoids the trap)")
    print(f"  INVALIDATE raised correctness: {inv_up}/{n} | assert-W deepened trap: {asr_dn}/{n}")
    print(f"  mean belief: cold {mc:+.2f} -> INVALIDATE {mi:+.2f}  (delta {mi-mc:+.2f})")
    print(f"  answers CORRECT (belief>0): cold {correct_cold}/{n} -> INVALIDATE {correct_inv}/{n}")
    ok = inv_up >= 0.7 * n and (mi - mc) > 0.5
    print(f"\n  RESULT: {'PASS — INVALIDATE suppresses the wrong path -> the frozen LM reasons better (structure does LOGIC)' if ok else 'FAIL — suppressing the trap does not help (operator does not aid reasoning)'}")


if __name__ == "__main__":
    main()
