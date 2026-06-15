"""Validate OPERATOR ATTENTION (the combine), not just single operators: does typed combine with
edge-routing BEAT a generic content-blend on contradictory evidence?

The kill-tests proved single operators (INVALIDATE subtract). Operator Attention is the COMBINE:
nodes typed by op_kind, edges routing (INVALIDATES). The whole point vs plain attention (a weighted
SUM) is that it can SUBTRACT a flagged node, not just blend it in. Decisive test, same 2 nodes:

  scenario per reasoning trap: the graph supplies the TRAP content (ASSERT grounding the wrong answer)
  AND a failure-pattern (INVALIDATE "that answer is wrong"), wired INVALIDATE --INVALIDATES--> ASSERT.

  cold        : no graph
  BLEND       : + v_assert + v_invalidate         (plain attention: add both -> generic mush)
  OPERATOR    : + v_assert - v_invalidate          (edge routes INVALIDATE to SUPPRESS the assert)
  belief = logit(correct) - logit(trap).  PASS = OPERATOR > BLEND and OPERATOR > cold
  (the typed combine suppresses the trap; blending it in does not).

Local 1.5B (sweet layer ~14): python -m v5.optest_operator_attention --model Qwen/Qwen2.5-1.5B --layer 14
4B (sweet layer ~26):         python -m v5.optest_operator_attention --layer 26
"""
from __future__ import annotations

import argparse
import os
import statistics as st
import torch

from v5.lm_loader import load_frozen_lm

TRAPS = [
    ("A bat and a ball cost $1.10 total; the bat costs $1.00 more than the ball. "
     "Does the ball cost (A) $0.05 or (B) $0.10? Answer: (", "A", "B"),
    ("If 5 machines make 5 widgets in 5 minutes, 100 machines make 100 widgets in "
     "(A) 5 minutes (B) 100 minutes. Answer: (", "A", "B"),
    ("A lily patch doubles daily and covers the lake in 48 days. Half-covered on "
     "(A) day 47 (B) day 24. Answer: (", "A", "B"),
    ("A farmer has 17 sheep; all but 9 run away. Left: (A) 9 (B) 8. Answer: (", "A", "B"),
    ("How many months have 28 days? (A) all 12 (B) only 1. Answer: (", "A", "B"),
    ("Emily's father has 3 daughters: April, May, and (A) Emily (B) June. Answer: (", "A", "B"),
]


def _layers(model):
    m = model
    for a in ("model", "layers"):
        m = getattr(m, a)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--layer", type=int, default=26)
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
        # graph nodes: ASSERT grounds the trap W; INVALIDATE flags W wrong (INVALIDATES edge -> assert)
        v_assert = (hlast(f"The answer is ({W}).\n{prompt}") - hlast(prompt)) * a.alpha
        v_inval = (hlast(f"The answer ({W}) is WRONG.\n{prompt}") - hlast(prompt)) * a.alpha
        cold = belief(prompt, R, W, None)
        blend = belief(prompt, R, W, v_assert + v_inval)     # plain attention: add both
        oper = belief(prompt, R, W, v_assert - v_inval)      # OPERATOR: edge routes INVALIDATE to subtract
        rows.append((cold, blend, oper))
        print(f"  cold {cold:+.2f}  BLEND {blend:+.2f}  OPERATOR {oper:+.2f}", flush=True)

    n = len(rows)
    op_gt_blend = sum(1 for c, b, o in rows if o > b)
    op_gt_cold = sum(1 for c, b, o in rows if o > c)
    mc = st.mean(r[0] for r in rows); mb = st.mean(r[1] for r in rows); mo = st.mean(r[2] for r in rows)
    print(f"\n=== Operator-Attention combine test (layer {L}, alpha {a.alpha}) ===")
    print(f"  belief = logit(correct) - logit(trap)  (higher = avoids the trap)")
    print(f"  OPERATOR beats BLEND: {op_gt_blend}/{n} | OPERATOR beats cold: {op_gt_cold}/{n}")
    print(f"  mean belief: cold {mc:+.2f} | BLEND {mb:+.2f} | OPERATOR {mo:+.2f}")
    ok = op_gt_blend >= 0.7 * n and (mo - mb) > 0.5 and mo > mc
    print(f"\n  RESULT: {'PASS — typed combine (subtract via edge) BEATS generic blend (structure>content-sum)' if ok else 'FAIL — operator combine no better than blending'}")


if __name__ == "__main__":
    main()
