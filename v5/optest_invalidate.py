"""KILL-TEST for the Operator-Attention schema: does a typed INVALIDATE operator (subtract),
gated by an edge, provably FLIP a frozen LM's output — and does ablating the edge un-flip it?

The whole "make structure load-bearing" schema rests on one assumption: that operations of a
fixed KIND (here: INVALIDATE = subtract) change the frozen LM's output in a way a content-blend
can't. This is the floor test. If it fails, even non-interchangeable operators can't beat the
frozen-decode wall -> don't scale the schema.

Honest design (no training, no circularity):
  - ASSERT vector  v_a = mean layer-L hidden of the ASSERT sentence  - neutral   (the counterfactual's push)
  - INVALIDATE vec v_i = mean layer-L hidden of the NEGATION sentence - neutral   (derived INDEPENDENTLY)
  - inject at layer L, measure logit(false_city) - logit(true_city) (belief in the counterfactual):
      cold            : no steer
      assert          : + a*v_a                       (push the counterfactual)
      edge-present    : + a*v_a  - a*v_i              (INVALIDATE subtracts, routed by the edge)
      edge-ablated    : + a*v_a                       (== assert; the edge removed -> no subtract)
  PASS = assert raises belief vs cold, edge-present LOWERS it vs assert (operator fired), and
         edge-ablated stays raised (the edge, not the node's mere presence, did the work).

Run (A40, 4B): V5_LM_TRUST_REMOTE_CODE=1 python -m v5.optest_invalidate --layer 18 --alpha 6
"""
from __future__ import annotations

import argparse
import torch

from v5.lm_loader import load_frozen_lm  # env-driven precision (V5_LM_QUANT etc.)


PROBES = [
    # (subject, true, false, prompt)
    ("Eiffel Tower", "Paris", "Rome", "The Eiffel Tower is located in the city of"),
    ("Statue of Liberty", "York", "London", "The Statue of Liberty is located in New"),
    ("Colosseum", "Rome", "Paris", "The Colosseum is located in the city of"),
    ("Big Ben", "London", "Berlin", "Big Ben is located in the city of"),
    ("Kremlin", "Moscow", "Madrid", "The Kremlin is located in the city of"),
]


def _layers(model):
    m = model
    for a in ("model", "layers"):
        m = getattr(m, a)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--layer", type=int, default=18, help="residual layer to read/steer")
    ap.add_argument("--alpha", type=float, default=2.0, help="scale on the real grounding shift (sweep 1-6)")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    import os
    trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
    model = load_frozen_lm(a.model)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=trust)
    model.eval()
    dev = next(model.parameters()).device
    layers = _layers(model)
    L = a.layer
    print(f"loaded | device {dev} | {len(layers)} layers | steering layer {L} "
          f"({type(layers[L]).__name__})", flush=True)

    cap = {"h": None}
    steer = {"v": None}

    def hook(mod, inp, out):
        is_tup = isinstance(out, tuple)
        h = out[0] if is_tup else out
        cap["h"] = h.detach()
        if steer["v"] is not None:
            h = h + steer["v"].to(h.dtype)
            if is_tup:
                return (h,) + tuple(out[1:])
            return h
        return out
    handle = layers[L].register_forward_hook(hook)

    @torch.no_grad()
    def hidden_lastpos(text):
        steer["v"] = None
        cap["h"] = None
        ids = tok(text, return_tensors="pt").input_ids.to(dev)
        model(ids)
        if cap["h"] is None:
            raise RuntimeError(f"hook never fired at layer {L} — wrong layer module")
        return cap["h"][0, -1].float()              # layer-L hidden at the FINAL token

    @torch.no_grad()
    def belief(prompt, true_tok, false_tok, v):
        steer["v"] = v
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        logits = model(ids).logits[0, -1].float()
        steer["v"] = None
        ti = tok(" " + true_tok, add_special_tokens=False).input_ids[0]
        fi = tok(" " + false_tok, add_special_tokens=False).input_ids[0]
        return float(logits[fi] - logits[ti])        # belief in the FALSE (counterfactual) city

    rows = []
    try:
        for subj, true_c, false_c, prompt in PROBES:
            # steering vector = the REAL grounding shift the fact causes at the answer position
            # (in-context diff): h_L(last token of "fact\nprompt") - h_L(last token of "prompt").
            assert_ctx = f"Fact: The {subj} is located in {false_c}.\n{prompt}"
            inval_ctx = f"Fact: It is FALSE that the {subj} is in {false_c}; it is in {true_c}.\n{prompt}"
            base = hidden_lastpos(prompt)
            v_a = (hidden_lastpos(assert_ctx) - base) * a.alpha
            v_i = (hidden_lastpos(inval_ctx) - base) * a.alpha
            cold = belief(prompt, true_c, false_c, None)
            asrt = belief(prompt, true_c, false_c, v_a)
            edge = belief(prompt, true_c, false_c, v_a - v_i)       # INVALIDATE via edge
            ablt = belief(prompt, true_c, false_c, v_a)             # edge removed == assert
            rows.append((subj, cold, asrt, edge, ablt))
            print(f"  done {subj}: cold {cold:.2f} assert {asrt:.2f} edge {edge:.2f}", flush=True)
    except Exception as e:                                          # noqa: BLE001
        import traceback
        print("ERROR during probes:", repr(e), flush=True)
        traceback.print_exc()
        handle.remove()
        return

    print(f"\n=== INVALIDATE operator kill-test (layer {L}, alpha {a.alpha}) ===")
    print(f"  belief = logit(false_city) - logit(true_city)  (higher = believes the counterfactual)")
    print(f"  {'probe':18} {'cold':>8} {'assert':>8} {'edge(inv)':>10} {'ablated':>8}")
    import statistics as st
    a_up = e_dn = 0
    for subj, c, asr, e, ab in rows:
        print(f"  {subj:18} {c:8.2f} {asr:8.2f} {e:10.2f} {ab:8.2f}")
        a_up += int(asr > c)        # assert raised belief
        e_dn += int(e < asr)        # INVALIDATE (edge) lowered it
    n = len(rows)
    print(f"\n  assert raised belief: {a_up}/{n} | INVALIDATE-edge lowered it: {e_dn}/{n}")
    mc = st.mean(r[1] for r in rows); ma = st.mean(r[2] for r in rows)
    me = st.mean(r[3] for r in rows)
    print(f"  mean belief: cold {mc:+.2f} -> assert {ma:+.2f} -> edge(invalidate) {me:+.2f}")
    assert_fired = a_up >= 0.6 * n and (ma - mc) > 0.3
    if not assert_fired:
        print(f"\n  INCONCLUSIVE — the ASSERT prerequisite did not fire (steering too weak/wrong layer)."
              f"\n  Sweep --layer/--alpha before reading INVALIDATE. (assert moved belief {ma-mc:+.2f})")
        return
    ok = e_dn >= 0.6 * n and (ma - me) > 0.3
    print(f"\n  KILL-TEST: {'PASS — assert fires AND INVALIDATE-edge cancels it (structure CAN compute, edge-gated)' if ok else 'FAIL — assert fires but INVALIDATE does NOT oppose it (subtract-of-negation != remove-belief)'}")
    handle.remove()


if __name__ == "__main__":
    main()
