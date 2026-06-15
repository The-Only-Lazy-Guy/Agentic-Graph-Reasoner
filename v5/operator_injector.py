"""Operator-Attention RUNTIME — inject the op-signed combine of typed graph nodes at a late layer.

The locked schema (v5/operator_schema.py) gives each node an op_kind. This runtime APPLIES it:
each node's grounding vector enters the residual SIGNED by its operation, not blended:

    inject = Σ_ASSERT (+v_i)  +  Σ_INVALIDATE (−v_i)  +  Σ_TRANSFORM (+v_i)      (GATE: multiplicative, v2)

INVALIDATE = SUBTRACT is the proven operation (optest_invalidate fact-flip 5/5; optest_reasoning
trap-suppression 8/8 on the 4B @ L26). The runtime generalizes it to a COMBINE over many nodes.

Node vector = the node's in-context grounding shift at the answer position:
    v_i = h_L(last token of "<node_text>\n<query>") − h_L(last token of "<query>")
(faithful; the trained-projector text→vector map collapses to generic, so we use the real shift).

__main__ validates the COMBINE cleanly vs a plain blend: ASSERT(correct) + INVALIDATE(trap pattern).
OPERATOR (+v_R − v_trap) vs BLEND (+v_R + v_trap) — the only difference is the trap's sign, the
proven subtract. PASS = OPERATOR avoids the trap better than BLEND.

  local 1.5B (L14): python -m v5.operator_injector --model Qwen/Qwen2.5-1.5B --layer 14
  4B (L26):         python -m v5.operator_injector --layer 26
"""
from __future__ import annotations

import argparse
import contextlib
import os
from typing import List, Tuple

import torch

from v5.lm_loader import load_frozen_lm

SIGN = {"ASSERT": 1.0, "INVALIDATE": -1.0, "TRANSFORM": 1.0, "GATE": 0.0, "SLOT": 0.0}


def _layers(model):
    m = model
    for a in ("model", "layers"):
        m = getattr(m, a)
    return m


class OperatorInjector:
    """Hooks one late layer; injects the op-signed sum of node grounding vectors."""

    def __init__(self, model, tok, layer: int, alpha: float = 4.0):
        self.model, self.tok, self.L, self.alpha = model, tok, layer, alpha
        self.dev = next(model.parameters()).device
        self.layers = _layers(model)
        self._cap = {"h": None}
        self._steer = {"v": None}
        self.layers[layer].register_forward_hook(self._hook)

    def _hook(self, mod, inp, out):
        is_tup = isinstance(out, tuple)
        h = out[0] if is_tup else out
        self._cap["h"] = h.detach()
        if self._steer["v"] is not None:
            h = h + self._steer["v"].to(h.dtype)
            return ((h,) + tuple(out[1:])) if is_tup else h
        return out

    @torch.no_grad()
    def _hlast(self, text: str) -> torch.Tensor:
        self._steer["v"] = None; self._cap["h"] = None
        ids = self.tok(text, return_tensors="pt").input_ids.to(self.dev)
        self.model(ids)
        return self._cap["h"][0, -1].float()

    def node_vector(self, node_text: str, query: str) -> torch.Tensor:
        """the node's in-context grounding shift at the query's answer position."""
        return self._hlast(f"{node_text}\n{query}") - self._hlast(query)

    def combine(self, nodes: List[Tuple[str, str]], query: str, normalize: bool = False) -> torch.Tensor:
        """nodes = [(text, op_kind)] -> op-signed steering vector at layer L.

        normalize=True unit-scales each node's grounding shift before applying alpha, so the steering
        DEGREE is length-invariant (long grown nodes else over-steer a big model — alpha 4 on 4B/L26
        gave +-12 logit swings). With normalize, alpha is the steering magnitude in shift-norm units.
        """
        base = self._hlast(query)
        v = torch.zeros_like(base)
        for text, op in nodes:
            s = SIGN.get(op, 0.0)
            if s != 0.0:
                shift = self._hlast(f"{text}\n{query}") - base
                if normalize:                       # length- AND model-invariant: unit shift * ||base||
                    shift = shift / (shift.norm() + 1e-6) * base.norm()   # alpha now = fraction of residual
                v = v + s * self.alpha * shift
        return v

    @contextlib.contextmanager
    def inject(self, v: torch.Tensor):
        self._steer["v"] = v
        try:
            yield
        finally:
            self._steer["v"] = None

    @torch.no_grad()
    def answer_logits(self, query: str, v=None) -> torch.Tensor:
        with self.inject(v) if v is not None else contextlib.nullcontext():
            ids = self.tok(query, return_tensors="pt").input_ids.to(self.dev)
            return self.model(ids).logits[0, -1].float()


# ── validation: op-signed COMBINE vs plain blend ─────────────────────────────────
TRAPS = [
    ("A bat and ball cost $1.10; the bat is $1.00 more than the ball. Ball: (A) $0.05 (B) $0.10. Answer: (", "A", "B"),
    ("5 machines make 5 widgets in 5 min; 100 machines make 100 widgets in (A) 5 (B) 100 min. Answer: (", "A", "B"),
    ("A lily patch doubles daily, covers the lake in 48 days. Half-covered on (A) day 47 (B) day 24. Answer: (", "A", "B"),
    ("17 sheep, all but 9 run away. Left: (A) 9 (B) 8. Answer: (", "A", "B"),
    ("How many months have 28 days? (A) all 12 (B) only 1. Answer: (", "A", "B"),
    ("Emily's father has 3 daughters: April, May, and (A) Emily (B) June. Answer: (", "A", "B"),
]


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
    inj = OperatorInjector(model, tok, a.layer, a.alpha)
    print(f"loaded | layer {a.layer} | alpha {a.alpha}", flush=True)

    def belief(q, R, W, v):
        lg = inj.answer_logits(q, v)
        return float(lg[tok(R, add_special_tokens=False).input_ids[-1]]
                     - lg[tok(W, add_special_tokens=False).input_ids[-1]])

    rows = []
    for q, R, W in TRAPS:
        # graph nodes: ASSERT the correct answer + INVALIDATE the trap pattern (a failure_pattern node)
        nodes = [(f"The answer is ({R}).", "ASSERT"), (f"The answer is ({W}).", "INVALIDATE")]
        v_op = inj.combine(nodes, q)                                   # +v_R - v_W  (op-signed)
        v_bl = inj.combine([(t, "ASSERT") for t, _ in nodes], q)      # +v_R + v_W  (plain blend)
        cold = belief(q, R, W, None)
        oper = belief(q, R, W, v_op)
        blend = belief(q, R, W, v_bl)
        rows.append((cold, blend, oper))
        print(f"  cold {cold:+.2f}  BLEND {blend:+.2f}  OPERATOR {oper:+.2f}", flush=True)

    import statistics as st
    n = len(rows)
    og_b = sum(1 for c, b, o in rows if o > b); og_c = sum(1 for c, b, o in rows if o > c)
    mc, mb, mo = (st.mean(r[i] for r in rows) for i in (0, 1, 2))
    print(f"\n=== Operator-Attention RUNTIME combine (layer {a.layer}, alpha {a.alpha}) ===")
    print(f"  belief = logit(correct) - logit(trap)")
    print(f"  OPERATOR beats BLEND: {og_b}/{n} | OPERATOR beats cold: {og_c}/{n}")
    print(f"  mean: cold {mc:+.2f} | BLEND {mb:+.2f} | OPERATOR {mo:+.2f}")
    ok = og_b >= 0.7 * n and (mo - mb) > 0.5
    print(f"\n  RESULT: {'PASS — op-signed combine (INVALIDATE subtracts) beats plain blend' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
