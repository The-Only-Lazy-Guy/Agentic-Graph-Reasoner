"""GATE operator kill-test — a MULTIPLICATIVE precondition (the last untested op_kind).

ASSERT/INVALIDATE add/subtract; GATE MULTIPLIES: a rule's conclusion should apply ONLY when its
precondition holds. A flat ASSERT injects the conclusion blindly -> it corrupts the case where the
precondition is VIOLATED. GATE gates the conclusion by g = the LM's own belief that the precondition
holds in the query's context, so g~=0 when violated -> the conclusion is suppressed. A sum cannot
express "C only if P" -> GATE is non-interchangeable with ASSERT.

Each rule -> two probes: a SATISFY query (precondition holds) and a VIOLATE query (precondition fails).
  cold   : no graph
  assert : inject the conclusion vector blindly (+v_C)                 -> right on satisfy, WRONG on violate
  gate   : inject g * v_C, g = P(precondition true | query)           -> right on BOTH
PASS = on VIOLATE probes GATE beats ASSERT (suppresses the inapplicable rule), and SATISFY is preserved.

  local 1.5B (L14): V5_LM_TRUST_REMOTE_CODE=1 python -m v5.optest_gate --model Qwen/Qwen2.5-1.5B --layer 14
  4B (L26, A40):    V5_LM_TRUST_REMOTE_CODE=1 python -m v5.optest_gate --layer 26 --alpha 1.0
"""
from __future__ import annotations

import argparse
import os
import torch

from v5.lm_loader import load_frozen_lm
from v5.operator_injector import OperatorInjector

# (query, kind, precondition, conclusion, correct, wrong) ; kind in {satisfy, violate}
PROBES = [
    ("A weighted graph has all NON-NEGATIVE edge weights. Is Dijkstra's algorithm guaranteed to find "
     "the correct shortest path? Answer:", "satisfy",
     "all edge weights are non-negative", "Dijkstra's algorithm correctly finds the shortest path",
     " yes", " no"),
    ("A weighted graph contains a NEGATIVE edge weight. Is Dijkstra's algorithm guaranteed to find the "
     "correct shortest path? Answer:", "violate",
     "all edge weights are non-negative", "Dijkstra's algorithm correctly finds the shortest path",
     " no", " yes"),

    ("The array is SORTED. Will binary search correctly find the target element? Answer:", "satisfy",
     "the array is sorted", "binary search correctly finds the target element", " yes", " no"),
    ("The array is UNSORTED. Will binary search correctly find the target element? Answer:", "violate",
     "the array is sorted", "binary search correctly finds the target element", " no", " yes"),

    ("The limit has the indeterminate form 0/0. Does L'Hopital's rule apply? Answer:", "satisfy",
     "the limit is an indeterminate form 0/0 or infinity/infinity", "L'Hopital's rule applies",
     " yes", " no"),
    ("The limit has the determinate form 2/3. Does L'Hopital's rule apply? Answer:", "violate",
     "the limit is an indeterminate form 0/0 or infinity/infinity", "L'Hopital's rule applies",
     " no", " yes"),

    ("The coefficient matrix is INVERTIBLE. Does the linear system have a unique solution? Answer:",
     "satisfy", "the coefficient matrix is invertible", "the linear system has a unique solution",
     " yes", " no"),
    ("The coefficient matrix is SINGULAR (non-invertible). Does the linear system have a unique "
     "solution? Answer:", "violate", "the coefficient matrix is invertible",
     "the linear system has a unique solution", " no", " yes"),
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
    print(f"loaded | layer {a.layer} | alpha {a.alpha} | {len(PROBES)} probes (4 rules x satisfy/violate)", flush=True)

    def tid(s):  # last-token id of a short word
        return tok(s, add_special_tokens=False).input_ids[-1]

    def belief(q, R, W, v=None):
        lg = inj.answer_logits(q, v)
        return float(lg[tid(R)] - lg[tid(W)])

    def gate_g(query, precond):
        # g = P(precondition is true | query context), the LM's own check -> [0,1]
        p = f"{query.split(' Answer:')[0]}\nIs it true that {precond}? Answer:"
        lg = inj.answer_logits(p)
        two = torch.softmax(torch.tensor([float(lg[tid(' yes')]), float(lg[tid(' no')])]), 0)
        return float(two[0])

    rows = []
    for query, kind, precond, concl, R, W in PROBES:
        g = gate_g(query, precond)
        v_C = inj.combine([(concl, "ASSERT")], query, normalize=True)
        cold = belief(query, R, W)
        asrt = belief(query, R, W, v_C)
        gate = belief(query, R, W, g * v_C)
        rows.append((kind, cold, asrt, gate, g))
        print(f"  [{kind:7}] g={g:.2f}  cold {cold:+.2f}  assert {asrt:+.2f}  gate {gate:+.2f}", flush=True)

    import statistics as st
    def grp(kind): return [r for r in rows if r[0] == kind]
    def acc(rs, i): return sum(1 for r in rs if r[i] > 0)
    sat, vio = grp("satisfy"), grp("violate")
    print(f"\n=== GATE kill-test (layer {a.layer}, alpha {a.alpha}) ===")
    print(f"  mean gate g: satisfy {st.mean(r[4] for r in sat):.2f} | violate {st.mean(r[4] for r in vio):.2f}  (want high|low)")
    for name, rs in (("SATISFY", sat), ("VIOLATE", vio)):
        print(f"  {name} ({len(rs)}): cold {acc(rs,1)}/{len(rs)} | assert {acc(rs,2)}/{len(rs)} | "
              f"gate {acc(rs,3)}/{len(rs)}  (mean assert {st.mean(r[2] for r in rs):+.2f} | gate {st.mean(r[3] for r in rs):+.2f})")
    gate_beats_assert_violate = sum(1 for r in vio if r[3] > r[2])
    ok = gate_beats_assert_violate >= 0.7 * len(vio) and acc(sat, 3) >= 0.7 * len(sat)
    print(f"\n  GATE beats ASSERT on VIOLATE: {gate_beats_assert_violate}/{len(vio)}")
    print(f"  RESULT: {'PASS — multiplicative gate suppresses the rule when its precondition fails (ASSERT cannot)' if ok else 'see numbers'}")


if __name__ == "__main__":
    main()
