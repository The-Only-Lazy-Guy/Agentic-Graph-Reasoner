"""GATE kill-test v2 — fix the two confounds that hid GATE on the 4B.

v1 confounds: (1) famous CS rules -> the 4B already knows them cold, so blind ASSERT never corrupts;
(2) the conclusion vector was grounded against the QUERY, so it leaked the violate reasoning (not
truly blind). v2 fixes both:
  - NOVEL entities (glorp/zid/...) with the rule STATED in context -> no prior to fall back on; the
    model must apply the stated rule, and a blind injected conclusion can override it.
  - BLIND assert: build the conclusion vector against a NEUTRAL stub, not the query -> a context-free
    "the conclusion is true" push that ignores the precondition (what a sum/ASSERT actually is).

  cold   : no graph                         (apply the stated rule)
  assert : inject blind +v_C                 (push the conclusion regardless of precondition)
  gate   : inject g * v_C, g=P(precond|query)   (push only when the precondition holds)
PASS = on VIOLATE, blind ASSERT corrupts (pushes the conclusion when it shouldn't) and GATE doesn't.

  local 1.5B (L14): V5_LM_TRUST_REMOTE_CODE=1 python -m v5.optest_gate2 --model Qwen/Qwen2.5-1.5B --layer 14
  4B (L26, A40):    V5_LM_TRUST_REMOTE_CODE=1 python -m v5.optest_gate2 --layer 26 --alpha 1.0
"""
from __future__ import annotations

import argparse
import os
import torch

from v5.lm_loader import load_frozen_lm
from v5.operator_injector import OperatorInjector

# (rule, entity_state, kind, precondition, conclusion, correct, wrong)
_RULES = [
    ("A glorp is sealed only if its valve is closed.", "valve is closed", "valve is open",
     "the valve is closed", "the glorp is sealed", "glorp", "sealed"),
    ("A zid is active only if it is powered.", "is powered", "is unpowered",
     "it is powered", "the zid is active", "zid", "active"),
    ("A kemp is valid only if it is signed.", "is signed", "is unsigned",
     "it is signed", "the kemp is valid", "kemp", "valid"),
    ("A fyle passes only if it is complete.", "is complete", "is incomplete",
     "it is complete", "the fyle passes", "fyle", "passing"),
    ("A wabe is open only if its key is present.", "key is present", "key is missing",
     "the key is present", "the wabe is open", "wabe", "open"),
    ("A trop is ready only if it is charged.", "is charged", "is drained",
     "it is charged", "the trop is ready", "trop", "ready"),
]


def _probes():
    out = []
    for rule, sat, vio, precond, concl, ent, prop in _RULES:
        q = lambda st: f"{rule} This {ent} {st}. Is the {ent} {prop}? Answer:"
        out.append((q(sat), "satisfy", precond, concl, " yes", " no"))
        out.append((q(vio), "violate", precond, concl, " no", " yes"))
    return out


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
    probes = _probes()
    print(f"loaded | layer {a.layer} | alpha {a.alpha} | {len(probes)} probes ({len(_RULES)} novel rules x sat/vio)", flush=True)

    def tid(s): return tok(s, add_special_tokens=False).input_ids[-1]
    def belief(q, R, W, v=None):
        lg = inj.answer_logits(q, v)
        return float(lg[tid(R)] - lg[tid(W)])

    def gate_g(query, precond):
        p = f"{query.split(' Answer:')[0]}\nIs it true that {precond}? Answer:"
        lg = inj.answer_logits(p)
        two = torch.softmax(torch.tensor([float(lg[tid(' yes')]), float(lg[tid(' no')])]), 0)
        return float(two[0])

    rows = []
    for query, kind, precond, concl, R, W in probes:
        g = gate_g(query, precond)
        v_C = inj.combine([(concl, "ASSERT")], "Answer:", normalize=True)   # BLIND: neutral stub, not the query
        cold = belief(query, R, W)
        asrt = belief(query, R, W, v_C)
        gate = belief(query, R, W, g * v_C)
        rows.append((kind, cold, asrt, gate, g))
        print(f"  [{kind:7}] g={g:.2f}  cold {cold:+.2f}  assert {asrt:+.2f}  gate {gate:+.2f}", flush=True)

    import statistics as st
    def grp(k): return [r for r in rows if r[0] == k]
    def acc(rs, i): return sum(1 for r in rs if r[i] > 0)
    sat, vio = grp("satisfy"), grp("violate")
    print(f"\n=== GATE v2 (novel rules, blind assert) layer {a.layer} alpha {a.alpha} ===")
    print(f"  mean gate g: satisfy {st.mean(r[4] for r in sat):.2f} | violate {st.mean(r[4] for r in vio):.2f}  (want high|low)")
    for name, rs in (("SATISFY", sat), ("VIOLATE", vio)):
        print(f"  {name} ({len(rs)}): cold {acc(rs,1)}/{len(rs)} | assert {acc(rs,2)}/{len(rs)} | gate {acc(rs,3)}/{len(rs)}  "
              f"(mean cold {st.mean(r[1] for r in rs):+.2f} | assert {st.mean(r[2] for r in rs):+.2f} | gate {st.mean(r[3] for r in rs):+.2f})")
    gba = sum(1 for r in vio if r[3] > r[2])
    ok = gba >= 0.7 * len(vio) and acc(sat, 3) >= 0.7 * len(sat)
    print(f"\n  GATE beats ASSERT on VIOLATE: {gba}/{len(vio)}")
    print(f"  RESULT: {'PASS — blind ASSERT corrupts when the precondition fails; the multiplicative GATE does not' if ok else 'see numbers'}")


if __name__ == "__main__":
    main()
