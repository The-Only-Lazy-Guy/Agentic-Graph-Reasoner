"""Raw GENERATION check for the code-reasoning result — does belief translate to actual output?

code_reasoning_suite measured BELIEF (logit margin): OP_correct +3.52 vs OP_WRONG -5.27. Belief is a
proxy. This decodes the ACTUAL answer greedily and checks the model OUTPUTS the right letter:
  COLD          -> wrong (fooled)
  OP(correct)   -> right   (the operator made it reason correctly, in its real output)
  OP(WRONG)     -> wrong   (a wrong code belief makes it OUTPUT the wrong answer -> content-specific)
  RAG(correct)  -> right   (context; for contrast)
Short answer (a single A/B letter) so steered generation can't run away. LM frozen.

  4B (A40): V5_LM_TRUST_REMOTE_CODE=1 python -m v5.code_reasoning_gen --layer 26 --alpha 0.6
"""
from __future__ import annotations

import argparse
import contextlib
import os
import re
import torch

from v5.lm_loader import load_frozen_lm
from v5.operator_injector import OperatorInjector
from v5.code_reasoning_suite import ITEMS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--layer", type=int, default=26)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--ntok", type=int, default=4)
    a = ap.parse_args()
    trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
    from transformers import AutoTokenizer
    model = load_frozen_lm(a.model); model.eval()
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=trust)
    inj = OperatorInjector(model, tok, a.layer, a.alpha)
    dev = next(model.parameters()).device
    print(f"loaded | layer {a.layer} | alpha {a.alpha} | {len(ITEMS)} items | GREEDY generation\n", flush=True)

    @torch.no_grad()
    def gen(prompt, v=None):
        with (inj.inject(v) if v is not None else contextlib.nullcontext()):
            enc = tok(prompt, return_tensors="pt").to(dev)
            out = model.generate(**enc, max_new_tokens=a.ntok, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()

    def letter(txt):
        m = re.search(r"[AB]", txt)
        return m.group(0) if m else "?"

    rows = []
    for tag, q, R, W, good, bad in ITEMS:
        g_cold = gen(q)
        g_good = gen(q, inj.combine([(good, "ASSERT")], q, normalize=True))
        g_bad = gen(q, inj.combine([(bad, "ASSERT")], q, normalize=True))
        g_rag = gen(f"{good}\n{q}")
        lc, lg, lb, lr = letter(g_cold), letter(g_good), letter(g_bad), letter(g_rag)
        cold_wrong = lc != R
        learned = lg == R
        wrong_makes_wrong = lb == W           # the WRONG insight makes it OUTPUT the wrong answer
        rag_right = lr == R
        flip = cold_wrong and learned          # genuine fooled -> fixed in actual output
        rows.append((tag, cold_wrong, learned, wrong_makes_wrong, rag_right, flip))
        print(f"[{tag}] correct={R}  COLD->{lc}{'(wrong)' if cold_wrong else ''}  "
              f"OP(correct)->{lg}{'(RIGHT)' if learned else ''}  OP(WRONG)->{lb}{'(wrong)' if lb==W else ''}  RAG->{lr}")
        print(f"     {q.splitlines()[0][:66]}", flush=True)

    n = len(rows)
    def c(i): return sum(1 for r in rows if r[i])
    print(f"\n========== GENERATION FINDINGS ==========")
    print(f"  cold WRONG in output:                 {c(1)}/{n}")
    print(f"  OP(correct) -> RIGHT output:          {c(2)}/{n}")
    print(f"  genuine FLIP (cold wrong -> OP right): {c(5)}/{n}   <- learning in actual generation")
    print(f"  OP(WRONG) -> WRONG output:            {c(3)}/{n}   <- content-specific in output, both ways")
    print(f"  RAG(correct) -> RIGHT output:         {c(4)}/{n}")
    print(f"\n  DIAGNOSE: FLIP > 0 and OP(WRONG)->wrong means the operator changes the model's ACTUAL")
    print(f"  ANSWER with specific code knowledge -- belief translated to generation. If FLIP=0, hollow.")


if __name__ == "__main__":
    main()
