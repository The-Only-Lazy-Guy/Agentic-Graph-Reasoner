"""algo_grr_prefix -- slots injected where ATTENTION can use them, instead of as a post-hoc logit bias.

WHY THE INTERFACE HAD TO CHANGE. The NSTM residual applies a rank-32 additive bias to the logits AFTER
the LM has computed its distribution. Measured ceiling of that channel, after fixing a broken write
path and forcing the readout through slot content:
    shuffling the issue cost 0.0374 CE, against 1.4940 CE that the same issue is worth IN THE PROMPT.
So it carried ~2.5% of the issue's value. An additive bias can re-weight tokens; it cannot make the
model REASON about the issue, because nothing in the forward pass ever attends to it.

Here the K slots are projected into the LM's embedding space and PREPENDED to the sequence. The model
attends over them exactly as it attends over prompt tokens, so the same 4x128 floats now enter the
computation rather than decorating its output. Gradient flows through the frozen LM into the
projection; no LM weights are trained.

ARMS, and the controls are the point:
    B  truncated       old line only, no prefix                  (the incumbent)
    P  prefix          issue -> slots -> K prefix embeddings     (the claim)
    F  fixed-prefix    K LEARNED embeddings, NOT issue-conditioned
                       -> isolates "a prefix helps" from "an ISSUE-CONDITIONED prefix helps".
                          This is the analogue of the bias-only arm that previously revealed the LM
                          hidden state was doing all the work.
    S  shuffled        P, but each example gets ANOTHER example's issue
                       -> the falsifier. Standing rule in this project after a 96.4% "memory" result
                          turned out to be a generic prior.

HONEST NOTE ON PRIOR PRESERVATION: unlike the zero-init logit residual, a prefix CANNOT be a no-op --
adding tokens changes the sequence the LM sees. Arm B is therefore the explicit incumbent to beat, and
arm F is what separates a generic prefix effect from genuine issue conditioning.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("HF_HOME", r"E:\cache\hf")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from v5.runtime.algo_grr_nstm_sup import embed_chunks, load_pairs, prompt_for


class PrefixSlots(nn.Module):
    """issue chunks -> K slots -> K embeddings in the LM's input space."""

    def __init__(self, d_model: int, n_slots: int = 4, d_slot: int = 128):
        super().__init__()
        self.n_slots, self.d_slot = n_slots, d_slot
        self.ctx_in = nn.Linear(384, d_slot)
        self.to_emb = nn.Linear(d_slot, d_model)
        # the FIXED control: K learned slot vectors with no issue input at all
        self.fixed = nn.Parameter(torch.randn(n_slots, d_slot) * 0.02)

    def forward(self, chunk_emb, fixed: bool = False):
        s = self.fixed if fixed else torch.tanh(self.ctx_in(chunk_emb))
        return self.to_emb(s)                                   # [K, d_model]


def loss_with_prefix(lm, mod, chunk_emb, prompt: str, target: str, arm: str):
    """Teacher-forced CE with the slots PREPENDED as embeddings. The LM is frozen; gradient reaches
    the projection through the LM's own activations, which is exactly what a logit bias never allowed."""
    tok = lm.tok
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, return_tensors="pt")
    ids = (ids if torch.is_tensor(ids) else ids["input_ids"]).to(lm.device)
    tgt = tok(target, return_tensors="pt", add_special_tokens=False).input_ids.to(lm.device)
    full = torch.cat([ids, tgt], 1)
    emb_layer = lm.model.get_input_embeddings()
    te = emb_layer(full)                                        # [1, T, d]
    if arm == "trunc":
        inp, off = te, 0
    else:
        pre = mod(chunk_emb, fixed=(arm == "fixed")).unsqueeze(0).to(te.dtype)   # [1, K, d]
        inp = torch.cat([pre, te], 1)
        off = pre.shape[1]
    out = lm.model(inputs_embeds=inp)
    lo = out.logits[0, off + ids.shape[1] - 1: off + full.shape[1] - 1, :].float()
    return F.cross_entropy(lo, tgt[0])


def main():
    ap = argparse.ArgumentParser(description="Slots as attended prefix embeddings.")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--lm", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rows = load_pairs(a.n, seed=a.seed)
    cut = int(len(rows) * 0.8)
    tr, held = rows[:cut], rows[cut:]
    print(f"{len(rows)} real single-line SWE-bench fixes | train {len(tr)} held {len(held)} | "
          f"K={a.slots} slots\n", flush=True)
    torch.manual_seed(a.seed)
    from v5.runtime.dcpd_latent import WhiteBox
    lm = WhiteBox(a.lm, quant="4bit")
    Etr = np.stack([embed_chunks(r["issue"], a.slots) for r in tr])
    Eh = np.stack([embed_chunks(r["issue"], a.slots) for r in held])

    def evaluate(mod, arm, shuffle=False):
        idx = list(range(len(held)))
        if shuffle:
            idx = idx[1:] + idx[:1]
        tot = 0.0
        with torch.no_grad():
            for i, r in enumerate(held):
                e = torch.tensor(Eh[idx[i]], device=lm.device)
                tot += float(loss_with_prefix(lm, mod, e, prompt_for("trunc", r["issue"], r["old"]),
                                              r["new"], arm))
        return tot / max(1, len(held))

    def train(arm):
        torch.manual_seed(a.seed)
        mod = PrefixSlots(lm.model.config.hidden_size, a.slots).to(lm.device)
        opt = torch.optim.Adam(mod.parameters(), lr=3e-4)
        for ep in range(a.epochs):
            run = 0.0
            for i, r in enumerate(tr):
                e = torch.tensor(Etr[i], device=lm.device)
                l = loss_with_prefix(lm, mod, e, prompt_for("trunc", r["issue"], r["old"]),
                                     r["new"], arm)
                opt.zero_grad(); l.backward()
                torch.nn.utils.clip_grad_norm_(mod.parameters(), 1.0)
                opt.step(); run += float(l)
            print(f"    [{arm}] epoch {ep+1} train CE {run/len(tr):.4f}", flush=True)
        return mod

    base = PrefixSlots(lm.model.config.hidden_size, a.slots).to(lm.device)
    b = evaluate(base, "trunc")
    print(f"  B truncated (no prefix)          held CE {b:.4f}\n", flush=True)

    mf = train("fixed")
    f = evaluate(mf, "fixed")
    print(f"  F fixed prefix (NOT issue-cond)  held CE {f:.4f}\n", flush=True)

    mp = train("prefix")
    p = evaluate(mp, "prefix")
    s = evaluate(mp, "prefix", shuffle=True)
    print(f"  P issue-conditioned prefix       held CE {p:.4f}")
    print(f"  S shuffled (WRONG issue)         held CE {s:.4f}")

    print(f"\n{'=' * 74}")
    print("SLOTS AS ATTENDED PREFIX (vs the logit-bias interface)")
    print(f"  B truncated            {b:.4f}")
    print(f"  F fixed prefix         {f:.4f}   (prefix alone buys {b - f:+.4f})")
    print(f"  P issue prefix         {p:.4f}   (issue-conditioning buys {f - p:+.4f} over F)")
    print(f"  S shuffled issue       {s:.4f}")
    print(f"\n  FALSIFIER instance-specific content: {s - p:+.4f} CE")
    print(f"    logit-bias interface managed        +0.0374")
    print(f"    the issue is worth IN THE PROMPT    +1.4940")
    print(f"    -> this channel carries {max(0.0, s - p) / 1.4940:.1%} of the issue's value")
    print(f"    -> {'GENUINE ISSUE MEMORY' if s - p > 0.05 else 'still not using the issue'}")
    print(f"{'=' * 74}")


if __name__ == "__main__":
    main()
