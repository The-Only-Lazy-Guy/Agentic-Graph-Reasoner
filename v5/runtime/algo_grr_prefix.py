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

from v5.runtime.algo_grr_nstm_sup import embed_chunks, group_split, load_pairs, prompt_for


class PrefixSlots(nn.Module):
    """issue chunks -> K slots -> K embeddings in the LM's input space."""

    def __init__(self, d_model: int, n_slots: int = 4, d_slot: int = 128):
        super().__init__()
        self.n_slots, self.d_slot = n_slots, d_slot
        self.ctx_in = nn.Linear(384, d_slot)
        # NO BIAS on the projection into embedding space. THIS WAS DESTROYING THE EXPERIMENT.
        # Measured at init: input chunk embeddings are well separated (across-instance cosine 0.0994),
        # but the prefix vectors coming OUT were collapsed at 0.9170 -- because ||bias|| = 2.27 against
        # a mean signal norm of 1.09, i.e. the bias was 2.09x the signal and every instance's prefix
        # was mostly the same vector. With bias removed: 0.5395.
        # If the prefix barely varies across instances, shuffling the issue CANNOT change much, so the
        # falsifier reads ~0 BY CONSTRUCTION rather than by finding. Every "instance-specificity is
        # zero" number from this module is contaminated by that.
        # This is the THIRD time a bias/collapse in a write path produced a false null here (SupNSTM
        # write() ran a GRU from zero slots: 0.9960; then the readout could bypass the slots entirely).
        self.to_emb = nn.Linear(d_slot, d_model, bias=False)
        # the FIXED control: K learned slot vectors with no issue input at all
        self.fixed = nn.Parameter(torch.randn(n_slots, d_slot) * 0.02)

    def forward(self, chunk_emb, fixed: bool = False):
        # NO tanh. It saturated and cost a second collapse on its own: 0.0994 -> 0.5630 before the
        # projection had even been applied. A plain linear map preserves the separation the encoder
        # produced; the LM sees the prefix through its own layer norms anyway.
        s = self.fixed if fixed else self.ctx_in(chunk_emb)
        return self.to_emb(s)                                   # [K, d_model]

    def separation(self, chunk_embs) -> float:
        """Across-instance cosine of the PREFIX VECTORS. Call it before trusting any falsifier from
        this module: if this is ~1.0 the prefix is a constant and the shuffle test cannot detect
        anything, however the rest of the pipeline behaves."""
        with torch.no_grad():
            P = torch.stack([self(torch.as_tensor(e)).flatten() for e in chunk_embs]).float()
        P = P / (P.norm(dim=1, keepdim=True) + 1e-9)
        S = (P @ P.T).cpu().numpy()
        import numpy as _np
        return float(_np.mean(S[~_np.eye(len(S), dtype=bool)]))


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
    ap.add_argument("--sweep", type=str, default="",
                    help="comma-separated slot counts, e.g. 1,4,16,64 -- tests whether "
                         "instance-specificity is CAPACITY limited")
    a = ap.parse_args()

    rows = load_pairs(a.n, seed=a.seed)
    tr, held = group_split(rows, seed=a.seed)   # by INSTANCE: row-splitting leaked 64% of issues
    print(f"{len(rows)} real single-line SWE-bench fixes | train {len(tr)} held {len(held)} | "
          f"K={a.slots} slots\n", flush=True)
    torch.manual_seed(a.seed)
    from v5.runtime.dcpd_latent import WhiteBox
    lm = WhiteBox(a.lm, quant="4bit")
    Etr = np.stack([embed_chunks(r["issue"], a.slots) for r in tr])
    Eh = np.stack([embed_chunks(r["issue"], a.slots) for r in held])

    # FALSIFIER, REBUILT. The previous version paired each held example with its MOST
    # ANTI-CORRELATED issue (np.argmin over the cosine matrix). That is not "a wrong issue" -- it is
    # the negation of the useful direction, and worse, its difficulty is a FUNCTION OF K: the
    # anti-correlation gap decayed 0.221 -> 0.045 from K=1 to K=64 by concentration of measure, so the
    # falsifier manufactured the very K-trend it was meant to test. Switching pairing rules alone moved
    # the same measurement 0.0239 -> 0.6461 (27x), which was the tell I missed.
    # Now: average over several RANDOM DERANGEMENTS (no fixed point), whose difficulty is K-invariant,
    # and report the spread.
    def derangements(nheld, reps, rng):
        outs = []
        for _ in range(reps):
            while True:
                perm = list(range(nheld))
                rng.shuffle(perm)
                if all(perm[i] != i for i in range(nheld)):
                    outs.append(perm)
                    break
        return outs

    def evaluate(mod, arm, shuffle=False, reps: int = 5, rand_prefix=False):
        """rand_prefix: feed a NORM-MATCHED RANDOM vector instead of any issue. Separates 'this
        channel is sensitive to perturbation' from 'this channel carries the issue' -- without it, a
        degraded shuffled arm proves only that the prefix matters, not that the ISSUE does."""
        rng = random.Random(4242)
        perms = derangements(len(held), reps, rng) if shuffle else [list(range(len(held)))]
        vals = []
        for perm in perms:
            tot = 0.0
            with torch.no_grad():
                for i, r in enumerate(held):
                    if rand_prefix:
                        src = torch.randn_like(torch.tensor(Eh[i]))
                        src = src / src.norm(dim=-1, keepdim=True) * float(
                            torch.tensor(Eh[i]).norm(dim=-1, keepdim=True).mean())
                        e = src.to(lm.device)
                    else:
                        e = torch.tensor(Eh[perm[i]], device=lm.device)
                    tot += float(loss_with_prefix(lm, mod, e,
                                                  prompt_for("trunc", r["issue"], r["old"]),
                                                  r["new"], arm))
            vals.append(tot / max(1, len(held)))
        return float(np.mean(vals)), float(np.std(vals))

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

    if a.sweep:
        ks = [int(x) for x in a.sweep.split(",") if x.strip()]
        base0 = PrefixSlots(lm.model.config.hidden_size, 1).to(lm.device)
        b0, _ = evaluate(base0, "trunc")
        print(f"  B truncated (no prefix, no training) held CE {b0:.4f}\n", flush=True)
        print(f"  {'K':>3} {'F fixed':>9} {'P issue':>9} {'S derange':>11} {'R rand-vec':>11} "
              f"{'P-vs-S (issue)':>15} {'P-vs-R':>9}")
        for k in ks:
            Etr = np.stack([embed_chunks(r["issue"], k) for r in tr])
            Eh = np.stack([embed_chunks(r["issue"], k) for r in held])
            a.slots = k
            # GUARD: if the prefix vectors are near-identical across instances the falsifier cannot
            # detect anything, whatever the rest of the pipeline does. Printed with every row so a
            # collapsed channel can never again be reported as 'no instance-specificity'.
            sep = PrefixSlots(lm.model.config.hidden_size, k).to('cpu').separation(Eh[:8])
            mfk = train("fixed"); fk, _ = evaluate(mfk, "fixed")
            mpk = train("prefix"); pk, _ = evaluate(mpk, "prefix")
            sk, sd = evaluate(mpk, "prefix", shuffle=True)
            rk, _ = evaluate(mpk, "prefix", rand_prefix=True)
            print(f"  {k:>3} {fk:9.4f} {pk:9.4f} {sk:8.4f}+-{sd:.3f} {rk:11.4f} "
                  f"{sk - pk:15.4f} {rk - pk:9.4f}   prefix-sep {sep:.3f}", flush=True)
        print("\n  P-vs-S is the issue-specific signal (random derangement, K-invariant difficulty).")
        print("  P-vs-R says only that SOME prefix beats a random one -- it is not memory.")
        return

    base = PrefixSlots(lm.model.config.hidden_size, a.slots).to(lm.device)
    b, _ = evaluate(base, "trunc")
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
