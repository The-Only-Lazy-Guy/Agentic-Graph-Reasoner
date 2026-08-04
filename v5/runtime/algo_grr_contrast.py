"""algo_grr_contrast -- THE POSITIVE CONTROL, and the session handoff document.

=====================================================================================
THE RESULT (run this file to reproduce; numbers below are n=900, seed 0, held 167)
=====================================================================================
Attach an auxiliary network to a FROZEN LM and train it with plain next-token
cross-entropy, and it converges to an INSTANCE-INDEPENDENT ADAPTER -- regardless of
where you inject it. Measured across three structurally different interfaces:

    interface                              adapter gain   instance-specific   ratio
    logit bias (rank-32 over vocab)           ~1.66 CE      0.0045-0.0075     ~250x
    prefix embeddings (K prepended vectors)   ~2.1  CE      0.0049            ~430x
    hidden-state adapter (read h, write dh)    1.0499 CE    0.0011            ~950x

Three unrelated injection points -- after the logits, before the embeddings, inside
the residual stream -- all converging on the same solution. Not three architecture
failures; ONE optimization phenomenon.

THE POSITIVE CONTROL (this file) is what makes that a claim rather than an absence.
Same MiniLM issue embedding, same linear-encoder shape, same instance split, same
derangement falsifier. ONLY THE LOSS CHANGES:

    chance                          0.1250
    F no-issue (constant query)     0.1377   <- at chance, as required
    R shuffled issue (derangement)  0.1257   <- falls to chance, as required
    P issue query                   0.4731   95% CI [0.4012, 0.5509]
    query separation                0.0238   (queries vary; no collapse)

    instance-specific signal = 0.4731 - 0.1257 = 0.3474
    versus 0.0011-0.0075 for the SAME information under CE.  ~50-300x more.

=> The information IS present in the representation. The encoder was never the
   bottleneck. The apparatus CAN detect memory when it exists -- which is exactly what
   three consecutive nulls needed in order to mean anything. THE FAILURE IS THE
   SUPERVISION SIGNAL: under CE a generic prior helps EVERY example while memory helps
   ONE, so gradient descent takes the prior and never needs instance identity. Under a
   loss where a prior scores at chance BY CONSTRUCTION, the same encoder finds that
   information immediately.

Corroborating: more DATA made specificity WORSE at fixed capacity (prefix K=1:
0.646 -> 0.077 going n=300 -> n=900), because more data makes the generic prior easier
to fit. Capacity and data act through one channel: how easy is it to fit a solution
that ignores the instance.

=====================================================================================
BUGS FOUND THIS SESSION. Every one produced a WRONG RESULT that was reported before it
was caught. Read this before trusting any measurement in this repo.
=====================================================================================
MEASUREMENT-APPARATUS BUGS (the dangerous class -- they fake nulls AND wins):
 1. write() ran a GRU from ZERO slots, so its bias dominated -> across-instance slot
    cosine 0.9960 while the inputs were 0.1547. Channel dead before training started.
    FALSE NULL.
 2. prefix to_emb carried a BIAS 2.09x the signal -> prefix cosine 0.9170 (inputs
    0.0994); tanh saturation cost a further 0.0994 -> 0.5630 on its own. FALSE NULL.
    Fixing both also dropped the fixed-arm train CE 3.8705 -> 1.6604, so the collapse
    was crippling TRAINING as well as the measurement.
 3. the readout could BYPASS the slots via h, so the module ignored the very thing
    under test. Blocking it moved the falsifier 0.0001 -> 0.0374.
 4. falsifier used np.argmin(cosine) = the most ANTI-CORRELATED partner, whose
    difficulty DECAYS WITH K (gap 0.221 -> 0.045). It manufactured an entire K-trend.
    Changing only the pairing rule moved the same number 0.0239 -> 0.6461 (27x).
    ALWAYS use a random derangement.
 5. test contamination: one row per HUNK with the same problem_statement copied, split
    by ROW -> 64% of held issues also appeared in train, 14% were exact triples.
    Split BY INSTANCE and dedupe on (issue, old, new).
 6. degenerate bootstrap: np.random.RandomState(1) constructed INSIDE the comprehension
    -> all 4000 resamples identical -> CI width 0.000000 -> the script printed
    "genuinely USES the hidden trajectory" for an effect of +0.0011.
 7. endpoint-only separation guard measured only Z[-1]; the real check is the whole
    TRAJECTORY at matched relative positions (it read 1.0000 while the tracked arm
    still beat the fixed arm by 0.0701, which is impossible for a constant state).
 8. patches that silently did not apply (globals() vs closure locals; unasserted
    anchors). ALWAYS assert the anchor and verify the symbol exists afterwards.

PIPELINE BUGS (wrong numbers, less subtle):
 9.  train() did ONE opt.step() per EPOCH -> 6 gradient updates for a whole run.
     Everything blamed on "not enough data" was near-zero optimization.
 10. reward paid +0.5 for patching ANY file but only +0.4 for the RIGHT one, so
     "open something, patch it" strictly dominated the actual objective.
 11. arg_fn truncated the candidate list to 24 BEFORE appending tool-discovered paths,
     so grep/locate results were dropped in ~96% of real issues.
 12. action parser used a non-greedy regex -> truncated code at the first inner ')' ->
     12 of 12 authored tools rejected. Looked like an LM failure; was a parser bug.
 13. the prompt advertised author(<python>) and the 3B copied the placeholder
     literally, so every action parsed to the string "<python>".
 14. SWE verifier: wrong runner (django ships no pytest), MISSING test_patch (the
     FAIL_TO_PASS tests do not EXIST without it, so even the GOLD patch failed),
     pass-marker scanned only the last line (git-apply stderr lands there), missing
     pipefail, cp1252 decode crash on real test output.
 15. prefix vectors were OUT OF DISTRIBUTION: nearest REAL token cosine 0.0881
     (max 0.1022) against 0.5797 for a real token's nearest neighbour -- with norms
     matched to 1.0x. Norm-matching does NOT make a vector in-distribution.

=====================================================================================
FILE MAP
=====================================================================================
  algo_grr_contrast.py   THIS FILE. Positive control; contrastive loss, 3 arms.
  algo_grr_state.py      Reasoning-state tracker (read h -> GRU -> residual back into
                         h). In-distribution by construction. Has trajectory
                         separation + the h-PERMUTATION control (feed another
                         example's hidden trajectory). Best CE gain: 1.0499.
  algo_grr_prefix.py     Prefix embeddings + separation() guard + derangement arm.
  algo_grr_nstm_sup.py   Supervised NSTM; load_pairs / group_split / embed_chunks.
  algo_grr_instr.py      Self-written instructions banked in the AtomGraph, with
                         validate_bank() = VERIFIER-GATED GRAPH EDITING: prunes
                         negative-utility entries. The only thing in this project that
                         EDITS the graph rather than only appending to it.
  algo_grr_toolsmith.py  LM authors executable repair tools; sandboxed, gated, banked.
  algo_grr_swetools.py   REAL SWE-bench tools + Docker verifier (--calibrate).
  algo_grr_thinkctl.py   Domain-general tool-use controller (SWE + HotpotQA).
  membrane.py            Main entry point; --prefix-slots wires the adapter with its
                         limits documented above demo_prefix().
  scripts/prep_gold_hunks.py, mine_verified_schemas.py, prep_swe_tests.py
                         Dataset prep. ALL torch-free on purpose: pyarrow after torch
                         SEGFAULTS SILENTLY here (exits with no traceback at all).

=====================================================================================
WHAT IS ACTUALLY REAL (survived its controls)
=====================================================================================
  * 1 verified SWE-bench fix, agent-chosen, 2 steps (author -> test), real Docker test
    suite, 4.24 GB, Qwen2.5-3B 4-bit. Verifier calibrated 3/3 in BOTH directions
    (no-op edit FAILS, real gold patch PASSES): algo_grr_swetools.py --calibrate
  * 22.7% of 9152 real SWE-bench hunks are one line replaced by one line.
  * Instruction pruning by MEASURED utility took the retrieval signal 0.0339 -> 0.1665
    (5x). A random draw from the top-5 BEATS rank-1, so cosine ORDERING is misleading
    while cosine MEMBERSHIP is informative. CI still includes 0 at n=75; needs n>=300.
  * Trained adapters DO cut prompt cost: prefix K=16 held CE 3.2299 -> ~1.13.
  * This file's positive control.

WHAT IS NOT REAL (do not rebuild without new evidence)
  * latent MEMORY of the instance under CE -- 8 attempts, 3 interfaces, all ~0.
  * thinker as a tool/string selector -- loses to plain grep (grep reaches gold 0.786),
    and end-to-end it DESTROYED the one solve the hand-driven pipeline gets.
  * self-authored TOOL transfer -- 0 replays / 40. Literals cannot generalise, and a 3B
    cannot write generalising regexes (11/20 compiled but matched nothing).
  * schema-based repair -- VERIFIED coverage 4/9152 = 0.04%. A hand-labelled miner had
    claimed 16.5%; its precision on one class measured 13%.

=====================================================================================
STANDING METHODOLOGY (earned the hard way, all of it)
=====================================================================================
  1. Before any latent claim: print the across-instance separation of the latent AT
     EVERY TIMESTEP. If it is ~1.0 the channel is dead and the falsifier is meaningless.
  2. Falsifier = random DERANGEMENT, never argmin. Report its spread.
  3. Always include a NOT-conditioned control (bias-only / fixed-prefix / fixed-state)
     AND a norm-matched RANDOM-vector control. "Better than nothing" is not "uses the
     context".
  4. Split BY INSTANCE. Dedupe. Verify leakage numerically; never assume it.
  5. Zero-init the write so the untrained module IS the incumbent, exactly.
  6. Clip the residual to a fraction of ||h||; do not rescale it.
  7. Bootstrap every headline number. A zero-width CI is a bug, not a result.
  8. Assert every patch anchor; verify the symbol exists after patching.
  9. A result that gets BETTER when you add a control is SUSPICIOUS: >100% "recovery"
     was the tell that the falsifier had been rigged.
 10. When a null appears, check the write path BEFORE concluding anything about the
     idea. Three of this session's nulls were dead channels.

=====================================================================================
NEXT STEP
=====================================================================================
Stop the architecture search. The open question is the LOSS: what objective forces the
latent to encode what the LM does not already represent? This file shows that
contrastive/retrieval supervision does it for a STATIC issue. The untested step is
whether a contrastive or mutual-information objective also works when the latent must
be used DURING generation (the state tracker's setting), where CE currently drives it
to a positional prior.

To make the cross-interface claim publishable: multiple seeds, a second LM, a second
task family. The DIRECTION is consistent across three interfaces -- that is what makes
it interesting -- but the magnitudes are not tight. Current scope is one LM
(Qwen2.5-3B 4-bit), one task family (SWE-bench single-line fixes), one seed per cell,
held sets of 75-167.
"""
from __future__ import annotations

import argparse
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

from v5.runtime.algo_grr_nstm_sup import group_split, load_pairs


class ContrastiveProbe(nn.Module):
    """issue -> query; candidate fix -> key; score = dot product. Deliberately the SAME shape of
    encoder that failed under CE (a linear map off the MiniLM embedding), so any difference is
    attributable to the LOSS and not to added capacity."""

    def __init__(self, d_in: int = 384, d: int = 128):
        super().__init__()
        self.q = nn.Linear(d_in, d, bias=False)      # no bias: a bias 2x the signal collapsed a
        self.k = nn.Linear(d_in, d, bias=False)      # previous module's separation to 0.917
        self.fixed_q = nn.Parameter(torch.randn(d) * 0.02)   # the no-issue control
        self.scale = nn.Parameter(torch.tensor(10.0))

    def forward(self, issue_emb, cand_emb, mode: str = "issue"):
        """issue_emb [B, d_in]; cand_emb [B, N, d_in] -> logits [B, N]."""
        if mode == "fixed":
            q = self.fixed_q.unsqueeze(0).expand(issue_emb.shape[0], -1)
        else:
            q = self.q(issue_emb)
        q = F.normalize(q, dim=-1)
        k = F.normalize(self.k(cand_emb), dim=-1)
        return self.scale * torch.einsum("bd,bnd->bn", q, k)

    @torch.no_grad()
    def separation(self, issue_emb) -> float:
        """If queries do not vary across instances the task is unsolvable regardless of the loss --
        the same guard that caught three false nulls today."""
        q = F.normalize(self.q(issue_emb), dim=-1)
        S = (q @ q.T).cpu().numpy()
        return float(np.mean(S[~np.eye(len(S), dtype=bool)]))


def build_batches(rows, emb_issue, emb_fix, n_cand: int, rng):
    """For each row: the TRUE fix plus n_cand-1 fixes sampled from OTHER instances. The true one is
    placed at a random index so position carries no information."""
    N = len(rows)
    Q, C, Y = [], [], []
    for i in range(N):
        others = [j for j in range(N) if j != i]
        negs = rng.sample(others, min(n_cand - 1, len(others)))
        cands = [i] + negs
        pos = rng.randrange(len(cands))
        cands[0], cands[pos] = cands[pos], cands[0]
        Q.append(emb_issue[i])
        C.append(np.stack([emb_fix[j] for j in cands]))
        Y.append(cands.index(i))
    return (torch.tensor(np.stack(Q)), torch.tensor(np.stack(C)), torch.tensor(Y))


def main():
    ap = argparse.ArgumentParser(description="Contrastive positive control for the adapter shortcut.")
    ap.add_argument("--n", type=int, default=900)
    ap.add_argument("--cand", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    rows = load_pairs(a.n, seed=a.seed)
    tr, held = group_split(rows, seed=a.seed)
    chance = 1.0 / a.cand
    print(f"{len(rows)} real single-line SWE-bench fixes | train {len(tr)} held {len(held)} "
          f"(BY INSTANCE)")
    print(f"{a.cand} candidates per query -> CHANCE = {chance:.4f}\n", flush=True)

    from embedder import encode_batch
    EI_tr = np.asarray(encode_batch([r["issue"][:600] for r in tr]), dtype=np.float32)
    EF_tr = np.asarray(encode_batch([r["new"][:200] for r in tr]), dtype=np.float32)
    EI_h = np.asarray(encode_batch([r["issue"][:600] for r in held]), dtype=np.float32)
    EF_h = np.asarray(encode_batch([r["new"][:200] for r in held]), dtype=np.float32)

    probe = ContrastiveProbe()
    opt = torch.optim.Adam(probe.parameters(), lr=3e-3)
    Qh, Ch, Yh = build_batches(held, EI_h, EF_h, a.cand, random.Random(999))

    for ep in range(a.epochs):
        Q, C, Y = build_batches(tr, EI_tr, EF_tr, a.cand, rng)   # fresh negatives every epoch
        opt.zero_grad()
        loss = F.cross_entropy(probe(Q, C), Y)
        loss.backward(); opt.step()
        if (ep + 1) % 20 == 0:
            with torch.no_grad():
                acc = float((probe(Qh, Ch).argmax(-1) == Yh).float().mean())
            print(f"    epoch {ep+1:3d}  train loss {float(loss):.4f}  held acc {acc:.4f}",
                  flush=True)

    with torch.no_grad():
        pred = probe(Qh, Ch).argmax(-1)
        acc = (pred == Yh).float()
        # SHUFFLED: derangement of the issue queries. Must fall to chance.
        perm = list(range(len(held)))
        while True:
            random.Random(4242).shuffle(perm)
            if all(perm[i] != i for i in range(len(perm))):
                break
        acc_s = float((probe(Qh[perm], Ch).argmax(-1) == Yh).float().mean())
        acc_f = float((probe(Qh, Ch, mode="fixed").argmax(-1) == Yh).float().mean())
        sep = probe.separation(torch.tensor(EI_h[:64]))

    A = float(acc.mean())
    rs = np.random.RandomState(0)
    an = acc.numpy()
    boot = np.array([an[rs.randint(0, len(an), len(an))].mean() for _ in range(4000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])

    print(f"\n{'=' * 74}")
    print(f"CONTRASTIVE POSITIVE CONTROL   (held {len(held)}, {a.cand} candidates)")
    print(f"  chance                          {chance:.4f}")
    print(f"  F no-issue (learned constant)   {acc_f:.4f}   [must be ~chance by construction]")
    print(f"  R shuffled issue (derangement)  {acc_s:.4f}   [must fall to ~chance]")
    print(f"  P issue query                   {A:.4f}   95% CI [{lo:.4f}, {hi:.4f}]")
    print(f"  query separation across instances: {sep:.4f} "
          f"{'<-- COLLAPSED, task unsolvable' if sep > 0.99 else '(queries vary)'}")
    above = lo > chance
    print(f"\n  -> {'ABOVE CHANCE: the issue DOES carry instance-specific information a small net can '
                   'extract. The CE nulls are about the LOSS, not the information.' if above else 'AT CHANCE: the information is not recoverable from this representation, so the CE nulls say nothing about supervision.'}")
    print(f"{'=' * 74}")


if __name__ == "__main__":
    main()
