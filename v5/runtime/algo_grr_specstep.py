"""algo_grr_specstep — STEP-level speculative decoding for agentic tasks (speculate IDEAS, not tokens).

The failed draft idea had the TRM GENERATE code tokens (dead: a tiny net can't write code). This inverts it
correctly: for a multi-step / agentic task the unit is a STEP (an atom call / sub-goal), not a token. The
LM is the (expensive) planner; the TRM (cheap) SPECULATES the next K steps as a chunk; the LM VERIFIES the
whole chunk in ONE call and supplies the correct step only at the first divergence. The TRM speculates
STRUCTURE (which steps) — the thing that works — never code. Win: when the TRM's plan is right, one verify
call advances K steps -> up to K× fewer LM planning calls, same (verify-gated) correctness.

  sequential:  L steps  -> L expensive LM planning calls
  speculative: TRM proposes K -> 1 LM verify accepts the correct prefix -> far fewer LM calls

This is the compounding lever for AGENTIC tasks (the TRM learns recurring step-motifs from solved tasks and
speculates whole motifs). Correctness is never sacrificed: every accepted step is one the LM verified.

    python -m v5.runtime.algo_grr_specstep --selftest   # no-GPU: LM-call savings + 100% correctness
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# recurring step-motifs: an agentic plan is a sequence of these chunks (learnable structure)
_MOTIFS = [[3, 4, 5], [6, 7], [8, 9, 10, 11], [4, 5, 3], [7, 6, 8], [10, 9]]
_A = 12                                            # step vocabulary size


def gen_plan(rng: random.Random, n_motifs: int) -> list[int]:
    plan: list[int] = []
    for _ in range(n_motifs):
        plan += rng.choice(_MOTIFS)
    return plan


def _oracle_prefix(plan: list[int], done: int, spec: list[int]) -> int:
    """LM verifies a speculated chunk in ONE call: length of spec that matches the true continuation."""
    n = 0
    for s in spec:
        if done + n < len(plan) and plan[done + n] == s:
            n += 1
        else:
            break
    return n


def spec_step_solve(plan: list[int], speculate, K: int) -> tuple[list[int], int]:
    """LM plans; TRM speculates K steps; LM verifies the chunk (1 call) + supplies the step at the first
    divergence (1 call). Returns (reconstructed_plan, lm_calls). Correctness is guaranteed (verify-gated)."""
    hist: list[int] = []
    lm_calls = 0
    while len(hist) < len(plan):
        spec = speculate(hist, K)                  # cheap TRM chunk (no LM call)
        lm_calls += 1                              # ONE LM verify call over the whole chunk
        n_ok = _oracle_prefix(plan, len(hist), spec)
        hist += spec[:n_ok]                        # accept the verified prefix
        if len(hist) < len(plan):                  # first divergence -> LM supplies the correct step
            lm_calls += 1
            hist.append(plan[len(hist)])
    return hist, lm_calls


def _build_speculator():
    import torch
    import torch.nn as nn

    class StepSpeculator(nn.Module):
        """Tiny GRU over step-embeddings: predict the next step from history. Speculate K = autoregressive
        rollout. Learns the recurring motifs -> speculates whole chunks correctly."""

        def __init__(self, vocab: int, d: int = 64):
            super().__init__()
            self.emb = nn.Embedding(vocab + 1, d)   # +1 = BOS
            self.gru = nn.GRU(d, d, batch_first=True)
            self.head = nn.Linear(d, vocab)
            self.bos = vocab

        def forward(self, seqs):                     # seqs [B,L] -> logits [B,L,vocab]
            y, _ = self.gru(self.emb(seqs))
            return self.head(y)

        @torch.no_grad()
        def speculate(self, hist, K):
            self.eval()
            seq = [self.bos] + list(hist)
            t = torch.tensor([seq])
            out = []
            for _ in range(K):
                nxt = int(self.forward(t)[0, -1].argmax())
                out.append(nxt)
                t = torch.cat([t, torch.tensor([[nxt]])], dim=1)
            return out

    return torch, nn, StepSpeculator


def train_speculator(model, plans, steps=800, lr=3e-3, seed=0):
    import torch
    rng = random.Random(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = torch.nn.CrossEntropyLoss()
    for it in range(steps):
        p = plans[rng.randrange(len(plans))]
        seq = torch.tensor([[model.bos] + p])
        logits = model(seq)[0, :-1]                  # predict p from [BOS]+p[:-1]
        loss = lossf(logits, torch.tensor(p))
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def _selftest() -> bool:
    print("algo_grr_specstep --selftest: STEP-level speculation for agentic tasks (no GPU)\n")
    torch, nn, StepSpeculator = _build_speculator()
    torch.manual_seed(0)
    rng = random.Random(1)
    train_plans = [gen_plan(rng, rng.randint(4, 8)) for _ in range(300)]
    test_plans = [gen_plan(random.Random(9000 + i), rng.randint(5, 9)) for i in range(60)]

    spec = StepSpeculator(_A)
    train_speculator(spec, train_plans, steps=1000)

    K = 4
    tot_seq = tot_spec = 0
    correct = 0
    for p in test_plans:
        recon, lm_calls = spec_step_solve(p, spec.speculate, K)
        tot_seq += len(p)                            # sequential = 1 LM plan call per step
        tot_spec += lm_calls
        correct += int(recon == p)                   # verify-gated -> must be exact
    n = len(test_plans)
    speedup = tot_seq / max(1, tot_spec)
    print(f"  held-out agentic tasks: {n}, chunk size K={K}\n")
    print(f"  correctness (verify-gated)       : {correct}/{n}  (every accepted step was LM-verified)")
    print(f"  LM planning calls — SEQUENTIAL   : {tot_seq}   (one per step)")
    print(f"  LM planning calls — SPECULATIVE  : {tot_spec}   (TRM speculates chunks, LM verifies)")
    print(f"  speedup (fewer LM calls)         : {speedup:.2f}x")
    ok = correct == n and speedup > 1.5
    print(f"\n  -> {'PASS' if ok else 'FAIL'}: the TRM speculates whole STEPS (learned motifs); the LM verifies\n"
          f"     chunks -> {speedup:.1f}x fewer LM calls at NO cost to correctness. Speculate ideas, not tokens.")
    print(f"\n  ALGO_GRR_SPECSTEP SELFTEST -> {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
