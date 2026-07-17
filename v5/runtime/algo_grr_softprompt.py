"""algo_grr_softprompt — Option 2: the owned LATENT reasoner (TRM -> soft prompt -> FROZEN LM).

The z-wall was measured for a foreign single-layer latent injected into a frozen LM. This routes the
TRM's reasoning latent into the LM's NATIVE input space instead: the TRM produces z, a small owned
projection turns z into K "virtual tokens" (soft-prompt embeddings) prepended to the LM input, and the
FROZEN LM attends to them like real tokens. Only the TRM + projection train (gradients flow THROUGH the
frozen LM into the prefix, never into the LM's weights) -> the LM's knowledge is untouched, so poison is
CONTAINED in the owned, resettable adapter. STaR-trained on VERIFIED code (verifier-grounded). On a failed
attempt, the error feeds back into the TRM -> new z -> new soft prompt -> retry (a LATENT feedback loop,
not a text handoff).

Open question this tests: does a TRM-computed soft prompt add CAPABILITY the frozen LM lacks, or does it
SATURATE (like the earlier L26 reader)? If it lifts solve-rate vs no-prefix, the owned latent reasoner
works and the TRM is the thing to scale. Verify gates every output, so it can never produce wrong code —
worst case it just doesn't help.

    selftest (no GPU):  python -m v5.runtime.algo_grr_softprompt --selftest
      proves the PLUMBING on a tiny frozen stub LM: (1) gradients reach the prefix, NOT the LM;
      (2) prefix-training conditions a FROZEN LM's output (loss drops from baseline); (3) failure
      feedback changes the latent. The capability A/B is molab (real 3B).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _build():
    import torch
    import torch.nn as nn

    class SoftPromptTRM(nn.Module):
        """Owned reasoner: (task, atoms, optional failure) -> latent z -> K soft-prompt vectors in the
        LM's embedding space. This is the ONLY thing that trains (with the tiny projection)."""

        def __init__(self, d_in: int, d_model: int, d: int = 128, K: int = 8, T: int = 3):
            super().__init__()
            self.K, self.d_model, self.T = K, d_model, T
            self.task = nn.Linear(d_in, d)
            self.atom = nn.Linear(d_in, d)
            self.fail = nn.Linear(d_in, d)
            self.q = nn.Linear(2 * d, d)
            self.f = nn.Linear(4 * d, d)                          # [task, atom-summary, z, failure]
            self.ln = nn.LayerNorm(d)                             # keeps z responsive (no tanh saturation)
            self.z0 = nn.Parameter(torch.zeros(d))
            self.proj = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, K * d_model))

        def forward(self, x_vec, atom_vecs, fail_vec=None):
            x = self.task(x_vec)                                  # [d]
            A = self.atom(atom_vecs)                              # [N, d]
            ft = self.fail(fail_vec) if fail_vec is not None else torch.zeros_like(x)
            z = self.z0
            for _ in range(self.T):
                y = torch.softmax((A @ self.q(torch.cat([x, z]))) / (A.shape[-1] ** 0.5), dim=0)
                asum = y @ A                                     # [d]
                z = self.ln(self.f(torch.cat([x, asum, z, ft])))     # failure is a first-class input
            sp = self.proj(z).view(self.K, self.d_model)          # [K, d_model]
            return sp, z

    class StubLM(nn.Module):
        """Tiny FROZEN causal LM for the no-GPU plumbing proof: embed -> 1 self-attn block -> lm_head.
        The soft-prompt tokens sit at the front and attend-mix into later positions (causal), so the
        prefix genuinely conditions the OUTPUT — a faithful mini-proxy for a real frozen LM."""

        def __init__(self, vocab: int, d_model: int):
            super().__init__()
            self.embed = nn.Embedding(vocab, d_model)
            self.attn = nn.MultiheadAttention(d_model, num_heads=2, batch_first=True)
            self.ln = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab)

        def forward_embeds(self, inputs_embeds):                  # [1, S, d_model] -> [1, S, vocab]
            S = inputs_embeds.shape[1]
            mask = torch.triu(torch.ones(S, S), diagonal=1).bool()
            h, _ = self.attn(inputs_embeds, inputs_embeds, inputs_embeds, attn_mask=mask)
            return self.head(self.ln(inputs_embeds + h))

    return torch, nn, SoftPromptTRM, StubLM


def _prefix_loss(torch, nn, lm, soft_prompt, input_ids, target_ids):
    """Next-token CE on target_ids, with the soft prompt prepended to the input embeddings. Differentiable
    through the FROZEN lm into soft_prompt (and thus the TRM + projection)."""
    emb = lm.embed(input_ids)                                     # [1, L, d]
    full = torch.cat([soft_prompt.unsqueeze(0), emb], dim=1)      # [1, K+L, d]
    logits = lm.forward_embeds(full)                             # [1, K+L, vocab]
    K = soft_prompt.shape[0]
    Lt = target_ids.shape[1]
    # positions that predict each target token: the K+L-1 ... slice covering the last Lt targets
    pred = logits[:, K + input_ids.shape[1] - Lt - 1: K + input_ids.shape[1] - 1, :]
    return nn.functional.cross_entropy(pred.reshape(-1, logits.shape[-1]), target_ids.reshape(-1))


# ═══════════════════════════════════════════════════════════════════════════════
# SELFTEST — plumbing proof on a frozen stub LM (no GPU)
# ═══════════════════════════════════════════════════════════════════════════════

def _selftest() -> bool:
    print("algo_grr_softprompt --selftest: owned latent reasoner plumbing (frozen stub LM)\n")
    torch, nn, SoftPromptTRM, StubLM = _build()
    torch.manual_seed(0)
    d_in, d_model, vocab, K = 32, 48, 40, 6
    ok = True

    lm = StubLM(vocab, d_model)
    for p in lm.parameters():
        p.requires_grad_(False)                                   # FROZEN LM
    sptrm = SoftPromptTRM(d_in, d_model, d=64, K=K, T=3)

    # a synthetic "task": a task vector + atom vectors + a target token sequence the LM should produce
    x = torch.randn(d_in)
    A = torch.randn(5, d_in)
    input_ids = torch.randint(0, vocab, (1, 6))
    target_ids = torch.randint(0, vocab, (1, 4))

    # ── [1] gradient isolation: grads reach the prefix (TRM+proj), NOT the LM's weights ──────────
    sp, _z = sptrm(x, A)
    loss = _prefix_loss(torch, nn, lm, sp, input_ids, target_ids)
    loss.backward()
    trm_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in sptrm.parameters())
    lm_grad = any(p.grad is not None for p in lm.parameters())
    print(f"  [1] gradient isolation: TRM+proj get grad={trm_grad}, LM gets grad={lm_grad} -> "
          f"{'PASS' if trm_grad and not lm_grad else 'FAIL'}")
    ok &= trm_grad and not lm_grad

    # ── [2] CONDITIONING: training only the prefix steers the FROZEN LM's output toward the target ─
    torch.manual_seed(0)
    lm = StubLM(vocab, d_model)
    for p in lm.parameters():
        p.requires_grad_(False)
    sptrm = SoftPromptTRM(d_in, d_model, d=64, K=K, T=3)
    with torch.no_grad():
        base = _prefix_loss(torch, nn, lm, torch.zeros(K, d_model), input_ids, target_ids).item()
    opt = torch.optim.Adam(sptrm.parameters(), lr=5e-3)
    for _ in range(150):
        sp, _z = sptrm(x, A)
        loss = _prefix_loss(torch, nn, lm, sp, input_ids, target_ids)
        opt.zero_grad(); loss.backward(); opt.step()
    trained = loss.item()
    print(f"  [2] conditioning a FROZEN LM: loss {base:.3f} (no prefix) -> {trained:.3f} (trained prefix) "
          f"-> {'PASS' if trained < base * 0.5 else 'FAIL'}")
    ok &= trained < base * 0.5

    # ── [3] FEEDBACK: a failure vector changes the latent z (the retry loop has an effect) ──────────
    sp0, z0 = sptrm(x, A)
    sp1, z1 = sptrm(x, A, fail_vec=torch.randn(d_in))
    delta = (z1 - z0).abs().sum().item()
    print(f"  [3] failure feedback shifts z by {delta:.3f} -> {'PASS' if delta > 1e-3 else 'FAIL'}")
    ok &= delta > 1e-3

    # ── [4] param budget: the owned adapter is TINY (deployable) ────────────────────────────────
    n_params = sum(p.numel() for p in sptrm.parameters())
    print(f"  [4] owned adapter size: {n_params/1e3:.1f}k params (frozen LM untouched) -> "
          f"{'PASS' if n_params < 5e6 else 'FAIL'}")
    ok &= n_params < 5e6

    print(f"\n  ALGO_GRR_SOFTPROMPT SELFTEST -> {'PASS' if ok else 'FAIL'}")
    print("  (plumbing proven no-GPU. CAPABILITY A/B — does the soft prompt lift a real 3B's solve rate")
    print("   vs no-prefix, or saturate? — is the molab experiment.)")
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
