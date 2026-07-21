# DCPD verdict — the latent Dual-Channel from NEW_DESIGN(v6), tested for real (2026-07)

The original design (`NEW_DESIGN(v6).md`) had two latent mechanisms. Both were replaced by discrete/text
stand-ins during implementation (`algo_grr_dcpd.py`) because of the z-wall (latent code-transport collapse).
This session finally built + ran BOTH on the real white-box network (`v5/runtime/dcpd_latent.py`, Qwen2.5-3B).
Every number below is measured on the actual model.

## Channel 1 — γ-gated pointer-decode (symbolic): WORKS. KEEP.
Force the graph's exact tokens (γ=0, byte-perfect, 0 syntax errors by construction) between LM prose (γ=1).
- `--exp gating2` on Qwen: forced code compiles exactly = True; explanation is fluent, on-topic, no code leak.
- Needed the chat template + code-token masking on the explain segment (raw-prompt v1 leaked code).
- **This is the real, validated deliverable: exact syntax + faithful explanation, LM frozen.**

## Channel 2 — concept-space repulsion (latent mistake-steering): REFUTED. DROP.
Steer the residual stream away from a mistake concept. Measured 3 escalating ways vs a one-line text baseline
("Do NOT mention X"). Trap = "bubble sort"; trap-rate lower is better; fluency = self-NLL lower is better.

| method | trap-rate | fluency | vs text |
|---|---|---|---|
| text "avoid X" (baseline) | **0.12–0.50** | 1.29 | — |
| v1 α-scaled subtraction (hand) | 0.83 | 1.43 | lost |
| v2 contrastive dir + directional ablation, band+strength sweep (hand) | 0.88 | 0.86–1.16 | lost (8/10 configs didn't move) |
| v3 TRAINED control vector, LM frozen (gradient to v only) | 0.75 | 1.66 | lost |

Three methods, escalating sophistication, all lose to a one-line text instruction. Not a tuning gap:
**residual-stream steering cannot reliably suppress a concept the model wants to emit without wrecking
fluency.** The z-wall extends fully to steering, not just code-transport.

## Consequence
The NEW_DESIGN(v6) → discrete modification was **HALF right, and correct on both halves**:
- symbolic γ-gating should have been kept — it works (and the discrete grammar-hole `dcpd` was approximating it);
- latent mistake-steering was correctly dropped — it doesn't work; the discrete negative-edge check + graph-
  narrated "why" is both more effective AND gives the required "explain why you avoided it".

**Locked:** keep γ-gating (real). Mistake-handling stays discrete. Latent steering: refuted, parked with these
numbers. LoRA on the LM (would likely suppress) crosses the frozen-LM poison invariant and is overkill (text is
free at 0.12) — not pursued.

Repro: `python -m v5.runtime.dcpd_latent --lm Qwen/Qwen2.5-3B-Instruct --exp gating2`
       `python -m v5.runtime.dcpd_latent --lm Qwen/Qwen2.5-3B-Instruct --exp trained`
