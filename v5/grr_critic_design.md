# GRR-Critic — internalized error-noticing + signed mistake memory (design, locked 2026-07-19)

## Why
Deployment target is a **verifier-parameterized reasoning substrate**, not a Python tool
([[grr-architecture]]). The compounding graph only stays clean because a verifier is the sole writer —
so the system generalizes exactly as far as the domain has a verifier. Two wants pull past that:

1. **Notice the mistake without always calling the verifier** (speed + reach into weakly-verifiable
   domains).
2. **Generalize reasoning patterns across domains** (code → math → physics → general).

Both are one mechanism: you cannot learn to notice failure without failures, so **storing mistakes is
the prerequisite for self-noticing.** The capability that generalizes lives in the *small trained
reasoner* (TRM / neural planner + a critic head), not the frozen author.

## The hard line (or we reopen poison)
The critic **amortizes** the verifier; it does **not replace it as the writer.** A learned critic that
is wrong = the LoRA/context poison channel we spent the project killing. Discipline:

- **verifier = the WRITER** of `(+)` atoms (banking gate), when present.
- **critic = the FILTER + confidence** everywhere: prioritize search, auto-reject gross errors, route.
- **no verifier available** (general domain) → critic gates a **provisional / quarantine** tier,
  promoted to verified only when a verifier or teacher-consensus later confirms. The critic never writes
  a permanent `(+)` atom alone.

"Reduce reliance" is a **decay curve, not a switch** — verifier-calls-per-task fall as the critic learns.

## Three-tier SIGNED memory
| tier | sign | content | retrieval | writer |
|---|---|---|---|---|
| content atoms | `+` | executable unit (function / lemma / law) | realizer uses it | verifier |
| reasoning-move atoms | `+` | domain-general how-to-think (decompose→verify, refine-on-fail, retrieve-analogue) — the F-bank (#69) / meta-nodes (#54) | planner composes moves | verifier-gated on outcome |
| **mistake nodes** | `−` | failed trace + verdict + reason/failure-mode | **WARNING only** — feeds critic + prunes search; **never realized as a solution** | critic (provisional) |

Mistakes key on the **failure signature** (task-shape + error-shape), not the exact wrong line, so they
generalize. They **abstract into failure modes** (off-by-one-at-boundary, missing-base-case,
dimensional-mismatch, wrong-invariant) via the existing sleep/prune machinery — negative atoms that
often transfer *better* than positive moves (a boundary off-by-one is the same bug in code indexing and
physics limits).

## How the model learns to notice itself
The teacher's reason is a **label**, not magic. Distill → amortize → drop:

```
1. model attempts → VERIFIER (truth) says pass/fail → TEACHER says WHY (reason)   [training time]
2. (trace, verdict, reason) = a training pair, stored as the (−) node
3. train CRITIC to PREDICT verdict/reason from the TRACE ALONE — no teacher, no verifier at input
4. gradient moves the judgment into the critic's weights
5. inference: critic emits "wrong because X" the teacher used to emit → teacher dropped
```

**"Notice itself" = predict the judge's verdict before the judge runs.** Same amortization already proven
in GRR-7 (search did the reasoning, net amortized) — new target is the *judge*. The **uncertainty head
(GRR-17, already built)** makes it calibrated: confident → skip verifier; unsure → escalate. That routing
IS the reduced-reliance curve.

### Decay loop
```
critic flags "looks wrong" → VERIFIER confirms (still truth) →
confirmed catches = NEW training pairs → critic sharpens → teacher/verifier called LESS
```
Teacher involvement decays over training. Self-supervision rides the verifier axis: safe novelty only
where a gate can still call it wrong.

## What the critic can / cannot notice (honest)
Without the oracle a critic reliably catches **gross, structural, semantic** errors — crashes,
type/shape violations, sanity violations (negative count, non-monotone where task implies monotone),
task↔behavior mismatch. It **cannot** catch a subtle plausible off-by-one — that genuinely needs the
verifier. So: critic handles the gross majority cheaply + transferably; verifier is reserved for the
subtle residue. This is exactly "reduce reliance, not eliminate."

## Roles / generalization axis
| piece | learns | generalizes? | poison-safe |
|---|---|---|---|
| neural planner / TRM | reasoning STRUCTURE (compose/decompose/retrieve/refine) | ✅ over graph, not syntax | proposes; verify gates |
| **critic head** | error-noticing | ✅ reads trace repr | filter only, never sole writer |
| frozen LM | domain content (writes the math/code) | ❌ domain-specific | frozen = no gradient poison |
| graph | verified memory (3 tiers) | topology general, atoms per-domain | verify-gated writes |

## Teacher (molab 32B) — two distillations, both verifier-gated
1. **patterns → planner**: teacher solves multi-domain tasks, student TRM imitates STRUCTURE (verified
   traces only).
2. **error-reasons → critic**: teacher gives the *why* (denser than pass/fail), critic predicts
   verdict + reason.

Guardrail: teacher traces enter training **only if the external verifier passes them.** Teacher = denser
proposer; verifier = truth. A teacher can be confidently wrong; the gate cannot.

## The experiment (one run proves all four wants)
Train the critic on the **code** verifier's labels + stored `(−)` nodes, then measure:
- **decay** — on held-out CODE: AUC + how many verifier calls it auto-saves at high precision.
- **self-noticing across domains** — on **MATH it never trained on**: does it flag wrong math (AUC vs
  chance vs a math-trained ceiling)?

One number demonstrates: mistakes-in-graph → self-noticing → cross-domain transfer → reduced verifier
reliance.

**Representation (domain-general, no oracle):** `emb(task) ‖ emb(attempt ‖ behavior) ‖ [crash, monotone,
has_neg, mean_magnitude, cos]`. Behavior = attempt run on a few probe inputs (observe, don't compare to
truth). Crash + magnitude + task↔behavior mismatch are the transferable symptoms.

**Module:** `v5/runtime/algo_grr_critic.py` (`--selftest` no-GPU stub embedder; `--run` real MiniLM +
real execution/sympy verdicts). Verifier stays the writer throughout — the critic only filters/routes.

## Honest risks
- MiniLM is a *sentence* embedder — weak on raw code; the code tier may need a code-model embedder. Math
  (natural-language-ish) should transfer better. Report both; the limiter is a finding.
- Subtle-numeric errors are out of a no-oracle critic's reach *by construction* — don't claim them.
- Quarantine tier must be pruned aggressively or `(−)` clutter degrades retrieval.

---

# v6 Dual-Channel realizer (from the user's 2 brainstorm designs, built 2026-07-20)

**Trigger:** nodes were bare code (`store[name] = code`) and the realizer PASTED atom bodies + hard-coded
wiring → the LM wrote nothing, couldn't explain, output was spliced. User brainstormed **Design A
(Dual-Channel Pointer-Decoding)** and **Design B (TRM owns computation, LM owns communication)**. Both =
one idea: **separate SEMANTIC INTENT from SYMBOLIC EXECUTION; graph stores meaning, not code.**

**The channels (both DISCRETE — the one design correction):**
- **Symbolic** = exact structure from the graph: verified atom closure (immutable) + typed HOLES the LM
  fills, grammar/AST-constrained. Zero syntax hallucination on the hard parts; a hallucinating filler can
  only damage a hole, caught by grammar+verify. (Design A γ=0 / Design B AST-fragments + syntax lattice.)
- **Semantic** = the explanation, **narrated from the execution-graph traversal + node cards** → faithful
  by construction (can only cite atoms actually in the verified program). Measured: **1.0 vs 0.40**
  post-hoc free-form.

**The design correction (measured, not opinion):** Design A's semantic channel is *continuous latent*
(`h_latent → LM` via soft-prompt/cross-attn). This project already measured that path failing — the
z-wall / softprompt 73%→15% routing collapse; *text is THE memory interface*. So **both channels are
discrete**: symbolic = grammar/AST, semantic = text. The continuous-latent channel stays a **registered,
swappable seam** (`LatentSemanticChannel`) that must **win `fair_ab` vs text** before adoption — the
beautiful idea gets a door and an honest gate, not a free pass and not the bin.

**Mistake handling:** Design B's **negative-edge check** (prune the forbidden candidate BEFORE generation)
gives the pink-elephant benefit (LM never sees the trap in-context; the reason is still narratable) with
NONE of Design A's latent-repulsion risk (no hidden-state hooks / custom runner). Symbolic, safe, shippable.

**Built (commit ceae538, all no-GPU green):** `AtomNode` rich nodes in `algo_grr_pipeline.py`;
`algo_grr_dcpd.py` = `build_skeleton` / `dual_channel_realize` / `TextSemanticChannel` /
`LatentSemanticChannel` (parked) / `MistakeNode`+`prune_candidates` / `fair_ab`. Selftest: code verifies
24/24, closure exact 24/24, faithful 1.00 vs 0.40, hallucinated glue caught + closure survives, mistake
prune fires, latent seam honestly NotImplemented.

**Next:** wire `dual_channel_realize` into `MembraneV2.solve` (optional, default unchanged); real
grammar-constrained hole-fill (Outlines / GBNF) on the `--lm` run; implement `LatentSemanticChannel` on a
white-box runner only to feed `fair_ab`. Keep the code-atom compounding number as the proposal headline;
dual-channel is the "explains its approach + stores meaning not code" story.

**REAL 3B PROOF (2026-07-20):** `algo_grr_dcpd --run --lm` (Qwen2.5-3B), n=40 — DUAL-CHANNEL 39/40 verifies,
**0 syntax errors**, faithfulness **1.00**; FREE-INLINE 18/40, **5** hallucinated syntax, no explanation.

---

# Deployment-scaling: teacher-verifier, gate-RL, TRM freezing (design 2026-07-20)

**Teacher-as-verifier, retire the critic.** The cross-domain critic failed (code→math AUC 0.46 = chance;
MiniLM too weak on code). Replace its training-time role with the teacher (molab 32B): execution verifier
stays PRIMARY where it exists; the teacher judges only where none exists; a teacher verdict trains the
student only if the real verifier passes it OR N-of-M teacher consensus gates a *provisional* tier. Teacher
= denser proposer (can be wrong) → never writes a permanent node alone. The signed mistake-tier + the
subprocess executor survive (reused by dcpd's negative-edge prune); only the error-noticer role retires.

**Gate/delegation RL without mode collapse.** RL-tuning how much the TRM delegates to the LM collapses to
**A (Silent Coder)** γ→0 everywhere (perfect code, no explanation) or **B (Yapper)** γ→1 everywhere (plain
hallucinating LLM). Fix = COMPOSITE reward: `R_exec` (terminal, big syntax penalty) + `R_bridge` (reward
narration ONLY if exec passes, **faithfulness-weighted not token-volume**) + `P_mode` (penalize graph-on-Sem
/ LM-on-Struct) + entropy `H` with decaying β; CURRICULUM syntax-sandbox → forced-explanation. The AST
skeleton supplies the `Struct/Sem` label per span **for free** (closure=Struct, holes=Sem). This trains the
DISCRETE delegation policy we ship (continuous γ is Design A, parked behind `fair_ab` — same reward trains
it if it wins). Keep a STaR fallback (search gate-schedules → verify → amortize) since sampling-RL was
unstable for program synthesis on this stack. VALIDATED no-GPU (`algo_grr_gate_rl --selftest`): R_exec-only
→ Collapse A (Sem→LM 0.00); composite curriculum → **synergy** (Struct→graph 0.00 / Sem→LM 1.00 / exec 1.00);
drop-syntax-penalty + no-P_mode + expensive-graph → Collapse B (Struct→LM 1.00, exec 0.76). Both traps
reproduced; composite reward avoids them.

**TRM freezing at deployment.** The LM stays FROZEN forever (weight-poison, measured). The TRM *can* unfreeze
but ONLY via verifier-gated STaR (learn from verified-solved traces), NEVER naive online RL/self-train (the
GRR-7 wanderer). Recommended: TRM FROZEN on-device by default, adaptation lives in the GRAPH (rebuild-net:
graph is the memory, net re-amortizes); the teacher re-trains the TRM offline (STaR + gate-RL) and pushes
updates. Optional research knob: a slow verifier-gated online-STaR trickle + a forgetting guard
(frozen-holdout replay / EWC). Unfreeze only behind the gate that already makes banking safe.
