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
