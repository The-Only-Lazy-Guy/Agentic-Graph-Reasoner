# Artifact Graph — Self-Growing Algorithm Memory (design)

**Status:** mechanism built + validated (no-GPU selftest + 3B run), 2026-07-11. Training **not
started** — this doc is the design to review before we build it.
**Code:** `v5/runtime/artifact_graph.py` (commit `50228a1` on `fix/swe-slot-plan-gate-real-file`).
**Relationship to prior work:**
- Successor to `tool_memory.py` / `tool_compose.py` / `tool_library.py`. Those proved
  induce→verify→compose→reuse but **hard-coded the taxonomy** (a fixed `PRIMS` registry, a fixed
  `TASKS` list — *we* decided which nodes exist). This removes that.
- Sibling to `graph-resident-learning.md` (the **learning-algorithm** store) and `LGGNv4_design.md`
  (latent retrieval). Same graph substrate, different layer.

---

## 1. The problem, and the principle

We want a graph memory that stores **algorithms** (DNA-like), and reasoning that *uses* them. Two
things forced this redesign:

1. **We cannot enumerate node types.** There are millions of algorithms. Any fixed schema
   ("tool node", "decomposition node", "learning node") is us imposing *what we need it to be*.
   The system must **create the kind of node it needs.**
2. **We cannot rely on the LM.** A 3B "predicts next token, not next idea." Trusting it to derive
   correct plans/abstractions in one pass means "if the LM derives it wrong, everything is wrong."

The principle that resolves both:

> **A node is just an ARTIFACT** — a piece of code the model wrote. It has **no declared type**;
> its "type" is emergent (where its embedding lands). The graph keeps a node by exactly **one law:**
> **reusing it must raise VERIFIED downstream success.**

We supply the *selection pressure* (verify + credit + prune); the graph supplies the *content* (the
vocabulary). This is the DreamCoder wake/sleep shape on a verified substrate.

---

## 2. The mechanism (the physics, not a taxonomy)

A stream of tasks, each specified **only** by `(target_fn, natural-language text, oracle, I/O)` —
never by "which primitive to build".

```
wake — per task:
  1. RETRIEVE candidate artifacts from the graph (embedding rank) and advertise them
  2. AUTHOR: the LM writes a solution — it MAY call advertised artifacts and/or DEFINE new
     helper functions for sub-computations it invents (we never tell it what to factor)
  3. VERIFY the solution end-to-end by execution against the oracle (grounded)
  4. CAUSAL CREDIT: an advertised artifact is credited only if it was CALLED *and* removing it
     BREAKS verification (counterfactual) — this kills credit for inert "present but useless" calls
  5. REGISTER every function the model defined (target + invented helpers) as candidate nodes

sleep — periodically:
  * PRUNE nodes never causally reused by a LATER task (one-offs decay out)
  * MERGE behavioral duplicates (same output fingerprint on a probe set) into one node
```

Two design choices are load-bearing and are the honest improvements over `tool_library`:

- **Causal (counterfactual) credit, not call-count.** `tool_library` counted any call as reuse. Here
  an artifact earns credit only if the solution *fails without it*. This is what makes the reward
  (§6) hack-resistant — you cannot farm credit by spamming useless calls.
- **Behavioral merge does the "find shared abstraction" work**, not the LM. Two solutions that each
  define an equivalent `is_prime` fingerprint-match → collapse to one node. The graph discovers reuse;
  the LM never cross-references solutions.

### 2.1 Node schema (`Artifact`)

| Field | Meaning |
|---|---|
| `name`, `code`, `sig` | the def and its signature |
| `doc` | one-line description (task snippet, or "helper invented while solving X") |
| `calls` | graph edges — other artifacts this one calls (transitive deps resolved at verify time) |
| `fp` | behavioral fingerprint (output vector over a mixed probe set) → dedup/merge key |
| `emb` | retrieval address ("emergent type") |
| `born`, `last_used`, `wins`, `uses` | provenance + credit stats (`wins` = causal reuses) |

No `type` field. The vocabulary is whatever survives.

---

## 3. What is validated

**Mechanism — proven with NO model** (`python -m v5.runtime.artifact_graph --selftest`, 5 gates):
1. behavioral-duplicate **merge** (renamed `is_prime` → one node);
2. **transitive dep closure** resolves + verifies;
3. **causal credit** — credits `is_prime`, refuses an inert `digits` call;
4. full 9-task StubLM run: **9/9 solved**, emergent vocabulary **`{is_prime, digits, digit_sum}`**
   (named in *no* task spec), **7 causal reuses on 3 distinct primitives** (compounds), one-off
   `c2f` and dead composites **pruned**;
5. **persistence** — save/reload the vocabulary, rerun reuses it, re-authors 0 primitives.

**Plumbing — Qwen2.5-0.5B smoke:** clean end-to-end (loads / generates / verifies / registers).

**Frozen Qwen2.5-3B run (`--store-dir artifacts/algo_graph`):** every graph mechanism fired
correctly — 9/9 solved; causal credit fired (`digital_root` reused `digit_sum`); versioning fired
(`digit_sum__7`, `count_prime_digits__7` kept separate); pruning fired; persistence wrote; survivor
= `digit_sum`.

### 3.1 The honest finding

**Compounding was weak: 1 reuse.** Root cause:

> The frozen 3B writes **monolithic** solutions — it **inlines** primality / digit-extraction instead
> of factoring reusable helpers. It only reused `digit_sum` because "repeatedly sum the digits" made
> the call linguistically obvious.

This **empirically confirms "we can't rely on the LM"**: it will not spontaneously author the reusable
atoms, so the graph cannot wait for it. The mechanism is sound; the *proposer's default behavior* is
the gap.

---

## 4. Rejected fixes (and why)

- **Post-hoc corpus mining** (LM reads N solutions, proposes common helpers): correct but **compute-
  heavy and scales badly** — the LM cross-references a set that grows every task.
- **Conditional refactor / subgoal-first prompting** (make the model emit separate helper defs):
  cheaper, but it **forces** the behavior with a prompt scaffold rather than the model learning it.

Decision (user): **train the model to synergize with the graph — don't force it.** The capability
should live in the weights, so the model factors and reuses *by default*, and it transfers.

---

## 5. Honest limits (the floor does not move)

- **Causal credit needs a grounded verifier at the leaves.** The graph can invent unlimited node
  *kinds*, but every win must trace to a verifiable check somewhere. Fully non-verifiable open-ended
  tasks ("design an app", assumptions with no oracle) → no credit signal → back to raw base-model
  judgment. That residual is the **base-model ceiling**, not something the mechanism removes.
- **Retrieval ranking at scale is unbuilt.** v1 advertises top-k by a cheap hash embedding. When the
  graph holds thousands of artifacts, "which stored algorithm applies" becomes the real policy — the
  natural home for the validated ranker (`compose_pool`, +0.95) or an HRM-latent selector over the
  algorithm graph.

---

## 6. Training to synergize (the plan — to review, not yet built)

Goal: fine-tune the proposer so it **factors, calls the library, and composes by default** — because
it is rewarded for it, not scaffolded into it. This is the *learning-algorithm* layer applied to the
implementation-algorithm store.

**The reward already exists in the loop — no labels:**
```
reward = verified_correct                 # grounded (execution vs oracle)
       + causal_reuse_bonus               # the counterfactual credit of §2 — NOT raw call-count
       (+ factoring_bonus)                # emitted a NOVEL, VERIFIED helper def (local proxy for a
                                          #   payoff that is otherwise delayed to future tasks)
```
The reuse term **must** be causal, or the model reward-hacks by spamming inert calls.

**Prerequisites (named up front — this is where RL projects burn time):**
1. **Task generator.** 9 tasks cannot RL. Need many small *verifiable* tasks that **share latent
   primitives** (so reuse is learnable + rewardable) — procedural, like `reason_rl.make_game`.
2. **Curriculum.** Reuse-reward can't fire on an empty graph → primitive-ish tasks before composites.
3. **Delayed credit.** A factored helper pays off on *future* tasks; reward the local proxy (novel
   verified helper) so the signal isn't purely delayed.

**Validation ladder (cheap → expensive; gate GPU on the cheap steps):**

| step | what | cost | proves |
|---|---|---|---|
| 1 | define reward + **selftest it no-GPU** | free | reward tracks good synergy behavior |
| 2 | **best-of-N with reuse-preference** (no training) | 1 molab run | the model *can* produce reuse-solutions + the reward is achievable |
| 3 | **GRPO LoRA** (reuse `reason_rl` GRPO infra) | molab | bake it into the weights → reuses first-try, no search |

**Step 2 is the honest gate.** If preferring reuse-solutions among N samples raises causal reuse at
equal correctness, the behavior is learnable and RL is worth the GPU. If the model can't produce them
even when selected for, RL won't conjure it — that points at a bigger base (`project-model-strategy`).

---

## 7. Where this sits in the vision

One graph substrate, three algorithm layers, all grown by the same law (propose → verify → credit →
keep):

| layer | node example | "does" | status |
|---|---|---|---|
| **compute** | `nash_solver`, `is_prime` | the operation | ✓ `tool_memory` / this |
| **decompose** | how to split a task-family | *how to plan* | emergent here (not a fixed type) |
| **learn** | an RL strategy | *how to train* | `graph-resident-learning.md` |

Precedent: **DreamCoder** (wake/sleep library learning — invents its own primitives by compression +
downstream usefulness) works because its leaves are verifiable I/O examples; same shape, same floor.
Long-term reasoning = retrieve proven decompositions + retrieve/induce tools + execute + verify,
deriving fresh only when memory has nothing — and **banking every verified derivation so it is never
re-derived.** The latent controller that sequences retrieve→bind→execute over a large graph is the
HRM-latent frontier.

---

## 8. Provenance

- **Code:** `v5/runtime/artifact_graph.py`. Reuses `verify_fn` (`tool_compose`), `_extract_code`/`_log`
  (`tool_memory`), `batch_generate` (`reason_rl`), `load_frozen_lm`.
- **Run:** `python -m v5.runtime.artifact_graph --selftest` (no GPU) ·
  `V5_LM_TRUST_REMOTE_CODE=1 python -m v5.runtime.artifact_graph --model Qwen/Qwen2.5-3B --store-dir artifacts/algo_graph`
  (`_log` → stderr; molab shows stderr).
- **Memory notes:** `artifact-graph-emergent-vocab.md`, `tool-induction-validated.md`.
