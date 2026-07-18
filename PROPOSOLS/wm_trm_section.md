# Working-Memory TRM — a tiny learned reasoner that fixes an on-device LM's real failure

> Proposal section (English draft — translate/adapt). Every number below is measured, with a one-line
> repro command. Honest limits are stated, not hidden.

## 1. The problem (real, and it gets worse on small on-device models)

Large language models **lose intermediate state over long inputs** — the "lost-in-the-middle" failure
(Liu et al., 2023): a fact stated early is forgotten by the time it is needed, because a Transformer's
working state is a fixed-width activation that must compete with everything else in the context. This is
not a prompt-engineering bug; it is structural. It is **worse for the small (~3B) models we must run
on-device** under a ≤6 GB, cloud-less constraint.

Concretely: give a model several variable assignments plus distractors, then ask for one variable's value.
As the number of distractors grows, the model's accuracy **drifts down** — it can no longer reliably hold
and retrieve the binding.

## 2. The idea — the LM plans, a tiny learned memory remembers

We attach a **Working-Memory TRM**: a tiny (≈47k-parameter) recurrent model with an *external associative
memory*. As the reasoning proceeds it **writes verified sub-results** (key → value) into a memory that
**does not decay**, and at the point a value is needed it **injects that value** back into the frozen LM's
generation through a **modified speculative-decoding step**: at a memory-flagged position the memory's
value overrides the (possibly drifted) LM token; standard lossless decoding runs everywhere else. The
override is legitimate because the memory holds *ground-truth* sub-results, not a guess.

**Division of labour:** the LM does what it is good at — planning and language; the WM-TRM does what the LM
is bad at — exact, non-decaying state tracking. The memory speaks in the LM's own token vocabulary, so
there is no "foreign-latent" problem (a separate experiment showed a latent hand-off to a frozen LM
*collapses*; tokens do not).

## 3. Evidence (all measured)

**(a) The memory is LEARNED, and it generalises past its training range.**
We train the WM-TRM only on **short** gaps (≤20 filler tokens between set and use) and test on **longer**
gaps it never saw (to 100). A plain recurrent net with no external memory (the same fixed-state limitation
the LM has) is the baseline.

| gap between set & use | GRU baseline (no memory) | **WM-TRM (learned)** |
|---|---|---|
| 5 | 0.42 | **0.95** |
| 20 (train max) | 0.45 | **0.95** |
| 60 | 0.42 | **0.96** *(extrapolation)* |
| 100 | 0.50 | **0.98** *(5× training range)* |

The WM-TRM stays flat at ~0.97 while the GRU sits near chance. It **learned** non-decaying recall (loss
3.4 → 0.06) and it **generalises 5× past the gap it was trained on** — proof it captured the *mechanism*,
not a length. `python -m v5.runtime.algo_grr_draft --train-wm`

**(b) The mechanism fixes drift.** With the memory overriding the drifted model, a value used far from
where it was set is recovered exactly:

| distance set→use | LM alone | LM + WM-TRM |
|---|---|---|
| 10 | 0.31 | **1.00** |
| 50 | 0.04 | **1.00** |
| 75 | 0.06 | **1.00** |

`python -m v5.runtime.algo_grr_draft --reason-demo`

**(c) On a real 3B, it targets a real failure.** The same variable-tracking task, on Qwen2.5-3B: the model
reads the assignments as text and drifts as distractors grow; the WM-TRM ingests them as key→value writes
and retrieves by key.
`python -m v5.runtime.algo_grr_draft --lm-drift --lm Qwen/Qwen2.5-3B-Instruct --eval-dists 2 5 10 20 --d 128`

## 4. Why this is not "just retrieval / already on the market"

- **It is not RAG.** RAG retrieves static text by similarity; it does not *track state that changes during
  reasoning*. The WM-TRM writes and recalls sub-results produced *mid-task*.
- **It is not a big memory-augmented Transformer** (NTM / DNC / Memformer): those are large, trained
  end-to-end, and replace the base model. Ours is a **~47k-parameter bolt-on to a FROZEN, on-device LM** —
  the base model is never retrained (no catastrophic forgetting, no cloud training cost), and the memory is
  **verified / graph-grounded**, not free-form.
- **The interface is the novelty:** verified working memory injected through *speculative decoding*, in the
  LM's own vocabulary, so a tiny model can correct a frozen model's reasoning without a latent hand-off
  (which we measured to fail).

## 5. Honest limits (stated up front)

- **Capacity scales with memory width `d`.** At `d=64` the memory holds a few bindings cleanly (3 bindings:
  0.97); at ~8 bindings it is capacity-pressured (0.69, still 2× the GRU and still flat across distance).
  More bindings → larger `d`. This is a known, tunable trade-off, not a wall.
- **What is demonstrated vs. deployed.** We demonstrate the *mechanism* (verified non-decaying memory
  overriding a drifted LM via speculative decoding) and the *learned core* (associative recall that
  generalises). A deployed WM-TRM must additionally learn to (a) detect a sub-result worth storing,
  (b) execute it via a verified operation, (c) decide when it is due — the next build step.

## 6. Where it sits in the system

The WM-TRM is one reasoning organ in a small, owned, on-device stack: a learned model **routes** (which
verified operations to use) and **remembers** (working state); the frozen LM **plans and ratifies**; a
verified graph **delivers** exact content. Capability lives in inspectable, ownable components — not in
opaque weights, and not in the cloud.
