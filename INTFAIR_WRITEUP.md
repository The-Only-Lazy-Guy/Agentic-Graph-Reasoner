# Agentic Graph Reasoner — grow the graph, not the parameters

> One line: a **frozen small LM (Qwen3.5-4B, ≤6 GB)** reasons by leaning on an **editable knowledge
> graph** — typed operators + a slot dependency-graph solved to a fixpoint — so the *graph* carries the
> reasoning, the reasoning is **visible + editable**, and capability grows by editing the graph (and a
> tiny skill-LoRA), not by retraining the model.

This document consolidates the **verified** results. Every claim below has a forcing-function check
(a log/assert/verifier), because the project's hardest discipline was refusing to count a component
that was merely *present* — it must be proven to *fire* (`INTEGRATION_CHECKLIST.md`).

---

## 1. The problem
Small models are cheap and inspectable but weak reasoners. The usual fixes — fine-tune in knowledge,
or RAG-dump text — either bury knowledge in opaque weights or let a weak model misread dumped context.
We want the *knowledge and the reasoning structure* to live **outside** the model, editable and legible,
with the frozen model as a per-step *executor*.

## 2. The architecture — operator-grounded slot-graph
A task = a **Task Slot-Graph**: slots (units of the answer) + DATAFLOW dependency edges + 5 filledness
states (empty / tentative / valid / **stale** / **insufficient**), solved to a **fixpoint**.

- **Slots store TEXT in a pool** (no latent collapse) — the reasoning state is on screen.
- **Typed operators** fill/judge slots: ASSERT (+v), **INVALIDATE (−v, the load-bearing subtract)**,
  GATE (×), TRANSFORM (derive), SLOT (register). Op kind is derived from node type (`op_kind_for`).
- **Retrieve-OR-DERIVE**: a slot fills from graph retrieval; if the graph has no fact, the frozen LM
  **derives** it from upstream slots.
- **Dependency-directed backtracking**: a slot that can't be satisfied revises its weakest upstream
  (de Kleer nogoods), recovering from a wrong retrieval.
- **Self-improving memory**: a solved slot-graph is saved as a template and retrieved for the next
  similar task → the graph accumulates *reasoning structures*, not just facts.

Engine: `v5/runtime/slot_coder.py` (14/14 deterministic asserts). Integrated: `v5/runtime/integrated_run.py`.

## 3. Evidence (all verified)

| # | claim | result | how it's proven |
|---|---|---|---|
| 1 | **Typed operators carry content RAG can't** | operator **beats RAG** on reasoning belief-margin **+5.66 vs −0.27** (cold −0.02), 5/6 code items, **through `solve()`** | logit-margin, manually read; RAG≈cold because untyped dump of insight+misconception cancels, the typed +/− resolves |
| 2 | **Slot decomposition lifts SWE resolve** | **SLOT 7/21 vs ONE-SHOT 6/21** on SWE-bench-Lite, **strict superset, zero regressions** | real **swebench Docker** verifier, gold-sanity 5/5 first |
| 3 | **Decomposition is right-for-the-right-reason** | on `astropy-14995` one-shot **gutted the method**; the diagnose-first SLOT wrote the **surgical None-guard** = the gold fix | read both diffs; verifier-confirmed flip |
| 4 | **Self-correcting, inspectable reasoning** | on `astropy-12907` the backtrack re-diagnosis went generic → **near-gold `_coord_matrix(...)`** | raw diagnosis dump |
| 5 | **Self-improving memory** | a solved slot-graph template, reused on a new task, makes the 2nd task solvable where cold fails | `slot_coder --selfimprove` / `--v3` |
| 6 | **A skill-LoRA closes a real capability gap** | SFT-then-RL on a frozen 4B: held-out reward **−0.15 → +0.725**, hallucination **65% → 30%**, **generalizing** | grounded reward (8/8, anti-hack); held-out random tasks (can't memorize) |
| 7 | **The whole stack runs as ONE system** | integrated run, real 4B: chain-correct on **fictional** entities, **every component fired**, nothing decorative | `integrated_run` WiringReport + `--preflight` |

## 4. The live demo (reproducible)
```bash
# the full stack in one solve, with a wiring report proving every piece fired:
V5_LM_TRUST_REMOTE_CODE=1 python -m v5.runtime.integrated_run --preflight   # asserts wiring, exits
V5_LM_TRUST_REMOTE_CODE=1 python -m v5.runtime.integrated_run               # runs + WiringReport

# operator > RAG, through the slot-graph (belief margin):
V5_LM_TRUST_REMOTE_CODE=1 python -m v5.runtime.slot_coder --reason --layer 26

# SWE resolve (Docker verifier): gold-sanity, then SLOT vs ONE-SHOT
python -m v5.graph_grower.swe_verify --gold-sanity --dataset lite --limit 5
python -m v5.graph_grower.swe_verify --predictions artifacts/swe_slot_preds.jsonl --dataset lite --run-id slot

# no-model proofs (engine + wiring), run anywhere:
python -m v5.runtime.slot_coder --test            # 14/14 engine asserts
python -m v5.runtime.integrated_run --selftest    # all components fire (forced-failure path)
python -m v5.runtime.derive_reward                # reward 8/8 (anti-hack)
```
The thesis demo: **edit a graph node → the frozen model's answer changes, with no retraining** —
the reasoning is on screen (the slot values) and editable.

## 5. Novelty (cite, do not claim invention)
The pieces are individually known; the **assembly** is the contribution.
- Truth-maintenance / dependency-directed backtracking (Doyle; de Kleer) — **reused**, not invented.
- Frame / slot-filling reasoning — classic NLP.
- Activation steering / representation editing — crowded; our typed, edge-gated, *non-interchangeable*
  operator algebra (INVALIDATE = subtract) is the specific instance.
- Save/reuse solved structures — Buffer of Thoughts (2024), Voyager skill-library, case-based reasoning.

**The fresh assembly:** an **operator-grounded slot-graph truth-maintenance system, executed through a
FROZEN SMALL LM, whose solved structures are saved back into the same graph that grounds it** — so
capability grows by editing the graph + a tiny skill-LoRA, and the reasoning stays legible/editable.

## 6. Honest limitations (stated, not hidden)
- **SWE walls**: localization recall ~0.30 and the 4B's emission ceiling bound end-to-end resolve;
  the slot-graph structures the problem but doesn't break those walls. The 7/21 is *given oracle
  localization* (synthesis-isolation), not full end-to-end.
- **The skill-LoRA is arithmetic, not coding** — it proves the *training pipeline* (reward→SFT→RL→
  generalizes), not a coding gain. Training on SWE/coding data (verifier reward) is the next step.
- **The integrated run's verifier** is the gold answer on a controllable fictional family, not the
  SWE Docker harness (the `solves_fn`/`grounded_fn` are pluggable for that apex swap).
- **N is small** for the operator margin (6 items) and SWE (21–24); reproduced + manually inspected,
  not large-scale.

## 7. What's next
- Apex: point the RL loop's `solves_fn` at the **SWE Docker verifier** + train on coding derives.
- Attack the localization wall (multi-file retrieval) and the emission wall (worked-exemplar retrieve-adapt).
