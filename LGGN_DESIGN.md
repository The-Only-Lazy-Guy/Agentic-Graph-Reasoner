# LGGN-Coder — Reasoning as Graph Traversal over a Growing Operator Library

**One principle:**
> The graph is a growing library of **typed repair operators**. Traversal **composes** them into a repair program. The frozen LLM only **realizes** operators as syntactically-correct code. Learning is **operator discovery by compressing successful trajectories**.

Not GraphRAG (retrieve→serialize→LLM). The graph is the **execution substrate**, not retrieval storage. Lineage: neurosymbolic **library-learning** (DreamCoder family) with an **LLM as the operator-realization backend** + a **latent graph as the execution substrate**. The research question this answers: *what should exist in latent graph space?* → **a discovered, typed vocabulary of repair transforms** (not files, not strategies-as-text).

This doc is the canonical design. It supersedes the single-vector "floor" (`swe_rl --distill`), which grounded relevance/localization — the wrong axis (see `memory/strong-path-resolve-ceiling`: retrieval has no resolve headroom; the wall is fix-reasoning).

---

## Results scorecard (all measured, 2026-06-30 / 07-01)
| Claim | Result | Status |
|---|---|---|
| Reasoner learns (goal→operator, semantic ops) | **40% SWE / 70% Fable** vs ~chance | VALIDATED |
| Scales with data | held first-op **33→48%** with train size | VALIDATED |
| Compositional manifold — intra-task (fix = Σ own parts) | **0.91** vs 0.26 random, 100% of golds | VALIDATED |
| Compositional manifold — cross-task (fix = Σ library ops) | **0.58** vs 0.38 random, 98% | VALIDATED |
| Topology load-bearing (op-change next-op, CV) | **38%** vs 6% random vs 3% marginal | VALIDATED (self-loops factored out) |
| Open-vocabulary growth (T.8) | Fable mints **+16** ops, 2421 novel edits | VALIDATED |
| POMDP obs-feedback on DEBUG data (issue+failure→op) | **+15pp** (40% vs 25%), failure-only>issue-only | SUPPORTED (n=33; scale to tighten) |
| Encoder is not the wall | 4B-hidden ≈ mpnet, gap≈0 all layers | ruled out |
| Realizer resolve (4B leaf) | **~20%** (6 ways) | OPEN — leaf wall |
| **Solution ladder** (4B, n=30, `solution_ladder.py`) | emit given issue **14%** / +operator-plan **10%** / **+EXACT gold 83%** | the wall is **DERIVATION**, not emission or the plan |

**Honest negatives the kill-tests caught** (right-for-the-right-reason): signature operators were goal-blind (~chance → embedding ops fixed it); raw topology 65% was a self-loop artifact (op-change is the real 38%); Fable-greenfield obs=0% (no failures → domain-mismatch, rescued on SWE debug); retrieval has no resolve headroom (best-of-3=top-1).

**One-line thesis status:** the reasoning substrate — a compositional, growing, topology-load-bearing operator library the frozen LLM realizes — is **validated end-to-end as science**; but the **solution ladder characterized the resolve wall as DERIVATION** (the 4B emits a *handed* fix at 83% and copies fine, but derives the correct fix from the issue only 14%, and **generic operator-plans don't help (−4%)**). So the substrate is the **wrong granularity for resolve**: resolve needs *specific fix content* (derivation), operators deliver *generic structure*. The graph can't retrieve the novel exact fix (no retrieval headroom) nor make the 4B derive it. **Resolve levers now:** deliver near-exact content (not generic ops), or improve 4B derivation directly (SFT/RL), or a stronger leaf — *not* more operator-planning.

**CORRECTION (2026-07-01) — resolve is UNBUILT, not closed.** The HRM lens exposed the gap: HRM iterates *against the puzzle state* (constraints propagate from the grid, in-input); we only ever ran a **one-shot** operator picker + a **frozen** executor. `iterative_refine.py` probe: internal latent iteration on a *static goal-embedding* is FLAT (K=1..8 ≈ 0.39) — EXPECTED, a fixed input has no constraints to propagate; **not a latent-size problem.** What's missing (none is "bigger latent"): (1) an **environment to iterate against** — code + test-feedback (T.9 proved failures carry signal), (2) a **TRAINED refiner** (frozen 4B resubmits the same patch; HRM's net *learns* to refine), (3) the **loop assembled** (`lggn.solve()` still a stub — traverse→realize→verify→refine has never actually run). So the real, untested mechanism = a **trained agentic refinement loop against tests** (HRM's recurrence transposed: iterate against the codebase+tests, not the Sudoku grid). Honest cost: that's RL-against-verifier (hard frontier) + a *trained* refiner (drifts from "frozen small model"). But it was never built → don't claim resolve is closed.

## 0. Why this shape (what the SWE experiments forced)
Measured, on SWE-bench Lite, 4-bit Qwen3.5-4B:
- File-level localization works; **emission applies ~80%**; **resolve ceiling ~20%** with exemplar grounding.
- **Retrieval has zero headroom** — best-of-3 exemplars = top-1; a better exemplar rescues nothing. The exemplar is a **format prior**, not the answer.
- The wall is the **decoder inventing the repair**: wrong-line edits, wrong logic, cosmetic non-fixes, under-scoped multi-part.

Conclusion: grounding the *inputs* can't fix weak *reasoning*. So move the reasoning OUT of the frozen decoder into a **trained traversal over typed operators**, and shrink the decoder's job to **realizing** a chosen operator trajectory.

---

## T. Theory — the latent-memory model (why latent operator-traversal should generalize)
*Tags: [VALIDATED] measured · [BUILT] code exists · [HYPOTHESIS] testable claim · [VISION] not yet built.*

### T.0 Lifetime table (the clarifier — read this first)
| Component | Lifetime | Role |
|---|---|---|
| **Latent state `h_t`** | one reasoning **episode** | transient working reasoning |
| **Session graph** | one **task** | task-local goals / evidence / artifacts |
| **Persistent Latent Memory** (the graph) | **across tasks** | permanent, growing knowledge |
| **Parameters** (LLM + planner) | **fixed after training** | frozen substrate |

**The latent state is transient; the graph is permanent.** Reasoning happens in `h_t`; knowledge accrues in the graph. This one distinction frames the whole architecture.

### T.1 Terminology — *Persistent Latent Memory*, not "operator graph"
The graph is a **Persistent Latent Memory.** **Operators are only its first node type.** It is meant to also hold ideas, proofs, constraints, goals, observations, and programs — each as a node with an **embedding** (its location in the latent manifold), so memory is *latent* and content-addressable. [BUILT: operator nodes. VISION: the richer node types.]

### T.2 Keystone hypothesis — local compositionality of the operator manifold
> **Hypothesis (compositionality).** The semantic operator manifold is *locally compositional*: learned repair operators correspond to approximately **stable directions (displacements)** in latent space, and composing operators corresponds to **composing their latent displacements** (`Δ_A + Δ_B`). Reasoning is a **trajectory through a continuous manifold guided by discrete operator selections.**

This is *why* latent operator-traversal should **generalize** (compose known operators for unseen tasks) and *why the graph is more than retrieval* (the directions **compute**). [HYPOTHESIS]
- **Evidence (partial)** [VALIDATED]: fix-embeddings cluster into stable operators (§4d); the goal predicts the operator (AMI ≈ 0.3; 40% vs 12% chance) → the directions are real and goal-aligned.
- **Falsification test — RUN, keystone SUPPORTED** [VALIDATED]: on n=110 multi-hunk golds, `cos(full_fix, Σ its OWN hunk-displacements) = 0.906±0.05` vs `cos(full_fix, Σ RANDOM hunk-displacements) = 0.259±0.11` — gap **+0.648, own>random in 100% of golds.** The full fix ≈ the **sum of its component displacements**; the manifold is **locally compositional.** The random baseline (0.26) rules out generic text-additivity — the decomposition is *specific* to the fix's own parts. **Cross-task composition — RUN, also SUPPORTED** [VALIDATED]: on the ≥2-distinct-operator golds (n=50), the fix is reconstructed by its **shared LIBRARY operators' centroids** `Σ centroid_op = 0.582` vs random-ops `0.376` (gap +0.206, own>random **98%**) — weaker than intra-task (lossy centroid averages) but strongly significant → **library operators compose *across* tasks.** Both levels confirm the keystone: intra-task `fix = Σ its parts` (0.91 vs 0.26, 100%); cross-task `fix ≈ Σ its library operators` (0.58 vs 0.38, 98%). Local compositionality — *the reason latent traversal should generalize* — is empirically real at both the task and library level.

### T.3 Latent memory IN the loop (not preprocessing retrieval)
Memory must **modify the latent state**, not merely precede it:
```
m_t = CrossAttention(h_t, G)          # consult the Persistent Latent Memory
h_{t+1} = F(h_t, Δ_t, m_t)            # memory enters the latent computation every step
```
The graph becomes part of the latent computation; retrieval stops being a preprocessing step. [BUILT: `GraphMemoryKV`, `cross_attention.py`, `adapter.py` — the machinery exists.]
**Honest reconciliation (empirical):** we measured that **untrained** cross-attn injection **hurt** — but at the **decode leaf** (the realizer). The fix: `m_t` cross-attention must be (a) **trained**, (b) applied to the **planner state `h_t`** (the traverse), **not** the realizer decode. Your formulation targets the planner — exactly right; our failed attempt was untrained + at the wrong site. Reconciled.

### T.4 Capacity scaling — `capacity ≈ parameters + persistent graph`
Parametric capacity is **fixed** after training. The graph adds **non-parametric capacity that grows with experience** — every solved task mints operators/observations. So capability rises with **tasks seen**, not only parameters: a different scaling law (LLM-capacity = params; LGGN-capacity = params + memory). [HYPOTHESIS]
- **Test = V6 (write-back compounding):** does solving task A measurably help a later similar B? If yes, the added graph-capacity is real and the scaling law holds.

### T.5 Traversal policy `π(Δ | h_t, G)`
The planner is an explicit **policy** over operators (+STOP), conditioned on the latent state **and** memory:
- **v1 [BUILT]:** learned **softmax** policy (the GRU head), greedy/sampled decode.
- **v2 [VISION]:** **best-first / beam** over the operator-graph **transitions** (the co-occurrence structure = a learned search prior).
- **v3 [VISION]:** **RL (RLVR)** — reward = resolve; the discrete operator action-space makes RL tractable.
- The graph **topology is the policy prior** → **V7 kill-test [RUN 2026-06-30, `topology_test.py`, mined 37 Fable trajectories, mean 33.5 steps]:**
  - **Naive raw transitions:** learned 65% vs random 1% — *but the kill-test caught it as an ARTIFACT*: 64% of transitions are **self-loops** (op→same op = trivial iteration); learned-on-CROSS (op→different op) = **1%** (below chance) → raw verdict **DECORATIVE beyond self-loops**.
  - **Real test (collapse self-loops → op-CHANGES, 5-fold CV):** learned **38%±5%** vs random **6%** vs marginal **3%** (n_cross=374) → **LOAD-BEARING**: when the operator *changes*, the next operator is predictable 6× over random topology, 12× over marginal. The graph's edges carry genuine sequencing info — *masked* by self-loops in the raw view, revealed by collapsing.
  - **So topology has two layers:** (1) strong self-loops (iterate the same operator — trivial), (2) a real op-change transition structure (the planning signal). Coarsening to 8 buckets *destroys* it (over-collapse, n=32 noise) — fine ops are right for transitions. Caveat: modest data (374 op-changes / 37 sessions). SWE co-occurrence precursor agreed (learned 44% vs marginal 16%, n=32).

### T.6 Goal **regions**, not goal nodes
The planner seeks a **region**, not a node — a neighborhood in the operator manifold (`Programming → Concurrency → Transactions → atomic-fix`). This yields **hierarchy / subgoals / abstraction without inventing every subgoal**: the manifold's natural **coarse→fine** structure *is* the region hierarchy (we already have it — `coarse_bucket` → fine embedding-cluster = a 2-level hierarchy). The traverse **descends**: pick the region (coarse), refine to the operator (fine). [coarse/fine BUILT; the descent policy VISION.]

### T.7 Nodes are **programs** (executable), not labels
An operator node is a typed **program**:
- `precondition` — when it applies (soft, embedding-matched) [BUILT]
- `execution` — `realize_hint` → the frozen LLM realizes it as code [BUILT]
- `expected_effect` — **the cluster centroid IS the fix-direction in embedding space** (we already have this — the operator's centroid is its expected latent effect) [VALIDATED]
- `failure_modes` — observed failures → INVALIDATE, from write-back [VISION]

Then traversal is **execute → execute → execute** — a planning system (STRIPS/PDDL-style operators with preconditions + effects), not move → move. Neurosymbolic planning with an LLM realizer.

### T.8 Operator assignment & library GROWTH (open-vocabulary, soft, residual-driven)
Hard `edit → nearest centroid → operator` is **bootstrap only** — it forces novelty into existing atoms (loses discovery) and locks early clustering errors permanently. The principled form is **dictionary learning / sparse coding** over the operator library:
- An edit = a **sparse combination** of operators **+ a residual**: `e ≈ Σ wᵢ·centroidᵢ + r`.
- **Reuse** if the edit is within `τ` of the operator manifold (residual small). **Grow** if not: a large residual `r` *is a candidate new operator atom* `D` (e.g. `e ≈ 0.6·A + 0.4·D`, `D` minted from the residual). So **Fable actually grows the library** instead of collapsing into the SWE-seeded vocab. [VISION; current `chunk_embedding` = the bootstrap hard-assign.]
- **Preserve uncertainty** — don't collapse `{A:0.62, B:0.59}` to argmax; store the **soft distribution** as a *candidate* and let later **trajectory optimization** (write-back, which operators actually recur across solved trajectories) resolve it. Early-clustering errors stay reversible. This is the operator-level retrieve-or-derive (`lggn.py` `OperatorLibrary` already has the conf/match/mint gates — wire the soft+residual path through it).

### T.9 Training data is POMDP-shaped (preserve observations, don't flatten)
The runtime is a **POMDP** (§2: traverse → decode → **OBSERVE** → update → write-back). The training data must match it. Fable carries what SWE golds largely don't: **temporal structure** — `Op → Observation → Revision → Op`, not just `Op → Op`. **Preserve it:**
```
Goal → Op₁ → Obs₁ → Op₂ → Obs₂ → …        (NOT flattened to Goal → [Op₁,Op₂,…])
```
The planner trains as a **policy `π(Δ_t | h_t, obs_{<t})`** where the observation **enters the latent state** (`h_{t+1}=F(h_t, Δ_t, m_t, obs_t)`, T.3) — a failed/observed result is a real INVALIDATE+re-plan signal, the gold supervision for the debug loop. Flattening discards exactly the signal the planner most needs. **[MEASURED 2026-06-30, n=300 cached sessions: Fable is edit-SPARSE — median 0 edits/session, mean 1.0, only 2% multi-edit; ~0.5 edits/session full-set. So the POMDP `Op→Obs→Op` data is a ~2% MINORITY (the heavy sessions), NOT the bulk. Fable's main value = ~2539 `goal→single-op` pairs (data scale for the goal→operator curve); the multi-step/POMDP signal is a smaller SECONDARY feed. Don't over-build the POMDP planner around thin data — mine the heavy sessions for it separately.]**

**Fable-feed RESULT (2026-06-30, 2539 records, grown vocab 40 = 24 SWE-seed + 16 Fable-minted, 2421/2539 edits novel):** goal→operator — **Fable-held 70%** (majority 12%, 5.8×) vs **SWE-held 28%** (majority 5%). Fable is the traverse's STRONG domain (intent-stated → operator far more predictable than SWE symptom-issues), vindicating the agentic-coder framing. **But NO cross-domain transfer:** SWE+Fable→SWE-held = 17% < SWE-only 28% (mixing HURTS, same lesson as the realizer Fable-mix) → **specialize per domain, don't blend.** T.8 `grow_library` confirmed (Fable mints its own 16 feature-building operators). Caveat: 70% partly the "easy" regime (intent states the op); the valuable multi-step regime is Fable-edit-sparse.

**POMDP policy ablation (2026-06-30, `pomdp_policy.py`, mined 37 trajectories, grouped 5-fold CV) — the architecture ASSEMBLED, next-op prediction:**
| feature | raw (n=1202) | op-changes (n=374) |
|---|---|---|
| goal-only | 28% | 20% |
| **topology (prev-op)** | **49%** | 37% |
| goal+prev | 45% | 40% |
| goal+prev+**OBS** | 44% | 41% |

- **Topology is the engine** (best single predictor); **goal** adds a little (op-changes 37→40); **OBS adds ~0%** (T.9 observation-feedback UNSUPPORTED here).
- **Why:** Fable is greenfield *build* (success-heavy) — failure-obs = only **2%** of transitions, and failures *don't even redirect* (15% op-change after failure vs 31% baseline). The observe→revise loop is **absent in greenfield data** → it needs **failure-rich** data (SWE-agent runs w/ test feedback), not Fable. *Not a refutation of T.9 — a data-domain mismatch (build ≠ debug).*
- So the validated planner = **topology-driven, goal-conditioned operator planning**; the observation-feedback (debug) loop stays designed-but-untested pending failure-rich trajectories.

**Fork A — obs-feedback ON FAILURE-RICH DATA (2026-07-01, `capture_failures.py` + `obs_signal_test.py`):** captured real FAIL_TO_PASS failures on 40 SWE-Lite instances (harness on unpatched repos, no-op patch; 38/38 this run). Does the test-failure help predict the gold operator? **issue+failure 40% vs issue-only 25% (+15pp), failure-only 30% (n=33, majority 24%) → T.9 SUPPORTED on debug data.** The observation carries fix-operator signal where FAILURES exist — the exact opposite of Fable greenfield (obs=0%), confirming the **domain-mismatch**: the POMDP observe→revise loop works on **bug-fixing** (the architecture's target), not greenfield build. `failure-only > issue-only` = the test-failure localizes the operator better than the issue text. Caveat: n=33, wide bands (±14-19%) → suggestive + domain-contrast decisive, but scale to n~100 (capture is now durable/resumable) to tighten.

---

## 1. The two graphs

**Persistent Operator Graph (long-term memory, grows across tasks).** Nodes are **operators** (typed transforms), plus code-knowledge nodes (`symbol`/`module`, with `calls`/`same_class`/`contains` edges) used for localization/preconditions. **No artifacts stored** — patches are outputs, never persisted; only their *compressed trajectory* (→ operator) persists. Keeps the graph knowledge-only and bounded.

**Session Graph (working memory, per task, transient).** Nodes: `goal` (a (sub)goal), `evidence` (knowledge bound to this task), `observation` (test/exec result). Artifacts live here transiently and are discarded after compression.

### Operator schema (every operator carries uncertainty — load-bearing, not cosmetic)
```
op_id, name                  # e.g. GuardType, FixInPlaceMutation
input_type, output_type      # soft types (memoryview→bytes, possibly-None→checked)
precondition                 # when it applies (soft: NL + embedding + learned matcher)
realize_hint                 # how the decoder emits it (compiler-mode instruction)
embedding                    # for soft retrieval
confidence                   # 0..1
source                       # seed | derived:<instance_id>
age                          # tasks since created
validation_count             # times reuse actually resolved
```
**Why uncertainty is mandatory:** without it, `wrong primitive → write-back → retrieved again → reinforced` is catastrophic. Operators below a confidence/validation floor are **not retrievable for reuse** — only for re-validation. Write-back can only *strengthen* an operator after an outcome-confirmed pass.

---

## 2. The loop (POMDP-shaped: plan → act → observe → belief-update)

```
INGEST    issue → root goal G0, latent h0
RETRIEVE  query operator graph by precondition/type match (conf+val gated)
loop until G0.GATE = solved or budget:
  TRAVERSE (latent): GNN/operators compose an OPERATOR TRAJECTORY toward the goal
        moves are typed ops:  SLOT (open subgoal) · ASSERT (bind evidence) ·
        INVALIDATE (subtract a failed approach) · MERGE · GATE (decode-ready / solved)
        a subgoal not covered by any operator → DERIVE branch (below)
  FREEZE → DECODE (tokens): the decoder REALIZES the trajectory as code (compiler mode).
        Input to decoder = "here is the chosen sequence of transforms + site + constraints",
        NOT "here is raw evidence, infer the repair".
  OBSERVE: run tests / self-check → observation node
  UPDATE (debugging in graph space):
        pass → GATE subgoal solved; continue traversal
        fail → observation becomes an INVALIDATE op on the failed approach; re-traverse
               (the failure changes the latent goal → don't repeat the same patch)
ASSEMBLE leaf artifacts → final patch
WRITE-BACK: compress (goal → op-seq → obs-seq → success) into a new/strengthened operator
```

### retrieve-or-derive **at the operator level** (resolves "decoder does too much")
- **Known repair** → compose existing operators → decoder *realizes* (reliable, cheap).
- **Novel repair** → no operator covers it → decoder *derives* the fix (today's hard path) → if it passes, **compress into a NEW operator**. Next time it's reuse.

So "decoder invents the repair" is the **exception that triggers learning**, not the steady state. The library grows; the decoder's burden **shrinks over tasks**. Day-1 the library is mostly empty (all derive); it fills.

---

## 3. How the knowledge graph ties in (three touchpoints)
1. **RETRIEVE** — traversal *reads* operators (by precondition/type) + code-knowledge (localization). The graph is the memory.
2. **TRAVERSE** — operators *are* the moves; the GNN message-passes over real edges to compose/select them. Topology drives reasoning (trained on outcome → not decorative).
3. **WRITE-BACK** — solved trajectories *become* operators. The graph evolves; reasoning compounds. Retrieve-or-derive made literal.

---

## 4. Training (what makes the graph functional, not decorative)
Decoder = frozen 4B + LoRA (realizer). The **traversal + operator selection + goal-update** is trained.

- **Decomposition is LOGICAL, not syntactic.** Two hunks ≠ two reasoning steps. We do **not** supervise decomposition from hunks directly. Instead the **operator trajectory IS the logical decomposition** (one op = one logical step), and logical units are **discovered by compression**: a sub-sequence recurring across tasks *becomes* an operator. "6 hunks → 1 conceptual repair" = 6 hunks compress to 1 operator; "1 hunk → 5 decisions" = 1 hunk expands to a 5-op trajectory. Hunks bootstrap the realizer weakly; the library corrects syntactic→logical automatically.
- **Realizer SFT (weak bootstrap):** train LoRA to emit code given (operator trajectory + site). Seed targets from **SWE gold** (SR format). NOT Fable (format clash — mixing it hurt held 27→20%).
- **Traverse SFT (the reasoning bootstrap — from FABLE):** train the planner to predict the **operator trajectory** from the goal, supervised by **Fable-5 sessions** (goal → CoT → edit-sequence; mine each edit → operator). This is *general reasoning/decomposition* (breadth), decoupled from bug-fix emission. SWE provides the verifiable outcome reward; Fable provides the supervised "how to plan." Then outcome-RL refines. Clean split: **reason (Fable) / realize (SWE) / grade (verifier).**
- **Traversal RL (GRPO/RLVR):** reward = resolve × efficiency (fewer ops / reuse) × write-back-reusability (a stored op that helps a later task). Outcome-supervised.
- **Operator discovery** (the crux, §6): how a recurring transform gets named, typed, confidence-scored, made retrievable.
- **Topology kill-test:** trained-edge traversal vs random-edge — learned topology must win, else the graph is decorative.

---

## 4b. Observability — the traversal must be plottable (or it isn't real)
A genuine latent traversal is **visualizable**, and the plot doubles as the V7 kill-test:
- **Operator-space graph** (`operator_discovery.visualize`, built): discovered operators = nodes (size ∝ freq), edges = consecutive-operator transitions across the corpus's gold trajectories → `artifacts/train_plots/operator_space.png`. This is the graph the traverse moves over, emerged self-supervised from data.
- **Model traversal overlay** (hook ready, `model_trajs=`): the traverse's actual chosen op-sequences drawn in red. **A LEARNED traverse hugs the high-frequency gold edges; a decorative one scatters** — V7 made visual.
- **Latent path** (when the latent traverse exists): `h_t` states in 2D (PCA) with `Δ_t` operator-moves as arrows — the literal "navigation in latent space."
If the traversal can't be plotted as a structured path, it isn't traversing — it's noise. Observability is a requirement, not a nicety.

## 4c. The TRAVERSE (HRM-templated planner) — spec
The reasoner: `goal → operator trajectory`. The frozen LLM realizes each operator; the verifier grades.
Cousin of HRM (latent hierarchical reasoning, no CoT) — but over a **frozen LLM** + a **growing,
inspectable operator library**, for open-ended coding. HRM's gains were traced to its refinement loop,
not its hierarchy → we **kill-test the structure** (V7), which HRM didn't.

**Architecture (v1, feedforward-autoregressive):**
- **Goal encoder (frozen):** RealEmbedder(issue + localized support) → `g` (768-d). No training cost.
- **Operator decoder (learnable, small):** autoregressive over the **discovered operator vocab** (~56
  ops + BOS/STOP), conditioned on `g`. At step t: `(g, op_1..op_{t-1}) → softmax(op_t)`. Tiny output
  space → small, fast, HRM-like efficient. Operators are *discrete latent tokens* (reasoning in
  operator-space, not text-token space).

**Training — SFT on `(goal → operator trajectory)`:**
- corpus = **SWE** chunked golds (verifiable, mostly short) **+ Fable** chunked sessions (long
  multi-edit = the multi-step decomposition SWE lacks). Operators abstract format → SWE+Fable unify
  with NO format clash (the reason Fable belongs HERE, not the realizer — proven by the realizer
  negative, held 27→20). Self-supervised labels via `operator_discovery.chunk`.

**Metrics (decoupled from the 4B resolve wall):**
- **trajectory-prediction accuracy on HELD goals** (op-level P/R, exact-traj match) — does the reasoner
  *generalize* to unseen tasks? This validates the reasoning claim independent of leaf capacity.

**V7 topology kill-test (the HRM differentiator):** learned operator-transition structure (graph /
co-occurrence prior over ops) vs **random** vs **none**. If learned beats random → the graph *reasons*;
if it ties → decorative (the exact ablation HRM skipped).

**v2 (ablation, only if v1 generalizes):** add HRM-style latent H/L recurrence (K refinement passes
before readout). Ablate K=1 (feedforward) vs K>1 — *does recurrence help?* (HRM's contested claim.)

**Observability:** capture the decoder hidden state at each emitted op → PCA → the **real latent-path
viz** (`operator_discovery.visualize_embedding(model_states=...)`). The model's actual `h_t` path,
not the operator-embedding proxy.

**Compose (end-to-end, needs a non-4B leaf to be unconfounded):** traverse → realizer (stronger leaf) →
verify. Resolve is the *downstream* metric; trajectory-accuracy + V7 are the *reasoning* metrics.

**Build order:** (1) traverse corpus `(goal→trajectory)` from SWE+Fable; (2) TraversePlanner + SFT,
measure held trajectory-accuracy; (3) V7 kill-test + latent-path viz; (4) compose w/ stronger leaf.

## 4d. Latent-state reasoning (the substrate — the core idea)
The traversal is **not** symbol-shuffling — it is a **path through latent state**. The reasoner holds a
latent state `h_t` (the planner's hidden state). Each step applies a **typed operator as a move toward
the goal region**: conceptually `h_{t+1} = h_t + Δ_op`, where **operators are SEMANTIC LATENT
DIRECTIONS** in the fix-embedding manifold.

**This is now VALIDATED, and it's why the carving matters:** operators discovered as **fix-EMBEDDING
clusters** are goal-predictable — `AMI(goal, fix-embedding clusters) ≈ +0.3` vs shuffled ~0; goal→operator
**40% vs 12% chance** (`traverse --emb-ops`). The earlier **surface structural-signature** operators were
**goal-blind** (~chance) — a path through the *wrong* latent space. So the latent state must live in the
**semantic operator manifold**, where the goal *does* determine the fix-direction. The reasoner scales
with data on this manifold (held first-op 33%→48% with train size) — flat on the signature manifold.

**FREEZE → DECODE → RESUME:** at a decode-ready state, **freeze** the traversal and **decode** the
operator into code via the frozen LLM realizer (the leaf — *language is only the output interface*).
**Observe** (tests). The observation **updates the latent goal** (a new Δ; a failure = INVALIDATE), and
the traversal **resumes with the changed goal** — *reasoning continues in latent space between
token-decodes*, not by re-serializing text. This is the §2 POMDP loop realized as **latent-state
evolution**.

**Why latent (HRM-aligned):** reasoning lives in `h_t` (cheap, no CoT tokens); language is emitted only
at leaves. **Observability (§4b):** `h_t` is plottable — a *real* traversal is a structured path through
the operator manifold; a *decorative* one scatters (V7, made visible). Operators are the discrete latent
vocabulary the path is built from; discovery (V5) grows that vocabulary; write-back (V6) compounds it.

## 5. Component map — reuse, don't reinvent
| Role | Existing piece |
|---|---|
| Persistent graph + edits | `graph_core.MemoryGraph`, `reasoning/graph_editor` (provenance/validated) |
| Session subgraph + KV-inject | `subgraph.build_active_subgraph`, `GraphMemoryKV`, `runtime/prefix_session` |
| Typed operators (Δ) | `operator_injector` (ASSERT/INVALIDATE/GATE/SLOT — proven algebra) |
| Traversal net (train it) | `gnn_encoder.RGCNEncoder` |
| Decompose/fixpoint/backtrack control | `runtime/slot_coder.SlotGraph.solve` |
| Decode leaf (realizer) | frozen 4B + LoRA + KV-inject |
| Observe | `graph_grower/swe_verify` (Docker) |
| Retrieve (localization) | `graph_grower/train_retriever` |
| Train traversal | `runtime/derive_rl` / GRPO |
| Operator library (NEW) | `runtime/lggn.py` |

---

## 6. The crux (publishable-vs-engineering): operator discovery
Everything hinges on the **operator vocabulary**:
- too **coarse** (ASSERT/INVALIDATE/GATE) → trajectory doesn't specify the repair → decoder still invents → back to today.
- too **fine / hand-coded** → doesn't scale past a demo.

So operators must be **discovered, typed, and confidence-scored automatically** from solved trajectories. **That discovery mechanism is the paper.** Put the research risk here, not in loop plumbing. Open sub-questions: how to *name/cluster* a recurring transform; what a *soft precondition/type* is in code (likely embedding + learned matcher, not hard logic); how to *merge/retire* operators (age, validation).

---

## 7. Honest risks
- **Operator discovery** (§6) — unproven; make-or-break.
- **Realization is still code-gen** — compiler mode lowers burden but a leaf op still emits tokens; the 4B can miss. Decomposition raises the floor, doesn't remove it.
- **Soft symbolic boundary** — "precondition/type" in code is fuzzy; necessary softness blurs the clean "compiler" story.
- **Write-back poison** — mitigated by confidence/validation gating (§1), but needs the outcome-confirmation discipline.
- **Leaf still 4B-bound** — decomposition shrinks leaves so the 4B succeeds more; a still-too-hard leaf fails.

---

## 8. Build order (skeleton-first — the whole shape thin, then deepen)
1. **Operator library + seed vocab** (`lggn.py`): schema w/ uncertainty, retrieve-by-precondition (conf/val gated), add_or_strengthen, compress_trajectory. Selftest (no model). ← **step 1, here.**
2. **Thin end-to-end loop**: retrieve ops → compose trajectory (hand-coded) → decoder *realizes* → test → compress. Hand-coded traverse.
3. **Measure the core bet:** does "realize a chosen operator trajectory" beat "decode from raw evidence" on the same tasks? (tests §2/§"decoder-as-realizer".)
4. Train the realizer (SFT on gold) → train traverse (operator selection) → add UPDATE/debugging → GRPO.
5. **Operator discovery** loop (derive→compress→type→confidence). ← the crux.
6. Make traverse **latent** (the efficiency form) once the token loop beats one-shot.

Each step yields a resolve number, so no step is taken on faith.

---

## Scorecard (self-assessment, to beat)
Originality 9 · Coherence 8.5 · Feasibility 7.5 · Incremental-buildability 9.5 · Risk High.
Risk concentrated in §6 (operator discovery). Strengthen by making the graph's output **procedural** (an operator trajectory the decoder *realizes*), not evidence the decoder *interprets*.

-------

# Progress log + insights — V1/V2 verification (2026-06-29)

## Built (committed, selftested)
- `lggn.py` — operator library (typed ops + confidence/source/age/validation; retrieve gated by conf+val+match; strengthen/weaken; compress_trajectory), 11 seed ops from the SWE taxonomy, retrieve-or-derive `solve()`. Selftest PASS.
- `swe_slot --lggn-realize` — decoder realizes an operator trajectory via `fix_user(plan=...)` (A/B vs decode-from-evidence exemplar). `--lggn-multileaf` — real decomposition: decode+apply each operator as a SEPARATE leaf against the live tree (`_multileaf_patch`).
- `data/swe/oracle_ops.jsonl` — hand-labeled op/trajectory per held instance (oracle upper bound).
- `v5/training/ingest_fable5.py` — probe-first Fable-5 ingester (AGPL-3.0). **CORRECTION: Fable feeds the TRAVERSE (reasoning), NOT the realizer.** Mixing Fable into the realizer SFT *hurt* held (27→20%): Fable = feature-building whole-file writes (different task + output format than SWE SR) + noisy intent → wrong axis for leaf emission. Fable's value is its **reasoning trajectories** (goal → CoT → edit-sequence) = supervised `goal → operator-trajectory` for the traverse. Reasoning trained broadly (Fable) → applied narrowly (SWE realizer + verifier).

## Results (7 held SWE-Lite, Docker-verified, leakage-checked)
- **Every grounding form ties at 4/7**: exemplar, best-of-3 retrieval, single-op realize, multi-op trajectory. Scale-30 ≈ 20%.
- **V1 (decoder-as-realizer): MECHANISM PASS.** The decoder *faithfully* realizes a handed operator (read: `transaction.atomic` / `output[:]=` / empty-guard emitted as specified). No resolve lift vs exemplar.
- **V2 (decomposition): MECHANISM PASS, resolve fail on the hard case.** Multi-leaf apply-between **fixed applyability** (11283's cumulative patch applies, where the 3-in-1-prompt was non-applyable). But 11283 resolve=0; patch read shows: AddImport emitted the WRONG symbol (`Prefetch`, not `transaction`/`IntegrityError` → caught `IntegrityError` unimported → NameError), no `with transaction.atomic()`, missing the WARNING/body.

## The wall (manually read, not inferred)
The 4/7 ceiling is **LEAF REALIZATION CORRECTNESS**, decomposed into two causes:
1. **Under-specified operators (FIXABLE):** operators are NAMES without OPERANDS — "AddImport" didn't say *what* to import, so the 4B guessed wrong. Fix = **parameterize operators** (operands + a realize prompt that fills them) = the decoder-as-compiler division of labor. Operator discovery must mine operands, not bare names.
2. **Leaf capacity (4B ceiling, real):** 11283 is a *feature* (4 imports + WARNING + atomic + except + print) — the hardest case. The 4B realizes EASY operators correctly (resolved 6938/11039/11133) and fails compound ones, even decomposed.

NOT the wall: localization (file-level works), emission (80% applyable), retrieval (no headroom — best-of-3=top-1), decomposition mechanism (multi-leaf applies).

## Corrections logged
- "Slot engine 0/10" was **WRONG** — every run used `--oneshot-only` (skips the slot/kv solve by design → 0 attempts). The decompose path was *disabled*, never tested.
- swebench harness crashes on report-gen for some instances (flask + certain django commits: `requirements.txt` path). Eval still runs; parse `logs/run_evaluation/<run_id>/*/report.json` per-instance when the aggregate raises.

## Future insights / next
1. **Parameterize operators** (operands in the schema + a realize prompt that fills them) → deterministic realization (compiler mode); fixes the AddImport-guessed-Prefetch class. **Re-run 11283 + a moderate multi-part.**
2. **Test decomposition on a MODERATE multi-part** (2 small correct leaves), not 11283 (adversarial feature) — separates operator-under-specification / 4B-capacity / 11283-too-hard.
3. **The 4B leaf is the resolve ceiling on hard cases.** To raise *absolute* resolve, a stronger LEAF model is the lever — the graph/LGGN earns its place as the reasoning/decomposition/efficiency substrate, NOT as a manufacturer of leaf-correctness the 4B lacks.
4. **Untested rungs:** V3 (rich debugging loop observe→INVALIDATE→re-traverse), V4 (operator SELECTION, drop oracle), V5 (operator DISCOVERY — the paper), V6 (write-back compounding + poison gate), V7 (topology kill-test), V8 (latent efficiency).
5. **Publishability hinges on V5/V6/V7** (discovery / compounding / topology) passing with controls. V1/V2 are validated necessary engineering, not the novelty.

-------

# Repository map (every important file + status)

## Core LGGN pipeline (built this arc)
| File | Lines | Role | Key API / flags | Status |
|---|---|---|---|---|
| `LGGN_DESIGN.md` | 303 | canonical design + Theory (§T) + this map | — | living |
| `v5/runtime/lggn.py` | 370 | **operator LIBRARY** | `Operator`, `OperatorLibrary` (retrieve/strengthen/weaken/`compress_trajectory`; conf+val+match gating = poison gate), 11-op seed vocab, `label_gold` (regex miner, SUPERSEDED), `render_op_program`, `solve()` retrieve-or-derive loop | built + selftest |
| `v5/runtime/operator_discovery.py` | 407 | **self-supervised DISCOVERY (V5)** | `signature()`, `discover()` (signature clusters), **`discover_embedding()`/`chunk_embedding()` = CANONICAL** (semantic fix-embedding clusters + centroids), `visualize()` (operator-space graph), `visualize_embedding()` (latent-traversal 2D PCA); CLI `--discover [--embedding --k]` | built + selftest |
| `v5/runtime/traverse.py` | 274 | **the REASONER (planner)** | `build_corpus_swe[_emb]`, `TraversePlanner` (GRU goal->operator), `train_traverse` (held traj-acc vs majority), `coarse_bucket`; CLI `--train/--curve/--emb-ops/--selftest` | **VALIDATED 40% vs 12%** |
| `v5/runtime/swe_rl.py` | 1180 | **the TRAINER** (SFT->GRPO->distill, LoRA) | `--realizer` (operator-plan->gold SFT), `--discovered-ops`, `--fable-corpus/--fable-frac`, `--distill`, `--plots-dir` (`_save_train_plots`), `--rep-penalty`, `--eff-coef`, `--use-exemplar`, `--staged`, `--graph` | built; realizer proxy-up resolve-flat |
| `v5/runtime/swe_slot.py` | 2121 | **the INFERENCE ENGINE** | oneshot/slot/kv solve; `--exemplar`(+`--exemplar-rank`), `--lggn-realize`(+`--lggn-multileaf`), `--test-feedback`, `--exact-verify`; `fix_user(plan=)`, `_multileaf_patch`, `_repair_sr_to_src` | built |
| `v5/training/ingest_fable5.py` | 226 | **Fable-5 ingestion** | per-session event streams -> (intent,edit) records; `--probe/--ingest/--selftest` | built (4781->2539 recs) |
| `v5/graph_grower/swe_verify.py` | 323 | **Docker SWE verifier** | gold-sanity gate, resolve; `--predictions --backend docker` | reused |

## Data / artifacts
- `data/swe/discovered_ops_emb.jsonl` - **CANONICAL operators** (24 semantic clusters + centroids).
- `data/swe/discovered_ops.jsonl` - signature ops (SUPERSEDED by embedding).
- `data/swe/oracle_ops.jsonl` - 7 hand-labeled ops (the A/B upper-bound set).
- `artifacts/train_plots/` - `training_dynamics.png` (SFT/GRPO/eval, split by source), `operator_space.png` (op transition graph), `operator_embedding.png` (latent-traversal 2D PCA), `metrics.json`.

## Supporting (reused, pre-existing — relevant to Theory §T)
- `v5/training/providers.py` (90) - `RealEmbedder` (mpnet, the goal/fix encoder), `FrozenQwenHInitProvider` (4B-hidden encoder, used in the separability probe).
- `v5/operator_injector.py` (171) - operator algebra (ASSERT/INVALIDATE/GATE/SLOT = signed latent Delta).
- `v5/subgraph.py` (210) - `build_active_subgraph`, `GraphMemoryKV` (the **latent-memory machinery for T.3**, `m_t=CrossAttn(h_t,G)`).
- `v5/adapter.py`, `v5/cross_attention.py`, `v5/gnn_encoder.py`, `v5/runtime/prefix_session.py` - cross-attn / GNN / KV-prefix (T.3 components; untrained cross-attn measured to hurt at decode -> train + at planner state).
- `v5/graph_grower/swe_load.py` (183) - `load_instances`, `checkout_repo`. `v5/lm_loader.py` (97) - `load_frozen_lm` (4-bit).

# Full progress log (chronological, with results)

1. **Pivot to LGGN** (from the single-vector distill floor, which grounded relevance = the no-headroom axis). Design: graph = growing operator library; LLM realizes; learning = operator discovery.
2. **Operator library skeleton** (`lggn.py`) - selftest PASS (poison gate, retrieve-or-derive, mint-on-novel).
3. **V1 decoder-as-realizer** (`swe_slot --lggn-realize`): A/B realize vs exemplar = **4/7 tie**, but manual read = decoder FAITHFULLY realizes a handed operator (mechanism PASS). Tie confounded by 1 mislabel + multi-part.
4. **V2 decomposition** (`--lggn-multileaf`): apply-between FIXED applyability on 11283; resolve still 0 (leaf capacity + under-specified ops). Mechanism PASS.
5. **Self-supervised discovery** (`operator_discovery`, signature): 56 ops, 79% coverage (vs regex 59%). + dual visualizations.
6. **Realizer training** (`swe_rl --realizer --discovered-ops`): proxy lifted (applyable 20->33%, gold-solve 7->13%) but **Docker resolve 1/5 = ~20% ceiling** (4B leaf-bound).
7. **Fable**: ingested (4781 sessions -> 2539 recs). Mixed into realizer -> **HURT held 27->20%** (format clash). CORRECTION (user): Fable -> TRAVERSE (reasoning), not realizer.
8. **Traverse built** (`traverse.py`, HRM-templated, frozen-embed + GRU; trains WITHOUT the 4B). On SWE signature-ops: **flat ~chance** (issue 4%, issue+code 2%, coarse 20% vs 10%; learning curve FLAT). Diagnosed "hard objective."
9. **Encoder probe**: 4B hidden states ALL layers ALSO gap~0 -> NOT the encoder.
10. **USER INSIGHT "the graph poisons it"** -> MI test: **AMI(goal, fix-EMBEDDING clusters) = +0.3 vs shuffled ~0** -> structure EXISTS; signature-ops were goal-blind carving. Goal->embedding-op: **LogReg/kNN 42-43%**; end-to-end GRU traverse **40% vs 12% (3.4x)** -> **the reasoner LEARNS.**
11. **Canonical embedding discovery** + **learning curve SCALES** (33->48% with data, vs flat signatures) -> Fable scale justified.
12. **Theory §T** (5 gaps integrated) + **KEYSTONE compositionality test RUN -> SUPPORTED** (n=110: own-displacement-sum 0.906 vs random 0.259, gap +0.648, 100%). Manifold is locally compositional.

## Current verdicts
- **Reasoner (traverse):** LEARNS (40%, scales) - VALIDATED. The architecture's reasoning claim is alive.
- **Compositionality (the "why"):** intra-task VALIDATED; cross-task = next test.
- **Realizer resolve:** ~20% on 4-bit 4B - leaf-capacity wall (separate; needs a stronger leaf, not graph tricks).
- **Operators:** semantic embedding clusters, canonical, 100% coverage.
- **Open builds:** latent-memory cross-attn into planner (T.3, trained), policy beam/RL (T.5), goal regions (T.6), node-programs failure_modes (T.7), V6 compounding, V7 topology kill-test, Fable->trajectory feed.

## Next (recommended order)
1. **Cross-task compositionality** probe (`centroid_A+centroid_B ~= unseen fix using both`) - airtight the keystone.
2. **Fable -> trajectory feed** + re-curve (scale the reasoner on the agentic-coder domain).
3. **V7 topology kill-test** (graph vs flat policy) + **real latent-path viz** (planner h_t).
