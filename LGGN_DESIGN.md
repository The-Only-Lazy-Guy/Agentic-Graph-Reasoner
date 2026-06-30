# LGGN-Coder — Reasoning as Graph Traversal over a Growing Operator Library

**One principle:**
> The graph is a growing library of **typed repair operators**. Traversal **composes** them into a repair program. The frozen LLM only **realizes** operators as syntactically-correct code. Learning is **operator discovery by compressing successful trajectories**.

Not GraphRAG (retrieve→serialize→LLM). The graph is the **execution substrate**, not retrieval storage. Lineage: neurosymbolic **library-learning** (DreamCoder family) with an **LLM as the operator-realization backend** + a **latent graph as the execution substrate**. The research question this answers: *what should exist in latent graph space?* → **a discovered, typed vocabulary of repair transforms** (not files, not strategies-as-text).

This doc is the canonical design. It supersedes the single-vector "floor" (`swe_rl --distill`), which grounded relevance/localization — the wrong axis (see `memory/strong-path-resolve-ceiling`: retrieval has no resolve headroom; the wall is fix-reasoning).

---

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
- **Falsification test (must-run, concrete):** on multi-hunk golds, embed `Δ(fix₁)`, `Δ(fix₂)`, and `Δ(fix₁∘fix₂)`. If `Δ(fix₁)+Δ(fix₂) ≈ Δ(fix₁∘fix₂)` (cosine, vs a random-pair baseline) → the manifold **composes**; if not → the model **reuses but doesn't compose**, and the keystone is false. *This single experiment supports or kills the central claim.*

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
- The graph **topology is the policy prior** → **V7 kill-test:** does it beat a flat (graph-less) policy?

### T.6 Goal **regions**, not goal nodes
The planner seeks a **region**, not a node — a neighborhood in the operator manifold (`Programming → Concurrency → Transactions → atomic-fix`). This yields **hierarchy / subgoals / abstraction without inventing every subgoal**: the manifold's natural **coarse→fine** structure *is* the region hierarchy (we already have it — `coarse_bucket` → fine embedding-cluster = a 2-level hierarchy). The traverse **descends**: pick the region (coarse), refine to the operator (fine). [coarse/fine BUILT; the descent policy VISION.]

### T.7 Nodes are **programs** (executable), not labels
An operator node is a typed **program**:
- `precondition` — when it applies (soft, embedding-matched) [BUILT]
- `execution` — `realize_hint` → the frozen LLM realizes it as code [BUILT]
- `expected_effect` — **the cluster centroid IS the fix-direction in embedding space** (we already have this — the operator's centroid is its expected latent effect) [VALIDATED]
- `failure_modes` — observed failures → INVALIDATE, from write-back [VISION]

Then traversal is **execute → execute → execute** — a planning system (STRIPS/PDDL-style operators with preconditions + effects), not move → move. Neurosymbolic planning with an LLM realizer.

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
