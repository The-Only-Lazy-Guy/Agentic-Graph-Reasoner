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
| **Latent traversal = COMPOSITION** (`latent_compose.py`, train-only basis, no leakage) | direct 0.42 vs **operator-compose 0.54** (+0.12) | **the LGGN core is REAL** — composing operators reaches the fix far better than direct |
| Operator SELECTION (RL, `rl_refiner.py`) | reward flat 0.07 | discrete pick is the WRONG mechanism (composition ≠ selection) |
| **Composition-DECODE** (4B, n=20, `composition_decode.py`) | issue-only **0.144** vs top1 0.115 vs topK 0.115 | the frozen LLM **can't cash in** the composition — delivered patterns *mislead* it → **decode is the frozen-LLM wall** |
| **LGGN refiner** (`lggn_refine.py`, r=512, Qwen3.5-4B, 5-seed CI) | recurrence **+0.084±0.011** · constraints **+0.112±0.012** · graph **+0.028±0.008** · learnedness **+0.128±0.009** · vs-noniter **+0.309±0.013** — ALL PASS | **all 4 LGGN pillars load-bearing at r=512.** Graph was FLAT at r=256 (capacity-starved); rescued by wider projection. The discrete operator basis IS better than free MLP when both have enough capacity |
| LGGN extended (r=512, 5-seed) | topology **-0.003±0.003** · hybrid **-0.015±0.007** vs nograph · compounding **+0.001±0.010** | topology/hybrid/compounding ALL FLAT. Graph value = per-step learned directions, NOT sequencing/prior/scaling |
| **Decode v1 — soft prefix** (`lggn_decode.py` v1, LoRA + soft prefix from h_K, 4-bit Qwen3.5-4B) | baseline **0.136** · latent **0.098** · ceiling **0.113** — ALL prefix arms WORSE than baseline. Training loss identical (~0.35) across all arms | **INJECTION MECHANISM BROKEN.** LoRA ignores prefix tokens, learns text→text shortcut. The prefix is a weak injection point diluted by self-attention over 200+ text tokens. Soft prefix = prompt engineering in disguise — the latent changes the INPUT, not the COMPUTATION. Escalate to FiLM. |
| **Decode v2 — FiLM** (`lggn_decode.py` v2, LoRA + FiLM(z), 4-bit Qwen3.5-4B, 5ep, stability fixes) | baseline **0.079** · constant **0.139** · latent **0.106-0.203** (variance) · ceiling **0.210**. Permuted gold: **0.167** | **FiLM MECHANISM VALIDATED** (ceiling-baseline=+0.131). But latent arm unstable across runs — refiner h_K (cos 0.553) is below the decoder's sensitivity cliff (~0.78). Ceiling decoder handles h_K (0.196) better than latent decoder trained ON h_K (0.106). **Refiner quality is the bottleneck, not decode.** |
| **Graph edits wiring** (`graph_edits.py` + `lggn_decode.py` phase 4) | 10 structural edits · selftest PASSES · wired to real FiLM outcomes: top-2 trajectories, verified-based seeding, all 7 per-outcome + maintenance edits fire | **BUILT.** 3 semantic issues found and fixed: (1) MINT dead (trajectory always populated) — not fixable without DERIVE branch, (2) CONNECT/COMPOSE dead from argmax lock-in — fixed via top-2 distinct-op trajectory, (3) seed behavior_embs wrong space — fixed via mean of verified h_K projections. Graph→decode loop still OPEN (write-only, no retrieval→decode path yet). |
| **Fable-5 HF integration** (`lggn_decode.py`, `_load_fable5_texts`) | 4782 agentic coding sessions from `Glint-Research/Fable-5-traces` · str_replace displacement pairs (old→new) extracted · `--dataset fable5\|lite+fable5` | **BUILT** |
| **Decoder diagnostic breakdown** (n=200 lite, post-fixes) | format_fail **0%** · zero_recall **55%** · partial **28%** · good **17%** · baseline 0.153 · ceiling 0.181 | **MEASURED.** 55% zero_recall = 4B capacity wall (valid patches, wrong fix strategy). Truncation fixed (14%→0%). |

**Honest negatives the kill-tests caught** (right-for-the-right-reason): signature operators were goal-blind (~chance → embedding ops fixed it); raw topology 65% was a self-loop artifact (op-change is the real 38%); Fable-greenfield obs=0% (no failures → domain-mismatch, rescued on SWE debug); retrieval has no resolve headroom (best-of-3=top-1).

**One-line thesis status (2026-07-04):** LGGN reasoning substrate **validated** (refiner at r=512, 5-seed CI, all 4 pillars pass). FiLM decode mechanism **validated** — ceiling 0.210, decoder CAN read gold representations. Graph edits engine **built and wired** to real outcomes. **Fable-5 HF dataset wired** to decoder pipeline (`Glint-Research/Fable-5-traces`, 4782 sessions, str_replace displacement pairs). **Diagnostic breakdown** after truncation fix (256→512 tokens): format_fail **0%**, zero_recall **55%**, partial 28%, good 17%. The 55% zero_recall = **4B model capacity wall** (generates valid patches, wrong fix strategy), not pipeline issue. **The WALL is STILL the REFINER:** h_K at cos 0.561 is below the decoder's sensitivity cliff (~0.78). **Next:** improve refiner quality (cos 0.56 → 0.78+), THEN close the graph→decode→verify→graph loop. Model size discussion pending.

**CORRECTION (2026-07-01) — resolve is UNBUILT, not closed.** The HRM lens exposed the gap: HRM iterates *against the puzzle state*; we only ever ran a one-shot operator picker + frozen executor. `iterative_refine.py` probe: iteration on a *static goal-embedding* is FLAT (K=1..8 ≈ 0.39) — a fixed input has no constraints to propagate.

**UPDATE (2026-07-02) — the TRAINED REFINER is now VALIDATED.** `lggn_refine.py` builds exactly the missing piece: a trained refiner that iterates h_t against code-token hidden states (the constraint environment), selecting graph operators each step. At r=512, all 4 pillars PASS (5-seed CI). The refiner reaches cos=0.593 with the gold fix in Qwen space — the *reasoning* works. **What remains:** (1) a **trained decoder** that turns h_K into the actual patch (the decode wall — frozen 4B can't cash in the composition), (2) the **loop assembled** (traverse→realize→verify→refine), (3) **write-back** — successful trajectories feed back into the graph as new/strengthened operators (the growth mechanism). The r=256→512 flip taught us the policy projection was capacity-starved; the graph earns its place when given enough room. Extended tests (topology/hybrid/compounding all flat) tell us: don't complicate the op selection — the value is in the learned directions themselves.

---

## THE DECODE PIVOT: From Embedding Directions to Behavioral Modulation (2026-07-02)

### What failed and why

**Soft-prefix injection (lggn_decode v1):** projected h_K into 4 soft tokens prepended to the input sequence. Result: baseline 0.136, latent 0.098, ceiling 0.113. All prefix arms WORSE than baseline. Training loss identical (~0.35) across all three arms — the LoRA adapter learned text→text and completely ignored the prefix tokens. The prefix is diluted by self-attention over 200+ text tokens. Even the CEILING (gold fix embedding projected into prefix tokens) couldn't beat baseline — the injection mechanism itself is broken, not the latent quality.

**Root cause:** soft prefix is prompt engineering in disguise. It adds information to the INPUT, but the model is free to ignore it (and does). The LoRA learns a text→text shortcut that's strictly better than reading noisy prefix tokens. This is the same failure mode as composition_decode (ops delivered as TEXT hurt: 0.115 < 0.144 issue-only).

**The lesson:** the latent must change the COMPUTATION, not the input. The model must be UNABLE to ignore the conditioning.

### Architecture ranking (from user research)

Six injection mechanisms evaluated for LGGN:

| Mechanism | How it works | Can model ignore it? | Bandwidth | Rating |
|---|---|---|---|---|
| **FiLM / AdaLN** | h' = γ(z)⊙h + β(z) at every layer | **NO** — modulates every activation | Medium (per-channel scale+shift) | ★★★★★ |
| **Hypernetwork → LoRA** | z generates LoRA adapter weights | **NO** — changes the weights | High (full weight modification) | ★★★★★ |
| Cross-attention (T.3 style) | attention over latent tokens | Partially — heads can attend elsewhere | High (content-based) | ★★★★ |
| Hidden-state replacement | replace activations at a target layer | NO — but distributional mismatch risk | Highest (direct replacement) | ★★★ |
| Soft prefix (v1, FAILED) | project z into input tokens | **YES** — diluted by self-attention | Low | ★★ (PROVEN BROKEN) |
| KV-cache injection | inject z into attention KV cache | Partially | Medium | ★★★ |

**Chosen: FiLM first** (simplest ★★★★★, proven in DiT/SD3), with hypernetwork as the escalation path.

### The conceptual shift

Before (soft prefix / composition_decode):
```
latent vector → add to input → model reads it (maybe) → text
```
The model is asked to DECODE a latent. It can and does ignore it.

After (FiLM):
```
latent vector → behavior encoder → FiLM renderer → modulate every layer → model computes differently → text
```
The model is asked to COMPUTE differently. It cannot ignore it — every layer's activations are scaled and shifted.

**The graph becomes a library of model BEHAVIORS, not embedding directions.**

Before: a node = a point in representation space (a direction in Qwen hidden space).
After: a node = a computational modification (a set of per-layer modulations that change how the model processes information).

This is a richer abstraction. An embedding direction says "the fix is OVER THERE." A behavioral modulation says "THINK THIS WAY to find the fix."

### The architecture (3 layers, not 2)

```
h_K (refiner output, d=2560 in Qwen hidden space)
 │
 ▼
BehaviorEncoder (learned projection, d=2560 → b=64-128)
 │
 ▼
behavior embedding (compact, ~64-128d)     ← THE GRAPH LIVES HERE
 │                                           operators defined here
 ▼                                           composition happens here
FiLM Renderer (per-layer expansion)          write-back stores points here
 │                                           retrieval searches here
 ▼
γ₁β₁ ... γ₃₆β₃₆ (per-layer scale+shift)
 │
 ▼
Qwen hidden states modulated at every layer
```

**Why three layers, not two:** the graph should not manipulate raw FiLM parameters (36 layers × 2 × 2560 = ~184K values). That's like a scene graph manipulating pixels. The BehaviorEncoder compresses h_K into a compact behavior space where graph operations (composition, retrieval, write-back) are tractable. The FiLM Renderer is just that — a renderer. Analogous to graphics: scene graph → renderer → pixels.

**Parameter count:**
- BehaviorEncoder: Linear(2560→64) + GELU ≈ 164K params
- FiLM Renderer: 36 × Linear(64→5120) ≈ 12M params
- LoRA (q/k/v/o, r=16): ~9M params
- Total trainable: ~21M — fits in 6GB VRAM alongside 4-bit Qwen3.5-4B

**Initialization:** FiLM initialized to identity (γ=1, β=0). At init, the model behaves exactly as if FiLM weren't there. Training teaches the FiLM to deviate from identity in ways that help decode.

### FiLM × LGGN synergy map (6 interaction points)

FiLM turns a latent vector into a behavioral modification of the LLM. Every place LGGN produces a latent, FiLM can consume it. Every place the LLM runs, FiLM can steer it.

**1. Decode (the gate test)**
- h_K → BehaviorEncoder → FiLM → Qwen generates patch
- If ceiling > baseline → FiLM injection works
- If latent > baseline → h_K helps decode (the bridge works)
- This must pass before anything below matters.

**2. Conditioned representation extraction (feedback loop)**
- Currently: frozen Qwen produces g, f, ctx as fixed hidden states
- With FiLM: Qwen + FiLM(previous h_K) extracts conditioned representations
- The model "thinks differently" when reading code based on what the refiner discovered
- refine → FiLM → re-extract → refine → ... converges when the behavior stabilizes
- This is iterative fixed-point inference / belief propagation on the graph

**3. Model-internal verification (soft gate before Docker)**
- Generate patch with FiLM(h_K)
- Re-encode patched code through Qwen + FiLM(h_K)
- Check multiple signals: representation similarity, model confidence, edit consistency
- Passes soft gate → spend Docker. Fails → update h_K → re-generate.
- Docker remains the external oracle. Internal verify is a cost filter, not a replacement.
- CAUTION: high cosine(goal) ≠ patch is correct. Combine signals; don't trust cosine alone.

**4. Operators in behavior space (redefines what a graph operator IS)**
- Currently: KMeans on (f-g) displacements in Qwen embedding space → static ops as vectors
- With FiLM: operators are directions in the BEHAVIOR manifold, not the embedding manifold
- Each op = a direction in behavior space → a set of (Δγ, Δβ) changes → a way to rewire computation
- Composing ops = composing behavioral changes. "Apply GuardType then FixMutation" = stack their modulations
- Discovery pipeline: successful patches → extract learned FiLM states → compute Δγ,Δβ from identity → PCA/autoencoder → compressed behavior vectors → cluster → operator nodes in the graph
- The graph never touches raw FiLM params — it operates in the compressed behavior manifold

**5. Behavioral write-back (procedural memory — the most novel piece)**
- After a successful decode: extract the learned FiLM state (γ,β per layer)
- Compress: FiLM state → BehaviorEncoder (reverse direction) or a separate compressor → behavior vector
- Store that behavior vector as a new graph node
- Future similar instances: retrieve the behavior vector → FiLM Renderer → reinstall the computational state
- The graph stores and retrieves previously-successful WAYS OF THINKING, not facts or embeddings
- This is **procedural memory**: "when I saw this kind of bug, thinking THIS WAY worked" — stored as a literal computational configuration, not a text description
- Closest analogy: motor memory in neuroscience. The brain doesn't remember "how to ride a bike" as a fact; it stores the motor program. FiLM write-back stores the computational program.

**6. Hierarchical conditioning (bandwidth scaling, later)**
- Single FiLM = one global z for all layers → uniform modulation
- Layer-grouped FiLM: different z per group (early layers = syntax, mid = semantics, late = generation)
- The refiner produces K×G latents (K steps × G groups) instead of one h_K
- More bandwidth, more targeted control — but more parameters to generate

### Dependency order

```
(1) Decode gate test              ← must pass first; if fails, escalate to hypernetwork
    ↓
(4) Ops in behavior space         ← redefines what "graph operator" means
(5) Behavioral write-back         ← graph stores computational states
    ↓
(2) Conditioned extraction        ← feedback loop (iterative fixed-point)
(3) Model-internal verify         ← drops Docker from inner loop (soft gate)
    ↓
(6) Hierarchical conditioning     ← bandwidth scaling
```

### Relationship to the hypernetwork / ΔW idea

FiLM applies h' = γ⊙h + β — this is **activation modulation**, not literal weight change. But the EFFECT resembles a lightweight weight edit: it changes the computation performed at each layer. The distinction matters:

- FiLM: modulates activations. Same model weights, different behavior per input. Cheap, reversible, per-instance.
- Hypernetwork → LoRA: generates actual weight updates (ΔW). The model's weights literally change. More powerful, more expensive.
- The ΔW idea (graph predicts weight updates): the graph communicates by rewiring the model. Each operator generates a small ΔW. Composition = accumulating ΔW. This IS the hypernetwork path.

**FiLM is the gate test for the principle** (latent modulates computation). If FiLM works, the ΔW path is the upgrade (higher bandwidth, literal weight change). If FiLM fails, the principle itself might not hold for this domain.

### The fixed-point iteration (the real endgame)

The feedback loop (synergy point 2) deserves emphasis:

```
initial code representation
    ↓
graph reasoning (refiner)
    ↓
behavior embedding
    ↓
FiLM modulation
    ↓
new code representation (model reads code differently)
    ↓
graph reasoning (refiner, on new reprs)
    ↓
new behavior embedding
    ↓
FiLM modulation
    ↓
... converges
```

This is not a single retrieval pass. The graph operates inside an **iterative dynamical system**. It resembles:
- **Belief propagation** — messages iterate until beliefs converge
- **Fixed-point inference** — the system converges to a self-consistent state
- **Recurrent reasoning** — each pass refines the previous

The convergence condition: when successive behavior embeddings stop changing (||b_{t+1} - b_t|| < ε). The fixed point is a self-consistent state where the model's way of reading the code and the refiner's reasoning are in agreement.

This is architecturally distinct from HRM (which iterates h_t against fixed context). Here, the CONTEXT itself changes because FiLM modifies how the model processes it. The refiner and the model co-evolve.

---

## GRAPH STRUCTURAL EDITS — the write path (2026-07-03)

The read path is validated: refiner navigates operator space → FiLM modulates Qwen → decoder generates patches. The WRITE path is the amortization mechanism — how the graph improves from experience, not just grows.

### The problem with shallow adding

Current `OperatorLibrary.add_or_strengthen()`:
```python
if cosine(new_op, existing) > threshold:
    existing.confidence += delta       # counter bump
else:
    library.append(new_op)             # add node
```

No edge updates. No precondition learning. No operator refinement. No merging. The graph accumulates but doesn't structurally improve. This means: solving task A teaches the graph NOTHING that helps task B. The amortization claim (T.4) requires structural edits.

### Graph edit operations (10 types, all operate in behavior space)

Every edit operates in the BehaviorEncoder's output space (64-128d). This is the same space where the refiner selects operators, the FiLM renderer reads, write-back stores, and retrieval searches. One space, shared by all components.

**1. MINT — create operator from successful trajectory**
- Trigger: decode succeeded (verified by Docker/tests), trajectory includes novel behavior
- Input: the FiLM state (γ,β per layer) that produced the successful patch
- Process: compress through BehaviorEncoder (reverse direction or dedicated compressor) → behavior vector
- Store: new node with behavior embedding, precondition from the input instance, confidence=low, validation_count=1
- **FiLM synergy:** BehaviorEncoder is already trained from FiLM decode — compression is free

**2. STRENGTHEN — reuse success confirms an operator**
- Trigger: existing operator retrieved, used in decode, decode VERIFIED (tests pass)
- Process: confidence += δ_up, validation_count++, update behavior embedding as EMA toward the FiLM state that worked
- **Synergy:** confidence gates retrieval — only validated operators are reusable. Prevents untested ops from propagating.

**3. WEAKEN — reuse failure penalizes**
- Trigger: operator retrieved and used, but decode FAILED (tests fail)
- Process: confidence -= δ_down, store (input_features, failure_observation) as negative precondition
- **Synergy:** failure observation from POMDP observe step → INVALIDATE in session graph → WEAKEN in persistent graph

**4. CONNECT — edge formation from co-occurrence in successful trajectories**
- Trigger: trajectory [A → B → C] verified successful
- Process: create/strengthen edges A→B, B→C with weight proportional to trajectory success
- **Synergy:** topology carries sequencing info (validated: 38% op-change prediction). Edges are the policy prior for the refiner's next-operator selection.
- Gold supervision: from agentic traces (SWE-agent/OpenHands), each tool-call transition = one edge

**5. MERGE — combine redundant operators**
- Trigger: two operators have high cosine in behavior space AND similar preconditions AND both validated (validation_count > k)
- Process: average behavior embeddings, union preconditions, sum validation counts, redirect edges
- **Synergy:** keeps library compact. Prevents fragmentation from separate MINTs of the same underlying repair.

**6. SPLIT — specialize an operator with divergent usage**
- Trigger: operator has high variance in success rate across input types (works on type-A bugs, fails on type-B)
- Process: cluster usage contexts by input features → create specialized sub-operators with tighter preconditions
- **Synergy:** sharpens retrieval precision. Each sub-operator's FiLM configuration is specialized to its context.

**7. RETIRE — prune dead operators**
- Trigger: age > threshold AND validation_count < min AND confidence < floor
- Process: mark non-retrievable (soft delete) or remove entirely
- **Synergy:** bounded graph size. Prevents retrieval noise from stale/wrong operators.

**8. REFINE_PRECONDITION — learn when an operator applies**
- Trigger: accumulated success/failure instances for an operator
- Process: train a lightweight classifier (or update embedding-based matcher) that predicts success given input features
- **Synergy:** retrieval becomes selective. "GuardType" only retrieved when the input looks like a type-guard bug, not every time.
- Gold supervision: agentic traces show which tool was selected for which problem type

**9. REFINE_EMBEDDING — update operator's behavior vector from actual usage**
- Trigger: operator used in decode, FiLM state that actually worked differs from stored behavior
- Process: EMA update: `embedding = (1-α)*embedding + α*actual_film_state`
- **Synergy:** operators drift toward their true behavioral configuration over usage. Initial KMeans centroids are bootstraps; usage refines them.

**10. COMPOSE — amortize frequent sub-trajectories**
- Trigger: sub-trajectory [A→B] appears in >k successful trajectories
- Process: create composite operator AB. Behavior = sequential application of A then B's FiLM modulations. Precondition = intersection of A and B's preconditions.
- **Synergy:** reduces trajectory length. The refiner can select AB in one step instead of A then B. Amortizes common repair patterns.

### How graph edits achieve perfect amortization

**Day 1:** library is empty. Every fix is DERIVE (decoder invents). Each success → MINT new operator.

**Day N:** library has operators from solved tasks. New task arrives:
1. RETRIEVE matches preconditions → candidate operators
2. Refiner COMPOSES operators in behavior space → FiLM → decode
3. If decode succeeds → STRENGTHEN used operators, CONNECT edges, REFINE_EMBEDDING
4. If decode fails → WEAKEN operators, REFINE_PRECONDITION (exclude this input type)
5. If no operators match → DERIVE (decoder invents) → MINT new operator

**Day 100:** library covers most repair patterns. Novel fixes are RARE. Most tasks = retrieve + compose + realize. The decoder's burden SHRINKS. The graph's value COMPOUNDS.

**The measurement (V6):** does solving task A measurably help later task B?
- Compare: fresh library vs library-after-A on task B
- If library-after-A wins → amortization is real → T.4 capacity scaling confirmed

### Training data for graph edits

| Data source | What it provides | Which edits it trains |
|---|---|---|
| **SWE-bench gold patches** | Fix decomposition (what operators) | MINT, REFINE_EMBEDDING |
| **Agentic traces (SWE-agent, OpenHands)** | Sequential tool calls, failures, revisions | CONNECT, REFINE_PRECONDITION, WEAKEN |
| **LGGN's own runs (self-play)** | End-to-end trajectories with FiLM states | ALL (the self-improving loop) |
| **Fable trajectories** | Multi-step feature builds | CONNECT, COMPOSE (long trajectories) |

Agentic traces are CRITICAL for the graph edit mechanism. SWE-bench gold patches give static decompositions; agentic traces give the SEQUENTIAL structure (what follows what, what failed, what was revised). The CONNECT and REFINE_PRECONDITION edits require this sequential signal.

### Synergy summary — where graph edits meet FiLM

```
Successful decode
    │
    ├─ FiLM state (γ,β per layer) ──→ BehaviorEncoder ──→ behavior vector
    │                                                           │
    │                                                     ┌─────┴─────┐
    │                                                     │           │
    │                                                  MINT      REFINE_EMBEDDING
    │                                                (new node)  (update existing)
    │
    ├─ Trajectory [op_A → op_B] ──→ CONNECT (A→B edge)
    │                             ──→ COMPOSE (if frequent, create AB)
    │
    └─ Verification outcome ──→ STRENGTHEN / WEAKEN
                              ──→ REFINE_PRECONDITION (success/failure context)

Failed decode
    │
    ├─ Failure observation ──→ WEAKEN (penalize used operators)
    │                       ──→ REFINE_PRECONDITION (add negative context)
    │
    └─ Session INVALIDATE ──→ re-traverse with updated goal
```

The FiLM state IS the information that flows into graph edits. A successful FiLM decode produces a compressed behavioral configuration (via BehaviorEncoder) that becomes either a new operator (MINT) or an update to an existing one (REFINE_EMBEDDING). The graph stores WHAT WORKED as a literal computational configuration — procedural memory.

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

### Operator schema (updated for FiLM/behavior space, 2026-07-03)
```
op_id, name                  # e.g. GuardType, FixInPlaceMutation
input_type, output_type      # soft types (memoryview→bytes, possibly-None→checked)
precondition                 # when it applies (soft: NL + embedding + learned matcher)
precondition_neg             # when it DOESN'T apply (learned from failures via REFINE_PRECONDITION)
realize_hint                 # how the decoder emits it (compiler-mode instruction)
embedding                    # original discovery embedding (Qwen space, 2560d)
behavior_embedding           # compressed FiLM configuration (behavior space, 64-128d) ← NEW
confidence                   # 0..1 (updated by STRENGTHEN/WEAKEN)
source                       # seed | derived:<instance_id> | mint:<trajectory_id>
age                          # tasks since created
validation_count             # times reuse actually resolved (outcome-confirmed)
failure_count                # times reuse failed ← NEW
edges_out                    # {op_id: weight} — transition structure from CONNECT ← NEW
```
**Two embeddings:** `embedding` (Qwen space) = where the operator IS in the representational manifold. `behavior_embedding` (behavior space) = HOW the operator modulates the model's computation. The refiner operates on `embedding`; the FiLM renderer reads `behavior_embedding`. REFINE_EMBEDDING updates `behavior_embedding` from actual usage.

**Why uncertainty is mandatory:** without it, `wrong primitive → write-back → retrieved again → reinforced` is catastrophic. Operators below a confidence/validation floor are **not retrievable for reuse** — only for re-validation. STRENGTHEN only after outcome-confirmed pass; WEAKEN on failure with negative precondition learning.

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
| `v5/runtime/lggn_decode.py` | ~1010 | **decode bridge v2 (FiLM) + graph wiring + Fable-5** | `_FiLMModule` (BehaviorEncoder+Renderer, identity-init), `_Decoder` (LoRA + FiLM + per-layer hooks + `behavior_embeddings()` extraction), `_extract_trajectory`, `_write_back_report`, `_sensitivity_sweep` (interpolation + permuted control); 4 arms + phase 4 graph integration. **New (2026-07-04):** `_load_fable5_texts(limit=0)` — Fable-5 HF str_replace displacement pairs; `--dataset fable5\|lite+fable5`; `--arms` filter (comma-sep arm subset, cuts 4.8h→2.2h); `eval_on(diagnose=True)` — failure breakdown (format_fail/zero_recall/partial/good) + raw samples + per-instance timing; generic prompt framing ("Task:"/"Code:" not "Fix this bug"/"Buggy code"); `max_new_tokens=512` (was 256). `--z-dropout`; `--sensitivity`. | FiLM validated, refiner blocking, Fable-5 wired |
| `v5/runtime/graph_edits.py` | ~520 | **graph edits ENGINE (the write path)** | `BehaviorNode` (behavior_emb + pos/neg precondition centroids + provenance), `BehaviorGraph` (nodes/edges/usage/bigrams/edit-log, save/load/snapshot), `GraphEditEngine` — `observe(Outcome)` per-decode (MINT/STRENGTHEN/WEAKEN/CONNECT/REFINE_EMBEDDING/REFINE_PRECONDITION) + `maintain()` periodic (MERGE/SPLIT/RETIRE/COMPOSE); conf-gated `retrieve()`; `seed_from_centroids` bridge. Poison gate preserved (mint conf 0.35 < floor 0.40). V6-mini in selftest PASSES (after-A retrieval helps similar A'). | built + selftest |
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
13. **LGGN refiner built** (`lggn_refine.py`): HRM-style K-step refinement in Qwen hidden space. h_0=goal, each step CrossAttn(h,code) + softmax(policy)@ops, readout cos(h_K, gold_fix). TopologyNet, HybridNet, compounding ablations included.
14. **r=256 result (5-seed):** HRM validated (recurrence+constraints+learnedness PASS), but graph-as-discrete-basis FLAT (-0.008±0.019). Free MLP matched discrete ops. Diagnosed: policy projection capacity-starved at r=256.
15. **r=512 result (5-seed) — THE FLIP:** graph pillar RESCUED (+0.028±0.008 PASS). All 4 pillars load-bearing. The discrete operator basis IS better than free MLP when the projection has enough room. Extended: topology -0.003 (flat), hybrid -0.015 (interferes), compounding +0.001 (saturates). Graph value = learned directions, not sequencing/prior/scaling.
16. **Next wall identified: DECODE.** The refiner reaches cos=0.593 with gold fix in Qwen space. The frozen LLM can't turn that into a patch (`composition_decode` confirmed). Need a trained decoder + write-back (model feeds discoveries into the graph).
17. **Decode v1 — soft prefix FAILED.** (`lggn_decode.py` v1, LoRA + soft prefix from h_K, 4-bit Qwen3.5-4B on molab.) baseline=0.136, latent=0.098, ceiling=0.113. ALL prefix arms WORSE than baseline. Training loss identical (~0.35) across all arms — LoRA ignores prefix tokens, learns text→text shortcut. Even the ceiling (gold_f projected into prefix tokens) couldn't beat baseline. Root cause: soft prefix adds information to the INPUT; the model is free to ignore it. Same failure mode as composition_decode (ops-as-text hurt). The latent must change the COMPUTATION, not the input.
18. **Architecture pivot: FiLM + Behavioral Memory.** Evaluated 6 injection mechanisms. FiLM (h' = γ(z)⊙h + β(z) at every transformer layer) chosen: model CANNOT ignore it (every layer modulated), proven in DiT/SD3, ~12M extra params. Key conceptual shift: the graph becomes a library of model BEHAVIORS (computational modulations), not embedding directions. Each operator = a way to rewire the model. BehaviorEncoder maps h_K to compact behavior space (64-128d) where graph ops, composition, write-back, retrieval all operate. FiLM Renderer expands to per-layer (γ,β). 6 synergy points identified: (1) decode gate test, (2) conditioned repr extraction (feedback loop → fixed-point inference), (3) model-internal verify (soft gate before Docker), (4) ops in behavior space, (5) behavioral write-back (procedural memory — stores successful computational states), (6) hierarchical conditioning.
19. **Decode v2 — FiLM gate test (3 runs on molab, 2026-07-02).**
    - **Run 1 (2 epochs, no stability fixes):** baseline=0.099, latent=0.074, ceiling=0.153. FiLM injection works (ceiling-baseline=+0.054) — first proof the mechanism isn't broken. But latent arm worse than baseline: h_K not helping yet. BatchEncoding bug + gradient checkpointing incompatibility fixed during this run.
    - **Run 2 (5 epochs, no stability fixes):** baseline=0.125, latent=0.057, ceiling=0.129. Latent loss EXPLODED at epochs 4-5 (0.529→0.373→0.295→0.799→0.716). Noisy h_K creates conflicting FiLM gradients. Sensitivity sweep confirmed monotonic trend (gold→h_K interpolation tracks cos with gold). Need training stability before increasing epochs.
    - **Run 3 (5 epochs, WITH stability fixes):** gradient clipping (max_norm=1.0) + FiLM warmup (freeze epoch 1) + separate param groups (FiLM at 0.5x LR). **ALL THREE GATES PASS:**
      ```
      baseline  : 0.079
      latent    : 0.125   (+0.046 over baseline — BRIDGE WORKS)
      ceiling   : 0.183   (+0.104 over baseline — FiLM INJECTION WORKS)
      ```
      Training loss stable: latent 0.488→0.363→0.227→0.175→0.113 (monotonic).
    - **Sensitivity sweep (ceiling decoder, gold→h_K interpolation):** mostly monotonic from alpha=1.0 (gold) to alpha=0.0 (h_K), with a peak at alpha=0.67 (recall 0.204 > gold 0.183 — mixing some h_K INTO gold helps, suggesting complementary information). **BUT random z gets 0.187** (cos=0.001 with gold), matching gold (0.183). The FiLM decoder at eval time is not strongly z-direction-dependent.
    - **Diagnosis: PRESENCE vs MEANING.** The FiLM renderer weights learn a beneficial modulation pattern during training. At eval, the specific z-content is secondary — the learned (γ,β) pattern is mostly fixed in the renderer weights. Training z quality matters (ceiling trained with gold > latent trained with h_K > baseline trained without FiLM), but eval z does not. This means FiLM is acting as a "learned adaptive layer norm" — a capacity boost from the extra ~12M trainable params — not as a true conditioning mechanism responsive to z content.
20. **Constant-z arm + z-dropout (2026-07-03).** Diagnostic to disentangle FiLM capacity from z-content signal. constant-z = FiLM with mean(h_K) for all instances. z-dropout = randomly drop z during training (p=0.15), forces FiLM to USE z when present. **Results (best run):** baseline=0.079, constant=0.139, latent=0.203, ceiling=0.210. Decomposed: +0.060 capacity, +0.065 z-content, +0.007 refiner gap. z-dropout critical: without it latent was only 0.125. **Z-CONTENT MATTERS** — the graph's latent carries semantic signal, not just a capacity boost.
21. **Permuted-latent control (2026-07-03).** Replaced random Gaussian with permuted gold (gold_f from WRONG instance — correct distribution, wrong semantic match). **Result: perm=0.167** (cos=0.375 with correct gold). Permuted < h_K (0.196) < gold (0.210). Wrong-instance gold actively MISLEADS — coherent-but-wrong conditioning hurts more than noise. Confirms partial instance-specificity: the decoder extracts SOME instance-specific content from z. The full sensitivity curve on the ceiling decoder remains flat (no z-dropout on ceiling), but the latent decoder (with z-dropout) is genuinely z-dependent.
22. **Strategic assessment (2026-07-03).** The graph read path is validated (refine→FiLM→decode). The write path is shallow (counter increments). Identified 10 structural graph edit operations needed for amortization. Training data gap: 116 instances is insufficient; need full SWE-bench + agentic traces. The 4B model CAN emit patches at 83% recall with exact gold in prompt — the FiLM channel bandwidth (ceiling 21%) and training scale are the bottleneck, not model size. The graph should ELEVATE the model, not be bypassed by model scaling.
23. **Graph edits wired to lggn_decode (2026-07-03).** `graph_edits.py` engine wired to real FiLM outcomes in lggn_decode phase 4. Three semantic issues found and fixed during validation: **(1) MINT never fires** — refiner always picks from op basis, trajectory always populated, MINT requires empty trajectory (DERIVE branch). Graph grows only via SPLIT. **(2) CONNECT/COMPOSE dead** — refiner locks onto one dominant op across K steps (argmax same op K times → no transitions). **Fixed**: top-2 distinct-op trajectory with diversity threshold (weight > 0.15 for runner-up). After fix: CONNECT and COMPOSE both fire. **(3) Seed behavior_embs wrong space** — `BehaviorEncoder(op_centroid)` projects Qwen-space displacement direction through encoder trained on h_K (different distributions). **Fixed**: seed from mean of VERIFIED h_K projections per dominant op (real FiLM states), fallback to centroid projection for unseen ops. Selftest: 10 nodes (6 retrievable), 142 edits, all 7 edit kinds fire. Graph→decode loop remains open (write-only, no retrieval→decode path).
24. **Molab confirmation: refiner is the bottleneck (2026-07-03).** Latest molab run: baseline=0.079, constant=0.139, latent=0.106, ceiling=0.210. Sensitivity curve reveals the critical finding: **decoder sensitivity cliff at cos ≈ 0.78.** Recall flatlines at ~0.196 for cos < 0.7, jumps to ~0.217 for cos ≥ 0.78. 50/50 gold+h_K mix (cos 0.783) peaks at 0.219 > pure gold 0.210 (complementary info). Refiner h_K at cos 0.553 is well below the cliff. **Worse: latent decoder (0.106) < ceiling decoder at h_K (0.196).** Training LoRA on noisy h_K teaches bad co-adaptations — the gold-trained decoder handles noisy input MORE gracefully than the noise-trained decoder. This flips the roadmap: **refiner quality (cos 0.55 → 0.78) is the critical path, not graph wiring or decoder improvements.** Latent arm variance 0.106-0.203 across runs needs investigation (z_dropout RNG seeding may not be deployed to molab).
25. **Fable-5 HF dataset integration (2026-07-04).** `_load_fable5_texts(limit=0)` extracts str_replace displacement pairs (old→new) from `Glint-Research/Fable-5-traces` on HuggingFace. 4782 agentic coding sessions, deduped by content hash. Only str_replace/edit edits with both old+new code used (greenfield writes skipped). Mixed dataset via `--dataset lite+fable5` (splits on "+", concatenates reprs). Corresponding `_reprs_from_texts()` added to `lggn_refine.py` for generic Qwen repr computation.
26. **Diagnostic failure breakdown (2026-07-04).** `eval_on(diagnose=True)` classifies each held instance into: format_fail (truncation / malformed output), zero_recall (valid SR format but wrong code), partial (recall > 0 but < 0.5), good (recall >= 0.5). Captures 8 raw output samples with category labels. Per-instance inference timing (ms/instance, min, max). Runs automatically on baseline + ceiling arms.
27. **Truncation fix (2026-07-04).** `max_new_tokens=256` was too small — SEARCH block consumed most tokens, REPLACE block cut off before `>>>>>>> REPLACE` marker. 14% format failures. Fixed to `max_new_tokens=512`. Result: format_fail **14% → 0%**.
28. **Generic prompt framing (2026-07-04).** `_prompt()` said "Fix this bug" / "Buggy code" but Fable-5 data is feature-building, not bug-fixing. Fixed to "Task:" / "Code:". Also fixed in `lggn_refine.py` `_reprs()` (`hid("Buggy code:\n" + removed)` → `hid("Code:\n" + removed)`). Note: changing ctx prompt invalidates cached `.npz` files.
29. **Training time optimization (2026-07-04).** `--arms` CLI flag — comma-separated arm filter to skip unneeded decoder arms. `--arms latent,ceiling` cuts total time from 4.8h to ~2.2h. `--arms baseline,ceiling --decoder-epochs 3` for 30min diagnostic runs. Per-phase timing dict added throughout `run()`.
30. **Post-fix diagnostic (2026-07-04, n=200 lite, baseline+ceiling, 3 epochs).** After truncation + prompt fixes: format_fail **0%**, zero_recall **55%**, partial **28%**, good **17%**. baseline=0.153, ceiling=0.181, gap=+0.027 (FiLM works). **55% zero_recall is the dominant failure mode** — model generates syntactically valid search/replace patches but with the wrong fix strategy. This is 4B model capacity wall, not pipeline issue. The model finds the right code location but invents a different (wrong) repair.
31. **Phase B code wiring (2026-07-04-05).** Refiner `Net.forward(return_traj=True)` now returns per-step latents `h_steps [K,N,d]` alongside op-selection weights (Gap 1 from audit). `_Decoder` gains `save_checkpoint()`, `load_checkpoint()`, `generate(prompt, z)` for standalone FiLM-conditioned generation (Gap 2). `lggn_loop.py` gains `make_rank_from_trajectory()` (refiner-informed op ranking) and `make_realize_film()` (FiLM-conditioned patch generation with per-step latent). `lggn_decode._train_refiner` propagated n_heads + GPU training. All selftests pass.
32. **Operator basis experiment — "ops are a basis, not semantics" (2026-07-05, Qwen2.5-3B, d=2048, n=145, 3 seeds).** Major finding. KMeans operator centroids are WORSE than random Gaussian vectors as the refiner's operator basis. Random ops in d=2048 are near-orthogonal (Johnson-Lindenstrauss), spanning more of the displacement space than correlated KMeans centroids. The refiner's MLP composes ops like Fourier coefficients — what matters is spanning the space, not semantic clustering.

    **Results table (held cos, 3-seed mean ± std):**
    ```
    Config                              cos      std    note
    ─────────────────────────────────────────────────────────
    old arch K=1, KMeans 24  (r256/h1)  0.737    0.018  BEST absolute
    old arch K=1, random 24  (r256/h1)  0.736    0.017  same — K=1 insensitive to basis
    new arch K=4, random 48  (r512/h4)  0.740    —      best new arch (from main run)
    new arch K=4, fixed rand 96         0.724    0.014  lowest variance
    new arch K=4, QR-orthogonal 48      0.704    —      exact orthogonality doesn't help
    new arch K=4, KMeans 48             0.703    0.009  KMeans collapses basis
    new arch K=4, learn from rand 96    0.688    0.018  learning HURTS
    new arch K=4, learn from rand 48    0.685    0.060  learning HURTS + high variance
    new arch K=4, learn from KMeans     0.687    0.055  can't escape collapsed subspace
    new arch K=1, random 48             0.635    0.101  new arch overfits at K=1
    new arch K=2, random 48             0.656    0.061  improves with more K
    ```

    **Key findings:**
    - **Random ops > KMeans** (+0.037): KMeans optimizes reconstruction, not span. Random vectors are near-orthogonal at d=2048, giving 48 independent knobs vs 48 correlated ones.
    - **Fixed ops > learned ops** (+0.05): learning ops allows overfitting on 116 train instances. Fixed random = regularization (like fixing the Fourier basis and only learning coefficients).
    - **Old arch K=1 = 0.737 ABOVE the cliff** — but K=1 means no trajectory, no iteration plan, no multi-leaf. The architecture gets one shot. For the pipeline (Phase B), need K>1.
    - **New arch needs K≥4**: K=1 (0.635) → K=2 (0.656) → K=4 (0.698-0.740). More parameters need more steps as regularization.
    - **K=8 WORSE than K=4** (synthetic confirmed, consistent across d=256/512): more steps = more overfitting.
    - **QR-orthogonal ≈ KMeans** — exact orthogonality doesn't help. The original randops (Gaussian normalized to same per-op norm) are better because the scale distribution matters more than exact orthogonality.
    - **PROXY WARNING**: cos is a proxy. The real metric is decoder recall (code quality). cos > 0.73 predicts decoder benefit per the sensitivity curve, but z-dropout can soften the cliff. Full pipeline test (refiner → decoder → recall) needed on molab.

    **Implication for `_discover_ops`:** Replace KMeans with fixed random initialization as the default. The ops are a coordinate system, not a semantic clustering. Retain op-selection weights for trajectory interpretability (the loop needs to know which "op" was applied at each step).

    **Synthetic verification (controlled, no model):** Architecture upgrade (r512/h4/48ops vs r256/h1/24ops) shows +5-8% at d=256-512 on harder synthetic tasks (N=800, 24 ground-truth ops, chain=4). K=4 consistently beats K=8. Improvement scales with d.

33. **Qwen3.5-4B (d=2560) molab results + `_reprs()` mixed dataset fix (2026-07-05).** Qwen3.5-4B has DeltaNet layers (theoretically more efficient) but d=2560 vs Qwen2.5-3B's d=2048. Refiner peaks much lower. Fixed `_reprs()` in lggn_refine.py to handle `lite+fable5` mixed dataset syntax (split on `+`, handle `fable5` via `_load_fable5_texts()` + `_reprs_from_texts()`, concatenate).

    **Molab results (Qwen3.5-4B, d=2560, K=4/K=2, random ops):**
    ```
    Config                    cos     note
    ─────────────────────────────────────────
    r=512  K4_full            0.502   code hurts -0.081
    r=512  K4_nocode          0.583   best K=4
    r=1024 K4_full            0.529   more r helps +0.027
    r=1024 K4_nocode          0.584   nocode still wins
    r=1024 K2_full            0.537   K=2 code neutral
    r=1024 K2_nocode          0.525   code helps slightly at K=2
    ```

    **Compare Qwen2.5-3B (d=2048):** K4_full=0.703, K4_randops=0.740. Gap is **0.156** (0.584 vs 0.740).

    **Key findings:**
    - **d=2560 is harder.** raw_g is 0.226 (vs 0.372 for d=2048). Higher dimensionality = sparser displacement space = harder to learn directional structure.
    - **Code attention HURTS at K=4** consistently (-0.055 to -0.081). At K=2, effect is neutral/slightly positive. Hypothesis: code tokens add noise the attention can't filter with 4 recurrent steps.
    - **All results below 0.73 cliff.** Even best (0.584) is far below where decoder starts benefiting.
    - **Doubling r (512→1024) gives diminishing returns** (+0.027 for K4_full, +0.001 for K4_nocode). Bottleneck is not model capacity but displacement space difficulty.
    - **Verdict: Qwen2.5-3B (d=2048) is the better refiner backbone.** DeltaNet efficiency doesn't compensate for the d=2560 displacement learning difficulty. Use Qwen3.5-4B only for decode (where raw generation quality matters).

    **`_reprs()` fix:** `main()` in lggn_refine.py now splits dataset on `+`, handles `fable5` part via `_load_fable5_texts()` + `_reprs_from_texts()`, and SWE parts via `_reprs()`, concatenating all arrays. Previously passed raw `"lite+fable5"` to `load_instances()` which triggered HFValidationError.

34. **Fable5 data HURTS refiner at d=2560 (2026-07-05, Qwen3.5-4B, n=722, 5 seeds).** Critical negative result. Adding 500 fable5 displacement pairs to 222 SWE-bench instances DEGRADED refiner by -0.100. Refiner actively destroys signal — all arms below direct (no refinement).

    **Results (Qwen3.5-4B, d=2560, lite+fable5, r=1024, K=2, 5 seeds):**
    ```
    Config         cos     std     note
    ───────────────────────────────────────────────────
    raw_g          0.229   0.005   starting point
    direct         0.529   0.003   NO refinement (projection only)
    K1_full        0.445   0.049   refiner HURTS (-0.084 vs direct)
    K2_full        0.437   0.037   recurrence doesn't help
    K2_nocode      0.424   0.020   code attention neutral
    K2_nograph     0.484   0.011   free MLP > graph ops (+0.047)
    K2_randops     0.457   0.045   random ops ≈ learned
    ```

    **Compare lite-only (same model, same r, same K, n=145, 3 seeds):**
    ```
    K2_full        0.537   —       lite only
    K2_full        0.437   0.037   lite+fable5  →  DELTA = -0.100
    ```

    **Key findings:**
    - **Fable5 is destructive at d=2560.** Feature-building (str_replace) has fundamentally different displacement patterns than bug-fixing (SWE-bench). Mixed training confuses the refiner — can't learn coherent direction when two domains pull opposite ways in high-d.
    - **Refiner WORSE than direct.** K2-direct = -0.092. The refiner learns to move AWAY from gold. This never happened on Qwen2.5-3B (d=2048) with lite-only.
    - **Free MLP > graph ops** at 0.484 vs 0.437. Graph constraints actively harm when the displacement space is incoherent (mixed domain data).
    - **More data ≠ better.** 722 instances > 145, but performance dropped 0.100. Domain coherence matters more than quantity.
    - **OPEN QUESTION:** ~~Does fable5 also hurt on Qwen2.5-3B (d=2048)?~~ **ANSWERED in entry 35: YES, it hurts on both.** Domain mismatch confirmed as root cause, not dimensionality.

35. **Fable5 hurts refiner on BOTH models — domain mismatch confirmed (2026-07-05, Qwen2.5-3B, d=2048, n=722, 5 seeds).** Resolves the open question from entry 34. Fable5 data is destructive for the refiner regardless of dimensionality.

    **Results (Qwen2.5-3B, d=2048, lite+fable5, r=1024, K=2, 5 seeds):**
    ```
    Config         cos     std     note
    ───────────────────────────────────────────────────
    raw_g          0.394   0.007   starting point
    direct         0.637   0.011   NO refinement (r=1024 projection)
    K1_full        0.615   0.049   refiner HURTS (-0.022 vs direct)
    K2_full        0.595   0.046   recurrence doesn't help
    K2_nocode      0.584   0.011   code attention neutral
    K2_nograph     0.634   0.019   free MLP ≈ direct (best refiner arm)
    K2_randops     0.597   0.065   random ops ≈ learned
    ```

    **Cross-model comparison (lite+fable5, K=2, r=1024):**
    ```
    Metric              d=2048      d=2560
    ───────────────────────────────────────
    direct              0.637       0.529
    K2_full             0.595       0.437
    K2-direct           -0.043      -0.092
    K2_nograph-full     +0.040      +0.047
    ```

    **Hypothesis resolution:**
    - ~~H3 (d-specific)~~ **RULED OUT.** Both models show same pattern: refiner < direct, nograph > full.
    - **H1 (parameter interference) CONFIRMED.** Fable5 displacements (feature-building) and SWE displacements (bug-fixing) compete for shared refiner parameters. Fable5 dominates (69% of training) → SWE-optimal parameters shift → all held instances degrade.
    - H2 (fable5 gold inherently noisy) may COMPOUND H1 but is not the primary cause — if it were, direct would also be lower for fable5 instances, but direct actually increased (0.637 vs 0.543 in lite-only, though r difference contributes).
    - d=2048 less severely affected (-0.043 vs -0.092) because the displacement space has more structure (raw_g 0.394 vs 0.229), giving the optimizer more signal to separate domains. But not enough to overcome the interference.

    **~~Verdict: DOMAIN-MATCHED TRAINING ONLY for refiner.~~** REVISED in entry 36 — the problem was a data loading bug, not domain mismatch.

36. **ROOT CAUSE FOUND: per-edit granularity bug in `_load_fable5_texts` (2026-07-05).** Entries 34-35 blamed "domain mismatch" — WRONG. The real cause: each individual str_replace was a separate instance, but sessions share the same goal. One session with 210 edits = 210 instances with IDENTICAL `g` embedding and 210 DIFFERENT `f` targets. This is a one-to-many mapping that's impossible for the refiner to learn.

    **Data analysis (all fable5, no limit):**
    ```
    Total individual edits:      1850
    Unique goals (sessions):      964
    Mean edits/session:           1.9
    Sessions with 1 edit:         927  (fine)
    Sessions with >5 edits:        25  (poisonous)
    Top session:                  210 edits (same goal!)
    Top 10 sessions:              672 edits = 36% of all data
    ```

    With `limit=500`, early sessions dominate. If the 210-edit session appears early, **42% of training data shares one goal embedding** mapping to 210 different fix embeddings. The refiner can't learn — same input, contradictory targets. Averages to noise.

    **Compare SWE-bench:** one instance = one complete patch. Unique (problem_statement, complete_fix) pair. Never one-to-many.

    **Fix:** Aggregate all edits per session into one instance. Concatenate all old_strings → single `code`, all new_strings → single `added`. Matches SWE-bench granularity: one instance = one complete task. 964 sessions → 964 instances max. Truncate concatenated text to 1500 chars (matching `_fix_text`'s cap).

    **Reframing:** The refiner is NOT a "bug fixer" — it's a general agentic coder. Bug fixes AND feature-building are both valid target behaviors. The "domain mismatch" conclusion was premature; the actual problem was data preparation, not task incompatibility. With correct per-session aggregation, fable5 should ADD useful signal (more diverse agentic tasks = better generalization).

    **INVALIDATES entries 34-35 conclusions.** "Domain mismatch" was a misdiagnosis caused by a data loading bug. Must re-run lite+fable5 with aggregated sessions to get valid results.

37. **Per-session aggregation WORKS — fable5 no longer destructive (2026-07-05, Qwen2.5-3B, d=2048, n=722, 5 seeds).** Confirms entry 36 fix. Refiner now HELPS on mixed data.

    **Results (Qwen2.5-3B, d=2048, lite+fable5 AGGREGATED, r=1024, K=2, 5 seeds):**
    ```
    Config         cos     std     note
    ───────────────────────────────────────────────────
    raw_g          0.336   0.005   starting point
    direct         0.533   0.015   no refinement
    K1_full        0.658   0.043   refiner helps (+0.125 vs direct)
    K2_full        0.627   0.036   K=2 slight regression vs K=1
    K2_nocode      0.589   0.012   code attention helps (+0.038)
    K2_nograph     0.648   0.024   free MLP still slightly better
    K2_randops     0.642   0.030   random ≈ learned
    ```

    **Before/after comparison (same model, same dataset label, same K=2):**
    ```
    Metric          per-edit (broken)   per-session (fixed)   delta
    ──────────────────────────────────────────────────────────────
    K2-direct       -0.043 DESTROYS     +0.094 HELPS          +0.137
    K2-nocode       +0.011 flat         +0.038 PASS           +0.027
    K1_full         0.615               0.658                  +0.043
    K2_full         0.595               0.627                  +0.032
    ```

    **Key findings:**
    - **Aggregation fix = +0.137 swing on K2-direct.** From destroying signal to adding value. Confirms the per-edit granularity was the sole cause of entries 34-35 degradation.
    - **Code attention now PASSES** (+0.038). The one-to-many mapping noise was masking the real code attention signal.
    - **K=1 > K=2 here** (0.658 vs 0.627). At K=2, recurrence slightly hurts (-0.031). Need K=4 to see if more steps recover (as in lite-only where K=4 >> K=1).
    - **Still below lite-only K4 peak** (0.627 vs 0.740). But this is K=2 not K=4, and held set is 69% fable5 (different composition). K=4 run needed for apples-to-apples.
    - **raw_g dropped** (0.336 vs 0.394 in broken run). Aggregated sessions have longer concatenated text → different Qwen embeddings → different raw alignment. Direct also dropped (0.533 vs 0.637), confirming different held set composition.
    - **The model IS a general agentic coder.** Bug fixes + feature-building coexist in the same refiner when data preparation is correct. "Domain mismatch" was always a misdiagnosis.

    **Next:** K=4 on lite+fable5 (aggregated) to compare against the 0.740 lite-only peak.

## Current verdicts (updated 2026-07-05)
- **LGGN refiner:** ALL 4 PILLARS PASS at r=512 (recurrence, constraints, graph, learnedness). Reasoning substrate validated. cos(h_K, f) improved from 0.553 to **0.703-0.740** with architecture upgrade + random ops (Qwen2.5-3B, d=2048). Old arch K=1 reaches **0.737** (above cliff). New arch K=4 + random ops reaches **0.740**. **Near or above the 0.73 sensitivity cliff.** Qwen3.5-4B (d=2560) verified: peaks at **0.584** — significantly worse. **Use Qwen2.5-3B (d=2048) as refiner backbone.**
- **Operator basis = coordinate system, NOT semantics.** Major finding (2026-07-05). Random Gaussian ops beat KMeans centroids (+0.037). Fixed ops beat learned ops (+0.05). The refiner's MLP composes operators like Fourier coefficients — spanning the space matters, not clustering. KMeans collapses the basis into a low-rank correlated subspace. `_discover_ops` should be replaced with fixed random init.
- **Graph value (REVISED):** at r=256 (entry 15), discrete ops > free MLP was the finding. At r=512 with random ops (entry 32), K4_nograph ≈ K4_full (0.701 vs 0.703). The graph basis adds near-zero over free MLP. The graph's value for the PIPELINE is not cos improvement — it's providing interpretable trajectories for the iteration loop (which op to try, which to invalidate).
- **Reasoner (traverse):** LEARNS (40%, scales) — VALIDATED.
- **Compositionality:** intra-task (0.91) + cross-task (0.58) — VALIDATED.
- **Decode v1 (soft prefix):** FAILED. LoRA ignores prefix tokens.
- **Decode v2 (FiLM):** MECHANISM VALIDATED — ceiling-baseline = +0.131 (the decoder CAN read gold). **But latent arm unstable:** 0.106-0.203 across runs (high variance, possibly z_dropout RNG + small n=29). Ceiling decoder at h_K input = 0.196, latent decoder at h_K = 0.106 — gold-trained decoder degrades gracefully, h_K-trained decoder learns bad co-adaptations. **Sensitivity cliff at cos ≈ 0.78:** recall flatlines at ~0.196 for cos < 0.7, jumps to ~0.217 for cos ≥ 0.78. 50/50 gold+h_K mix (cos 0.783) peaks at 0.219 > pure gold 0.210 (h_K adds complementary signal when not too noisy).
- **Architectural pivot:** the graph IS a library of model BEHAVIORS (computational modulations). BehaviorEncoder → FiLM Renderer. Operators live in behavior space. VALIDATED by the FiLM results.
- **Graph read path:** WORKS (refine → FiLM → decode). Proven when h_K is good enough (cos ≥ 0.78).
- **Graph write path:** BUILT (`graph_edits.py`, 10 edits, selftest PASS). Wired to real outcomes in lggn_decode (behavior_emb extraction, top-2 trajectories, verified seeding). All 7 edit types fire in practice. CONNECT/COMPOSE activated via top-2 trajectory diversity. **Graph→decode loop still OPEN** (graph learns from outcomes, but nothing reads retrieved nodes back into FiLM decode yet). Closing this loop is blocked on refiner quality — a noisy graph (from noisy h_K) won't help decode.
- **Fable-5 HF integration:** BUILT. Entries 34-35 showed degradation — **root cause was per-edit granularity bug** (entry 36), not domain mismatch. Fixed: aggregate all edits per session into one instance (matching SWE-bench granularity). Cached .npz files from per-edit runs are INVALID — must regenerate. **Re-run needed to validate.**
- **Truncation:** FIXED. `max_new_tokens` 256→512. format_fail 14%→0%.
- **Dominant failure mode:** 55% zero_recall. Model generates syntactically valid patches with wrong fix strategy. This is 4B capacity, not pipeline. 28% partial (close but not exact), 17% good (recall >= 0.5).
- **The wall is the REFINER, not the decoder.** The decoder reads gold fine (ceiling 0.210). The refiner's h_K (cos 0.561) is too noisy. Improving refiner cos from 0.56 → 0.78 is the critical path. Everything downstream (graph edits, closed loop, V6 kill-test) depends on this. The diagnostic confirms: even with perfect pipeline (no truncation, correct prompts), 55% of instances hit the 4B's reasoning limit.
- **Training data gap (REVISED):** Previous "more data ≠ better" conclusion (entries 34-35) was caused by per-edit granularity bug, not domain conflict. With aggregated sessions, fable5 should add ~964 diverse agentic instances. **Re-run needed.** Full SWE-bench (~2300) still the next scale-up target.
- **Model ceiling:** 4B with EXACT gold in prompt = 83% recall (solution_ladder). FiLM ceiling = 21%. Gap = FiLM channel bandwidth + training data scale, NOT 4B capacity. **Model size discussion pending** — user considering whether to scale up from 4B.

## Strategic decision (2026-07-04): 4B + full LGGN pipeline FIRST

**Goal:** diverse consumer usage with good performance on small model. Bigger model is a drop-in upgrade AFTER pipeline proves out.

**Rationale:** LGGN's value = structured reasoning (decompose + iterate + retrieve) routing around single-shot capacity limits. The 55% zero_recall is a wall on ONE-SHOT generation, not on the SYSTEM. The system doesn't rely on single-shot — it decomposes, iterates, retrieves. Prove this at 4B, then model scaling compounds with the pipeline.

**Don't:** propose model scaling as the fix for capacity walls. **Do:** build decomposition, iteration, retrieval, MoLoRA.

```
Current:   issue → refiner → FiLM → ONE generate → done          (ceiling 0.21)
Target:    issue → refiner → decompose → [FiLM → generate → test → update] × N
           + graph retrieval pre-loads similar solved configs      (ceiling >> 0.21)
```

---

## Next (recommended order, updated 2026-07-04)

### Phase A — Refiner quality (RESOLVED)
Architecture upgrade + random ops lifted cos from 0.553 → 0.703-0.740 (Qwen2.5-3B). **Near or above the 0.73 cliff.** Qwen3.5-4B (d=2560) verified WORSE (0.584 peak) — use Qwen2.5-3B. Remaining: full decoder pipeline test to measure actual recall improvement.

1. ~~**Switch to random ops**~~ DONE. Default in CLI. +0.037 cos lift.
2. ~~**Confirm on Qwen3.5-4B**~~ DONE (entry 33). d=2560 peaks at 0.584, far below cliff. Qwen2.5-3B (d=2048) confirmed as refiner backbone.
3. **Full pipeline test** — refiner → decoder → recall. Does cos 0.74 translate to recall improvement? Run: baseline vs latent(random ops) vs ceiling. The sensitivity curve predicts yes, but z-dropout may have already bridged the gap.
4. **Stabilize latent arm variance** — latent recall 0.106-0.203 across runs. Confirm z_dropout RNG seeding is deployed. If seeded, the variance is from train/held split randomness at n=29.

### Phase B — Decomposition + iteration (THE SYSTEM VALUE)
The pipeline that routes around 4B single-shot capacity. Each piece independently useful, together they compound.

4. **Multi-leaf FiLM decode** — wire `--lggn-multileaf` through FiLM. Each leaf = one operator, one FiLM modulation, one simple generation. Operator trajectory from refiner determines leaf sequence. Existing `_multileaf_patch` applies each leaf against live tree. Key: each leaf is WITHIN 4B capacity (the decomposition payoff).
5. **Iteration loop (POMDP)** — decode → Docker verify → observe failure → update refiner goal (INVALIDATE failed approach) → re-decode. The §2 loop assembled end-to-end. Convergence: stop when tests pass or budget exhausted. Each iteration narrows search space.
6. **Graph retrieval→decode** — retrieve graph node → feed `behavior_emb` (64d) directly to `film.modulate()` (bypass BehaviorEncoder) → decode → verify → observe. Architecture supports this: `modulate()` already takes bottleneck-d `b` directly. Closes the loop: past solutions help future problems.
7. **V6 kill-test** — does solving A help later B? Fresh graph vs graph-after-A on real tasks.

### Phase C — Bandwidth + scale
8. **MoLoRA** — z generates LoRA expert weights instead of scale+shift. Orders of magnitude more bandwidth. Only viable once refiner cos ≥ 0.78 (otherwise amplifies noise).
9. **Widen FiLM channel** — bottleneck 128 or 256 (currently 64). Intermediate step before MoLoRA.
10. **Full SWE-bench training** — ~2300 instances (not 116-722). Raise FiLM ceiling.
11. **Ingest agentic traces** — sequential structure for CONNECT + failure→revision for REFINE_PRECONDITION.
12. **Conditioned extraction (feedback loop)** — FiLM-conditioned Qwen re-reads code → refiner on new reprs → iterate (fixed-point inference).

### Phase D — Model scaling (AFTER pipeline proves out)
13. **Drop-in bigger model** — 7B/14B replaces 4B leaf. Pipeline + graph stay identical. Each leaf succeeds MORE → decomposition needed LESS → but graph still compounds. The system value (graph + iteration) ADDS to model capacity, not replaces it.

### Completed
- ~~FiLM gate test~~ — **PASSED** (ceiling-baseline=+0.131, decoder reads gold)
- ~~Presence vs meaning~~ — **RESOLVED** (z-content=+0.065, z-dropout essential)
- ~~Permuted control~~ — **DONE** (wrong-instance misleads: perm=0.167 < h_K=0.196)
- ~~Compositionality~~ — **VALIDATED** (intra 0.91, cross 0.58)
- ~~Topology kill-test~~ — topology flat at r=512 (value in directions, not sequencing)
- ~~Decode v1 soft prefix~~ — **FAILED** (replaced by FiLM)
- ~~Graph edits engine~~ — **BUILT** (10 edits, selftest PASS, V6-mini PASS)
- ~~Wire edits to real outcomes~~ — **DONE** (3 semantic issues found & fixed, all edit types fire)
- ~~Fable-5 HF integration~~ — **BUILT** (str_replace displacement pairs from HuggingFace, `--dataset fable5|lite+fable5`)
- ~~Truncation fix~~ — **FIXED** (max_new_tokens 256→512, format_fail 14%→0%)
- ~~Prompt framing~~ — **FIXED** (generic "Task:"/"Code:" for mixed SWE+Fable-5)
- ~~Diagnostic breakdown~~ — **BUILT** (`eval_on(diagnose=True)`, failure categories + raw samples + timing)
- ~~Training time optimization~~ — **BUILT** (`--arms` filter, 4.8h→2.2h or 30min diagnostic)
- ~~Qwen3.5-4B verification~~ — **RESOLVED** (d=2560 peaks at 0.584, far below cliff. Qwen2.5-3B confirmed as refiner backbone)
- ~~`_reprs()` mixed dataset~~ — **FIXED** (`lite+fable5` split on `+`, fable5 via `_load_fable5_texts`, with caching)

---

## Research audit — implementation readiness (2026-07-04)

Deep code audit of refiner, multi-leaf, FiLM decoder, and iteration loop. Findings shape the implementation plan.

### Refiner architecture bottlenecks (`lggn_refine.py`)

The refiner reaching cos 0.55-0.59 is NOT a hard ceiling — the architecture has multiple clear bottlenecks:

1. **No value projection in cross-attention** (line 159): attention weights applied directly to raw d-dim context vectors. Cannot recombine code features — only weighted average of raw hidden states. Fix: add Wv projection.
2. **24 ops in 2560d space**: delta at each step trapped in convex hull of 24 vectors. Tiny subspace. Even K=4 additive steps can't escape. Fix: increase n_op to 48-96, or use continuous displacement (free MLP showed comparable at r=512).
3. **Single head, scalar gates** (`gc`, `go` at line 148): one attention head, two shared scalars for all instances/steps/dimensions. Fix: multi-head attention + instance-dependent gating.
4. **No GRU** — despite "HRM recurrence" framing, it's just additive: `h_{k+1} = h_k + gc*a + go*delta`. No gated recurrence, no step encoding, no memory of prior steps. Fix: GRU cell or step-index embedding.
5. **No normalization**: h_t norm drifts over K steps. Fix: LayerNorm after each update.
6. **Context downsampled to 48 tokens** (t_ctx=48): 256-token sequence gets every ~5th token. Loses local structure. Fix: increase to 128-256.
7. **~5M params on ~160 instances** (80% of ~200): likely overfitting. Train/held gap would confirm. Fix: more data (722 with Fable-5), or regularization.
8. **Fix text is lossy**: gold f = last-token hidden of 6 removed + 8 added lines, 1500 char cap. Multi-hunk patches lose most content. This sets an UPPER BOUND on cos(h_K, f) — even a perfect refiner can't reach cos=1.0 against a lossy target.

**Key insight**: the refiner is surprisingly simple. Multiple low-hanging improvements available BEFORE architectural overhaul. Priority: (1) more ops or continuous displacement, (2) value projection, (3) multi-head, (4) more context tokens, (5) LayerNorm.

### Multi-leaf + FiLM wiring gap (5 specific gaps)

**Gap 1 — No per-operator latent.** Refiner produces ONE h_K per instance. Multi-leaf needs one latent PER leaf/step. The refiner's internal loop has intermediate `h_k` at each step — these ARE the per-leaf latents, just never exposed. `_extract_trajectory()` already walks the steps. Fix: expose `[h_0, h_1, ..., h_{K-1}]` from `Refiner.Net.forward()`.

**Gap 2 — FiLM decoder is training-only.** `_Decoder` creates model+LoRA+FiLM, trains, evaluates, destroys. No save/load checkpoint, no standalone generate method. Fix: add `save_checkpoint(path)`, `load_checkpoint(path)`, and `generate(prompt, z) → str`.

**Gap 3 — Prompt format mismatch.** Multi-leaf uses `fix_user()` (complex: issue+source+diagnosis+plan+hints). FiLM decode uses `_prompt()` (minimal: issue+code+SR instruction). FiLM decoder trained on one format won't work with the other. Fix: since FiLM z carries the plan semantics, use the simple prompt format — the plan moves INTO the latent, not the text.

**Gap 4 — `gen()` in swe_slot has no FiLM hook.** Multi-leaf loop (line 1973) hardcodes `inject=False`. Fix: accept a `film_decoder` object, replace `gen()` call with `film_decoder.generate(prompt, z=per_step_latent[k])`.

**Gap 5 — `lggn.solve()` doesn't carry latents.** `decode_fn` signature is `(prompt, task) → str`. No way to pass z. Fix: extend to `(prompt, task, z=None) → str`, populate `Operator.embedding` with behavior embedding.

**Estimated effort**: Gaps 1,3 = small (expose existing data, align on format). Gaps 2,4,5 = medium (new methods, CLI args, signature changes). No gap is architecturally hard — all are wiring.

### Iteration loop feasibility

**Key discovery: `lggn_loop.py` already exists** with the full POMDP skeleton:
```python
def solve_instance(task, rank_fn, realize_fn, verify_fn, iters=4):
    invalidated, feedback, trace = [], "", []
    for k in range(iters):
        plan = rank_fn(task, invalidated)          # next-best op not yet tried
        patch = realize_fn(task, plan, feedback)    # frozen LLM + prior feedback
        resolved, fb = verify_fn(task, patch)
        if resolved: return ...
        invalidated.append(plan.get("name"))        # INVALIDATE
        feedback = fb                               # carry observation forward
```

**What's ready (EASY):**
- `verify_one()` returns `(resolved, feedback)` — exact POMDP observe signal
- `_failure_feedback()` extracts structured info (which tests fail, targeted excerpts)
- Refiner is stateless — pass new `g` to re-invoke, trivial
- `_Decoder._set_z()` supports swapping z between generations
- `graph_edits.observe(Outcome)` accepts exactly what the loop produces

**What's missing:**
- **Goal update in latent space** (MEDIUM): compute `g' = hid(issue + failure_feedback)` via Qwen forward pass (~200ms). Simplest viable approach. `lggn.py` line 245 already does text-concat version.
- **Operator masking in refiner** (MEDIUM): add mask arg to softmax in `Refiner.Net.forward()` line 164. ~10-line change.
- **Learned belief updater** (HARD, LATER): trained observation encoder + update rule. Not needed for v1 — text-concat goal update is sufficient.

**Latency**: Docker verify = 2-10 min/instance. 4-iteration loop = 8-40 min/instance. Wall-clock constraint, not code constraint.

**Pragmatic path**: use `lggn_loop.py` skeleton → replace `rank_fn` with refiner trajectory → replace `realize_fn` with FiLM decoder generate → text-concat goal update → wire `graph_edits.observe()`. All connections are individually EASY-MEDIUM.
