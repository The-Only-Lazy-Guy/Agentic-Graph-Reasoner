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
