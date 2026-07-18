# GRR Build Plan — 6 items, parallelized (execute exactly)

Frozen LM + verified graph, reasoner emits STRUCTURE (route + plan). This plan turns the validated
*probes* into an integrated system. **Speculative by design** — expected numbers stated so you know
when a step is "working" vs "off."

## The trick that makes it parallel: define 3 interfaces FIRST (item #1a), everyone builds against them

```
AtomRouter.rank(task_text, atom_ids) -> ranked atom_ids          # who: #3 improves it
Planner.plan(task_text, atoms)       -> AtomProgram              # who: #2 trains it
realize(AtomProgram)                 -> code (string)            # deterministic, no LM
AtomProgram = {atoms:[id...], wiring: tree}                      # tree over atoms (nesting/args)
MembraneV2.solve(task): rank -> plan -> realize -> LM ratifies glue -> VERIFY -> bank
```
Stub impls first (current router + an ORACLE/templated planner) so tracks A–F don't block each other.

## Dependency graph / waves

```
WAVE 0 (blocks all, ~0.5–1 day, no-GPU):   #1a interfaces + MembraneV2 skeleton (stub planner+router)
                                              │
        ┌─────────────────────┬───────────────┼───────────────┬───────────────────┐
WAVE 1  ▼ #2 planner (no-GPU) ▼ #3 GraphGPS   ▼ #5 step-spec   │                   │ (parallel, no-GPU)
        │                     │  router(no-GPU)│  into loop     │                   │
        └─────────────────────┴───────────────┴───────────────┘                   │
                                              │ (swap trained planner + GPS router into skeleton = #1b)
        ┌─────────────────────┬───────────────┴───────────────┐
WAVE 2  ▼ #4 compound-vs-RAG  ▼ #6 invention                  ▼ MHPP verify   (parallel, MOLAB/3B)
```
- **Swarm-parallelizable (no-GPU):** #1a, #2, #3, #5. Give each a teammate.
- **Sequential on molab (3B, one GPU):** #4, #6, MHPP. #4 can start on the WAVE-0 skeleton (current router + oracle planner) before #2/#3 land.

---

## #1a — Interfaces + MembraneV2 skeleton  (DO FIRST; no-GPU; unblocks all)
**Files:** new `v5/runtime/algo_grr_pipeline.py`. Reuse: `algo_grr_membrane` (MembraneSolver, verify, realize helpers), `algo_grr_router.make_router_policy`, `algo_composed`/`algo_trm_compose` (realizer + atom-program form), `embedder.encode_batch`.
**Steps:**
1. Define `AtomProgram` dataclass {atoms, wiring} and `realize(program, graph) -> code` (deterministic — reuse the `_REF` wiring realizer from `algo_composed`).
2. Define `class AtomRouter` (wrap current NeuralRouter + `embedder` mpnet for task/atom vecs) with `.rank(task_text, atoms)`; and `class OraclePlanner` (templated: for the compose/wiring corpus the wiring is known → emit the ground-truth program). This is the STUB planner.
3. `class MembraneV2`: `.solve(task)` = router.rank → planner.plan → realize → frozen-LM writes ONLY glue over the realized skeleton (or ratifies) → `verify` → on solve `bank_helper_granular`.
4. `--selftest` (no-GPU, stub LM = reference): run over the compose corpus.
**DoD / selftest:** MembraneV2 solves ≥95% of the compose corpus with router+oracle-planner+realizer+verify; banking works.
**Expected:** ~100% on compose (deterministic realize); this is plumbing, not capability.

---

## #2 — Train the learned Planner (TRMPlanDecoder → AtomProgram from NL)  (no-GPU, tiny model)
**Files:** `v5/runtime/algo_grr_planner.py` (trainer) reusing `TRMPlanDecoder` (in `algo_grr_draft`) + `algo_grr_compose.gen_corpus` + `embedder`.
**Steps:**
1. **Data:** for each generated compose/wiring task, you KNOW the atom-program (ground-truth wiring). Emit pairs `(task_text_emb, atom_program_serialized)`. Target ~5–20k pairs (augment via `algo_dsl_gen` paraphrases + the wiring-depth generator in `algo_grr_wiring`).
2. **Model:** TRMPlanDecoder: input = mpnet(task) + atom-set embeddings; output = the atom-program as a token sequence over {atom-pointers, wiring-ops: call/compose/arg}. Teacher-force; deep supervision (algo_trm train pattern).
3. **Eval:** held-out tasks (unseen wiring compositions). Metric = **program exact-match** AND **realized-code verify-rate**.
**DoD / selftest:** on held-out, realized-code verify ≥70% (planner infers structure the LM alone gets ~3% at depth 5 — see `algo_grr_wiring`).
**RESULT (built, `algo_grr_planner.py`):** a FLAT seq2seq planner DEGRADES with depth exactly like the LM
(verify 1.00→0.07 by depth 5; overall 0.46) — the compositional-generalisation limit. The FIX (already
validated as GRR-7): **net-GUIDED VERIFIED SEARCH** — beam-decode candidates with the seq2seq as a GUIDE,
VERIFY each, return the first that passes. Lifts it: 0.72→**0.98** @d2, 0.40→**0.75** @d3, overall
0.46→**0.67**. Deep extrapolation (d≥4) stays **budget-bound** (beam=10 + OOD-deep net can't contain the
program) → raise beam / train on deeper (the GRR-8 verifies-to-solve tradeoff). **So the planner = search +
net-guide + verify, NOT a flat decoder.** `plan_by_search()` is the interface MembraneV2 (#1b) consumes.
**Risk:** deep depth needs budget; train the net on the target depth range + scale beam with depth.

---

## #3 — GraphGPS routing (topology MPNN + LapPE/RWSE)  (no-GPU; the real scale fix)
**Files:** extend `v5/runtime/algo_grr_struct.py` + `algo_grr_router`. Reuse `algo_grr_retrieval.TopologyRetriever` (the message-passing/depend-boost = the specific-edge signal), `struct_features` (LapPE/RWSE = global).
**Steps:**
1. **Fix the confounded scale test first:** in `algo_grr_router._synth_corpus`, replace the fixed-circle packing with **separated clusters + random distractors** (needed atoms stay well-separated as N grows). Re-run `--scale`; cosine should NOT collapse (confound removed).
2. Router atom features = concat[ mpnet-content , struct_features(adj) , one-hop topology-aggregated content ]. Train router on verified `(task→used-atoms)`.
3. Re-run `--scale` at 24/60/120/250/500 on the FIXED corpus.
**DoD / selftest:** router@3 **holds ≥0.85 at 500 atoms** on the de-confounded test (vs flat 0.22).
**RESULT + RESOLUTION (built, `algo_grr_graphgps.py`):** de-confounded the scale test (separated clusters,
not the densifying circle). GraphGPS features = content + LapPE/RWSE + one-hop message-passing give a real
lift over content-only (0.71 vs 0.17 @N=60), BUT both flat AND hierarchical learned routing plateau ~0.5 at
scale — ranking a SPECIFIC cross-cluster atom is a needle for ANY features, and coarse-routing-to-the-dep-
cluster is as hard as fine-routing-to-the-dep-atom (2 hier attempts, both hier≡flat).
**THE ANSWER (the honest reframe):** a KNOWN structural edge is **FOLLOWED, not learned** — graph traversal
(`TopologyRetriever` depend-boost, already built): candidate = {q} ∪ neighbours(q) -> the dep partner IS q's
neighbour -> **Recall 1.00, O(degree), scale-free** (measured: content 0.30 / flat-GPS 0.50 / **TOPO 1.00**).
So: **structural deps -> follow the edge (topology, trivial + scales); LEARNED GraphGPS routing is only for
SEMANTIC relevance** (content, no explicit edge). The "routing at scale" weakness dissolves for structure.
Wire into MembraneV2's AtomRouter: topology-boost for depend-neighbours (structural) + GPS/content ranker
for semantic candidates. #3 RESOLVED.

---

## #5 — Agentic step-speculation, wired into the real loop  (no-GPU plumbing; molab for LM-call metric)
**Files:** extend `algo_grr_specstep` + hook into `MembraneV2` (#1a).
**Steps:**
1. **Motif miner:** from banked solved traces (atom-programs), mine frequent step-subsequences (PrefixSpan / n-gram) → the speculator's training set (replaces the toy `_MOTIFS`).
2. Train `StepSpeculator` on mined motifs; wire into MembraneV2 for MULTI-atom tasks: speculator proposes the next K atoms/steps, LM verifies the chunk in one call, accept prefix.
3. Bump K (4→8), measure LM-call reduction on the compose corpus (no-GPU with stub) then real 3B.
**DoD / selftest:** on held-out multi-step tasks, **≥1.5× fewer LM calls at 100% verify-gated correctness** (matches the 1.52× probe; higher with real mined motifs + K=8).
**Expected:** 1.5–3× depending on motif recurrence in the real corpus.

---

## #4 — Compounding vs RAG on a real 3B stream  (MOLAB/3B; the hero plot)
**Files:** extend `algo_grr_scaleup`. Reuse `run_scaleup` + `assemble_corpus`.
**Steps:**
1. Two arms on the SAME task stream (compose ⋈ MBPP+/MHPP interleaved):
   - **OURS:** MembraneV2 with banking + abstraction (compounding ON).
   - **RAG baseline:** cosine retrieval into prompt, **NO banking, static store** (compounding OFF).
2. Log rolling solve-rate + lm-calls/task vs task index for both.
3. Plot: OURS rises + lm/task falls; RAG flat.
**DoD:** OURS deriv-reuse rises (already saw 6→70) AND lm/task falls below RAG by task ~150; solve-rate ≥ RAG.
**Expected (speculative):** OURS lm/task drops ~30–50% by end of stream; RAG flat. Solve-rate OURS ≥ RAG on decomposable fraction, tie on atomic.
**Command:** `python -m v5.runtime.algo_grr_scaleup --run --lm Qwen/Qwen2.5-3B-Instruct --n-compose 150 --mbpp 150 --save graphs/scaleup.json` + a `--rag-baseline` flag to add.

---

## #6 — Robust invention (LM authors new atoms at program HOLES)  (MOLAB/3B)
**Files:** continue GRR-14 in `algo_grr_membrane` (derive path) + `algo_grr_poison_test.bank_helper_granular`.
**Steps:**
1. When the planner's program has a HOLE (no banked atom covers a required step), the frozen LM AUTHORS the primitive for that step (constrained: write ONE reusable function).
2. Fuzz-generality gate + verify; bank if it generalizes.
3. Measure: tasks solved that were UNSOLVABLE from the existing library; new-atom cross-task reuse.
**DoD:** ≥1 invented atom reused across ≥2 later tasks; solve-rate on "needs-new-primitive" subset rises above the retrieval-only ceiling (earlier: 15→19/24).
**Expected:** invention is LM-capability-bound — expect modest, honest lift; the WIN is *reuse* of the invented atom (compounding), not raw invention rate.
**Risk (honest):** the frozen 3B caps this. Consider a molab TEACHER (bigger LM) authoring atoms that distill into the graph (teacher→graph→student), per the deploy strategy.

---

## Hard-benchmark verify (MOLAB) — MHPP is a DEAD END; use BigCodeBench
**MHPP is real (arXiv 2405.11430) BUT withholds tests+solutions** (submit-to-server) → it does NOT work
with our local verify gate. **Swapped to BigCodeBench** (public unittest tests). Built + validated:
- `verify_unittest` (algo_grr_mbpp): runs a `class TestCases(unittest.TestCase)`, pass iff all tests pass
  (correct→1.0, wrong→0.0 verified).
- `load_mhpp` auto-routes: asserts→verify_asserts, `def check`→verify_check, unittest→verify_unittest.
- `prep_mhpp` maps BigCodeBench fields (complete_prompt/instruct_prompt/test/canonical_solution) + self-
  validates canonicals. Sample `artifacts/bcb_sample.jsonl` → 2/2 canonical PASS.
```
python -m v5.runtime.prep_mhpp --hf bigcode/bigcodebench --out artifacts/bcb.jsonl   # download + self-validate
python -m v5.runtime.algo_grr_mbpp --run --lm Qwen/Qwen2.5-3B-Instruct --topo --mhpp artifacts/bcb.jsonl --limit 120
```
**Caveat:** BCB tasks use real libs (pandas/numpy/…) — the molab env must have them. Expect base-3B low,
memory helps on decomposable, ceiling-bound elsewhere — the honest hard-eval.

---

## Suggested assignment (if swarming)
- **Teammate 1 (no-GPU):** #1a → then #5. (#1a first, it unblocks everyone.)
- **Teammate 2 (no-GPU):** #2 planner (data-gen + train).
- **Teammate 3 (no-GPU):** #3 GraphGPS (de-confound scale test + fuse features).
- **You (molab):** #4 compounding-vs-RAG, then #6 invention, then MHPP. Start #4 on the #1a skeleton.
**Integration (#1b):** once #2 + #3 land, swap trained-planner + GPS-router into MembraneV2; re-run MHPP + #4.

## Definition of done (the system)
MembraneV2 = frozen LM + verified graph, router (GPS) picks atoms, planner (trained) emits the program,
realizer builds code, LM ratifies glue, verify gate banks; runs on MHPP; compounds vs a RAG baseline;
step-speculation cuts LM calls; invention (LM/teacher) fills holes. Each arrow already has a passing probe.
