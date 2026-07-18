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
**Expected (speculative):** shallow (depth≤2) ~90%, depth 3–4 ~60–75%, depth 5 ~40–60%. This is the number that replaces the "oracle" in #1b.
**Risk:** underfit → scale data (10k+), or shrink the wiring grammar first.

---

## #3 — GraphGPS routing (topology MPNN + LapPE/RWSE)  (no-GPU; the real scale fix)
**Files:** extend `v5/runtime/algo_grr_struct.py` + `algo_grr_router`. Reuse `algo_grr_retrieval.TopologyRetriever` (the message-passing/depend-boost = the specific-edge signal), `struct_features` (LapPE/RWSE = global).
**Steps:**
1. **Fix the confounded scale test first:** in `algo_grr_router._synth_corpus`, replace the fixed-circle packing with **separated clusters + random distractors** (needed atoms stay well-separated as N grows). Re-run `--scale`; cosine should NOT collapse (confound removed).
2. Router atom features = concat[ mpnet-content , struct_features(adj) , one-hop topology-aggregated content ]. Train router on verified `(task→used-atoms)`.
3. Re-run `--scale` at 24/60/120/250/500 on the FIXED corpus.
**DoD / selftest:** router@3 **holds ≥0.85 at 500 atoms** on the de-confounded test (vs flat 0.22).
**Expected:** topology term does the heavy lifting (specific-edge), LapPE/RWSE adds the global/cluster prior; hierarchy optional if still sliding.
**Risk:** if it still slides, add the coarse-cluster gate (2-level) on TOP of GPS features (route to cluster → within).

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

## MHPP verify (after #1a; MOLAB)
Already built: `prep_mhpp.py` + `load_mhpp` + `verify_check` (sample 3/3 validated).
```
python -m v5.runtime.prep_mhpp --hf <mhpp-hf-path> --out artifacts/mhpp.jsonl   # self-validates canonicals
python -m v5.runtime.algo_grr_mbpp --run --lm Qwen/Qwen2.5-3B-Instruct --topo --mhpp --limit 120
```
Once MembraneV2 (#1a) lands, point it at `--mhpp` too. **Expect base-3B low, memory helps on decomposable, ceiling-bound elsewhere** — the honest hard-eval.

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
