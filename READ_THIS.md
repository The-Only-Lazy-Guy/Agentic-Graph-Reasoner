# READ_THIS — GRR: Graph Recursive Reasoner (2026-07-13)

> At-a-glance dump of the latest session (raw numbers, decisions, repro commands).
> Updated each working session. Branch: fix/swe-slot-plan-gate-real-file.

## What GRR is
A tiny OWNED reasoner (not a language model) whose vocabulary + memory ARE a verified, self-compressing
graph. Content lives symbolically in the graph (executed); strategy lives in the latent. Rewarded for
its own future capability, not for matching us. Core bet — **memory is load-bearing only for a model
FORCED to compose** — is PROVEN and EMBODIED.

## The stack — all built + real-tested this session (no stubs-echoing-themselves)
| module | what | proof |
|---|---|---|
| algo_quality (GRR-1) | fuzz-generality store-gate | rejected 31% overfit in the real harvest |
| algo_graph_reason.Budget (GRR-2) | budget in wake loop | budget really halts the solve |
| algo_capability (GRR-3) | counterfactual Δcapability | real ablation drops solve; coupled to fuzz bar |
| algo_abstract (GRR-4) | MUTATE + ABSTRACT | mutate repairs 25→100%; str_dp2 fuzz-equiv + MDL-compresses |
| algo_sleep (GRR-5) | gated compress + prune | lossy rewrite VETOED by fuzz-equivalence |
| resolve_deps (GRR-5b) | transitive dep graph-walk | abstractions run + get credited |
| algo_composed (GRR-6.1) | compose-forced solver | **str_dp2: Δ0 (free 4B) → TOP atom Δ+0.50** |
| algo_trm_compose (GRR-6.2/3) | TRM policy + realizer | trained from scratch, drives compose-forced solve |
| algo_dsl + algo_dsl_trm (5b) | combinator DSL + program decoder | 86% synthetic / 100% mpnet |

## Key numbers (real 4B / real mpnet)
```
cross-attend adapter (earlier): compose +43% @200tok, +5% @400tok  = AMORTIZATION, not capability
Δcapability (hard graph, general): only lcs_length/edit_distance load-bearing; rest replaceable
str_dp2 under compose-forcing: Δ+0.50, TOP atom (vs Δ0 free-form 4B)  <-- THE thesis
DSL program decoder (mpnet): 100% held-out INSTANCES (base + dp2 graphs)
compo-gen (leave-one-family-out, mpnet): 0/66 = 0%  <-- IMITATION alone is RECALL, not reasoning
GRR-7 STaR (search+verify+consolidate), REAL MPNET --compo-gen-star --hard, leave-one-family-out x3 seeds:
                   zero-shot(recall)=0%  ->  with-search(reasoning)=100%  (ALL 6 families 100% [min 100,
                   max 100] every seed) <-- recall->reasoning, PERFECTLY STABLE. matches synthetic selftest.
                   (first real run: 89%, max_lis wandered 0-100%; fixed by MDL-minimal keep-gate, see below)
(superseded) RL+GRPO dense reward: discovered structure once but HIGH VARIANCE / unstable -> replaced by STaR
```

## Honest frontier
- Imitation alone → RECALL (memorizes family→program; 0% compo-gen on synthetic AND mpnet). GRPO was the
  unstable fix (the wanderer); worse, pure policy-sampling can't even EXPLORE an unseen family.
- GRR-7 = STaR / expert iteration: SEARCH (bounded enumeration of valid pipelines) -> VERIFY oracle I/O
  (never the reference program) -> KEEP fuzz-general + MDL-MINIMAL (shortest solving length, dedup by
  realized code) -> SFT the net so decode amortizes it. Search does the reasoning; the net consolidates.
  Deterministic search => ZERO variance (GRPO pain solved). REAL mpnet: 0% recall -> 100% with-search,
  all 6 families 100% every seed. The MDL-minimal keep-gate was load-bearing: without it the search kept
  weak-input-only variants (digit_sum-as-identity on single-digit values) that pass search but fail eval
  fuzz -> max_lis wandered 0-100%; minimality drops them (they're longer than the true ref) -> 100% stable.
  The NET alone is still recall zero-shot; the SYSTEM (net+search) generalizes — the honest DreamCoder-wake framing.
- GRR-8 (design-complete pass) = ALL THREE next-levers BUILT + selftested:
  (1) net-GUIDED search + verify budget (algo_dsl_trm._guided_search): verifies-to-solve 54 -> 5 after
      consolidation (selftest [5]) — amortization measured, search scales past brute force;
  (2) solve_with_search: the design's retrieve->reason->VERIFY->update->retry-UNTIL-SOLVE inference
      primitive (decode -> guided search -> consolidate; via flips search->decode);
  (3) algo_grr_loop.py = THE unified wake/sleep compounding loop over ONE graph: wake -> consolidate ->
      SLEEP writes discovered programs INTO THE GRAPH (impl nodes, pipeline SYMBOLIC in metadata, depend
      edges to atom closure, health-gated) -> re-index -> measure. Selftest: zero-shot RISES 4/6 -> 6/6
      fams while verifies-to-solve FALLS 8.8 -> 1.0; --rebuild-net: FRESH net + graph only -> 6/6
      (the graph IS the memory — survives net/box resets; the net just re-amortizes);
  (4) harder task inputs (arrays len<=10, vals<=49): weak-input degenerates now fail fuzz directly.
  Also fixed a selftest keyword collision ("increasing" before "subsequence") that had collapsed
  max_lis+sum_lcs to one embedding — explains the old synthetic-86%-vs-mpnet-100% gap.
- Deferred to SCALE-UP by design (user call): more dataset, RL-with-LM synergy (LM authors new atoms),
  hierarchical tasks (--programs-as-atoms wins), RGCN structured read.

## Repro (molab, no 4B — mpnet + tiny nets, minutes)
```
python -m v5.runtime.algo_composed --selftest                    # thesis: str_dp2 load-bearing
python -m v5.runtime.algo_dsl_trm --compo-gen-star --hard --graph graphs/algo_reason_hard.json  # 0% -> 100%
python -m v5.runtime.algo_grr_loop --loop --graph graphs/algo_grr_loop.json     # THE loop (compounding table)
python -m v5.runtime.algo_grr_loop --rebuild --graph graphs/algo_grr_loop.json  # graph-only net rebuild
# every module: python -m v5.runtime.<algo_quality|algo_capability|algo_abstract|algo_sleep|algo_graph_edits|algo_graph_mg|algo_compose_tasks|algo_composed|algo_trm_compose|algo_dsl|algo_dsl_trm|algo_grr_loop|algo_meta|algo_anticheat> --selftest   # 14/14 PASS
```

## Files (all v5/runtime/)
algo_quality · algo_capability · algo_abstract · algo_sleep · algo_composed · algo_trm_compose ·
algo_dsl · algo_dsl_trm (STaR + _guided_search + solve_with_search; legacy train_rl kept for comparison) ·
algo_grr_loop (GRR-8: wake_sleep_loop/rebuild_net/_sleep_store) · algo_meta · algo_anticheat.
Reuses: algo_graph_mg (MGRetriever.resolve_deps), algo_compose_tasks (_REF/_NEEDS/ALL_ATOMS/gen),
algo_graph_edits+graph_grower (health-gated writes), subgraph/gnn_encoder/goal_encoder (read stack).
Trained: artifacts/grr6_trm.pt, artifacts/grr6_dsl.pt.

## DEPLOYMENT CONSTRAINT (hard, user): <= 6GB VRAM. Big LMs (32B/72B class) = OFFLINE TEACHERS only,
##   never deployed. The GRR stack IS the distillation channel (graph-mediated: teacher -> gate ->
##   symbolic nodes -> rebuild tiny net). Deployed stack = mpnet ~220MB fp16 + TRM <1MB + graph (CPU)
##   = <0.5GB. Teacher upgrades are drop-in re-runs with zero deployment change.
## D2 FREEZES (10-day final-architecture clock, decided 2026-07-15):
##   TRAINER = gru. Lower-ceiling h2h (48 fams, para 2/3, ceiling 92->84%): gru 84% == vib 84% (dead
##   tie, mixed per-seed) — VIB's earlier every-seed edge was ceiling-inflated; honest reversal,
##   simplicity wins, vib stays behind a flag. recursive 83%.
##   VQ = my implementation BUG, not a concept refutation: the "shrunk residual" term
##   resid_scale*(mu - mu.detach()) is VALUE-ZERO in forward -> z collapsed to 32 prototypes for 48
##   fams -> 9%. Correct form: proto + s*(mu - proto). PARKED with bug documented (clock).
##   HINT = ON (reasoning-state sketch): lm discoveries 6 -> 9 (+50%), zero-shot 24 -> 26/48, third
##   consecutive positive, never harmful.
## GRR-16 SECOND-BRAIN COUPLING (both user ideas, real-3B/real-mpnet measured):
##   (a) sketch-hint (TRM thought -> LM prompt as confidence-gated TEXT; latent FiLM/cross-attn/KV
##   ruled out earlier: z-wall + amortization-not-capability): intent-tier A/B lm discoveries 6 -> 8,
##   zero-shot 26 -> 28/48, FASTER early consolidation, NO harm (ads lesson held via tau-gate + soft
##   phrasing). Single seed each: suggestive positive, kept ON (--lm-hint).
##   (b) VIB loss (user objective min I(z;task) max I(z;solution), variational form: stochastic goal
##   encoder + beta*KL; decode = mu, parameter-identical at inference): held-out phrasings 93.2% vs
##   gru 91.7% vs recursive 90.1%; vib >= gru in EVERY seed (worst vib >= best gru). Small (+1.6pp at
##   a 92% ceiling) but sign-consistent + theoretically right -> ADOPTED as trainer arch of choice.
##   True effect size needs a lower-ceiling benchmark (fewer train phrasings / more fams).
## GRR-14 RAW-INSPECTION PAYOFF (user push: inspect the pipeline, don't argue from aggregates):
##   raw dump decomposed the 9 zero-solves -> 5 were OUR defects: 2x prep bug (entry `set` extracted
##   from `assert set(inner(...))` — model read the intent RIGHT and our harness NameError'd it),
##   1x case mismatch (find_volume vs find_Volume), 2x prose-in-code-fence (extractor swallowed prose).
##   Fixes (entry-name from reference defs; repair_code = compile-trim + case alias; gate untouched):
##   67% -> 78% (93/120), syntax failures 10 -> 0, curve flat. ~82% plain-MBPP-equivalent, stock 3B,
##   best-of-4-with-verifier. Honest residual: 21 assert_fails (~17%) = the real capability gap —
##   the "undertrained" hypothesis now testable clean (STaR/LoRA on the 93 verified solutions).
##   REAL cross-task reuse on MBPP: 0 (post-regex-fix) — still unproven.
## GRR-14 ABLATION VERDICT (the decline investigation, 3-arm, real 3B, 120 MBPP+ tasks):
##   off (graph ablated): 80/120 = 67%, curve flat/bumpy — NO decline. sig (status quo bare-sig ads +
##   hard call-these directive): 46/90 = 51%, marginals 14,12,8,8 — monotonic decline as graph grows.
##   => the ADVERTISEMENT channel was net-negative (~16pp by task 90) and CAUSED the decline; the
##   graph's memory role unaffected (80 atoms banked in off). Default now ad-style=off.
##   plus_only_fail=4 -> dense gate costs only ~3pp vs plain MBPP: true 3B capability ~70%
##   plain-equivalent (published 7B range); earlier 55% was ads-harm not model weakness.
##   Reuse metric had false positives (`lst.count(` matched atom `count`) -> fixed; REAL cross-task
##   reuse on MBPP = 0 so far (unproven). PENDING: purpose arm (repaired ads: sig + purpose line +
##   soft directive) vs off — can advertisement EVER pay here?
## GRR-14 INVENTION RUNG (LM authors NEW ATOMS, real 3B on real MBPP+, 5m17s):
##   baseline first (algo_grr_inspect --mbpp-baseline): current ladder on MBPP+ = 1/40 (2%) — outside
##   its atom vocabulary the system is dead. With authoring: 39/60 SOLVED (65%, 32x baseline), 39 atoms
##   banked (origin=lm_author, health-gated, depend edges). Cross-task reuse was 0 — diagnosed: banking
##   unit was whole solutions under entry-point names (nobody calls another task's entry point) ->
##   FIXED: STORE-action helpers now bank as their own atoms (origin=lm_author_helper). Full-378 run
##   with helper granularity = the real reuse measurement (pending).
## GRR audit (user ask, algo_grr_inspect, local real mpnet): census 26 nodes = 4% NL-only concept /
##   35% pure-code atoms / 62% code+SYMBOLIC-pipeline+NL programs; 96% carry executable code; NL is
##   descriptive retrieval keys, never how-to prose; triviality ~20% (incl. one degenerate-but-CORRECT
##   minimal count program — MDL ignoring a red-herring hint = the gate working). Raw trace: rebuilt
##   net decodes consolidated fams at 1.00 head confidence -> realize -> graph-walk deps -> verified.
## GRR-13 REASONING-vs-TRANSLATION (intent tier, real 3B, molab): texts describe WHAT never HOW
##   ("exactly two positive divisors", zero method vocabulary). RESULT vs method tier:
##     lm discoveries 11 -> 7 (-36%) | lm held-out reuse 91% -> 57% | beam control 8 -> 8 (unchanged =
##     experiment valid). VERDICT: the 3B genuinely REASONS (7 fams from pure intent, incl 4-step
##     chains) but translation carried ~40% of method-tier performance. The delta is now a measured
##     benchmark for bigger models (drop-in via --lm).
## GRR-13b MBPP+ PREPPED (real open-source corpus): 378/378 kept, 0 dropped at validation (every
##   reference passes its full EvalPlus test script in subprocess); pipeline-shaped 163 / LM-author
##   territory 215. artifacts/mbpp_plus_prepped.jsonl COMMITTED (survives resets). Step-2 hunting ground.
## GRR-12 LM PROPOSER (real Qwen2.5-3B-Instruct + real mpnet, 24 fams, 6m28s incl. model download):
##   escalation ladder decode -> beam+eps -> LM (task TEXT only -> candidate pipelines -> SAME verify
##   gate, MDL-first, origin=lm). RESULT: 19/24 banked — the twice-measured blind-search ceiling (15/24)
##   BROKEN by language understanding. Provenance: lm 11 discovered -> 11 reused (20/22 held-out inst,
##   incl. the gen5/gen6 deep resisters), beam 8 -> 8 (14/16). Round 0 alone: 19 discoveries.
##   "LM teaches ONCE, graph+TRM remember FOREVER" = demonstrated with a real model.
##   zero-shot 15/24 fams (34/48 inst); rebuild 12/24 (31/48). Open: 5 fams resist LM k=6; late-find
##   under-consolidation (19 banked vs 15 full-zero-shot). Next: LM authors NEW ATOMS (step 2), then
##   reconsolidation (#71).
##   Repro: --loop --factory --families 24 --rounds 8 --budget 800 --n-wake 3 --sft-steps 1200 \
##          --explore 8 --lm Qwen/Qwen2.5-3B-Instruct --lm-k 6
## GRR-9c FACTORY LOOP (real mpnet, 24 generated fams, paraphrased goals):
##   beam-only STALLED at 10/24 (rich-get-richer, twice reproduced) -> EPSILON slots fix discovery.
##   PERF: 1h40m -> 3m46s (~26x): vectorized level scoring (_score_pipes_batch, one forward per level;
##     the TRM is 150k params on CPU — cost was python-loop overhead, NOT compute; the 90GB VRAM matters
##     at the LM phase) + batched embed cache + search-once-per-stuck-fam-per-round.
##   FINAL RUN (3m46s): 15/24 banked | zero-shot 8/24 fams (21/48 held-out inst) |
##     rebuild-net 9/24 (24/48) — a fresh net trained PURELY from the graph BEATS the online net.
##   PROVENANCE->REUSE (per-node origin/found_round; "does exploration find load-bearing programs?"):
##     run A: beam 10/10 reused (19/20 inst), eps 5 disc/3 reused | run B: eps 8 disc/7 reused (11/16),
##     beam 7 disc/6 reused (10/14) -> epsilon ~ HALF of all knowledge, definitively load-bearing.
##   SEARCH CEILING (knob-sweep verdict, 14m28s run): 4x budget (3000) + 2x epsilon (16) + 14 rounds =
##     SAME 15/24 — budget is NOT binding. Everything discovered is perfectly amortized (beam 12/12
##     reused 24/24 inst, eps 3/3 6/6; zero-shot 15/24 = exactly the discovered set; rebuild 12/24).
##     The 9 resisters are structural: a len-6 program's 5-prefix must survive 4 consecutive beam
##     prunings (~5e-5/search) — the blind-search wall. THE LM-PHASE MOTIVATION: those fams' texts
##     ("sum of the digit-reversal of the square of...") are compositionally PARSEABLE — an LM proposer
##     emits the pipeline directly, same verify gate.
##   Repro: python -m v5.runtime.algo_grr_loop --loop --factory --families 24 --rounds 8 \
##            --budget 800 --n-wake 3 --sft-steps 1200 --explore 8 --graph graphs/algo_grr_factory.json
## GRR-9 h2h VERDICT (real mpnet, 32 factory fams, held-out-PHRASING eval, 3 seeds):
##   gru 92% (153k params) vs recursive(TRM-merge) 90% (186k) -> TIE, no length bucket favors
##   recursion -> GRU stays production; recursion + F-bank PARKED (revisit: len10+/adaptive-T/nested DSL).
##   Benchmark lesson: fixed-text recall saturates (100% = point memorization); PARAPHRASE held-out
##   is the real generalization test — both nets ~90% there (decoder maps meaning REGIONS, not points).
## GRR-7 DONE: real mpnet 0% recall -> 100% with-search, all 6 families stable every seed.
## GRR-8 DONE + REAL-MPNET CONFIRMED (molab): zero-shot 4/6 -> 6/6 fams (18/18 inst) while
##   verifies-to-solve 13.2 -> 1.0 (r2: ZERO searches); rebuild-net: fresh net + graph only -> 6/6.
##   Matches the synthetic selftest exactly. 14/14 selftests PASS. Graph: graphs/algo_grr_loop.json
##   (15 nodes 22 edges, 6 program nodes banked) — committed? artifacts/ dies at box reset; the graph
##   json is re-derivable in ~2 min via --loop (deterministic search), so no artifact dependency.
## NEXT = SCALE-UP: more dataset + RL-with-LM synergy (LM authors new atoms) + hierarchical tasks.


# ═══════════════════════════════════════════════════════════════════════════════
# GRR-Tool — TRM-Driven Reasoner with Tool MLPs (design, 2026-07-15)
# ═══════════════════════════════════════════════════════════════════════════════

## Core idea
TRMReasoner (tiny ~7M net) produces step-by-step reasoning traces (latent states z_1..z_T).
Each step: tool MLPs consume the TRM's reasoning state → execute graph operations → results
feed back into the next step. The final trace is fed to the LM which decodes it into
answers + code + explanations. The LM is the *realizer*; the TRM is the *reasoner*.

Each graph node has:
- `text`: concise purpose string (the retrieval key, e.g. "computes prime factorization")
- `metadata.code`: executable implementation
- `depend` edges: which atoms this one calls (composition structure)
The step-by-step *reasoning* lives in the TRM trace (z_t states), NOT in the nodes.
Nodes carry what they DO and their DEPENDENCIES — the TRM figures out the ORDER/WHY.

### Inference is a CLOSED LOOP (not linear)

```
TRM reasons with tools ──→ LM decodes code ──→ Verify ──→ Solved? ──→ DONE (→ SLEEP)
       ↑                                                        │
       │                                  ┌─────────────────────┘
       │                                  │ (fail: error + failed code + partial trace)
       └────────── FEED BACK ─────────────┘
                         
```

The TRM gets to "think again" with failure context. This mirrors the existing
`solve_with_search` pattern (algo_dsl_trm.py:551): decode → if fail → search →
if found → consolidate → if still fail → LM proposer. The new tool MLPs make
the TRM's reasoning visible and steerable at each step, but the outer closed
loop stays the same.

### Node explanations (what each atom carries)

Each implementation node stores:
- `text`: one-line purpose (retrieval key, ~200 chars)
- `metadata.code`: the full implementation
- `metadata.kind`: "authored" | "program" | "helper"
- `depend` edges: closure of atoms this one calls
- `part_of` edge: which concept domain it belongs to

The step-by-step *explanation of how to use it* is NOT stored in the node —
it emerges from the TRM's reasoning trace during inference. The TRM learns
WHEN and WHY to call each atom; the atom only stores WHAT it does.

## Key design decisions (agreed 2026-07-15)
1. TRM scratchpad z_t should represent *search state, partial hypotheses, uncertainty* —
   not just a blended feature. Accomplished by: larger d (256+), auxiliary prediction
   heads (confidence, usefulness, "did the last retrieval help?") that force the latent
   to encode these quantities.
2. Tool MLPs are NOT one affine transform — they use residual blocks or small
   transformer blocks (Linear → GELU → Linear → GELU → residual), ~10K params each.
3. Retrieval is ITERATIVE, not one-shot cosine. MLP_ret produces a query vector →
   TraversalRanker retrieves → result feeds back → TRM decides "good enough" or
   "another hop" → MLP_stop gates whether to continue. This makes retrieval compositional:
   "need theorem A → need theorem B → need impl C".
4. Health gate is PROBABILISTIC: write_prob = σ(α·confidence + β·novelty + γ·verification - δ).
   Verified solutions still usually write, but the TRM can choose NOT to bank something
   it's unsure about. Prevents pollution even when the test gate passes (degenerate
   solutions). Learned, not hardcoded.

## Architecture

```
Task text ──→ mpnet ──→ x_vec [768]
                           │
                    ┌──────▼──────────┐
                    │ TRMReasoner      │  T recursion steps (d=256)
                    │ z_t: search      │  each step: attend atoms,
                    │ state, partial   │  refine scratchpad, produce
                    │ hypotheses,      │  tool-feedback vector for
                    │ uncertainty      │  next step
                    └──────┬──────────┘
                           │
              ┌────────────┼─────────────┐
              │            │             │
         ┌────▼────┐ ┌────▼──────┐ ┌────▼──────┐
         │Residual  │ │Residual   │ │Residual   │ ...
         │MLP_ret   │ │MLP_write  │ │MLP_edge   │
         │(3 layer) │ │(3 layer)  │ │(3 layer)  │
         └────┬────┘ └────┬──────┘ └────┬──────┘
              │            │             │
         ┌────▼────┐ ┌────▼──────┐ ┌────▼──────┐
         │Traversal│ │  LM       │ │  grow     │
         │Ranker   │ │  decodes  │ │(edits)    │
         │(iter-   │ │  code     │ │           │
         │ ative)  │ │  text     │ │           │
         └────┬────┘ └───────────┘ └───────────┘
              │            │             │
              └────────────┼─────────────┘
                           │ Each tool result feeds BACK
                           ▼ into next TRM step (z_{t+1})
                      ┌──────────┐
                      │  Graph   │  MemoryGraph (nodes + edges)
                      │  (CPU)   │  persistence, retrieval, composition
                      └──────────┘
                           │
                    TRM trace (z_1..z_T + tool outputs)
                           │
                           ▼
                    LM decodes final answer + code + explanation
```

## Components

### TRMReasoner (algo_trm.py)
- `d=256` (was 64), `T=5` (was 3)
- `forward(x_vec, atom_vecs, tool_feedback=T×d_feedback tensor)`:
  For t=1..T:
    y_t = atom_pointer(x, A, z_{t-1})
    z_t = f([x, ysum, z_{t-1}, fb_t])
  Returns: (z_1..z_T, y_1..y_T, per-step auxiliary predictions)
- Auxiliary heads (deep supervision targets):
  - `confidence_t`: scalar (how sure the plan so far is correct)
  - `usefulness_t`: scalar (did the last retrieval/add/write help?)
  - `stop_t`: binary (should we halt and decode?)

### Tool MLPs (algo_trm.py — ToolHead base)
- `ToolHead(d_in, d_out, hidden=None)`: 3-layer residual MLP
  `Linear(d_in, h) → GELU → Linear(h, h) → GELU → Linear(h, d_out)` + optional skip
- `RetrievalHead(d, d_feedback)`: produces (query_vec, stop_logit, feedback_vec)
- `WriteHead(d, d_feedback)`: produces (write_latent, node_pointer, feedback_vec)
- `EdgeHead(d, d_feedback)`: produces (src_ptr, dst_ptr, relation_logits, feedback_vec)

### TraversalRanker wrapper (algo_graph_mg.py or new)
- `TRMRetriever(retriever, embed_fn)`: called by MLP_ret
  - `retrieve(query_vec, k)`: cosine search + optional multi-hop refinement
  - Returns: (atom_ids, code_embeddings, metadata) → packed into feedback vector

### Probabilistic health gate (algo_graph_edits.py)
- `write_prob = sigmoid(α·confidence + β·novelty + γ·verification - δ)`
- Sample: write ~ Bernoulli(write_prob)
- confidence = TRM's own confidence head (learned)
- novelty = 1 - cosine_sim(code, existing atoms) (deterministic)
- verification = 1 if code passes tests else 0
- α, β, γ, δ: learned scalars (or fixed hyperparameters initially)

### LM trace decoder (algo_lm_author.py or new)
- Collects TRM trace: [(z_1, tool_outputs_1), ..., (z_T, tool_outputs_T)]
- Formats as structured text: each step's search state, retrieved atoms, write decisions
- Feeds as prompt to LM → LM generates code + explanation
- On VERIFY FAIL: error message + failed code + partial trace feeds back into TRM
  as a special "failure context" vector → TRM re-reasons with this context →
  new trace → LM decodes updated code → verify again (closed loop)
- SFT on (trace, verified_code) pairs

### Closed-loop inference (one task)

```
1. EMBED task text → x_vec [768]

2. TRM REASON (T steps, with tool MLPs):
   For t = 1..T:
     a. Attend to atoms → y_t
     b. Refine z_t with tool_feedback from previous step
     c. Auxiliary heads: confidence_t, usefulness_t, stop_t
     d. Tool MLPs: retrieve, write, edge proposals → feedback for next step
   Output: TRM trace (z_1..z_T, y_1..y_T, tool_outputs_1..T)

3. LM DECODE: trace → structured prompt → LM generates code + helpers + explanation

4. EXECUTE VERIFY: run code against tests

5. LOOP BACK IF FAIL:
   If verify fails:
     a. Collect: error message + failed code + partial trace
     b. Format as failure context (embedded or tokenized)
     c. Prepend to TRM input for a NEW reasoning pass
     d. Go to step 2 (TRM reasons again WITH failure context)
     e. Budget: max N retries per task

6. SLEEP (if solved):
   Build candidates: new atom node + part_of + depend edges
   Probabilistic gate: write_prob = σ(α·conf + β·novelty + γ·1.0 - δ)
   grow() → health gate → persist
```

## Building Blocks That Exist
| Block | File | Status |
|-------|------|--------|
| TRMReasoner | algo_trm.py | T-step recursion, atom-pointer — needs expansion |
| ProgramDecoder | algo_dsl_trm.py | GRU decoder with heads — reference pattern |
| MGRetriever | algo_graph_mg.py | Cosine retrieval on impl embeddings |
| TraversalRanker | traversal_ranker.py | Multi-hop latent retrieval with RefinerNet |
| node/edge_candidate, grow | algo_graph_edits.py | Health-gated graph writes |
| LM author | algo_lm_author.py | LM decodes code from prompt |
| MemoryGraph | graph_core.py | Typed nodes/edges, JSON persist |

### Per-round training flow (STaR)

```
for each round:

  ┌─ WAKE ─────────────────────────────────────────────────────┐
  │  For each task in batch:                                   │
  │    solved = False                                          │
  │    retries = 0                                             │
  │    while not solved and retries < max_retries:             │
  │      TRM reason with tools  ──→ trace                      │
  │      LM decode trace ──→ code                              │
  │      Execute verify                                        │
  │      if verified: solved = True; collect (trace, code)     │
  │      else: feed error back to TRM; retries += 1            │
  │                                                            │
  │    if solved:                                              │
  │      add to SFT pool (deep supervision on TRM steps)       │
  │      bank solution: probabilistic gate → grow → graph      │
  └────────────────────────────────────────────────────────────┘
  
  ┌─ CONSOLIDATE ──────────────────────────────────────────────┐
  │  SFT TRM on pool (deep supervision over all T steps)       │
  │  SFT LM on (trace → verified_code) pairs                   │
  └────────────────────────────────────────────────────────────┘
  
  ┌─ SLEEP ────────────────────────────────────────────────────┐
  │  For each newly banked atom:                               │
  │    write node + part_of + depend edges to graph JSON       │
  │  Re-index retriever                                        │
  └────────────────────────────────────────────────────────────┘
  
  ┌─ MEASURE ──────────────────────────────────────────────────┐
  │  Zero-shot decode on held-out (no search, no retries)      │
  │  Track: solved rate, verifies-to-solve, graph size         │
  └────────────────────────────────────────────────────────────┘
```

The compounding effect: each round, the graph has MORE atoms → TRM retrieves more
relevant context → composes better → solves harder tasks → banks more atoms.
Zero-shot rises; verifies-to-solve falls. Rebuild: fresh net + same graph → same
solve rate (graph IS the memory).

## Build Order (implemented below in algo_trm.py)
1. TRMReasoner: larger d, tool-feedback input, auxiliary heads
2. ToolHead base class with residual blocks
3. RetrievalHead + TRMRetriever wrapper
4. TRMWithTools orchestrator + selftest
5. Probabilistic health gate (deferred to graph write integration)
6. LM trace decoder (deferred to LM integration phase)