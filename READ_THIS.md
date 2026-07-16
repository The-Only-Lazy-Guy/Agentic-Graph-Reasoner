# READ_THIS — GRR: Graph Recursive Reasoner (2026-07-17)

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

## Repro (molab, no 4B — mpnet + tiny nets, minutes; poison test needs GPU)
```
python -m v5.runtime.algo_composed --selftest                    # thesis: str_dp2 load-bearing
python -m v5.runtime.algo_dsl_trm --compo-gen-star --hard --graph graphs/algo_reason_hard.json  # 0% -> 100%
python -m v5.runtime.algo_grr_loop --loop --graph graphs/algo_grr_loop.json     # THE loop (compounding table)
python -m v5.runtime.algo_grr_loop --rebuild --graph graphs/algo_grr_loop.json  # graph-only net rebuild
python -m v5.runtime.algo_grr_seed --selftest                    # clean 25-node seed graph
python -m v5.runtime.algo_grr_membrane --selftest                # frozen+ membrane closed loop (5/5)
python -m v5.runtime.algo_grr_membrane --run --stub              # 6/6 on curriculum (no GPU)
python -m v5.runtime.algo_grr_poison_test --selftest             # two-arm structural test (no GPU)
python -m v5.runtime.algo_grr_policy --selftest                  # ComplementPolicy TRM policy
# Poison thesis (GPU, Qwen2.5-3B-Instruct):
python -m v5.runtime.algo_grr_poison_test --run --lm Qwen/Qwen2.5-3B-Instruct  # NEW arm only
python -m v5.runtime.algo_grr_poison_test --inspect --lm Qwen/Qwen2.5-3B-Instruct  # R3/R4 derive inspect
V5_LM_QUANT=8bit python -m v5.runtime.algo_grr_poison_test --run --lm Qwen/Qwen2.5-3B-Instruct --old-arm  # both arms
# every module: python -m v5.runtime.<algo_quality|algo_capability|algo_abstract|algo_sleep|algo_graph_edits|algo_graph_mg|algo_compose_tasks|algo_composed|algo_trm_compose|algo_dsl|algo_dsl_trm|algo_grr_loop|algo_meta|algo_anticheat|algo_grr_seed|algo_grr_membrane|algo_grr_poison_test|algo_grr_policy> --selftest   # 16/16 PASS
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

## POISON DIAGNOSIS + frozen-compiler resolution (2026-07-16)
The failure both the old STaR loop AND a naive GRR-Tool share: **the graph poisons the LM as
STaR progresses** (solve-rate declines over rounds). Traced to TWO channels, both = LM absorbing the graph:
  - **Channel 1 — WEIGHT poison (the STaR LoRA, commit 425d649 unfrozen loop):** each epoch SFTs the
    LM on its own verified traces. Narrow verified set -> memorization collapse (the 210-variant augment
    271a48b was fighting exactly this), plus_only overfit, catastrophic forgetting; graph-conditioned
    traces bake shifting graph-dependency into weights.
  - **Channel 2 — CONTEXT poison (graph-in-prompt):** as the graph grows, more atoms flood the prompt;
    a wrong retrieved atom drags the LM. This is what the measured "advertisement net-negative / over-time
    decline" (GRR-14 ablation, ~16pp) actually was — context distraction, not the graph's memory role.
RESOLUTION (user's principle, made literal): **LM = FROZEN COMPILER, un-poisonable. ALL learning lives
in TRM + graph.** LM is a pure stateless `compile(spec)->code`; weights NEVER change -> no gradient path
graph->LM -> Channel 1 dead. **The TRM is the MEMBRANE:** graph never touches the LM directly — the TRM
retrieves, tentatively composes, verifies partial coverage, and hands the LM ONLY a clean curated spec
(subgoals + chosen atoms + wiring + holes), never a raw top-k dump -> Channel 2 dead. A bad atom now only
costs if the TRM retrieves it AND it survives the hard verify gate (which already rejects 31% overfit);
compiling a bad spec -> verify fails -> not banked. Loop self-cleans. Authoring novel primitives uses the
FROZEN LM's capability ("LM teaches once, graph remembers forever" = teaching accumulates in the GRAPH, not
LM weights) -> zero LM training ever. Two dead components deleted from the original design: (a) WriteHead's
"latent for LM code generation" = the measured z-wall / amortization-not-capability, replaced by DISCRETE
atom-pointer + spec (text only, never a latent handoff); (b) gameable health-gate scalars (learnable delta
can open the gate on unverified code) -> gate is VERIFICATION-DOMINATED, novelty is tie-break only.
COMPOUNDING TARGET MOVES: solve-rate rises not because the LM improves (poisons) but because the graph
covers more subgoals -> TRM composes more from memory -> **LM does strictly LESS per task** (token-burden
falls as the graph grows, instead of flooding). rebuild-net already proves the graph is the memory.
FALSIFIABLE TEST — DONE (2026-07-17, real Qwen2.5-3B-Instruct on molab, 4 rounds of the designed seed
curriculum, R1 recall / R2 compose / R3 derive / R4 reuse):

  NEW (frozen 3B + membrane):             OLD (LoRA SFT + raw flood):
    round  solved  reuse  prompt_atoms       round  solved  reuse  prompt_atoms
      1     3/3      1     1.0                  1     2/3      0     21.3
      2     3/3      4     1.7                  2     1/3      0     23.0
      3     2/2      0     1.0                  3     2/2      0     24.5
      4     2/2      2     4.0                  4     1/2      0     26.5

RESULT — NEW 10/10 (100%), OLD 6/10 (60%). Two poison channels both confirmed:
  (a) Weight poison: LoRA mean loss collapses 0.577→0.117→0.009→0.002 as the pool grows —
      the LM overfits to its own traces, solve rate drops 2/3→1/3→1/2 across rounds.
  (b) Context poison: raw-flood prompt grows 21→27 atoms with the graph, overwhelming the LM
      (vs NEW bounded at ≤4 atoms). OLD reuse = 0 structurally (whole-solution banking can't compose);
      NEW compounds (R3 derives helper atoms, R4 reuses them).
  VERDICT: frozen-compiler + membrane premise CONFIRMED — the poison thesis is experimentally
  validated with a real 3B. All learning stays in TRM + graph; the LM stays frozen forever.

COMPOUNDING CONFIRMED IN RAW (2026-07-17, --inspect on the real 3B — not from aggregates):
  R3 t_sumsq -> frozen 3B factors a TOP-LEVEL `sum_of_squares` -> BANKED, graph 25->26.
  R4 t_sumsq_rev -> solve selects=['sum_of_squares'] = REUSES the R3-banked atom (the payoff, VISIBLE).
  R4 t_fib_prime -> factors + BANKS `fib`, reuses is_prime, graph ->27.
  Graph grew 25->27, two derived atoms banked, one reused across rounds. Earlier runs had banked=0
  (3B wrote MONOLITHIC/NESTED code); two fixes closed it: (i) compose prompt demands TOP-LEVEL factoring
  + one-shot shape + strip_module_exec (drops the LM's own print()/check() so its grader never runs in
  our sandbox = anti-cheat hygiene); (ii) membrane.bankable_pure_defs banks a helper even when the LM
  NESTS it inside the entry (AST purity walk; capturing closures rejected) -> robust compounding that
  does NOT depend on the LM factoring top-level. poison_test selftest now 5 checks (incl. nested-banking).
CHANNEL ISOLATION — MEASURED (2026-07-17, --isolate, clean 2x2, 3 OLD-variants share ONE compile path so
  only prompt{bounded|flood} x train{off|on} differ; an untrained LoRA is zero-init == frozen):
    NEW          (neither: frozen + membrane) : 10/10  compounds (graph 25->27)
    CONTEXT-only (flood prompt, frozen)       :  3/10  <- flood ALONE drops it 10->3
    WEIGHT-only  (bounded prompt, LoRA SFT)   :  6/10  <- LoRA  ALONE drops it 10->6
    OLD          (both: flood + LoRA)         :  1/10  <- channels STACK, worst
  BOTH channels are independently load-bearing (each < NEW) and they stack -> "both channels confirmed"
  HOLDS. (A first buggy run had CONTEXT-only 0/10 — artifact of a mismatched compile path, fixed 09ba327;
  it also showed WEIGHT-only 9/10, which was noise.) HONEST NOISE CAVEAT: single seed, 10 tasks, temperature
  0.6 -> the per-round curves are NON-MONOTONIC and the RNG floor is +-1-2 tasks (R1 OLD 0/3 vs CONTEXT-only
  1/3 is the IDENTICAL config = pure sampling noise). So the DIRECTION (NEW >> each single poison >> both) is
  robust and the mechanism is real, but this is a MECHANISM DEMO, not a clean dose-response curve. To harden:
  greedy decode (do_sample=False) + multiple seeds + more tasks. Repro:
  `V5_LM_QUANT=8bit python -m v5.runtime.algo_grr_poison_test --run --lm Qwen/Qwen2.5-3B-Instruct --isolate`.

## GRR-Tool BUILD STATUS (2026-07-17) — poison thesis CONFIRMED with real 3B
All four modules, selftested no-GPU, plus the real-3B two-arm molab run completed:
- `algo_grr_seed.py` -> `graphs/grr_seed_clean.json` (clean 25-node seed with depend edges).
- `algo_grr_membrane.py` = the frozen-compiler + membrane closed loop. MembraneSolver: iterative
  verifier-gated retrieval, curated spec, `policy_fn` seam, `make_lm_compiler(gen_fn)` = FROZEN 3B.
  Selftest 5/5. Real-3B on curriculum: 100% solved, prompt ≤4 atoms, compounds (R3 derives, R4 reuses).
- `algo_grr_poison_test.py` = the two-arm test (R1 recall / R2 compose / R3 derive / R4 reuse).
  Real-3B result (Qwen2.5-3B-Instruct, FP16, 4 rounds):
    NEW (frozen + membrane + helper-granular derive-bank): 10/10 solved, prompt ≤4, compounds.
    OLD (LoRA SFT on own traces + raw-flood prompt + whole-solution banking): 6/10 solved, declining
    accuracy per round, prompt floods 21→27 atoms, zero reuse, LoRA loss collapses to 0.002.
  VERDICT: poison thesis experimentally confirmed — two channels both validated. The LM STAYS FROZEN.
  OLD-arm LoRA adapter saved to `artifacts/old_arm_adapter/` (on molab).
  Repro: `V5_LM_QUANT=8bit python -m v5.runtime.algo_grr_poison_test --run --lm Qwen/Qwen2.5-3B-Instruct --old-arm`
  Inspect R3/R4 derive code: `python -m v5.runtime.algo_grr_poison_test --inspect --lm Qwen/Qwen2.5-3B-Instruct`
- `algo_grr_policy.py` (B2a) = the trained TRM retrieval policy. ComplementPolicy: tiny pointer net
  scoring atoms | (task, atoms-selected-so-far), trained to rank the STILL-MISSING atom highest, drops
  into MembraneSolver.policy_fn. Selftest PASS: complement rank avg 1.00 top1 vs cosine 1.50.
  PENDING (next): drop ComplementPolicy into MembraneSolver (currently uses cosine baseline), scale to MBPP+.

## CLEAN SEED GRAPH (2026-07-16) — replaces the polluted grown graphs
The grown graphs (grr_grown 377n, grown_graph* 4-13MB) are POLLUTED: whole task-solutions banked under
entry-point names (`impl_similar_elements` = raw MBPP prompt + full solution), reusable helpers TRAPPED as
nested inner fns, and topology FLAT (**371 part_of, ZERO depend** -> nothing composes -> cross-task reuse
mechanically 0). New clean substrate: `v5/runtime/algo_grr_seed.py` -> `graphs/grr_seed_clean.json`:
**25 nodes (21 verified primitive atoms + 4 concept hubs), 28 edges (21 part_of + 7 REAL depend), 0 dangling.**
Every atom is fuzz-verified against its dep-closure before it enters the graph (the store-gate), helper-
granular (is_prime/gcd/lcm->gcd/divisors/sum_divisors->divisors/is_palindrome_number->reverse_digits/
is_anagram->char_freq/most_common->count_occurrences/is_perfect->sum_divisors...), text = concise PURPOSE
key (never a task prompt). Loads through graph_core.MemoryGraph unchanged (all 28 edges kept). Selftest:
`python -m v5.runtime.algo_grr_seed --selftest` (21/21 atoms pass, no GPU). This is the reuse-bearing
starting topology the TRM membrane composes over.

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