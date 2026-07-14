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
