# Graph-Agent Route/RL Project — Task List

**Last updated:** 2026-05-16
**Current repo:** new clean repo  
**Graph storage:** `./graphs/*.json`  
**Active architecture:** goal-conditioned executor  
**Old architecture:** archived `NGR-v1a` action-policy path and traversal heuristic predictor prototype  
**Primary rule:** keep executor deterministic and temporary-only; do not hide predictor failures inside executor heuristics  
**Official active script:** `traverse_threshold_draft_edit.py`  

**POLICY** Please do not use heuristics to patch the model. It will make the spaghetti code unmaintainable, undebuggable, and unpatchable in the long term, SO TO PATCH PLEASE THINK OF BETTER ALOGORITHM THAT COULD BENEFIT THE WHOLE SYSTEM.!!!

Key : Aim for breakthroughs, not shorterm patches. Don't give up easily, the road to success will not be easy!!!

---

## Answerer-v2 Novelty Eval v0 - 2026-05-16

Status:

```text
GRAPH PROTOCOL    PASSED
GROUNDING RULES   TIGHTENED
NOVELTY SUITE     BUILT (20 hand-designed questions)
MOCK BASELINE     0/20 pass (gate has teeth)
LOCAL SAMPLE      RUNNING (4 questions, one per category)
FULL LOCAL SUITE  NOT YET RUN
```

Files:

```text
data/novelty_eval.jsonl          20 rows across 4 categories
novelty_metrics.py               7 graph-grounded metrics + composite gate
eval_novelty.py                  runner (mock | local, --ids filter)
answer_query_v2_with_session     v2 API that returns final session + trace
```

Tightening fixes already applied to answerer_v2:

```text
1. Support cap 1-4 per NOTE/CONCLUDE (executor-enforced)
2. Reject plan_step / Q0 / hypothesis as direct supports
3. Reject contradict-pair supports (force VERIFY_EDGE)
4. Evidence quality score in briefing (relevance/hub/degree/contradict penalties)
5. Coverage-complete short-circuit (loop stops once every plan_step grounded)
6. _import_missing_edges (pulls in edges expand_node misses, e.g. contradict)
```

Composite novelty_pass gate:

```text
graph_dependency        >= 0.5  (transitive evidence coverage)
max_support_depth       >= 2    (real chain, not direct cite)
no_shortcut             True    (no forbidden_shortcut_nodes)
contradiction_clean     True    (no contradict-pair cited together)
no_false_claims         True    (no must_not_claim substring)
plan_coverage           >= 0.5  (>=half plan_steps grounded)
```

Next priorities (in order):

```text
1. Full 20-question LocalGLM run (~2 hours)
2. Anchor retrieval quality
     question-blind anchors leaked irrelevant CS nodes into BFS/DFS run
     fix lives in answerer_v1.retrieve_anchors
3. Ablation sensitivity metric
     re-run each question with one required_evidence node removed
     measure how much the answer degrades
4. PRED preview during PLAN phase (deferred to follow-up)
5. RL/training data harvest from successful answerer traces
     traces are already saved to artifacts/trace_*.json
```

---

## EXEC-v0 Cutover - 2026-05-11

Active controller:

```text
traverse_threshold_draft_edit.py --controller-mode executor
```

Current held-out correctness run:

```text
tasks:
  artifacts/tasks_trv_executor_20260511/ngr_v1_val.jsonl

output:
  out_traverse_executor_val_20260511.json

overall:
  task_complete_proxy_rate  = 1.0
  task_complete_strict_rate = 1.0
  session_node_precision    = 1.0
  session_edge_precision    = 1.0
  attachment_precision      = 1.0
  false_edge_rate           = 0.0
  false_attachment_rate     = 0.0
```

What this means:

```text
The executor is no longer the place where prediction heuristics live.
It now executes the goal spec directly and produces exactly goal-sized draft structure.
```

Active next work:

```text
1. Keep EXEC-v0 stable as the reference executor.
2. Treat the old traversal path only as a predictor prototype.
3. Build predictor-side evaluation separately from executor correctness.
4. Remove remaining dead heuristic branches from the active file once predictor replacement is ready.
```

---

## PRED-v2 Aligner Baseline - 2026-05-11

New files:

```text
pred_model.py
train_pred_v1.py
eval_pred_v1.py
```

Current trained baseline:

```text
checkpoint:
  out_pred_v1_train_20260511_fix1/best_pred_v1.pt

held-out val:
  span_top1_acc             = 0.7598
  span_top1_acc_nonnull     = 0.7852
  commit_acc                = 0.9033
  edge_precision            = 0.5309
  edge_recall               = 0.8692
  edge_f1                   = 0.6592
  edge_relation_acc_on_gold = 0.7090
  row_complete_rate         = 0.1462
```

Meaning:

```text
This is the first learned predictor baseline on pred_v1 data.
It is goal-conditioned: it aligns spans and predicts edge structure for known session specs.
It is not yet a free-form goal generator.
```

Current next patch:

```text
PRED-v2.1

Goal:
  inspect the learned aligner by task and by null-span bridge cases

Priority:
  1. per-task span / edge metrics
  2. explicit analysis of best_span_id = null bridge rows
  3. decide whether the next move is better edge precision or unguided session-spec prediction
```

---

## PRED-v2.11 Memory-Context Edge Heads - 2026-05-11

Active baseline:

```text
checkpoint:
  out_pred_v1_train_20260511_fix12/best_pred_v1.pt  (ep7, best score)
  out_pred_v1_train_20260511_fix12/best_cover_pred_v1.pt  (ep7, best cover_f1+recall)
```

Best held-out val (ep7):

```text
row_complete_rate             = 0.2241
cover_precision               = 1.000
cover_recall                  = 0.992
cover_f1                      = 0.996
span_top1_acc_nonnull         = 0.7245
commit_acc                    = 1.000
edge_f1                       = 0.7717
edge_relation_acc_on_gold     = 0.7845
attachment_f1                 = 0.9894
attachment_relation_acc_on_gold = 0.7937
```

Per-task row_complete:

```text
covered_long_signal           = 0.561   (was 0.024 before fix12 — 23× improvement)
long_decompose                = 0.000   (unchanged — FP edge problem)
mixed_add_link                = 0.244
multi_region_attach           = 0.524
```

What changed over PRED-v2.10:

```text
1. pred_model.py: edge_exist_head and edge_rel_head input widened from
   hidden_dim×3 → hidden_dim×5 by appending state_h to every (spec_i, spec_j) pair.
   state_h = cat([signal_h, pooled_mem.detach()]) carries memory context.
   Covered rows have zero gold session edges; memory context now signals this
   and suppresses false-positive edge predictions for covered rows.

2. train_pred_v1.py: score formula extended to include +0.25*cover_f1.
   Dual checkpoint save: best_pred_v1.pt (score) + best_cover_pred_v1.pt
   (cover_f1 + cover_recall).

3. eval_pred_v1.py: added covered_row_failure_breakdown() diagnostic function.
```

Meaning:

```text
covered_long_signal is no longer the dominant bottleneck.
The failure modes are now split cleanly by task type:
- covered_long_signal: mostly span accuracy ceiling (~56% row-complete)
- long_decompose: entirely FP edge problem (row_complete=0, edge_precision=0.504)
- mixed_add_link: attachment relation label errors (~24% row-complete)
- multi_region_attach: span accuracy ceiling (~52% row-complete)
```

Active next decision:

```text
PRED-v2.12

Goal:
  keep the successful decode-side suppression
  then target the remaining long_decompose relation-heavy FP residue
  long_decompose row_complete already moved off zero (0.194)

Priority:
  1. preserve anti-symmetry + transitive reduction in edge decode
  2. inspect remaining relation-error edges on long_decompose
  3. do not touch covered or multi_region until long_decompose improves again
```

---

## PRED-v2.12 Long-Decompose Edge Precision - DIAGNOSED / DECODE ABLATION COMPLETE

Status:

```text
PHASE 1 COMPLETE
DECODE-SIDE ABLATION COMPLETE
```

Problem:

```text
Baseline before decode suppression:
  long_decompose row_complete = 0.0
  edge_precision = 0.504

Dominant failure was structural false-positive edges.
After the decode-side anti-symmetry + transitive reduction experiment:
  long_decompose row_complete = 0.1938
  edge_precision = 0.6750

So the zero-complete deadlock is broken, but relation-heavy residual FP edges remain.
```

Completed diagnosis:

```text
Ran long_decompose_edge_debug() on fix12 baseline.

Before decode suppression:
  FP total = 370
  spurious_pair   = 179 (48.4%)
  direction_error = 101 (27.3%)
  relation_error  = 90  (24.3%)

This ruled out a pure threshold fix.
Most errors were structural:
  - transitive shortcut edges such as 0→2
  - reverse-direction duplicates

Applied decode-side ablation globally:
  1. anti-symmetry: keep higher-confidence direction when both (i,j) and (j,i) fire
  2. transitive reduction: drop (i,k) when predicted path (i,j),(j,k) exists

Safety check:
  long_decompose gold contains no intentional transitive shortcuts,
  so transitive reduction is safe for that task family.

After decode suppression:
  FP total = 171
  relation_error  = 67  (39.2%)
  spurious_pair   = 58  (33.9%)
  direction_error = 46  (26.9%)
```

Observed outcome:

```text
Global:
  row_complete_rate = 0.2972
  edge_f1           = 0.8085

long_decompose:
  row_complete      = 0.1938   (first nonzero result)
  edge_precision    = 0.6750
  edge_recall       = 0.6750
  edge_f1           = 0.6750

No meaningful regression on covered / mixed / multi_region tasks.
```

Next step:

```text
Keep the decode-side suppression path.

Then move to training-side long_decompose cleanup:
  1. inspect the 67 relation-error residuals
  2. avoid inverse-frequency relation CE; it over-corrected toward rare labels
  3. revisit remaining spurious-pair negatives only after relation errors are reduced
```

Rejected follow-up:

```text
PRED-v2.13 inverse-frequency edge_rel CE from fix12:
  out_pred_v1_train_20260511_fix13_relweight

Result:
  global row_complete_rate = 0.2571
  long_decompose row_complete = 0.0688
  long_decompose edge_relation_acc_on_gold = 0.3875

Failure:
  relation_error count rose to 133
  relation confusion shifted into rare-class overprediction:
    support -> contradict = 67
    part_of -> contradict = 29

Decision:
  keep fix12 + decode suppression as active baseline
  do not use inverse-frequency edge_rel CE as the next path
```

Neutral follow-up:

```text
PRED-v2.14 focal edge_rel loss from fix12:
  out_pred_v1_train_20260511_fix13_focal_g2

Result:
  global row_complete_rate = 0.2925
  long_decompose row_complete = 0.1938
  long_decompose edge_relation_acc_on_gold = 0.6906

Compared with active fix12 + decode:
  global row_complete_rate = 0.2972
  long_decompose row_complete = 0.1938

Decision:
  focal loss is stable but not a meaningful upgrade
  keep fix12 + decode suppression as active baseline
```

Informative follow-up:

```text
PRED-v2.15 relation pair features:
  out_pred_v1_train_20260511_fix15_pairfeat

Added optional edge_rel_pair_feat with:
  jaccard_sim
  containment_ij
  containment_ji
  length_ratio
  position_delta

Result:
  global row_complete_rate = 0.2925
  long_decompose row_complete = 0.1875
  long_decompose edge_relation_acc_on_gold = 0.6937

Compared with active fix12 + decode:
  global row_complete_rate = 0.2972
  long_decompose row_complete = 0.1938

Decision:
  pair features are not a meaningful upgrade
  keep fix12 + decode suppression as active baseline
  next relation path should be encoder-side, not another small scalar feature
```

Encoder-side follow-up:

```text
PRED-v2.16 frozen spec sentence embeddings:
  artifacts/spec_emb_cache/pred_v1_fix8_minilm_spec.npz
  out_pred_v1_train_20260511_fix16_specemb

Added:
  precompute_pred_embeddings.py
  optional spec_emb channel in PredBatch / PredAlignNet
  --spec-emb-cache for train/eval

Result:
  global row_complete_rate = 0.2689
  global edge_f1 = 0.8476
  global edge_relation_acc_on_gold = 0.8379

  long_decompose row_complete = 0.1750
  long_decompose edge_f1 = 0.7410
  long_decompose edge_relation_acc_on_gold = 0.7250

Compared with active fix12 + decode:
  global row_complete_rate = 0.2972
  global edge_f1 = 0.8085
  global edge_relation_acc_on_gold = 0.8122
  long_decompose row_complete = 0.1938
  long_decompose edge_f1 = 0.6750
  long_decompose edge_relation_acc_on_gold = 0.6813

Decision:
  sentence embeddings improve the edge/relation axis
  but the current spec-only additive integration regresses span/covered rows
  keep fix12 + decode suppression as active baseline

PRED-v2.17 spec-embedding warm-start from fix12:
  out_pred_v1_train_20260512_fix17_specemb_warm

Added:
  --init-checkpoint out_pred_v1_train_20260511_fix12/best_pred_v1.pt
  zero-initialized spec_emb_proj so step-0 behavior matches fix12 on shared weights

Result:
  global row_complete_rate = 0.2948
  global edge_f1 = 0.8059
  global edge_relation_acc_on_gold = 0.8232
  cover_f1 = 0.9620

  long_decompose row_complete = 0.2000
  long_decompose edge_f1 = 0.6698
  long_decompose edge_relation_acc_on_gold = 0.7000

Compared with active fix12 + decode:
  global row_complete_rate = 0.2972
  global edge_f1 = 0.8085
  global edge_relation_acc_on_gold = 0.8122
  cover_f1 = 0.9959
  long_decompose row_complete = 0.1938
  long_decompose edge_relation_acc_on_gold = 0.6813

Decision:
  warm-start recovers most of the fresh spec-embedding regression
  and gives a small long_decompose relation gain
  but it still slightly trails the active baseline globally and regresses cover
  keep fix12 + decode suppression as active baseline

PRED-v2.18 edge-only spec-embedding routing:
  out_pred_v1_train_20260512_fix18_specemb_edgeonly

Changed:
  spec_h stays BoW-only for span and memory heads
  spec_h_for_edges = spec_h + zero-init spec_emb_proj(spec_emb)
  only edge_exist_head and edge_rel_head consume spec_h_for_edges

Best-score result:
  global row_complete_rate = 0.2972
  global edge_f1 = 0.8104
  global edge_relation_acc_on_gold = 0.8269
  cover_f1 = 0.9620

  long_decompose row_complete = 0.2062
  long_decompose edge_f1 = 0.6772
  long_decompose edge_relation_acc_on_gold = 0.7063

Compared with active fix12 + decode:
  global row_complete_rate = 0.2972
  global edge_f1 = 0.8085
  global edge_relation_acc_on_gold = 0.8122
  cover_f1 = 0.9959
  long_decompose row_complete = 0.1938
  long_decompose edge_relation_acc_on_gold = 0.6813

Decision:
  edge-only routing is the strongest encoder diagnostic so far
  it matches global row_complete and improves long_decompose
  but cover still regresses during continued training
  keep fix12 + decode suppression as active safe baseline unless cover regression is acceptable

PRED-v2.19 frozen non-edge warm-start:
  out_pred_v1_train_20260512_fix19_specemb_edgeonly_freeze

Changed:
  added --freeze-except-edge-emb
  trainable prefixes:
    spec_emb_proj
    edge_exist_head
    edge_rel_head
  all fix12-loaded non-edge paths are frozen

Result:
  global row_complete_rate = 0.3090
  global edge_f1 = 0.8129
  global edge_relation_acc_on_gold = 0.8343
  cover_f1 = 0.9959

  long_decompose row_complete = 0.2250
  long_decompose edge_f1 = 0.6823
  long_decompose edge_relation_acc_on_gold = 0.7188

Compared with old active fix12 + decode:
  global row_complete_rate = 0.2972
  global edge_f1 = 0.8085
  global edge_relation_acc_on_gold = 0.8122
  cover_f1 = 0.9959
  long_decompose row_complete = 0.1938
  long_decompose edge_relation_acc_on_gold = 0.6813

Decision:
  promote PRED-v2.19 as the active aligner baseline
  it preserves fix12 span/cover/memory behavior and improves edge/relation quality

PRED-v2.20 candidate embeddings for span scorer:
  out_pred_v1_train_20260512_fix20_candemb_span_freeze

Added:
  artifacts/spec_emb_cache/pred_v1_fix8_minilm_cand.npz
  cand_emb in PredBatch
  cand_emb_proj routed only into span_scorer
  --cand-emb-cache
  --freeze-except-cand-emb-span

Result:
  global row_complete_rate = 0.3137
  span_top1_acc = 0.7321
  edge_f1 = 0.8129
  edge_relation_acc_on_gold = 0.8343
  attachment_f1 = 0.9895
  cover_f1 = 0.9959

Per task:
  covered_long_signal row_complete = 0.5366
  long_decompose row_complete = 0.2250
  mixed_add_link row_complete = 0.2562
  multi_region_attach row_complete = 0.5397

Compared with fix19:
  global row_complete_rate = 0.3090 -> 0.3137
  span_top1_acc = 0.7245 -> 0.7321
  edge/cover/memory metrics preserved

Decision:
  promote PRED-v2.20 as the active aligner baseline
  the gain is modest but clean under frozen-component constraints
```

```text
PRED-v2.21 memory embeddings for mem_rel_head:
  out_pred_v1_train_20260512_fix21_mememb_rel_freeze

Added:
  artifacts/spec_emb_cache/pred_v1_fix8_minilm_mem.npz
  mem_emb in PredBatch
  mem_emb_proj routed only into mem_rel_head
  --mem-emb-cache
  --freeze-except-mem-emb-rel

Result:
  global row_complete_rate = 0.3208
  span_top1_acc = 0.7321
  edge_f1 = 0.8129
  edge_relation_acc_on_gold = 0.8343
  attachment_f1 = 0.9895
  attachment_relation_acc_on_gold = 0.8042
  cover_f1 = 0.9959

Per task:
  covered_long_signal row_complete = 0.5366
  long_decompose row_complete = 0.2250
  mixed_add_link row_complete = 0.2750
  multi_region_attach row_complete = 0.5397

Compared with fix20:
  global row_complete_rate = 0.3137 -> 0.3208
  span/edge/cover/attachment-target metrics preserved
  global attachment_relation_acc_on_gold = 0.8112 -> 0.8042
  mixed_add_link attachment_relation_acc_on_gold = 0.6625 -> 0.6500

Decision:
  promote PRED-v2.21 as the active row-complete aligner baseline
  do not count it as a successful memory-relation fix
  the memory-relation bottleneck remains open
```

Next PRED-v2 task:

```text
PRED-v2.22

Goal:
  improve residual structural edge false positives from the fix21 baseline.

Reason:
  the single-component embedding series is exhausted:
    spec embeddings helped materially
    candidate embeddings helped modestly
    memory embeddings did not fix attachment relation labels

Starting checkpoint:
  out_pred_v1_train_20260512_fix21_mememb_rel_freeze/best_pred_v1.pt

Priority:
  1. preserve fix21 as the active row-complete baseline
  2. keep decode-side anti-symmetry and transitive reduction enabled
  3. diagnose residual edge FP categories under fix21:
       relation_error
       direction_error
       spurious_pair
  4. if training is needed, target edge structural tails only
  5. do not add another embedding path without a new diagnosis

Alternate branch:
  if mixed_add_link becomes the priority, first diagnose attachment relation
  errors directly; fix21 showed memory embeddings alone are not the answer.
```

PRED-v2.22 result:

```text
Implemented:
  train_pred_v1.py
    build_edge_hard_negative_mask()
    --hard-negative-weight
    --hard-negative-max-per-row
    --freeze-except-edge

After rejection:
  --hard-negative-weight defaults to 1.0 so future training is not silently
  affected by the rejected weighting.

Run A:
  out_pred_v1_train_20260512_fix22_edge_hardneg_freeze
  hard_negative_weight = 2.0
  global row_complete_rate = 0.3019
  global edge_f1 = 0.7930
  long_decompose row_complete = 0.1750
  long_decompose edge_f1 = 0.6478

Run B:
  out_pred_v1_train_20260512_fix22b_edge_hardneg15_freeze
  hard_negative_weight = 1.5
  global row_complete_rate = 0.3113
  global edge_f1 = 0.8011
  long_decompose row_complete = 0.2000
  long_decompose edge_f1 = 0.6625

Compared with active fix21:
  global row_complete_rate = 0.3208
  global edge_f1 = 0.8129
  long_decompose row_complete = 0.2250

Decision:
  reject hard-negative edge supervision as an active baseline
  keep PRED-v2.21 active
  do not raise hard_negative_weight further
```

PRED-v2.23 full unfreeze from fix21:

```text
Run:
  out_pred_v1_train_20260512_fix23_unfreeze_lr1e5

Setup:
  freeze_mode = none
  lr = 1e-5
  epochs = 8
  warm-start = fix21

Result:
  global row_complete_rate = 0.3066
  span_top1_acc = 0.7293
  edge_f1 = 0.8074
  edge_relation_acc_on_gold = 0.8324
  attachment_f1 = 0.9859
  attachment_relation_acc_on_gold = 0.8147
  cover_f1 = 0.9750
  long_decompose row_complete = 0.1875

Compared with fix21:
  global row_complete_rate = 0.3208
  cover_f1 = 0.9959
  long_decompose row_complete = 0.2250

Decision:
  reject full-unfreeze follow-up
  fix21 remains the final PRED-v2 aligner baseline
  the remaining aligner-only upside is not worth pursuing under this architecture
```

PRED-v2 series conclusion:

```text
The aligner-polishing returns are now closed.

Final oracle-conditioned aligner baseline:
  PRED-v2.21
  out_pred_v1_train_20260512_fix21_mememb_rel_freeze/best_pred_v1.pt

Meaning:
  this is the upper bound to carry into proposer work
  until the goal proposer exists, the pipeline is still not deployable
```

Next active stage:

```text
PRED-v3 proposer design
```

---

## PRED-v3 Goal Proposer - ACTIVE DESIGN

Status:

```text
ACTIVE
```

Current active baseline:

```text
PRED-v3+ unified end-to-end model (fix4)
checkpoint: out_unified_v1_20260513_fix4_slot_pos/best_unified_v1.pt

best val (epoch 9):
  row_complete_rate              = 0.4505
  text_faithful_row_complete     = 0.4505
  span_top1_acc                  = 0.7722
  text_faithful_acc              = 0.8084
  edge_f1                        = 0.7823
  edge_relation_acc_on_gold      = 0.7808
  attachment_f1                  = 0.7730
  cover_f1                       = 0.9960

Per-task row_complete:
  covered_long_signal:  0.4146
  long_decompose:       0.3750
  mixed_add_link:       0.5000
  multi_region_attach:  0.5397

The old split path remains archived for comparison:
  proposer  = fix5
  aligner   = fix21
  split e2e = 0.1745 lenient / 0.0000 text-faithful
```

Rejected experiments since fix1:

```text
fix2 (cand_self_attn, 2026-05-13): rejected
  - helped span/edge on covered + long_decompose
  - but disturbed slot_query and regressed attach on mixed_add_link
  - row_complete: 0.4387 -> 0.4316

fix3 (mem_rel class-weighted CE, 2026-05-13): rejected
  - targeted "predict support" bias on mixed_add_link
  - inverse-freq weights clipped to [0.5, 2.5]
  - resolved the headline "X -> support" pattern but introduced
    "support -> X" + new "X -> example_of" overcorrection
  - wrong_relation count rose 35 -> 46 on mixed_add_link
  - row_complete unchanged globally (0.4387)
  - same over-correction pattern as PRED-v2 fix13
```

Promoted: fix4 (slot_pos + mem_pair_feat bundle, 2026-05-13):

```text
- intended target: wrong_slot bucket (20 mis-routed attaches on mixed_add_link)
- actual mechanism: encoder reorganization improved edge_f1 (+2.6pp) and
  edge_relation_acc (+2.4pp), driving a +8.75pp row_complete gain on
  long_decompose
- diagnosed target (wrong_slot) unchanged at 20 cases
- net global win: row_complete 0.4387 -> 0.4505 (externally confirmed)
- side regressions: mixed_add_link -2.5pp, multi_region_attach -7.94pp
- three bundled changes mean we cannot attribute which sub-change drove
  the gain; future isolation experiment could split slot_pos vs mem_pair_feat
```

Active next decision:

```text
Two known-unaddressed problems remain:

(a) wrong_slot on mixed_add_link (20 cases, unchanged since fix1)
    - slot_pos embedding alone did not move this number
    - next attempt could try explicit task_type conditioning or
      decode-time tie-breaking toward slot 1 for mixed_add_link rows

(b) multi_region_attach regression (-7.94pp since fix1)
    - new bottleneck introduced by fix4
    - needs its own diagnostic before any further change
    - hypothesis: shared encoder reallocated away from bridge synthesis

Recommended order:
1. Diagnose multi_region_attach regression first (cheap, ~30 min).
   Same pattern as failure_attribution_breakdown(): which components
   fail on fix4's multi_region rows vs fix1's? If bridge text or attach
   regressed, that's the lever.
2. Based on result, decide whether to chase the multi_region recovery
   (could close to fix1's 0.6190 baseline level) or move to wrong_slot.

Fix3b (milder mem_rel weights) is no longer the highest-priority option:
fix4 already captured the encoder-reorganization benefit that motivated
fix3b, so the residual upside there is small.
```

What this stage must do:

```text
PredAlignNet (PRED-v2) is a goal-conditioned aligner.
It takes gold session_nodes as input and cannot run without them.

PRED-v3 is the stage that closes the loop:
  input:  signal + candidate spans + graph context (no oracle goals)
  output: session_node candidates usable by the aligner
```

PRED-v3 v2 Contract Alignment (COMPLETED):

```text
The v1 model hit 88% on procedural val but 0% on manual holdout.
We diagnosed a schema mismatch and aligned the generator (ngr_v1_tasks.py) 
to match the environment's extractive capabilities.
Result: Holdout row_complete jumped from 0% -> 30%. mixed_add_link is solved (3/3).
```

Next active stage:

```text
PRED-v3.2 Procedural Generator Coverage & Evaluation Polish (COMPLETED):

```text
Goal: Fix evaluation noise masking true capability gaps.
1. 'long_decompose' naturally predicts 0 attachments. 'diagnose_holdout.py' was predicting edges for padded slots. Fixed.
2. 'multi_region_attach' bridge templates aligned between environment and test data.
Result: multi_region_attach passes text_faithful metrics cleanly!
```

Next active stage:

```text
PRED-v3.3 Integration into Trajectory Aligner Loop

Goal:
  Now that PRED-v3 produces reliable session_nodes (1-node outputs without manual oracle goals), it needs to be wired into the phase 9 pipeline (PredAlignNet / Tool-Policy Trajectory Refinement). 
  We need to test the end-to-end integration of PRED-v3 candidate proposer -> PRED-v2 AlignNet -> Tool Trajectory.
```

```text
1. Supervision / data:
   reuse the existing pred_v1 rows directly
   goal.session_nodes are already the proposer targets
   no new data generation is required for v1

2. Output schema:
   start with a fixed max-slot proposer plus per-slot use gating
   each slot predicts:
     - use_this_slot
     - spec_type
     - span pointer or none
   this is less expressive than a ranked list with stop token, but much simpler
   to train and enough to establish an end-to-end baseline

3. Encoder:
   reuse the same BoW + MiniLM cached embedding inputs already used in PRED-v2
   to avoid representation drift between proposer and aligner

4. Integration target:
   first end-to-end metric is:
     aligner(proposer(signal, graph)) -> row_complete
   fix21's 0.3208 remains the oracle-proposer upper bound
```

PRED-v3+ unified status:

```text
IMPLEMENTED / FIRST BASELINE TRAINED

new files:
  unified_proposal_aligner_model.py
  train_unified_v1.py

smoke validation:
  - finite forward pass
  - finite weighted loss components
  - oracle text reconstruction exact on sampled rows
  - 20-row / 50-step subset loss 4.5926 -> 2.3870

critical implementation notes:
  1. unified state_h intentionally does not detach pooled_mem
  2. slot->memory attention must handle zero-memory rows without all-masked MHA
  3. non-synthesis node text reconstruction must use canonical slot text for
     the predicted span_id, not raw spans[*].text

next:
  PRED-v3+.2
  1. add eval_unified_v1.py
  2. add per-task diagnostics
  3. verify the unified baseline directly against the old split pipeline on the
  same held-out metric definitions
```

PRED-v3+.2 status:

```text
COMPLETE / EXTERNAL EVAL CONFIRMED

new file:
  eval_unified_v1.py

fresh-load checkpoint eval:
  checkpoint = out_unified_v1_20260512_fix1/best_unified_v1.pt
  report     = out_unified_v1_20260512_fix1/eval_report.json

parity with trainer checkpoint metrics:
  exact on every checked metric
  delta = 0.0

per-task row_complete:
  covered_long_signal = 0.4146
  long_decompose      = 0.2875
  mixed_add_link      = 0.5250
  multi_region_attach = 0.6190

dominant failure components:
  span   = 176
  text   = 140
  edge   = 117
  attach = 60

Interpretation:
  the unified baseline is real, externally reproducible, and should now replace
  the old split pipeline as the primary PRED-v3+ baseline.

failure-attribution summary:
  covered_long_signal:
    dominant combo = span+text
    slot span acc  = [0.8049, 0.6341, 0.7805]

  long_decompose:
    dominant combo        = edge+span+text
    isolated edge-only    = 43 rows
    slot span acc         = [0.8000, 0.7125, 0.8875]

  mixed_add_link:
    isolated attach-only  = 19 rows
    isolated span-only    = 8 rows

  multi_region_attach:
    isolated span-only    = 10 rows
    slot span acc         = [0.7778, 0.6667]

Implication:
  the next focused iteration should target slot-1 span quality on
  covered_long_signal / long_decompose first, then revisit attachment decoding
  for mixed_add_link as a secondary lever.
```

PRED-v3.1 status:

```text
COMPLETE

prepare_proposer_data.py generated:
  artifacts/proposer_v1_20260512/proposer_{train,val}.jsonl
  k_max = 3
  dropped slots = 0 on both splits

eval_proposer_roundtrip.py validated the schema against fix21:
  row_complete_rate               = 0.3208
  edge_f1                         = 0.8129
  edge_relation_acc_on_gold       = 0.8343
  attachment_relation_acc_on_gold = 0.8042
  cover_f1                        = 0.9959

This matches the fix21 aligner upper bound exactly.
Canonical slot ordering changes raw session-node order on some rows
(~24.3% of val), but preserves named session-node content and all
non-session goal structure 100%.
```

Immediate implementation plan:

```text
1. keep the plain fixed-slot proposer baseline as active
2. design the next proposer improvement without autoregressive previous-slot coupling
3. only after that decide whether to move to a more flexible ranked-list proposer
```

Why this order:

```text
The fixed-slot proposer is the cheapest path to a real end-to-end baseline.
Do not over-design ranked generation before the first loop-closing result exists.
```

PRED-v3.2 status:

```text
IMPLEMENTED / SMOKE VALIDATED

new files:
  proposer_model.py
  train_proposer_v1.py
  eval_proposer_v1.py

bridge invariant verified:
  multi_region_attach => exactly 1 bridge
  other tasks         => 0 bridges

first smoke run:
  out_proposer_v1_20260512_smoke/best_proposer_v1.pt

proposer metrics:
  use_acc                = 0.9733
  span_acc_on_used       = 0.3937
  bridge_acc_on_used     = 0.9399
  slot_row_complete_rate = 0.0519
  slot2_use_acc          = 0.9198

end-to-end through fix21:
  row_complete_rate = 0.0024

Meaning:
  the pipeline now closes end-to-end, but the first proposer baseline is
  dominated by span-pointer error. use-gating and bridge classification are
  already easy relative to span prediction.
```

PRED-v3.3 status:

```text
REJECTED

attempted change:
  MLP span scorer with slot_query*cand_h interaction and previous-slot
  pointer features, trained with teacher forcing and decoded autoregressively

result:
  baseline fix1:
    span_acc_on_used       = 0.7722
    slot_row_complete_rate = 0.5778
    top3 span recall       = 0.9056
    true row_complete      = 0.2476

  fix2 scorer upgrade:
    span_acc_on_used       = 0.7607
    slot_row_complete_rate = 0.5448
    top3 span recall       = 0.8761
    true row_complete      = 0.2311

meaning:
  the upgrade hurt both top-k candidate quality and final ranking.
  It did not improve slot 1 or shared-anchor rows.

active proposer baseline remains:
  out_proposer_v1_20260512_fix1_baseline10/best_proposer_v1.pt
```

PRED-v3.4 status:

```text
REJECTED

purpose:
  isolate whether the regression came from AR features or from the
  interaction-MLP scorer itself

change:
  kept interaction scorer
  removed autoregressive previous-slot features

result:
  baseline fix1:
    span_acc_on_used       = 0.7722
    slot_row_complete_rate = 0.5778
    top3 span recall       = 0.9056
    true row_complete      = 0.2476

  fix3 interaction-only:
    span_acc_on_used       = 0.7607
    slot_row_complete_rate = 0.5377
    top3 span recall       = 0.8875
    true row_complete      = 0.2311

meaning:
  AR features were part of the regression, but not the whole story.
  The interaction-MLP scorer also fails to beat the plain baseline.

active proposer baseline stays:
  out_proposer_v1_20260512_fix1_baseline10/best_proposer_v1.pt
```

PRED-v3.5 status:

```text
REJECTED

attempted change:
  keep the simple dot-product span scorer
  add DETR-lite slot-query refinement:
    - slot->candidate cross-attention
    - slot self-attention

result:
  active baseline fix1:
    span_acc_on_used       = 0.7722
    slot_row_complete_rate = 0.5778
    top3 span recall       = 0.9056
    true row_complete      = 0.2476

  fix4 DETR-lite + dot:
    span_acc_on_used       = 0.7035
    slot_row_complete_rate = 0.4717
    top3 span recall       = 0.8646
    true row_complete      = 0.2099

meaning:
  cross-slot / cross-candidate attention does not help on the current
  candidate representation.
  It degrades candidate ranking quality, with the largest collapse on slot 1:
    slot 1 span_acc_on_used = 0.5354

active proposer baseline remains:
  out_proposer_v1_20260512_fix1_baseline10/best_proposer_v1.pt
```

PRED-v3.6 status:

```text
PROMOTED / NEW ACTIVE BASELINE

purpose:
  isolate whether the fix4 regression came from slot attention itself or from
  the simultaneous scorer swap

change:
  keep DETR-lite slot-query refinement:
    - slot->candidate cross-attention
    - slot self-attention
  but restore the original fix1 span scorer:
    concat_mlp(slot_query, cand_h)

checkpoint:
  out_proposer_v1_20260512_fix5_attn_concat10/best_proposer_v1.pt

result vs old active baseline fix1:
  fix1:
    span_acc_on_used       = 0.7722
    slot_row_complete_rate = 0.5778
    top3 span recall       = 0.9056
    true row_complete      = 0.2476

  fix5 attention + concat:
    span_acc_on_used       = 0.7798
    slot_row_complete_rate = 0.6014
    top3 span recall       = 0.8770
    true row_complete      = 0.2594

meaning:
  the attention layers are not the source of the fix4 regression.
  The bad result in fix4 came from changing both:
    - scorer: concat -> dot
    - attention: none -> detr

  With the original scorer restored, slot attention gives a real gain:
    slot 1 span_acc_on_used: 0.6981 -> 0.7193
    needs_synthesis_false true row_complete: 0.2239 -> 0.2438

new active proposer baseline:
  out_proposer_v1_20260512_fix5_attn_concat10/best_proposer_v1.pt
```

PRED-v3.7 status:

```text
REJECTED / CEILING TEST

purpose:
  test whether the current fix5 proposer still has meaningful architectural
  headroom under simple scaling

change:
  keep fix5 architecture unchanged
  scale:
    hidden_dim 256 -> 512
    epochs 10 -> 30

checkpoint:
  out_proposer_v1_20260512_fix6_capacity512_30ep/best_proposer_v1.pt

result vs active baseline fix5:
  fix5:
    span_acc_on_used       = 0.7798
    slot_row_complete_rate = 0.6014
    true row_complete      = 0.2594
    needs_synthesis_false  = 0.2438

  fix6 capacity scan:
    span_acc_on_used       = 0.7922
    slot_row_complete_rate = 0.6156
    true row_complete      = 0.2547
    needs_synthesis_false  = 0.2239

meaning:
  extra capacity improves proposer-internal slot metrics, but does not improve
  the deployable downstream metric.
  The synthesis-free subset actually regresses back to the old fix1-level
  result, so simple scaling does not unlock more useful headroom.

active proposer baseline remains:
  out_proposer_v1_20260512_fix5_attn_concat10/best_proposer_v1.pt
```

PRED-v3.8 status:

```text
IMPLEMENTED / DIAGNOSTICALLY SUCCESSFUL

what changed:
  - added a strict text-faithful metric to end-to-end proposer eval
  - added deterministic template synthesis as post-processing
  - verified from data that both synthesis tasks are exactly reconstructible
    from row context only

template verification:
  mixed_add_link heuristic exact match   = 868 / 868 = 1.000
  multi_region_attach heuristic exact match = 374 / 374 = 1.000

strict floor on fix5 without synthesis:
  overall text_faithful_row_complete_rate        = 0.0000
  needs_synthesis_true text_faithful_acc         = 0.0000
  needs_synthesis_true text_faithful_row_complete_rate = 0.0000

with deterministic synthesis enabled:
  overall row_complete_rate                      = 0.2759   (was 0.2594)
  overall text_faithful_acc                      = 0.1973   (was 0.1659)
  needs_synthesis_true row_complete_rate         = 0.3049   (was 0.2735)
  needs_synthesis_true text_faithful_acc         = 0.0740   (was 0.0000)
  text_faithful_row_complete_rate                = 0.0000   (still zero)

meaning:
  deterministic synthesis is the right v1 design and should remain in the
  pipeline.
  But strict whole-row exactness is now bottlenecked by proposer anchor errors
  on the non-synthesized slots, not by missing text generation.
```

PRED-v3.8b status:

```text
BUG FIXED / REAL SYNTHESIS BASELINE RECORDED

bug:
  the first synthesizer version keyed off reconciled gold names
  ("source_note", "new_note", "support_note", "bridge")
  instead of structural role.
  When proposer span picks were wrong, reconciliation renamed slots to
  orphan_k and synthesis silently skipped them.

fix:
  - carry slot_idx + predicted is_bridge through predicted_slots_for_row
  - identify synthesis roles structurally:
      mixed_add_link: used-slot order
      multi_region_attach: predicted is_bridge / node_type

re-run after the fix:
  report:
    out_proposer_v1_20260512_fix5_attn_concat10/eval_report_textfaithful_synth_fixed.json

  overall:
    row_complete_rate               = 0.1745
    text_faithful_acc               = 0.1983
    text_faithful_row_complete_rate = 0.0000

  needs_synthesis_true:
    row_complete_rate               = 0.1121
    text_faithful_acc               = 0.0762
    text_faithful_row_complete_rate = 0.0000

meaning:
  this is the real post-synthesis baseline.
  Deterministic synthesis is still correct, but once synthesis actually fires
  broadly, the fix21 aligner suffers a strong input-distribution shift from the
  rewritten proposer text.

next implication:
  separate future diagnostics into:
    1. slot-text exactness before the aligner
    2. aligner robustness to synthesized proposer text
```

PRED-v3.8c status:

```text
COMPLETE / DECISIVE DIAGNOSTIC

run definition:
  A = oracle slots + oracle text            -> fix21 aligner   = 0.3208
  B = oracle slots + synthesized text       -> fix21 aligner   = 0.2170
  C = real proposer + synthesized text      -> fix21 aligner   = 0.1745

gap decomposition:
  A - B = 0.1038   aligner robustness gap
  B - C = 0.0425   proposer error gap

meaning:
  the larger loss is aligner sensitivity to synthesized proposer text.
  Proposer slot error is still real, but it is the secondary gap.

next main patch:
  aligner adaptation to template-synthesized session-node text
  is now higher ROI than another proposer-only experiment.
```

PRED-v3.8d status:

```text
REJECTED / SYNTH-AWARE ALIGNER FINE-TUNE DID NOT HIT THE REAL DISTRIBUTION

run:
  out_pred_v1_train_20260512_fix24_synth_finetune

training setup:
  - warm-start fix21
  - freeze_mode = synth_finetune
  - synth_swap_prob = 0.5
  - lr = 1e-5
  - epochs = 8

training result:
  best val_gold row_complete_rate  = 0.3090
  best val_synth row_complete_rate = 0.3090

critical finding:
  val_gold and val_synth were bit-identical through the run.
  The synth-swap path over gold goal session_nodes was effectively an identity
  transform, so the aligner never saw the problematic proposer/reconciled text
  distribution during training.

post-run deployment diagnostics:
  oracle slots + synthesized text -> new aligner = 0.2052
  previous fix21 Run B baseline                  = 0.2170

  real proposer + synthesized text -> new aligner = 0.1509
  previous fix21 Run C baseline                   = 0.1745

meaning:
  this adaptation attempt regressed both oracle-synth and real end-to-end
  performance.
  The failure mode is now understood:
  training-time synthesis over gold session_nodes is not the same distribution
  as synthesized/reconciled proposer slots.

next aligner adaptation requirement:
  train on proposer-style reconstructed rows, not identity-rewritten gold rows.

active downstream aligner remains:
  out_pred_v1_train_20260512_fix21_mememb_rel_freeze/best_pred_v1.pt
```

PRED-v3 prerequisite note:

```text
The active row-complete aligner baseline is still fix21.
PRED-v3 should use it as the downstream consumer for the first loop-closing run.
```

Next decision if PRED-v3 v1 underfits:

Recommended next stage:
  proposer-only architecture tuning is now in the diminishing-returns regime.
  Synthesis is now in place and the next dominant lever is aligner adaptation
  to synthesized proposer text.
  After that, proposer slot-anchor accuracy remains the secondary ROI lever,
  especially source/support slot quality on synthesis-needing rows.
```

---

## PRED-v0 Prototype Loop Closure - 2026-05-11

Archived predictor prototype:

```text
traverse_threshold_draft_edit.py --controller-mode predictor_prototype
```

Current held-out predictor metrics:

```text
output:
  out_traverse_predictor_proto_val_20260511.json

overall:
  predictor_session_node_recall       = 0.3231
  predictor_session_node_precision    = 0.3204
  predictor_session_edge_recall       = 0.1332
  predictor_session_edge_precision    = 0.1887
  predictor_attachment_recall         = 0.9843
  predictor_attachment_precision      = 0.9917
  predictor_covered_recall            = 0.0
  predictor_task_complete_proxy_rate  = 0.1203
  predictor_commit_type_accuracy      = 1.0
```

Meaning:

```text
The loop is now clean:
- predictor_prototype predicts a goal spec
- executor runs that spec deterministically
- predictor quality is measurable separately from executor correctness
```

Verified invariant:

```text
Synthesis goal specs in artifacts/tasks_trv_executor_20260511 have non-empty span_text:
  mixed_add_link::source_note   160 / 160
  mixed_add_link::new_note      160 / 160
  multi_region_attach::support_note 63 / 63
  multi_region_attach::bridge       63 / 63
```

PRED-v1 completed:

```text
pred_tasks.py
artifacts/pred_v1_20260511/pred_{train,val}.jsonl

Each row:
  input:  signal, spans, graph_path, initial_memory_node_ids
  target: goal (effective, covered tasks auto-expanded)
  labels: span_oracle (per-node best_span_id + all_scores)
          is_pseudo_goal
          meta: num_nodes, num_edges, num_attachments, num_covered

Train: 1885 rows, span_coverage 0.984
Val:   424 rows,  span_coverage 0.9676
```

Current next patch:

```text
PRED-v2

Goal:
  build the predictor model that reads (signal, spans, graph_path) and
  predicts goal spec structure

Targets:
  1. span ranker: score each span per goal session position
  2. node count / type: predict num_nodes and node_type sequence
  3. edge predictor: predict (src_idx, dst_idx, relation) for each edge
  4. attachment predictor: match session to memory node + relation
  5. commit classifier: add_node vs no_op
```

---

## TRV-v1.1 / SYN-v0.2 Match Ownership Fix - 2026-05-11

Active controller:

```text
traverse_threshold_draft_edit.py
```

Latest held-out 20-row smoke:

```text
output:
  out_traverse_draft_smoke20_synv0_matchfix2.json

overall:
  synthesis_used_rate = 0.5
  task_complete_proxy_rate = 0.85
  session_node_recall = 1.0
  session_edge_recall = 0.90625
  attachment_recall = 1.0
  covered_complete_rate = 1.0
```

What this patch fixed:

```text
- SYN-v0-created nodes now own their intended goal slots during final matching
- synthesis no longer adds support edges / attachments to stale lexical draft nodes
- mixed_add_link now completes on the smoke slice
- multi_region_attach now completes on the smoke slice
```

What remains:

```text
long_decompose:
  task_complete_proxy_rate = 0.5
  session_edge_recall = 0.75

Reason:
  the missing second edge is conceptual
  it is not directly backed by a memory-graph edge
  lowering post_edge_threshold did not fix it
```

Current next patch:

```text
TRV-v1.2

Goal:
  add a conceptual post-edge heuristic for long_decompose second-edge completion
  without regressing the repaired SYN-v0 path
```

---

## TRV-v1.3 Precision Cleanup v2 - 2026-05-11

Active controller:

```text
traverse_threshold_draft_edit.py
```

Full 47-row val after subtractive cleanup:

```text
output:
  out_traverse_draft_val47_precision_v2.json

overall:
  task_complete_proxy_rate = 0.9574
  task_complete_strict_rate = 0.6383

  session_node_precision = 0.9539
  session_edge_precision = 0.8515
  attachment_precision = 0.9787

  false_edge_rate = 0.1485
  false_attachment_rate = 0.0213
```

What this patch fixed:

```text
- mixed_add_link is now strict-clean on the held-out val set
- multi_region_attach is now mostly strict-clean
- synthesis rows no longer keep extra target-attachment competitors
- synthesis rows no longer keep most extra lexical competitor nodes
```

Remaining concentration of error:

```text
long_decompose:
  task_complete_strict_rate = 0.0
  session_edge_precision = 0.5849
  false_edge_rate = 0.4151

This is now the dominant remaining precision problem.
```

Current next patch:

```text
TRV-v1.4

Goal:
  improve long_decompose strict completion only

Priority:
  1. reduce extra draft-node creation on decomposition rows
  2. reduce extra non-gold structural edges on decomposition rows
  3. preserve current strict gains on mixed_add_link and multi_region_attach
```

---

## TRV-v1.3 Evaluation Integrity / Precision Gap - 2026-05-11

Active controller:

```text
traverse_threshold_draft_edit.py
```

Full graph-held-out val run:

```text
output:
  out_traverse_draft_val47_precision.json

overall:
  task_complete_proxy_rate = 1.0
  task_complete_strict_rate = 0.2128

  session_node_precision = 0.8121
  session_edge_precision = 0.6954
  attachment_precision = 0.7766

  false_edge_rate = 0.3046
  false_attachment_rate = 0.2234
  avg_extra_nodes = 0.5957
```

Interpretation:

```text
The active controller now has perfect recall on this held-out val split,
but the previous "complete" metric was recall-only.

Strict completion is low because the controller still:
- creates extra nodes
- adds extra session edges
- adds extra attachments
```

Most important task-level precision gaps:

```text
long_decompose:
  task_complete_strict_rate = 0.0
  session_edge_precision = 0.5849
  false_edge_rate = 0.4151

mixed_add_link:
  task_complete_strict_rate = 0.0909
  session_edge_precision = 0.3939
  attachment_precision = 0.5909

multi_region_attach:
  task_complete_strict_rate = 0.1429
  attachment_precision = 0.5714
```

Current next patch:

```text
TRV-v1.3

Goal:
  improve strict completion, not recall

Priority:
  1. suppress extra conceptual edges on long_decompose
  2. suppress duplicate / extra support edges on mixed_add_link
  3. suppress extra non-gold attachments on multi_region_attach
  4. keep graph-held-out recall at current level while raising precision
```

Important evaluation note:

```text
The 20-row smoke slice is no longer an unbiased checkpoint because thresholds
were tuned repeatedly against that slice.

Use the full 47-row graph-held-out val set for controller comparisons until a
fresh untuned graph split is generated.
```

---

## TRV-v1.2 Conceptual Post-Edge Fix - 2026-05-11

Active controller:

```text
traverse_threshold_draft_edit.py
```

Latest held-out 20-row smoke:

```text
output:
  out_traverse_draft_smoke20_synv0_conceptual3.json

overall:
  synthesis_used_rate = 0.5
  task_complete_proxy_rate = 1.0
  session_node_recall = 1.0
  session_edge_recall = 1.0
  attachment_recall = 1.0
  covered_complete_rate = 1.0
```

What this patch fixed:

```text
- long_decompose reverse-neighbor second-edge misses
- conceptual fallback now uses signal order for direction
- reverse-edge fallback uses relation = related, matching the task generator
- covered rows are protected from false conceptual edges
```

Result:

```text
covered_long_signal:
  task_complete_proxy_rate = 1.0

long_decompose:
  task_complete_proxy_rate = 1.0
  session_edge_recall = 1.0

mixed_add_link:
  task_complete_proxy_rate = 1.0

multi_region_attach:
  task_complete_proxy_rate = 1.0
```

Result on the 20-row smoke slice:

```text
proxy complete:
  1.0
```

---

## TRV-v1 / SYN-v0 Capability Split - 2026-05-11

Active controller remains:

```text
traverse_threshold_draft_edit.py
```

New routing contract:

```text
1. Run traversal-first draft editing on every row.
2. Inspect unresolved add_node commits after draft scoring.
3. If unresolved add_node work remains, run SYN-v0.
4. SYN-v0 fills only the missing sessions / support edge / required attachments.
```

SYN-v0 limitations by design:

```text
- deterministic template text writer only
- no LM dependency yet
- support relation hardcoded for synthesized session edges
- meant as a structural capability patch, not a final text generation system
```

Held-out 20-row smoke:

```text
output:
  out_traverse_draft_smoke20_synv0.json

overall:
  synthesis_used_rate = 0.15
  task_complete_proxy_rate = 0.5
  session_node_recall = 1.0
  session_edge_recall = 0.46875
  attachment_recall = 0.75
  covered_complete_rate = 1.0
```

Task split:

```text
covered_long_signal:
  task_complete_proxy_rate = 1.0

long_decompose:
  task_complete_proxy_rate = 0.5

mixed_add_link:
  task_complete_proxy_rate = 0.5
  synthesis_used_rate = 0.5

multi_region_attach:
  task_complete_proxy_rate = 0.1667
  synthesis_used_rate = 0.1667
```

Current next patch:

```text
TRV-v1.1 / SYN-v0.1

Goal:
  improve multi_region_attach attachment completion and support-edge completion
  without regressing covered_long_signal or mixed_add_link gains
```

---

## TRV-v0.5 Structural Fixes - 2026-05-11

Applied to the active controller:

```text
graph_core.py
traverse_threshold_draft_edit.py
```

What changed:

```text
- directed traversal and directed edge lookup
- multiple draft nodes per memory when spans differ
- low-overlap span blocking
- weak bridge pruning / bridge disabled by default
- post-traversal global draft-edge scoring
- post-traversal global cover / attach realignment
- neighbor fanout cap to reduce traversal sprawl
```

Held-out 20-row smoke result:

```text
output:
  out_traverse_draft_smoke20_fixed_v2.json

overall:
  task_complete_proxy_rate = 0.35
  session_node_recall = 0.9
  session_edge_recall = 0.28125
  attachment_recall = 0.45
  covered_recall = 1.0
  covered_complete_rate = 1.0
  avg_visited_nodes = 17.1
```

Task split:

```text
covered_long_signal:
  task_complete_proxy_rate = 1.0

long_decompose:
  task_complete_proxy_rate = 0.5
  session_edge_recall = 0.75

mixed_add_link:
  task_complete_proxy_rate = 0.0
  session_edge_recall = 0.0

multi_region_attach:
  task_complete_proxy_rate = 0.0
  attachment_recall = 0.4167
```

Current next patch:

```text
TRV-v0.6 mixed-task node-composition patch

Goal:
  improve mixed_add_link and multi_region_attach without regressing the new
  covered-task and long_decompose gains
```

---

## Current Stage: TRV-v0.4 Active - Edit While Traversing

The project has pivoted again.

The official active goal is no longer to make the `NGR-v1a` action-sequence controller work as the main path.

The active goal is:

```text
Traverse the graph, create temporary draft edits on the way, and score completion directly against goal structure.
```

The archived `NGR-v1a` path remains available only for diagnostics and historical comparison.

---

## Architecture Summary

```text
signal + graph
→ candidate subgraph
→ text/node/edge embeddings
→ graph encoder
→ recurrent state encoder
→ policy heads
→ discrete tool/action/target/relation decisions
→ environment executes tools
→ final edit sequence
```

The model predicts structure. The system serializes JSON.

---

## Non-Negotiable Design Rules

### Rule 1 — No fallback as planner teacher

Do not allow:

```text
fallback add_node
bootstrap_seed
system recovery
heuristic repair
```

to compete with the learned policy and then become training labels.

If the policy fails, record a failed trajectory.

Do not rescue it and train on the rescue.

---

### Rule 2 — Hard constraints are allowed

Allowed:

```text
node must exist
relation must be in enum
schema must be valid
target must be in candidate/retrieved set
no mutation after weak evidence
step budget limit
```

These are environment validity rules, not heuristic planning.

---

### Rule 3 — Planner does not generate raw JSON

The graph policy outputs:

```text
action IDs
node pointers
region pointers
relation class
stop/continue
```

Then the system constructs JSON.

---

### Rule 4 — LM is not the planner

Allowed LM usage:

```text
write new node text
write updated node text
write conflict resolution text
write summary text
```

Not allowed LM usage:

```text
choose tool sequence
choose final edit type
choose graph target IDs
choose relation labels
```

---

## Gate Table

```text
Gate      Name                                      Status
────────────────────────────────────────────────────────────────────
N0        Repo reset + plan pivot                    DONE
NGR-v0    One-step edit policy                       BASELINE / KEEP
NGR-v0.1  Coverage-vs-novelty patch                  PARTIAL / USEFUL
NGR-v1a   Session-graph edit-program policy          ACTIVE
NGR-v1a.1 Span + weak-retrieval environment patch     DONE
NGR-v1a.2 Progress-state data + tuple-set loss        PARTIAL PASS
NGR-v1a.3 Duplicate blocking + weighted tuple loss    PARTIAL PASS
NGR-v1a.4 Tuple-beam decode + exclusive phase rows    PATCHED / RETRIEVAL COLLAPSE EXPOSED
NGR-v1a.6 No-retrieval full-graph ablation            ACTIVE / SMOKE RUN DONE
NGR-v1b   Real multi-retrieval + action-history model DEFERRED UNTIL v1a.6 COMMITS
NGR-v2    Graph policy + text writer                  PLANNED
NGR-v3    Graph-grounded answer/reasoning agent       PLANNED
```

---

## Inspection Fix Update â€” 2026-05-09

The active evaluator path was patched after a Python-code inspection found three
behavioral mismatches.

Changed file:

```text
eval_ngr_v1.py
```

Applied fixes:

```text
1. STOP runtime gate is now structural, not count-only.
   Runtime must match goal node/edge/attachment/add structure before STOP is
   allowed.

2. MARK_COVERED is now task-aware.
   It is blocked on non-covered edit tasks.

3. Empty tuple beams no longer silently force STOP.
   The decoder now retries with exhaustive tuple enumeration first and only then
   uses a safe fallback action path.
```

Status implication:

```text
Old v1a.4 rollout metrics were produced under weaker evaluator semantics.
They should be treated as stale until rollout evaluation is rerun.
```

Next command to run:

```powershell
python eval_ngr_v1.py `
  --checkpoint .\out_ngr_v1a4\best_ngr_v1a.pt `
  --val-jsonl .\artifacts\tasks_v1a_spanfix\ngr_v1_val.jsonl `
  --max-rollout-rows 200 `
  --save-rollouts-jsonl .\out_ngr_v1a4\rollouts_eval_after_inspection_fix.jsonl
```

---

## Smoke Pipeline Run â€” 2026-05-09

A reduced end-to-end smoke pipeline was executed successfully.

Commands run:

```powershell
python .\ngr_v1_tasks.py `
  --graphs-dir .\graphs `
  --out-dir .\artifacts\tasks_v1a_smoke_20260509 `
  --max-tasks 240 `
  --per-type-per-graph 8

python .\ngr_v1_progress_tasks.py `
  --goal-train-jsonl .\artifacts\tasks_v1a_smoke_20260509\ngr_v1_train.jsonl `
  --goal-val-jsonl .\artifacts\tasks_v1a_smoke_20260509\ngr_v1_val.jsonl `
  --out-dir .\artifacts\tasks_v1a_progress_smoke_20260509 `
  --states-per-goal 8

python .\train_ngr_v1.py `
  --train-jsonl .\artifacts\tasks_v1a_progress_smoke_20260509\ngr_v1_progress_train.jsonl `
  --val-jsonl .\artifacts\tasks_v1a_progress_smoke_20260509\ngr_v1_progress_val.jsonl `
  --out-dir .\out_ngr_v1a_smoke_20260509 `
  --epochs 2 `
  --batch-size 32

python .\eval_ngr_v1.py `
  --checkpoint .\out_ngr_v1a_smoke_20260509\best_ngr_v1a.pt `
  --val-jsonl .\artifacts\tasks_v1a_smoke_20260509\ngr_v1_val.jsonl `
  --max-rollout-rows 48 `
  --save-rollouts-jsonl .\out_ngr_v1a_smoke_20260509\rollouts_eval.jsonl
```

Observed result:

```text
The full smoke pipeline works, but rollout quality is still failing.

overall:
  avg_steps            = 12.0
  invalid_action_rate  = 0.0
  repeated_action_rate = 0.6892
  session_node_f1      = 0.2604
  session_edge_f1      = 0.0
  memory_attachment_f1 = 0.1202
  commit_f1            = 0.0
  no_op_accuracy       = 0.0
```

Most common action:

```text
RETRIEVE_RELATED = 446
```

Current conclusion:

```text
The evaluator patch is active and the pipeline is runnable.
The next bottleneck is rollout policy collapse into repeated retrieval,
not data generation or training-script breakage.
```

---

# N1 — GraphPolicyEnv

## Goal

Create:

```text
graph_policy_env.py
```

This file defines the environment for the learned graph policy.

It should be independent from the old LLM planner.

## Required API

```python
class GraphPolicyEnv:
    def __init__(self, graph: MemoryGraph, config: GraphPolicyEnvConfig): ...

    def reset(self, signal: str, task: dict | None = None) -> dict:
        ...

    def observe(self) -> dict:
        ...

    def valid_action_mask(self) -> dict:
        ...

    def step(self, action: dict) -> tuple[dict, float, bool, dict]:
        ...

    def serialize_trajectory(self) -> dict:
        ...
```

## State Fields

The environment state should contain:

```text
signal
step
budget_left
active_regions
anchor_nodes
visited_nodes
frontier_nodes
candidate_nodes
candidate_edges
candidate_paths
candidate_edits
tool_history
last_action
done
```

## Supported Primitive Actions

```text
route_regions
expand_frontier
inspect_path
find_conflicts
propose_edit
stop
```

## Supported Final Edit Types

```text
no_op
add_node
update_node
link_nodes
create_bridge
resolve_conflict
summarize_cluster
```

## Candidate-Limited Targeting

At each step, target choices should be restricted to candidate objects:

```text
candidate regions
candidate nodes
candidate paths
candidate edits
```

Never train the policy to generate node IDs.

The model should select candidate indices/pointers.

## Hard Validity Checks

Invalid actions should return negative reward and not mutate the graph:

```text
unknown action type
target index out of range
node ID absent from graph
relation not in canonical enum
link_nodes src/dst same when not allowed
update_node target missing
resolve_conflict target not false/hypothesis/conflict-like
stop before any valid evidence when task requires evidence
```

## Pass Criteria

```text
loads all ./graphs/*.json
reset works on arbitrary signal
candidate subgraph is built
primitive actions update state
invalid actions are blocked
trajectory serialization works
no fallback action exists
```

---

# N2 — Corruption Task Generator

## Goal

Create:

```text
generate_graph_policy_tasks.py
```

It generates gold tasks from graph transformations, not heuristic winners.

## Input

```text
./graphs/*.json
```

## Output

```text
artifacts/tasks/graph_policy_train.jsonl
artifacts/tasks/graph_policy_val.jsonl
artifacts/tasks/graph_policy_summary.json
```

## Task Types

### 1. edge_mask

Original graph has:

```text
src --relation--> dst
```

Training graph removes the edge.

Gold:

```text
link_nodes(src, dst, relation)
```

### 2. node_mask

Original graph has a node with useful edges.

Training graph removes the node.

Gold:

```text
add_node(node_text, node_type, edges_to)
```

### 3. relation_corrupt

Original edge relation is changed.

Gold:

```text
correct relation
```

or:

```text
link_nodes(src, dst, correct_relation)
```

### 4. false_claim

Inject a false node.

Gold:

```text
resolve_conflict(target_id)
```

### 5. duplicate_signal

Signal repeats content already covered by existing nodes.

Gold:

```text
no_op
```

### 6. freshness_missing_target

Signal asks for an edit involving a node not present in retrieved/candidate context.

Gold:

```text
no_op
```

or later:

```text
request_retrieve
```

### 7. summarize_cluster

Pick a dense local cluster.

Gold:

```text
summarize_cluster(cluster_node_ids)
```

## Task Row Schema

```json
{
  "id": "edge_mask_000001",
  "task_type": "edge_mask",
  "graph_path": "graphs/cs4.json",
  "signal": "Radix sort depends on counting sort as a stable digit subroutine.",
  "masked": {
    "edge": {
      "src": "radix_sort_linear",
      "dst": "counting_sort_linear_time",
      "relation": "depend"
    }
  },
  "gold_trajectory": [
    {"action": "route_regions"},
    {"action": "expand_frontier", "target": "radix_sort_linear"},
    {"action": "inspect_path", "src": "radix_sort_linear", "dst": "counting_sort_linear_time"},
    {
      "action": "propose_edit",
      "edit": {
        "action": "link_nodes",
        "src": "radix_sort_linear",
        "dst": "counting_sort_linear_time",
        "relation": "depend"
      }
    },
    {"action": "stop"}
  ],
  "gold_final_edits": [
    {
      "action": "link_nodes",
      "src": "radix_sort_linear",
      "dst": "counting_sort_linear_time",
      "relation": "depend"
    }
  ]
}
```

## Pass Criteria

```text
>= 1000 generated tasks total
all task types represented
train/val split exists
no fallback/system-recovery labels
every gold target exists in the task graph or is intentionally masked
summary report shows action distribution
```

---

# N3 — GraphPolicyNet Supervised Training

## Goal

Create:

```text
graph_policy_model.py
train_graph_policy.py
```

## Model

Minimal v1:

```text
frozen text encoder
2-layer R-GCN-lite
small state transformer or GRU
policy heads
value head
```

## Inputs

```text
candidate node embeddings
edge index
edge relation IDs
node type IDs
retrieval/candidate scores
visited/frontier flags
tool history
signal embedding
```

## Outputs

```text
next_action distribution
edit_type distribution
source pointer over candidate nodes
target pointer over candidate nodes
region pointer
relation distribution
stop probability
value estimate
```

## Supervised Loss

```text
L =
  action_type_loss
+ edit_type_loss
+ source_pointer_loss
+ target_pointer_loss
+ relation_loss
+ region_pointer_loss
+ stop_loss
+ value_loss
```

## Pass Criteria

```text
valid action accuracy >= 95%
edit_type accuracy >= 70%
target pointer top-5 accuracy >= 80%
relation accuracy >= 75%
no_op precision >= 80%
invalid target rate <= 5%
```

---

# N4 — Tool-Sequence Imitation

## Goal

Train the model to output full action/tool sequences.

## Required Sequence Pattern

```text
route/expand/inspect/find_conflicts
→ propose_edit
→ stop
```

Not every task needs every tool, but the model must learn when each one is useful.

## Pass Criteria

```text
tool sequence exact match >= 50%
final edit correctness >= 60%
invalid action rate <= 5%
average steps <= budget
```

---

# N5 — Offline RL

## Goal

Use structured rewards from generated corruption tasks and graph QA probes.

## Rewards

Positive:

```text
recovered masked edge
recovered masked node
correct relation
resolved injected false claim
correct no_op on duplicate
correct no_op on missing target
useful summarize_cluster
```

Negative:

```text
wrong target
wrong relation
duplicate add_node
unsafe mutation
target missing node
stopped too early
tool spam
```

## Algorithms to Try

Start simple:

```text
pairwise action ranking
Q-learning over discrete actions
actor-critic with value head
best-of-N trajectory ranking
```

Do not start with PPO/GRPO until the environment is stable.

## Pass Criteria

```text
offline RL beats supervised-only policy on held-out corruption tasks
masked-edge recovery improves
duplicate add_node rate decreases
unsafe mutation rate = 0
```

---

# N6 — Text Writer Integration

## Goal

Use LM only after the graph policy chooses structure.

## Text Writer Inputs

```text
signal
chosen edit type
selected target nodes
selected evidence nodes
selected relation labels
old node text when updating
```

## Text Writer Outputs

```text
new_node.text
updated node text
conflict resolution text
summary text
```

## Pass Criteria

```text
text is not signal-copy
text uses selected evidence
text passes duplicate check
graph policy still owns action/target/relation decisions
```

---

# N7 — Compare Against Old LM Planner

## Baseline

Use archived old planner artifacts only as comparison.

## Evaluation Set

```text
held-out graph corruption tasks
held-out CS/math signals
old crossref signals if available
```

## Metrics

```text
final edit accuracy
tool sequence quality
schema validity
hallucinated ID rate
invalid target rate
duplicate add_node rate
no_op precision
runtime
VRAM
```

## Expected Win

```text
GraphPolicyNet should beat the LM planner on:
  schema validity
  target validity
  speed
  RL trainability
  hallucinated ID avoidance
```

---

## Files to Create First

```text
graph_policy_env.py
generate_graph_policy_tasks.py
graph_policy_model.py
train_graph_policy.py
eval_graph_policy.py
```

---

## Reusable Files from Old Repo

Use these as infrastructure only:

```text
graph_core.py
graph_walk_env.py
graph_dc_env.py
consistency_reward.py
qa_probe_reward.py
critic.py
action_repair.py
editor.py
graph_visualizer_streamlit.py
```

Do not preserve the old fallback-driven planner loop as the main training loop.

---

## Immediate Next Commands

After placing the JSON graphs in `./graphs`, first implement and test environment loading:

```powershell
python graph_policy_env.py --graph .\graphs\cs4.json --signal "Dijkstra fails with negative edges because finalized distances can later improve." --debug
```

Then generate tasks:

```powershell
python generate_graph_policy_tasks.py --graphs-dir .\graphs --out-dir .\artifacts\tasks --max-tasks 1000
```

Then train supervised policy:

```powershell
python train_graph_policy.py --train-jsonl .\artifacts\tasks\graph_policy_train.jsonl --val-jsonl .\artifacts\tasks\graph_policy_val.jsonl --out-dir .\out_graph_policy_v1
```

---

## Current One-Line Summary

The project has moved away from training a tiny language model as the planner. The active plan is to train a graph-native Neural Graph Reasoner that directly chooses discrete graph/tool/action sequences from graph state, with no fallback teacher and no JSON-token generation planner.


---

# Active Patch Task — NGR-v0.1 Coverage/Novelty Boundary

## Problem

The first shuffled NGR-v0 model learned graph pointers and relations, but failed the coverage-vs-novelty gate.

Observed validation:

```text
overall edit_acc: 0.770
relation_acc:    0.879
target_top5:     0.984
src_top5:        1.000
dst_top5:        1.000
```

Weakness:

```text
duplicate_signal / no_op accuracy: 0.323
node_mask / add_node accuracy:     0.545
```

Confusion:

```text
gold no_op:
  predicted no_op:    40
  predicted add_node: 70
  predicted link:     14

gold add_node:
  predicted add_node: 55
  predicted no_op:    37
  predicted link:      8
```

## Goal

Teach NGR-v0 the boundary:

```text
covered existing knowledge → no_op
missing concept            → add_node
```

without reintroducing fallback or heuristic planner labels.

## Files patched

Replace these files:

```text
graph_policy_model.py
train_graph_policy.py
graph_policy_tasks.py
eval_graph_policy.py
```

## Regenerate tasks

```powershell
python graph_policy_tasks.py `
  --graphs-dir .\graphs `
  --out-dir .\artifacts\tasks_v0_coverage `
  --max-tasks 3000 `
  --per-type-per-graph 180 `
  --false-claim-multiplier 6 `
  --node-mask-max-candidate-overlap 0.86
```

## Train

```powershell
python train_graph_policy.py `
  --train-jsonl .\artifacts\tasks_v0_coverage\graph_policy_train.jsonl `
  --val-jsonl .\artifacts\tasks_v0_coverage\graph_policy_val.jsonl `
  --out-dir .\out_graph_policy_v0_coverage `
  --epochs 10
```

Optional if no_op is still weak:

```powershell
python train_graph_policy.py `
  --train-jsonl .\artifacts\tasks_v0_coverage\graph_policy_train.jsonl `
  --val-jsonl .\artifacts\tasks_v0_coverage\graph_policy_val.jsonl `
  --out-dir .\out_graph_policy_v0_coverage_noopstrong `
  --epochs 10 `
  --edit-weights "no_op=3.25,add_node=1.75,link_nodes=1.0,update_node=0.7,resolve_conflict=0.7"
```

Optional if add_node becomes too weak:

```powershell
python train_graph_policy.py `
  --train-jsonl .\artifacts\tasks_v0_coverage\graph_policy_train.jsonl `
  --val-jsonl .\artifacts\tasks_v0_coverage\graph_policy_val.jsonl `
  --out-dir .\out_graph_policy_v0_coverage_addstrong `
  --epochs 10 `
  --edit-weights "no_op=2.50,add_node=2.25,link_nodes=1.0,update_node=0.7,resolve_conflict=0.7"
```

## Evaluate

```powershell
python eval_graph_policy.py `
  --checkpoint .\out_graph_policy_v0_coverage\best_graph_policy.pt `
  --val-jsonl .\artifacts\tasks_v0_coverage\graph_policy_val.jsonl
```

## Pass criteria

```text
overall edit_acc >= 0.75
relation_acc >= 0.80
pointer top5 >= 0.95
no_op recall >= 0.70
add_node recall >= 0.70
no_op precision >= 0.70
add_node precision >= 0.70
```

## Decision

If this passes:

```text
NGR-v0 PASSED
move to NGR-v1 multi-step sequence policy
```

If this fails:

```text
keep NGR-v0 active
add explicit binary coverage head:
  covered vs novel
then condition edit_type on coverage probability
```


---

# Active Patch Task — NGR-v1a.2 Progress-State Data

## Problem

The current NGR-v1a policy can create useful session nodes but does not yet execute a complete edit program.

Rollout symptoms:

```text
avg_steps = 12.0
early_stop_rate = 0.0
repeated_action_rate ≈ 0.6667
session_edge_f1 = 0.0
memory_attachment_f1 = 0.0
no_op_accuracy = 0.0
```

The data currently teaches components, not phase transitions.

## Goal

Generate progress-state training rows.

Each goal row becomes many rows:

```text
partial state
→ allowed_next action tuples
```

The model should learn:

```text
when to create
when to link session nodes
when to attach session nodes to memory
when to mark covered
when to propose no_op
when to stop
```

## Files

```text
ngr_v1_progress_tasks.py
train_ngr_v1.py
eval_ngr_v1.py
```

## Generate progress rows

First generate normal goal rows if needed:

```powershell
python ngr_v1_tasks.py `
  --graphs-dir .\graphs `
  --out-dir .\artifacts\tasks_v1a_spanfix `
  --max-tasks 2000 `
  --per-type-per-graph 80
```

Then generate progress-state rows:

```powershell
python ngr_v1_progress_tasks.py `
  --goal-train-jsonl .\artifacts\tasks_v1a_spanfix\ngr_v1_train.jsonl `
  --goal-val-jsonl .\artifacts\tasks_v1a_spanfix\ngr_v1_val.jsonl `
  --out-dir .\artifacts\tasks_v1a_progress `
  --states-per-goal 12
```

## Train

```powershell
python train_ngr_v1.py `
  --train-jsonl .\artifacts\tasks_v1a_progress\ngr_v1_progress_train.jsonl `
  --val-jsonl .\artifacts\tasks_v1a_progress\ngr_v1_progress_val.jsonl `
  --out-dir .\out_ngr_v1a_progress `
  --epochs 8
```

## Evaluate rollout on goal validation rows

```powershell
python eval_ngr_v1.py `
  --checkpoint .\out_ngr_v1a_progress\best_ngr_v1a.pt `
  --val-jsonl .\artifacts\tasks_v1a_spanfix\ngr_v1_val.jsonl `
  --max-rollout-rows 200 `
  --save-rollouts-jsonl .\out_ngr_v1a_progress\rollouts_eval.jsonl
```

## Metrics to watch

```text
avg_steps
repeated_action_rate
early_stop_rate
session_node_f1
session_edge_f1
memory_attachment_f1
commit_f1
no_op_accuracy
invalid_action_rate
```

## Pass target for this patch

```text
invalid_action_rate <= 0.05
avg_steps < 12.0
repeated_action_rate < 0.35
session_node_f1 >= 0.65
commit_f1 >= 0.45
no_op_accuracy > 0.0
memory_attachment_f1 > 0.0
```

This is not the final NGR-v1 pass target. It is the first progress-state sanity target.

## If this still fails

Next patch:

```text
1. Add explicit phase embedding to state.
2. Add action-history GRU instead of count vector.
3. Add rollout-aware best-of-N training.
4. Add stronger STOP/no_op terminal states.
```

---

# Active Patch Task — NGR-v1a.3 Finish-Program Behavior

## Problem

v1a.2 improved session-node creation and rollout length, but did not learn link/attach/no-op behavior.

Current failure:

```text
session_edge_f1 = 0.0
memory_attachment_f1 = 0.0
no_op_accuracy = 0.0
```

## Files patched

```text
ngr_v1_env.py
ngr_v1_progress_tasks.py
train_ngr_v1.py
eval_ngr_v1.py
```

## Regenerate progress rows

```powershell
python ngr_v1_progress_tasks.py `
  --goal-train-jsonl .\artifacts\tasks_v1a_spanfix\ngr_v1_train.jsonl `
  --goal-val-jsonl .\artifacts\tasks_v1a_spanfix\ngr_v1_val.jsonl `
  --out-dir .\artifacts\tasks_v1a_progress_finish `
  --states-per-goal 12
```

## Train

```powershell
python train_ngr_v1.py `
  --train-jsonl .\artifacts\tasks_v1a_progress_finish\ngr_v1_progress_train.jsonl `
  --val-jsonl .\artifacts\tasks_v1a_progress_finish\ngr_v1_progress_val.jsonl `
  --out-dir .\out_ngr_v1a_finish `
  --epochs 8
```

## Evaluate rollout

```powershell
python eval_ngr_v1.py `
  --checkpoint .\out_ngr_v1a_finish\best_ngr_v1a.pt `
  --val-jsonl .\artifacts\tasks_v1a_spanfix\ngr_v1_val.jsonl `
  --max-rollout-rows 200 `
  --save-rollouts-jsonl .\out_ngr_v1a_finish\rollouts_eval.jsonl
```

## Watch

```text
action_counts
repeat_action_counts
invalid_errors
session_edge_f1
memory_attachment_f1
no_op_accuracy
commit_f1
```


---

# Active Patch Task — NGR-v1a.4 Tuple-Beam + Exclusive Phases

## Why this is next

v1a.3 fixed duplicate loops but the policy collapsed into:

```text
CREATE_SESSION_NODE
→ PROPOSE_ADD_SESSION_NODE
→ STOP
```

Latest rollout:

```text
avg_steps = 3.01
invalid_action_rate = 0.0
repeated_action_rate = 0.0
session_node_f1 = 0.6833
session_edge_f1 = 0.0
memory_attachment_f1 = 0.0
commit_f1 = 0.4858
no_op_accuracy = 0.0
```

Action counts:

```text
CREATE_SESSION_NODE:        201
PROPOSE_ADD_SESSION_NODE:   198
STOP:                       200
MARK_COVERED:                 3
LINK_SESSION_NODES:            0
PROPOSE_LINK_SESSION_TO_MEMORY:0
PROPOSE_NO_OP:                 0
```

## Root cause

Training uses tuple-set loss:

```text
P(action + arguments)
```

Rollout still uses action-first greedy decoding:

```text
argmax(action_logits)
then argmax(pointer_heads)
```

This does not match the training objective.

## Files to patch

```text
eval_ngr_v1.py
ngr_v1_progress_tasks.py
train_ngr_v1.py
```

Optional if needed:

```text
ngr_v1_env.py
```

## Patch 1 — Tuple-beam decoding

Replace rollout decoding with tuple-beam decoding.

Candidate tuple examples:

```text
CREATE_SESSION_NODE(span_i, node_type)
LINK_SESSION_NODES(src_session_i, dst_session_j, relation)
MARK_COVERED(session_i, memory_j)
PROPOSE_ADD_SESSION_NODE(session_i)
PROPOSE_LINK_SESSION_TO_MEMORY(session_i, memory_j, relation)
PROPOSE_NO_OP
STOP
```

Tuple score:

```text
score(tuple)
=
log P(action)
+ log P(required pointers)
+ log P(relation if needed)
+ log P(node_type if needed)
```

The rollout should select the highest-scoring valid tuple.

## Patch 2 — Exclusive phase rows

In `ngr_v1_progress_tasks.py`, make hard phase rows exclusive:

```text
link phase:
  allowed_next = LINK_SESSION_NODES only

attach phase:
  allowed_next = PROPOSE_LINK_SESSION_TO_MEMORY only

cover phase:
  allowed_next = MARK_COVERED only

noop phase:
  allowed_next = PROPOSE_NO_OP only

stop phase:
  allowed_next = STOP only
```

Keep create/add rows too, but do not let `PROPOSE_ADD_SESSION_NODE` appear as a shortcut inside attach/link/cover phases.

## Patch 3 — Reduce add-node shortcut

Keep weighted tuple-set loss.

If the model still collapses into add-node:

```text
PROPOSE_ADD_SESSION_NODE weight: 0.75 → 0.45
CREATE_SESSION_NODE weight:      0.90 → 0.80
LINK_SESSION_NODES weight:       2.00 → 2.50
PROPOSE_LINK_SESSION_TO_MEMORY:  2.00 → 2.50
MARK_COVERED:                    2.75 → 3.00
PROPOSE_NO_OP:                   4.00 → 4.50
STOP:                            2.00 → 2.00
```

## Regenerate progress rows

```powershell
python ngr_v1_progress_tasks.py `
  --goal-train-jsonl .\artifacts\tasks_v1a_spanfix\ngr_v1_train.jsonl `
  --goal-val-jsonl .\artifacts\tasks_v1a_spanfix\ngr_v1_val.jsonl `
  --out-dir .\artifacts\tasks_v1a_progress_v1a4 `
  --states-per-goal 12
```

## Train

```powershell
python train_ngr_v1.py `
  --train-jsonl .\artifacts\tasks_v1a_progress_v1a4\ngr_v1_progress_train.jsonl `
  --val-jsonl .\artifacts\tasks_v1a_progress_v1a4\ngr_v1_progress_val.jsonl `
  --out-dir .\out_ngr_v1a4 `
  --epochs 8
```

## Evaluate

```powershell
python eval_ngr_v1.py `
  --checkpoint .\out_ngr_v1a4\best_ngr_v1a.pt `
  --val-jsonl .\artifacts\tasks_v1a_spanfix\ngr_v1_val.jsonl `
  --max-rollout-rows 200 `
  --save-rollouts-jsonl .\out_ngr_v1a4\rollouts_eval.jsonl
```

## Pass target

```text
invalid_action_rate <= 0.05
repeated_action_rate <= 0.10
session_node_f1 >= 0.65
session_edge_f1 > 0.0
memory_attachment_f1 > 0.0
no_op_accuracy > 0.0
commit_f1 >= 0.50
```

---

# Planned Stage — NGR-v1b Retrieval-Conditioned Multi-Step Reasoning

## Do not start v1b yet

v1b should not start until v1a.4 proves that the policy can perform the basic edit-program grammar.

Required before v1b:

```text
session_edge_f1 > 0.0
memory_attachment_f1 > 0.0
no_op_accuracy > 0.0
invalid_action_rate <= 0.05
repeated_action_rate <= 0.10
```

## v1a vs v1b

```text
v1a:
  learn session-graph edit program grammar

v1b:
  learn retrieval-conditioned multi-step reasoning
```

## v1b new skills

```text
decide when to retrieve
choose retrieval query source:
  signal
  span
  session node
  memory node
use weak retrieval flags
attach only after evidence exists
mark covered only after evidence exists
maintain action history with GRU/Transformer
handle multiple retrievals before commit
best-of-N / offline RL over final commit F1
```

## v1b files to create later

```text
ngr_v1b_env.py
ngr_v1b_tasks.py
ngr_v1b_model.py
train_ngr_v1b.py
eval_ngr_v1b.py
```

Alternative: keep names as `ngr_v1_*` and add `--mode v1b`, but separate files may be cleaner while the design is changing.

## v1b first benchmark

Use same goal validation rows, but require retrieval behavior:

```text
retrieval_call_rate > 0.0
gold_memory_retrieved_rate > 0.50
memory_attachment_f1 improves over v1a
covered/no_op improves over v1a
commit_f1 improves over v1a
```

## v1b warning

Do not let retrieval become a new heuristic fallback.

Retrieval may return candidates.

The policy must still choose:

```text
whether evidence is enough
which session node to attach
which memory node to attach
whether to mark covered
whether to stop
```


---

# Active Patch Task — Apply NGR-v1a.4 Inspection Fixes

## Replace files

```text
eval_ngr_v1.py
ngr_v1_progress_tasks.py
train_ngr_v1.py
```

## Generate rows

```powershell
python ngr_v1_progress_tasks.py `
  --goal-train-jsonl .\artifacts\tasks_v1a_spanfix\ngr_v1_train.jsonl `
  --goal-val-jsonl .\artifacts\tasks_v1a_spanfix\ngr_v1_val.jsonl `
  --out-dir .\artifacts\tasks_v1a_progress_v1a4 `
  --states-per-goal 12
```

## Check summary

```text
phase_exclusive should be true
multi_action_type_rows should be 0
```

## Train

```powershell
python train_ngr_v1.py `
  --train-jsonl .\artifacts\tasks_v1a_progress_v1a4\ngr_v1_progress_train.jsonl `
  --val-jsonl .\artifacts\tasks_v1a_progress_v1a4\ngr_v1_progress_val.jsonl `
  --out-dir .\out_ngr_v1a4 `
  --epochs 8
```

## Evaluate

```powershell
python eval_ngr_v1.py `
  --checkpoint .\out_ngr_v1a4\best_ngr_v1a.pt `
  --val-jsonl .\artifacts\tasks_v1a_spanfix\ngr_v1_val.jsonl `
  --max-rollout-rows 200 `
  --save-rollouts-jsonl .\out_ngr_v1a4\rollouts_eval.jsonl
```

## Optional decoder tuning

```powershell
python eval_ngr_v1.py `
  --checkpoint .\out_ngr_v1a4\best_ngr_v1a.pt `
  --val-jsonl .\artifacts\tasks_v1a_spanfix\ngr_v1_val.jsonl `
  --max-rollout-rows 200 `
  --arg-weight 1.5 `
  --stop-penalty 0.5
```

## Expected behavior change

The rollout should no longer be limited to:

```text
CREATE_SESSION_NODE
PROPOSE_ADD_SESSION_NODE
STOP
```

We want non-zero counts for:

```text
LINK_SESSION_NODES
PROPOSE_LINK_SESSION_TO_MEMORY
PROPOSE_NO_OP
MARK_COVERED
```

---

# Active Patch Task — NGR-v1a.6 No-Retrieval Ablation

## Status

```text
IMPLEMENTED / THREE-MODE EVAL ADDED / PHASE HEAD ADDED
```

## Files changed

```text
ngr_v1_env.py
ngr_v1_model.py
ngr_v1_progress_tasks.py
train_ngr_v1.py
eval_ngr_v1.py
PROGRESS.md
TASK_LIST.md
```

## Implemented behavior

```text
remove RETRIEVE_RELATED from active action set
expose full graph memory from reset
build progress rows against full-graph memory
train/eval only over structural actions
keep tuple-beam decoding and structural STOP gate
```

## Smoke pipelines for v1a.6

```text
artifacts/tasks_v1a6_smoke_20260509
artifacts/tasks_v1a6_progress_smoke_20260509
out_ngr_v1a6_smoke_20260509
artifacts/tasks_v1a6_progress_phasegate_smoke_20260509
out_ngr_v1a6_phasegate_smoke_20260509
```

## Observed result

```text
baseline no-retrieval:
  repeated_action_rate = 0.0
  RETRIEVE_RELATED     = 0
  session_node_f1      = 0.5119
  session_edge_f1      = 0.0042
  memory_attachment_f1 = 0.1590
  commit_f1            = 0.0243
  no_op_accuracy       = 0.0

phase-progress follow-up:
  repeated_action_rate = 0.0
  session_node_f1      = 0.6793
  session_edge_f1      = 0.0042
  memory_attachment_f1 = 0.2637
  commit_f1            = 0.5617
  no_op_accuracy       = 0.8

exact-progress eval:
  repeated_action_rate = 0.0
  session_node_f1      = 0.6875
  session_edge_f1      = 0.1250
  memory_attachment_f1 = 0.5625
  commit_f1            = 0.9688
  no_op_accuracy       = 1.0

three-mode phase-head eval:
  guided_exact_progress:
    commit_f1          = 0.9688
    session_edge_f1    = 0.1250
    memory_attachment_f1 = 0.5625
    no_op_accuracy     = 1.0
  phase_guided:
    commit_f1          = 0.5256
    session_edge_f1    = 0.0117
    memory_attachment_f1 = 0.2240
    no_op_accuracy     = 0.9333
  policy_only:
    commit_f1          = 0.2374
    session_edge_f1    = 0.0128
    memory_attachment_f1 = 0.1444
    no_op_accuracy     = 0.0
```

## Interpretation

```text
retrieval collapse is removed
phase-progress gating fixed most add/no-op failure
exact-progress decoding fixes the remaining edge/add completion path
three-mode eval now makes that boundary explicit
remaining weakness = policy_only autonomous rollout
```

## Next required patch

```text
1. recover policy_only edge completion while keeping the new no_op gains
2. keep the three-mode metric split explicit
3. preserve repeated_action_rate at 0.0
4. only begin v1b after policy_only mechanics are materially stronger
```

## Acceptance checks for next rerun

```text
guided_exact_progress stays recorded as upper-bound only
phase_guided stays recorded as middle-ground v1a metric
policy_only improves over current commit_f1 0.4244 in the unfiltered setting
policy_only no_op_accuracy improves over 0.0 only in the top-k ablation
repeated_action_rate stays at 0.0
```

---

# Active Patch Task — NGR-v1a.7 Policy-Only Phase Control

## Status

```text
PATCHED / SWEEP RUN DONE
```

## What changed

```text
policy_only rollout phase diagnostics
policy_only soft phase prior sweep
imperfect recovery rows in progress-state training
phase-head retrain on the noisier recovery set
```

## Current best unfiltered point

```text
policy_only phase_prior_lambda = 0.50
policy_only compat_weight      = 0.50
commit_f1            = 0.4244
session_edge_f1      = 0.0203
memory_attachment_f1 = 0.2960
no_op_accuracy       = 0.0
repeated_action_rate = 0.0
```

## Main diagnostic finding

```text
compatibility scoring helps policy_only broadly
top-k filtering fixes noop better but currently over-prunes link recovery
```

## Immediate next patch

```text
1. preserve top-k/noop gains without zeroing edge completion
2. reduce over-pruning of link candidates under policy_only top-k
3. keep policy_only phase prior + compatibility as fair self-guidance only
4. rerun the same policy_only sweep before changing guided metrics
```

---

# Active Patch Task - NGR-v1a.9 Protected Top-K Link Recovery

## Status

```text
PATCHED / EVAL-ONLY ABLATION RUN DONE
```

## What changed

```text
policy_only protected-link phase preservation
policy_only link_pair_k widened to 64
policy_only link candidate relations widened to all relations
guided_exact_progress and phase_guided left unchanged
```

## Rerun setup

```text
checkpoint  = out_ngr_v1a8_policy_phase_smoke_20260509/best_ngr_v1a.pt
phase_prior = 0.50
compat      = 0.50
topk        = 2 and 3
```

## Observed result

```text
topk=2:
  commit_f1            = 0.5524
  session_edge_f1      = 0.0000
  memory_attachment_f1 = 0.2179
  no_op_accuracy       = 0.8
  repeated_action_rate = 0.0

topk=3:
  commit_f1            = 0.5606
  session_edge_f1      = 0.0052
  memory_attachment_f1 = 0.2483
  no_op_accuracy       = 0.6667
  repeated_action_rate = 0.0
```

## Main diagnostic finding

```text
The protected-link decoder ablation preserved no-op gains but did not recover
edge completion.

This is not just full link-family pruning anymore:
  topk=2 still emitted LINK_SESSION_NODES 27 times
  topk=3 still emitted LINK_SESSION_NODES 49 times

So the next bottleneck is more likely link tuple scoring/ranking after
candidates survive filtering.
```

## Immediate next patch

```text
1. keep the current three-mode metric split unchanged
2. keep policy_only repeated_action_rate at 0.0
3. target link tuple ranking / relation-pair scoring under policy_only
4. avoid retraining or retrieval changes until this eval-side diagnosis is exhausted
```

---

# Active Architecture Task - Traversal + Evidence Compiler Integration

## Status

```text
PLANNED / STAGED ADOPTION ONLY
```

## Decision

```text
Do not replace the v1a edit program with free-form traversal actions.

Adopt the new design in layers:
  traversal = bounded evidence and candidate builder
  discrete edit grammar = final mutation mechanism
  graph context compiler = v2/v3 reasoning bridge
```

## Why not rewrite v1a now

```text
Current failure is still policy_only link tuple ranking.

A full traversal-first rewrite would make it harder to tell whether failure
comes from:
  routing
  evidence selection
  tuple ranking
  relation choice
```

## Staged plan

```text
v1a.10:
  eval-only link-rank probe

v1a.11:
  traversal-assisted candidate narrowing for edit tasks
  keep add/link/attach/noop/stop as the commit grammar

v2:
  graph_context_compiler.py for compact evidence packets
  use for edit explanations and conflict review

v3:
  traversal + evidence packet + validator for answer/verification mode
  keep permanent graph mutation behind a separate commit gate
```

## Non-negotiable boundaries

```text
Traversal may:
  collect evidence paths
  rank frontiers
  narrow candidate edit sites
  surface conflicts

Traversal may not replace:
  typed graph edit actions
  relation-labeled edge commits
  no_op/stop grammar
  commit validation
```

## Immediate next concrete work

```text
1. use v1a.10 link-rank results before deciding whether v1a.11 needs traversal-assisted narrowing
2. do not move v1b ahead of unresolved v1a policy_only link control
3. keep NEW DESIGN PROPOSOL.md as the v2/v3 architecture reference
4. keep traversal as an upstream evidence builder, not a replacement for typed edit commits
```

---

# Active Patch Task - NGR-v1a.10 Link-Rank Probe

## Status

```text
PATCHED / DIAGNOSTIC RUN DONE
```

## What changed

```text
eval_ngr_v1.py now supports --link-rank-probe

For each rollout step where gold_phase_if_available == link:
  build all valid LINK_SESSION_NODES candidates
  map runtime session nodes to gold nodes
  identify missing gold link tuples
  score all link candidates with the current tuple scorer
  record rank and error-reason diagnostics
```

## Main result

```text
This is not a gold-candidate absence problem.

Observed:
  gold_link_present_rate = 1.0
  gold_absent_count = 0
  no_gold_node_match_count = 0

So the dominant failure is ranking, not candidate generation.
```

## Important split

```text
policy_only topk=2:
  link_probe_steps = 0

This means the current top-k/noop-friendly policy_only setting often never
reaches a gold link phase on this smoke slice.
```

## Strongest policy-only control for diagnosis

```text
policy_only, phase_topk = 0:
  commit_f1               = 0.4244
  session_edge_f1         = 0.0203
  link_probe_steps        = 14
  link_gold_top1_rate     = 0.1429
  link_pair_top1_rate     = 0.2143
  chosen_link_gold_rate   = 0.1429
  wrong_pair_count        = 7
  wrong_direction_count   = 3
  wrong_relation_count    = 2
```

## Immediate next patch

```text
1. target wrong-pair link ranking first
2. treat wrong-direction ranking as the secondary link fix
3. only address relation ranking after pair ranking improves
4. keep guided metrics frozen and keep v1b deferred
```

---

# Active Data Audit - V1a Progress Data

## Status

```text
PATCHED / SMOKE RECHECK DONE
```

## Cleared checks

```text
1. progress train/val exact overlap = 0
2. goal train/val exact overlap = 0
3. same state never maps to conflicting phases
4. same state never maps to conflicting allowed_next tuples
5. raw index and memory-visibility invariants passed
```

## Structural risks

```text
1. stop rows still have no alternatives by construction
2. full retrain/eval on the patched data has not been run yet
```

## Practical meaning

```text
The broken-label poisoning hypothesis did not hold.

The original skew findings were real, and the new smoke data patch fixed the
largest ones:
  graph-held-out validation now exists
  mixed_add_link now has link rows
  multi_region_attach now has link rows
  add/noop ranking pressure is materially stronger
```

## Data-side follow-ups

```text
1. keep graph-held-out validation as the default path going forward
2. preserve the stronger add/noop negatives and broader link supervision
3. decide later whether stop-phase distractors are worth adding
4. move the next patch to decoder/phase balance, not back to raw data integrity
```

## Smoke retrain result

```text
patched-data smoke checkpoint:
  out_ngr_v1a10_datafix_smoke_20260510/best_ngr_v1a.pt

unfiltered policy_only:
  commit_f1            = 0.3916
  session_edge_f1      = 0.1702
  memory_attachment_f1 = 0.1604
  no_op_accuracy       = 0.0

policy_only topk=2 after structural-phase rescue:
  commit_f1            = 0.4488
  session_edge_f1      = 0.2264
  memory_attachment_f1 = 0.1755
  no_op_accuracy       = 0.0
```

## What this means

```text
The data patch helped the real autonomous link problem.

It did not solve phase/decoder balance:
  top-k policy_only starvation is fixed
  but covered/no-op completion is broken again
  phase_guided is still unstable on the new held-out split
```

## Immediate next patch

```text
1. recover covered/no-op completion on top of the structural-phase rescue
2. preserve the new top-k link/attach gains
3. use the link-rank probe to keep targeting wrong-pair and wrong-direction errors
4. do not revert the held-out data patch
```

## No-op fallback fix

```text
Applied in eval_ngr_v1.py.

Result:
  policy_only metrics stayed essentially unchanged
  phase_guided no_op_accuracy dropped from 0.6 to 0.0

Meaning:
  the old phase_guided no-op win was partly a decoder fallback artifact
  not a trustworthy learned capability
```

## Honest no-fallback eval

```text
Applied in eval_ngr_v1.py.

All decoder fallback behavior is now removed.
When tuple enumeration fails, rollout emits __NO_VALID_TUPLE__ instead of
forcing a rescue action.

Held-out smoke rerun on:
  out_ngr_v1a10_datafix_smoke_20260510/best_ngr_v1a.pt

policy_only topk=2 with structural-phase rescue:
  commit_f1            = 0.4382
  session_edge_f1      = 0.2264
  memory_attachment_f1 = 0.1699
  no_op_accuracy       = 0.0
  invalid_action_rate  = 0.0739
  __NO_VALID_TUPLE__   = 40

Meaning:
  the evaluator is now exposing real no-candidate dead ends
  the next blocker is honest dead-end reduction, not fallback tuning
```

## Current next patch

```text
1. keep no-fallback rollout behavior
2. reduce __NO_VALID_TUPLE__ frequency honestly
3. improve candidate coverage / ranking without synthetic rescue
4. continue treating guided_exact_progress as upper-bound only
```

## V1a.11 dead-end audit

```text
Applied in eval_ngr_v1.py via --dead-end-probe.

Held-out policy_only topk=2 rerun:
  commit_f1            = 0.4382
  session_edge_f1      = 0.2264
  memory_attachment_f1 = 0.1699
  invalid_action_rate  = 0.0739
  __NO_VALID_TUPLE__   = 40

Dead-end reason counts:
  phase_topk_pruned_all_candidates = 40

Predicted phase on dead-end steps:
  stop = 16
  noop = 15
  link = 9

Structurally required phase on dead-end steps:
  create = 25
  cover  = 15

Meaning:
  current policy_only dead ends are not a candidate-coverage problem first
  they are a hard phase-topk pruning problem
  the next eval patch should relax or reshape top-k pruning on create/cover
  states without adding fallback rescue
```

## Create/cover top-k softening

```text
Applied in eval_ngr_v1.py via:
  --policy-only-soften-topk-on-create-cover

Held-out policy_only topk=2 rerun:
  commit_f1            = 0.3912
  session_edge_f1      = 0.2011
  memory_attachment_f1 = 0.1680
  invalid_action_rate  = 0.0
  repeated_action_rate = 0.0
  __NO_VALID_TUPLE__   = 0

Meaning:
  hard top-k pruning dead ends are fully removed
  but the next problem is now ranking / action-family choice after candidates
  are admitted
  this should remain an eval ablation, not the new default metric yet
```

## V1a.11 link-pair auxiliary loss

```text
Applied in train_ngr_v1.py.

Changes:
  direct link_pair_aux_loss on link-phase rows
  reverse-direction margin penalty
  validation metric: link_pair_candidate_hit

Smoke retrain:
  out_ngr_v1a11_linkpair_smoke_20260510/best_ngr_v1a.pt

Validation:
  link_pair_candidate_hit = 0.8716
  phase_accuracy          = 0.6222

Policy-only rollout:
  unfiltered:
    commit_f1            = 0.3798
    session_edge_f1      = 0.1337
    memory_attachment_f1 = 0.1361
    no_op_accuracy       = 0.0
    link_pair_top1_rate  = 0.2745
    link_gold_top1_rate  = 0.2353

  topk=2 + create/cover softening:
    commit_f1            = 0.3854
    session_edge_f1      = 0.1657
    memory_attachment_f1 = 0.1386
    no_op_accuracy       = 0.0
    invalid_action_rate  = 0.0
    link_pair_top1_rate  = 0.4082
    link_gold_top1_rate  = 0.3061

Meaning:
  the pair head improved
  policy_only rollout still does not convert that into a larger commit win
  next blocker is post-admission action-family choice and covered/no-op
```

## V1a.12 phase transition / terminal-control patch

```text
Applied in:
  eval_ngr_v1.py
  train_ngr_v1.py

Added rollout diagnostics:
  phase_compatible_rate
  premature_stop_count
  attach_before_edge_complete_count
  noop_available_not_chosen_count

Added training control losses:
  weighted action-family CE by phase
  link > attach margin
  attach > link margin
  noop > stop margin
  non-stop gold-action > stop margin

Smoke retrain:
  out_ngr_v1a12_control_smoke_20260510/best_ngr_v1a.pt

Validation:
  tuple_candidate_hit      = 0.8061
  link_pair_candidate_hit  = 0.9189
  phase_accuracy           = 0.7414

Held-out policy_only topk=2:
  commit_f1                   = 0.2867
  session_edge_f1             = 0.1692
  memory_attachment_f1        = 0.0652
  no_op_accuracy              = 0.0
  phase_compatible_rate       = 0.3922
  premature_stop_count        = 36
  attach_before_edge_complete = 117
  noop_available_not_chosen   = 0

Meaning:
  this first control-loss formulation regressed rollout
  attach-before-link is now the clearest explicit control error
  noop is still not reached often enough to become an available action
```

## V1a.12b attach rebalance

```text
Applied in train_ngr_v1.py.

Changes:
  stronger link phase weight
  reduced attach phase weight
  added non-attach > attach margin

Smoke retrain:
  out_ngr_v1a12b_attach_smoke_20260510/best_ngr_v1a.pt

Held-out policy_only topk=2:
  commit_f1                   = 0.3195
  session_edge_f1             = 0.1761
  memory_attachment_f1        = 0.1119
  no_op_accuracy              = 0.0
  phase_compatible_rate       = 0.2914
  premature_stop_count        = 41
  attach_before_edge_complete = 92

Meaning:
  partial recovery from the first v1a.12 regression
  attach drift reduced, but terminal control is still weak
  next blocker remains premature stop + weak covered/no-op reachability
```

## V1a.13 covered/no-op recovery patch

```text
Applied in:
  ngr_v1_progress_tasks.py
  train_ngr_v1.py
  eval_ngr_v1.py

Artifacts:
  artifacts/tasks_v1a13_progress_smoke_20260510
  out_ngr_v1a13_covered_smoke_20260510/best_ngr_v1a.pt
  out_ngr_v1a13_covered_smoke_20260510/eval_summary.json
```

## What was changed

```text
1. covered_long_signal recovery rows split into:
   cover_incomplete
   cover_complete_no_noop
   false_terminal_drift

2. covered rows removed from the broad global action-family control loss

3. new covered-specific control loss:
   cover > create/link/stop
   noop > create/link/mark_covered/stop
   covered premature-stop penalty

4. policy_only eval now logs covered-specific rollout metrics:
   covered_reaches_cover_complete_rate
   covered_reaches_noop_available_rate
   covered_noop_chosen_when_available_rate
   covered_premature_stop_count
   covered_create_after_all_nodes_present_count
   covered_link_on_noop_goal_count

5. policy_only covered-task admission preserves cover/noop structurally
```

## Result

```text
Smoke train:
  tuple_candidate_hit      = 0.8081
  link_pair_candidate_hit  = 0.9904
  phase_accuracy           = 0.7460

policy_only topk=2:
  commit_f1                     = 0.3752
  session_edge_f1               = 0.1797
  memory_attachment_f1          = 0.1261
  no_op_accuracy                = 0.0
  invalid_action_rate           = 0.0199
  repeated_action_rate          = 0.0127
  phase_compatible_rate         = 0.3388
  premature_stop_count          = 41
  attach_before_edge_complete   = 97
```

## Covered-task verdict

```text
covered_long_signal still fails honestly:
  commit_f1                           = 0.0
  no_op_accuracy                      = 0.0
  covered_reaches_cover_complete      = 0.0
  covered_reaches_noop_available      = 0.0
  covered_noop_chosen_when_available  = 0.0
  covered_premature_stop_count        = 10
  covered_link_on_noop_goal_count     = 31

Dead-end probe on covered rows:
  reason = phase_topk_pruned_all_candidates
  predicted phases = stop/noop
  structural phase = cover
```

## Current priority

```text
The no-op bottleneck is now clearly upstream:
  policy_only does not reach cover-complete states
  no-op is not being rejected after availability
  covered rows still drift into stop/noop predictions and link actions

Next patch should target covered-phase progression itself:
  create -> cover reachability
  cover candidate admission/ranking under policy_only
  stop/noop suppression specifically on uncovered covered-task states

Do not:
  change guided-mode meaning
  add retrieval
  move to v1b
```

## Training-side covered/no-op metric check

```text
Added explicit validation metrics in train_ngr_v1.py:
  covered_cover_action_accuracy
  covered_noop_action_accuracy
  covered_stop_action_accuracy
  covered_cover_phase_accuracy
  covered_noop_phase_accuracy
  covered_stop_phase_accuracy

Smoke validation on:
  out_ngr_v1a13_covered_smoke_20260510_metrics/best_ngr_v1a.pt

epoch 2:
  covered_cover_action_accuracy = 0.8875
  covered_noop_action_accuracy  = 1.0
  covered_stop_action_accuracy  = 1.0
  covered_cover_phase_accuracy  = 0.5
  covered_noop_phase_accuracy   = 1.0
  covered_stop_phase_accuracy   = 1.0
```

## Meaning

```text
The model does know how to choose PROPOSE_NO_OP on clean noop rows.
The honest rollout failure is upstream:
  covered rows are not reaching cover-complete / noop-available states
  covered cover-phase control is still weak
```

## Covered progression patch

```text
Applied in:
  ngr_v1_progress_tasks.py
  train_ngr_v1.py
  eval_ngr_v1.py

Artifacts:
  artifacts/tasks_v1a13b_progress_smoke_20260510
  out_ngr_v1a13b_coverprogress_smoke_20260510/best_ngr_v1a.pt
  out_ngr_v1a13b_coverprogress_smoke_20260510/eval_summary.json
```

## Result

```text
Smoke val:
  covered_create_action_accuracy = 0.6667
  covered_cover_action_accuracy  = 1.0
  covered_noop_action_accuracy   = 1.0
  covered_cover_phase_accuracy   = 0.35

policy_only topk=2:
  commit_f1                     = 0.3501
  session_edge_f1               = 0.1365
  memory_attachment_f1          = 0.1142
  no_op_accuracy                = 0.0
  phase_compatible_rate         = 0.4482
```

## New diagnosis

```text
This patch removed covered-task link drift:
  covered_link_on_noop_goal_count = 0
  covered_premature_stop_count    = 0

But covered rollout still does not reach no-op:
  covered_reaches_cover_complete_rate = 0.0
  covered_reaches_noop_available_rate = 0.0

The new bottleneck is beam omission of MARK_COVERED on covered rollout states.

Probe evidence:
  regular no-topk candidate families on covered dead ends:
    LINK_SESSION_NODES
    CREATE_SESSION_NODE
    STOP

  exhaustive candidate families on the same dead ends:
    LINK_SESSION_NODES
    CREATE_SESSION_NODE
    MARK_COVERED
    STOP
```

## Current priority

```text
Next patch should be eval-only first:
  widen MARK_COVERED candidate coverage for covered tasks
  or add a covered-specific exhaustive mark_covered probe path

Do not:
  change guided-mode meaning
  add retrieval
  move to v1b
```

## Covered env/eval recovery fixes

```text
Applied:
  ngr_v1_env.py
  eval_ngr_v1.py

Changes:
  1. MARK_COVERED reassignment is now allowed unless the target memory is the same
  2. policy_only covered pair beam widened with --policy-only-cover-pair-k=256
  3. covered evidence gate added for MARK_COVERED candidates
```

## Result

```text
Same checkpoint:
  out_ngr_v1a13b_coverprogress_smoke_20260510/best_ngr_v1a.pt

policy_only topk=2:
  commit_f1            = 0.3501
  no_op_accuracy       = 0.0
  invalid_action_rate  = 0.0107

covered_long_signal:
  commit_f1                           = 0.0
  no_op_accuracy                      = 0.0
  covered_reaches_cover_complete      = 0.0
  covered_reaches_noop_available      = 0.0
```

## Meaning

```text
These fixes were real and useful:
  MARK_COVERED count increased
  covered dead ends dropped sharply
  covered link drift stayed at zero

But they did not solve the honest no-op failure.

Current blocker:
  the policy emits many MARK_COVERED actions,
  but they still do not assemble into coverage_complete under runtime mapping
```

## Current priority

```text
Next patch should target covered-state correctness itself:
  inspect runtime_pred_to_gold behavior on covered rollouts
  inspect whether repeated MARK_COVERED is hitting wrong session-to-memory alignment
  add a covered-rank probe analogous to the link-rank probe

Do not:
  change guided-mode meaning
  add retrieval
  move to v1b
```

## Post-mask-fix check

```text
After fixing ngr_v1_env.py::_has_uncovered_session():
  covered dead_end steps = 0
  covered MARK_COVERED count increased to 67

But:
  covered_reaches_cover_complete_rate = 0.0
  covered_reaches_noop_available_rate = 0.0
  covered_no_op_accuracy = 0.0
```

## Meaning

```text
The action mask bug was a real blocker and is now removed.
The remaining failure is deeper:
  policy emits many MARK_COVERED actions
  but they are still not the right covered assignments under runtime mapping
```

## Fresh smoke verification

```text
Fresh run on current code:
  artifacts/tasks_v1a13c_progress_smoke_20260510
  out_ngr_v1a13c_maskfix_smoke_20260510/best_ngr_v1a.pt

policy_only topk=2:
  commit_f1            = 0.3206
  session_edge_f1      = 0.1011
  memory_attachment_f1 = 0.1013
  no_op_accuracy       = 0.0
  invalid_action_rate  = 0.0

covered_long_signal:
  covered_reaches_cover_complete_rate = 0.0
  covered_reaches_noop_available_rate = 0.0
  covered_no_op_accuracy              = 0.0
  dead_end steps                      = 0
```

## Verified status

```text
The covered deadlock is fixed.
The covered rollout objective is still failing.

So the current next patch should not chase masking/gating anymore.
It should directly probe covered assignment correctness under runtime mapping.
```

## Manual covered rollout dump

```text
Manual case:
  artifacts/manual_eval_case_covered_20260510.jsonl

Latest checkpoint:
  out_ngr_v1a13c_maskfix_smoke_20260510/best_ngr_v1a.pt

Observed rollout:
  4x CREATE_SESSION_NODE
  8x MARK_COVERED
  0x PROPOSE_NO_OP

This was a 2-target covered task.
```

## Meaning

```text
The failure starts before no-op.
The model is creating semantically bad session nodes for covered tasks,
then repeatedly marking those misaligned nodes as covered.

So the next patch should add a covered-create / covered-alignment probe:
  inspect runtime_pred_to_gold during covered create
  measure whether created session nodes match the intended covered spans
  measure whether MARK_COVERED is being applied to the right runtime node
```

---

## Traversal-only ablation status

Added:

```text
traverse_threshold_eval.py
```

Role:

```text
diagnostic traversal-only path
no session graph
no typed edit actions
thresholded adjacency expansion only
```

Current result:

```text
On a 20-row held-out slice,
target_memory_recall = 1.0 on all rows that have explicit goal memory targets,
because the signal already names those target memories strongly enough to anchor them.

Raising traverse_threshold from 0.18 to 0.55 reduces avg_visited_nodes:
  58.7 -> 40.45
but does not change target-memory hit rate.
```

Interpretation:

```text
Traversal-only is useful as an evidence/candidate narrowing primitive.
It is not a replacement for v1a edit-program learning.

It does not address:
  create-node quality
  session-edge control
  covered assignment correctness
  no-op completion
```

Decision:

```text
Keep traversal-only as a side ablation / future upstream module.
Do not replace the active v1a controller with traversal-only.
```

Current next target remains:

```text
covered-create / covered-alignment probe under the edit-program path
```

---

## Traversal-first draft-edit status

Added:

```text
traverse_threshold_draft_edit.py
```

Role:

```text
safe traversal-first prototype
edit while traversing
all edits remain temporary
no real graph mutation
```

Current behavior:

```text
Traversal can create draft session nodes from matched spans,
link draft nodes when traversed neighbors support it,
and add temporary covered / attach edits while walking.
```

Current result:

```text
manual covered case:
  recovered both covered targets in draft form
  but also produced extra junk draft nodes

4-row smoke:
  cover_hit_rate      = 0.3333
  attachment_hit_rate = 0.6667
  avg_draft_nodes     = 5.5
```

Interpretation:

```text
This path proves "edit on the way" is implementable safely if edits stay draft-only.
But the current lexical create heuristic is too loose and overproduces structure.
```

If we continue this ablation, the next patch should be:

```text
traversal-first draft-node quality control:
  prefer clause spans over merged/full spans
  tighten bridge_merge threshold
  prune low-quality draft nodes before cover/link propagation
```

This still does not replace the main v1a controller.

## Draft-edit tightening status

Implemented:

```text
clause/item preference
full/merged penalties
high-quality source requirement for bridge_merge
```

Current result:

```text
manual covered case improved sharply:
  draft_nodes 4 -> 2
  draft_edges 3 -> 0
  still hits both covered targets

4-row smoke improved modestly:
  avg_draft_nodes 5.5 -> 4.75
  avg_draft_edges 5.75 -> 3.75
```

Interpretation:

```text
This path is now much cleaner on the exact covered debug case.
It is still noisy on broader tasks and still mainly lexical.
```

If continuing traversal-first draft edits, the next patch should be:

```text
add local pruning of weak draft nodes and weak draft edges before later edits use them
```

---

## Main-path sequencing decision

Do this before any traversal-assisted candidate narrowing touches the active `v1a` controller:

```text
fix covered phase supervision in ngr_v1_progress_tasks.py
```

Implemented now:

```text
1. cover-phase history includes MARK_COVERED
2. covered create-phase no longer always starts from scratch
3. covered cover-phase no longer always has zero covered nodes
```

Verified on regenerated smoke progress data:

```text
covered create rows now include partial-created states
covered cover rows now include 0/1/2 covered partial states
cover action history now includes MARK_COVERED when cover_count > 0
```

Why this order matters:

```text
If traversal narrows candidates first,
phase-head errors become less visible and the real supervision bug gets masked.

So the order should be:
  1. fix covered phase data
  2. retrain / re-eval the active v1a path honestly
  3. only then decide whether traversal-assisted narrowing adds value
```

Current next main-path step:

```text
train + eval on the phase-fixed progress data
```

## Result of phase-fixed retrain

Completed:

```text
artifacts/tasks_v1a13d_progress_phasefix_smoke_20260511
out_ngr_v1a13d_phasefix_smoke_20260511/best_ngr_v1a.pt
out_ngr_v1a13d_phasefix_smoke_20260511/eval_summary.json
out_ngr_v1a13d_phasefix_smoke_20260511/rollouts.jsonl
```

Observed:

```text
train val:
  phase_accuracy           = 0.7596
  covered_cover_phase_acc  = 0.525
  covered_noop_action_acc  = 1.0

policy_only topk=2:
  commit_f1            = 0.3123
  session_edge_f1      = 0.1820
  memory_attachment_f1 = 0.1108
  no_op_accuracy       = 0.0
  invalid_action_rate  = 0.0
  repeated_action_rate = 0.0305
```

Covered task specifically:

```text
covered_long_signal:
  commit_f1                           = 0.0
  no_op_accuracy                      = 0.0
  covered_reaches_cover_complete_rate = 0.0
  covered_reaches_noop_available_rate = 0.0
```

Interpretation:

```text
The progress-data fix was necessary and did reduce the old supervision defect.
But it did not solve covered completion under rollout.

The remaining active bug is now narrower:
  covered session-node / covered-assignment alignment under runtime mapping
```

Current next main-path target:

```text
covered-create / covered-alignment probe in the active v1a rollout path
```

Meaning:

```text
Inspect whether CREATE_SESSION_NODE on covered tasks is still producing the wrong runtime nodes,
and whether MARK_COVERED is then being applied to the wrong runtime node even when the predicted phase is reasonable.
```
## 2026-05-11 Active Switch

The official active controller is now:

```text
traverse_threshold_draft_edit.py
```

The old `NGR-v1a` action-policy path remains in the repo for diagnostics and
historical comparison only. It is no longer the official architecture.

Current traversal-first contract:

```text
signal + graph
-> anchor scoring
-> thresholded traversal
-> temporary draft session nodes / edges / covered mappings / attachments
-> goal-matched completion scoring
```

Current held-out evidence:

```text
manual covered case:
  session_node_recall = 1.0
  covered_recall = 1.0
  covered_complete_rate = 1.0
  task_complete_proxy_rate = 1.0

held-out 20-row smoke:
  task_complete_proxy_rate = 0.15
  session_node_recall = 0.925
  session_edge_recall = 0.09375
  attachment_recall = 0.2
  covered_recall = 0.9167
  covered_complete_rate = 0.75
```

Interpretation:

```text
Traversal-first draft editing is already much stronger on covered tasks than the
archived NGR-v1a path.

The next bottleneck is draft-node pruning plus better session-edge and attachment
completion on mixed edit tasks.
```

New active gate table:

```text
Gate       Name                                         Status
N0         Repo reset + plan pivot                      DONE
NGR-v1a    Session-graph action policy                  ARCHIVED / DIAGNOSTIC ONLY
TRV-v0     Traversal-only evidence collector            DONE
TRV-v0.1   Draft-edit traversal prototype               DONE
TRV-v0.2   Span-quality tightening                      DONE
TRV-v0.3   Goal-matched draft metrics                   DONE
TRV-v0.4   Held-out traversal smoke                     ACTIVE
TRV-v1     Prune weak draft structure before reuse      NEXT
TRV-v2     Commit-capable traversal controller          PLANNED
TRV-v3     Traversal + evidence packet + validator      PLANNED
```

---
