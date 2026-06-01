# READ_THIS — V5 latest raw results & quick reference

> At-a-glance dump of the latest runs (raw outputs, numbers, repro commands) so
> you don't have to dig through commits/logs. Updated each working session.

**Last updated:** 2026-06-02
**HEAD:** latest pushed commit on branch `main`

---

## Session update (2026-06-01)

- Corpus scale is now 288 unique traces after merging the local + vast.ai
  shards.
- We found a teacher mismatch: raw V4 rows over-expose the tool-call path,
  while V5 wants `candidate subgraph -> planning -> evidence -> answer`.
- Added `project_corpus_to_v5_targets.py` + `v5.training.projection` to attach
  V5-native targets: `candidate_node_ids`, soft planning / evidence / support /
  distractor maps, and per-loop targets.
- `dataset.py` + `bridge.py` now prefer projected targets over raw anchor-only
  masks, and the held-out eval paths were updated to treat soft labels as
  positive when `> 0`.
- Real projected-corpus stats on the merged 288 traces:
  - planning rows 189
  - evidence rows 288
  - support rows 253
  - loop-supervised rows 288
  - mean candidate nodes 9.2
- After substrate + bridge on the projected corpus: +685 substrate nodes,
  +1151 relations, 288 examples, planning labels on 239/288 (83%), evidence
  labels on 288/288, average subgraph size 29.38 nodes.
- First held-out pilot completed end-to-end on the projected corpus
  (`234 train / 59 held-out`, Qwen2.5-0.5B-Instruct, `e1=30 e2a=20 e2b=20`):
  - plan P@1 0.28 / recall 0.25
  - evidence P@1 0.88 / recall 0.54
  - slot 0.66, epi all-node 0.30, shortcut 0.62, inv 0.00
  - epi per-node 0.96
  - fallback applicable=1.00, blocked=1.00, negative=1.00
- Added `v5.training.fallback_write_diag` and found a slot-alias mismatch in
  `task_frame.required_slots`: fallback was often checking the shared `unknown`
  slot even though slot targets were trained on canonical names. Canonicalizing
  task-frame slots moved projected-corpus fallback to applicable=0.83,
  blocked=0.82, negative=1.00 on the same `30/20/20` diagnostic schedule.
- Remaining failure after the alias fix: applicable still trips mostly on
  missing slots (30/47), low epistemic (19/47), and top-k invalidators (12/47).
  Gold-all oracle only drops applicable fallback to 0.64, so this is not one
  isolated head.
- Write remains suspicious: negative held-out total write is highest at 0.224
  and planning write is especially high (0.449). Treat this as a write-safety
  diagnostic target before Stage 3/4.
- Added differentiable write-ratio tensors plus a Stage 2B negative penalty:
  no-graph negatives now train for low write, low slots, low epistemic, and low
  shortcut. Expanded no-graph negatives from 5 to 15 prompts so held-out negative
  metrics are less single-example fragile.
- Latest `fallback_write_diag` with that penalty (`30/20/20`, 288 projected rows):
  applicable fallback=0.83, blocked=0.73, negative=1.00; negative write is now
  the lowest bucket again at total=0.107, though still above the ideal 0.00-0.05
  target. Main remaining issue is relational_explanation invalidator over-firing.
- Added the invalidator semantics pass:
  `invalidator_candidate_nodes()` now only marks well-formed active-subgraph
  negative edges, ignores self-invalidating graph noise, and the bridge now
  trains inactive structural invalidators as zero instead of only supervising
  positive deprecate labels. `fallback_write_diag` also dumps edge context for
  manual invalidator inspection.
- Latest comparable `fallback_write_diag` (`30/20/20`, 288 projected rows):
  applicable fallback=0.74, blocked=1.00, negative=1.00. `invalidator_active`
  is now 0.00 for applicable/blocked/negative, and `relational_explanation`
  moved from invalidator_active=0.93 to 0.00. Negative total write remains the
  lowest bucket at 0.150, but negative evidence-write is still worth watching.
- Added an applicable-only calibration report plus deterministic `--seed`
  support to `fallback_write_diag`. Seeded `30/20/20` read (`--seed 7`):
  applicable fallback=0.64, blocked=0.91, negative=0.67. Applicable failures
  are now mostly `missing_slot+low_epistemic` (24/47), with failed slots dominated
  by `reason` (25) and `verdict` (16). `relational_explanation` applicable is
  clean (`fallback=0.00`); direct_judgment remains the hard family.
- Calibration read: the gate is not mainly too strict by threshold margin.
  Failed slot margins are usually very negative, and low epistemic often occurs
  on gold evidence (`low_epi_on_gold_evidence=18`). So the next fix is label /
  head calibration for direct_judgment reason/verdict and support epistemic,
  not another invalidator or threshold-only tweak.
- Safety note: seeded negative fallback slipped to 2/3 held-out negatives (n=3).
  Keep the negative write/fallback guard; do not move Stage 3/4 until negative
  fallback is consistently high again.
- Added the direct_judgment calibration pass to `v5.training.fallback_write_diag`:
  held-out direct_judgment failure table, per-case question/required slots/slot
  scores/primary evidence/best gold evidence/planning hit, plus a negative
  no-graph safety table that dumps false `shortcut_verified` cases.
- Found and fixed another label/contract mismatch: finalized rows sometimes had
  `required_slots=['verdict','reason']` while the gold slot target still had one
  of those slots at 0. For finalized traces, `dataset.py` now marks all
  canonical required slots as filled and mirrors them into `task_frame.filled_slots`.
  Regression coverage was added in `reasoning/tests/test_v5_projection.py`.
- Post-fix seeded diagnostic
  (`--seed 7 --e1 30 --e2a 20 --e2b 20`, log:
  `artifacts/run_logs/fallback_write_diag_20260601_finalized_slot_fix_seed7.log`):
  applicable fallback=0.51, blocked=0.73, negative=0.67. Applicable missing-slot
  failures dropped to 9/47; remaining applicable failures are mostly
  low_epistemic 23/47. Direct_judgment applicable is now slot=0.23 but epi=0.63;
  relational_explanation applicable remains clean at fallback=0.00.
- Negative safety is still not solved: one held-out no-graph case exits
  `shortcut_verified` on an irrelevant but plausible chemistry node
  (`bond_polarity_depends_on_electronegativity_difference`) with slots near 1.0,
  epi=1.0, shortcut=0.998. This is now the clearest safety blocker before any
  Stage 3/4 move: protect negatives/shortcut verification before lowering
  thresholds or chasing answer quality.
- Current read after the direct_judgment pass: slot labels are much healthier;
  the remaining direct_judgment failures are primarily evidence/planning
  selection and epistemic calibration, not raw slot aliasing. Oracle gold-all
  only drops applicable fallback to 0.40 and leaves negative at 0.67, so do not
  treat this as solved by per-slot threshold tuning alone.
- Added the negative shortcut safety pass. No-graph negatives now carry explicit
  question/case ids, zero slot/epi/shortcut targets, and a task-frame contract:
  `graph_context=no_graph`, `allow_shortcut_exit=False`, `force_fallback=True`.
  `should_exit_loop()` refuses early graph-certification exits for forced-fallback
  contexts, and `fallback_needed()` now receives the injector task frame.
- Latest seeded diagnostic after the shortcut guard
  (`--seed 7 --e1 30 --e2a 20 --e2b 20`, log:
  `artifacts/run_logs/fallback_write_diag_20260601_negative_shortcut_guard_seed7.log`):
  applicable fallback=0.53, blocked=0.82, negative=1.00. Negative exits are now
  all `max_loops_reached`; no held-out no-graph case exits `shortcut_verified`.
  Oracle gold-all is applicable=0.43, blocked=0.82, negative=1.00.
- Negative audit now prints graph context, active node type counts, substrate
  counts, top evidence nodes, question, slots, epi, shortcut, and write ratios.
  The audit still shows irrelevant active substrate around negatives
  (`strategy`, `solved_subgoal`, `reasoning_atom`), but the gate no longer lets
  that substrate certify a no-graph answer. Negative total write is 0.132, lower
  than applicable no-fallback 0.164 but still above the ideal early 0.03-0.07 band.
- Current read after the negative shortcut safety pass: safety is repaired enough
  that negatives no longer leak through shortcut, but Stage 3/4 still stay held.
  The main remaining blocker is direct_judgment evidence/planning + epistemic
  calibration: direct_judgment applicable fallback is still 23/35 (0.66), mostly
  low epistemic / wrong primary evidence.
- Added an applicable failure focus report to `fallback_write_diag`, splitting
  held-out applicable failures into routing vs confidence buckets. It prints
  predicted/gold plan nodes, predicted/gold evidence nodes, slot scores,
  epistemic on predicted vs best-gold evidence, shortcut, and write ratio.
- Latest seeded focused diagnostic
  (`--seed 7 --e1 30 --e2a 20 --e2b 20`, log:
  `artifacts/run_logs/fallback_write_diag_20260601_applicable_focus_seed7.log`):
  applicable fallback=0.47, blocked=0.73, negative=1.00. Negative still fails
  closed; no no-graph `shortcut_verified` leak returned. Oracle gold-all drops
  applicable only to 0.36, so the remaining gap is not one head/threshold.
- Focused read: applicable failures are now almost entirely direct_judgment:
  21/22 applicable failures are direct_judgment, plus 1 design_synthesis.
  In failed direct_judgment, slot failure is 0.38, low_epistemic is 0.90,
  plan@1 is 0.10, plan@3 is 0.48, wrong_evidence_selected is 0.33,
  right_evidence_low_epi is 0.43, and gold_evidence_low_epi is 0.14.
  Translation: the next blocker is evidence/planning routing plus epistemic
  confidence on selected evidence; slot is now secondary.
- Safety caveat: negative total write is 0.154 in this run, which is still above
  the early ideal. The forced fallback guard keeps no-graph cases safe, but write
  pressure should remain monitored before Stage 3/4.
- Added a direct_judgment routing audit to `fallback_write_diag`. It reports
  plan/evidence P@1, hit@3, R@3, R@gold, pool coverage, raw/predicted node-type
  buckets, and per-case top-3 plan/evidence vs gold anchors with `pool=0/1`.
- Latest pool-aware seeded diagnostic
  (`--seed 7 --e1 30 --e2a 20 --e2b 20`, log:
  `artifacts/run_logs/fallback_write_diag_20260601_direct_routing_pool_seed7.log`):
  applicable fallback=0.49, blocked=0.73, negative=1.00. Oracle gold-all is
  applicable=0.36, blocked=0.73, negative=1.00. Negative total write is 0.128.
- Direct_judgment routing audit read:
  applicable direct_judgment has plan P@1=0.17, plan hit@3=0.46, plan R@gold=0.20;
  evidence P@1=0.80, evidence hit@3=0.97, evidence R@gold=0.61. For applicable
  fallback direct_judgment, plan P@1=0.14, plan hit@3=0.52, evidence P@1=0.67,
  evidence hit@3=0.95. So evidence top-k is usually near a usable in-pool anchor,
  while planning rank remains weak.
- Projection/pool warning exposed by the audit: direct_judgment plan labels are
  fully selectable (`planCov=1.00`), but evidence labels only have ~0.69-0.70
  in-pool coverage. The out-of-pool gold evidence labels are mostly
  `epistemic_state` (20), `example` (7), plus principle/equation/summary/etc.
  This is a label-contract smell: do not treat those out-of-pool anchors as
  model routing misses until projection or evidence-pool semantics are aligned.
- Current read after the routing audit: the next blocker is direct_judgment
  projection/pool alignment plus epistemic calibration on selected evidence.
  Do not strengthen direct_judgment planning loss until the label contract is
  cleaned or explicitly scoped to in-pool anchors.
- Added the direct_judgment projection/pool alignment pass:
  `projection.py` no longer makes `add_epistemic_state` nodes direct evidence
  targets; `dataset.py` preserves epistemic substrate status instead of
  defaulting all epistemic states to planning-pool `uncertain`; `bridge.py`
  masks projected plan/evidence labels to the block's legal pool; non-finalized
  blocked rows are marked `graph_context=weak_evidence`,
  `allow_shortcut_exit=False`, `force_fallback=True`. Regression tests cover
  status preservation, pool masking, and blocked forced-fallback task frames.
- Reprojected the local 288-row corpus after the projection cleanup. Stats are
  still 288 rows, planning rows 189, evidence rows 288, support rows 253, loop
  rows 288, and mean candidate nodes now 8.5 (down from 9.2 after dropping
  direct epistemic-state evidence targets).
- Earlier guarded seeded diagnostic after pool alignment
  (`--seed 7 --e1 30 --e2a 20 --e2b 20`, log:
  `artifacts/run_logs/fallback_write_diag_20260601_pool_alignment_blocked_guard_seed7.log`):
  applicable fallback=0.51, blocked=1.00, negative=1.00. Oracle gold-all is
  applicable=0.40, blocked=1.00, negative=1.00. Negative total write is 0.129.
- Pool alignment result: direct_judgment plan/evidence `poolCov` is now 1.00
  and `gold evid pool=0` is empty. Direct_judgment applicable plan P@1/hit@3
  improved to 0.40/0.73, while evidence P@1/hit@3 is 0.60/0.83 under the cleaner
  in-pool target contract. This is a label-contract win, not a final quality
  win: applicable fallback remains too high.
- New direct_judgment bucket report shows the remaining applicable failures are
  mostly multi-condition: raw flags are wrong_evidence=11, other_low_epistemic=11,
  slot_failed=10, right_evidence_low_epi=9, wrong_plan=8. Next likely pass is
  evidence/epistemic calibration on selected in-pool evidence plus direct_judgment
  planning substrate quality, while preserving the blocked/negative forced
  fallback guard.
- Added deterministic `--seed`, adapter-init reseeding, and plan/evidence `hit@3`
  reporting to the corpus/diagnostic harnesses. Provider/GNN setup can consume
  RNG, so both `corpus_scaling` and `fallback_write_diag` now reseed immediately
  before adapter construction. Canonical cleaned-label Stage 1 -> 2A -> 2B pass
  (`--seed 7 --e1 30 --e2a 20 --e2b 20`, log:
  `artifacts/run_logs/corpus_scaling_20260601_clean_projection_reseed_seed7.log`).
  Coverage over 288 positives: plan 189/288 (66%), evidence/slot/shortcut
  288/288, epi 227/288, inv 69/288.
- Controlled held-out metrics: plan P@1=0.34, plan hit@3=0.71, plan R@gold=0.34;
  evidence P@1=0.83, evidence hit@3=0.98, evidence R@gold=0.66; slot=0.66,
  epi strict=0.28, epi per-node=0.96, shortcut=0.69, inv=0.77. Fallback remains
  applicable=0.60, blocked=1.00, negative=1.00. Write ratio is applicable=0.168,
  blocked=0.160, negative=0.104.
- Added `direct_judgment calibration v2` to `fallback_write_diag` and ran the
  canonical diagnostic (log:
  `artifacts/run_logs/fallback_write_diag_20260601_direct_cal_v2_reseed_seed7.log`).
  Applicable direct_judgment split: no-fallback n=8 vs fallback n=24. No-fallback
  has evidence P@1/hit@3=0.88/1.00, verdict/reason=1.000/1.000, primary
  epi=0.920, best-gold epi=0.970. Fallback has evidence P@1/hit@3=0.46/0.79,
  verdict/reason=0.833/0.786, primary epi=0.069, best-gold epi=0.614, and
  best-gold epi clears threshold only 0.54 of the time.
- Direct_judgment v2 read: the pool contract is clean (`poolCov=1.00`, no
  out-of-pool gold targets), and evidence top-k is still useful, but fallback
  cases are failing exit alignment: primary evidence is often wrong/low-epi, and
  verdict/reason slot scores sit just below the 0.85 gate. Strict epi remains a
  poor example-level conjunction metric (`epi exact` near zero while per-node is
  ~0.94-0.96). Next pass should improve direct_judgment slot/epistemic
  calibration and selected-evidence control before adding a learned fallback head.
- Added a narrow support-primary reranker head. It is trained from projected
  `support_target` labels as a binary in-pool evidence target, then fallback
  chooses the primary evidence node by reranking only the existing evidence top-k.
  This is not Stage 3 and not a broad learned fallback head: hard safety rules
  still force no-graph/weak-evidence fallback and invalidators/slots/epi remain
  explicit gates.
- Important implementation note: support-primary uses BCE-with-logits over
  binary targets (`target > 0`), not softmax CE over weighted projection scores.
  The first attempt let weighted labels like 25.75 enter BCE, producing negative
  Stage 1 loss and unsafe write ratios. The support head is also kept frozen
  during Stage 2B so it cannot buy exits by pushing the residual write path.
- Canonical support-primary run (`--seed 7 --e1 30 --e2a 20 --e2b 20`, logs:
  `artifacts/run_logs/corpus_scaling_20260601_support_primary_binary_seed7.log`
  and
  `artifacts/run_logs/fallback_write_diag_20260601_support_primary_binary_seed7.log`):
  coverage over 288 positives now includes support 236/288 (82%). Held-out
  corpus metrics: plan P@1/hit@3=0.34/0.63, evidence P@1/hit@3=0.83/0.98,
  support-primary P@1/hit@3=0.62/0.78, slot=0.72, epi strict=0.28,
  epi per-node=0.96, shortcut=0.74, inv=0.85. Fallback is applicable=0.49,
  blocked=1.00, negative=1.00. Write ratio is applicable=0.191, blocked=0.166,
  negative=0.130.
- Direct_judgment support-primary diagnostic: applicable fallback improved from
  0.75 to 0.44; no-fallback direct_judgment has support@1/@3=0.83/0.94,
  primary epi=0.960, verdict/reason=1.000/1.000. Remaining fallback direct
  judgment has support@1/@3=0.43/0.64, primary epi=0.144, best-gold epi=0.496,
  verdict/reason=0.870/0.796. The remaining issue is still selected-support +
  epistemic calibration, but the reranker moved the bottleneck without breaking
  blocked/negative safety.
- Added `hard vs soft fallback calibration` to `fallback_write_diag` (log:
  `artifacts/run_logs/fallback_write_diag_20260602_hard_soft_seed7.log`). This
  is diagnostic only; runtime fallback is unchanged. It classifies hard blockers
  (`force_fallback`, empty evidence pool, invalidator active) separately from
  soft underconfidence (slot/epi below threshold) and reports possible
  answer-with-caveat/calibration candidates.
- Hard/soft read: in this seeded CUDA run, applicable fallback was 0.55, while
  blocked/negative remained 1.00. Among applicable fallbacks, hard=0.00 and
  soft=1.00. Among blocked/negative fallbacks, hard=1.00 and soft=0.00. So the
  remaining over-fallback is not hard safety doing too much; it is calibration
  fallback. Applicable soft rows had support@1=0.50, support@3=0.65, evidence@3
  0.77, slot-near-threshold 0.54, primary-epi-near-threshold 0.08, and
  best-gold-epi-near-threshold 0.46. This confirms the next blocker is selected
  support + epistemic/slot calibration, not removal of fallback.
- Promoted the hard/soft split into runtime policy semantics. `fallback_needed()`
  remains the backward-compatible boolean, but `fallback_policy()` now returns
  `normal_answer`, `soft_fallback`, or `hard_fallback` with explicit hard/soft
  reasons and gate bits. `GraphAttentionInjector.get_fallback_policy()` exposes
  this to controller code. Runtime answering is still unchanged for now: soft
  fallback is only first-class state, not automatic silent answering.
- Control contract going forward: `hard_fallback` should route to full V4
  fallback/tool loop; `soft_fallback` is eligible for answer-with-caveat or one
  cheap verifier step; `normal_answer` is the current confident V5 path. Stage
  3/4 stay held until this soft-exit policy is tested, because future training
  should learn against the final three-state control semantics, not the old
  "all uncertainty equals total fallback" behavior.
- Added the first `soft-exit runtime policy simulation` report to
  `fallback_write_diag`. It compares three held-out routes:
  `old_full_fallback`, `answer_with_caveat`, and `verify_once`. The report tracks
  policy state by case type, V5-answer rate, verifier-needed rate, full-V4 rate,
  accept/reject proxy, safety-leak proxy, changed-to-hard rate, write ratio, and
  relative latency units. This is explicitly offline: it uses routing/slot/epi
  labels as proxy outcomes because the current 288-row corpus does not include
  rubric terms and the harness does not generate fresh V5 answers. Real
  accepted/rejected, catastrophic, and gibberish rates still require a later
  real-LM runtime run.
- Small control alignment fix: `fallback_policy(..., evidence_pool_empty=True)`
  now treats an empty evidence pool as `hard_fallback`, and
  `GraphAttentionInjector` passes that evidence-mask signal through. This keeps
  runtime policy semantics aligned with the diagnostic hard blocker.
- Checkpoint discipline update: from here on, every meaningful training,
  diagnostic, or control-policy checkpoint should update this file and be pushed
  to GitHub before the work is considered handed off. Treat chat-only metrics as
  incomplete unless they are mirrored here.
- Soft-exit seeded rerun complete (log:
  `artifacts/run_logs/fallback_write_diag_20260602_soft_exit_seed7.log`). Split
  stayed at 61 held-out rows: applicable=47, blocked=11, negative=3. Fallback is
  now applicable=20/47 (0.43), blocked=11/11 (1.00), negative=3/3 (1.00).
  Policy states are cleanly separated: applicable normal=0.57, soft=0.43,
  hard=0.00; blocked/negative normal=0.00, soft=0.00, hard=1.00.
- Soft-exit route simulation from that rerun: `old_full_fallback` answers 57% of
  applicable rows with V5 and sends 43% to full V4 fallback (latency unit 1.43).
  `answer_with_caveat` answers all applicable rows, but proxy acceptance remains
  0.57 and proxy rejection is 0.43 (latency 1.00). `verify_once` uses the
  verifier on 43% of applicable rows, accepts 60%, sends 40% to full fallback,
  and has latency 1.51. Blocked/negative stay hard-fallback with no proxy safety
  leaks on all three routes.
- Stage 3/4 verdict after soft-exit rerun: still HOLD. The safety floor is good,
  and applicable fallback improved, but soft-exit is not useful enough yet:
  `verify_once` only moves applicable proxy acceptance from 0.57 to 0.60.
  Direct_judgment soft rows remain the main blocker: support@1=0.38,
  support@3=0.62, evidence@3=0.75, slot-near=0.56, primary-epi-near=0.06,
  best-gold-epi-near=0.44, verifier-eligible only 0.06. Next work should improve
  direct_judgment support/epistemic + verdict/reason slot calibration, then rerun
  this exact soft-exit report.
- Read from the controlled pass: the cleaned projection restores strong evidence
  routing and safety, but the same `30/20/20` recipe does not reduce applicable
  fallback below the current band. The next lever is calibration/label quality
  on direct_judgment slots + epistemic confidence and planning substrate, not
  Stage 3/4 or a blind second schedule.
- Frontend demo collection checkpoint: the requested `E:\PROJECT\front-end`
  path is actually `E:\PROJECT\graph-front-end` in this workspace. That demo API
  now defaults to `GRAPH_BACKEND_DIR=..\graph_v5` instead of `..\graph_final`,
  imports the V5 `answerer_v4` / `V4OpencodeController`, and passes
  `collect_corpus=True` with explicit V5 paths. Finalized demo/test runs append
  to `graph_v5/data/distillation_corpus/sessions.jsonl`; persisted session
  subgraphs go to `graph_v5/data/session_subgraphs`; live signature stats use
  `graph_v5/data/signature_stats`.
- `answer_query_v4()` now accepts explicit `corpus_root`, `corpus_file`,
  `corpus_extra_metadata`, and `session_root` arguments. This prevents frontend
  demo runs from accidentally writing generated data under the frontend cwd.
  The frontend `/api/health` response also reports `backend_name`, `corpus_path`,
  `session_subgraph_root`, and `demo_collects_corpus` so the team can verify the
  demo is harvesting into V5 before running public tests.
- Verification for the frontend collection checkpoint: `py_compile` passed for
  `graph-front-end/api/frontend_api.py`, `graph-front-end/api/test_frontend_api.py`,
  and `graph_v5/answerer_v4.py`; `python -m unittest api.test_frontend_api -v`
  passes 4/4. A clean health import reports backend `graph_v5`, graph dir
  `graph_v5/graphs`, corpus `graph_v5/data/distillation_corpus/sessions.jsonl`,
  session root `graph_v5/data/session_subgraphs`, and `demo_collects_corpus=True`.
- Push note: V5 backend/readme checkpoint is on Agentic-Graph-Reasoner `main`
  at `9ea1a9f`. The frontend code checkpoint was pushed to
  `The-Only-Lazy-Guy/PROJECT` branch `frontend-graph-v5-demo` at `e6dc85cc`.
  I used this clean branch because the local `reasoning-architecture` branch is
  12 commits ahead of `origin/main`; pushing it directly would drag unrelated
  history instead of only this demo-collection change.
- V4 quality checkpoint after inspecting
  `data/session_subgraphs/v4_bb5257d3c6e3`: the matching corpus sample is
  `data/distillation_corpus/sessions.jsonl` line 174. The run was creative but
  overconfident: `finalized=True`, `coverage_pct=1.0`, 54 tool calls, 12
  searches, one model-supplied `verify_hypotheses`, controller `VERIFY=0`,
  `controller_fallback_used=True`, and design slot fill only 2/10. This is not
  a clean V5 positive even though it produced an answer.
- V4 verification/finalization fix: `verify_hypotheses()` now only accepts a
  `verified` verdict when the evidence cites a node already read in the session,
  the cited node overlaps the user's question, and the cited text has minimal
  lexical support for the hypothesis. Weak analogies should be discarded or
  caveated, not stamped as verified. The finalization loop also adds one repair
  round when quality issues are present (`unverified_hypotheses`,
  `weak_verified_hypotheses`, `low_design_slot_coverage`, etc.).
- V5 data-shape fix: corpus rows now persist `trace.turn_summaries`,
  `trace.controller_raw_trace`, `trace.controller_raw_trace_summary`,
  `metrics.finalization_quality`, and `quality.training_eligible` /
  `quality.v5_label_status`. This makes downstream filtering cheap: train on
  rows where support/verification is clean; keep creative-but-weak rows as
  `needs_review` or negative/soft-fallback calibration data.
- Prompt/data guidance added for V4: complex/design tasks should emit a private
  `<evidence_audit>` before `<answer>` with claims, support node IDs, status,
  confidence, and open questions. This is the useful V5 supervision signal; the
  goal is not more raw hidden CoT, but structured evidence/support/quality labels
  that prevent expensive post-hoc parsing.
- Verification for this V4 checkpoint: `py_compile` passed for `answerer_v4.py`,
  `reasoning/distillation_corpus.py`, `reasoning/tests/test_v4_corpus_quality.py`,
  and `run_repeat_learning_experiment.py`. Targeted tests passed:
  `pytest -q reasoning/tests/test_v4_corpus_quality.py reasoning/tests/test_model_patch_extraction.py test_raw_trace_capture.py`
  -> 12/12.
- Frontend live-steps checkpoint: V4 `answer_query_v4()` now accepts an
  `event_callback` and emits public progress events while the run is still in
  flight: `model_turn`, `plan_update`, `tool_result`, and `answer_candidate`.
  These events intentionally carry summaries, counts, tool names, and small
  metadata only; they do not expose raw hidden reasoning. The frontend V4 stream
  path now uses the same threaded SSE queue as the other modes, so V4 events are
  forwarded immediately instead of waiting for the final payload.
- Demo UI live-step behavior: `graph-front-end` renders a "Live steps" timeline
  during streaming and mirrors those events into the synthetic live session
  graph. This makes the demo show search/read/verify-style progress and answer
  candidates while still collecting finalized V4 rows into
  `graph_v5/data/distillation_corpus/sessions.jsonl`.
- Verification for the live-steps checkpoint: `python -m py_compile
  answerer_v4.py` passed; `python -m unittest api.test_frontend_api -v` passed
  4/4 with the frontend API wired to pass the event callback into V4;
  `npm.cmd run typecheck` passed; `npm.cmd run build` passed with only Vite's
  large chunk warning. Stage 3/4 remain held; this is a demo/data-collection and
  observability checkpoint, not a V5 training-quality gate.
- Read: the projected pipeline is now real and end-to-end, evidence routing is
  materially stronger than planning. The false-invalidator blocker is repaired;
  the remaining fallback failures are mostly missing slots + low epistemic on
  top evidence. Do not move to Stage 3/4 yet.

---

## TL;DR claim boundary

- ✅ V5 trains end-to-end on **real graph states + real LM hidden states, including
  planning**, on a substrate-rich V4 corpus.
- ✅ V5 **generates** end-to-end with the adapter live; injection is numerically
  stable (stays coherent even with untrained projections).
- ✅ Random-init injection is **95% non-catastrophic** over 20 questions (perfect
  1/1 hook control) → Stage 2 starts from a stable injected-generation baseline.
- ✅ **Stage 2 core (synthetic)**: residual gate + 2A/2B trainer learns attention
  routing (plan/evid precision → 1.0) with bounded gated write (~11% of ‖h‖);
  negatives stay diffuse (entropy ln 3), positives confident, no collapse.
- ✅ **Stage 2A on REAL corpus**: routing plan 0.76→1.00, evid 0.37→1.00;
  perturbation re-check 0/20 catastrophic, hooks 20/20, sim 0.95 (W_o/gate frozen
  → generation untouched, as intended for "learn to look").
- ✅ **Stage 2B on REAL corpus (write-safety)**: write path trained, all 6 gates
  pass — write_ratio 0.047 (negatives lowest 0.034), catastrophic 0/20, hooks
  20/20, sim 0.94. Generation stable with real writing.
- ✅ **Integrated Stage 1→2A→2B (one adapter)**: 7/8 gates. Heads retained
  (head-retention loss fixed epi 0.38→0.88), routing 1.0, write 0.109 (negatives
  least 0.057), catastrophic 0/20, fallback blocked/negative HIGH.
- ⚠️ **Fallback applicable-drop**: only 1.00→0.94 (1/17). The fallback gate needs
  slot≥0.85 AND primary-evidence epi≥0.70; the 20-example corpus doesn't calibrate
  the heads to cross those thresholds. A calibration + corpus-scale issue (motivates
  a support-pointer head), not a training-mechanism failure.
- ✅ **Scaled 20 → 46 traces** (local GGUF gen, 35 finalized, 382 patches) and
  re-ran held-out (10 eval). KEY FINDING: 2.3× data improved slot/shortcut/node
  generalization but did NOT fix **fallback-applicable (still 1.00)** or
  **epistemic generalization (all-node 0.00)** → points to an architecture/label
  problem (the **support-pointer head**), NOT pure data scale. (n=10 still < 100–300
  bar; indicative not conclusive.)
- 🔬 **Two diagnostics ruled out the obvious fixes; the fallback gate is
  data-scale-bound, not a single-head fix.**
  - Oracle support-pointer: gold support node did NOT drop applicable fallback
    (7/7) → **support-pointer head ruled out**.
  - Slot sweep (0.50→0.85): applicable fallback **flat at 1.00**, slot recall 0.60
    (40% of required slots predicted <0.50) → **threshold tuning ruled out**.
  - Gold-slot oracle: applicable only 1.00→0.86 → **slots alone aren't the
    blocker; epi/inv also fail** (conjunctive gate). The two runs disagree on the
    dominant failing condition → **n=7 held-out is variance-dominated.**
  → Verdict: **multi-head calibration that only generalizes with more data.**
    Next = SCALE the corpus (vast.ai pipeline), get held-out n≥30, THEN calibrate
    per-condition. Stop running n=7 diagnostics (noise).
- ⏸️ **Held on purpose**: support-pointer head, slot-threshold change, Stage 3,
  Stage 4, any quality claim.
- ❌ NOT yet: V5 **generalizes** (corpus is 20 traces → train-fit only).
- ❌ NOT yet: V5 **improves** generation (Stage 2 not yet on the real 1536-d adapter;
  LoRA untrained).
- ⏳ Staged from full Stage 2 spec: KL-vs-base-LM stability loss (use
  `perturbation_baseline` on real LM), explicit head-retention loss, "no-graph"
  negative cases, and real-LM post-Stage-2 catastrophic-rate check.

---

## 1. Inference demo (raw) — `python -m v5.infer_demo`

Qwen2.5-1.5B, greedy, binary-search applicability question. **n=1, random-init
projections — anecdotal, NOT a quality claim.**

```
question: Is binary search applicable to find a target in this array,
          and what precondition must hold?

BASELINE (no adapter):
  "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]\n\nTo determine if binary search is
   applicable to find a target in the given array, we need to check if the
   array is sorted in ascending order. Binary search is an efficient algorithm
   for finding an"   <- rambles, hallucinates array, never answers

V5-INJECTED (untrained projections):
  "The array is sorted in ascending order. Yes, binary search can be applied to
   find a target in a sorted array. The precondition that must hold is that the
   array must be sorted in ascending order. Binary search works by repeatedly
   dividing the search interval in half..."   <- direct: verdict + precondition

hook call counts: {'planning': 1, 'evidence': 1}   (decode steps skipped)
fallback_needed: True   (heads untrained)
DIFF: outputs differ
```

Key finding: injection stayed **coherent** with random `W_o` → residual magnitude
is sane → Stage 2 will *shape* it, not fight catastrophic perturbation.

---

## 1b. Perturbation baseline (raw) — `python -m v5.perturbation_baseline`

20 corpus questions, baseline vs V5-injected (random-init projections), Qwen2.5-1.5B.
Goal: prove the adapter is *usually non-catastrophic* before Stage 2 — NOT improvement.

```
AGGREGATE (n=20):
  hook control ok (1/1)    : 20/20
  baseline gibberish       : 1/20
  injected gibberish       : 1/20
  CATASTROPHIC (inj broke) : 1/20  (5%)
  non-catastrophic rate    : 95%
  mean baseline length     : 272 chars
  mean injected length     : 265 chars   (no length collapse)
  mean semantic sim        : 0.73        (1=identical; moderate drift, stays related)
```

Read: random injection rarely breaks generation (1 catastrophic case, sim→0.43);
moderate drift is expected with untrained projections; hook control perfect.
=> Stage 2 starts from a stable injected-generation baseline.

---

## 1c. Stage 2 core (raw) — `python -m v5.training.stage2`

Synthetic, lm_dim=128, gate_init=0.02. Trains attention routing (2A) then gated
write (2B). NOT answer-quality; proves the projections can learn where to look and
write a bounded residual.

```
--- Stage 2A (learn to LOOK: train Q/K/V; W_o/gate frozen) ---
  BEFORE:  plan_attn 0.10  evid_attn 0.53  write_ratio 0.017
  AFTER 2A: plan_attn 1.00  evid_attn 1.00  write_ratio 0.111

--- Stage 2B (learn to WRITE: train W_o + gate; Q/K/V on) ---
  AFTER 2B: plan_attn 1.00  evid_attn 1.00  write_ratio 0.116
            gates (plan/evid) = 0.002 / 0.023

success criteria: plan>=0.9 OK · evid>=0.9 OK · write bounded (<=0.35) OK
```

Note: lr 1e-3 diverged (attn loss 1.8→9.6) on the small attention pool; lr 2e-4
converges. Residual gate keeps the write ~11% of ‖h‖.

With negatives (15+15 positive, 10 negative): positives confident (entropy
0.00/0.35), negatives maximally diffuse (entropy 1.10 = ln 3), top-1 freq 0.5/0.5
(no collapse).

---

## 1d. Stage 2A on REAL corpus + perturbation re-check — `python -m v5.training.stage2_real`

Qwen2.5-1.5B, substrate graph, gate=0.02, W_o frozen (learn to LOOK only).

```
attention routing (real corpus): plan 0.76 -> 1.00,  evid 0.37 -> 1.00

PERTURBATION RE-CHECK (Stage-2A adapter, 20 questions):
  catastrophic           : 0/20
  non_catastrophic_rate  : 1.000
  hooks_ok               : 20/20
  mean_base_len          : 272
  mean_inj_len           : 265
  mean_sim               : 0.948   (generation barely changes — W_o/gate frozen)
  injected_gibberish     : 0
```

HONEST CONFOUND: this used gate=0.02 vs the random baseline's gate=1.0, so 0%
catastrophic is partly because 2A barely writes by design. Write-safety is only
truly tested in Stage 2B (W_o + gate trained) — intentionally held.

---

## 1e. Stage 2B on REAL corpus — write-safety milestone — `python -m v5.training.stage2b_real`

Qwen2.5-1.5B. Train W_o + gate (Q/K/V lower LR), real positives + real negatives.
NOT a quality milestone — tests whether the adapter can WRITE without breaking safety.

```
per-case-type (after 2B):
  tag          n   write_ratio   fallback
  applicable  17     0.048         1.00
  blocked      3     0.044         1.00
  negative     5     0.034         1.00   <- negatives write LEAST
gates plan/evid: 0.012 / 0.008    overall write_ratio 0.047

perturbation re-check (20q): catastrophic 0/20, hooks 20/20, gibberish 0, sim 0.940

WRITE-SAFETY GATES (all OK): catastrophic ~0 · hooks 20/20 · no gibberish ·
  sim>=0.5 · write<=0.20 · negatives <= positives
```

HONEST CAVEAT: standalone 2B run -> aux heads are random (untrained), so
fallback_needed is 1.0 for ALL case types (retained, not regressed, but NOT the
desired "drops for applicable"). The applicable-fallback-drop needs Stage 1 heads
+ Stage 2 on ONE adapter (pipeline integration) — separate from write-safety.

---

## 1f. Integrated Stage 1->2A->2B — `python -m v5.training.stage_integrated`

One adapter through Stage 1 (heads) -> 2A (routing) -> 2B (write + head-retention).
Qwen2.5-1.5B. 7/8 integrated gates pass.

```
head metrics retained (after 2B): plan 1.0 evid 1.0 slot 1.0 epi 0.88 sc 1.0
  (head-retention loss fixed the regression: epi WAS 0.38 without it)
routing retained: plan 1.00 evid 1.00

per-case-type (write_ratio | fallback before->after):
  applicable  17   0.109   1.00 -> 0.94
  blocked      3   0.117   1.00 -> 1.00
  negative     5   0.057   1.00 -> 1.00     <- negatives write least

perturbation (20q): catastrophic 0/20, hooks 20/20, gibberish 0, sim 0.88

GATES: 7/8 OK
  [OK] head retained · routing retained · write bounded · negatives least ·
       catastrophic<=baseline · fallback blocked HIGH · fallback negative HIGH
  [FAIL] fallback applicable LOW  (0.94 — see below)
```

UNMET GATE: applicable fallback barely drops (1.00->0.94). fallback_needed wants
slot>=0.85 AND primary-evidence epi>=0.70; the 20-example corpus doesn't calibrate
the heads to cross those specific thresholds. Calibration + corpus-scale issue
(motivates an explicit support-pointer head), NOT a training-mechanism failure —
heads, routing, and write all train and retain.

---

## 2. Substrate Population Pass (raw) — `python -m v5.training.substrate`

```
base: 831 nodes, 1454 edges
substrate nodes added: 47
  epistemic_state    27
  strategy            7
  reasoning_atom      5
  solved_subgoal      4
  failure_pattern     4
relations added: 79
total: 878 nodes, 1533 edges  -> graphs/merged_graph_substrate.json
planning-pool substrate nodes added: 16
```

---

## 3. Bridge coverage (raw) — `python -m v5.training.bridge`

```
subgraph size (avg nodes/example):
  anchors-only         : 5.0    (~0.5 edges)
  persisted 1-hop nbhd : 17.8   (real edges -> real R-GCN message passing)

per-head label coverage     base graph   substrate-enriched
  plan                       0/20 (0%)    17/20 (85%)
  evid                      19/20 (95%)   19/20 (95%)
  slot                      20/20 (100%)  20/20 (100%)
  epi                        8/20 (40%)    8/20 (40%)
  inv                        1/20 (5%)     1/20 (5%)
  shortcut                  20/20 (100%)  20/20 (100%)
```

---

## 4. Real Stage 1 training (raw) — `python -m v5.training.stage1_real`

Qwen2.5-1.5B (hidden=1536, anchor_layer=8), substrate-enriched graph, 20 examples,
150 epochs, loss 18.5 → 2.99. **Train-fit, no held-out split.**

```
head        before   after
planning    0.94  -> 1.00   (now supervised via substrate anchors)
evidence    0.32  -> 1.00
slot        0.00  -> 1.00
epistemic   0.00  -> 1.00
shortcut    0.65  -> 1.00
```

Synthetic trainability (`python -m v5.training.trainability_test`): all heads
0.5 → 1.0, fallback_applicable 1.0 → 0.0, fallback_blocked stays 1.0.

---

## 5. Repro commands (Windows PowerShell)

```powershell
python -m v5.smoke_test_toy                         # deterministic invariants (fast)
python -m v5.training.trainability_test             # synthetic head trainability
python -m v5.training.substrate                     # build substrate-enriched graph
python -m v5.training.bridge                        # corpus -> Stage1Example + coverage
python -m v5.training.stage2                         # Stage 2 (2A routing + 2B write), synthetic
$env:KMP_DUPLICATE_LIB_OK="TRUE"; python -u -m v5.training.stage2_real   # real Stage 2A + perturbation re-check
$env:KMP_DUPLICATE_LIB_OK="TRUE"; python -u -m v5.training.stage2b_real  # real Stage 2B write-safety milestone
$env:KMP_DUPLICATE_LIB_OK="TRUE"; python -u -m v5.training.stage_integrated  # integrated 1->2A->2B (7/8 gates)
python merge_shards.py --out data/corpus_merged.jsonl
python project_corpus_to_v5_targets.py --corpus data/corpus_merged.jsonl
pytest reasoning/tests/test_v5_projection.py -q
$env:KMP_DUPLICATE_LIB_OK="TRUE"; python -u -m v5.training.corpus_scaling --corpus <corpus.jsonl>  # scale + held-out metrics
$env:KMP_DUPLICATE_LIB_OK="TRUE"; python -u -m v5.training.fallback_write_diag --corpus data/corpus_merged_v5proj.jsonl --seed 7  # fallback/write + calibration diagnosis
```

## 1g. Corpus scaling 20 -> 46 traces (held-out) — `v5.training.corpus_scaling`

Local GGUF generation (run_gen_llama.py -> llama-server :6768), 46 traces
(35 finalized, 382 patches), 41 train / 10 held-out.

```
coverage: plan 76% · evid 98% · slot 100% · epi 76% · inv 2% · shortcut 100%

HELD-OUT (10 unseen):
  plan node  P@1=0.57  recall=0.63   (n=7)
  evid node  P@1=0.67  recall=0.58   (n=9)
  head acc (strict all-node): slot=0.89  epi=0.00  shortcut=0.89
  fallback:  applicable=1.00  blocked=1.00  negative=1.00
  write:     applicable 0.097 · blocked 0.128 · negative 0.056 (negatives least)
```

vs n=20: node attention now generalizes modestly (the earlier P@1=1.0 was noise);
slot/shortcut generalize (0.89). BUT fallback-applicable stayed 1.00 and epi
stayed 0.00 across BOTH scales -> the epistemic/fallback gate is an architecture/
label issue (support-pointer head), not data-scale. Caveats: n=10 < 100-300;
epi all-node match is strict (per-node metric added).

---

## 1h. Oracle support-pointer diagnostic — `v5.training.oracle_support_diag`

Decisive pre-build test on 46-trace held-out applicable (n=7):
```
applicable fallback:  standard 7/7   ORACLE gold-support 7/7   (oracle does NOT help)
trips:  slot 7/7  ·  inv 1/7  ·  epi_top 2/7
mean epi:  top-attended 0.71 · gold-support 0.86   (mostly >= 0.70 threshold)
VERDICT: NOT support selection -> SLOT calibration is the blocker.
```
=> under the old raw-anchor label regime, support-pointer alone was not the fix.

## 1i. Slot-calibration sweep — `v5.training.slot_calibration_diag`

```
SWEEP (held-out, applicable n=7):  applicable fallback = 1.00 at EVERY thresh 0.50..0.85
  slot_P 1.00  slot_R 0.60  (40% of required slots predicted <0.50 -> fail at any thresh)
GOLD-SLOT ORACLE:  predicted@0.85 applicable 1.00  ->  GOLD slots applicable 0.86 (barely)
RECOMMENDATION: no threshold helps; gold slots barely help -> NOT a slot-threshold fix.
```
Combined with the oracle, the fallback gate looked CONJUNCTIVE (slots AND
no-inv AND epi); multiple heads under-calibrated on held-out and the dominant
failing condition varied between runs (n=7 = variance-dominated). This was a
useful historical diagnostic, but the later 288-trace inspection exposed a more
basic teacher-projection mismatch too, so treat this section as small-n
diagnostics, not the final verdict on the projected-label pipeline.

---

## 1j. V5-native corpus projection + 288-trace held-out pilot

Multi-machine `opencode` generation produced 2 shard files (144 local + 144
vast) which merged into 288 unique traces. Inspecting the raw rows showed the
main new problem: V4 supervision was too tool-path-shaped for V5. We therefore
added a projection pass that converts each row into V5-native targets before
training.

Repro:
```
python merge_shards.py --out data/corpus_merged.jsonl
python project_corpus_to_v5_targets.py --corpus data/corpus_merged.jsonl
pytest reasoning/tests/test_v5_projection.py -q
$env:KMP_DUPLICATE_LIB_OK="TRUE"; python -u -m v5.training.corpus_scaling --corpus data/corpus_merged_v5proj.jsonl --model Qwen/Qwen2.5-0.5B-Instruct --e1 30 --e2a 20 --e2b 20
```

Projection / bridge stats:
```
merge_shards:
  merged 288 unique traces  (46 dups dropped)

project_corpus_to_v5_targets:
  rows 288
  planning rows 189
  evidence rows 288
  support rows 253
  loop-supervised rows 288
  mean candidate nodes 9.2

substrate + bridge smoke:
  +685 substrate nodes
  +1151 relations
  288 examples
  planning labels 239/288 (83%)
  evidence labels 288/288 (100%)
  avg nodes/example 29.38
```

Held-out pilot (`234 train / 59 held-out`, shortened schedule for turnaround):
```
plan  node  precision@1=0.28  recall@gold=0.25  (n=47)
evid  node  precision@1=0.88  recall@gold=0.54  (n=58)
head acc (strict all-node): slot=0.66  epi=0.30  shortcut=0.62  inv=0.00
epi per-node acc: 0.96
fallback: applicable=1.00  blocked=1.00  negative=1.00
write ratio: applicable=0.161  blocked=0.183  negative=0.197
```

Read: the projected pipeline is now real and end-to-end. Evidence routing
generalizes much better than planning, but planning remains weak and fallback
still fires on every held-out case type. So the immediate blocker is no longer
"can we scale data?" or "can the architecture train?" - it is planning /
fallback supervision and calibration on the projected corpus.

Follow-up diagnostic (`v5.training.fallback_write_diag`) found a concrete
bug in the first pilot: `task_frame.required_slots` was not canonicalized even
though `slot_fill_target` was. That made fallback check `unknown` for aliases
like `answer`, `relationship`, and `explanation`. After canonicalizing task-frame
slots, the same short diagnostic schedule reported:
```
fallback: applicable=0.83  blocked=0.82  negative=1.00

applicable trip reasons:
  missing_slot       30/47
  low_epistemic      19/47
  invalidator_active 12/47

oracle fallback:
  predicted_all     applicable=0.83
  gold_slots_only   applicable=0.70
  gold_epi_only     applicable=0.81
  gold_inv_only     applicable=0.83
  gold_all          applicable=0.64

write ratio:
  applicable fallback total=0.147
  applicable no_fallback total=0.199
  blocked fallback total=0.160
  negative fallback total=0.224
```

Read after the diagnostic: the alias bug was real and fixing it helps, but the
pipeline is still over-conservative. Gold labels do not fully rescue applicable
fallback, and negative write remains the highest, especially in the planning
block. Do not move to Stage 3/4 yet.

Follow-up fix: Stage 2B now applies a differentiable negative write penalty plus
negative head suppression, and the no-graph negative bank is larger (15 prompts).
The comparable diagnostic now reports:
```
fallback: applicable=0.83  blocked=0.73  negative=1.00

write ratio:
  applicable fallback total=0.143
  applicable no_fallback total=0.180
  blocked fallback total=0.183
  blocked no_fallback total=0.160
  negative fallback total=0.107

task-family signal:
  relational_explanation fallback=1.00, invalidator_active=0.93
```

Read after the write fix: the safety ordering is sane again (negative writes
least and still falls back), but the target is not yet met. The next bottleneck
is invalidator over-firing in `relational_explanation`, where high-confidence
support claims are being marked invalidators without gold invalidator labels.

Invalidator semantics pass:
```
changes:
  invalidator candidate = source of INVALIDATED_BY/CONTRADICTS edge
                          only when the destination is also in the active subgraph
  self invalidated_by edges ignored as malformed graph noise
  bridge invalidator loss now uses the structural candidate mask
  inactive structural candidates train as inv=0

diagnostic after pass:
  fallback: applicable=0.74  blocked=1.00  negative=1.00
  invalidator_active: applicable=0.00  blocked=0.00  negative=0.00
  relational_explanation: fallback=0.50  invalidator_active=0.00

write ratio:
  applicable fallback total=0.161
  applicable no_fallback total=0.166
  blocked fallback total=0.173
  negative fallback total=0.150

oracle fallback:
  predicted_all applicable=0.74
  gold_slots_only applicable=0.68
  gold_epi_only applicable=0.74
  gold_inv_only applicable=0.74
  gold_all applicable=0.60
```

Read after invalidator semantics: the `relational_explanation` false-invalidator
issue is fixed. The remaining fallback bottleneck is now slots + epistemic
calibration, with planning misses still visible (`top3_near_miss`, `diffuse_miss`,
`confident_miss`). Blocked/negative still fall back, so safety is retained, but
do not move to Stage 3/4 yet.

Applicable fallback calibration pass (`fallback_write_diag --seed 7`):
```
repro:
  $env:KMP_DUPLICATE_LIB_OK="TRUE"; python -u -m v5.training.fallback_write_diag \
    --corpus data/corpus_merged_v5proj.jsonl \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --e1 30 --e2a 20 --e2b 20 --seed 7 \
    --invalidator-limit 4 --applicable-limit 14

fallback:
  applicable=30/47 (0.64)
  blocked=10/11 (0.91)
  negative=2/3 (0.67)

applicable gate combos:
  missing_slot+low_epistemic: 24
  low_epistemic only:          4
  missing_slot only:           2
  no_fallback:                17

failed applicable slots:
  reason: 25
  verdict: 16
  condition/alternative/complexity: 1 each

low-epi buckets:
  low_epi_on_gold_evidence:          18
  wrong_primary_gold_evidence_ok:     7
  all_gold_evidence_low:              3

family calibration:
  direct_judgment        n=35 fallback=0.83 slot=0.71 epi=0.77 plan@1=0.14 plan@3=0.49
  relational_explanation n=11 fallback=0.00 slot=0.00 epi=0.00 plan@1=0.64 plan@3=0.82
  design_synthesis       n=1  fallback=1.00 slot=1.00 epi=1.00

write:
  applicable fallback total=0.189
  applicable no_fallback total=0.190
  blocked fallback total=0.194
  negative fallback total=0.101
  negative no_fallback total=0.089
```

Read after calibration: direct_judgment is the real remaining cluster. The
failure is usually not "slot is barely under 0.85"; the failed reason/verdict
slots are often near zero. Epistemic is also not just wrong-primary selection:
18 low-epi cases are on gold evidence. This points to direct_judgment slot /
epistemic target quality or head calibration. Also, the seeded run shows one
negative held-out case did not fall back, so negative safety needs another guard
check before any Stage 3/4 move.

---

## Multi-machine UNIQUE data gen (local + vast.ai) — push & merge

```
# bank is sharded by index (disjoint -> unique). 100 Qs -> 50 local / 50 vast1.
LOCAL :  python run_gen_llama.py --run-id local --shard-index 0 --num-shards 2
VAST  :  RUN_ID=vast1 SHARD_INDEX=1 NUM_SHARDS=2 bash gen_and_push.sh   # gens + pushes its shard
LOCAL :  git pull ; python merge_shards.py --out data/corpus_merged.jsonl ; \
         python project_corpus_to_v5_targets.py --corpus data/corpus_merged.jsonl ; \
         python -m v5.training.corpus_scaling --corpus data/corpus_merged_v5proj.jsonl
```
Each machine writes data/corpus_shards/<run-id>.jsonl (distinct file, no conflict;
.gitignore exception lets these jsonl push). merge_shards de-dups by (session, question).

---

## Scaling the corpus (data-gen on a fresh box / cloud)

```
# fresh environment setup (venv + deps + LLM backend + env vars)
setup_datagen_env.bat            # Windows   (--gpu for CUDA torch, --run to generate)
bash setup_datagen_env.sh        # Linux/cloud

# backend: opencode CLI (npm i -g opencode-ai; opencode auth login)
#          OR llama-server serving a GGUF on :6768  (LOCAL_LLM_BASE_URL)

# generate traces, then scale + measure held-out calibration:
python run_phase15_corpus.py --dataset <questions.json> --graph graphs/merged_graph.json --mode harvest
python merge_shards.py --out data/corpus_merged.jsonl
python project_corpus_to_v5_targets.py --corpus data/corpus_merged.jsonl
python -m v5.training.corpus_scaling --corpus data/corpus_merged_v5proj.jsonl
$env:KMP_DUPLICATE_LIB_OK="TRUE"; python -u -m v5.realstack_test       # real-stack prefill
$env:KMP_DUPLICATE_LIB_OK="TRUE"; python -u -m v5.training.stage1_real # real Stage 1 (planning incl.)
$env:KMP_DUPLICATE_LIB_OK="TRUE"; python -u -m v5.infer_demo           # baseline vs injected generation
$env:KMP_DUPLICATE_LIB_OK="TRUE"; python -u -m v5.perturbation_baseline --n 20  # non-catastrophic baseline
```

Env note: `sentence_transformers` segfaults when co-loaded with `torch_geometric`
here — we use `transformers.AutoModel` for mpnet. Always set
`KMP_DUPLICATE_LIB_OK=TRUE` for the heavy combos.

---

## 6. Next true milestone

UPDATE 2026-06-01:

No longer raw corpus scale by itself - we now have 288 traces and a projected
V5-shaped corpus. The next milestone is to improve held-out planning / fallback
generalization on that projected corpus:

1. tighten planning / support supervision in the projection pass
2. rerun a longer held-out Stage 1 -> 2A -> 2B schedule on `data/corpus_merged_v5proj.jsonl`
3. calibrate fallback conditions only after planning / epi stop collapsing on eval
4. only then talk about LoRA / Stage 4 / inference-quality claims

Not architecture. **Scale the V4 corpus** (more traces) → enables an 80/20
held-out split and the first *generalization* metrics: node precision/recall by
pool, slot/epistemic/invalidator/shortcut accuracy, fallback decision accuracy.
Then Stage 2 (train cross-attn projections) + LoRA before any inference-quality
claim. Full detail in `v5_PROGRESS.md`.
