  PRED-v3+ Reframing Plan

     Decision

     Committed direction: Option 1 — Unified end-to-end model. Build a single UnifiedProposalAlignerNet that produces full goal structure (session_nodes + edges + memory links + commits) in one forward pass.
     Train from scratch. No fix21 warm-start. Estimated effort: 1.5–2 days.

     Alternatives considered (Options 2–4) are documented below as context for the decision and as fallbacks if Option 1's training fails to reach viable performance.

     Context

     After 6+ independent attempts to improve the proposer/aligner pipeline (fix22 hard-negatives, fix23 full unfreeze, fix4 dot+attention, fix24 gold-synth-swap, fix25 augmented-synth fine-tune, plus capacity
      scans), every aligner adaptation has regressed or held flat. The current deployable pipeline is:

     - proposer: fix5 (DETR-lite attention + concat_mlp scorer)
     - synthesizer: deterministic template post-processing
     - aligner: fix21 (frozen, no adaptation has improved it)
     - end-to-end row_complete: 0.1745 lenient / 0.0000 text-faithful

     Two ceilings are now visible:

     ┌────────────────────────────────────┬────────┬─────────────────────────────────────────────────────────────────────────┐
     │              Ceiling               │ Value  │                                  Means                                  │
     ├────────────────────────────────────┼────────┼─────────────────────────────────────────────────────────────────────────┤
     │ Oracle (gold slots → fix21)        │ 0.3208 │ The aligner itself caps the pipeline at ~32% even with perfect upstream │
     ├────────────────────────────────────┼────────┼─────────────────────────────────────────────────────────────────────────┤
     │ A→B gap (template text robustness) │ 0.10   │ The aligner can't generalize to synthesized text                        │
     ├────────────────────────────────────┼────────┼─────────────────────────────────────────────────────────────────────────┤
     │ B→C gap (proposer accuracy)        │ 0.04   │ Proposer errors compound on top of aligner brittleness                  │
     └────────────────────────────────────┴────────┴─────────────────────────────────────────────────────────────────────────┘

     Six adaptations have failed for structural reasons, not tuning reasons:
     1. Parameter sharing via state_h means selective freezes don't isolate components
     2. Training on noisy synthesized text against gold edge supervision breaks the text-edge association
     3. fix21 sits at a sharp local optimum on gold distribution; gradient-based adaptation moves away from it

     The pipeline-mismatch lens is exhausted as an improvement vector. This plan considers what reframings of the problem could plausibly raise both ceilings.

     ---
     Decision space: four reframings

     The user selected "reframe the project" from the strategic options. The reframing space has four distinct directions, with different cost/risk profiles. This plan walks each one, then recommends.

     Option 1: Unified end-to-end model (eliminate proposer/aligner split)

     Concept: One model takes (signal, graph, spans, memory) and outputs the full goal structure: session_nodes (with text), session_edges, memory_attachments, covered_mappings, commits. No intermediate
     "session_nodes" abstraction that gets corrupted at the pipeline seam.

     Architecture sketch:
     - Shared encoder: BoW + MiniLM over signal, spans, memory (reuse PRED-v2 encoders)
     - Transformer-style decoder with K learnable slot queries (DETR pattern, but for joint output)
     - Multiple heads off slot queries:
       - use, span_pointer, is_bridge (proposer-equivalent outputs)
       - edges between slots (relation type + existence)
       - memory_attachments / covered_mappings (mem_kind + mem_rel)
       - text for synth slots: structured slot-to-template-arg pointers, not free generation

     Key insight: The session_node TEXT becomes a derived output, not an intermediate signal. The model predicts span pointers and (for synthesis slots) template arguments. Text is reconstructed
     deterministically from those pointers and arguments at output time — same template synthesis logic, but now generated FROM the model's predictions rather than supervising the model's intermediate state.

     What this removes:
     - No distribution mismatch between proposer outputs and aligner inputs (no seam)
     - No text-edge supervision incoherence (the model never sees its own noisy text as input)
     - No reconciliation step (everything is one prediction)

     What this requires:
     - New model architecture (~600 lines)
     - New training script (~400 lines)
     - Same data (pred_v1 train/val)
     - Same evaluation harness

     Expected outcome: Hard to predict without trying. Best case: row_complete > 0.32 because joint training finds configurations that pipeline-trained components can't. Worst case: doesn't reach 0.32 because
     end-to-end optimization is harder than the staged version.

     Risk: medium-high. Architecture is more complex; harder to debug; may not converge.

     Cost: 1.5–2 days for v1 implementation + training.

     ---
     Option 2: Joint training of existing proposer + aligner (keep split, fix the seam)

     Concept: Keep proposer (fix5) and aligner (fix21) as separate models, but train them jointly with a single end-to-end loss. The proposer learns to produce outputs the aligner can handle; the aligner
     learns to handle the proposer's actual output distribution. They co-adapt.

     Architecture: unchanged. Just changes the training loop.

     Training procedure:
     - Single forward pass: proposer → reconciler → synthesizer → aligner
     - Single loss: aligner's standard losses (span, edge, mem, etc.) computed against gold
     - Backward through the synthesizer (treat as differentiable structured op) and reconciler (treat as non-differentiable, gradient passes through proposer's slot logits via straight-through or surrogate)
     - The proposer's gradient signal is "produce outputs that lead to correct aligner predictions"

     Tricky part: the reconciler is non-differentiable (Hungarian matching, name assignment by argmax). Need straight-through estimator or soft relaxation.

     What this fixes:
     - A→B gap (aligner adapts to proposer's actual distribution)
     - B→C gap (proposer adapts to what aligner can handle)

     What this doesn't fix:
     - Aligner's gold ceiling at 0.32 (joint training shifts the local optimum but doesn't fundamentally change capacity)

     Expected outcome: row_complete climbs from 0.17 toward 0.25–0.30. Doesn't break 0.32.

     Risk: medium. Joint training with non-differentiable bottlenecks is finicky.

     Cost: 1 day for joint trainer + scheduled-sampling scaffolding.

     ---
     Option 3: Scale the data (keep architecture, retrain from scratch on larger pred_v1)

     Concept: 1885 training rows is small. Pipeline ML systems are notoriously data-hungry; what looks like an architectural ceiling is often a data ceiling. Generate 5K–10K pred_v1 training rows via the
     existing generator (ngr_v1_tasks.py) with broader content variety, retrain both proposer and aligner from scratch.

     What this requires:
     - Data generator extension to produce more diverse content (might already work — depends on generator's parameter space)
     - Compute time for retraining (~10x longer training runs)
     - No architecture changes

     What this addresses:
     - May raise both the proposer ceiling and the aligner ceiling
     - Doesn't fix the pipeline mismatch but reduces its magnitude (better-trained components have more robust representations)

     Expected outcome: row_complete improves modestly across the board. Maybe 0.20 → 0.25 deployable. Doesn't fundamentally change the architecture's limits.

     Risk: low. No architectural risk. Just more training.

     Cost: 1–3 days depending on data generation complexity. Compute is the main cost.

     ---
     Option 4: Direct prediction (eliminate session_nodes abstraction entirely)

     Concept: The intermediate session_nodes representation is what creates the synthesis problem and the pipeline mismatch. What if the model predicts edges/mem DIRECTLY between spans and memory nodes,
     without an explicit "session_nodes" intermediate?

     Re-statement of the task:
     - Input: signal + graph + spans + memory
     - Output:
       - Which spans are "active" (replaces session_node identification)
       - For each active span pair: is there a session_edge, what relation
       - For each active span × memory: attach or cover, what relation
       - Commit family

     No session_nodes. No text synthesis. No reconciliation.

     What this loses:
     - The system can't produce "new" session_nodes whose text isn't a span (mixed_add_link new_note, multi_region_attach bridge). These tasks become unrepresentable in this formulation.

     What this requires:
     - Rethinking 2 of the 4 task types (mixed_add_link, multi_region_attach)
     - Either changing the task to fit the new formulation, or accepting partial coverage (only covered/long_decompose tasks work)

     Cost: 1 day for the new model + training, plus rethinking the failure modes.

     Why this is on the list: it's the cleanest formulation if the goal is "what spans/relations exist" without text generation. But it doesn't match the original problem statement, so it's likely a
     non-starter.

     ---
     Recommendation: Option 1 (Unified end-to-end model)

     Reasoning:

     Why not Option 2 (joint training): Doesn't address the 0.32 aligner ceiling. Joint training can recover some of the A→B and B→C gap, but the fundamental capacity issue of fix21 remains. The work-to-payoff
      ratio is moderate at best.

     Why not Option 3 (scale data): Best low-risk option, but the diminishing-returns trend across 10+ training runs suggests data isn't the dominant bottleneck. The cover_f1 regression in fix25 wasn't a data
     problem — it was a parameter-sharing problem. More data wouldn't change that.

     Why not Option 4 (direct prediction): Loses synthesis-task coverage, which is 53% of the dataset. Not a viable reframing of the original problem.

     Why Option 1:
     - It addresses both ceilings simultaneously (joint optimization + unified representation)
     - It removes the specific failure mode that has killed every adaptation (parameter sharing leaking gradients across heads with conflicting objectives)
     - It's a clean architectural reset; no fix21 baggage to carry forward
     - The implementation cost (1.5–2 days) is comparable to what was already spent on fix22-25
     - The diagnostic signal will be unambiguous: if a single-model end-to-end can't reach 0.32, no architecture in this family can, and the project pivots to other problems entirely

     Concrete plan for Option 1:

     1. Phase 1: design the architecture (half day)
       - Single UnifiedProposalAlignerNet model
       - Encoder: reuse pred_v2.x encoders (signal_proj, cand_proj, mem_proj, plus their _emb_proj siblings)
       - K_max=3 learnable slot queries (like fix5's DETR-lite)
       - Slot query refinement: cross-attention over spans + memory, self-attention across slots (also like fix5)
       - Per-slot heads: use, span_pointer, is_bridge (proposer-equivalent)
       - Pairwise heads from slot queries: edge_exist[K, K], edge_rel[K, K]
       - Slot × memory heads: mem_kind[K, M], mem_rel[K, M]
       - Synth template-arg pointers: for synthesis slots, predict which memory node is the dst (mixed_add_link) or which two memories are text_a/text_b (multi_region_attach)
       - Output text reconstruction: deterministic application of templates using model's predicted pointers
     2. Phase 2: training script (half day)
       - Reuse PredDataset components for input encoding
       - New ProposalAlignerBatch / collate function
       - Combined loss: per-slot use + span_pointer + is_bridge + pairwise edge_exist + edge_rel + per-(slot,mem) mem_kind + mem_rel + commit + template-arg-pointer loss
       - Reuse decoder constraints (anti-symmetry + transitive reduction) from PRED-v2
     3. Phase 3: end-to-end evaluation (existing infra works)
       - Single forward pass produces a complete goal
       - Compare to gold under both row_complete (lenient) and text_faithful_row_complete (strict)
       - Per-task breakdown to identify which row types benefit most from joint training
     4. Phase 4: decision point
       - If row_complete > 0.30 lenient: the unified model is the new active baseline; iterate on this architecture
       - If row_complete is 0.20–0.30 lenient: the architecture is plausibly viable but needs work; consider data scaling next
       - If row_complete < 0.20 lenient: unified models don't fit this problem at this data scale; pivot to a fundamentally different problem formulation or accept v1 and ship

     ---
     Critical files to be modified

     New files:
     - unified_proposal_aligner_model.py — UnifiedProposalAlignerNet, ~500 lines
     - train_unified_v1.py — joint training script, ~400 lines
     - eval_unified_v1.py — unified eval producing both lenient and strict metrics, ~300 lines

     Existing files referenced (read-only):
     - pred_model.py — encoder patterns to reuse (signal_proj, cand_proj, mem_proj, _maybe_emb_proj, LOGIT_MASK_VALUE)
     - proposer_model.py — slot query / DETR-lite attention pattern (encode() method, lines 207–215)
     - train_pred_v1.py — Dataset construction, decode_edge_predictions (anti-symmetry + transitive reduction)
     - train_proposer_v1.py — ProposerDataset for input format
     - synthesize_node_text.py — template definitions for output text reconstruction
     - eval_proposer_roundtrip.py — strict / lenient metric definitions
     - eval_proposer_v1.py — text_faithful_row_complete metric

     Critical existing utilities to reuse:
     - pred_model.py:_maybe_emb_proj — zero-init projection helper
     - train_pred_v1.py:decode_edge_predictions — vectorized anti-symmetry + transitive reduction
     - train_pred_v1.py:build_edge_hard_negative_mask — if hard negatives needed during training
     - synthesize_node_text.py:_memory_text, _best_matching_memory_id, clean — template synthesis primitives

     ---
     Verification plan

     End-to-end verification has three layers:

     1. Component sanity (during development):
       - Forward pass: confirm shapes match expected (no silent broadcasting bugs)
       - Loss components: each loss term should be nonzero on first batch, decrease over first 100 steps
       - Unit test: synthesis-template-arg-pointer head's output, when fed gold pointers, should produce gold synthesized text byte-identical
     2. Training-time diagnostics (each epoch):
       - Per-component loss tracking (use/span/edge/mem/text-arg)
       - Per-task row_complete on val
       - Per-task text_faithful_row_complete on val
     3. End-to-end comparison (after best checkpoint):
       - Run on pred_v1 val
       - Compare to current pipeline (fix5 + synth + fix21) on:
           - Global row_complete (lenient): 0.1745 → ?
         - Global text_faithful_row_complete (strict): 0.0000 → ?
         - Per-task row_complete: covered, long_decompose, mixed_add_link, multi_region_attach
       - Compare to oracle ceiling (0.3208 lenient):
           - If unified beats 0.32: real architectural win
         - If unified is in 0.25–0.32: viable, needs iteration
         - If unified < 0.25: pivot
     4. Pre-decision sanity check before kicking off training:
       - Generate 3 forward passes on randomly-initialized model, manually trace outputs to confirm template-arg pointers produce coherent synthesized text
       - Verify the trainable parameter count is reasonable (likely ~3M total, similar to PRED-v2 + proposer summed)
       - Run one epoch on a 20-row subset to confirm training infrastructure works before launching the full run

     ---
     Open questions before implementation

     1. Output text reconstruction location: should the template synthesis happen inside the model's forward (with differentiable template-arg pointers) or outside as a post-processing step (model outputs
     pointers, deterministic synthesis happens after)? Post-processing is simpler and matches the current synthesizer; inside-model is more end-to-end but adds complexity. Recommendation: post-processing, like
      the current pipeline.
     2. Should the unified model also predict commits and covered_mappings? Yes — these are part of the goal structure. They were predicted by the aligner alone; in the unified model they're additional output
     heads.
     3. Initialization strategy: train from scratch, or warm-start individual heads from fix5/fix21? Recommendation: train from scratch. Warm-starting risks inheriting the same local-optimum brittleness we've
     been trying to escape.
     4. K_max for unified model: stay at K_max=3 (matches data: every row has 2 or 3 slots). No reason to change.
     5. Are there pred_v1 rows that this formulation can't represent? The unified model produces the same output structure as the current pipeline, so anything pred_v1 expresses should be representable. Worth
     confirming with a smoke test on 5 representative rows before training.

● Plan is approved and saved. Sticking with your established pattern — you've been doing implementation, I've been planning. Want to take it from here, or do you want me to start on Phase 1 (architecture
  design for unified_proposal_aligner_model.py)?

  A few things worth flagging before you (or I) start coding:

  1. Trainable parameter budget check: rough estimate for the unified model is ~3M params (encoders ~1.5M reused from pred_v2, attention layers ~0.5M, all the heads combined ~1M). Comparable to fix21 + fix5
  summed. No surprises expected.
  2. The smoke test before training (open question #5 in the plan) is worth doing first: pick 5 representative pred_v1 rows (one of each task type plus an edge case), manually trace what the unified model
  should output to fully reconstruct the gold goal, confirm every output field maps to a model head. If any gold goal field doesn't have a corresponding head, the architecture is incomplete and worth catching
  before writing code.
  3. One detail not in the plan but worth deciding upfront: should the model see the proposer-style canonical-ordered slot positions as input bias, or learn slot semantics entirely from scratch? The current
  proposer uses positional slot embeddings (slot 0 = earliest anchor, slot 1 = middle, slot 2 = latest). Inheriting that convention makes the synthesis template-arg pointers easier (slot 0's predicted span is
  the source for mixed_add_link). Without it, the model has to learn slot semantics. I'd vote for keeping canonical-ordered slots — same convention as fix5.
