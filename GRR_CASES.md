# GRR CASE SPEC — what the model SHOULD do, step by step, per case

> The operating contract for the LONG training run on real open-source data. Each case: trigger →
> expected step-by-step behavior → what gets written where → the metric that says it worked.
> Grounded in the audited code paths (algo_grr_loop / algo_dsl_trm / algo_lm_proposer /
> algo_lm_author / algo_mbpp_prep), 2026-07-14. Annotate freely.

## The ladder (one task, cheapest rung first)

```
A recall -> B search -> C LM-propose -> D LM-author -> (fail: logged, becomes training signal)
each rung: same GATE (dense verify), same BANK (health-gated graph write), same provenance stamp
```

---

### CASE A — task matches consolidated knowledge (recall / amortized)
- **trigger**: TRM decode of the goal embed produces a pipeline that passes the gate (1 verify).
- **steps**: embed text (mpnet, cached) → decode (head distributions ~1.00 on consolidated fams) →
  realize (code CALLS atoms, never inlines) → resolve deps via graph walk → verify (two disjoint
  input sets).
- **writes**: nothing (already known).
- **metric**: share of tasks solved at rung A rises over the run; verifies-to-solve → 1.0.
- **expected cost**: ~ms, no LM call.

### CASE B — unseen composition of KNOWN atoms (search discovers)
- **trigger**: decode fails the gate.
- **steps**: beam+epsilon search under verify budget (net-guided ordering, depth-ascending = MDL-first;
  epsilon slots escape the prior's corner) → first fully-general hit wins → consolidate (SFT on the
  discovery + replay pool) → SLEEP banks a program node (code + SYMBOLIC pipeline + goal-region texts
  + origin=beam|epsilon + found_round) + depend edges to its atom closure.
- **writes**: program node; next encounter of this family = CASE A.
- **metric**: per-origin provenance→reuse table; search cost collapses after consolidation (measured
  54→5 verifies).
- **failure sub-case B1**: budget exhausted → escalate to C. NOT an error.

### CASE C — composition beyond search (language understanding needed)
- **trigger**: search exhausts budget (measured wall: deep prefixes ~5e-5/search survival).
- **steps**: LM prompted with task TEXT + DSL grammar + atom vocabulary (NEVER the reference, NEVER
  the oracle) → K sampled candidate pipelines → parse/validate → same gate, MDL-first → hit →
  consolidate + bank (origin=lm).
- **writes**: program node origin=lm; the LM is never needed again for this family (measured: rounds
  after LM-discovery run pure decode).
- **metric**: method-tier vs intent-tier first-encounter solve = the reasoning-vs-translation delta
  (3B measured: 11→7 discoveries, reuse 91%→57%). Bigger teacher must beat it HERE first.
- **failure sub-case C1**: no candidate passes → escalate to D (if authoring enabled for the domain).

### CASE D — no atom exists (out-of-vocabulary → invention)
- **trigger**: task needs a primitive the graph lacks (measured baseline: 2% without authoring).
- **steps**: prompt = task text (+ optional purpose-line ads, SOFT directive — bare-sig ads with hard
  directives are BANNED: measured −16pp and caused the accuracy decline) → k samples →
  repair_code (compile-trim prose-in-fence + entry-name case alias; gate still decides) → DENSE gate
  (original asserts + full plus script, subprocess) → bank solution (origin=lm_author, task text =
  retrieval key, depend edges to called atoms) + model's STORE helpers as their OWN atoms
  (origin=lm_author_helper — the reuse-granular units).
- **sub-cases**:
  - **D1** solved sample 1 → bank. Expected majority.
  - **D2** solved sample k>1 → bank. Best-of-k-with-verifier is the system's legitimate edge.
  - **D3** passes original asserts, fails plus script (`plus_only_fail`) → **NOT banked**. This is the
    gate doing its job (~5% measured). Must NOT rise during training (gate-integrity metric).
  - **D4** all samples fail asserts → not banked; failure logged with taxonomy + graph-size context →
    **becomes the STaR training signal** (a failure is data, never silently dropped).
  - **D5** model misreads the spec → looks like D4 in aggregates; only the RAW DUMP distinguishes it.
    Rule: when a number disappoints, dump raw BEFORE comparative experiments (this rule recovered
    +11pp: 5 of 9 "model failures" were harness bugs).
- **metric**: solve% vs the 2% no-authoring baseline (now 78%); assert_fail share = the true
  capability residual (~17%) — the ONLY number training is allowed to claim.

### CASE E — duplicate / already-known task
- **trigger**: `impl_<name>` already in graph, or behavioral dedup match.
- **steps**: solve normally (cheap rungs), skip banking. Idempotent.
- **metric**: banked count grows SUB-linearly with tasks as the corpus repeats structure.

### CASE F — bad/malformed data record
- **trigger**: record whose own reference solution fails its own tests, or unparseable fields.
- **steps**: prep validation drops it BEFORE the loop (0 tolerated; 378/378 clean on MBPP+).
- **long-run (GRR-15)**: the MODEL normalizes FORM (entry-point, canonical text, its own paraphrases
  for goal-region coverage); the gate keeps TRUTH (tests come from the corpus; the model NEVER writes
  its own grader — anti-cheat at corpus scale).

### CASE G — knowledge conflict / update (reconsolidation) — NOT BUILT (task #71)
- **expected when built**: new info correlates with existing nodes → LM merges old+new (token-level,
  LM = the merge engine) → behavioral A/B gate (updated node must not DROP solve) → non-destructive
  write (version edge, old kept). OUT OF SCOPE for long-run v1.

### CASE H — sleep (offline, between epochs)
- **steps**: replay-SFT the pool → prune Δ≈0 nodes → behavioral dedup of near-identical helpers →
  ABSTRACT shared skeletons (fuzz-equivalence + MDL gated) → **hygiene: strip dead trailing
  statements from banked code** (found: a mangled STORE line `store_row_sums: lambda` survived
  because bare annotations parse — dead code, gate-safe, but memory should be clean).
- **status**: replay+bank wired; prune/abstract/dedup exist as validated modules, NOT yet wired into
  the author domain. Wire before the long run's epoch 2.

---

## TRM's ROLE (clarification, user question 2026-07-14)

- Rung A: TRM **is** the decoder — amortized program synthesis (goal → program, 1.00-confidence heads
  on consolidated families). Not traversal: the point is making the LM unnecessary after first discovery.
- Rung B: TRM is the **search prior** — beam candidates ordered by its logprobs (verifies 54→5).
- Rungs C/D: TRM is **inert. It does NOT help the LM reason or decode.** The only TRM↔LM coupling is
  graph-mediated and OFFLINE (LM discovers → gate → graph → TRM consolidates between rounds). Within a
  single solve they never talk; the ladder is fallbacks, not collaborators.
- Possible couplings (none built, deliberately): (1) TRM sketch → LM prompt hints (symbolic, z-wall-safe
  — but the ads lesson applies: unconfident hints can cost points); (2) TRM as ad-RANKER for the author
  (Fix C #45 reborn — most defensible, relevant when purpose-ads prove ≥ off); (3) TRM reranks LM
  pipeline proposals pre-verify (rung C only). Deferred: the long run's bottleneck is rung D, where TRM
  structurally can't participate until authored atoms become DSL-composable (programs-as-atoms at scale).

## THE LONG RUN (design)

- **corpus**: MBPP+ (378, epoch-0 sanity: expect ≈78% pre-training) → **APPS-intro class (~2-3k
  problems with test cases)** as the real large corpus, prepped through the same
  normalize→VALIDATE→cache pipeline (GRR-15 self-ingestion when built).
- **outer loop (STaR epochs)**: run corpus → collect gate-VERIFIED solutions → LoRA-SFT the student
  3B on (prompt → verified code) → re-run → repeat. Machinery: train_star_mg (exists).
- **KPIs per epoch** (in priority order):
  1. assert_fail share ↓ (the only claimable training win)
  2. plus_only_fail flat (gate integrity — if it RISES, the model is learning to overfit tests)
  3. solve curve flat across the run (no memory-growth harm)
  4. cross-task reuse events ↑ (the compounding KPI — still 0 on MBPP, needs helper granularity +
     task clustering + possibly purpose-ads)
  5. banked atoms sub-linear (dedup working)
- **checkpointing (molab resets ~4h — HARD REQUIREMENT)**: commit LoRA + graph + failure logs every
  N=50 tasks; loop must RESUME (skip banked/solved). A lost run is a lesson we already paid for.
- **deployment constraint (permanent)**: everything trained must fit the ≤6GB cloud-less stack
  (student 3B@4bit + mpnet + TRM + graph). Big teachers only ever produce (a) graph content and
  (b) verified traces for student SFT.
