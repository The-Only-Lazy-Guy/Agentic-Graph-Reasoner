# Post-3E: signal usefulness investigation plan

**Status:** archived — superseded by 3E closure on 2026-05-23. No further work planned.

**Duration estimate:** 3–5 days (planned, not executed).

**Duration estimate:** 3–5 days.

**Scope:** signal quality and selection only. No graph schema redesign, no procedure/session-object lane changes, no broad answer-surface retuning.

**Parent docs:**
- `PHASE3E_PROGRESS.md` — 3E closeout results and compounding diagnosis
- `PHASE3E_SUCCESS_CRITERIA.md` — compounding gate definition

---

## 1. Problem

The compounding gate fails because warm-start signals are not helping enough:

| Task | Cold calls | Warm calls | Ratio | Diagnosis |
|---|---|---|---|---|
| `payment_psp` | 1.0 | 1.7 | 1.70 | Signals **pollute**: repair and hard-violation rates rise from 0% to 30%. Prior signals (4.7 activated) contain procedural and checker-residue fragments that mislead the model. |
| `migration_zd` | 1.0 | 1.6 | 1.60 | Same pollution pattern. Signals (2.4 activated) include migration-prose fragments that trigger unnecessary checker hard violations. |
| `inventory_flash` | 1.0 | 1.2 | 1.20 | Saturated: cold is already 1 call. Not actionable. |
| `segment_beats` | 4.2 | 3.3 | 0.79 | Best candidate: ratio improves. But repair fires on every run (100% cold and warm). Prior signals (5 activated) reduce back-and-forth within repair but don't eliminate it. |

Common pattern: **signal activation and reuse work (100%), but the activated signals contain low-value content** — procedural fragments, checker-violation residue, verbose model output — that trigger unnecessary repair or dilute useful invariants.

---

## 2. Investigation steps

### 2.1 Signal-classification audit (1 day)

For each of the 4 cold-warm tasks, collect the activated prior signals from a warm run and classify each signal into:

| Category | Definition | Example |
|---|---|---|
| **Reusable invariant** | Domain constraint or invariant that holds across sessions | "Segment tree second max + count_max + sum invariants" |
| **Useful risk/constraint** | Specific hard constraint that prevents a known failure mode | "Use long long for all sums" |
| **Procedural fragment** | Step-level instruction or meta-commentary | "First, I'll write the algorithm..." |
| **Checker residue** | Violation text from a previous checker pass | "segment_tree_merge_missing: Dynamic max-subarray answer should..." |
| **Answer-surface noise** | Domain-irrelevant wording or boilerplate | "Thank you for your question..." |
| **Irrelevant** | Unrelated to the current task | Signals from a different domain entirely |

Deliverable: per-task signal-inventory table showing category mix and count.

### 2.2 Warm-run comparison (1 day)

For `payment_psp` and `migration_zd` (the two regressing tasks):

1. Compare cold vs warm packet content directly — what extra signals arrive in warm that are not in cold?
2. Check whether any prior signal directly causes the increased repair rate: does a prior violation-text signal trigger the checker on the warm run?
3. Check whether repair-child focus in warm runs contains prior-signal fragments, meaning the model is copying old prose instead of writing fresh analysis.

### 2.3 Signal-promotion policy changes (1–2 days)

Based on the audit findings:

1. **Downweight procedural fragments**: if a signal's provenance includes `produced_by=regex_fallback` or its text contains step-level meta (`I'll`, `First,`, `Let me`), reduce its activation confidence below the active-signal threshold.

2. **Suppress checker-residue signals**: signals whose text starts with `"violation:"` or matches known checker violation codes (`segment_tree_*`, `long_long_*`, `backfill_*`, etc.) should be excluded from warm-start activation. Checker violations are for the current run's checker, not for polluting the next session's signal stack.

3. **Boost compact domain invariants**: signals that are short (< 200 chars) and match domain-keyword patterns (idempotency, durable state, backfill+live-sync+verification, segment tree beats state, single-writer ownership) get a confidence bonus.

4. **Add activation scoring debug view**: a `--debug-signals` flag to `run_phase3e_benchmark.py` that prints per-signal scores, category, and whether it was activated.

### 2.4 Re-run compounding benchmark (1 day)

After signal-policy changes, re-run only the compounding suite:

```powershell
python run_phase3e_benchmark.py --tasks bench\cold_warm_adversarial.json --judge-mode dual --cold-warm --replicate 5
```

Success criteria:
- `payment_psp`: warm repair rate drops from 30% toward ≤ 10%; warm pass rate rises from 6/10 toward ≥ 8/10.
- `migration_zd`: warm repair rate drops from 30% toward ≤ 10%; warm pass rate rises from 8/10 toward ≥ 9/10.
- `segment_beats`: warm ratio stays ≤ 0.85; warm pass rate ≥ 9/10.
- No quality regression on `core_20` (v2.judge_passed ≥ baseline + 2).
- No cost regression on `core_20` (v2.mean_llm_calls ≤ 1.20).

---

## 3. Success criteria

The investigation succeeds when:

1. The signal-classification audit produces a clear category breakdown for all 4 tasks.
2. At least one policy change demonstrably reduces warm-start pollution (measured by repair rate or hard-violation rate on a regressing task).
3. The compounding ratio on at least one regressing task improves (payment or migration), and segment_beats does not regress.
4. Core-20 quality and cost are not harmed.

If no policy change improves the compounding picture after 2 iterations, the conclusion is: **signals are working structurally but the model capability gap prevents compounding on this edge model.** The next move is a stronger model baseline — diagnostic only, not a production/runtime change or a replacement for the local 4B target — to disambiguate whether the bottleneck is signal policy or model reasoning depth.

---

## 4. Non-goals

Explicitly out of scope for this investigation:

- Graph schema redesign (3F territory).
- Procedure / session-object lane changes.
- Broad answer-surface or composer retuning.
- New deterministic checker plugins.
- New benchmark suites or rubric expansion.
- Storage collapse (3F) work of any kind.
