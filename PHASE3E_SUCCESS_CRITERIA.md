# Phase 3E â€” definition of success

**Status:** acceptance criteria. The design ships when every gate in Â§2 is
passing, every reasoning archetype in Â§3 matches its expected shape, and every
invariant in Â§4 holds for 30 consecutive sessions.

**Parent docs:**
- `PHASE3E_REASONING_SUBSTRATE_V2.md`
- `PHASE3E_PROGRESS.md`

**Date:** 2026-05-22

---

## 0. Definition

3E succeeds when:

1. **Quality:** v2 beats the baseline on a frozen, dual-judged 20-task suite.
2. **Cost:** v2 does not pay a measurable latency or call-count penalty for
   that quality win.
3. **Compounding:** repeat-domain sessions cost strictly fewer calls than
   cold sessions on the same task. This is the only test that the
   *substrate* (not the prompt protocol) is doing the work.
4. **Discipline:** every recursion event has a named gap, every promoted
   signal has a traceable provenance, and every persisted session replays
   deterministically from its journal.
5. **Failure containment:** the negative-control suite rejects 100% of
   known-bad answers, and no failure mode in Â§5 has occurred in the last
   30 sessions.

If 1 + 2 are true but 3 is not, **3E is a better prompt protocol, not a
substrate.** Ship it, but don't start 3F until 3 is also true.

If 3 is true but 1 + 2 are not, **the substrate compounds but the surface
is wrong.** Keep iterating on composer/checker; don't touch graph mechanics.

---

## 1. Suites that must exist before measurement

Frozen and committed to version control before the next benchmark run.

| Suite | Size | Purpose |
|---|---:|---|
| `bench/core_20.json` | 20 tasks | the 3E-4 quality + cost gate |
| `bench/cold_warm_adversarial.json` | 4 tasks Ã— 3 runs | compounding gate (run 1 cold, runs 2â€“3 warm) |
| `bench/negative_controls.json` | â‰¥ 40 cases | known-bad answers per checker plugin |
| `bench/replay_corpus/` | â‰¥ 50 sessions | persisted journals for deterministic replay |
| `bench/recursion_fuzz.json` | â‰¥ 100 cases | malformed deltas, deep gaps, repair stress |

The judge is **dual:** rubric (synonym-set, deterministic) + LLM-judge with
fixed criteria prompt. A task counts as `judge_passed` only when both agree.
Per-judge pass rates are also reported, but only the agreement number gates.

---

## 2. Quantitative gates

All gates measured on `bench/core_20.json` unless otherwise noted.

### 2.1 Quality

| Metric | Gate |
|---|---|
| `v2.judge_passed_agreed` | **â‰¥ baseline.judge_passed_agreed + 2** |
| Cases where v2 fails but baseline passes | **0** |
| Per-task regression on previously-passing v2 tasks | **0** |
| Rubric/LLM judge disagreement rate | **â‰¤ 20%** (otherwise judges aren't trustworthy) |

### 2.2 Cost

| Metric | Gate |
|---|---|
| `v2.mean_llm_calls` | **â‰¤ 1.20** |
| `v2.mean_llm_calls` on tasks baseline also resolves in 1 call | **â‰¤ 1.10** |
| `v2.median_elapsed_sec` per task | **â‰¤ 1.5Ã— baseline median** on the same edge model |
| `v2.p95_elapsed_sec` per task | **â‰¤ 15s** (a single hard cap so one outlier can't hide) |

### 2.3 Compounding (the real headline)

Measured on `bench/cold_warm_adversarial.json`. Each task is run 3 times against the
same persisted graph; first run is cold, runs 2â€“3 are warm. The default runner
(`run_phase3e_benchmark.py --cold-warm`) already uses this suite.

| Metric | Gate |
|---|---|
| `mean_llm_calls(warm) / mean_llm_calls(cold)` | **â‰¤ 0.70** |
| `judge_passed_agreed(warm)` | **â‰¥ judge_passed_agreed(cold)** |
| Warm runs that activated â‰¥ 1 prior-session signal | **â‰¥ 80%** |
| Warm runs where activated prior signals appeared in the resulting delta | **â‰¥ 50%** |

If warm runs don't get cheaper at maintained quality, the substrate isn't
compounding â€” it's just inert storage. This is the hardest gate and the one
that justifies the whole architecture.

### 2.4 Discipline

| Metric | Gate |
|---|---|
| Recursion events with no named gap in parent delta | **0** |
| `repair â†’ repair` chains observed | **0** |
| Replay determinism (same journal â†’ same outcome over 50 sessions) | **100%** |
| Skimmed deltas that promoted to reusable signals | **0** |
| Canonical gap-id collisions on a sampled 100 pairs | **0** semantic collapses |

### 2.5 Failure containment

Measured on `bench/negative_controls.json`.

| Metric | Gate |
|---|---|
| Negative controls rejected by their target plugin | **100%** |
| Negative controls that produced a final answer (not `failed`) | **0** |
| Recursion budget exhausted on a malformed-delta fuzz case | **â‰¤ 5%** of cases |

---

## 3. What a passing reasoning trace looks like

Every persisted session must match one of these three archetypes. Anything
else is a bug.

### 3.1 Archetype A â€” trivial direct answer (target: 80% of tasks)

```text
session:
  steps: 1
  total_llm_calls: 1
  total_elapsed_sec: â‰¤ 4s

step_0:
  status: resolved
  packet:
    active_signals: 2â€“6
    hard_constraints: â‰¥ 1 derived from task statement
  delta:
    parsed (not skimmed)
    decisions: â‰¥ 1
    constraints_honored: âŠ‡ packet.hard_constraints
  checker:
    passed: true
    confidence: â‰¥ 0.80
    plugins_fired: â‰¥ 2
```

Pass condition: one call, parseable delta, checker satisfied, hard
constraints honored verbatim.

### 3.2 Archetype B â€” adversarial recursion (target: â‰¤ 15% of tasks)

```text
session:
  steps: 3   (parent, child, parent-resumed)
  total_llm_calls: 3
  total_elapsed_sec: â‰¤ 15s

step_0 (parent):
  status: need_info
  delta.missing.question: present and parseable
  delta.gaps: â‰¥ 1 canonical gap id

step_1 (child):
  parent_step_id: step_0
  focus: derived from step_0.missing.question
  status: resolved
  delta.evidence: â‰¥ 1 entry

step_0_resumed:
  parent_step_id: null  (same step_id as step_0)
  packet.active_signals: includes child's evidence
  status: resolved
  checker: passed
```

Pass condition: child fires only on a named gap, child resolves, parent
resumes with child evidence visible in its packet, total â‰¤ 3 calls.

### 3.3 Archetype C â€” checker reject â†’ repair (target: â‰¤ 5% of tasks)

```text
session:
  steps: 3   (parent, repair child, parent-resumed)
  total_llm_calls: 3

step_0 (parent):
  status: resolved   (model thinks it's done)
  checker.passed: false
  checker.violations: â‰¥ 1 named

step_1 (repair):
  focus: derived from checker.violations
  status: resolved
  delta.repair: â‰¥ 1 entry

step_0_resumed:
  packet: includes repair signal
  status: resolved
  checker: passed
```

Pass condition: repair fires only on a hard-fail checker violation; repair
child cannot itself spawn another repair child; parent must close cleanly
or fall back to best-effort.

### 3.4 What must NOT appear anywhere

- A step with `mode: focus | plan | execute | check | revise | finalize`.
  3E deleted modes; if the trace contains them, the wrong loop ran.
- A step whose `result.status = resolved` and `checker.passed = false` with
  no follow-up repair step.
- A child step whose `parent_step_id` is null.
- A `delta` block whose `constraints_honored` is empty when
  `packet.hard_constraints` is non-empty.
- More than 3 violations on a single checker pass (should short-circuit
  to `failed` per Â§4.1 of the design).

---

## 4. Substrate health invariants (multi-session)

Measured over a rolling 30-session window. Each invariant ships with an
assertion in the audit pipeline.

| Invariant | Target | Why it matters |
|---|---|---|
| Signal kind distribution | constraint 20â€“30%, decision 20â€“30%, evidence 15â€“25%, risk 10â€“20%, gap 5â€“15%, repair 2â€“8%, procedure â‰¤ 5% | drift means the loop is regressing toward old failure modes |
| Median age of activated signals | bimodal: cluster < 24h + cluster > 7d | flat distribution = no consolidation; old-only = no fresh evidence |
| Canonical gap-id reuse rate | â‰¥ 30% of gap_ids appear in â‰¥ 2 sessions | proves dedupe is working |
| Prior-failure-signal recall | â‰¥ 70% when same gap_id recurs | proves the graph remembers what it couldn't do |
| `produced_by` breakdown for promoted signals | 0% `regex_fallback`, â‰¥ 80% `llm_delta`, rest `checker` / `consolidation` | skimmed deltas must not promote |
| Decision-conflict count | â‰¤ 1 active pair per 30 sessions | passive decay isn't enough; active deprecation needed if this rises |
| Writer attribution on legacy node types | drops to ~0 within 30 days of 3E-4 ship | gates storage collapse |

---

## 5. Hard failure criteria (design is wrong, not just incomplete)

If any of these is true after 3E-4 ships and runs for 30 days, **stop and
redesign before continuing to 3F**:

1. Cold/warm call-count ratio does not drop below 0.90 on `cold_warm_adversarial`.
   The substrate is not compounding; it's a journal with extra steps.
2. Negative-control rejection rate falls below 95% on any plugin.
   The checker isn't a safety net; it's theater.
3. `repair â†’ repair` chain count > 0. The recursion-explosion guarantee
   from Â§4.1 of the design has leaked.
4. Replay determinism falls below 99%. The journal is lossy, which kills
   the entire 3F-Î² training-data story.
5. v2 produces a final answer on more than 5% of negative-control inputs.
   Fail-open default is too permissive.

These are not "tune the thresholds" failures. They invalidate load-bearing
assumptions and force a redesign.

---

## 6. Required telemetry per session

Every persisted session journal must contain these fields. Missing fields
on any session fails the discipline gate (Â§2.4).

```yaml
session_summary:
  session_id:               str
  task_id:                  str
  enable_substrate_v2:      bool
  total_llm_calls:          int
  total_tokens_used:        int
  elapsed_sec:              float
  outcome:
    judge_rubric_pass:      bool
    judge_llm_pass:         bool
    judge_agreed:           bool
    failure_mode:           str | null
  steps:
    - step_id, parent_step_id, status, depth, llm_calls, tokens
  packets:
    - cache_key, cache_hit, active_signal_ids, hard_constraints
  deltas:
    - delta_status, produced_by, parse_error, constraints_honored
  checks:
    - plugin_names, passed, confidence, violations
  signals_touched:
    - id, kind, activation_score, age_days, source_step
  failure_modes_observed: list[str]   # references the Â§5 list
```

This is the schema the dashboard reads. If the schema slips, all the gates
above become unverifiable.

---

## 7. Decision matrix at 3E-4 close-out

After the bench runs, classify the result and act accordingly. No
ambiguity, no "let's keep tuning indefinitely."

| Quality gate (Â§2.1) | Cost gate (Â§2.2) | Compounding gate (Â§2.3) | Verdict | Next move |
|:---:|:---:|:---:|---|---|
| âœ“ | âœ“ | âœ“ | **3E succeeded** | Begin storage collapse â†’ 3F-Î± |
| âœ“ | âœ“ | âœ— | Better prompt, no substrate | Ship v2, **do not start 3F**; investigate why warm runs don't compound |
| âœ“ | âœ— | âœ“ | Substrate works but too slow | Profile composer/checker; cap shaper calls; do not redesign |
| âœ— | â€” | â€” | Surface still wrong | Iterate composer/coverage; **do not touch graph mechanics** |
| âœ— on Â§5 | â€” | â€” | Architectural failure | Redesign before continuing |

---

## 8. What this document is not

- Not a finish line for the project. 3F and 3G have their own success
  criteria (to be written when those phases start).
- Not a substitute for human review of failed traces. Gates catch the
  things you'd measure; humans catch the things you didn't know to
  measure.
- Not negotiable mid-benchmark. If a gate seems wrong, change it *before*
  the run, not after. Goalpost-moving invalidates the comparison.
