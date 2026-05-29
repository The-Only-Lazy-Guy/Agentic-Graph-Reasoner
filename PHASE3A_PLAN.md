# Phase-3A implementation plan — trigger-signal-react meta-procedures

**Status:** plan phase. No code written yet. Plan must be approved before implementation.

**Parent docs:**
- `REASONING_ARCHITECTURE.md` — three-phase architecture (Phase 3 = meta, chunking, dynamic embeddings)
- `PHASE2_PLAN.md` / `PHASE2_PROGRESS.md` — composition substrate
- `PHASE1_PLAN.md` / `PHASE1_PROGRESS.md` — substrate foundations

**Date:** 2026-05-20

---

## 0. Executive summary

Phase 3A introduces **meta-procedures**: deterministic Python-level rules that observe substrate state, detect patterns (cycles, contradictions, budget pressure, mis-dispatches, etc.), and inject *signals* into the next prompt. The model then reacts to those signals as part of its normal reasoning — no extra LLM call needed.

The load-bearing claim: **we get LLM-level flexibility on meta-decisions at near-zero LLM cost**, because the model's next-turn output IS the reaction mechanism. We piggyback on the LLM call we were going to make anyway.

This phase explicitly **avoids** the naive design where meta-procedures are themselves dispatched via LLM. The original architecture doc treated meta-procedures as another procedure-shaped abstraction; that costs hundreds-to-thousands of tokens per session. The signal-injection design costs **zero tokens for ~80% of meta-cognition** and a tiny prompt-overhead for the rest.

Decay (Phase 2B) and macro-extraction (Phase 3B+) remain deferred.

---

## 1. Goal of Phase 3A

A working substrate where:

1. Meta-procedures fire on three hook points: **pre-iteration**, **post-dispatch**, **end-of-session**
2. Each meta-procedure has a **trigger predicate** (pure Python, runs in microseconds) and an **action** (mostly pure Python; may emit signals)
3. **Signals** are typed messages emitted by meta-actions; they flow into the next iteration's prompt as a dedicated `# System signals` section
4. The model reads the signals in its normal next-turn reasoning and adapts its `<reasoning>` / `<answer>` accordingly — **no separate LLM call** for meta-reasoning
5. Signals also persist into the session subgraph as `node_type="signal"` for replay / debugging / UI rendering
6. At least 5 concrete meta-procedures shipped, covering the most common failure modes observed in Phase 1 + 2A
7. Conservative predicates: false-positive signals would mislead the model, so triggers err on the side of precision over recall
8. No Phase 1 or 2A regression

What's NOT in Phase 3A:
- LLM-dispatched meta-procedures (the naive design)
- Macro extraction / trace mining (Phase 3B)
- Dynamic embeddings (Phase 3C)
- Decay activation (Phase 2B, still gated on real corpus data)
- Multi-session concurrency

---

## 2. File structure

Additions only. No Phase 1 / 2A files are renamed or removed.

```
graph_final/
├── reasoning/
│   ├── meta.py                           ← NEW: MetaProcedure schema + MetaPool + signal injection
│   ├── signals.py                        ← NEW: Signal schema + rendering
│   ├── reasoning_loop.py                 ← MODIFIED: 3 hook points + signal-aware prompt build
│   ├── schemas.py                        ← MODIFIED: add SignalNode (node_type="signal")
│   ├── meta_procedures/                  ← NEW package, parallel to procedures/
│   │   ├── __init__.py
│   │   ├── cycle_detector.py
│   │   ├── budget_warner.py
│   │   ├── contradiction_detector.py
│   │   ├── dispatch_miss_nudge.py
│   │   └── repeated_anchor_observation.py
│   └── tests/
│       ├── test_meta.py                  ← MetaPool + hook firing
│       ├── test_signals.py               ← Signal rendering + persistence
│       ├── test_meta_procedures.py       ← Each predicate + action in isolation
│       └── test_signal_injection.py      ← Integration: signal appears in next prompt
```

---

## 3. Schemas (`reasoning/signals.py`, `reasoning/meta.py`, `reasoning/schemas.py`)

### 3.1 Signal

```python
SignalSeverity = Literal["info", "warn", "error"]
SignalHook    = Literal["pre_iter", "post_dispatch", "end_of_session"]

@dataclass
class Signal:
    id: str                                       # short slug e.g. "cycle_detected_so_abc"
    type: str                                     # short kind e.g. "cycle_detected"
    severity: SignalSeverity
    message: str                                  # one-line natural-language summary for the model
    emitted_at_step: int                          # session step when the signal fired
    emitted_by: str                               # meta_procedure name
    related_node_ids: list[str] = field(default_factory=list)  # session_object / procedure ids it references
    metadata: dict[str, Any] = field(default_factory=dict)
    # Lifetime: per-turn (cleared after one iteration consumes it) vs sticky
    # (persists across iterations until resolved). Errors are sticky by default.
    sticky: bool = False
    # Once-per-session debounce: if True, the same (type, related_node_ids)
    # tuple won't re-fire even if the predicate keeps matching.
    once: bool = False
```

### 3.2 MetaProcedure

```python
@dataclass
class MetaProcedure:
    id: str
    name: str
    purpose: str                                  # one-line documentation
    fires_on: SignalHook
    # Predicate: takes a snapshot of session state + budget + dispatch list.
    # Returns False (no fire) or a list of Signal objects to emit.
    # MUST be deterministic and side-effect-free.
    predicate: Callable[["MetaContext"], list[Signal]]
    # Optional action: runs after predicate returns >0 signals.
    # Receives the same context and the emitted signals.
    # May mutate substrate (rare); MUST NOT call LLM.
    action: Optional[Callable[["MetaContext", list[Signal]], None]] = None
    # If once=True, the meta-procedure can fire at most one time per session
    # for any given (type, related_node_ids) tuple.
    once_per_session: bool = False
```

### 3.3 MetaContext

A read-mostly snapshot the predicate / action sees. Avoids passing 6+ arguments individually.

```python
@dataclass
class MetaContext:
    session: SessionSubgraphController            # read-only by convention
    budget: BudgetTracker                         # read-only
    dispatch_outcomes: list[DispatchOutcome]      # cumulative
    raw_outputs: list[str]                        # per-iteration model outputs so far
    anchor_ids: list[str]
    current_iteration: int
    previous_signals: list[Signal]                # signals from prior hook fires in same session
```

### 3.4 SignalNode (for session subgraph persistence)

A new node_type ("signal") added to `schemas.NodeType`. Each emitted Signal lands in the session subgraph as a node so it persists, is replayable, and the front-end UI can render it alongside session_objects.

```python
# In schemas.py, NodeType gets one more literal:
NodeType = Literal[..., "signal"]
```

A `Signal` serializes into a node dict via `Signal.to_dict()` + `from_dict()`.

---

## 4. Trigger-signal-react mechanics

### 4.1 Per-iteration flow (the load-bearing diagram)

```
ITERATION N:

  context = build_meta_context(session, budget, ...)
  signals_this_iter = []

  # Hook 1: pre-iteration
  for mp in meta_pool.fires_on("pre_iter"):
    new_signals = mp.predicate(context)
    if new_signals:
      if mp.action: mp.action(context, new_signals)
      signals_this_iter.extend(new_signals)

  # Prompt build (NEW: pass signals)
  prompt = _build_prompt(req, ..., active_signals = signals_this_iter + sticky_signals)

  output = llm_call(prompt)
  raw_outputs.append(output)

  # Standard dispatch
  matches = dispatcher.scan(output)
  for match in matches:
    outcome = dispatcher.invoke(match, session, llm_call, budget=budget)
    dispatch_outcomes.append(outcome)

  # Hook 2: post-dispatch
  for mp in meta_pool.fires_on("post_dispatch"):
    new_signals = mp.predicate(context)
    if new_signals:
      if mp.action: mp.action(context, new_signals)
      signals_this_iter.extend(new_signals)

  # Persist non-once-fired signals to session subgraph as "signal" nodes
  for sig in signals_this_iter:
    session.subgraph.nodes[sig.id] = sig.to_dict()
```

```
END OF SESSION:

  # Hook 3
  for mp in meta_pool.fires_on("end_of_session"):
    new_signals = mp.predicate(context)
    if new_signals:
      if mp.action: mp.action(context, new_signals)
    # End-of-session signals also persist as nodes
```

### 4.2 Signal lifetime semantics

- **`sticky=False`** (default for info/warn): signal is rendered into the next prompt once, then dropped from `active_signals`. Still persisted to the session subgraph for replay.
- **`sticky=True`** (default for severity=error): signal stays in `active_signals` across iterations until something resolves it (currently: signal never resolves automatically — relies on a separate "signal_resolved" meta-procedure firing, which is out of scope for v1).

### 4.3 Predicate-and-action contract

- **Predicates MUST be**: pure (no I/O, no LLM), side-effect-free, fast (<1ms target), deterministic (same context → same output).
- **Predicates MAY**: read any substrate state including budget counters and prior signals.
- **Actions MAY**: emit signals, mutate substrate state (add nodes, add edges, set deprecated flags), call other Python helpers. Actions MUST NOT call the LLM.
- **If a predicate raises**: log via Python logging, skip that meta-procedure for this hook fire. Other meta-procedures still run. This keeps the meta layer fault-tolerant.

### 4.4 Conservative predicates (precision over recall)

Because signal false-positives mislead the model, predicates target HIGH-PRECISION patterns:
- "Same procedure + same args called **3+** times" not "more than twice" — clearly cyclic
- "Two session_objects with **opposing boolean** verdicts on the same instance_description" not "any disagreement" — clearly contradictory
- "Budget at **75%+**" not "50%+" — clearly approaching exhaustion

Recall can be improved later; v1 prefers silence to noise.

---

## 5. Hook points in `reasoning_loop.py`

Three insertion points (each minimal):

```python
# Before the LLM call this iteration:
signals_this_iter = meta_pool.run_hook("pre_iter", context)
sticky = [s for s in carrier_signals if s.sticky]
active_signals = sticky + signals_this_iter

prompt = _build_prompt(req, graph, anchor_ids, procedure_pool, dispatch_outcomes,
                      iteration=iteration, signals=active_signals)

# After dispatch this iteration:
signals_this_iter += meta_pool.run_hook("post_dispatch", context)

# Persist all signals fired this iteration:
for sig in signals_this_iter:
    session.subgraph.nodes[sig.id] = sig.to_dict()

# Update carry-over for next iteration:
carrier_signals = [s for s in active_signals + signals_this_iter if s.sticky]
```

And at the very end of `run_reasoning()`:

```python
signals_at_end = meta_pool.run_hook("end_of_session", context)
for sig in signals_at_end:
    session.subgraph.nodes[sig.id] = sig.to_dict()
```

Total integration: ~50 lines in the reasoning loop.

---

## 6. Signal rendering in the prompt

A new section in `_build_prompt`, inserted between dispatch results and the directive:

```
# System signals (deterministic detectors)

- WARN  cycle_detected: VerifyShortestPath was invoked twice with identical args. Further invocations suppressed.
- INFO  budget_at_75pct: 7 of 10 LLM calls used. Consider wrapping up reasoning soon.
- ERROR contradiction: VerifyAlgorithmPreconditions and VerifyNonNegativeEdges produced conflicting verdicts (safe_to_apply true vs false).
```

Three severity prefixes: `ERROR` (sticky), `WARN`, `INFO`. The model has no special instruction telling it what to do with each — the directive just adds:

> "If the System signals section reports anything, address those concerns in your reasoning before answering. ERROR signals especially must be acknowledged."

This single directive line is enough to make the model read+react. We don't need separate sub-prompts or hand-crafted reactions.

When `active_signals` is empty, the section is omitted entirely (no empty header).

---

## 7. The five initial meta-procedures

### 7.1 `CycleDetector`
- **Hook**: `post_dispatch`
- **Trigger**: count `(procedure_id, args_text)` tuples across all dispatch_outcomes; any tuple with count ≥ 3 fires
- **Action**: emit `WARN cycle_detected` signal naming the offending procedure + args. Add a `failure_pattern` node to the session subgraph linking back to the procedure (for consolidation to consider promoting if cited across sessions).
- **Once per session per tuple**: yes

### 7.2 `BudgetWarner`
- **Hook**: `pre_iter`
- **Trigger**: any of `llm_calls / fan_out_max_per_step / tokens` at ≥ 75% of cap
- **Action**: emit `INFO budget_at_<pct>` signal listing which axis and current usage. **Sticky=False** (per-iter only — the next iteration will re-check).
- **Once**: no — fires every iteration the threshold is met

### 7.3 `ContradictionDetector`
- **Hook**: `post_dispatch`
- **Trigger**: scan session_objects for any pair where:
  - Both share a `parent_object_id` (siblings in the call tree), AND
  - Both have a `safe_to_apply` (or equivalent boolean verdict) field, AND
  - The verdicts disagree
- **Action**: emit `ERROR contradiction` signal naming both session_object ids + their conflicting verdicts. **Sticky=True**.
- **Once per session per pair**: yes

### 7.4 `DispatchMissNudge`
- **Hook**: `post_dispatch`
- **Trigger**: model's reasoner output contains an `apply_intent` or `invoke` phrase that DIDN'T resolve to any procedure (silent miss in `dispatcher.scan`). Concretely: scan the raw output for the pattern but check if the matched name resolved.
- **Action**: emit `INFO dispatch_miss` signal naming the unmatched procedure name + listing the procedures that ARE available. Model can then retry with a correct name next iteration.
- **Once per session per name**: yes

### 7.5 `RepeatedAnchorObservation`
- **Hook**: `pre_iter` (from iteration 2 onward)
- **Trigger**: anchor retrieval returned the same top-K set as the previous iteration (token-equality on the sorted anchor_ids list)
- **Action**: emit `INFO repeated_anchors` signal suggesting the model reformulate the question or invoke a procedure (since direct anchor-based reasoning isn't producing new context). **Sticky=False**.
- **Once**: no — could legitimately fire each iteration if the model keeps not progressing

---

## 8. Test plan

### Unit tests (per sub-phase)

- `test_signals.py` — Signal dataclass round-trip, signal-to-node serialization, severity ordering
- `test_meta.py` — MetaPool registration, hook iteration, exception handling in predicates, once-per-session debounce
- `test_meta_procedures.py` — each predicate fires on the right state, doesn't fire on the wrong state, action emits expected signal
- `test_signal_injection.py` — signals appear in the next iteration's prompt; model output that mentions a signal is preserved

### Integration tests

- **CycleDetector end-to-end**: scripted LLM that invokes the same procedure 3+ times → assert WARN signal fires, prompt of iteration N+1 contains the signal text
- **ContradictionDetector end-to-end**: scripted scenario where two children of one composer produce contradicting verdicts → assert ERROR signal fires, sticky across iterations
- **Signal-persistence round-trip**: a session with 2-3 signals persists, reloads, and the signal nodes are intact

### Real-LLM smoke (manual, user-side)

Same harness as Phase 2A's 2.11. Pick a question where one of the meta-procedures should reasonably fire — e.g., a deeply recursive composition that triggers BudgetWarner.

---

## 9. Phased build order

### 3.1 — Signal + MetaProcedure schemas
Build `signals.py` + `meta.py` core dataclasses. Round-trip JSON tests. SignalNode addition to schemas.py.

### 3.2 — MetaPool + hook orchestration
`MetaPool` class: registers meta-procedures, iterates by hook, applies once-per-session debouncing. Pure orchestration, no procedures yet.

### 3.3 — Reasoning-loop hooks + signal injection in prompt
Wire the three hook points into `reasoning_loop.run_reasoning()`. Extend `_build_prompt` to take an `active_signals` argument and render `# System signals` section when non-empty. Update directive text to acknowledge signals.

### 3.4 — Signal persistence (session subgraph)
Signals emitted during a session land in `session.subgraph.nodes` as nodes with `node_type="signal"`. Round-trip tests verify they persist.

### 3.5 — Five initial meta-procedures
One module per procedure under `reasoning/meta_procedures/`. Each ~50-80 lines with its own unit tests.

### 3.6 — Integration test: cycle detection round-trip
Scripted LLM that triggers CycleDetector → assert signal fires + next prompt contains it + model output reacts. The canonical end-to-end test for Phase 3A.

### 3.7 — Integration test: contradiction sticky-signal lifecycle
Multi-iteration test that one ERROR signal persists across iterations until end of session.

### 3.8 — Manual real-LLM smoke
User-side. Same workflow as Phase 2A's 2.11.

---

## 10. Acceptance criteria for Phase 3A complete

1. All five initial meta-procedures land with unit tests asserting both true-positive and true-negative behavior
2. Signals appear in the next iteration's prompt as a dedicated section
3. ERROR-severity signals persist across iterations (sticky); INFO/WARN clear after one consumption
4. A scripted end-to-end test demonstrates: meta-procedure fires → signal in prompt → model reasoning visibly addresses the signal
5. Signals persist to the session subgraph as `node_type="signal"` nodes
6. Token cost overhead per session for meta-cognition: **0 LLM tokens output** (only prompt-side overhead for rendering active signals)
7. No Phase 1 or 2A regression — all 160 backend tests still pass
8. Backend test count grows by approximately 30+ new tests
9. Predicates are conservative: at the current corpus (~17 real sessions), the five meta-procedures' false-positive rate is observably zero (verified by replaying persisted sessions through the new meta-pool offline)
10. The front-end's `__diag__` node continues to populate (Phase 3A doesn't replace it; signals complement it)

---

## 11. Decisions (resolved 2026-05-20)

| Decision | Resolved as | Notes |
|---|---|---|
| **Sticky signal resolution** | **Never auto-clear in v1.** | ERROR signals persist for the rest of the session. Phase 3B (or later) can add explicit `signal_resolved` mechanism. |
| **Max signals per prompt** | **Cap at 5.** | Errors first, then warn, then info. Excess rendered as a single meta-signal "N additional signals suppressed for prompt brevity." |
| **Signal visibility format** | **Verbatim.** | The `message` field is hand-crafted in each meta-procedure to read naturally; no re-wrapping. |
| **Multi-firing same meta-procedure per iteration** | **Disallowed by default.** | Returns >0 signals = "fired" for that hook tick. `once_per_session=True` is the stricter debounce variant for things like contradiction detection. |
| **Signal `metadata` field** | **Free-form dict v1.** | Can tighten to a schema later if metadata shape stabilizes. |
| **UI rendering** | **DEFERRED.** | Per user 2026-05-20: raw-log inspection is sufficient for now. Signals persist to session subgraph as `node_type="signal"` nodes — front-end can adopt at any future point with no schema change required. |
| **Predicates on `raw_outputs`** | **Allowed, with caution.** | DispatchMissNudge needs it. Predicates that read raw output have higher false-positive risk; this is flagged in each such procedure's docstring. |

---

## 12. What this is NOT

- **Not LLM-dispatched meta-reasoning.** Meta-procedures are kernel rules, not procedures the model calls.
- **Not a replacement for the audit log.** Signals are operational observations for the model; the audit log is the ground-truth mutation history.
- **Not a planner / search algorithm.** Meta-procedures observe and signal; they don't choose what to do next.
- **Not a full executive function.** Real executive function requires goal management, backtracking, plan revision. This v1 catches specific failure modes only.
- **Not adaptive predicates.** Predicates are hand-coded. No learning, no auto-tuning. Phase 3D+ territory.

---

## 13. Effort estimate

| Sub-phase | Lines | Time |
|---|---|---|
| 3.1 Schemas (Signal, MetaProcedure, SignalNode) | ~150 | 0.5 day |
| 3.2 MetaPool + hook orchestration | ~120 | 0.5 day |
| 3.3 Reasoning-loop hooks + prompt-builder signal rendering | ~100 | 0.5 day |
| 3.4 Signal persistence | ~50 | 0.25 day |
| 3.5 Five initial meta-procedures (50-80 lines each) | ~350 | 1.5 days |
| 3.6 Cycle-detection end-to-end test | ~120 | 0.5 day |
| 3.7 Contradiction sticky-signal test | ~80 | 0.25 day |
| 3.8 Manual real-LLM smoke | n/a | 0.25 day |

**Total: ~970 lines, ~4 working days for Phase 3A.**

Smaller than Phase 2A (~1220) and Phase 1 (~2150) because the substrate (audit log, session subgraph, budgets, dispatcher) is reused entirely — Phase 3A is mostly orchestration on top.

---

## 14. Prior art

- **Soar's production system** — condition-action rules that fire on pattern-matched state, deterministic, no deliberation cost. The closest precedent for trigger-action meta-procedures.
- **ACT-R's procedural memory** — similar production-rule pattern.
- **OS interrupt handlers** — hardware detects an event, kernel injects context into the process. Direct analog for signal injection into prompts.
- **Exception handling in programming languages** — system raises, program handles via natural control flow. Same shape as our model reacting to signals via its normal reasoning.
- **LLM-as-judge / model-as-supervisor patterns** — what we're *not* doing. Those use additional LLM calls; we use predicates and let the existing LLM call handle reaction.

---

## 15. Status / next move

This doc is the **design phase output**. No code has been written for Phase 3A yet.

**Next step:** user resolves the OPEN decisions in §11. Once those are pinned, we start Sub-phase 3.1 (Signal + MetaProcedure schemas). Same discipline as Phase 1 / Phase 2A: design first, code second, behavior-inspect per sub-phase.
