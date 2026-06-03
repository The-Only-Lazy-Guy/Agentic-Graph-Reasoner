# V5 v2 — Graph-Grounded Local Coder (design)

**Status:** design / direction. Supersedes the v1 framing for the *implementation*
goal. v1 (GNN + L8/L20 cross-attention soft-injection over a STEM knowledge graph
for grounded QA) is not thrown away — it is the proof that graph-grounding works,
and several pieces are reused (see §10).

**Last updated:** 2026-06-03

---

## 1. Goal & scope

A **local** assistant that does implementation / side-task / debugging work on a
6 GB GPU, where the language model is a **small (Qwen3-4B, 4-bit) and assumed to
know little**. The graph carries the heavy load — *what is correct, where it goes,
how to check it* — and the model only **recognizes the task, adapts a grounded
example, emits a small diff, and reacts to a verifier**.

Design line, repeated everywhere below:

> The graph supplies **verbatim ground truth** + a way to **check** the output.
> The 4B only **adapts and glues**. It is never a smart retriever or a smart planner.

**Non-goals:** open-ended autonomous SWE; the model inventing/executing its own
tools (the dropped "procedure nodes that call procedures" — a workflow DSL +
interpreter, too risky); replacing the big teacher on the novel tail (that
escalates — §11).

---

## 2. Why a weak 4B can work here

Small coder models fail in predictable ways. The graph is built to remove exactly
those failure modes:

| weak-model failure | graph remedy |
|---|---|
| hallucinated API names/args/imports | **API cards** — model copies, doesn't recall |
| doesn't know the codebase | **symbol/structure graph** — hands it the target + style |
| can't plan multi-step | **decomposition skeleton** + V4 micro-controller |
| forgets constraints mid-gen | **constraint ledger** (reuse `control_rule`/slots) |
| can't tell if it's right | **verifier loop** with oracles in the graph |
| reinvents known solutions | **worked exemplars** (retrieve + adapt) |
| repeats finished subtasks | **task ledger** (maintained state, §6) |

What the graph **cannot** offload: execution itself (reading/writing real code,
interpreting tool output, recovering from genuine surprise). It cuts *planning &
knowledge* cost, not *execution* cost. Design around that line.

---

## 3. What changes from v1

| | v1 (QA) | v2 (coder) |
|---|---|---|
| graph contents | STEM facts | code artifacts (§4) |
| GNN + cross-attn | inject K/V into LM hidden states **every token** | run **once per task** as a retrieval **ranker** that assembles a brief |
| grounding delivery | soft hidden-state bias | **verbatim text in context** (code needs exact tokens) |
| who retrieves | the loop, implicitly per token | a cheap **controller/policy**, never the 4B |
| reliability | epistemic fallback (abstain) | **verifier-retry loop** (act, check, fix) |

**The mechanism shift is the crux:** for implementation, soft injection biases
"vibes" but loses the exact arg names code needs. So the GNN becomes a *ranker*
that selects nodes; the selected artifacts go into the prompt as text. The L8/L20
injector is demoted to (optional) routing/epistemic signal.

---

## 4. Graph node schemas (grounding artifacts)

All are distilled offline from the big teacher running real tasks (§9). All are
**terse** (cards, not docs; diffs, not files) to fit the 6 GB KV cache.

### 4.1 API card  (`node_type: api_card`)
```json
{
  "id": "api_<hash>",
  "node_type": "api_card",
  "symbol": "requests.post",
  "signature": "post(url, data=None, json=None, **kwargs) -> Response",
  "import": "import requests",
  "returns": "requests.Response",
  "min_example": "r = requests.post(url, json=payload); r.raise_for_status()",
  "gotchas": ["use json= not data= for JSON bodies", "call raise_for_status()"],
  "source_ref": "<repo path or doc url>"
}
```

### 4.2 Symbol / structure node  (`node_type: symbol`)
```json
{
  "id": "sym_<hash>",
  "node_type": "symbol",
  "kind": "function|class|module",
  "name": "AuthHandler.login",
  "file": "app/auth/handler.py",
  "signature": "def login(self, user: User, pw: str) -> Token",
  "conventions": ["logs via self.log", "raises AuthError on failure"]
}
```
Edges: `calls`, `imports`, `defined_in`, `tested_by` (reuse relation vocab where
possible; add code relations as needed).

### 4.3 Worked exemplar  (`node_type: exemplar`)
```json
{
  "id": "ex_<hash>",
  "node_type": "exemplar",
  "task_kind": "add_structured_logging",
  "task_text": "add logging to a request handler",
  "diff": "<unified diff that worked>",
  "constraints_met": ["used existing logger", "no new deps"],
  "verifier": "pytest tests/test_handler.py::test_logs"
}
```

### 4.4 Error→fix card  (`node_type: error_fix`)
```json
{
  "id": "err_<hash>",
  "node_type": "error_fix",
  "signature": "ModuleNotFoundError: No module named 'X'",
  "causes": ["missing dep", "wrong venv", "relative import"],
  "fixes": ["pip install X", "check pyproject deps", "use absolute import"]
}
```

### 4.5 Constraint / control_rule  (reuse v1 `control_rule`)
```json
{ "id":"cr_<hash>", "node_type":"control_rule",
  "task_kind":"add_endpoint", "must":["validate input","add test"],
  "must_not":["break public API"], "style":"black, type hints" }
```

---

## 5. Verifier loop (the reliability multiplier)

A weak generator + a strong checker ≫ a weak generator alone. Oracles live in the
graph; the **runtime** runs them.

- Each subtask has a **verifier spec** (from exemplar / control_rule): tests to
  run, lint rules, invariants, a build/typecheck command.
- After the model emits a diff: apply (in a sandbox) → run verifier → parse result.
- **Pass** → commit to ledger. **Fail** → fetch the matching `error_fix` card,
  feed failure + card back, retry (bounded retries, then escalate §11).
- Retrieval expansion is triggered by **verifier failure (ground truth)**, never by
  the model's self-assessed uncertainty.

---

## 6. Three-layer working memory (multi-subtask state without heavy re-retrieval)

Tasks split into subtasks (different goals each), so naive "retrieve per subtask"
looks heavy. Resolve by **separating maintained STATE from retrieved GROUNDING**,
and making per-subtask retrieval a small **delta** on a **pinned** base.

### Layer 1 — Task Ledger (maintained, never retrieved, always in context)
```json
{
  "goal": "<task>",
  "plan": [
    {"id":"st1","goal":"locate handler","status":"done","output":"app/auth/handler.py:login","verifier":"n/a"},
    {"id":"st2","goal":"add logging","status":"in_progress","output":null,"verifier":null}
  ],
  "open_bottlenecks": ["test fixture missing for auth"],
  "current": "st2"
}
```
- Tiny; **append-updated mechanically** by the system (verifier result + structured
  diff summary) — **not** 4B free-text, or the "what's done" record corrupts.
- This *is* the task stack → solves "know what's done / don't repeat subtasks /
  what's the bottleneck."

### Layer 2 — Task Brief (retrieved ONCE at task start, pinned in KV cache)
Shared grounding for the whole task: project conventions, target files, core API
cards for the feature area, the decomposition skeleton. Assembled once by the
GNN-ranker; **frozen in the KV cache** → zero context loss across subtasks, no
re-retrieval. (This is "single strong retrieval, no loss" — scoped to the *shared*
part only.)

### Layer 3 — Subtask Delta (per subtask, small, cheap)
Only the artifacts unique to the current subtask goal (a couple cards / one
exemplar). Small embedding lookup + small prefill on top of the pinned base. This
is "fire many times cheaply" — each fire is a delta, not a full brief.

**Refresh rule:** if a subtask's needs have low overlap with the pinned brief
(domain jump), re-assemble the brief. Rare, controlled.

---

## 7. Retrieval policy — take it OFF the model

The 4B deciding *what to retrieve* is circular (if it knew, it wouldn't be weak).

- A **cheap policy** (mostly rules + embedding/GNN rank) assembles brief & deltas.
- Driven by **task-type → retrieval recipe**: one cheap classification ("this is an
  add-logging task") → a fixed bundle of artifact types to pull. One decision, not N.
- The GNN's job: **rank candidate nodes** for the recipe slots; offline, once per
  task (brief) + tiny per subtask (delta). No per-token cost.

---

## 8. Decomposition grounding (don't let the weak model over-split)

- Retrieve a **decomposition skeleton** for the task *type* from the graph
  ("feature-add ≈ locate → modify → test → wire").
- Reuse the **V4 micro-controller**: `task_family → required_slots → micro_steps`.
- The 4B **adapts a known plan**; it does not invent one → the "splits wrong /
  over-reduces complexity" risk drops hard.

---

## 9. Data: how to build the corpus (the retro-grounding recipe)

**Problem:** the opencode/codex teacher won't produce *grounded* implementation
traces — asked to implement, it writes a straight solution referring to nothing. So
there is no grounding behavior to distill from natural traces.

**Fix — construct the grounded form offline (don't rely on teacher grounding):**
1. Teacher solves the task **ungrounded** → solution diff.
2. **Retro-link** the solution to the artifacts it *actually* used:
   - functions/APIs it calls → those `api_card`s
   - files/symbols it touches → those `symbol` nodes
   - the failing→passing transition → an `error_fix` (if any)
   - the whole thing → an `exemplar`
   High-precision **lexical/AST linking** (only include a card if the solution
   literally references that symbol — verifiable).
3. Train the student on **`(task + assembled brief) → solution`** — exactly the v2
   inference shape. The student learns "given this grounding in context, produce
   this," sidestepping the teacher's refusal to ground.

**Reuse:** this is the v1 `answer_support_ids` + override-detection machinery
("what did the output rest on"), pointed at code. Same idea, AST/lexical-precise.

Grounding-first still holds: STEM grounding (current work) proves the graph→answer
grounding mechanism before re-pointing the grower at code.

---

## 10. Reused vs new

**Reused from v1:**
- GNN encoder (repurposed: ranker, not per-token injector)
- micro-controller planning (`task_family`/slots/steps) → §8
- `control_rule`/slot machinery → constraint ledger
- `answer_support_ids` + override-detection → retro-grounding (§9)
- graph-grower extract→stitch→hub-wire→gated-apply pipeline → re-point at code
- quantized LM loader (`v5/lm_loader`), 4-bit 4B deploy

**New to build:**
- code node schemas (§4) + a code extractor (API cards / symbols / exemplars /
  error-fix) for the grower
- retrieval policy + task-type recipes (§7)
- 3-layer working memory: ledger format + pinning/KV discipline (§6)
- verifier-retry runtime (sandbox apply, run tests/lint, parse, escalate) (§5)
- retro-grounding data builder (§9)

---

## 11. Open questions, risks, escalation

- **Novel tail:** tasks with no card/exemplar → the model must reason → 4B weak.
  **Escalate to the big teacher** (and capture that trace as a new exemplar). Design
  the escalation explicitly; do not pretend the brief always exists.
- **Ledger upkeep** must be mechanical; free-text 4B summaries will drift.
- **KV-cache discipline** to pin the brief while delta+ledger rotate — real but
  standard engineering.
- **Brief staleness** on domain jumps → overlap-triggered refresh.
- **Retro-link precision:** over-attributing artifacts the solution didn't use
  poisons training → keep linking AST/lexical-exact.
- **tok/s on 4050:** verifier loops + retries multiply token volume; keep diffs
  small, contexts terse, retries bounded.

---

## 12. Cheap de-risk probe (do this before building the runtime)

Hand-author **5–10 API cards + 1–2 exemplars** for ONE concrete side-task type
(e.g. "add structured logging to a function"). Wire a dumb
**retrieve-brief → 4B emits diff → run tests** loop on a bare 4B. Measure:

> how much further does the 4B get **with** the brief vs **cold**?

Big lift → the planning/knowledge-offload thesis holds; build the runtime.
Small lift → the bottleneck is execution, not grounding, and more graph won't save
it (rethink before investing).
