# V5 v2 — Graph-Grounded Local Coder (design)

**Status:** design / direction. Supersedes the v1 framing for the *implementation*
goal. v1 (GNN + L8/L20 cross-attention soft-injection over a STEM knowledge graph
for grounded QA) is not thrown away — it is the proof that graph-grounding works,
and several pieces are reused (see §10).

**Last updated:** 2026-06-03

---

## 0. Roadmap & decision gates

The **spine is settled** (this doc); the **branches are decided by experiments**, not
opinion. Three experiments collapse most open variables — until they land, the
branches stay open.

**Track A — corpus/training (local, when the route-2 regen lands):**
1. merge `baseline_oc` + `baseline_cx` → project to V5 targets (`projection.py`)
2. Stage 1 (heads) → 2A (look) → 2B (write) on the fresh corpus vs `grown_graph4`
3. measure the guardrail: **per-domain fallback rate** (cs/physics/sysdesign spike =
   domain shortcut), plan P@1, support accuracy
4. rerun the retrieval baseline with the bigger gold set

**Track B — rented GPU (inference only, no training) — `scripts/cloud_run.sh`:**
1. **embedder A/B** — `causal-hidden` (Qwen3.5-4B) + `st-embed` (Qwen3-Embedding) vs
   the mpnet anchor (Hit@5 0.55 / MRR 0.47)
2. **`realstack_test` on Qwen3.5-4B** — validate injection-into-hybrid (hooks fire at
   L8/L20, hidden states extract, generation stable)
3. re-embed `grown_graph4` with the winner + re-calibrate dedup/attach thresholds

**Convergence:** B's wins (embedder/model) feed back into A (re-run training on the
improved substrate) — that is the "prove step 3 = the lift" experiment.

**Decision gates (open until measured):**

| open variable | decided by | status |
|---|---|---|
| graph embedder | retrieval A/B | ✅ **Qwen3-Embedding-0.6B** (Hit@5 0.63 > mpnet 0.55 > raw-hidden 0.41) |
| injection-into-hybrid works? | realstack on Qwen3.5 | ✅ **VALIDATED** on Qwen3.5-4B 4-bit — hooks fire, loops sane, pools route; fork dropped |
| trained ranker beats raw on code? | held-out A/B (#2) | ✅ **VALIDATED + LOCKED** — contrastive bi-encoder lifts code retrieval over raw (held-out Hit@5 0.40 / MRR 0.28; +19% MRR, +32% Hit@1). Config locked in-batch/2ep; hard negs gave no clean win. Plateau = query↔symbol semantic gap → #3 |
| graph TOPOLOGY helps retrieval? | mixed-pool A/B | ⚠️ **untrained heuristic FAILED** (hand-rolled 1-hop `cos+α·max_neighbor` HURT: Hit@1 −45%, MRR −14%, recall flat — even near-oracle). Lesson: topology needs **learned gating** (R-GCN), not a fixed formula. Open → decided by the trained GNN-as-ranker AFTER the trained bi-encoder is benchmarked on the mixed pool. NO more heuristics. |
| generalism cost (mixed vs code-only) | retrieval A/B | 📏 **MEASURED** — STEM distractors cost ~5pt Hit@5 / ~20% rel (code-only 0.377 → mixed 0.322). Real; ranker/fallback must absorb it. ONE general graph, eval on the mixed pool. |
| DeltaNet carries grounding? | realstack observation | open (bonus if true; nothing lost if not) |
| extractor size | held-out (0.5B = 50% ceiling) | open → likely step to 1.5B |
| visual nodes | cross-modal retrieval eval | open |
| domain shortcut | per-domain fallback after stage 2 | open (Track A) |

After these three land, the architecture goes from **designed** to **decided**.

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
| GNN + cross-attn | inject K/V every token (the *only* grounding path) | **kept** for REASONING-grounding + also a once-per-task **ranker** for the brief |
| exact tokens (APIs) | — (blurred by soft attn) | **graph-gated constrained decoding** (§3.5) — hard guarantee, not prompt text |
| who retrieves | the loop, implicitly per token | a cheap **controller/policy**, never the 4B |
| reliability | epistemic fallback (abstain) | **verifier-retry loop** (act, check, fix) |

### 3.5 The grounding mechanism — two jobs, two mechanisms (the crux)

Earlier drafts demoted the L8/L20 injector and shipped grounding as **verbatim text
in the prompt**. That was wrong: text-in-prompt is just RAG / a tool call — the
model can ignore it, lost-in-the-middle, no guarantee of use. It is *weak*
grounding. The fix is to recognize that grounding has **two distinct jobs** that
need **different mechanisms**:

- **Reasoning-grounding** — condition the *thinking* on graph structure (which node,
  how they relate, "use Dijkstra not Bellman"). This is exactly what soft K/V
  injection at L8/L20 is good at. **Keep v1 here — promote it, it's the spine.**
- **Emission-exactness** — get the *exact* tokens out (function names, args,
  imports, paths). Soft attention blurs this; *but the answer is NOT "dump text in
  the prompt"* (RAG). The answer is a mechanism that makes exact tokens
  **structurally guaranteed at decode time.**

So you do not choose injection vs text. Use **injection for reasoning AND a hard
emission-grounding mechanism for exactness:**

1. **Graph-gated constrained decoding (primary, frozen-LM-compatible).** Build a
   trie/grammar from the retrieved cards' valid symbols. In a "call position," a
   logits processor **masks the vocab to only valid API tokens from the graph** —
   the model *cannot* emit a hallucinated name; it is not in the allowed set.
   Grounding by construction, not by bias. Composes with injection (injection picks
   *which* API; the constraint guarantees you *spell it right*). No retraining;
   standard logits-masking (outlines/guidance-style). **This is the strong grounding
   — the graph GATES, it does not suggest.**
2. **Copy / pointer head over the graph K/V (deeper, trained).** Let the decoder
   **copy a token verbatim from a graph node it is attending to** (pointer-generator
   / CopyNet) instead of generating from vocab. Deep AND exact; the literal "use the
   cross-attention for exact emission." Cost: a trained head.
3. **kNN-LM / output-distribution interpolation (no training, latency cost).** At
   each step, retrieve nearest graph-node tokens and interpolate the next-token
   distribution toward them (RETRO / kNN-LM). Pushes exact API tokens up at the
   output layer. Cost: a datastore + per-step NN lookup (watch tok/s on 6 GB).

**The stack to run:**
```
v1 injection (L8/L20)          -> grounds the REASONING (relational/structural/plan)   [PROMOTED]
graph-gated constrained decode -> guarantees EXACT emission (cannot hallucinate an API) [the strong grounding]
verbatim card text (optional)  -> human-readable backup / the model's notes, NOT load-bearing
```
Why this beats both earlier options: v1-alone = deep but blurry (wrong API names);
text-RAG = exact but ignorable (weak). **injection + constrained-decode = deep
reasoning-grounding AND hard exactness.** Verbatim text stays only as a readable
backup, never the load-bearing path. The GNN-as-ranker (§7) still assembles the
brief; it is now *in addition to* the injector, not a replacement for it.

Cost order: constrained decoding first (moderate engineering, no retrain, highest
grounding/effort ratio) -> copy head (train, biggest deep+exact payoff) -> kNN-LM
(no train, per-token latency). Backing: GMT/"Beyond Prefixes" (injection into a
frozen LM); grammar/guided decoding for code; RETRO/kNN-LM (token-level retrieval).

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
- GNN encoder + L8/L20 cross-attention injector — **kept as the reasoning-grounding
  path** (§3.5), AND also used as a once-per-task ranker for the brief (§7)
- micro-controller planning (`task_family`/slots/steps) → §8
- `control_rule`/slot machinery → constraint ledger
- `answer_support_ids` + override-detection → retro-grounding (§9)
- graph-grower extract→stitch→hub-wire→gated-apply pipeline → re-point at code
- quantized LM loader (`v5/lm_loader`), 4-bit 4B deploy

**New to build:**
- **graph-gated constrained decoding** (§3.5) — logits processor + symbol trie from
  the brief; the strong emission-grounding. (Later: copy head, kNN-LM variants.)
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
