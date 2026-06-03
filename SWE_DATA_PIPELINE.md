# SWE code-data pipeline — the cheap rung (gold-patch grounded traces)

**Status:** spec. First concrete step toward v2 *code* data (see `V5_V2_DESIGN.md`).
**Scope of THIS doc:** the cheap rung only — build a **code graph** + **grounded code
traces** from SWE-bench, using the **gold patch** as the solution + support signal.
**No Docker, no test runs, no agentic teacher.** (Verifier harness + agentic
trajectories are the *next, expensive* rung — out of scope here.)

**Why cheap:** the gold patch already tells us the answer AND (via AST diff) exactly
which symbols the fix touched → the brief/support fall out for free. We get real
code-grounding data + the retrieval ranker's gold without running anything.

---

## 1. SWE-bench fields we use (per instance)

From `princeton-nlp/SWE-bench_Verified` / `SWE-bench_Lite` / SWE-gym:

| field | use |
|---|---|
| `instance_id` | row id (e.g. `django__django-12345`) |
| `repo` + `base_commit` | checkout to build the code graph |
| `problem_statement` | the **task** |
| `patch` | the **gold solution** (unified diff) |
| `test_patch`, `FAIL_TO_PASS`, `PASS_TO_PASS` | recorded as the **verifier spec** (not run in this rung) |

**Split (no leakage):** train on **SWE-gym**, hold out **SWE-bench Verified** for eval.
Lite (300) = smoke while building.

---

## 2. Pipeline (per instance, no Docker)

```
SWE instance
 1. checkout repo @ base_commit   (shallow + sparse: touched files + neighborhood)
 2. code_extract: parse files -> symbol / api_card nodes + edges -> code graph
 3. parse gold `patch` (unidiff): hunks -> touched files -> map lines to symbols (AST spans)
 4. brief    = touched symbols + their 1-hop callers/callees/imports (what the fix needed)
    support  = symbols the patch actually modifies/adds (exact, from the diff)
 5. emit a grounded-trace row (v2_grounding shape):
       task=problem_statement, solution=patch,
       brief.retrieved_ids=brief symbols, support_ids=touched symbols,
       verifier={FAIL_TO_PASS, PASS_TO_PASS}  (reference only)
 6. apply the code-graph nodes via the existing grower (stitch + hub-wire + gated apply)
```

Output per instance: **(a)** code-graph nodes/edges for that repo, **(b)** one
grounded-trace row, **(c)** retrieval gold = (issue -> touched symbols).

---

## 3. Node schemas (code; reuse the `raw_edit` envelope)

### code_symbol  (`node_type: symbol`)
```json
{ "id": "sym_<repo>_<hash>", "node_type": "symbol",
  "kind": "function|class|method|module",
  "name": "AuthHandler.login", "file": "app/auth/handler.py",
  "lineno": [120, 168],
  "signature": "def login(self, user: User, pw: str) -> Token",
  "text": "<signature + 1-line docstring summary>",   // for embedding/grounding
  "metadata": {"repo": "django/django", "commit": "<base_commit>"} }
```
### api_card  (from imports/calls; `node_type: api_card`) — as in V5_V2_DESIGN §4.1.
Edges (relation): `calls`, `imports`, `defined_in`, `contains` (extend the relation
vocab as needed; structural).

`artifact_kind` in the grounded-trace row = `"code_symbol"` (was `"graph_node"` for
STEM). Same `v2_grounding` schema otherwise — the row is born v2-shaped.

---

## 4. Scoping — do NOT graph the whole repo

Django/sympy/matplotlib are huge. Per instance, graph the **relevant neighborhood**:
- the files the gold patch touches,
- + symbols those files **import** and **call** (1-hop),
- + sibling functions/classes in the same files.

Across instances of the same repo, **accumulate** into one per-repo code graph (django
symbols shared across all django instances) via the grower's dedup/attach.

NOTE the asymmetry: the gold patch reveals the relevant files **post-hoc** (great for
*training* the ranker — that's the gold). At **inference** the ranker must find them
from the issue text alone — which is exactly the retrieval skill we're training.

---

## 5. Reused vs new

**Reused:**
- grower pipeline: `extract -> stitch -> hub-wire -> gated apply` (re-point the
  extractor from `fetch_cot` chunks to parsed code).
- `v2_grounding` row schema (task + brief -> solution + support).
- `answer_support_ids` concept -> here it's **AST-diff-derived** touched symbols (exact).
- `retrieval_eval.py` (Recall@k/MRR) -> now on code symbols; gold = patch-touched symbols.
- `Qwen3-Embedding-0.6B` (chosen embedder) for symbol embeddings.

**New to build (this rung):**
- `swe_load.py` — load SWE-bench/SWE-gym via `datasets`; iterate instances.
- repo checkout helper — shallow clone + `git checkout base_commit` + sparse paths.
- `code_extract.py` — Python `ast` (start) / tree-sitter (later) -> symbol/api_card
  nodes + edges, scoped to the neighborhood.
- patch parser — `unidiff` -> hunks -> (file, line-range) -> map to symbols via AST spans.
- grounded-trace builder — emit the `v2_grounding` rows.

---

## 6. Honest gotchas

- **Repo checkout cost**: big repos, specific commits. Shallow + sparse checkout, or use
  SWE-bench's provided repo snapshots. Cache per (repo, commit).
- **Python first**: `ast` for Python (the SWE-bench distribution). Multi-language = later
  (tree-sitter). Boundary: a "fix Python OSS-lib bugs" coder, not arbitrary code.
- **Hunk -> symbol mapping**: get function/class line spans from `ast`, map patch hunk
  lines into them. Hunks touching module-level / non-symbol lines -> attach to the file node.
- **Brief is approximate** in this rung (patch-touched + 1-hop). Good enough to train the
  ranker; the agentic rung later refines what's actually needed.
- **Symbol text for embedding**: signature + a 1-line summary (not the whole body) -> terse,
  fits the embedder + the eventual 6GB context.

---

## 7. Sequencing + first build step

1. `swe_load.py` + repo checkout — get instances + repos on disk (smoke on SWE-bench Lite).
2. `code_extract.py` — repo files -> symbol graph (apply via grower).
3. patch parser + grounded-trace builder -> `v2_grounding` rows + retrieval gold.
4. run `retrieval_eval` on the code graph (Qwen3-Embedding) — first **code** retrieval number.

**First step to implement:** `swe_load.py` + `code_extract.py` on **SWE-bench Lite**
(300, small) — parse a handful of repos into a code graph and eyeball the symbol nodes.
That validates the parse + scoping before wiring the grounded-trace builder.

This rung produces: a **code graph**, **grounded code traces** (v2-shaped), and the
**code retrieval gold** — the first real v2 data — with **zero Docker**. The verifier
harness + agentic trajectories come after, on a Linux box.
