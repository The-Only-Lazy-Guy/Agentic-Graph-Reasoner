# READ_THIS — V5 latest raw results & quick reference

> At-a-glance dump of the latest runs (raw outputs, numbers, repro commands) so
> you don't have to dig through commits/logs. Updated each working session.

**Last updated:** 2026-06-03
**HEAD:** latest pushed commit on branch `main`

---

## Session update (2026-06-07) — VERIFIED RESOLVE NUMBERS + the grounding lift (official harness, local Docker)

Stood up the **real verifier locally**: WSL2 Ubuntu (on E: for space) + docker.io + swebench.
**gold-sanity 3/3 resolved -> HARNESS PROVEN** (sb-cli hosted-lite is broken: marks correct gold
"failed"; use Docker). Ran the loop grounded vs `--no-graph` cold (n-eval 30), verified both:

| condition | applyable | **resolved** |
|---|---|---|
| **grounded** | 19/30 (63%) | **4** (4/17 submitted, ~13% of attempted) |
| **cold (--no-graph)** | 1/24 (4%) | **0** |

**Grounding accounts for 100% of the resolves** — remove it (no read-source, no injection) and the
4B solves NOTHING (0/24). Lift: **applyable ~15x (4%->63%), resolve 0->4.** Clean — no loop/SR-format
confound. astropy-14365's gold (`re.IGNORECASE`) = exactly our read-source bullseye -> resolves for us.
Modest absolute (4B reality), lift unambiguous = **the thesis, verified.** Scaling levers: retries
4->6, full n-eval 300, graph coverage, and `grounded_coder --gate verify` (now UNBLOCKED — Docker
works) = self-improving loop capturing only RESOLVED patches into memory. Verify is CPU-only
(~2-7 min/instance). See `DOCKER_VERIFY_RUNBOOK.md`.

---

## Session update (2026-06-07) — MILESTONE: first correct end-to-end fix (read-the-source unlocks edits)

**The execution wall diagnosed + broken.** Exhaustive manual inspection of generated patches
showed: grounding gets the 4B to the RIGHT FILE but the edits were garbage (no-op / wrong
function / unmatchable). Root cause (proven, not "model ignores graph"): the brief gives symbol
SIGNATURES (where), not the function BODY (what to edit) — so the 4B wrote SEARCH blocks for code
it never saw -> blind edits. The frozen LM still USES grounding (stage-3 NLL dropped, localization
works); it was just editing blind.

**Fixes built this session (all infra-free):**
- **SEARCH/REPLACE edit format** (`v5/runtime/search_replace.py`) — the 4B writes exact old->new
  Python (its strength), no line-number/hunk math -> no `@@ -XXX` garbage; blocks apply
  deterministically (verifiable). Same idea as Aider for weak models.
- **Read-the-source step** (`v5/runtime/sr_withcode.py`) — checkout repo@base_commit, read each
  support symbol's ACTUAL body (metadata.file+lineno), put it in context, then emit SR. NO
  graph-search tool, no verifier.

**RESULT (15 lite, adapter_code_s3): read-the-source CONFIRMS the diagnosis.**
inject(signatures) -> withcode(real source): **search_in_file 0.067 -> 0.333 (5x)** (SEARCH now
matches the real file = APPLYABLE), **edit_cov 0.067 -> 0.20 (3x)**. well_formed dropped
0.73->0.40 (long source -> over-reasons -> emits fewer blocks; tunable). **astropy-14365 = a
PERFECT fix**: SEARCH `re.compile(_type_re)` -> REPLACE `re.compile(_type_re, re.IGNORECASE)` =
EXACTLY the gold patch. First correct surgical fix end-to-end — because the model saw the source.

**Architecture VALIDATED (the V4 loop, for code):** localize (graph ranker) -> READ the source
(repo, not graph) -> edit (SR, now matchable) -> verify. Single-shot failed only because it
skipped the read step. Remaining gap = hit-rate (no-op edits + emission drop on long context) ->
tune the read step (fewer/shorter bodies, tighter anti-echo prompt, keep it FAST) then the
verifier-retry loop (no-op -> tests red -> retry). RL = expensive endgame, needs the verifier
1000x + breaks frozen-LM; the retry loop is RL-lite (feedback, no training). Reuse V4's
`reasoning_loop`/`micro_controller` (swap verify-step -> tests), don't reinvent.

---

## Session update (2026-06-05) — STAGES 3/4: grounding → better generation (the v2 thesis demonstrated)

**The full v2 grounding stack is now proven end-to-end on code.** Progress map:

PROVEN (mechanisms, all on code):
- retrieval: trained bi-encoder ranker beats raw + survives mixed pool (GNN-ranker shelved)
- emission: constrained-decode selection valid 1.0 / 2.1× random; in-patch constraint built
- reasoning injection (stage 1/2): adapter routing plan/evid attn 1.0, write 0.13, non-destructive
- **injection→generation (stage 3, `stage3_answer.py`):** held-out gold-patch NLL cold 1.30 →
  injected 0.98–1.18 (87–98% better) — grounding makes the gold patch more likely
- **grounded generation runtime (stage 4, `stage4_generate.py`):** injection → is_diff 0.40→0.73
  (valid patches) + slight file-cov lift; exact-symbol emission (edit_cov) flat → motivates the
  in-patch constrained decode (`--constrain`, just built) to force the exact symbols.

Key fixes this session: planning targets wired into the code corpus (plan precision 0→1.0);
nan loss = cosmetic empty-slot BCE (guarded); Stage-3 inject-at-ALL-positions (last-token-only
had zero gradient); adapter checkpointing (`--adapter-ckpt`, skips slow 1/2 rebuild).

REMAINING (product, not proof): verifier-retry loop + REAL test-pass (Tier-2, infra-blocked on
Docker/sb-cli — the headline resolve-rate); 3-layer working memory; retrieval recipes; scale
(strategy distill on Verified, SWE-gym) + polish. **Architecture risk ~retired; what's left is
engineering + the verifier infra.**

---

## Session update (2026-06-03c) — #3 scaled (300) + generalism cost MEASURED + naive topo rerank FAILED

**#3 strategy distiller scaled to 300/300 (SWE-bench Lite, opencode):** 998 content nodes
(299 strategy / ~409 reasoning_atom / 308 solved_subgoal) + 302 hubs, 4535 edges incl
**2101 `leveraged`** (strategy→symbol) + 232 `transfers_to` (cross-instance). Strategy
bridge nodes ~94% general (sample-checked + lint-confirmed). Generality lint made
**type-aware** (`swe_strategy._LINT_TYPES`): hard-drop leaky retrieval-entry nodes
(strategy/atom), KEEP solved_subgoal (instance resolution, grounded by edge).
Retroactive filter (`filter_strategy_leaks.py`) on the pre-lint 300-run: dropped 70
nodes (18 strategy + 52 atom) + 341 edges, 0 orphans.

**KEY DECISION — ONE general graph, not a code-only silo.** grown_graph4 is multi-domain
(cs/algo/sysdesign/logic + physics/chem/bio/math); the product is generalist (same
assistant tutors a student AND fixes code). Eval the code task on the **MIXED pool**
(distractors present) — a sanitized code-only number hides the real deploy condition.
`retrieval_eval --mix-graph` unions graph + code symbols.

**Generalism cost MEASURED (Qwen-embed, code gold, Blackwell box):** code-only pool
(21.6k) Hit@1 0.159 / Hit@5 0.377 / MRR 0.262 → **MIXED pool (29.8k, +STEM)** Hit@1
0.112 / Hit@5 0.322 / MRR 0.209. STEM distractors cost **~5pt / ~20% relative**. Real,
expected; the fallback/ranker must absorb it.

**Naive topology rerank FAILED (and that's fine — it was a bad probe).** `topo_rerank_eval`
= a hand-rolled, UNTRAINED 1-hop heuristic (`rerank=cos + α·max_neighbor_cos`, α=0.5,
bidirectional, ungated). On the mixed pool it HURT: base Hit@1 0.112 → 0.062 (−45%),
MRR 0.209 → 0.180 (−14%); Hit@5/10/20 flat. Damning: Hit@10/@20 didn't move even in a
near-ORACLE setup (not held-out — the strategy minted from issue X is in the pool when
querying X) → the bridge surfaced ZERO correct symbols base missed. **Diagnosis:** an
untrained heuristic has no gate, so it boosts every node with a vaguely-matching
neighbor → noise reshuffles the top. **Lesson: this architecture's value IS learned
gating; a fixed formula can't prove/disprove it. NO more heuristics — the cheapest VALID
test is the smallest TRAINED model.** topo_rerank kept only for its base-pool byproduct.

**RESULT — #2 ranker SURVIVES the mixed pool (2026-06-03c, Blackwell).** Trained locked
ranker (in-batch, 2ep, 35s) eval'd base vs ranker on the MIXED 28,495-node pool, 114
held-out: base Hit@1 0.105 / Hit@5 0.360 / Hit@20 0.526 / MRR 0.226 → **ranker Hit@1
0.167 / Hit@5 0.386 / Hit@20 0.597 / MRR 0.285** (+58% Hit@1, +26% MRR, +13% Hit@20).
**Money contrast — SAME mixed pool:** naive topo heuristic Hit@1 −45% / MRR −14%;
**trained ranker +58% / +26%.** Learned gating works, heuristics don't — proven on
identical data. **BANKED: trained bi-encoder ranker = the generalist retrieval baseline
(~Hit@5 0.39 / MRR 0.28 mixed).** Ranker absorbs most of the ~5pt generalism penalty.
Caveats: absolutes modest (code retrieval hard); ranker uses ZERO topology (2101
`leveraged` edges untapped).

**Next (decided): GNN-as-ranker** — must BEAT the 0.39 bi-encoder by using topology with
LEARNED gating. Reuse `v5/gnn_encoder.RGCNEncoder` (2-layer R-GCN, 12 relations, node
input = frozen text-embed + type/epistemic/confidence → [N,256]) + a query projector
(issue Qwen-embed → GNN space), contrastive on (issue→support) like #2 but over GNN node
embeddings (so a symbol's embedding absorbs its strategy/`leveraged` neighbors). Design
rule: GNN must REFINE the strong Qwen features (concat/residual), not replace them through
the 256 bottleneck (else it loses to the 0.6B bi-encoder). New: `train_gnn_ranker.py`.
NOTE the R-GCN currently expects 768-d (mpnet) text feats; Qwen-embed is 1024 → init with
`text_embed_dim=1024`. The SAME RGCNEncoder is also the injection KV path (Stage 1/2A/2B,
heads still untrained) — ranker first (cheaper, isolates "does topology help retrieval").

**GNN-ranker VERDICT (2026-06-04) — topology does NOT beat the bi-encoder; STOP grinding it.**
v1 architecture bug (separate query/node projections destroyed the shared-embedding
alignment) → ~random (Hit@5 0.026). Fixed: query = frozen Qwen (no projection); node =
Qwen + delta(gnn), delta ZERO-INIT → starts at the raw base. Fixed run: `[init/base]`
Hit@5 0.307 (== raw embed, harness sound) → climbs to BEST **epoch 5 Hit@5 0.360 / MRR
0.293**, then OVERFITS hard (train loss 8.6→0.78, held-out collapses to 0.23 by ep29;
584 train queries can't support a full GNN). GNN best vs bi-encoder #2 (same held-out):
GNN edges MRR (0.293 vs 0.272) but **LOSES Hit@5 (0.360 vs 0.412)** and is unstable.
**Not a clean win.** Topology carries *some* signal (lifts MRR over its own raw base
0.186→0.293) but a strong bi-encoder suffices for RETRIEVAL. **DECISION: bank the trained
bi-encoder as the ranker; stop the retrieval-GNN.** Topology's real home = the INJECTION
path (stage 6 reasoning K/V), not retrieval ranking. Held my own "one clean attempt" line.

**PIVOT — the generation half (the real unknown).** Retrieval is settled. Everything so
far (#1 data, #2 ranker, #3 strategy, #4 GNN) answered "can we RETRIEVE grounding?" The
v2 bet is the other half: does a weak 4B USE the brief to write better code than cold?
Next: (5) stand up the **SWE verifier harness** (Docker, FAIL_TO_PASS) — deferred
expensive rung; (6) the **§12 brief-vs-cold probe** — 4B+brief vs cold, measure
test-pass lift = the GO/NO-GO for v2. New: `swe_verify.py`.

---

## Session update (2026-06-03b) — #2 ranker CLOSED (config locked) + hard-neg verdict + #3 deps

**#2 (trained retrieval ranker) is DONE — locked, stop tuning it.**

- **Confirmed (2nd A40 run):** the ranker beats raw retrieval again. Leakage-free
  held-out (114 q, SAME held-out set within the run — the only valid comparison):
  raw Qwen-embed Hit@1 0.132 / Hit@5 0.377 / Hit@10 0.474 / Hit@20 0.526 / MRR 0.238
  → **ranker Hit@1 0.175 / Hit@5 0.404 / Hit@10 0.500 / Hit@20 0.640 / MRR 0.284.**
  Lift biggest on Hit@1 (+32% rel), MRR (+19%), Hit@20 (+11 pts). Thesis re-proven.
- **Hard-negatives VERDICT = no clean win.** This run (hard-neg, 4ep) scored ranker
  0.404 / MRR 0.284 / Hit@1 0.175; the prior run (in-batch, 2ep) scored 0.414 / 0.300 /
  0.190. **Do NOT read 0.404-vs-0.414 as a regression** — the two runs use DIFFERENT
  held-out splits (114 vs 116 q; rows changed → different shuffle), so cross-run
  absolute deltas at ~115 q are split noise, not signal. Within each run the ranker
  beats raw; that's the robust claim. Hard negs neither clearly helped nor hurt.
- **Root cause of the plateau (the real signal):** the bottleneck is the
  **query↔symbol semantic gap** — query = bug-symptom issue, target = bare `def`
  signature, low overlap. Negative-sampling tricks don't close that gap → diminishing
  returns. The gap is closed by **#3 strategy nodes** (behavior-level text matches the
  issue) + **graph-topology rerank** (strategy `leveraged`→symbol edges), NOT by more
  ranker tuning.
- **Config LOCKED** (`train_ranker.py` + `scripts/cloud_run.sh`): in-batch negatives,
  2 epochs (`--num-hard 0 --epochs 2`, now the defaults). Hard-neg code path stays,
  re-enable via `NUM_HARD>0`/`--num-hard`. Ship ranker as the v2 retrieval baseline
  (~Hit@5 0.40 / MRR 0.28 held-out).

**#3 (strategy-node distiller) deps — for the cheap CPU box (no GPU):**
- python libs: **`datasets`** (swe_load.load_dataset — was MISSING from
  `requirements-datagen.txt`), `huggingface_hub`, + `sentence-transformers`/`numpy`/
  `requests` (pulled at import of answerer_v4 controller). NO torch-cuda / transformers /
  bitsandbytes / torch_geometric — #3 is pure data-gen.
- non-pip: **git CLI** (swe_grounded shells git to checkout repo@base_commit) +
  **opencode CLI** (`npm i -g opencode-ai`, Node 18+, model API key) — or `--backend codex`.
- output: `swe_strategy_candidates.jsonl` (strategy nodes + `leveraged` edges → support
  symbols). Candidates only — apply onto a `code_graph.json` next.

**Next after #3 lands:** build `code_graph.json` (apply symbol candidates → grown) →
apply strategy candidates → **graph-topology rerank eval** (issue → strategy match →
1-hop to symbols, vs raw bi-encoder Hit@5 0.40). That's the test of whether the
semantic gap actually closes.

---

## Session update (2026-06-03) — Qwen3.5-4B validated + embedder decided + SWE code-data pipeline (v2 pivot)

Big arc. The project pivoted from "STEM grounded-QA proof" toward the real product:
a **local agentic coder** (v2). See `V5_V2_DESIGN.md` + `SWE_DATA_PIPELINE.md`.

**Model + architecture (Track-B gates CLOSED, on an L40):**
- **Base LM = Qwen/Qwen3.5-4B** (hidden 2560 — same as the stale Qwen3-4B target so
  no adapter-dim change; 32 layers; hybrid Gated-DeltaNet+attention; multimodal;
  4-bit for 6GB). **Injection-into-hybrid VALIDATED**: realstack passed — hooks fire
  at L8/L20, GNN encodes, pool routing correct, loops sane, generation intact,
  fallback correct (untrained heads). The "drop injection" fork is OFF.
- Needed 5 hybrid fixes (committed): config dims -> derive from model
  (get_input_embeddings); 4-bit device_map None -> cuda:0; output_hidden_states in
  the forward CALL; +torch_geometric dep; **dtype seam** bf16 LM <-> fp32 adapter
  (cast at the hook boundary).
- **Graph embedder = Qwen3-Embedding-0.6B** (retrieval A/B on grown_graph4, 78 gold):
  **Hit@5 0.628 / MRR 0.489** > mpnet 0.551/0.470 > raw Qwen3.5-hidden 0.410. Adopt.
  PENDING production swap (re-embed + recalibrate dedup/attach thresholds).

**v2 design (graph-grounded coder) — key decisions:**
- **Dual grounding**: keep L8/L20 injection for REASONING-grounding (promoted, NOT
  demoted) + **graph-gated constrained decoding** for EXACT emission (the strong
  grounding; logits-mask to valid API tokens -> can't hallucinate). Verbatim-text-in-
  prompt is just RAG (weak) -> rejected as the load-bearing path.
- **GNN = once-per-task RANKER** assembling a brief, AND the injector (both).
- **3-layer memory**: Task Ledger (carried, never retrieved) + Task Brief (pinned
  once, KV-cached) + Subtask Delta (cheap). Decompose via V4 micro-controller skeleton.
- **Data = retro-grounding**: teacher solves -> link solution to artifacts used (reuse
  answer_support_ids). V4 corpus now emits a `v2_grounding` block (additive).
- Prior-art validated (GMT/Beyond-Prefixes = injection twin; ReVeal = verifier;
  Voyager/Trainable-Graph-Memory = skill memory; small-agents-beat-big = the bet).
- Procedure-nodes-that-EXECUTE dropped (too risky). Strategy nodes = DECLARATIVE only.

**SWE code-data pipeline (the v2 data spine, FREE — git+ast+gold patch, no LLM):**
- `swe_load.py` (load SWE-bench/gym + partial-clone checkout), `code_extract.py`
  (Python ast -> symbol nodes: signature+docstring+line spans + symbol_at_line),
  `swe_grounded.py` (gold patch -> AST-mapped support symbols -> v2_grounding trace +
  retrieval gold + code graph).
- **Cheap rung scaled (SWE-bench Lite 300/300):** 300 grounded traces, 300 gold,
  **19,223 code nodes** (19,008 symbols). **Code retrieval mpnet: Hit@5 0.272 / MRR
  0.200** — ~2x HARDER than STEM (issue-symptom vs def-signature, low lexical
  overlap). Motivates a TRAINED ranker + constrained-decode. (Qwen3-Embedding code
  number pending the L40; sentence-transformers segfaults on Windows.)

**#1+#2 RESULTS (A40 cloud, 2026-06-03):**
- #1 scale DONE: SWE Lite(300)+Verified(499) = 799 grounded traces, 799 gold, 21,621
  unique code symbols.
- code retrieval (698 q, 21.6k symbols): mpnet Hit@5 0.281 / MRR 0.196; **Qwen3-Embedding
  Hit@5 0.374 / MRR 0.264** (Qwen > mpnet on code too).
- **#2 ranker VALIDATED** (contrastive bi-encoder, 680 q / 1727 pairs, 2 ep, 73s):
  leakage-free held-out (116 q) raw Qwen 0.353/0.236/Hit@1 0.112 -> **trained ranker
  Hit@5 0.414 / MRR 0.300 / Hit@1 0.190** (+6pts Hit@5, +27% MRR, +70% Hit@1).
  Training lifts code retrieval -> the "train a ranker" thesis is PROVEN. Big headroom
  (no hard negs, no graph topology, 2 ep). Run: `TRAIN_RANKER=1 bash scripts/cloud_run.sh`.

**Current plan (in progress): 1+2+3, STEM Q&A dropped.**
1. **Scale SWE -> Verified (500) + gym** (running). 2. **Trained GNN-ranker** (the
   0.27 floor demands it; train on code+STEM gold, contrastive). 3. **Strategy nodes
   via opencode session-graph** — repurpose opencode (was STEM Q&A) to distill
   (issue+gold patch) -> strategy/reasoning subgraph via V4's audit/apply (the SESSION
   lane). symbols ground tokens; strategies ground approach.
- **STEM regen HUNG** at 108/274 (opencode unreliable on long unattended batches);
  killed. 122 v2-shaped traces banked (baseline_oc 108 + baseline_cx 14) -- enough as
  the grounding-trains proof; finish remainder on a cheap cloud opencode box if wanted.
- **codex**: pure-gen isolated (read-only sandbox + scratch cwd + guard; writes
  verified blocked) for the self-contained STEM regime; for AGENTIC code it explores
  in an ISOLATED throwaway workspace (not the project repo). Save codex budget (~20%)
  for the SWE agentic rung, not STEM.

**Deploy/infra:** `v5/lm_loader` env-driven precision (V5_LM_QUANT=4bit,
V5_LM_TRUST_REMOTE_CODE=1 for Qwen3.5). `scripts/cloud_run.sh` on an L40 runs the
embedder A/B + realstack + (now) code retrieval. `grown_graph4.json` (6874 nodes,
committed) is the current STEM graph; SWE code graph is separate.

**Key commands:**
- code data: `python -m v5.graph_grower.swe_grounded --dataset lite --limit 300`
- code retrieval: `python -m v5.graph_grower.retrieval_eval --nodes-file <swe_code_candidates> --gold-file <retrieval_gold_code> --embedder mpnet`
- cloud: `export HF_TOKEN=...; bash scripts/cloud_run.sh`
- finish STEM (cloud box): `GRAPH=graphs/grown_graph4.json SKIP_EXISTING=1 RUN_ID=cloud_oc BACKEND=opencode bash gen_and_push.sh`

---

## Session update (2026-06-02i) — Qwen-0.5B extractor SMOKE-TRAINED + verified (opencode-retirement path live)

- LoRA-SFT'd Qwen2.5-0.5B on the 225 collected pairs (all cot/math). **bf16 LoRA,
  NOT QLoRA** — 0.5B fits ~1 GB so 4-bit quant is pointless/lossy here. RTX 4050
  (6 GB). 87 steps / 12.5 min, **train_loss 0.527, token-acc 0.92**. Adapter saved
  to `models/extractor-0.5b` (35 MB, gitignored).
- **Verified end-to-end:** `load_extractor_fn(model_dir)` -> `extract_fn` ->
  `parse_extraction` -> `conform_edits` on 5 held chunks: **4/5 parseable +
  vocab-correct** (atomic reasoning_atoms, valid edges; 1 empty). The student
  reproduces the teacher's extraction format and drops into
  `extract_documents(extract_fn=...)`. Opencode-retirement loop is real.
- Real bugs fixed in `train_extractor.py` to get training+inference working on this
  Windows box (trl 1.5 / transformers 5.9):
  - import `datasets` BEFORE `torch` (else native pyarrow/OpenMP **segfault**).
  - launch with `PYTHONUTF8=1` (trl reads a UTF-8 file w/ cp1252 -> charmap error).
  - `SFTConfig`: `max_seq_length` -> `max_length` (trl rename).
  - `generate`: `apply_chat_template(..., return_dict=True)` + `**enc` (tf 5.x
    returns a dict, not a tensor).
  - inference truncates input (`max_input_tokens=1536`) — the cot chunk branch is
    uncapped so a 12k-token chunk OOM'd attention.
  - default `max_length` 4096 -> 1536 (+ gradient_checkpointing): p90 of pairs is
    ~826 tok; 4096 spilled to shared VRAM (9.9 GB, 33 s/it). 1536 -> 8 s/it, fits.
- **trl install bumped transformers 4.54 -> 5.9.** Verified safe: core modules
  import; reasoning suite **461 passed / 1 failed** (the 1 is a missing data
  fixture, not an API break).
- **Next:** scale + diversify the SFT corpus (more fetch_cot batches across
  physics/chem/bio/logic + fact-mode docs, all `--collect-sft`), retrain, eval the
  student vs opencode, then swap it in for `extract_fn` to grow the graph locally.

---

## Session update (2026-06-02h) — batch-1 grow: stitch + hub-wiring (gate PASS, health UP)

- Ran the full batch-1 extraction (`extract --link-graph --collect-sft`, opencode):
  12 OpenThoughts math docs → 235 chunks → **2411 candidates (1262 nodes / 1149
  edges)**, **225 SFT pairs** banked to `data/external_kb/extractor_sft.jsonl`.
- **Apply FAILED hard first: health 0.6948 → 0.4145 (−0.2803).** Diagnosed via
  `graph_growth_apply.json`: +242 new components + 40 orphans. Root cause = the
  extractor only links nodes WITHIN a chunk (`n0..nk`), never across chunks/docs,
  so each of the 235 chunks became its own island.
- **Fix 1 — `stitch_candidates()` (extract.py):** union-find over the batch, add a
  minimal bridge between consecutive disconnected nodes (chain_step within a cot
  doc, related across docs). 244 bridges → components 250→8, orphans 40→0,
  connectivity 0.33→0.92. Delta −0.28 → **−0.11** (still FAIL).
- **Residual cause:** `graph_health.hub_reachability_3hop` only counts nodes within
  3 hops of a `node_type=="hub"` node, and all 40 existing hubs are old-domain →
  bulk-added math scored ~0 reachability. No hub-wiring/rewire pass existed.
- **Fix 2 — `wire_hubs()` (extract.py):** per source doc, add one topical `hub`
  node linked to every node of that doc (1 hop → reachable), thread per-doc hubs
  through a parent `kb_hub_external_root` into the existing mesh (optional
  `link_existing_hub`, used `algo_design_hub`). 12 hubs + parent.
- **Result — gate PASS, health IMPROVES: 0.6948 → 0.8497 (+0.1549).**
  components 8→7, connectivity 0.79→0.94, clustering 0.39→**0.57**, reachability
  0.60→**0.84**, orphans 0, hubs 40→53. `graphs/merged_graph.json` (831n/1454e)
  → `graphs/grown_graph.json` **2092n / 4108e** (+1261 nodes).
- Both passes default ON in `extract_documents` (`stitch`, `wire_hub_layer`);
  `--no-stitch` to disable. 12 extract tests pass (added stitch + hub-wire tests).
- **Takeaway:** graph growth is a 3-stage pipeline now — extract → **stitch (no
  islands) → hub-wire (reachable)** → gated apply. A bulk new-domain dump must be
  hub-wired or it tanks reachability even when fully connected.
- **Next:** more batches (physics/chem/bio/logic via fetch_cot) to balance the
  math-heavy graph; grow SFT corpus toward enough pairs to train the Qwen-0.5B
  extractor; then point V5 at `grown_graph.json` and re-check label coverage.

---

## Session update (2026-06-02g) — extractor --link-graph + --judge (no more islands)

- Answered: at apply, the model only saw graph state for exact-id skip + edge-
  endpoint validation; paraphrases DUPLICATED (id=sha1(text)) and external nodes
  attached only to each other → ISLANDS (the −0.008 health dip). Fixed.
- `extract.py` now does entity resolution vs the TARGET graph (`--link-graph`):
  - cosine ≥ dup_threshold (0.92) → "duplicate": drop new node, remap its edges
    onto the existing node.
  - attach_threshold..dup (0.80–0.92) → "ambiguous": KEEP node + add a `related`
    attach edge to the nearest existing node → joins topology, no island.
  - `--judge`: edit_judge accept/reject/merge_into (merge → remap onto target).
- Live smoke (sample docs vs merged_graph, --link-graph): **3 nodes attached** to
  the existing dijkstra/bellman cluster:
  - "Dijkstra assumes non-negative weights" → `dijkstra_requires_nonnegative_edge_weights` (0.86)
  - "Dijkstra can produce wrong paths" → `wrong_shortest_path_may_be_negative_edge_hyp` (0.87)
  - "Bellman-Ford handles negative edges" → `bellman_ford_handles_negative_edges` (0.82)
  - scalar/vector facts: no existing match → stay a valid NEW component (correct —
    can't link to knowledge that isn't there yet; interlinks as more physics lands).
- epistemic_state: confirmed the extractor correctly does NOT emit it — it's a V4
  runtime verification judgment (16-d status embedding, status-conditioned pool),
  not external fact content. Owned by the session/substrate path.
- 9 extract tests pass (incl. attach / dedupe-remap / judge-merge, offline stubs).
- Full clean chain: `fetch_cot → extract --link-graph [--judge] → apply --candidates`.

---

## Session update (2026-06-02f) — FIX: extractor node types must match GNN design

- Verification caught a real bug: the extractor emitted `concept` (fact mode) and
  `chain_step` (cot node type) — NEITHER is in `v5.gnn_encoder.NODE_TYPE_VOCAB`
  NOR in any `v5.subgraph` pool. Such nodes become `unknown` AND fall in no
  planning/evidence pool → **invisible to both cross-attention layers**. The
  scalar/vector smoke nodes were `concept` → would never have been attended.
- Fix (`extract.py`): emit only pooled design types + alias common LLM outputs:
  - fact mode → `fact`, `claim` (EVIDENCE pool); aliases concept/definition/
    principle→claim, theorem/law/equation→fact
  - cot mode → `reasoning_atom`, `reasoning_chain`, `strategy` (PLANNING),
    `solved_subgoal` (EVIDENCE); aliases chain_step/step→reasoning_chain, etc.
- Added a **drift-guard test** that imports the real GNN vocab + pools and asserts
  every emittable type is embedded AND attended (can't silently diverge again).
- Re-ran live opencode extraction: types now fact/claim/reasoning_atom/
  reasoning_chain/solved_subgoal, **all in a pool** (fact/claim/solved_subgoal→
  EVIDENCE, reasoning_atom/chain→PLANNING). 6 extract tests pass.
- Confirmed Q: extraction uses **opencode** (big default, no --model, cost 0).

---

## Session update (2026-06-02e) — CoT dataset chosen + HF adapter

- Question-bank domain mix (from ids): cs 42, physics 39, algo 39, math 36,
  sysdesign 36, logic 28, chem 27, bio 27 (+ extreme variants). STEM + reasoning.
- Dataset → domain map decided:
  - **`open-thoughts/OpenThoughts-114k`** (primary) → math/algo/cs + partial science
  - `camel-ai/{physics,chemistry,biology}` (CC-BY-NC) → science depth
  - `csitfun/LogiCoT` → logic
  - sysdesign → NO CoT dataset exists; synthesize via opencode or use fact mode
- Built `v5/graph_grower/fetch_cot.py` — streams OpenThoughts-114k **metadata**
  config (`problem` + `deepseek_reasoning` + `deepseek_solution` + `domain`),
  filters by OT domain + optional keywords, emits `{id,text,domain,mode:"cot"}`
  for the extractor. Streaming = no 3.5 GB download.
- Live smoke (datasets 4.8.5 installed): fetched 3 real math traces; HF dep is
  lazy so the transform is unit-tested offline (5 tests pass).
- Caution: OT reasoning traces are long (12k–42k chars) → many chunks/LLM calls;
  use modest `--limit` + keyword targeting so growth lands on question-bank topics.
- Full chain now runnable:
  `fetch_cot → extract → apply --candidates → grown graph`.

---

## Session update (2026-06-02d) — Source B: external-knowledge extractor (facts + CoT)

- Built `v5/graph_grower/extract.py` — external docs (Wikipedia/paragraph OR CoT
  traces) → ATOMIC graph-edit candidates in the SAME `raw_edit` schema, so it
  plugs straight into the existing apply path. Shared extractor, two modes:
  - `fact`: paragraph → atomic claim/concept nodes + entails/supports/contradicts/related
  - `cot` : reasoning steps → reasoning_atom/chain_step/solved_subgoal + chain_step/leveraged
- Refactored `apply.py`: new `apply_candidates(...)` core + `--candidates <queue>`
  CLI, so external queues apply through the same health gate / provenance / staging.
- Pipeline: chunk → LLM extract (opencode big, injectable) → conform (vocab +
  atomicity 12–400 chars + stable sha1 ids) → [optional] entity-resolve
  (`semantic_dedupe.classify`) → [optional] judge (`edit_judge`) → candidate queue.
- **Why this matters:** session/Phase-B growth only adds reasoning substrate, NOT
  new facts. This is the first path that injects atomic declarative knowledge —
  the thing that was missing (e.g. "scalar vs vector" had no node → false grounding).
- Smoke (opencode) on `data/external_kb/sample_docs.jsonl` (1 physics paragraph +
  1 Dijkstra CoT): **16 atomic nodes / 15 edges**, all typed + LINKED:
  - `[concept] A scalar is a physical quantity described by magnitude alone` (5 edges)
  - `[concept] A vector ... has both magnitude and direction` (5 edges)
  - `[concept] Vectors require direction ...; scalars do not` (2 edges)
  - CoT: reasoning_atom/chain_step/solved_subgoal for Dijkstra→Bellman-Ford choice
  - apply onto merged: 831→847 nodes, health 0.695→0.687 (delta −0.008, gate PASS)
- The scalar/vector gap is now FILLABLE: that question would anchor to the scalar
  concept node instead of bond_polarity.
- 10 graph-growth tests pass (3 audit + 3 apply + 4 extract).
- **Next:** expose `--link-graph` (build dedupe index → link new facts to existing
  nodes; lifts health, kills dup bloat) + `--judge`; then point V5 training at the
  grown graph and measure coverage / override-negative-rate drop.

---

## Session update (2026-06-02c) — graph grower Phase B: gated apply

- Built `v5/graph_grower/apply.py` — turns audit queues into actual graph growth,
  non-destructive. `python -m v5.graph_grower.apply --lanes substrate`.
- Why it matters: bridge.py keeps a substrate node only if `nid in graph.nodes`
  (lines ~199/210); proposed-but-unpersisted substrate nodes float with NO edges
  ("R-GCN message passing is shallow"). Phase B persists them so they gain
  TOPOLOGY for real message passing.
- First real apply (substrate lane, on the 179-row sessions.jsonl):
  - `graphs/merged_graph.json` 831 nodes / 1454 edges
    → `graphs/grown_graph.json` **1305 nodes / 2218 edges** (+474 nodes, +764 edges)
  - **health 0.6948 → 0.7208 (delta +0.0259, gate PASS)** — growth improves health
  - 88 dangling edges dropped (session-local `claim_v4_*_h_1` hypothesis endpoints)
  - every grown node/edge stamped `metadata.auto_grown / batch_id / grow_lane /
    grow_source_session` → ablatable / rollback-by-batch
- Safety: refuses to write onto the base graph (`--out` must differ); health gate
  (`degradation_threshold=-0.02`, `--force` to override); backup of `--out` if it
  exists; `--dry-run`. Generated graph + reports are gitignored (regenerable).
- 6 graph-growth tests pass (3 audit + 3 apply).
- **Critical-path note:** this ran on the OLD-schema corpus (175/179 rows lack the
  new quality schema). Substrate growth is permissive so that's OK, but persistent
  promotion (strict lane) and the true gate read still wait on the 288→1000 regen
  with override-detection labels. Order remains: GROW → REGEN CORPUS → TRAIN.
- Next: point the V5 bridge/training at `graphs/grown_graph.json` (vs merged) and
  measure whether substrate topology lifts planning-label coverage / plan P@1.

---

## Session update (2026-06-02b) — V4 1-sample verify + override-detection fix

- Generated 1 V4 sample via opencode (free big default model, cost 0) to verify
  the pipeline: `python run_gen_llama.py --backend opencode --opencode-config-dir
  pure-opencode --out-dir artifacts/datagen_probe --run-id verify_v4_0602 --limit 1`.
- Wiring confirmed RIGHT: row well-formed, `outputs.answer_support_ids` present,
  flows through projection. cost 0.
- **But the sample exposed a real label bug — FALSE GROUNDING.** Q="scalar vs
  vector" has no matching node; V4 anchored on junk, read a category-mismatched
  node (`bond_polarity_depends_on_electronegativity_difference`), the model
  explicitly rejected it and answered from its own knowledge — yet finalize set
  `shortcut_anchor_ids=[that node]` → `answer_support_ids=[that node]` →
  `v5_label_status=positive`. V5 would learn a junk node "supports" the answer
  (poisons epistemic/fallback).
- **Fix (option 1): override-detection.** `reasoning/distillation_corpus.py`:
  if a FINALIZED answer has lexical_overlap(answer, support-node-text) < 0.08,
  set `answer_support_ids=[]`, `v5_label_status="unsupported"`,
  `quality.answer_overrides_graph=True`. `v5/training/projection.py` honors the
  flag: keeps candidate pool + planning, ZEROes support/evidence/evidence-loop →
  trains as a clean NEGATIVE (epistemic ~0 on attended node → fallback fires).
- Verified by regenerating the same sample: support_target {bond_polarity:24}→{},
  evidence {bond_polarity:34}→{}, label positive→unsupported, planning + pool
  preserved. Positive regression check (dijkstra probe): support intact, stays
  positive (missing flag → not overridden, backward-compatible).
- **Take on root cause:** graph too small is the *frequency* driver (expand graph
  long-term, also needed for positive:negative class balance), but override-
  detection is the *correctness* fix and is independent of graph size — these
  override traces, correctly relabeled, are gold negatives that calibrate the
  fallback gate. Skipped upstream question-filtering (option 2): it discards that
  signal and misses partial overrides. Track positive:negative ratio when scaling.

---

## Session update (2026-06-02) — answer_support_ids wired into projection

- V4 now emits `outputs.answer_support_ids` (commit 92ff383): nodes the FINAL
  answer rests on. Finalize → `shortcut_anchor_ids`; loop-finalized → cited
  `read_node` ids; non-finalized → empty.
- **Wired `v5/training/projection.py` to consume it** as the dominant
  support/evidence signal: `_add_many(support, answer_support_ids, 5.0)`,
  `_add_many(evidence, ..., 4.0)`, `_add_many(outer, ..., 1.5)`. Outweighs
  trajectory-derived weights so V5 support/epistemic targets match what the
  answer actually cited. Also exposed standalone `answer_support_ids` field +
  `diagnostics.answer_support_count`.
- End-to-end path verified on probes (re-projection):
  - probe1 (finalized): 3 answer_support_ids dominate `support_target`
    (24.2 / 23.3 / 21.3) and top evidence — answer-grounded nodes on top.
  - probe2 (non-finalized, empty): graceful fallback to trajectory-derived
    support, no contamination.
- Path: `outputs.answer_support_ids` → projection `support_target`/
  `evidence_target` → `dataset.support_target_map`/`evidence_target_map` →
  bridge. Single source of truth (`project_corpus_file`); root
  `project_corpus_to_v5_targets.py` delegates to it.
- **Next:** regenerate corpus + scale question bank 288 → ~1000, then re-run
  `fallback_write_diag` to confirm gold_all / applicable-fallback hold or
  improve with answer-grounded labels at scale.

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
- V4 quality checkpoint after inspecting
  `data/session_subgraphs/v4_a8816f136f9b`: the matching corpus sample is
  `data/distillation_corpus/sessions.jsonl` line 175. Question:
  "Design a better way to update neural network than back propogation." The run
  finalized after 10/12 steps with 46 tool calls, but it is not a clean V5
  positive. Slot coverage is only 2/10, controller action counts are
  `REUSE=2 QUERY=3 DERIVE=1 VERIFY=0 FINALIZE=0`, and controller fallback was
  used. The answer introduces external ML methods (`InfoNCE`, `SimCLR`,
  forward-forward, predictive coding) while the graph support is mostly
  analogy/bridge nodes from data structures, DP, topology, networking, and C++.
- The `v4_a8816f136f9b` row shows a task-frame contract smell: a neural-network
  design prompt was assigned required slots like `rank_query`, `pagination`,
  `tie_policy`, `latency_budget`, and `consistency_model`. Those are poor slots
  for this design question and would create noisy V5 slot labels. Treat this row
  as `needs_review` / soft-fallback calibration, not answer-quality supervision.
- Patch quality for `v4_a8816f136f9b` agrees with that read: 37 proposed scoped
  patches, with `needs_review=23`, `accept=8`, `soft_only=6`; 20 were medium
  risk. The accepted patches are mostly epistemic/failure-pattern scaffolding,
  while the main claim/strategy patches inherit `needs_review` or have very low
  evidence support. The row also predates the newest quality fields
  (`training_eligible`, `v5_label_status`, `finalization_quality`), so it needs
  post-hoc filtering if kept in the corpus.
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
- Push note for the live-steps checkpoint: V5 backend/docs are on
  Agentic-Graph-Reasoner `main` at `1eef435`. Frontend demo code is on
  `The-Only-Lazy-Guy/PROJECT` branch `frontend-graph-v5-demo` at `7a782108`.
  The frontend push stayed on that clean branch to avoid dragging unrelated
  local `reasoning-architecture` history.
- Read: the projected pipeline is now real and end-to-end, evidence routing is
  materially stronger than planning. The false-invalidator blocker is repaired;
  the remaining fallback failures are mostly missing slots + low epistemic on
  top evidence. Do not move to Stage 3/4 yet.
- Graph growth/richness planning checkpoint: graph-growth machinery already
  exists, but it is split across several paths. `answer_query_v4()` produces raw
  `graph_edits`, scoped patches, validation summaries, and session artifacts;
  `reasoning.scoped_edits` adds scope/evidence/risk/support validation;
  `reasoning.graph_editor` can apply edits with backups and health checks;
  `scripts/run_batch.py --apply-edits` supports inline accumulation;
  `scripts/process_session.py --apply` supports offline reflection edits; and
  `v5.training.substrate` can build a substrate-enriched graph for V5 from safe
  scoped patches.
- Baseline from `data/distillation_corpus/sessions.jsonl` on 2026-06-02:
  175 rows, all with scoped patches. Patch statuses are `accept=1416`,
  `soft_only=553`, `needs_review=643`, `reject=31`. Main patch types are
  `add_relation=1229`, `reinforce_existing=674`, `add_epistemic_state=204`,
  `add_strategy=153`, `add_fact=100`, `add_reasoning_atom=82`,
  `add_control_rule=81`, and `add_solved_subgoal=78`.
- Baseline graph richness: `graphs/merged_graph.json` is 831 nodes / 1454 edges.
  Building substrate from the current 175-row corpus would add 464 safe substrate
  nodes and 788 relations, yielding 1295 nodes / 2242 edges in
  `graphs/merged_graph_substrate.json` if regenerated locally. This substrate
  path is good for V5 planning/evidence supervision, but should remain separate
  from persistent graph promotion.
- Proposed graph-growth control contract: three lanes, not one giant apply
  button. Lane A: persistent graph promotion, only high-confidence accepted
  patches with good support, low review risk, no conflicts, and healthy graph
  delta. Lane B: V5 substrate growth, broader accepted/soft patches including
  strategy, solved_subgoal, reasoning_atom, failure_pattern, control_rule, and
  epistemic_state. Lane C: manual review queue for `needs_review` patches,
  especially creative design-synthesis analogies and weakly verified claims.
- First implementation slice is now live as `v5.graph_grower.audit`. It is
  intentionally non-mutating: it reads session/corpus JSONL, computes base graph
  health, summarizes patch status/type/risk by task family and session, flags
  row-quality issues, and writes three queues: Lane A persistent promotion,
  Lane B V5 substrate, and Lane C review.
  ```
  $env:PYTHONPATH="E:\PROJECT\graph_v5"
  python -m v5.graph_grower.audit ^
    --corpus data/distillation_corpus/sessions.jsonl ^
    --graph graphs/merged_graph.json ^
    --out artifacts/graph_growth/graph_growth_audit.json
  ```
- Graph-growth audit result on 2026-06-02: 179 rows, all with scoped patches;
  2707 scoped patches total (`accept=1467`, `soft_only=561`,
  `needs_review=647`, `reject=32`). Base graph is still 831 nodes / 1454 edges.
  Strict Lane A found only 2 persistent candidates; Lane B found 1419 substrate
  candidates with an estimated +472 nodes / +762 edges; Lane C found 1837
  review/blocked candidates. Row flags: `missing_quality_schema=175`,
  `controller_fallback_used=97`, `low_slot_coverage=57`,
  `suspicious_design_slots=17`, `answer_overrides_graph=1`,
  `weak_label_status=1`.
- Read: this is the right conservative shape. Old-schema rows and weak V4 runs
  can still provide V5 substrate signals with warnings, but they are blocked
  from persistent graph promotion. Do not auto-promote design-synthesis analogy
  claims like `v4_a8816f136f9b`; send them through review or a stricter evidence
  audit first.
- Second implementation slice should be a conservative offline apply command:
  load the queue, apply only Lane-A candidates into a graph copy, compute
  before/after `graph_health`, write a backup and health report, and require an
  explicit `--apply` flag to mutate `merged_graph.json`. Do not promote
  design-synthesis analogy claims like `v4_a8816f136f9b` unless they pass manual
  review or a stricter evidence audit.
- Third implementation slice should add tool/procedure richness as graph nodes:
  represent callable tools/procedures as typed nodes (`procedure`, `tool`,
  `control_rule`) with edges such as `requires_slot`, `produces_slot`,
  `invalidated_by`, `uses_tool`, `verifies`, and `fallback_route`. V5 should
  learn these as planning/control substrate, while V4 can retrieve them for
  better tool choice and verification behavior.

## Session update (2026-06-02) — full-recipe control test: NOT undertraining; it's the label/gate contract

- Ran the SAME diagnostic + corpus (`data/corpus_merged_v5proj.jsonl`, 288 proj,
  Qwen2.5-0.5B, `--seed 7`) at the **full recipe `e1=200 e2a=120 e2b=150`** (vs the
  `30/20/20` diagnostic schedule). Log:
  `artifacts/run_logs/fallback_write_diag_fullrecipe_seed7.log`.
- Stage 1 loss converged 7.53→1.48 (at epoch 40 it was 3.14 — i.e. `e1=30` quit at
  ~2× final loss; the heads ARE better trained now), BUT:
  **applicable fallback = 0.57** — same band as `30/20/20` (0.43–0.60).
  **7× training did NOT move applicable fallback.** Undertraining hypothesis REFUTED.
- Oracle ablations (held-out) are the decisive finding:
  `predicted_all 0.57 · gold_slots_only 0.47 · gold_epi_only 0.55 · gold_inv_only
  0.57 · gold_all 0.45`. **Even with ALL gold labels, applicable fallback floors at
  0.45** → not a head/training/capacity problem; the **fallback gate's "answerable"
  definition is inconsistent with the corpus labels** for ~45% of applicable cases.
- Where the mismatch lives (both on direct_judgment + design_synthesis):
  1. SLOTS: `failed slots reason 13 / verdict 10`; `gold_slots_only` only 0.47 →
     the GOLD slot target itself under-marks required slots (same `reason`/`verdict`
     underfill seen in the friend's opencode trace: `required=[answer,reason]`,
     `filled=[answer]`). This is a **V4-side slot-detection** problem in the labels.
  2. EPISTEMIC: `right_evidence_low_epi 12`; `gold_epi_only` barely moves (0.55) →
     the epi-on-primary gate and the gold-epi-positive node don't align.
- Safety floor holds at full training: blocked/negative fallback = 1.00 across
  EVERY oracle variant; negative total write lowest at 0.102; no negative
  `shortcut_verified` leaks. Combined with the earlier hard/soft result (applicable
  fallback = 100% soft), **"applicable fallback rate" is a mis-specified Stage-3/4
  gate.** The legitimate gate (safety) is met.
- VERDICT / next step: do NOT chase this with more training, Stage 3/4, or a new
  head. Fix the LABEL CONTRACT — V4 `reason`/`verdict` slot detection (mark filled
  when the answer contains them) and the epi-on-primary definition — then either
  applicable fallback drops because the gate matches labels, or proceed to Stage
  3/4 on the (already-met) safety floor.

## Session update (2026-06-02 b) — EPI LABEL FIX is the unlock (gold_all 0.45 → 0.06)

- 5-trace careful inspection (`data/corpus_inspect5.jsonl`, worst finalized
  direct_judgment, raw `filled=[]`):
  - SLOTS already correct (dataset.py finalized fix yields `[verdict,reason]`) →
    stale-slot hypothesis REFUTED.
  - EPISTEMIC under-marked: finalized traces cite 1-5 evidence nodes but
    `epi_target` marked only the SINGLE support node. The gate checks epi on the
    PRIMARY attended evidence node → routing onto any other cited node → epi 0 →
    fallback. This was the `right_evidence_low_epi` bucket and the `gold_all=0.45` floor.
- FIX (`bridge.py`, commit `ce24679`): for finalized traces mark ALL cited
  evidence-pool anchors epi-supported (+ support node + raw accessed evidence).
  No re-projection needed (applied at load time). Verified on the 5: epi now
  covers all cited evidence.
- End-to-end confirmation on full 288 (same diagnostic, `--seed 7 --e1 30 --e2a 20
  --e2b 20`, log `fallback_write_diag_epifix_seed7.log`):
  ```
                    before    after
  applicable fb      0.57  →  0.21
  gold_slots_only    0.47  →  0.17
  gold_epi_only      0.55  →  0.21
  gold_all           0.45  →  0.06   <- label floor lifted
  blocked/negative   1.00  →  1.00   <- safety preserved
  ```
  The 45% floor was a mislabeled epistemic contract, NOT training/capacity/Stage-3/4.
- Remaining gap (predicted 0.21 vs gold-floor 0.06) is head calibration, which the
  full recipe (loss 3.1→1.48) should close. Next: full-recipe rerun to push
  predicted toward 0.06, then Stage 3/4 are genuinely unblocked (safety floor met).

## Session update (2026-06-02 c) — full recipe does NOT beat short; epi-fix is the lever, residual is data/variance

- Full recipe (200/120/150) WITH the epi fix, same `--seed 7` 47-applicable split,
  log `fallback_write_diag_epifix_fullrecipe_seed7.log`:
  ```
                  30/20/20+epifix   full+epifix    pre-epifix full
  applicable fb        0.21      →     0.30            0.57
  gold_all             0.06      →     0.13            0.45
  gold_epi_only        0.21      →     0.28            0.55
  blocked/negative     1.00            1.00            1.00
  ```
- Reads:
  1. The EPI LABEL FIX is the confirmed major lever (0.57→0.21-0.30; gold_all
     0.45→0.06-0.13). Solid.
  2. Training length is NOT the remaining lever: full recipe (0.30) did not beat
     30/20/20 (0.21) on the same split — my "train longer -> 0.06 floor"
     prediction was WRONG. ~4 cases / 47 = within variance.
  3. Residual: `gold_all=0.13` = ~13% of applicable cases the model's PRIMARY
     attended node lands OUTSIDE cited evidence (routing/support gap), so even
     gold epi (now marking all cited evidence) can't rescue it. Plus persistent
     direct_judgment/design_synthesis missing_slot+low_epistemic.
  4. n=47 single-seed held-out is noisy; can't separate 0.21 from 0.30.
- Verdict: label contract was the dominant blocker (FIXED). What's left is
  (a) small-held-out variance and (b) a smaller routing/support + slot/epi
  calibration gap — NOT epochs. Next levers: SCALE the corpus (bigger held-out
  -> stable number; opencode shards) + the V4 data-quality fixes (answer-cited
  evidence/support labels, answer-content slot fills) which target exactly the
  residual routing/slot gap. Safety floor still met (blocked/negative 1.00).

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
