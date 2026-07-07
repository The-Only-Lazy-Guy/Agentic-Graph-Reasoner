# LGGN v3 — Total-Memory Graph + Agentic Loop

**One principle:**
> The graph is TOTAL MEMORY — every single bit, even syntax — in two levels: memory of
> SOMETHING (semantic concepts) SHAPES memory of IMPLEMENTATIONS (episodic verified
> artifacts); the model consumes implementations. The LM proposes and realizes; the
> SANDBOX supplies the state feedback (HRM's missing ingredient); the memory compounds.

Canonical doc for v3 (2026-07-07). Product goal: a cheap, easy-access agentic coder
(Qwen2.5-3B, consumer GPU) that debugs + writes code and GETS BETTER WITH USE. SWE-bench
is a passive metric. Predecessors: `LGGN_DESIGN.md` (v1, substrate + conditioning),
`LGGNv2_design.md` (v2, reasoner/realizer — realizer validated, tracer falsified).

## 1. Why this shape (verdict chain)

- v2 M1: the REALIZER works — (trace + span) → edit, 0.21 vs 0.004/0.003 controls. Text
  content channel is real when content is instance-accurate.
- v2 M2: derivation cannot be moved into the latent or into thin-context generation —
  z steers but never carries content; (goal, span) UNDERDETERMINES Fable-5 edits; flat
  retrieval of whole implementations = 0.002 (wrong bindings poison right strategies).
- HRM analysis: HRM closes task/space/feedback/readout in one trained system. We rent a
  frozen pretrained space — so import what transfers: **state feedback** (failing tests =
  constraint propagation), **stay in-space until forced out** (latents select/steer/rank;
  tokens carry content), **only determined sub-tasks** reach the LM.
- Channel inventory: content moves ONLY via tokens (KV-prefix paid-once, raw-format text);
  zero-token channels (OperatorInjector 5/5, MoLoRA/FiLM, constrained decode) = steering/
  constraints. That split IS the LM↔graph contract, and it is speed-optimal.

## 2. Architecture

```
                 ┌──────────── TOTAL MEMORY (disk, any size) ────────────┐
                 │ L2 SEMANTIC concepts — v5/memory/semantic.py          │
                 │   BehaviorNode (graph_edits engine UNTOUCHED):        │
                 │   emb_skel centroid + ctx preconditions + confidence  │
                 │   lifecycle MINT/STRENGTHEN/WEAKEN/REFINE/MERGE/      │
                 │   SPLIT/RETIRE + poison gate  + members[impl_ids]     │
                 │        │ shapes (conf-gated retrieve + rank)          │
                 │        ▼                                              │
                 │ L1 EPISODIC implementations — v5/memory/episodic.py   │
                 │   {ctx, old, new, trace, outcome, verified}           │
                 │   emb_ctx (when) + emb_skel (what strategy, masked)   │
                 │        │ grounded by                                  │
                 │        ▼                                              │
                 │ L0 SYNTAX — v5/memory/syntax.py                       │
                 │   AST symbols/signatures/identifiers, ident_overlap,  │
                 │   per-file vocab (constrained decode source)          │
                 └──────┬──────────────────────────────┬─────────────────┘
        read: ctx→concepts→member impls→          write: outcome → L1 append,
              local fit (0.6 ctx-cos + 0.4              L2 observe(), L0 upsert
              ident_overlap) → ONE short payload
                     │
   LM channels: content = raw-format slot (v1) / KV-prefix session (P5)
                steering = OperatorInjector on retries (P5)
                constraints = symbol-trie constrained decode (P5)
                     │
   LOOP (v5/runtime/agent_loop.py): data-only SEP prompt → propose → apply →
   SANDBOX (v5/runtime/sandbox.py, ~100ms) → pass: write verified / fail: obs retry (≤3)
```

Storage (v5/memory/store.py): JSONL WALs + float16 .npy embedding shards, cosine search
with within-subset restriction; linear v1, ANN behind the same interface at >50k rows.
mpnet-768 everywhere (RealEmbedder); embedders injected so all selftests run modelless.

## 3. Benchmark A — MBPP + mutation-debug (superseded by B, kept as diagnostic; see §5)

Local sandbox verification in ~100ms (no Docker) → fast episodes → memory actually
accumulates → compounding measurable. DEBUG tasks: gold mutated by 12 operators
(`< ↔ <=`, off-by-one range, negate, and/or swap, const flips...), mutant must COMPILE
and FAIL ≥1 test (equivalents rejected by execution). Splits (deterministic, disjoint,
per-split-only gold derivation): lora_train 398 · pool_a 392 · pool_b 264 · dev 123
(= 1177 tasks from 971 locally-verified golds).

Prompt = DATA only (SEP format): BUILD `spec+assert ###T [memory|obs] ###N` → code;
DEBUG `spec+buggy ###T [memory|obs] ###N` → fixed code. One shared task LoRA (trained
empty-slot + obs-conditioned debug pairs); memory arms differ ONLY in slot content.

## 3b. Benchmark B — repo-continuity project chains (active; `v5/runtime/project_gen.py`)

One instance = a small Python project evolved over an ordered session chain
(CREATE → CROSS → DEBUG → EXTEND). Dependency sessions reference earlier CONVENTIONS
without restating them ("the same line format as inventory receipts") — the seeded
convention (one of several inline format strings / function names / id schemes / rates)
exists only in the repo the agent itself built; a stateless agent must guess, an agent
with L0 (own-repo symbols) + L1 (own past implementations) has it. 2 archetypes
(inventory, logparse) × seeds → 4-5 sessions each; import-proof withholding verified
by selftest (withheld tokens never appear in specs). Agent context per session = spec +
CURRENT target file only.

Arms: **off** (spec + file) / **memory** (+ TotalMemory payload, relevance-gated) /
**ceiling** (+ whole repo dumped in prompt — what memory approximates). Prompt = data-only
SEP format (unchanged from A). Gold-chain LoRA trained on 30 held-out train seeds with a
three-way slot mixture (empty / memory-shaped / full-repo) — applies the MBPP interference
lesson (§5) preemptively instead of discovering it again.

## 4. Gates

**Benchmark A (MBPP, diagnostic):**
- **GM0** loop sanity: arm=off dev solve-rate.
- **GM1** memory earns: concept > flat > off on pool_a; concept − flat ≥ +3pp.
- **GM2** compounding (the product claim): pool_a with write-back into a COPIED memory
  root → pool_b experienced-vs-fresh, same proposer; ≥ +3pp (or −0.3 attempts/solve).
- **GS** speed: memory wall-clock overhead ≤ 15%; payload ≤ ~300 tok (logged per episode).

**Benchmark B (repo-continuity, active):**
- **GB1** memory > off on DEPENDENCY sessions, ≥ +10pp (off structurally lacks the info).
- **GB2** memory reaches ≥90% of ceiling's dependency solve-rate at ≤0.6× ceiling's payload
  tokens (the scale claim — real repos won't fit in a prompt, memory must approximate).

## 5. Progress log

**2026-07-07 — P1 memory package BUILT (commit `f9552e9`).**
store/syntax/episodic/semantic/memory, all selftests PASS (modelless). Real seed:
936 Fable-5 weak-verified impls → 32 KMeans concepts, payload ≤300 tok. Windows notes:
np.save appends .npy to handles-less paths (write via handle); no mmap on active stores
(Windows can't replace open maps).

**2026-07-07 — P2 harness BUILT (commit `a7ebbdb`).**
sandbox (100ms/verify local; timeout/crash/unicode covered), loop_tasks (974 MBPP cached;
mutation generator; pools built: 1177 tasks, splits disjoint), agent_loop (wave-batched
retries proven with stub LM: easy@1 / medium@2-with-obs / unsolvable never; write-back
records both polarities; GM report). Selftests PASS.

**2026-07-07 — MBPP grid v1 (empty-slot LoRA): GM2 "PASS" was poison-differential.**
GM0 PASS 0.626. GM1 catastrophic FAIL with clear mechanism: empty-slot-trained proposer
craters when the slot is filled (0.617 → 0.18, flat≈concept — payload PRESENCE, not
retrieval mode). GM2 read +25.4pp (0.167→0.421 pool_b experienced-vs-fresh) — later
reinterpreted: relevant payload poisons LESS than irrelevant, not positive lift (both far
below the no-memory 0.583).

**2026-07-07 — MBPP grid v2 (relevance gate `MIN_FIT=0.35` + slot-augmented LoRA, `3bbda6e`):**
```
dev:off 0.577 · pool_a off=flat=concept 0.543 (gate silences fable5 junk; mem_tok 0; GS +0.4% PASS)
pool_b: off 0.583 · concept(fresh) 0.587 · concept_exp 0.567 (mem_tok 50) -> GM2 FAIL (-0.020)
att/solve: exp 1.09 vs 1.12-1.14 (weak speed signal). Augmented LoRA cost the baseline
-4/-7pp (payload pairs diluted empty-slot share 80%→36%).
```
**Structural verdict:** write-back stores what the model already SOLVED — self-experience
memory consolidates ability, it does not extend the frontier on independent-task benchmarks
where the model is competent and nothing non-re-derivable exists to remember. MBPP is the
wrong habitat for TOTAL memory: no shared repo, no conventions, no prior decisions.

**PIVOT B (user-selected): repo-continuity benchmark — the deployment habitat.**
Parametric PROJECT CHAINS: one small codebase evolved over N ordered sessions
(CREATE → EXTEND → DEBUG → CROSS). Later specs REFERENCE earlier choices without restating
them ("apply the same tax rate as pricing") — the seeded values live only in the repo the
agent built, and the agent's context is spec + CURRENT FILE only. Memory (L0 symbols of own
repo + L1 own session implementations) supplies what a stateless agent cannot re-derive.
Arms: off / memory / ceiling (whole repo in prompt). Gates:
- **GB1** memory > off on dependency-bearing sessions (the frontier test — off lacks the info)
- **GB2** memory tokens << ceiling tokens at ≥90% of ceiling's solve rate (the scale claim)
Files: `v5/runtime/project_gen.py` (archetype templates, seeded params, gold chains,
withholding verified by selftest), sandbox project mode (multi-file), TotalMemory.read
gains L0 symbol payloads, `v5/runtime/project_loop.py` (chain harness, repo state owned by
the agent, per-depth metrics).

**2026-07-07 — B built: project_gen + sandbox project mode + project_loop (`73a2049`, `248369e`).**
2 archetypes (inventory, logparse), 4-5 session chains, import-proof withholding (33 tokens
verified absent from specs). Bug the harness's own selftest caught in project_gen: writer_gold
for logparse layout-2 built its code TEMPLATE by calling the format lambda with identical
`'{}'` placeholder args for ts/level/msg — positional info collapsed, so the template's slot
order silently diverged from the fixed `.format(ts, level, msg)` call; layout-2 gold failed
its own tests. Fixed with an explicit per-layout (template, arg-order) table. `memory.py`
gains L0 delivery (own-repo symbols, file-mention boost — a spec naming a prior file reliably
retrieves it). `project_loop.py`: agent-owned repo state per chain, arms off/memory/ceiling,
healing (default on, `--no-heal` for entangled realism), gold-chain LoRA (empty + memory-
shaped + full-repo slot mixture — the MBPP interference lesson applied preemptively), GB1/GB2
report. Selftest needed a real fix too: the stub LM originally checked whether the withheld
TEST VALUE (evaluated output) appeared in the payload — but payloads carry source CODE
(templates), not evaluated strings, so it always failed. Rewrote as `_shares_code`
(verbatim-chunk overlap with earlier gold source) — the correct proxy for "is the convention
actually visible." All selftests PASS (off dep=0.00, ceiling dep=1.00, memory dep=1.00 with a
FAKE embedder on the file-mention boost alone).

**2026-07-07 — GB1 PASS on real molab data, then user challenge, then Stages 1+2 planned.**

GB1 first real run (Qwen2.5-3B, 40 chains, 180 sessions): **+0.337** on dependency sessions
(0.362 off → 0.700 memory, after two rounds of same-file-boost fixes — see `5440c90`,
`b94c1f7` for the full debug→extend see-saw diagnosis). Core v3 claim validated on real
repo-continuity data.

User then challenged the architecture directly: *"why are we just in basic RAG + prompt
engineering? I think this design won't last."* Fair, and traced precisely: `LGGN_DESIGN.md`
entry 27's refiner (5-seed CI, ALL 4 pillars load-bearing, cos(h_K,gold)=0.70-0.74) is
genuinely validated HRM-style latent computation — and is used NOWHERE in `v5/memory/*` or
`project_loop.py` (confirmed by grep). What actually failed across v1/v2 was narrower:
DECODING novel content from a compressed latent (MoLoRA ceiling 0.12-0.19, v2 M2 tracer's
embedding inversion, oracle-best-of-8 still 0.027) — not latent computation itself. v3 threw
out the refiner's working capability along with the dead decode step.

**Answer: reuse both proven components for the job each is proven to do, not the job that
failed.** v2 M1's trace-writing skill (validated 0.21) → query formation (Stage 1), not a
z-conditioned generation target. v1's Refiner.Net convergence (validated cos 0.70-0.74) →
candidate ranking (Stage 2), not FiLM/MoLoRA content conditioning. Full plan at
`C:\Users\Ace\.claude\plans\buzzing-puzzling-sunrise.md` (also ADR'd here, §7 below).

**Stage 1a+1b BUILT (`efd85c5`, `91cab62`), all selftests PASS:**
- `lggn_realizer.py`: `SEP_W = "\n###W\n"`, `why_prompt(spec, current)`, `why_pairs(triples)`
  — Call A supervision from REAL Fable-5 `(goal, old, trace)` triples (genuine skill
  transfer, not spec-echo).
- `project_loop.py`: `run_chain` gains `query_mode` (`"spec"` default = byte-identical to the
  GB1-validated path; `"why"` = Call A + why_text as the memory query; `"refiner"` = Stage 2
  stub, gated). `train_lora` mixes `why_pairs` into the one shared LoRA. `_report_gb3`.
  Result-key scheme keeps `"memory"` literal for the default mode specifically so GB3 can
  reuse the ALREADY-VALIDATED GB1 molab result as its baseline without a re-run.
- Caught before running: `TotalMemory.__init__` doesn't accept `query_fn` yet (that's Stage
  2, task-gated) — `run_chain` only passes it when actually set, so Stage 1 constructs
  `TotalMemory` exactly as GB1 did.

**2026-07-07 — Stage 1 local smoke PASS, molab GB3 run: flat headline, real story underneath.**

Local 0.5B smoke: clean, exit 0, no crashes (`why_tok`/`mem_tok` both non-zero, flowing
correctly). Real molab (Qwen2.5-3B, `--train-lora` mixed, then `--run --query-mode why`):

```
              solve   DEP(n=80)  indep   mem_tok  why_tok  by_kind
ceiling       0.961   1.000      0.930   80       0        create 53/60 cross 40/40 debug 40/40 extend 40/40
memory(spec)  0.756   0.700      0.800   109      0        create 50/60 cross 20/40 debug 30/40 extend 36/40
memory_why    0.833   0.700      0.940   113      120      create 54/60 cross 36/40 debug 40/40 extend 20/40
off           0.656   0.362      0.890   0        0        create 49/60 cross 9/40  debug 40/40 extend 20/40

GB1 PASS +0.337 (unchanged, reused from the original run — no re-run needed)
GB3 +0.000 NO-REGRESSION (headline) — but by_kind: cross +16 (+40pp), extend -16 (-40pp),
                                        EXACT cancellation, not noise
```

**Diagnosis (from eyeballing `[why]` log samples — boundary checklist item 2):** SEP_W's
completion habit is an implicit ANSWER-GUESS (code), not natural-language intent —
`def order_id(n): return "ORD-" + "ITEM-{:03d}".format(n)`, not "I need inventory's id
format." For `cross` (copy an id/format string) the guess is usually right, so it works
great as a retrieval query by accident. For `extend` (reuse pricing's SPECIFIC tax
rate/bulk threshold) the guess hallucinates plausible-but-WRONG numbers from OTHER chains'
training distribution — this chain's actual seeded value is structurally unknowable
(`TRAIN_SEEDS`/`EVAL_SEEDS` disjoint) — and that wrong-but-specific guess actively misleads
retrieval away from `pricing.py`, worse than the raw spec text would have.

Root cause traced further: pooled training loss plateaued at **~1.2** vs the code-only
baseline's **0.041** at the same 2 epochs. Not a mix-ratio problem (why-pairs were already
the majority by count, 936 vs 690) — a converged-vs-not gap: repetitive synthetic code
converges almost immediately, so most of the 2-epoch gradient budget was already going to
the harder, still-learning why-pairs; they just hadn't converged.

**Fix built (`3ff0177`):** `_oversample(pairs, factor, seed)` + `train_lora`'s
`why_oversample` param / `--why-oversample` CLI — gives the under-converged class more full
passes per epoch. Selftest covers identity/2x/1.5x-split/determinism. **Boundary checklist
verdict: FAIL on item 2 (not real natural language) — do not advance to Stage 2 yet.**
Retrain with `--epochs 4 --why-oversample 1.5`, rerun `--query-mode why` only (off/ceiling/
memory-spec unaffected, reused as-is) before trusting GB3 as a real Stage 1 verdict.

**Two follow-ups queued (not blocking the retrain), both surfaced by this run:**
- **Reasoning loop gap (user-caught):** `memory.read()` already accepts `obs` and threads it
  into the query embedding, but `run_chain` builds `payload` ONCE before the retry loop and
  never re-reads memory on failure — retrieval is single-shot per session, only generation
  gets the obs feedback. `extend`'s hallucinated-wrong-number failure is exactly the case
  this would fix: fail → obs reveals the guess was wrong → re-query with that signal →
  second attempt gets a chance at the right L1 record. Orthogonal to Stages 1/2.
- **Coverage gap (user-caught):** the 4 session kinds all hand the model a pre-specified
  target file + signature + tests — none test open-ended "here's a vague request, decide the
  structure yourself" work, which is what a real deployed agent's user query looks like.
  `create`/`cross` (100/180 sessions, 56%) DO test from-scratch coding respecting project
  conventions — this isn't a debugging-only benchmark — but fully open-ended greenfield
  planning is a real, separate gap. Proposed as a 5th `project_gen.py` session kind.

**Pending:**
- [ ] molab: retrain (`--epochs 4 --why-oversample 1.5`) → `--run --query-mode why` → GB3 v2
- [ ] Stage 1/2 boundary checklist re-check with real natural-language why_text
- [ ] Stage 2 (gated): `source_session_idx` in `project_gen.py`, `query_fn` in `memory.py`,
  new `memory_refiner.py` (refiner-as-ranker), GB4
- [ ] Reasoning-loop: obs-informed re-query on retry (queued, orthogonal)
- [ ] Greenfield/open-ended session kind (queued, orthogonal)
- [ ] P5 channels (KV-prefix priming — directly relevant: repo memory paid once per session)
- [ ] SWE-bench passive slice (later)

## 7. ADR — why RAG-critique → Stages 1+2, not a rewrite

Recorded because it'll look like scope creep without the reasoning: the user's "this is just
RAG" challenge could have been answered by defending the current design (delivery-as-text is
proven, GB1 passed) or by a rewrite (bring z-conditioning back). Both were wrong. The right
move was narrower: identify EXACTLY what v1/v2 proved (refiner convergence: yes; latent
decode: no) and EXACTLY what v3 was missing as a result (any use of the refiner at all), then
re-point the validated piece at a job that doesn't require decode. Two falsifiable stages,
strictly sequential (Stage 2's query input IS Stage 1's output), each with its own gate and
an explicit "what a FAIL means" so a negative result still teaches something instead of just
being scope creep that didn't pan out.

## 6. Repro

```bash
# selftests (no GPU/network)
python -m v5.memory.store --selftest && python -m v5.memory.syntax --selftest
python -m v5.memory.episodic --selftest && python -m v5.memory.semantic --selftest
python -m v5.memory.memory --selftest
python -m v5.runtime.sandbox --selftest && python -m v5.runtime.loop_tasks --selftest
python -m v5.runtime.project_gen --selftest && python -m v5.runtime.project_loop --selftest
python -m v5.runtime.agent_loop --selftest

## Benchmark A (MBPP) — diagnostic, kept for reference
python -m v5.runtime.loop_tasks --fetch && python -m v5.runtime.loop_tasks --build-pools
python -m v5.memory.memory --seed                    # fable5 -> L1, KMeans -> L2 (mpnet)
python -m v5.runtime.agent_loop --smoke
V5_LM_QUANT=4bit python -u -m v5.runtime.agent_loop --train-lora --batch-size 16
V5_LM_QUANT=4bit python -u -m v5.runtime.agent_loop --run --arm off --pool dev --eval-batch 16       # GM0
V5_LM_QUANT=4bit python -u -m v5.runtime.agent_loop --run --arm off --pool pool_a --eval-batch 16    # GM1 (x3 arms)
V5_LM_QUANT=4bit python -u -m v5.runtime.agent_loop --run --arm flat --pool pool_a --eval-batch 16
V5_LM_QUANT=4bit python -u -m v5.runtime.agent_loop --run --arm concept --pool pool_a --eval-batch 16
V5_LM_QUANT=4bit python -u -m v5.runtime.agent_loop --run --arm concept --pool pool_a \
    --write-back --copy-memory-to data/memory_expA
V5_LM_QUANT=4bit python -u -m v5.runtime.agent_loop --run --arm concept --pool pool_b
V5_LM_QUANT=4bit python -u -m v5.runtime.agent_loop --run --arm concept --pool pool_b \
    --memory-root data/memory_expA --result-key pool_b:concept_exp

## Benchmark B (repo-continuity) — active
python -m v5.runtime.project_loop --smoke                          # 0.5B local end-to-end
V5_LM_QUANT=4bit python -u -m v5.runtime.project_loop --train-lora --batch-size 16
V5_LM_QUANT=4bit python -u -m v5.runtime.project_loop --run --arm off        # baseline
V5_LM_QUANT=4bit python -u -m v5.runtime.project_loop --run --arm memory     # GB1
V5_LM_QUANT=4bit python -u -m v5.runtime.project_loop --run --arm ceiling    # GB2
```
(No cross-chain eval batching yet — `run_chain` generates one prompt at a time per session,
since repo state and slot content differ per chain; the LoRA batches within `train_on`.)
