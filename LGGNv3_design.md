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

**Pending:**
- [ ] local 0.5B smoke (project_loop, arms off+memory)
- [ ] molab: gold-chain `--train-lora`, then GB1/GB2
- [ ] P5 channels (KV-prefix priming — directly relevant: repo memory paid once per session)
- [ ] SWE-bench passive slice (later)

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
