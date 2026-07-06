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

## 3. Benchmark (user-selected): MBPP + mutation-debug

Local sandbox verification in ~100ms (no Docker) → fast episodes → memory actually
accumulates → compounding measurable. DEBUG tasks: gold mutated by 12 operators
(`< ↔ <=`, off-by-one range, negate, and/or swap, const flips...), mutant must COMPILE
and FAIL ≥1 test (equivalents rejected by execution). Splits (deterministic, disjoint,
per-split-only gold derivation): lora_train 398 · pool_a 392 · pool_b 264 · dev 123
(= 1177 tasks from 971 locally-verified golds).

Prompt = DATA only (SEP format): BUILD `spec+assert ###T [memory|obs] ###N` → code;
DEBUG `spec+buggy ###T [memory|obs] ###N` → fixed code. One shared task LoRA (trained
empty-slot + obs-conditioned debug pairs); memory arms differ ONLY in slot content.

## 4. Gates

- **GM0** loop sanity: arm=off dev solve-rate.
- **GM1** memory earns: concept > flat > off on pool_a; concept − flat ≥ +3pp.
- **GM2** compounding (the product claim): pool_a with write-back into a COPIED memory
  root → pool_b experienced-vs-fresh, same proposer; ≥ +3pp (or −0.3 attempts/solve).
- **GS** speed: memory wall-clock overhead ≤ 15%; payload ≤ ~300 tok (logged per episode).

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

**Pending:**
- [ ] local 0.5B smoke (loop end-to-end, arms off+concept)
- [ ] molab: `--train-lora` then GM0 (`--run --arm off --pool dev`)
- [ ] P3: GM1 + GS (pool_a, arms off/flat/concept)
- [ ] P4: GM2 compounding
- [ ] P5: KV-prefix session priming, OperatorInjector retry steering, constrained decode
- [ ] SWE-bench passive slice (after GM2)

## 6. Repro

```bash
# selftests (no GPU/network)
python -m v5.memory.store --selftest && python -m v5.memory.syntax --selftest
python -m v5.memory.episodic --selftest && python -m v5.memory.semantic --selftest
python -m v5.memory.memory --selftest
python -m v5.runtime.sandbox --selftest && python -m v5.runtime.loop_tasks --selftest
python -m v5.runtime.agent_loop --selftest

# data (once)
python -m v5.runtime.loop_tasks --fetch && python -m v5.runtime.loop_tasks --build-pools
python -m v5.memory.memory --seed                    # fable5 -> L1, KMeans -> L2 (mpnet)

# local end-to-end
python -m v5.runtime.agent_loop --smoke

# molab
V5_LM_QUANT=4bit python -u -m v5.runtime.agent_loop --train-lora --batch-size 16
V5_LM_QUANT=4bit python -u -m v5.runtime.agent_loop --run --arm off --pool dev --eval-batch 16       # GM0
V5_LM_QUANT=4bit python -u -m v5.runtime.agent_loop --run --arm off --pool pool_a --eval-batch 16    # GM1 (x3 arms)
V5_LM_QUANT=4bit python -u -m v5.runtime.agent_loop --run --arm flat --pool pool_a --eval-batch 16
V5_LM_QUANT=4bit python -u -m v5.runtime.agent_loop --run --arm concept --pool pool_a --eval-batch 16
# GM2: learn on A into a copy, then eval B fresh vs experienced
V5_LM_QUANT=4bit python -u -m v5.runtime.agent_loop --run --arm concept --pool pool_a \
    --write-back --copy-memory-to data/memory_expA
V5_LM_QUANT=4bit python -u -m v5.runtime.agent_loop --run --arm concept --pool pool_b   # fresh
V5_LM_QUANT=4bit python -u -m v5.runtime.agent_loop --run --arm concept --pool pool_b \
    --memory-root data/memory_expA --result-key pool_b:concept_exp                       # experienced
```
