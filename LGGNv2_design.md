# LGGN v2 — Graph Uses LM (Reasoner/Realizer Split)

**One principle:**
> The GRAPH reasons; the LM realizes. The latent z carries WHAT-TO-DO (semantic — fits a
> bottleneck); the code SPAN carries the identifiers (literal — free, already in context).
> Each channel does what it physically can. v1 died asking z to do both.

Canonical design + progress doc for the v2 architecture (2026-07-06). Supersedes the
decode-conditioning line of `LGGN_DESIGN.md` (v1: FiLM/MoLoRA conditioning the code generator
directly). The refiner, operator-basis, and graph-edits findings from v1 carry over unchanged.

---

## 1. Why v2 exists — the v1 post-mortem

v1 pipeline: `issue+code reprs → refiner h_K → MoLoRA(z=h_K) conditions a 3B LM → search/replace patch`.

Five verified failures (code audit + molab runs):

1. **z cannot carry content.** MoLoRA ceiling arm (z = GOLD fix repr) = recall 0.120–0.192 with
   66% zero_recall. The recall metric demands literal gold tokens (`.clone()`, `atomic` —
   identifiers that appear nowhere in the input), but z (2048d → 64d bottleneck → 4-expert
   router) transmits semantics only. Conditioning selects strategies; it cannot inject
   identifiers. **Architectural, not tunable.** (FiLM failed for the same reason one level
   shallower — scale+shift can't even select well.)
2. **The task was partly impossible.** `extract_hunks` drops ALL diff context lines; the issue
   was truncated to 700 chars. Gold REPLACE lines frequently reference identifiers invisible in
   the prompt. Part of the 66% zero_recall was unwinnable by any model.
3. **One LM call entangled** understanding + localization + generation. The 3B fails at the
   first two; conditioning can't fix what the model doesn't know.
4. **Eval was noise.** n=29 held, single split. 12% vs 19% across runs = the same noise band
   (if ~15% of instances are 3B-solvable, a random 29-instance draw gives 10–28%).
5. **The best supervision was dropped on the floor.** `ingest_fable5.parse_session` already
   extracted `intent` = the assistant's thinking text immediately before each edit tool-call —
   real (reasoning → edit) pairs at scale. The v1 pipeline read only the old/new strings.

**MoLoRA run history (the evidence):**

```
Run                              baseline   latent   ceiling   note
2ep (1 effective)                0.117      ~equal   0.134     experts unspecialized, noise
8ep (6 effective)                0.121      0.192    timeout   first latent > baseline (+0.071)
8ep ceiling-only (separate run)  —          —        0.120     66% zero_recall, 21% partial, 10% good
```

Ceiling landing in the baseline band despite gold z = the content-through-z line is dead.

## 2. The architecture

```
issue+code reprs → LGGN refiner → h_K     (RETARGETED: trace-repr space, not fix-repr space)
                       ↓
   TRACER LM   (input: old span only; MoLoRA z=h_K)  →  ~100-token reasoning trace
                       ↓
   REALIZER LM (frozen; input: old span + trace)     →  edited span
```

Three components, two trained LM roles:

- **REALIZER (M1)** — `(reasoning trace + old span) → new span`. Trained on REAL Fable-5
  (thinking, old_string, new_string) triples. The falsifiable half: do real reasoning traces
  steer realization at all?
- **TRACER (M2)** — decodes the graph latent into a short trace. Input = old span only (NO goal
  text → z is the only intent channel → any latent-over-baseline delta is attributable to z).
  The trace is semantic and ~100 tokens: it FITS the z bandwidth, unlike literal code.
- **REFINER (exists, retargeted)** — same `lggn_refine`/`_train_refiner` machinery; the target
  repr `f` becomes the GOLD TRACE repr instead of the gold fix repr. h_K now lives in
  reasoning space. New cache key (`lggn_reprs_fable5trace_*`) — old .npz caches don't collide.

**Hard constraints (user-set):**
- Minimal fixed format, ZERO instruction text. Raw completion on the BASE model (no chat
  template). The whole scaffold is two fixed separators (~5 tokens). No prompt engineering —
  conditioning is learned (z) or data (the trace).
- Short sequences (consumer-PC friendly; molab A40 for real runs).
- Qwen2.5-3B now; bigger model later as a drop-in (strategy: prove the SYSTEM at small scale).
- Multi-seed eval with paired per-seed deltas — never again a single n=29 split.
- Falsifiable GO/NO-GO gates decided BEFORE any run.

## 3. Format (the entire LM-visible scaffold)

```
SEP_T = "\n###T\n"          # trace follows
SEP_N = "\n###N\n"          # new span follows

realizer prompt = old + SEP_T + trace + SEP_N      target = new + EOS
notrace prompt  = old + SEP_N                      target = new + EOS     (ablation arm)
tracer prompt   = old + SEP_T                      target = trace + EOS   (M2)
```

- The tracer prompt is a **strict prefix** of the realizer prompt → a future single-pass merged
  model (`old ###T <gen trace> ###N <gen new>`) stays open.
- **No output parsing at inference.** Generated text up to EOS IS the new span. The v1
  `_emitted_replace` regex failure mode (format_fail) ceases to exist.
- Loss masking: prompt and target tokenized SEPARATELY, concatenated
  (`labels = [-100]*len(p_ids) + t_ids + [EOS]`). Exact boundary, matches inference. EOS is
  supervised so the model learns to stop. Qwen2.5 base: no BOS, pad = eos.

## 4. Data — Fable-5 (thinking → edit) triples

`Glint-Research/Fable-5-traces`: 4782 session event-streams (AGPL-3.0, research only).

**Extraction** (`parse_session_triples`, v5/training/ingest_fable5.py): walk each session's
events; carry the latest text/thinking part forward as `intent`; for each edit tool-call with
BOTH `old_string` and `new_string` non-empty, emit
`{goal, intent, old, new, file_path, tool, session_id, fresh}`.
`fresh=True` only for the FIRST edit after a thinking block — consecutive edits share one stale
thinking (one-to-many supervision noise). Note: v1's `_edit_payload` kept only ONE field and
dropped `old_string` entirely; this is why a new extractor was needed.

**Real ingest (2026-07-06):** 4781 sessions → 1863 triples (1284 fresh-intent).

**Filter funnel** (`load_triples`, each stage counted by `--stats`):

```
raw                      1863
goal_ok                  1653      (drop <local-command-caveat> + empty goals)
intent_ok                1653      (drop intent < 10 chars)
edit_ok                  1653      (drop no-ops, empty sides)
fresh_only=True          1191
dedup (old,new)          1007
caps <= (1800,1800)       936      (caps are FILTERS, never truncation of a supervised pair)
trace_ok                  936      (post-transform degenerate-trace guard)
= 936 usable | 737 unique goals (1.3 triples/goal)
len(old) p50 = 145 chars | len(new) p50 = 272 | trace tail-capped at 400 chars (~100 tok)
```

- `trace = collapse_ws(intent)[-400:]` — TAIL truncation: the text immediately before the tool
  call is the concrete plan; earlier text is exploration. Whitespace runs (≥4 spaces) collapsed
  BEFORE the slice so the budget buys content, not code-block indentation (3/936 traces were
  near-blank without this).
- **Sample (real):** trace = "…Opens a `<Box flexDirection=\"column\" height={dynRows}>` to
  allocate the dynamic height…" — old/new spans show exactly that edit. Traces ARE concrete plans.

**Split:** goal-level, `md5(goal[:400])` (builtin `hash()` is process-salted), whole goal-groups
to held until ≥ 20%. Stricter than session-level: the same goal recurs across sessions with
near-duplicate edits. Asserted leak-free.

## 5. Metrics

- **`added_recall` (primary):** recall over gold lines ABSENT from the input span
  (whitespace-normalized line sets, reusing `solution_ladder._norm_lines`). Copy-the-input
  scores exactly **0**. Rationale: str_replace old/new overlap heavily — plain `_fidelity`
  recall rewards a copy-the-input degenerate policy. Instances whose edit adds no new lines
  (pure rearrange/delete) return None and are skipped.
- **copy_rate (diagnostic):** fraction of outputs whose normalized line set equals the input's —
  the collapse detector.
- `_fidelity` recall/exact retained as secondary continuity metrics.

## 6. Experiments + gates

### M1 — realizer (`v5/runtime/lggn_realizer.py`)

| Arm | Train | Eval | Question |
|---|---|---|---|
| `trace` | (old + trace) → new | true traces | do real traces steer realization? |
| `notrace` | old → new | — | span-only floor |
| `shuffled` | (no extra training) | trace model + DERANGEMENT of held traces | content-specificity control |

Protocol: seeds {0,1,2} (fresh goal-split + init each), eval ≤200 held/seed, paired per-seed
deltas. Config: 2 epochs, lr 2e-4, LoRA r=16 q/k/v/o, 4-bit on molab.

**Gates:**
- **G1a** trace added_recall ≥ 0.15 — floor for M2 usability.
- **G1b** trace − notrace ≥ +0.05 AND positive EVERY seed — **the falsifiable claim**.
- **G1c** true − shuffled ≥ +0.05 AND shuffled ≈ notrace (±0.03) — content, not length/format.
- **NO-GO on G1b/G1c ⇒ traces don't carry realization signal ⇒ premise falsified ⇒ M2 is never built.**

### M2 — tracer (`v5/runtime/lggn_tracer.py`, pending M1 gates)

Refiner retarget first: `f = repr(gold trace)`, same `_train_refiner` (K=4, r=512, 48 random
ops — random > KMeans per v1 finding), same goal-split as M1 seed k.

| Arm | z | Question |
|---|---|---|
| baseline | none (pure LoRA) | what does the span alone predict? |
| constant | mean h_K broadcast | extra-params control |
| latent | h_K per instance, z_dropout 0.1 | **THE TEST** |
| ceiling | gold trace repr | max injectable through z |

End-to-end eval: generated trace → frozen M1 realizer (seed-matched checkpoint) → added_recall.
Anchors: gold-trace-through-realizer (upper), M1 notrace (lower). Intermediate:
cos(repr(gen trace), f_trace).

**Gates:**
- **G2** (before tracer training): held cos(h_K, f_trace) ≥ 0.45 AND ≥ raw cos(g,f) + 0.05.
  Fallbacks: contrastive 0.1, composite start g' = hid(goal+old).
- **G3** (ceiling-first, 1 seed, ceiling+baseline only): e2e ceiling − baseline ≥ +0.05.
  If z can't carry trace semantics given GOLD trace reprs, stop — rethink the bridge
  (literature fallback: soft-prefix capsule à la R-Capsule, targeting the trace embedding).
- **G4** (3 seeds, all arms): e2e latent − baseline ≥ +0.03 every seed; trace-cos latent > constant.

## 7. Related work (grounding)

- **Coconut** (arXiv 2412.06769) — latent-space reasoning outperforms discrete CoT. Validates
  reasoning in h_K. We differ: refinement in an external graph module, decoded to text once.
- **R-Capsule** (2509.22131) — plans compressed through a low-capacity bottleneck condition
  generation. Nearly our tracer; we decode z INTO text (interpretable; realizer trains
  independently on real traces).
- **iCLP** (2512.24014) — latent plan space avoids token-level plan hallucination.
- **CodePLAN** (2403.13271) — plan distillation lifts small-model code gen (+130% APPS). We use
  REAL agent traces, not LLM-synthesized plans.
- **Agentless** (2407.01489) — decomposed localize→repair pipeline beats agents on SWE-bench
  Lite. Validates decomposition; v2 is the learned-latent analog.
- **GrACE / Coeditor** (2305.14129 / 2305.18584) — intent-conditioned edit prediction works with
  small models. Validates trace+span → edit.

Nobody combines: real agent reasoning traces as supervision + graph-refined latent → trace
decode + frozen realizer. That's the novel bet.

## 8. Progress log

**2026-07-06 — design + M1 built + validated locally (commits `d35e438`, `dc7b978`):**

- Design settled with user: trace+span realizer input; z-conditioned generation bridge; minimal
  raw-completion format; local-first validation policy (nothing goes to molab unproven).
- `parse_session_triples` + `--ingest-triples` (ingest_fable5.py). Selftest: fresh/stale flag,
  create_file excluded, string-encoded args parsed, both sides kept. **PASS.**
- `lggn_realizer.py`: loader funnel, goal-level split, `_encode_pair` masking, `RawLM`
  (4-bit-or-env + LoRA, optional MoLoRA hooks for M2), added_recall, arms + shuffled control,
  gates, `--selftest/--stats/--smoke`. Selftest (funnel, masking+padding, metrics edge cases,
  split leakage across seeds, derangement, prefix property). **PASS.**
- Real ingest + funnel: **936 usable triples** (predicted band 700–1300). Triples eyeballed —
  traces are concrete edit plans.
- **Smoke #1** (0.5B, RTX 4050, real loop end-to-end): ran; found missing attention_mask
  (pad==eos warning) → fixed; found 3/936 whitespace-blank traces → whitespace-collapse +
  trace_ok filter.
- **Smoke #2**: clean. n=8 metrics are noise at 0.5B/1ep (flip-flopped between runs — exactly
  why the gates demand 3 seeds × 200).
- **Batched training + generation** (user: molab looked hung, VRAM 4GB idle): length-sorted
  batches (uniform padding, bounded memory; batch order reshuffled per epoch — no length
  curriculum), labels -100 on pads, mask from LENGTHS (mask can't be `ids != pad` — the
  supervised EOS equals pad), LEFT-padded batched generation, per-batch progress logs,
  gradient checkpointing on the non-quant path too (local OOM at batch 4 without it),
  `_set_z_batch` for M2. Smoke #3: epoch 24 pairs 60s → 6s. **PASS.**
- Note for Windows local runs: console is cp1252 — use `PYTHONIOENCODING=utf-8` (traces contain
  unicode). Artifacts are always written utf-8.

**2026-07-06 — M1 molab runs: ALL GATES PASS (the premise is validated):**

First molab attempt (350 min, one process, all 6 arms sequential) was walltime-killed before the
single end-of-run results write → resilience fixes (commit `bb0e508`): merge-write results after
EVERY arm, checkpoint saved before eval, shuffled samples dumped, `--seed-list` per-seed jobs.
Buffering fix (`e231217`): piped stdout was block-buffered while tqdm streamed on stderr — runs
looked hung while training fine.

Re-run as 3 per-seed jobs (Qwen2.5-3B 4-bit, 748 train / ~188 eval per seed, 2 epochs,
batch 16 — ~35 min/seed, eval ~1.5-2s/inst):

```
arm        added_recall (3 seeds)         per-seed trace: 0.174 / 0.238 / 0.217
notrace    0.004 ± 0.000
shuffled   0.003 ± 0.001
trace      0.210 ± 0.027

G1a  trace >= 0.15            0.210                    PASS
G1b  trace - notrace          +0.205 ± 0.026 all +     PASS   (41x the floor)
G1c  true - shuffled          +0.207 ± 0.026           PASS
     shuffled - notrace       -0.001  (predicted ~0)   exactly on prediction
```

Reading:
- **The span alone predicts NOTHING (0.004).** All realization signal comes from the trace.
- **A WRONG trace = no trace (0.003 ≈ 0.004).** The signal is 100% trace CONTENT — not format,
  not length, not "any plausible text after ###T".
- **copy_rate drops under tracing** (0.04-0.18 vs 0.19-0.25 notrace): traces pull the model off
  the copy-the-input attractor.
- Training loss: trace arm converges lower (0.64-0.71) than notrace (0.91-0.93) — the trace
  genuinely disambiguates the target.
- All 3 seed checkpoints saved (`artifacts/lggn_realizer/seed{0,1,2}_trace/`) — M2's frozen
  realizers.

**2026-07-06 — M2 G3 (span-only tracer): FAIL, and the failure is the finding:**

Two molab attempts, seed 0, arms ceiling+baseline:
1. 2 epochs / warmup 1 = ONE effective conditioner epoch → ceiling training loss batch-for-batch
   IDENTICAL to baseline (2.845/2.407 vs 2.843/2.407) — the v1 "MoLoRA needs ≥6 effective
   epochs" finding, not carried over. Fixed: defaults 8 ep / warmup 2 + loud WARN (`8f7c4b2`).
2. 8 epochs / warmup 2 (proper training): ceiling loss now DIVERGES (ep8 1.038 vs 1.158) and
   trace_cos moves (ceiling 0.541 vs baseline 0.493) — **z is read and steers** — but e2e stays
   at the notrace floor (ceiling 0.002, baseline 0.005; gold anchor 0.165). **G3 FAIL.**

**Verdict:** z cannot RECONSTRUCT ~100 tokens of specific trace text from one repr through a
64-d/4-expert router — that is embedding inversion (vec2text needs millions of pairs + iterative
correction; we have 936). The v1 bottleneck argument, one level up: moving the target from code
to trace made z's job easier but reconstruction is still the wrong job. z steers; it cannot
dictate content. Also on the books: M2 smoke caught a CRITICAL silent bug first — transformers
≥4.5x returns bare tensors from decoder layers; `output[0]` sliced the first batch row and
broadcasting rebuilt corrupt [B,T,D] batches (every row = row 0). Batch-1 paths (all v1 runs)
unaffected. Fixed version-agnostically in lggn_realizer + lggn_decode (`54878ab`).

**BRIDGE v2 (active):** the tracer sees GOAL TEXT + old span; z only steers.
- The goal is DATA available at inference (hiding it was for attribution cleanliness, not
  deployment realism). Fable-5's goal = whole-session request; the trace = edit-specific plan —
  goal→trace stays a real derivation task.
- `tracer_goal_prompt = goal[:700] + "\n###O\n" + old + "\n###T\n"` (still zero instructions).
  Prefix property vs the realizer prompt is dropped (belonged to the span-only design).
- New **retrieval arm** (free, no LM): h_K nearest-neighbor over TRAIN gold-trace reprs → that
  train instance's trace TEXT. Full text fidelity, wrong instance — measures how much trace
  SPECIFICITY vs generality matters. Smoke: retrieval trace_cos 0.677 > generated arms (real
  text wins in repr space, as expected).
- G3/G4 keep their form; baseline is now goal+span (stronger, honest): G4 = does the GRAPH's
  h_K add anything over the raw goal text — the actual LGGN system question.

**Pending:**
- [x] molab M1 3 seeds → G1a-c ALL PASS.
- [x] M2 span-only G3 → FAIL (z steers, cannot reconstruct) → bridge v2.
- [ ] molab bridge-v2 G3: `--seed-list 0 --arms ceiling,baseline,retrieval` (8ep/warmup2
  defaults), then G4 full matrix per seed.
- [ ] SWE-bench transfer — DEFERRED until M2 passes; needs a localization stage (à la
  Agentless / `code_retrieve.retrieve_support`, the one repr-based span scorer in the repo).

## 9. File map

| File | Role |
|---|---|
| `v5/training/ingest_fable5.py` | `parse_session_triples`, `--ingest-triples` → `data/fable5/realizer_triples.jsonl` (gitignored; regenerate anywhere) |
| `v5/runtime/lggn_realizer.py` | M1: loader funnel, `RawLM`, format constants, added_recall, arms, gates, smoke |
| `v5/runtime/lggn_tracer.py` | M2: refiner retarget (trace-repr space, new cache ns), arms baseline/constant/latent/ceiling, e2e through frozen M1 ckpt, trace-cos diagnostic, gates G2-G4 |
| `LGGN_DESIGN.md` | v1 canonical doc; §Architecture v2 summary + entries 39-40 point here |
| `READ_THIS.md` | per-session raw-results dump |
| Reused untouched | `_MoLoRAModule`, `_train_refiner`, `_find_layers` (lggn_decode.py); `_reprs_from_texts`, `Refiner` (lggn_refine.py); `_norm_lines`/`_fidelity` (solution_ladder.py) |

## 10. Repro

```bash
# wiring (no GPU, no network)
python -m v5.training.ingest_fable5 --selftest
python -m v5.runtime.lggn_realizer --selftest

# data + funnel (network; ~6 min cold)
python -m v5.training.ingest_fable5 --ingest-triples
python -m v5.runtime.lggn_realizer --stats

# local proof of the real loop (0.5B, 6GB GPU, ~3 min)
PYTHONIOENCODING=utf-8 python -m v5.runtime.lggn_realizer --smoke

# M1 (molab A40)
V5_LM_QUANT=4bit python -m v5.runtime.lggn_realizer --seeds 3 --save-ckpt \
    --batch-size 16 --eval-batch 16
```
