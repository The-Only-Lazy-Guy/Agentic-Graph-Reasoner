# READ_THIS — LGGN v2: M1 GATES ALL PASS (2026-07-06)

> At-a-glance dump of the latest session (raw numbers, decisions, repro commands).
> Updated each working session.

## M1 REALIZER — molab, 3 seeds, Qwen2.5-3B 4-bit, ~188 eval/seed

```
notrace  : 0.004 ± 0.000     span alone predicts NOTHING
shuffled : 0.003 ± 0.001     wrong trace = no trace (perfect control)
trace    : 0.210 ± 0.027     seeds 0.174 / 0.238 / 0.217

G1a  trace >= 0.15                     0.210    PASS
G1b  trace - notrace, all seeds +      +0.205   PASS  (41x the floor)
G1c  true - shuffled                   +0.207   PASS  (shuffled-notrace = -0.001, predicted ~0)
```

**Premise validated: real reasoning traces carry essentially ALL realization signal, and the
3B realizer cashes them.** Copy-rate drops under tracing (0.04-0.18 vs 0.19-0.25 notrace).
Trace-arm loss converges lower (0.64-0.71 vs 0.91-0.93) — the trace disambiguates the target.
Checkpoints: `artifacts/lggn_realizer/seed{0,1,2}_trace/` (M2's frozen realizers).

Ops lessons burned in: results merge-written after EVERY arm (walltime kill lost 4 arms once),
stdout line-buffered (piped runs looked hung), per-seed jobs via --seed-list, batched
train/generation (seed = ~35 min).

## M2 TRACER — built, next up

`v5/runtime/lggn_tracer.py`: refiner retargeted to TRACE-repr space (f = repr(gold trace), new
cache ns `lggn_reprs_fable5trace_*`), arms baseline/constant/latent/ceiling, e2e through frozen
seed-matched M1 realizer, trace-cos diagnostic, gates G2-G4. Selftest PASS.

```bash
# G3 ceiling-first (~1.5h) — if z can't carry traces given GOLD reprs, stop:
V5_LM_QUANT=4bit python -u -m v5.runtime.lggn_tracer --seed-list 0 --arms ceiling,baseline
# then full (per-seed jobs):
V5_LM_QUANT=4bit python -u -m v5.runtime.lggn_tracer --seed-list 0
V5_LM_QUANT=4bit python -u -m v5.runtime.lggn_tracer --seed-list 1
V5_LM_QUANT=4bit python -u -m v5.runtime.lggn_tracer --seed-list 2
```

## THE PIVOT (this session)

MoLoRA validated the conditioning MECHANISM but killed the content-through-z line:

```
Run                              baseline   latent   ceiling
2ep (1 effective)                0.117      ~equal   0.134     experts unspecialized
8ep (6 effective)                0.121      0.192    timeout   first latent > baseline
8ep ceiling-only (separate)      —          —        0.120     66% zero_recall
```

- Ceiling (z = GOLD fix repr) lands in the same 0.12-0.19 band as baseline → **n=29 noise**.
- 66% zero_recall WITH gold z: valid-format wrong-content patches.
- **Diagnosis:** z (2048d → 64d bottleneck → 4-expert router) transmits SEMANTICS; the recall
  metric demands LITERAL identifiers (`.clone()`, `atomic`) that aren't even in the input
  (`extract_hunks` drops context lines, issue truncated to 700 chars). Task partly impossible.
- **Conclusion: conditioning selects strategies, it cannot inject content. Architectural, not tunable.**

## Architecture v2 — graph reasons, LM realizes

```
issue+code reprs → LGGN refiner → h_K   (retargeted to TRACE-repr space)
                       ↓
   TRACER LM  (input: old span; MoLoRA z=h_K)  → ~100-token reasoning trace
                       ↓
   REALIZER LM (frozen; input: old span+trace) → edited span
```

- z carries WHAT-TO-DO (semantic, fits the bottleneck); the SPAN carries identifiers (literal, free).
- Supervision that was dropped on the floor: Fable-5 `thinking` immediately before each edit
  tool-call = real (reasoning → str_replace) pairs. `parse_session` extracted it; nothing used it.
- Zero instruction text: raw completion on base Qwen2.5-3B, two fixed separators (`###T`, `###N`),
  ~5 tokens of scaffolding total. No chat template. No output parsing (text-to-EOS IS the new span).
- New metric **added_recall**: recall over gold lines ABSENT from the input span — copying scores 0.
- Related work: Coconut (2412.06769), R-Capsule (2509.22131), iCLP (2512.24014),
  CodePLAN (2403.13271), Agentless (2407.01489), GrACE/Coeditor (2305.14129/2305.18584).

## Data (real, ingested this session)

```
ingested 4781 sessions -> 1863 (intent,old,new) triples (1284 fresh-intent)
filter funnel: raw 1863 -> goal_ok 1653 -> fresh_only 1191 -> dedup 1007 -> caps 936
936 usable triples | 737 unique goals (1.3 triples/goal)
len(old) p50=145 chars | len(new) p50=272 | trace tail-capped at 400
```

Sample (real): trace = "…Opens a `<Box flexDirection=\"column\" height={dynRows}>` to allocate the
dynamic height…" → old/new spans show exactly that edit. Traces ARE concrete plans.

## Built this session (all selftests PASS, no GPU needed)

| What | Where |
|------|-------|
| `parse_session_triples` + `--ingest-triples` (BOTH old+new + intent + fresh flag) | `v5/training/ingest_fable5.py` |
| M1 realizer: loader funnel, goal-level md5 split, RawLM (no chat template), added_recall, arms trace/notrace/shuffled, gates | `v5/runtime/lggn_realizer.py` |
| Design section + entries 39-40 | `LGGN_DESIGN.md` §Architecture v2 |

## Gates (falsifiable, decided BEFORE runs)

- **G1a** trace added_recall ≥ 0.15 (floor)
- **G1b** trace − notrace ≥ +0.05, positive EVERY seed ← the claim
- **G1c** true − shuffled ≥ +0.05, shuffled ≈ notrace (content-specificity)
- NO-GO on G1b/G1c = traces don't carry realization signal = premise falsified, M2 never starts.
- M2 gates: G2 refiner-retarget cos ≥ 0.45 · G3 e2e ceiling−baseline ≥ +0.05 (ceiling-first, 1 seed) · G4 latent−baseline ≥ +0.03 every seed.

## Repro

```bash
python -m v5.training.ingest_fable5 --selftest        # triples extraction wiring
python -m v5.runtime.lggn_realizer --selftest         # funnel/masking/metrics/split
python -m v5.training.ingest_fable5 --ingest-triples  # -> data/fable5/realizer_triples.jsonl
python -m v5.runtime.lggn_realizer --stats            # filter funnel + length percentiles
python -m v5.runtime.lggn_realizer --smoke            # 0.5B, real loop, local GPU (6GB OK)
# molab (A40):
V5_LM_QUANT=4bit python -m v5.runtime.lggn_realizer --seeds 3 --save-ckpt
```

## Status / next

- [x] Design + related-work grounding
- [x] Data layer + M1 code + selftests
- [x] Ingest + `--stats` funnel: 936 usable triples
- [ ] Local `--smoke` (0.5B) — prove the loop end-to-end before molab
- [ ] molab M1 3 seeds → gates G1a-c
- [ ] M2 tracer (`lggn_tracer.py`) — only after G1 passes
- SWE-bench eval DEFERRED until M2 passes (needs localization stage à la Agentless).
