# V5 Graph Grower — Design Doc

> Standalone, **offline** component that expands `MemoryGraph` from two sources —
> normal-session graph edits **and** external documents — using a fine-tuned
> Qwen-0.5B extractor + a big-model judge. Decoupled from inference and from V4
> trace generation. Goal: raise graph coverage so more questions have real
> support → better positive:negative balance for V5 (see READ_THIS 2026-06-02b).

Status: **Phase A audit + Phase B gated apply implemented.** Phase A
(`v5/graph_grower/audit.py`) is non-mutating: audits scoped patches into
persistent/substrate/review queues. Phase B (`v5/graph_grower/apply.py`) applies
the queues to a SEPARATE grown graph (never the base graph), with provenance
stamps, dangling-edge filtering, and a health gate. First substrate apply:
831→1305 nodes / 1454→2218 edges, health 0.695→0.721 (gate PASS).

---

## 1. Why separate

Graph growth and graph *reasoning* are different jobs with different cadences:

- **Reasoning** (V5 adapter) is online, per-question, latency-sensitive, frozen graph.
- **Growth** is offline, batch, throughput-sensitive, mutates the graph.

Coupling them risks a **feedback loop**: a biased grower mutates the graph, V4
generates traces on the mutated graph, those traces train V5 — errors compound.
Separation + a staging/promote gate breaks that loop.

Most of the machinery already exists inside V4; this component **lifts it into a
standalone service**, adds external-doc ingestion, and (optionally) distills the
extractor into a local 0.5B. It is NOT a rewrite.

---

## 2. Reuse map (existing modules — do not rebuild)

| Stage | Module / entry point | Notes |
|---|---|---|
| Propose (session) | `reasoning/post_processing.py:produce_graph_edits`, `apply_graph_edits` | V4 already emits candidate node/edge adds |
| Propose (reflection) | `reasoning/graph_editor.py:edits_from_reflection`, `edits_from_reflection_v2` | reflection → edits |
| Patch shaping | `reasoning/scoped_edits.py:patches_from_graph_edits`, `approved_raw_edits_from_patches` | raw edits ↔ scoped patches |
| Entity-resolve / dedup | `reasoning/semantic_dedupe.py:build_dedupe_index` → `DedupeIndex.query_topk` | nearest existing nodes |
| Judge | `reasoning/edit_judge.py:judge_edit`, `judge_edits_batch` | LLM-as-judge → `accept` / `reject` / `merge_into`; sees edit + 3 nearest + 1-hop; **fail-open** |
| Validate | `reasoning/scoped_edits.py:validate_patches`, `reasoning/graph_health.py:compute_health` | schema + health delta |
| Copy/quality repair | `action_repair.py:repair_add_node_copy` | compress near-duplicate add_node |
| Apply (gated) | `reasoning/graph_editor.py:apply_edits` (`dry_run`, `allowed_tiers=("soft",)`) | inline per-edit |
| Apply (batch + health) | `reasoning/graph_editor.py:apply_edits_offline` (`allowed_tiers=("soft","add","mutate")`, `degradation_threshold=-0.02`) | health before/after FULL batch, no auto-revert |
| Persist | `reasoning/session_to_graph.py:apply_graph_updates`, `save_graph_atomically(backup=True)` | atomic write + backup |

**Vocabulary the grower must conform to:**
- 12 relation types (`reasoning/graph_relations.py:RELATION_TYPE_ID`):
  `entails(0) contradicts(1) overlaps(2) supports(3) leveraged(4) related(5)
  used_signal(6) epistemic_of(7) invalidated_by(8) requires_slot(9)
  transfers_to(10) chain_step(11)`. New relations append at END (keeps R-GCN IDs stable).
- Node types (`reasoning/schemas.py`): `procedure, failure_pattern, session_object,
  strategy, solved_subgoal, reasoning_atom, control_rule, epistemic_state, …`.
- Unknown relation → falls back to `related` (id = len). Extractor output MUST be
  validated against these sets; reject/repair off-vocabulary.

---

## 3. Pipeline

```
            ┌─────────────── SOURCE A: session traces ───────────────┐
            │  data/corpus_shards/*.jsonl  (trace.scoped_patches,     │
            │  metrics.graph_edits, reflection) + judge verdicts      │
            └────────────────────────────────────────────────────────┘
            ┌─────────────── SOURCE B: external docs ─────────────────┐
            │  docs/textbooks/notes → chunk → extract                 │
            └────────────────────────────────────────────────────────┘
                                   │
                                   ▼
   (1) EXTRACT      Qwen-0.5B (fine-tuned)  → candidate {node_type, content, edges[]}
                    [bootstrap: opencode big model; distill → 0.5B once data exists]
                                   │
                                   ▼
   (2) ENTITY-RESOLVE  semantic_dedupe.DedupeIndex.query_topk(content)
                    → exact dup (drop) | near (→ merge_into target) | novel
                                   │
                                   ▼
   (3) JUDGE        edit_judge.judge_edits_batch(controller=BIG)
                    → accept | reject | merge_into  (fail-open)
                                   │
                                   ▼
   (4) VALIDATE     scoped_edits.validate_patches + graph_health.compute_health
                    + action_repair (copy guard)
                                   │
                                   ▼
   (5) STAGE        apply_edits_offline → STAGING graph copy
                    tag provenance: {auto_grown:true, source, batch_id, judge_rationale}
                                   │
                                   ▼
   (6) EVAL GATE    health delta ≥ -0.02 AND positive:negative ratio on a
                    held-out question probe does NOT regress
                                   │
                          ┌────────┴────────┐
                       PASS                FAIL
                          │                  │
                          ▼                  ▼
   (7) PROMOTE   save_graph_atomically   quarantine batch + report
                 (backup) → main graph    (no mutation to main)
```

**Quarantine rule:** nothing reaches the graph used for V4 trace-gen until it
clears the eval gate. Grown nodes always carry `provenance.auto_grown=true` so we
can ablate / roll back a batch by `batch_id`.

---

## 4. The Qwen-0.5B fine-tune

Two candidate roles; **fine-tune the extractor first**, keep judge on big model.

### 4a. Extractor (high volume → local 0.5B)
- **Task:** `text span → JSON {node_type, content, edges:[{relation, target_hint}]}`,
  constrained to the type vocab in §2.
- **Why 0.5B:** repetitive, high-throughput, runs offline/local, no API cost.
- **Bootstrap:** run opencode big model over external docs → judge-filter → keep
  accepted extractions as gold. Then distill into 0.5B (SFT).
- **Decoding:** constrained / grammar to the type vocab; post-validate, repair
  off-vocabulary instead of dropping.

### 4b. Judge (lower volume, higher stakes → stay big model initially)
- Keep `edit_judge` on opencode big model. Distill to 0.5B **only** once volume
  justifies it and accept/reject agreement is measured.

### Training data
| Source | Extractor SFT pairs | Judge SFT pairs |
|---|---|---|
| **Session** (free, in-distribution) | accepted edits = `(context, edit)` gold | logged `(edit + neighbors → accept/reject/merge_into)` verdicts |
| **External** (abundant, broadens coverage) | `(doc span, big-model extraction kept by judge)` | big-model verdicts on external candidates |

Session verdicts are **already logged** by V4's judge path → ready-made labels.
First job is a miner that pulls them into SFT format (see §6 Phase B).

---

## 5. Risks & mitigations

1. **Feedback loop** → staging graph + eval gate + provenance tags + per-batch
   rollback. Main graph only changes on PASS.
2. **Schema drift** (extractor invents relations/types) → validate against §2
   vocab; repair, don't silently coerce; new relations append-only.
3. **Duplicate bloat** → entity-resolution step (semantic_dedupe) before judge;
   `merge_into` instead of add when near-dup.
4. **Class-balance regression** (the actual goal) → eval gate checks
   positive:negative on a held-out question probe; a batch that grows the graph
   but worsens balance is rejected. Backstop: the override-detection negative
   (READ_THIS 2026-06-02b) protects V5 even if junk slips through.
5. **Health degradation** → `apply_edits_offline` health delta; reject batch if
   `< degradation_threshold (-0.02)`.

---

## 6. Suggested build order (each = one PR, gated)

- **Phase A - audit/queue skeleton. IMPLEMENTED.** `v5.graph_grower.audit`
  reads session/corpus JSONL rows, computes base graph health, classifies scoped
  patches into three lanes, and writes a compact report plus queue JSONL files:
  `promotion_queue.jsonl`, `substrate_queue.jsonl`, and `review_queue.jsonl`.
  It does not mutate the graph.
  ```
  $env:PYTHONPATH="E:\PROJECT\graph_v5"
  python -m v5.graph_grower.audit ^
    --corpus data/distillation_corpus/sessions.jsonl ^
    --graph graphs/merged_graph.json ^
    --out artifacts/graph_growth/graph_growth_audit.json
  ```
  First audit run on 2026-06-02: 179 rows, 2707 scoped patches, base graph
  831 nodes / 1454 edges, 2 strict persistent candidates, 1419 substrate
  candidates, and 1837 review/blocked candidates. Persistent promotion is
  intentionally strict: old-schema rows, controller fallback rows, weak labels,
  low slot coverage, and suspicious design-synthesis slot frames are blocked
  from long-term promotion.
- **Phase B — Extractor data miner.** Pull session edits + judge verdicts → SFT
  pairs; external-doc ingestion + chunker; big-model bootstrap extractions.
- **Phase C — 0.5B extractor SFT.** Fine-tune + constrained decode + validate;
  swap into Phase A in place of big-model extraction; compare accept-rate.
- **Phase D — Eval gate + promote.** Held-out positive:negative probe, health
  delta, atomic promote with backup + rollback-by-batch_id.
- **Phase E (optional) — distill judge to 0.5B** once volume/agreement justify.

Success metric throughout: **held-out positive:negative ratio** for V5 trace-gen
improves (more questions gain real support) **without** health regression.

---

## 7. Open questions for owner

- External-doc corpus: what sources / domains first (match the question bank)?
- Promote cadence: per-N-sessions auto, or manual review of each staged batch?
- Keep grown nodes in a separate graph file (union at load) or merge into
  `graphs/merged_graph.json`? (Separate file = cleaner ablation.)
- Should Lane B substrate warnings block V5 substrate builds, or only persistent
  promotion? Current audit lets safe substrate patches through with row warnings,
  while blocking those rows from Lane A.
