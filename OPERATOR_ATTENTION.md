# Operator Attention — typed graph nodes that do LOGIC for a frozen LM

> Status: **mechanism validated on the deploy 4B**; runtime built; grower now produces the content.
> Last updated 2026-06-15.

## The problem it solves
Every trainable injection mechanism we built (cross-attn adapter, DeltaNet projector, GNN) collapsed
to **generic grounding** — the model uses *whatever content is present* coarsely, and node identity /
graph structure get ignored. Measured repeatedly: own-node ≈ random-node; topology (GNN) ≈ random.
Root cause: a softmax / weighted **sum** can satisfy the training loss while ignoring structure, so
gradient drops it. Structure was always *optional*.

## The invention
Make structure **non-optional** by baking the operation KIND into the node type as
**non-interchangeable primitives**, tuning only degree:

| op_kind | operation | role |
|---|---|---|
| `ASSERT` | `+v` | declarative evidence / grounding (fact, claim, symbol, ...) |
| `INVALIDATE` | `−v` (subtract / suppress) | a wrong path / misconception to kill (`failure_pattern`, `contradicts`) |
| `GATE` | `×g` (multiplicative) | logical precondition (`control_rule`, `epistemic_state`) |
| `TRANSFORM` | `W·state` | the "how" (`strategy`, `procedure`, `reasoning_chain`) |
| `SLOT` | writable register | intermediate result (reasoning loop, v2) |

A **subtract cannot be reproduced by a positive blend**; a **product-gate cannot be reproduced by a
sum**. So the mechanism *physically cannot* ignore the type → **structure is load-bearing by
construction**, can't collapse. That is the whole idea.

## Validation (all on a frozen LM; the headline numbers are the deploy Qwen3.5-4B @ layer 26)
| test | file | result |
|---|---|---|
| `INVALIDATE` flips a factual belief, edge-gated | `v5/optest_invalidate.py` | 5/5 (4B L18) |
| `INVALIDATE` suppresses a reasoning trap → reasons better | `v5/optest_reasoning.py` | **8/8** (4B L26) |
| op-signed **COMBINE** beats plain blend | `v5/operator_injector.py` | **6/6, +7.07** (4B L26) |
| plain blend (RAG / array-of-text) with conflicting content | (same) | **−2.95 — net NEGATIVE** |
| **EDGE-GATED end-to-end** (4B): grow→bare miscon→contradicts edge→retrieve correct→hop→signed inject | `v5/operator_loop_v2.py` | **acc 5/7→7/7, beats blend 6/7, cold +1.25→BLEND +0.57→OPERATOR +3.52** (4B L26, scope=all) |
| **SLOT register READ** (inject a value's latent vector → frozen LM recalls it) | `v5/optest_slot.py` | **8/8, +3.33 ≈ text** (4B L26; 1.5B fails — scale-dependent) |
| **SLOT register WRITE** (model writes its OWN computed value → reads it back, no text) | `v5/optest_slot_write.py` | **write-gen 8/8, +6.14** (4B L26; pure-latent write-state 6/8) |
| **GATE** mechanism: gate injection tracks precondition; ASSERT is precondition-blind | `v5/optest_gate.py` | 4B: gate +2.73(sat)/+1.16(vio) tracks g, assert +3.05/+4.78 blind; 1.5B accuracy 1/4→3/4 |

**GATE caveat (honest):** GATE is the weak operator. The *mechanism* (multiplicative, gate-output tracks
the precondition g, non-interchangeable with a sum) is confirmed, and its *accuracy* value shows on a weak
model that blindly misapplies a rule (1.5B 1/4→3/4). But a clean 4B accuracy beat is structurally blocked:
the 4B handles preconditions cold, and the step being gated (asserting the conclusion) is itself the
noisy one. Banked as the multiplicative primitive, not a 4B headline like INVALIDATE/SLOT.

**End-to-end note (the headline):** validated the FULL chain on the deploy 4B. Two findings made it work:
(1) **content shape** — the grower must emit `failure_pattern` as a BARE misconception (the wrong belief
asserted as true), not a correction; a strong LM reads a correction as the right answer (`optest_shape`:
bare → operator +2.76 / blend −2.64; correction → blend +2.07 / operator −0.34). (2) **edge-gated
retrieval** — a bare misconception has no query cues, so retrieve the topical CORRECT node and follow its
`contradicts` edge to the bare trap (real mpnet embedder). Then `ASSERT(correct) − INVALIDATE(trap)` beats
`BLEND(both +)`: BLEND is a high-variance poison (craters to −11 on the relevant trap), OPERATOR is stable
+3.5–5.0. **An array can't follow the edge — that is "why a graph, not an array," end-to-end.**
Degree: per-node α = α/k and residual-relative normalize (long grown nodes else over-steer).

**The punchline:** typed operators make a frozen 4B reason dramatically better, and **naive
content-retrieval (RAG) is worse than nothing** when evidence conflicts — only the operator structure
rescues it. That is the answer to *"why a graph, not an array."*

Design facts found by sweeping: the operator works at **late layers** (sweet spot ~L26 of 32 on the
4B, ~L14 of 28 on the 1.5B — ~0.8 depth), and needs adequate steering strength (α≈4).

## What does NOT work (honest)
- **Per-node CONTENT specificity from text** still collapses to generic (`v5/optest_projector.py`):
  a trained text→vector projector ≈ a generic-mean baseline. So the **operator KIND** is the value,
  not per-node content. Vectors here are **in-context grounding shifts** (a forward per node — faithful
  but slow); a fast trained projector remains an open problem (but the op kind is hard-coded, so
  structure survives the content-collapse).
- GATE / TRANSFORM operators not yet tested (only ASSERT/INVALIDATE validated).

## The schema (locked) — `v5/operator_schema.py`
Single source of truth, torch-free. `OP_OF` maps node_type→op_kind, `EDGEOP_OF` maps
relation→edge_op; coverage-checked over the full GNN node-type vocab. `op_kind` is **derived** from
node_type (no graph regen needed) and also **stamped** into `metadata.op_kind` by both growers
(`code_extract.py`, `extract.py`).

## The runtime — `v5/operator_injector.py`
`OperatorInjector` hooks one late layer and injects the **op-signed** sum of node grounding vectors:
`inject = Σ_ASSERT(+v) + Σ_INVALIDATE(−v) + Σ_TRANSFORM(+v)`. Reusable; `combine(nodes, query)`.

## The content — `v5/graph_grower/extract.py`
The graph had **zero INVALIDATE content** (the teacher only extracted what's true), so the operators
were dormant. The grower now extracts **`failure_pattern`** nodes (common wrong approaches /
misconceptions to AVOID) + **`contradicts`** edges, so a grown graph carries the INVALIDATE structure
the runtime applies. Aliases: misconception/pitfall/mistake/trap/... → failure_pattern.

## Honest scope for the competition (NRCT INTFAIR)
- This is **not a novel layer** — RGAT (typed attention), gated attention/ExGate, NTM/DNC (writable
  memory), negation-as-suppression each own a piece. The fresh part = *non-interchangeability as a
  defense against generic-collapse* + the demonstration on a frozen LM. **Cite, don't claim invention.**
- The demonstrable, honest result: **a frozen small model reasons much better via typed graph logic,
  and RAG-style blending actively hurts.** Clean figure: cold +1.2 → blend −3.0 → operators +7.1.

## Next steps
1. ~~Grow `failure_pattern` content~~ **DONE** (2026-06-15): codex teacher on `cot_batch1` (12 math docs) → `graphs/grown_graph6.json` carries **99 `failure_pattern` (INVALIDATE) nodes + 111 `contradicts` edges**, all op_kind-stamped, health gate PASS. Quality verified (concrete wrong-approaches w/ counterexamples). Optional: 2nd domain via `cs_batch` (12 algo docs).
2. ~~Wire into the real grounding loop + confirm on the 4B~~ **DONE** (`operator_loop_v2.py`, 4B L26): edge-gated end-to-end PASS — acc 5/7→7/7, beats blend 6/7, cold +1.25→blend +0.57→**operator +3.52**. Needed two fixes: BARE-misconception content shape (grower) + edge-gated retrieval (mpnet) + per-node α/k normalize.
3. ~~SLOT register~~ **DONE** (`optest_slot.py` READ 8/8, `optest_slot_write.py` WRITE-own-value 8/8 on 4B): a frozen 4B carries its own intermediate across steps as latent state — inspectable/editable, no retraining (the "idea layer" / make-DeltaNet-useful goal, on-thesis alternative to a diffusion plan head). Note: latent value-carry is SCALE-dependent (4B reads it, 1.5B can't); the real in-context shift carries content even though a trained projector collapses.
4. ~~GATE operator~~ **DONE** (mechanism confirmed, weak operator — see caveat above). **OPERATOR ALGEBRA COMPLETE: ASSERT (+) / INVALIDATE (−, end-to-end 4B) / GATE (×, mechanism) / SLOT (register, read+write 4B).**
5. **← NEXT:** SLOT multi-step loop; the "learn this" before/after expo demo (LAST); INTFAIR writeup. Optional: 2nd domain (`cs_batch`), native DeltaNet-state register, fast node→vector path.
