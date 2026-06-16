# DESIGN — Slot-Graph Reasoning (the self-improving reasoning substrate)

> Status: **DESIGN (v0)** — for theoretical-validity + novelty review BEFORE building.
> Converged 2026-06-16 from: the proven operator algebra + the code-reasoning win + the user's
> slot/task-pool insight. This is the architecture the whole project converges on.

## 0. One sentence
A task's reasoning is a **dependency graph of slots** solved to a **fixpoint** by a **frozen LM** through
**typed operators**, and the **solved slot-graph is saved back into the main graph** so the system grows
in *reasoning capability* (not parameters) and the reasoning is *visible + editable*.

---

## 1. Thesis it serves
- **Grow the graph, not the parameters** — now for *reasoning*: the graph accumulates solved reasoning
  STRUCTURES, not just facts.
- **Frozen small LM** (Qwen3.5-4B, ≤6GB) — the executor; never trained to memorize knowledge.
- **Visible + editable reasoning** — the slot-graph IS the reasoning state, on screen; edit a slot/edge
  → it re-reasons.

---

## 2. Core abstraction — the Task Slot-Graph (TSG)
A task instantiates a **TSG = (Slots, DataflowEdges)**.

**Slot** = a unit of the answer the task must produce:
```
Slot { id, role, value(register), state, justification }
  role          : what it holds (LOCALIZE / DIAGNOSE / FIX, task-typed)
  value         : the filled content (a SLOT register — text and/or latent)
  state         : one of the 5 filledness states (§3)
  justification : {upstream_slot_ids, evidence_node_ids} that support the value
```
**DataflowEdge** `A → B` = B's value **derives from** A's (B depends on A). This is the
`operator_schema` **DATAFLOW** relation (contains / leveraged / chain_step).

The TSG is itself a small graph — the same shape as the main graph, so it can be **saved into it** (§6).

**SWE instantiation (the first target):**
| slot | role | depends on | evidence (retrieved) | operator on evidence |
|---|---|---|---|---|
| LOCALIZE | target file+symbol | issue | ranked repo symbols | ASSERT |
| DIAGNOSE | root cause | LOCALIZE, source | failure_pattern / bug nodes + code | INVALIDATE pitfalls, ASSERT cause |
| FIX | the SR edit | DIAGNOSE | strategy / exemplar nodes | TRANSFORM / ASSERT |

---

## 3. Filledness states (not binary)
```
empty        : no value
tentative    : value, justification weak/incomplete
valid        : value, justification complete + consistent + GATE-sufficient
STALE        : was valid, but a dependency (upstream slot OR graph evidence) changed -> re-evaluate
INSUFFICIENT : value exists, but a DOWNSTREAM slot needs more depth from it
```
Transitions: fill(empty→tentative/valid); update-upstream(valid→STALE); downstream-demand(valid→INSUFFICIENT);
re-fill(STALE/INSUFFICIENT→tentative/valid). **The states are the control — no confidence head.**

---

## 4. The solve loop (fixpoint over the TSG)
```
instantiate TSG for the task (from a retrieved template §6, or the default per task_family)
repeat until FIXPOINT (all slots valid+sufficient) or max_steps:
   pick a non-(valid+sufficient) slot, upstream-first (topological order over DATAFLOW)
   FILL(slot):
       q   = slot.role + upstream slot values            # the slot's targeted sub-query
       ev  = retrieve(q, slot.evidence_pool)              # graph retrieval, per-slot pool
       v   = operator_inject(ev, q) -> frozen LM fills/derives the value   # §5
       slot.value, slot.justification = v, (upstream + ev ids)
       slot.state = valid if (justification complete AND GATE_sufficient(slot)) else tentative
   PROPAGATE(slot):
       forward : dependents of slot -> STALE      # DATAFLOW forward
       backward: if slot INSUFFICIENT or unfillable -> mark weakest justifier needs-deepen / INVALIDATE
   BACKTRACK (dependency-directed): if a slot is unfillable (no evidence / contradiction),
       trace justification edges backward, INVALIDATE the weakest upstream value, re-derive forward
emit answer from terminal slot(s) (FIX)
```
- **Trigger** = any empty/stale/insufficient slot. **Termination** = fixpoint (or max_steps guard).
- **retrieve-OR-derive** per gap: if `retrieve` returns nothing usable, DERIVE (the LM reasons the value
  from upstream slots — TRANSFORM operator).
- **Graph update during a task** (new info / an edit) re-enters via PROPAGATE: touched evidence → its
  slots STALE → re-fill. (belief revision.)

---

## 5. The operators are the primitives (the convergence — already proven)
| TSG mechanism | operator (this session, validated) |
|---|---|
| slot value storage | **SLOT** register (read/write/multi-step 8/8 on 4B) |
| inject supporting evidence to fill a slot | **ASSERT** (+v) |
| suppress a wrong path / mark stale | **INVALIDATE** (−v) |
| "is the basis sufficient?" precondition | **GATE** (×g) |
| derive a slot from upstream | **TRANSFORM** (W·) |
| slot depends-on slot | **DATAFLOW** edge |
Fill = grounded operator inject (query-grounded; latent cache collapses — must be grounded, §validity).
The frozen LM is the per-slot **executor**; the TSG + operators are the scaffold.

---

## 6. Memory — save/retrieve/generalize solved TSGs (the self-improving core)
**On solve success** (fixpoint reached, and — when available — verifier pass):
1. **Generalize** the solved TSG into a TEMPLATE: strip instance specifics (this file, this symbol),
   keep the reasoning STRUCTURE (slot roles + DATAFLOW + the strategy/pitfall content). Reuse the
   strategy distiller's instance→general lint (`swe_strategy`).
2. **Save** the template into the main graph (gated apply, `graph_grower/apply.py`): slots→nodes,
   DATAFLOW→edges, tagged with a **task signature** (issue embedding / bug-type).
**On a new task:**
3. **Retrieve** the nearest saved TSG template by task signature (the trained ranker) →
   **instantiate it as the starting TSG** (the reasoning template is given) → fill its slots for the
   new instance (fewer steps, the structure is pre-built).

**Result:** the graph accumulates solved REASONING STRUCTURES → reasoning capability grows with use.
Generalizes `grounded_coder`'s "capture solved patch as exemplar" → capture the whole reasoning slot-graph.

---

## 7. Reuse map (this is mostly assembly of built parts)
| need | existing component |
|---|---|
| slots / task_frame / required_slots | `reasoning/` V4 substrate (`schemas`, `micro_controller`, `reasoning_loop`) |
| per-slot evidence pools (PLANNING L8 / EVIDENCE L20 typing) | `v5/subgraph.py` pools + `operator_schema` op-kinds |
| the solve-loop control skeleton | `reasoning/reasoning_loop.py` + `micro_controller.py` |
| retrieval (per-slot evidence + TSG template) | trained ranker `models/ranker-code` + `code_retrieve` |
| operator inject (fill) | `v5/operator_injector.py` (this session) |
| generalize-on-save (instance→template) | `v5/graph_grower/swe_strategy.py` lint |
| save into graph (gated) | `v5/graph_grower/apply.py` |
| the exemplar-store precedent | `v5/runtime/grounded_coder.py` (generalize it) |
| verifier (save trigger + RL reward later) | `v5/graph_grower/swe_verify.py` |
| frozen LM executor | `v5/lm_loader.py` (Qwen3.5-4B @ 4-bit) |
**New code = the TSG datastructure + the fixpoint/propagate/backtrack loop + the generalize-and-save.**

---

## 8. The frozen LM's role + the emission boundary (honest)
- LOCALIZE / DIAGNOSE = **reasoning DECISIONS** → operators carry content-specific signal (proven:
  code-reasoning +3.52 vs −5.27, beats RAG). The frozen LM's strength.
- FIX = **exact code emission** → the operator channel does NOT deliver exact tokens (proven: emission
  collapses; "constrained-decode = RAG"). So FIX uses **content-in-context** (source + retrieved
  exemplar) + the operator-injected *strategy* for the HOW. Exact code = retrieval, reasoning = operators.

---

## 9. Training (optional, later — CONTROL only, never knowledge)
- **SFT** the control (which slot next, retrieve-vs-derive) from solved-TSG traces. Precedent:
  `stage_sr_sft` (moved resolve 4→6).
- **RL** refine with the verifier reward. Knowledge stays in the graph (thesis intact).
- Substrate first — can't RL a policy with no action space.

---

## 10. To check BEFORE building (the user's gate)
**Theoretical validity:**
- V1 **Termination** — does fixpoint converge? (DATAFLOW must be a DAG or have cycle-breaking; max_steps guard.)
- V2 **Fill reliability** — can the frozen 4B fill each slot type well enough? (DIAGNOSE/LOCALIZE = reasoning ✓ proven; FIX = emission, needs context not operators ✓ §8.)
- V3 **2nd-task-easier** — does retrieving a saved TSG measurably reduce steps / raise accuracy? (the core empirical claim; prior strategy-retrieval was mixed — the TSG is richer, must be tested.)
- V4 **Generalize-on-save** — can instance→template abstraction be done reliably without leaking specifics? (strategy distiller does it imperfectly.)
- V5 **Invalidation correctness** — does STALE propagation re-derive consistently (no oscillation)?

**Novelty (cite, don't claim invention):**
- Truth-maintenance / dependency-directed backtracking (Doyle, de Kleer) — classic. We reuse, not invent.
- Frame/slot-filling reasoning — classic NLP.
- Skill libraries / case-based reasoning / Voyager — save+reuse solved structures.
- **The fresh assembly to defend:** operator-grounded slot-graphs — the SLOT/DATAFLOW/INVALIDATE/GATE
  operators as the TMS primitives, *executed through a frozen LM*, *saved back into the same graph that
  grounds it*, growing reasoning-not-parameters. The combination + the frozen-LM-as-executor is the claim.

---

## 11. Build order (if validity+novelty pass)
1. **TSG datastructure** + the 5 states + DATAFLOW.
2. **fixpoint solve loop** (fill via operators, propagate, backtrack) — toy 2–3 slot task first.
3. **save/retrieve TSG** — prove **2nd-task-easier** on a controllable pair (the go/no-go).
4. **port to SWE** (LOCALIZE→DIAGNOSE→FIX) + the verifier.
5. **scale** (more tasks → graph of TSG templates) + (optional) SFT/RL the control.
