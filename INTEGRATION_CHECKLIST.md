# Integration Checklist — "everything enabled" run (the full system, real verifier target)

> DO-CONFIRM checklist. Killer failure mode (proven repeatedly this project): a component is
> **decorative / bypassed** and the run looks fine but isn't testing what we claim. So every item
> below is verified by a **forcing function** — a log line, assert, or grep that PROVES the component
> actually fired in THIS run. If an item can't be confirmed, the component is decorative → **HALT**.
>
> Scope: the integrated derive/reason run on the REAL target (`solves_fn` = SWE verifier or real
> graph grader), with every piece we built wired into one `SlotGraph.solve`. ~60-90s to run preflight.

---

## PAUSE 1 — PREFLIGHT (before spending GPU/Docker; a `--preflight` assert that HALTS on any miss)

1. **Engine, not proxy.** The runner imports `SlotGraph` and calls `.solve()` per task — NOT a flat
   prompt pipeline. ✔ Forcing function: `grep "SlotGraph\|\.solve("` the runner; runtime log
   `solve: fixpoint=… steps=…` per task. (swe_slot bypassed this once — never again.)

2. **LoRA loaded.** The RL-trained `artifacts/derive_lora` is attached to the 4B (not the bare base).
   ✔ `print("LoRA loaded:", adapter_path, "| modules:", n)`; assert the adapter dir exists + non-empty.

3. **Operators armed + in-loop.** `OperatorInjector` hook registered; reason-slot fill calls
   `combine`+`inject`, op-signed by `op_kind_for`. ✔ Log `[op] inject ENTERED slot=JUDGE signs=[+1,-1]`
   on the first reason-slot fill. (Operators were inert in #8/#9/#13 — this MUST log ENTERED.)

4. **Retrieval real + slot-aware.** `RealEmbedder` (mpnet-768) + per-slot query built from the pool —
   NOT a constant `[{src}]`. ✔ Log each slot's `query=` and the retrieved node ids; assert two
   different slots produce different queries/evidence. (swe_slot's retr was a constant — flag it.)

5. **Verifier/grader reachable.** `solves_fn` = the real verifier; gold-sanity passes FIRST. ✔ Run
   gold-sanity (SWE Docker 5/5, or arithmetic gold check) and assert PASS before trusting any number.

## PAUSE 2 — PER-TASK RUNTIME (each task must show these components actually fired)

6. **Pool = text, no latent.** Slot values stored/read as TEXT in the `Pool` (no cached-vector
   collapse). ✔ Dump `{slot: value}` per task; values are readable strings.

7. **Retrieve-OR-DERIVE fires correctly.** Derive only on retrieval-miss (fallback, T14); on a real
   gap the LoRA derive runs. ✔ Log `DERIVE …` rows; grounded-check via `derive_reward`.

8. **Backtrack + INSUFFICIENT active.** A failed/insufficient slot triggers BACKTRACK + re-fill — not
   a silent skip. ✔ Log `BACKTRACK … revise …` rows when a downstream slot can't fill.

9. **Reward gated on grounding.** `derive_reward` scores every derive; ungrounded is PUNISHED even if
   it equals gold. ✔ Log `reward=… grounded=… verdict=…`; an ungrounded fill must show `<0`.

## PAUSE 3 — POST-RUN (right-for-the-right-reason; the no-cheap-tests rule)

10. **Manual inspection.** READ the actual outputs (queries, retrieved evidence, derived values,
    diagnoses, patches) — not just the aggregate `X/N`. Confirm passes are for the right reason.

11. **No decorative component.** Every item 1–9 logged ACTIVE at least once this run. If any NEVER
    fired (e.g. backtrack 0 times, operator never ENTERED), it's decorative → say so / fix wiring.

12. **Self-improving memory.** The solved slot-graph is saved + retrievable by signature for the next
    task. ✔ Assert the template store grew and a 2nd similar task retrieves it.

---

### Deviation policy
Skipping an item is an INFORMED choice, logged — e.g. "localization held fixed (oracle), retrieval
item 4 intentionally constant for the synthesis-isolation run." Never a silent omission.

### The forcing function to build
A `--preflight` mode in the integrated runner that asserts items 1–5 and **exits non-zero** if any
fails, and a `--wiring-report` printed at the end listing which of 1–12 fired (the "no decorative
component" audit). Can't claim "everything enabled" without this report showing all green.
