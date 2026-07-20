# ISSUES — the problem GRR faces right now (2026-07-20)

Written for thinking, not for a decision. Every number is measured. The machinery (retrieval, planning,
dual-channel realization, compounding, anti-poison gate) works; the problems below are what the scaling
stress-test surfaced.

---

## PROBLEM 1 (headline) — the AUTHORING-ERROR TAX = the invention ceiling, now quantified

**Statement.** The whole system's capability to *grow* is bounded by one thing: the frozen ~3B must
**author each new atom correctly** at least once, or that atom never banks. The verify gate correctly
rejects wrong atoms (the graph stays clean — anti-poison holds), but a rejected atom is *re-authored on
every reappearance* (wasted LM calls) or *skipped* (its tasks fail). The 3B cannot author roughly **half**
of the atoms in our synthetic domain.

**Evidence (real Qwen2.5-3B, `algo_grr_scale --run`, K=100 T=400 zipf=1.1):**

| author strategy | banked /100 | late floor (author/task) | author_calls | wasted |
|---|---|---|---|---|
| sim (perfect author) | 75 | **0.12** | ~75 | ~0 |
| real 3B, tries=1, no cap | 45 | **0.25** | ~? | — |
| real 3B, tries=1, **cap=3** | 39 ↓ | 0.25 | 125 | 86 |
| real 3B, **best-of-4**, cap=3 | 42 | **0.75** ↑↑ | **363** | **321** |

**Why both attempted fixes failed:**
- **Blind mistake-node cap** (skip an atom after N failed authorings): *hurt* — banked 45→39, floor
  unchanged. It gives up on atoms that were only *unluckily* failed, and skipping them fails their tasks.
- **Best-of-N** (resample the author N times): *much worse* — floor 0.25→**0.75**, calls 125→363. **The
  3B's authoring failures are SYSTEMATIC, not independent noise.** It misreads a given atom the *same* way
  every sample, so N samples give N copies of the same wrong code. Best-of-N assumes independence that does
  not hold → it burns N× compute for near-zero rescue. **Resampling cannot fix understanding.**

**So the core tension:** the frozen LM is a fixed-capability author. For the ~half of atoms it *can* write,
the graph compounds beautifully (reuse, dual-channel, retrieval all proven). For the ~half it *cannot*
write, no retry strategy over the *same frozen model on the same prompt* helps — the failure is in the
model's understanding, not in sampling luck.

**Candidate directions (unranked — for thinking):**
1. **Teacher escalation.** Atoms the 3B systematically fails → hand to the molab 32B teacher (offline),
   which authors + verifies, and either (a) bank the teacher's atom into the graph, or (b) STaR-distill
   that specific skill into the student. Poison-safe (verify-gated). This is the *designed* teacher role;
   the tax makes it necessary, not optional.
2. **Change the input, not the sample.** Systematic failures may be prompt-fragile: rephrase the spec,
   add a worked example, or DECOMPOSE the atom into sub-atoms the 3B can write. (Different input → different
   understanding, unlike best-of-N.) Cheap to try; unknown yield.
3. **Immediate skip (cap=1) + record the mistake, escalate.** Stop wasting calls the moment an atom fails
   once; route it to (1) or (2). Trades solve-rate on the hard atom for zero waste.
4. **Accept the ceiling honestly.** The frozen-3B floor (0.25 amortized authoring cost, ~45% atom coverage)
   IS the on-device capability; the graph's job is to make the *other* ~half free forever. Report it as the
   measured ceiling and let the teacher/STaR lift it offline.

**The meta-question this forces:** is "freeze the LM forever" the right hill for *authoring*? The graph
machinery doesn't need the LM unfrozen. But *invention* (writing a genuinely new primitive) is exactly
where a frozen weak model caps out. Options: keep frozen on-device + teacher-authored atoms pushed as
graph updates (adaptation lives in the graph, not weights — consistent with everything); OR allow a
verify-gated STaR trickle to teach the student the failing atoms (unfreeze the TRM/critic, never the LM's
core — the poison line). Both keep the anti-poison spine; they differ in where invention capability lives.

---

## PROBLEM 2 — routing on low-semantic descriptions

`route_ok` stayed **0.02–0.04** across the compounding runs because the synthetic atoms have *numeric*
descriptions ("the value of 3 n² + 2 n + 1 modulo 7") — semantic needles a content embedder can't
disambiguate. It didn't block compounding only because `OraclePlanner` supplied the atoms. **In a real
deployment the router must actually find the atom.** Mitigant already proven: content-ANN + follow-edge
(`--scale-fix`, recall 1.00 to N=10k) works when atoms have *meaningful* descriptions and *structural
edges*; it does NOT rescue a domain where atoms are semantically indistinguishable and unlinked. Open: do
real domains (code helpers) have enough semantic/structural signal? (MBPP evidence says yes — is_prime etc.
are nameable; the poly domain is a worst case.)

---

## PROBLEM 3 — the compounding floor is a law, not a bug (but state it)

Amortized LM cost → a floor = **the rate a never-seen atom appears**, set by the reuse distribution, not the
task count (measured: skewed workload 11% authoring, flat 25%). This is honest and fine — but it means the
"cost → 0" story is wrong; the real claim is "cost → the domain's novelty rate." A novelty-heavy workload
(lots of one-off tasks) barely compounds. Deployment targeting must pick *repetitive, verifiable* domains.

---

## Known-negative / parked (don't re-litigate without new evidence)

- **Latent semantic channel** — parked behind `fair_ab`; the z-wall (softprompt 73→15 routing collapse)
  says text wins until a white-box runner proves otherwise.
- **Cross-domain critic** — measured negative (code→math AUC 0.46 = chance); MiniLM too weak on code.
  Retired in favor of teacher-as-verifier; the mistake-tier itself survives (used in dcpd + the tax fix).
- **Composition-depth wall** — UNMEASURED. Planner holds to depth 5 when *given* structure; how deep it can
  *infer* structure before collapse is the last unmapped edge.
- **Real MBPP cross-task reuse ≈ 0** — corpus-bound (370/378 tasks are monolithic, non-decomposable);
  strong self-derived compounding needs a decomposable corpus (APPS/curriculum).

---

## What is SOLID (so the problems sit in context)

- **Dual-channel realization** — real-3B 39/40 verifies, 0 syntax errors, faithful explanation 1.00.
- **Compounding vs RAG** — pure-neural real-3B held-out 31/40 vs 15; LM calls fall with the graph.
- **Composition** — structure removes the 0.03→1.00 wiring collapse.
- **Retrieval at scale** — content-ANN + follow-edge, recall 1.00 to N=10k, no training, ms-fast.
- **Anti-poison** — verify gate is the only writer; frozen LM = no weight poison; graph stays clean even
  under a weak author (the tax is *wasted calls*, never *wrong banks*).

---

## The one-line framing

**The graph is a great memory and reasoner on top of a frozen author whose ceiling is now measured: it can
reuse/compose the ~half of atoms it can write, forever and cheaply, but it cannot invent the other half —
and no same-model retry fixes that. The next real lever is where invention capability comes from (teacher /
STaR / decomposition), not more of the frozen model.**
