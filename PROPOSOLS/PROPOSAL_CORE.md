# A Frozen On-Device Reasoner with a Verified Graph — what survives when you refuse to train the model

> Core proposal draft (English). Every headline number is measured; honest limits and open problems are
> stated, not hidden. Section files: `composition_ceiling_section.md`, `wm_trm_section.md`.

## 1. The problem

Useful AI for schools, labs, small teams, and privacy-bound organisations must be **cheap, private, and
verifiable** — not a cloud chatbot. But the dominant recipe puts all capability **in the model's weights**,
which forces a choice between (a) paying to retrain/scale the model whenever knowledge changes (rising
energy + cost), (b) fine-tuning on-device (catastrophic forgetting), or (c) sending data to the cloud
(privacy/PDPA risk). We target the opposite: a **frozen, small (~3B, 4-bit), on-device** model whose
capability lives in an **inspectable, ownable, verified graph** — not in opaque weights.

## 2. The constraint that defines the contribution

Our hard constraint — **freeze the model, learn only in an external graph + tiny nets, ≤6 GB, no cloud** —
is exactly what most published techniques *don't* assume. DeepSeek and others "do it right" by **training
the weights end-to-end** (RL for reasoning, extra heads for drafting, latent attention baked into the
architecture). We refuse that (cost, forgetting, ownership). So the real question — and our contribution —
is: **which ways of assisting a model survive when the model is frozen?** We mapped it empirically.

**The law we measured (three independent confirmations):** a frozen LM is a **text-in / text-out** function;
the only lossless channel into it is **text**. Therefore an external system can only help by (a) choosing
*what verified text to feed it* and (b) letting it do what it is good at.

| channel we tried | what the small model produced | result (real 3B) |
|---|---|---|
| **latent** hand-off | a vector for the frozen LM to read | **fails** — routing collapse (73→15 when scaled) |
| **generated draft** | code tokens (spec-decode) | **fails** — 0/3, garbage (a tiny net can't write code) |
| **recall memory** | key→value state | **no target** — clean recall isn't even an LM failure |
| **structure (route)** | *which* atoms (a pointer) | **works** — cosine-RAG 0.50 → **0.98** |
| **structure (plan)** | *how* atoms wire (a program) | **works** — **0.03 → 1.00** |

Every *non-structural* channel fails; both *structural* channels win. That is the architecture.

## 3. The headline result — the frozen LM's real ceiling is COMPOSITION, and structure removes it

Given verified helper atoms, a frozen Qwen2.5-3B must produce a function equal to a nested expression
(*"one more than the negation of the sum of the product of n and n, and n"* → `inc(neg(add(mul(n,n),n)))`).
As **depth** grows, it mis-wires and collapses; given the **structure**, a deterministic realizer is perfect:

| depth | free-form frozen 3B | planned (structure given) |
|---|---|---|
| 1 | 0.73 | **1.00** |
| 3 | 0.30 | **1.00** |
| 5 | **0.03** | **1.00** |

`python -m v5.runtime.algo_grr_wiring --run --lm Qwen/Qwen2.5-3B-Instruct` — exact verifier, deterministic
planned arm. **The reasoner's job is to emit structure** (an *atom-program*: which atoms + how wired); the
verified graph delivers the code as text; the frozen LM composes the small glue and runs it.

## 4. The architecture

```
 TASK ─▶ ROUTER (which atoms; learned, structural — beats cosine 0.50→0.98)
          └▶ PLANNER (how wired; atom-program — turns 3%→100% on deep composition)
               └▶ VERIFIED GRAPH delivers the atoms' exact code as TEXT (the "membrane")
                    └▶ FROZEN LM ratifies / writes the small glue ─▶ VERIFY gate (only writer)
                         └▶ on solve: BANK new verified atoms (fuzz-general) ─▶ graph grows, cost falls
```
- **Frozen LM** — never trained → no forgetting, no cloud, no per-change retraining cost.
- **Verify gate is the only writer** → the graph is correct *by construction*; a weak model + search + verify
  still accretes a correct library (expert-iteration).
- **Compounding** — verified atoms are reused/abstracted; per-task LM cost falls with use (measured:
  derived-reuse 6 → 70 on a real 3B stream). RAG is static; this compounds.
- **No weight-poison** — measured: frozen+membrane **10/10** vs LoRA-on-own-traces **6/10**.

## 5. Honest limits and roadmap (the self-criticism the work needs)

- **The frozen LM is the ceiling for *invention*.** The graph reuses/wires known atoms; a task needing a
  novel primitive the 3B can't write is capped. Invention (LM authors new atoms) is the frontier.
- **The planner is not yet learned from language.** Our composition result *supplies* the structure; a
  from-scratch recursive model that *emits* atom-programs is validated on algorithmic tasks (a banked
  abstraction scored Δ+0.50 and became the top atom under compose-forcing) — generalising it to arbitrary
  NL tasks is the roadmap.
- **Routing does not yet scale cleanly.** A flat router degrades as the graph grows; the fix is a
  **structure-aware graph encoder** — plain message-passing GNNs are 1-WL-limited (blind to motifs), so we
  will add **structural/positional encodings (Laplacian PE, random-walk SE)** and **substructure/motif
  features (GSN-style)** so the router routes by *subcomponent*, not per-atom. This is also the natural
  representation for the abstraction hierarchy (recurring subgraphs = the higher-order skills that
  sleep-compression mints).
- **The moat is scoped to verifiable domains.** Safe self-growth needs a machine oracle (code, math,
  simulation). Open-ended domains fall back to retrieval. We state this scope rather than overclaim.
- **Benchmarks.** We are moving evaluation from the too-easy MBPP (≈2% factorable) to **MHPP** and
  BigCodeBench, where multi-step reasoning creates real headroom (and where the frozen-LM ceiling is
  honestly exposed).

## 6. Why this is not "graph-RAG" and not "already on the market"

RAG retrieves static text by similarity into the prompt — cosine-blind to structure, no self-growth. Ours
**learns** which verified operations are relevant (router, beats cosine by structure), **plans** how they
compose (planner, removes a measured 3%-ceiling), **grows and verifies its own library** (compounds,
cheaper with use), all on a **frozen on-device model that is never retrained**. The failures we report —
latent, draft, recall — are not gaps; they are the **map of what the frozen-weight constraint forbids**, and
they justify precisely this design.
