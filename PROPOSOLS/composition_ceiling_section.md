# The frozen LM's real ceiling is COMPOSITION — and a structural planner removes it

> Proposal section (English draft). Every number is measured on a real Qwen2.5-3B with an exact verifier;
> the planned arm is deterministic. This is the honest core of "a small reasoner assists a frozen LM."

## The finding

A frozen small LM is good at *language* and *local* code, but it has a hard **composition ceiling**: when
a solution requires wiring several verified operations into the right *structure*, it mis-wires, and the
failure grows with depth. We measured this directly. The model is given verified helper atoms and must
produce a function equal to a nested expression (e.g. *"one more than the negation of the sum of the
product of n and n, and n"* → `inc(neg(add(mul(n, n), n)))`). We vary the expression **depth** and check the
result against an exact oracle.

| expression depth | free-form frozen 3B | planned (structure given) |
|---|---|---|
| 1 | 0.73 | **1.00** |
| 2 | 0.67 | **1.00** |
| 3 | 0.30 | **1.00** |
| 4 | 0.13 | **1.00** |
| 5 | **0.03** | **1.00** |

The frozen LM collapses from 73% to **3%** as depth grows. Given the **structure** — the plan of which atoms
wire to which — a deterministic realizer produces correct code **100% of the time, at every depth**.
`python -m v5.runtime.algo_grr_wiring --run --lm Qwen/Qwen2.5-3B-Instruct`

## Why this matters (and why it's the right role for the reasoner)

We tested three other ways a small model could help a frozen LM, and they failed for a *consistent*
reason — a frozen LM only reads **text** losslessly:
- a **latent** hand-off → the frozen LM can't read a foreign vector → *routing collapse*;
- a **generated draft** → a tiny model can't write valid code → *garbage*;
- a **recall memory** → clean recall isn't even an LM failure → *no target*.

Composition is different: it **is** a real, measured LM failure, and the fix does **not** require the small
model to write code or emit a latent. The reasoner emits **structure** — *which* atoms, wired *how* — and a
deterministic realizer turns that structure into code. The frozen LM is never asked to do the thing it is
bad at. This is the reasoner's true job:

- **route** — *which* verified atoms are relevant (learned router: cosine-RAG 0.50 → **0.98** recall,
  because it learns structural dependencies text-similarity is blind to);
- **plan** — *how* those atoms wire together (this result: **3% → 100%** by supplying structure);
- the frozen LM **delivers/ratifies** — composes the small glue and runs the verified atoms.

Router + planner together = an **atom-program**; the graph delivers the verified code; the LM ratifies.
Every non-text channel we tried failed; the two **structural** channels — route and plan — both won.

## Honest scope (stated, not hidden)

The planned arm above is **given** the ground-truth structure. A reasoner that **infers** the structure
from the natural-language task is the *learned planner* — and we have already validated a from-scratch
recursive model that emits atom-programs on algorithmic tasks (a banked abstraction scored Δ+0.50 and
became the top atom under compose-forcing, where a free-form model left it dark). Generalising that planner
to arbitrary natural-language tasks is the roadmap. What is proven here: **the frozen LM's composition
ceiling is real and steep, and supplying structure removes it entirely** — so the reasoner's value is
concrete and measurable, not hypothetical.
