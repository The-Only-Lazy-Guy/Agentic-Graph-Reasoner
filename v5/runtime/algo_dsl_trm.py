"""GRR-6 phase B: the TRM emits a DSL PROGRAM autoregressively — the piece where composition is real
reasoning, not a 4-way lookup.

At each decode step the reasoner, conditioned on the goal + partial program + the graph's atoms, picks:
an OP (FILTER/MAP/MAP2/KEEP_MAKEABLE/REDUCE), and its ARG (an ATOM pointer for the ops, an AGG for
REDUCE), halting at REDUCE. The realizer (algo_dsl) compiles the program to code that CALLS its atoms
(compose-forced). Trained by imitating the reference programs (algo_dsl._PROGRAMS), teacher-forced
cross-entropy per step. This is a tiny autoregressive program synthesizer over the graph's action space.

  selftest (no LM):  python -m v5.runtime.algo_dsl_trm --selftest
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from v5.runtime.algo_dsl import _PROGRAMS, Op, atoms_of, realize_program

OPS = ["FILTER", "MAP", "MAP2", "KEEP_MAKEABLE", "REDUCE"]
AGGS = ["sum", "max", "count", "len"]
_REDUCE = OPS.index("REDUCE")


def program_to_steps(pipeline, atom_idx):
    """[(op_i, atom_i|-1, agg_i|-1)] teacher-forcing targets for a reference program."""
    steps = []
    for op in pipeline:
        oi = OPS.index(op.kind)
        if op.kind == "REDUCE":
            steps.append((oi, -1, AGGS.index(op.arg)))
        else:
            steps.append((oi, atom_idx[op.arg], -1))
    return steps


def _build():
    import torch
    import torch.nn as nn

    class ProgramDecoder(nn.Module):
        """Autoregressive: (goal, partial-program state, atoms) -> next (op, arg). GRU state carries the
        program so far; op/agg heads + an atom POINTER head choose the step."""
        def __init__(self, d_in=768, d=64):
            super().__init__()
            self.goal_proj = nn.Linear(d_in, d)
            self.atom_proj = nn.Linear(d_in, d)
            self.op_emb = nn.Embedding(len(OPS), d)
            self.agg_emb = nn.Embedding(len(AGGS), d)
            self.rnn = nn.GRUCell(d, d)
            self.op_head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, len(OPS)))
            self.agg_head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, len(AGGS)))
            self.q_atom = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))
            self.d = d

        def _heads(self, g, state, A):
            h = torch.cat([g, state])
            return self.op_head(h), self.agg_head(h), A @ self.q_atom(h)      # op, agg, atom-pointer

        def loss(self, goal_vec, atom_vecs, steps):
            ce = nn.CrossEntropyLoss()
            g = self.goal_proj(goal_vec)
            A = self.atom_proj(atom_vecs)
            state = torch.zeros(self.d)
            total = torch.zeros(())
            for op_i, atom_i, agg_i in steps:
                op_l, agg_l, atom_l = self._heads(g, state, A)
                total = total + ce(op_l.unsqueeze(0), torch.tensor([op_i]))
                if op_i == _REDUCE:
                    total = total + ce(agg_l.unsqueeze(0), torch.tensor([agg_i]))
                    arg = self.agg_emb(torch.tensor(agg_i))
                    break
                total = total + ce(atom_l.unsqueeze(0), torch.tensor([atom_i]))
                arg = A[atom_i]
                state = self.rnn((self.op_emb(torch.tensor(op_i)) + arg).unsqueeze(0), state.unsqueeze(0))[0]
            return total

        @torch.no_grad()
        def decode(self, goal_vec, atom_vecs, atom_names, max_len=5):
            g = self.goal_proj(goal_vec)
            A = self.atom_proj(atom_vecs)
            state = torch.zeros(self.d)
            pipe = []
            for _ in range(max_len):
                op_l, agg_l, atom_l = self._heads(g, state, A)
                op = int(op_l.argmax())
                if OPS[op] == "REDUCE":
                    pipe.append(Op("REDUCE", AGGS[int(agg_l.argmax())])); break
                ai = int(atom_l.argmax())
                pipe.append(Op(OPS[op], atom_names[ai]))
                state = self.rnn((self.op_emb(torch.tensor(op)) + A[ai]).unsqueeze(0), state.unsqueeze(0))[0]
            return pipe

    return torch, nn, ProgramDecoder


def train_decoder(traces, atom_vecs, atom_names, d=64, steps=2000, lr=1e-3, seed=0):
    import random
    import torch
    torch, nn, ProgramDecoder = _build()
    torch.manual_seed(seed)
    model = ProgramDecoder(d_in=atom_vecs.shape[1], d=d)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    A = torch.as_tensor(atom_vecs, dtype=torch.float32)
    rng = random.Random(seed)
    for _ in range(steps):
        gv, tgt = traces[rng.randrange(len(traces))]
        loss = model.loss(torch.as_tensor(gv, dtype=torch.float32), A, tgt)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # stabilize (lr-too-high diverged)
        opt.step()
    return model


def solve_dsl(model, task, input_kind, atom_names, atom_vecs, embed_fn, graph_path):
    """TRM decodes a program -> realize -> resolve deps through the graph -> fuzz-verify."""
    import torch
    from graph_core import MemoryGraph
    from v5.runtime.algo_compose_tasks import ALL_ATOMS
    from v5.runtime.algo_graph_mg import MGRetriever
    from v5.runtime.algo_quality import fuzz
    gv = torch.as_tensor(np.asarray(list(embed_fn({"q": task.text}).values())[0], dtype=np.float32))
    pipe = model.decode(gv, torch.as_tensor(atom_vecs, dtype=torch.float32), atom_names)
    try:
        code = realize_program(task.name, input_kind, pipe)
    except (ValueError, KeyError):
        return False, pipe
    retr = MGRetriever(MemoryGraph.load_json(graph_path), embed_fn)
    used = atoms_of(pipe)
    deps = retr.resolve_deps(used) if any(a in ALL_ATOMS or a in atom_names for a in used) else ""
    passed, total = fuzz(code, task.name, deps, n=40)
    return bool(total and passed == total), pipe


def train_dsl_grr6(graph_path: str, hard: bool = True, steps: int = 3000, d: int = 64, seed: int = 0,
                   out: str = "artifacts/grr6_dsl.pt"):
    """molab: train the DSL program decoder with REAL mpnet embeddings on the graph's atoms. Imitates
    the reference programs; eval held-out = decode->realize->verify solve over the covered families."""
    from pathlib import Path
    import torch
    from graph_core import MemoryGraph
    from v5.memory.store import make_mpnet_embedder
    from v5.runtime.algo_compose_tasks import gen_compose_tasks, seed_atom_graph
    from v5.runtime.algo_graph_mg import _fn_name
    if not Path(graph_path).exists():
        seed_atom_graph(graph_path, hard=hard)
    embed = make_mpnet_embedder()
    g = MemoryGraph.load_json(graph_path)
    impls = [(nid, n) for nid, n in g.nodes.items()
             if n.node_type == "implementation" and n.metadata.get("code")]
    atom_names = [_fn_name(n.metadata["code"]) or nid[len("impl_"):] for nid, n in impls]
    atom_idx = {a: i for i, a in enumerate(atom_names)}
    vecs = embed({nid: n.text for nid, n in impls})
    atom_vecs = np.asarray([vecs[nid] for nid, _ in impls], dtype=np.float32)
    fams = {f: (k, p) for f, (k, p) in _PROGRAMS.items() if atoms_of(p) <= set(atom_names)}
    ts = list(gen_compose_tasks(300, seed=0)) + list(gen_compose_tasks(300, seed=0, hard=hard))
    traces = [(list(embed({"q": t.text}).values())[0], program_to_steps(fams[t.name][1], atom_idx))
              for t in ts if t.name in fams]
    print(f"grr6 DSL train: {graph_path} | {len(atom_names)} atoms | {len(fams)} families | "
          f"{len(traces)} traces | steps={steps}", flush=True)
    model = train_decoder(traces, atom_vecs, atom_names, d=d, steps=steps, seed=seed)
    held = list(gen_compose_tasks(60, seed=999)) + list(gen_compose_tasks(60, seed=999, hard=hard))
    held = [t for t in held if t.name in fams]
    solved = sum(int(solve_dsl(model, t, fams[t.name][0], atom_names, atom_vecs, embed, graph_path)[0])
                 for t in held)
    print(f"  held-out ({len(held)} tasks, {len(fams)} families): program decode->realize->verify "
          f"solve={solved/max(1,len(held)):.0%}", flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "atom_names": atom_names, "atom_vecs": atom_vecs, "d": d,
                "fams": {f: k for f, (k, _p) in fams.items()}}, out)
    print(f"  DSL reasoner -> {out}  ({sum(p.numel() for p in model.parameters())} params)", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SELFTEST — the reasoner DECODES a correct program per family (a real op+atom sequence) + solves.
# ═══════════════════════════════════════════════════════════════════════════════

def _selftest() -> bool:
    import json
    import tempfile
    from pathlib import Path
    from v5.runtime.algo_compose_tasks import ALL_ATOMS, gen_compose_tasks
    print("algo_dsl_trm --selftest: TRM autoregressively decodes a DSL PROGRAM per family + solves\n")

    atom_names = ["is_prime", "digit_sum", "build_adj", "dijkstra", "edit_distance", "lcs_length",
                  "coin_change", "lis_length"]
    atom_idx = {a: i for i, a in enumerate(atom_names)}
    rng = np.random.default_rng(0)
    d_in = 64
    atom_vecs = rng.standard_normal((len(atom_names), d_in)).astype("float32")

    # families we cover with the DSL (need their atoms present)
    fams = {f: (k, p) for f, (k, p) in _PROGRAMS.items()
            if atoms_of(p) <= set(atom_names)}
    kw = [("largest digit-sum", "max_prime_digitsum"), ("prime", "sum_digitsum_primes"),
          ("edit", "sum_edit_distance"), ("subsequence", "sum_lcs"),
          ("increasing", "max_lis"), ("coin", "count_makeable")]     # specific before general
    fam_base = {f: rng.standard_normal(d_in).astype("float32") for f in fams}

    def embed(d):
        out = {}
        for k, text in d.items():
            f = next((f for w, f in kw if w in text.lower()), None)
            out[k] = ((fam_base[f] if f in fam_base else np.zeros(d_in, "float32"))
                      + 0.3 * rng.standard_normal(d_in)).astype("float32")
        return out

    # traces: (goal_vec, target steps) from generated tasks + the reference programs
    def make_traces(seed, n):
        ts = list(gen_compose_tasks(n, seed=seed)) + list(gen_compose_tasks(n, seed=seed, hard=True))
        out = []
        for t in ts:
            if t.name not in fams:
                continue
            gv = list(embed({"q": t.text}).values())[0]
            out.append((gv, program_to_steps(fams[t.name][1], atom_idx)))
        return out

    with tempfile.TemporaryDirectory() as td:
        gp = str(Path(td) / "g.json")
        nodes = [{"id": "concept_algorithms", "text": "algorithms", "node_type": "concept"}]
        for a in atom_names:
            nodes.append({"id": f"impl_{a}", "text": ALL_ATOMS[a][0], "node_type": "implementation",
                          "metadata": {"code": ALL_ATOMS[a][1]}})
        Path(gp).write_text(json.dumps({"metadata": {}, "nodes": nodes, "edges": []}))

        model = train_decoder(make_traces(0, 120), atom_vecs, atom_names, steps=2500)

        # [1] decode a full program for a task + it CALLS atoms (multi-step reasoning, not a pick)
        import torch
        t_ed = next(t for t in gen_compose_tasks(30, seed=1, hard=True) if t.name == "sum_edit_distance")
        pipe = model.decode(torch.as_tensor(np.asarray(list(embed({"q": t_ed.text}).values())[0], dtype=np.float32)),
                            torch.as_tensor(atom_vecs), atom_names)
        assert [o.kind for o in pipe] == ["MAP2", "REDUCE"] and pipe[0].arg == "edit_distance", pipe
        print(f"  [1] decoded program for sum_edit_distance: "
              f"{[(o.kind, o.arg) for o in pipe]} (a real op+atom SEQUENCE) -> PASS")

        # [2] held-out: decode -> realize -> solve, across families
        held = list(gen_compose_tasks(60, seed=999)) + list(gen_compose_tasks(60, seed=999, hard=True))
        held = [t for t in held if t.name in fams]
        solved = sum(int(solve_dsl(model, t, fams[t.name][0], atom_names, atom_vecs, embed, gp)[0])
                     for t in held)
        acc = solved / len(held)
        assert acc >= 0.75, acc
        print(f"  [2] held-out ({len(held)} tasks, {len(fams)} families): decode->realize->verify "
              f"solve={acc:.0%} -> PASS")

        # [3] a program is genuinely structured: >=2 decode steps, right ops+atoms+agg (not a lookup)
        lens = [len(model.decode(torch.as_tensor(np.asarray(list(embed({'q': t.text}).values())[0], dtype=np.float32)),
                                 torch.as_tensor(atom_vecs), atom_names)) for t in held[:12]]
        assert max(lens) >= 3, lens            # sum_digitsum_primes = FILTER,MAP,REDUCE = 3 steps
        print(f"  [3] programs are structured: decode length up to {max(lens)} steps (FILTER->MAP->REDUCE), "
              f"each an op+atom choice -> PASS")

    print("\n  ALGO_DSL_TRM SELFTEST -> PASS  (the reasoner synthesizes programs; composition is real)")
    return True


def main():
    ap = argparse.ArgumentParser(description="GRR-6 phase B: TRM autoregressive DSL program decoder.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--train", action="store_true", help="train the DSL decoder with real mpnet (molab)")
    ap.add_argument("--graph", default="graphs/algo_reason_hard.json")
    ap.add_argument("--hard", action="store_true")
    ap.add_argument("--steps", type=int, default=3000)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    if a.train:
        train_dsl_grr6(a.graph, hard=a.hard, steps=a.steps)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
