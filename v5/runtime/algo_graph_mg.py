"""The coherent algorithm-graph loop ON graph_core.MemoryGraph — every piece supports the next, as
designed. This replaces the flat TotalMemory store (0 edges, health-less, pollution) with the mature
substrate the repo already has.

  RETRIEVE  MGRetriever: embed impl-node text + cosine (retrieval_eval's mechanism) + concept
            neighbourhood -> candidate atoms (code rides in node.metadata)
      |     the model is SHOWN the atoms' signatures
  AUTHOR    the LM composes a solution (CALLS atoms) and CHOOSES what to store (STORE actions, Fix D)
      |
  VERIFY    execution (verify_fn / MBPP asserts) — grounds everything
      |
  REWARD    derive_reward.code_reward + grounded_code (compose > novel > bare, fail < 0)
      |
  EDIT      model STORE -> add_node(implementation, code in metadata); COMPOSE -> add_edge
      |     (derived_from / part_of) — the MODEL edits its graph (honors 'model chooses')
  GROW      graph_grower.apply_candidates (health-gated, provenance, EDGES) via algo_graph_edits
      |
  (loop)    graph grew -> re-embed -> retrieval surfaces the new atom next round -> reuse compounds

Supervision: STaR / rejection-SFT (keep composed+verified -> SFT) — denser than GRPO, which only
amplifies. Reuses: graph_core.MemoryGraph, v5.runtime.algo_graph_edits (bridge to graph_grower),
derive_reward, tool_compose.verify_fn, algo_graph_run (tasks/prompt/verify/parse), memory embedders.

  selftest (no model):  python -m v5.runtime.algo_graph_mg --selftest
  run (GPU):            V5_LM_TRUST_REMOTE_CODE=1 python -m v5.runtime.algo_graph_mg --star --tasks mbpp
"""
from __future__ import annotations

import argparse
import builtins
import json
import keyword
import re
import sys
from pathlib import Path

import numpy as np

from graph_core import MemoryGraph
from v5.runtime.algo_graph_edits import grow, propose_edits
from v5.runtime.algo_graph_run import (_author_prompt, _code_fingerprint, _parse_store_actions,
                                       _task_verify, _top_defs, load_tasks)
from v5.runtime.derive_reward import _def_names, code_reward, grounded_code
from v5.runtime.tool_memory import _extract_code


def _fn_name(code: str) -> str:
    m = re.search(r"def\s+([A-Za-z_]\w*)\s*\(", code or "")
    return m.group(1) if m else ""


_UNSAFE_REUSE_NAMES = set(dir(builtins)) | set(keyword.kwlist) | {"self", "cls"}


def _is_safe_atom_name(name: str) -> bool:
    """True iff a function name is safe to treat as a graph-callable atom.

    Builtin-colliding atoms such as `sum` are valid task entry points, but they are
    unsafe as reusable graph atoms: ordinary Python code that calls builtin sum()
    would otherwise look like cross-task reuse and can create giant false hubs.
    """
    return bool(
        name
        and re.match(r"^[A-Za-z_]\w*$", name)
        and not name.startswith("_")
        and name not in _UNSAFE_REUSE_NAMES
    )


def _calls_atom(code: str, name: str) -> bool:
    """Detect a real unqualified call to a safe graph atom, not a method/builtin."""
    return bool(
        _is_safe_atom_name(name)
        and re.search(rf"(?<![\w.]){re.escape(name)}\s*\(", code or "")
    )


# ═══════════════════════════════════════════════════════════════════════════════
# RETRIEVAL on MemoryGraph — embed impl-node text + cosine (retrieval_eval mechanism)
# ═══════════════════════════════════════════════════════════════════════════════

class MGRetriever:
    def __init__(self, graph: MemoryGraph, embed_fn):
        self.graph = graph
        self.embed_fn = embed_fn
        self._index()

    def _index(self):
        impls = []
        for nid, n in self.graph.nodes.items():
            if n.node_type != "implementation" or not n.metadata.get("code"):
                continue
            if not _is_safe_atom_name(_fn_name(n.metadata.get("code", ""))):
                continue
            impls.append((nid, n))
        self.ids = [nid for nid, _ in impls]
        if self.ids:
            vecs = self.embed_fn({nid: n.text for nid, n in impls})
            self.mat = np.asarray([vecs[i] for i in self.ids], dtype=np.float32)
            self.mat /= (np.linalg.norm(self.mat, axis=1, keepdims=True) + 1e-9)
        else:
            self.mat = np.zeros((0, 1), dtype=np.float32)

    def retrieve_vec(self, vec, k: int = 6, min_cos: float = 0.2):
        """Return [(fn_name, code)] for the top-k impl nodes by cosine of node.text vs a query VECTOR
        (so a shared Query.vec — ranker output or embed — drives retrieval)."""
        if not self.ids:
            return []
        q = np.asarray(vec, dtype=np.float32)
        q /= (np.linalg.norm(q) + 1e-9)
        scores = self.mat @ q
        out = []
        for j in np.argsort(-scores)[:k]:
            if scores[j] < min_cos:
                continue
            code = self.graph.nodes[self.ids[j]].metadata["code"]
            out.append((_fn_name(code) or self.ids[j], code))
        return out

    def retrieve(self, query: str, k: int = 6, min_cos: float = 0.2):
        """Retrieve by a query STRING (embeds then delegates to retrieve_vec)."""
        if not self.ids:
            return []
        vec = list(self.embed_fn({"q": query}).values())[0]
        return self.retrieve_vec(vec, k=k, min_cos=min_cos)

    def _code_by_fn(self):
        d = {}
        for nid in self.ids:
            c = self.graph.nodes[nid].metadata.get("code", "")
            fn = _fn_name(c)
            if _is_safe_atom_name(fn):
                d[fn] = c
        return d

    def resolve_dep_names(self, called_names) -> set:
        """Reference-CLOSURE: the called atoms PLUS everything they transitively reference (edit_distance
        -> str_dp2). Used both to assemble deps AND to credit transitively-needed atoms in Δcapability."""
        code_by_fn = self._code_by_fn()
        needed, frontier = set(), [n for n in called_names if n in code_by_fn]
        while frontier:
            fn = frontier.pop()
            if fn in needed:
                continue
            needed.add(fn)
            body = code_by_fn[fn]
            for other in code_by_fn:                       # any other impl fn this one calls
                if other != fn and other not in needed and _calls_atom(body, other):
                    frontier.append(other)
        return needed

    def resolve_deps(self, called_names) -> str:
        """Graph WALK: the code of the transitive closure (so an abstracted leaf pulls its skeleton).
        Without this every ABSTRACT breaks at compose time."""
        code_by_fn = self._code_by_fn()
        return "\n\n".join(code_by_fn[fn] for fn in self.resolve_dep_names(called_names) if fn in code_by_fn)


# ═══════════════════════════════════════════════════════════════════════════════
# SOLVE + GROW — retrieve -> author -> verify -> reward -> model edits -> health-gated grow
# ═══════════════════════════════════════════════════════════════════════════════

def solve_mg(retriever: MGRetriever, gen_fn, task, k: int = 6, samples: int = 1):
    advertised = retriever.retrieve(task.text, k=k)
    adv_names = [n for n, _ in advertised]
    best = ("", [], False, "")                                   # code, reused, verified, raw
    for gen in gen_fn([_author_prompt(task, advertised)] * samples):
        code = _extract_code(gen)
        defined = _def_names(code)
        called = [n for n in adv_names if n not in defined and _calls_atom(code, n)]
        deps = "\n\n".join(c for n, c in advertised if n in called)
        if _task_verify(task, code, deps):
            best = (code, called, True, gen); break
        if not best[0]:
            best = (code, called, False, gen)
    code, reused, verified, raw = best
    _, used = grounded_code(code, adv_names)
    new_helpers = [n for n in _top_defs(code) if n != task.name and n not in adv_names]
    R, _ = code_reward(verified, composed_used=used,
                       authored_new_verified=len(new_helpers) if verified else 0)
    return dict(name=task.name, verified=verified, reward=round(R, 3), reused=used,
                code=code, raw=raw)


def _edits_from_solve(res, task, concept_id: str = "concept_algorithms"):
    """MODEL-directed edits, verify-gated: its STORE actions -> add_node(implementation); the atom
    part_of the concept; and it DEPENDs on each atom it composed (composition edges)."""
    defs = _top_defs(res["code"])
    stores, edges = [], []
    for name, purpose in _parse_store_actions(res["raw"]):
        src = defs.get(name)
        if not src or _code_fingerprint(src, name) is None:      # only standalone verifiable atoms
            continue
        nid = f"impl_{name}"
        stores.append((nid, src, purpose))
        edges.append((nid, concept_id, "part_of"))
        for atom in res["reused"]:                               # composed FROM these -> depend edges
            edges.append((nid, f"impl_{atom}", "depend"))
    return stores, edges


_SEED_CONCEPTS = ("concept_algorithms", "concept_number_theory", "concept_strings",
                  "concept_lists", "concept_math", "concept_search_sort")


def seed_graph(path: str, concepts: tuple = None):
    """A base MemoryGraph with concept nodes so atoms attach via part_of + cold-start isn't empty.
    Defaults to all 6 domain concept nodes so _classify_concept can route atoms to the right hub."""
    if concepts is None:
        concepts = _SEED_CONCEPTS
    nodes = [{"id": c, "text": c.replace("concept_", "").replace("_", " "), "node_type": "concept"}
             for c in concepts]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"metadata": {}, "nodes": nodes, "edges": []}), encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════════
# STaR / rejection-SFT on MemoryGraph — the loop, trained. Keep composed+verified -> SFT; the model's
# edits GROW the graph (health-gated) each round; re-index retrieval so reuse compounds.
# ═══════════════════════════════════════════════════════════════════════════════

def train_star_mg(model_name, rounds, tasks, graph_path, k=8, batch=16, lr=1e-4, r_lora=8,
                  layers=(20, 22, 24, 26, 28), seed=0, temperature=1.0):
    import os
    import random
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from v5.lm_loader import load_frozen_lm
    from v5.memory.store import make_mpnet_embedder

    if not Path(graph_path).exists():
        seed_graph(graph_path, _SEED_CONCEPTS)
    trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
    embed = make_mpnet_embedder()
    retr = MGRetriever(MemoryGraph.load_json(graph_path), embed)
    base = load_frozen_lm(model_name)
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
    dev = next(base.parameters()).device
    leaf = sorted({n.split(".")[-1] for n, m in base.named_modules()
                   if isinstance(m, nn.Linear) and ".layers." in n
                   and not any(x in n.lower() for x in ("lm_head", "embed"))})
    model = get_peft_model(base, LoraConfig(r=r_lora, lora_alpha=2 * r_lora, lora_dropout=0.0,
                           task_type="CAUSAL_LM", target_modules=leaf,
                           layers_to_transform=list(layers))); model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr)
    ce = nn.CrossEntropyLoss(); rng = random.Random(seed)
    print(f"STaR/MG: LoRA r={r_lora} | graph {graph_path} | {len(tasks)} tasks", flush=True)

    def encode(prompt):
        m = [{"role": "user", "content": prompt}]
        kw = dict(add_generation_prompt=True, return_tensors="pt", return_dict=True)
        try:
            return tok.apply_chat_template(m, enable_thinking=False, **kw)["input_ids"].to(dev)
        except TypeError:
            return tok.apply_chat_template(m, **kw)["input_ids"].to(dev)

    @torch.no_grad()
    def evaluate(held):
        model.eval(); solved = reused = 0
        for task in held:
            adv = retr.retrieve(task.text, k=k); names = [n for n, _ in adv]
            pids = encode(_author_prompt(task, adv))
            out = model.generate(pids, do_sample=False, max_new_tokens=460, pad_token_id=tok.eos_token_id)
            code = _extract_code(tok.decode(out[0, pids.shape[1]:], skip_special_tokens=True))
            called = [n for n in names if _calls_atom(code, n)]
            deps = "\n\n".join(c for n, c in adv if n in called)
            ok = _task_verify(task, code, deps)
            solved += ok; reused += ok and bool(grounded_code(code, names)[1])
        model.train(); return solved / max(1, len(held)), reused / max(1, len(held))

    held = list(tasks)[:24]
    bs, brz = evaluate(held); print(f"[round 0] solve={bs:.0%} reuse={brz:.0%}", flush=True)
    for rnd in range(1, rounds + 1):
        model.eval(); kept, ncomp, stores, edges = [], 0, [], []
        for _ in range(batch):
            task = rng.choice(tasks)
            adv = retr.retrieve(task.text, k=k); names = [n for n, _ in adv]
            prompt = _author_prompt(task, adv); pids = encode(prompt)
            with torch.no_grad():
                outs = model.generate(pids, do_sample=True, temperature=temperature, top_p=0.95,
                                      max_new_tokens=460, num_return_sequences=k, pad_token_id=tok.eos_token_id)
            pick = None
            for j in range(k):
                gen = tok.decode(outs[j, pids.shape[1]:], skip_special_tokens=True)
                code = _extract_code(gen)
                called = [n for n in names if n not in _def_names(code) and _calls_atom(code, n)]
                deps = "\n\n".join(c for n, c in adv if n in called)
                if not _task_verify(task, code, deps):
                    continue
                composed = bool(grounded_code(code, names)[1])
                res = dict(name=task.name, code=code, raw=gen, reused=grounded_code(code, names)[1])
                s, e = _edits_from_solve(res, task)               # model-directed edits from this win
                stores += s; edges += e
                if pick is None or (composed and not pick[1]):
                    pick = (gen, composed)
            if pick:
                kept.append((prompt, pick[0])); ncomp += int(pick[1])
        # SFT on kept
        model.train(); losses = []
        rng.shuffle(kept)
        for prompt, comp in kept:
            pids = encode(prompt)
            cids = tok(comp + tok.eos_token, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
            logits = model(torch.cat([pids, cids], dim=1)).logits
            s = pids.shape[1]
            loss = ce(logits[:, s - 1:s - 1 + cids.shape[1]].reshape(-1, logits.shape[-1]).float(), cids.reshape(-1))
            loss.backward(); torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step(); opt.zero_grad(); losses.append(float(loss.detach()))
        # GROW the graph from this round's model edits (health-gated), swap in, re-index retrieval
        if stores or edges:
            newp = graph_path + ".grown"
            gr = grow(graph_path, newp, propose_edits(f"r{rnd}", stores, edges))
            if gr.get("persisted"):
                Path(newp).replace(graph_path)
                retr = MGRetriever(MemoryGraph.load_json(graph_path), embed)
        s, rz = evaluate(held)
        ml = sum(losses) / max(1, len(losses))
        print(f"[round {rnd}] kept {len(kept)}/{batch} ({ncomp} composed) sft_loss={ml:.3f} | "
              f"solve={s:.0%} reuse={rz:.0%} | graph {len(retr.graph.nodes)} nodes "
              f"{len(retr.graph.edges)} edges", flush=True)
    model.save_pretrained("artifacts/algo_graph_mg_lora")
    print("  LoRA saved -> artifacts/algo_graph_mg_lora", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SELFTEST (no model) — the FULL loop connects: seed -> solve(store atom) -> grow(node+edge) ->
# re-embed -> retrieval SURFACES the new atom for a related task (the self-improving loop)
# ═══════════════════════════════════════════════════════════════════════════════

def _selftest() -> bool:
    import tempfile
    from v5.memory.store import make_fake_embedder
    from v5.runtime.algo_graph_run import MBPPTask
    print("algo_graph_mg --selftest: retrieve(MemoryGraph)->solve->model-edit->grow->retrieve-again\n")
    embed = make_fake_embedder()
    with tempfile.TemporaryDirectory() as td:
        unsafe = str(Path(td) / "unsafe.json")
        Path(unsafe).write_text(json.dumps({"metadata": {}, "edges": [], "nodes": [
            {"id": "concept_algorithms", "text": "algorithms", "node_type": "concept"},
            {"id": "impl_sum", "text": "unsafe builtin collision", "node_type": "implementation",
             "metadata": {"code": "def sum(xs):\n    return 0"}},
            {"id": "impl_digit_total", "text": "sum decimal digits", "node_type": "implementation",
             "metadata": {"code": "def digit_total(n):\n    return 0"}},
        ]}), encoding="utf-8")
        r_unsafe = MGRetriever(MemoryGraph.load_json(unsafe), embed)
        assert "impl_sum" not in r_unsafe.ids and "impl_digit_total" in r_unsafe.ids
        assert not _calls_atom("def f(xs):\n    return sum(xs)", "sum")
        print("  [0] unsafe atom names: builtin-colliding impl_sum is not retrievable/callable -> PASS")

        base = str(Path(td) / "g0.json")
        seed_graph(base, ("concept_number_theory",))
        g = MemoryGraph.load_json(base)
        r0 = MGRetriever(g, embed)
        assert r0.retrieve("is the number prime") == [], "empty graph -> nothing retrieved"
        print("  [1] seed graph (concept only) -> retrieval empty -> PASS")

        # solve a task; a storing stub composes nothing (cold) but writes is_prime + a STORE action
        IS_PRIME = ("def is_prime(n):\n    if n < 2:\n        return False\n    i = 2\n"
                    "    while i * i <= n:\n        if n % i == 0:\n            return False\n"
                    "        i += 1\n    return True")
        task = MBPPTask("check_prime", "Write `check_prime(n)`: True iff n is prime. "
                        "ctx: prime number test",
                        ["assert check_prime(7) == True", "assert check_prime(8) == False"])

        def _stub(prompts):
            body = "def check_prime(n):\n    return is_prime(n)"
            return [f"```python\n{IS_PRIME}\n\n{body}\n```\n"
                    "STORE is_prime: primality test — True iff n is prime"] * len(prompts)

        res = solve_mg(r0, _stub, task)
        assert res["verified"], f"solved: {res}"
        stores, edges = _edits_from_solve(res, task, "concept_number_theory")
        assert stores and stores[0][0] == "impl_is_prime", f"model chose to store is_prime: {stores}"
        print(f"  [2] solve -> verify -> model STORE is_prime (+part_of edge) -> PASS")

        # GROW via graph_grower (health-gated) -> node + edge land on MemoryGraph
        out = str(Path(td) / "g1.json")
        gr = grow(base, out, propose_edits("sess", stores, edges))
        assert gr["edit_stats"]["node_edits"] == 1 and gr["edit_stats"]["edge_edits"] == 1, gr
        assert gr["gate_passed"] and gr["persisted"], gr
        print(f"  [3] grow: +1 node / +1 EDGE, health {gr['health_before']}->{gr['health_after']} "
              f"gate PASS -> PASS")

        # RE-EMBED the grown graph -> retrieval now SURFACES is_prime for a related query (the loop!)
        g1 = MemoryGraph.load_json(out)
        assert "impl_is_prime" in g1.nodes and g1.edge_between("impl_is_prime", "concept_number_theory")
        r1 = MGRetriever(g1, embed)
        got = r1.retrieve("primality test — True iff n is prime", k=3, min_cos=-1.0)
        assert got and got[0][0] == "is_prime", f"grown atom is retrievable: {got}"
        print("  [4] re-embed grown graph -> retrieval SURFACES the new atom (loop compounds) -> PASS")

    print("\n  ALGO_GRAPH_MG SELFTEST -> PASS")
    return True


def main():
    ap = argparse.ArgumentParser(description="Algorithm-graph loop on graph_core.MemoryGraph (coherent).")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--star", action="store_true", help="STaR/rejection-SFT on MemoryGraph (health-gated grow)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--tasks", default="mbpp", choices=["curriculum", "mbpp"])
    ap.add_argument("--task-limit", type=int, default=200)
    ap.add_argument("--graph", default="graphs/algo_graph_mg.json", help="MemoryGraph path (seeded if missing)")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r-lora", type=int, default=8)
    ap.add_argument("--layers", type=int, nargs="+", default=[20, 22, 24, 26, 28])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=1.0)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    if a.star:
        tasks = load_tasks(a.tasks, a.task_limit)
        print(f"tasks: {a.tasks} ({len(tasks)})", file=sys.stderr)
        train_star_mg(a.model, a.rounds, tasks, a.graph, k=a.k, batch=a.batch, lr=a.lr,
                      r_lora=a.r_lora, layers=a.layers, seed=a.seed, temperature=a.temperature)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
