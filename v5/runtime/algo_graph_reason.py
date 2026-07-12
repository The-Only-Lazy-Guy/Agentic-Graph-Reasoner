"""Reasoning-over-graph, SYNERGIZED — "graph reasons, LM realizes", made strong (propose-verify, not
predict) with every piece feeding the next through ONE shared QUERY.

The bottleneck was: compose is the LM's RARE spontaneous decision -> can't be trained. Fix: make
composition a guided SEARCH + VERIFY (the 18%->100% tool_memory pattern / DreamCoder wake), and let
the LM only write GLUE (call named atoms) — a trivial job that sidesteps the 0.21 realizer wall.

The synergy — one Query threads the whole loop (the user's ask):

    make_query(task)  ->  Query{text, vec}          (what the task NEEDS)
         |  vec  ---------------------------------->  RETRIEVE ranked atoms (MGRetriever / ranker)
         |  text ---------------------------------->  a FEATURE in the realizer prompt (LM knows what
         |                                             it's composing) -> glue-realize (best-of-N)
         v
      VERIFY (execution) -> keep the composition that passes
         |
      GROW edges: task-concept `uses` atoms, atom `depend` atom  (graph_grower, health-gated)
         |                                             = the composition MEMORY that biases the NEXT
         `-------------------------------------------> search, and the trainable target for the query
                                                        ranker (verified task->atoms pairs).

Reuses: MGRetriever + MemoryGraph (v5.runtime.algo_graph_mg), graph_grower bridge (algo_graph_edits),
tool_compose.verify_fn, derive_reward, algo_graph_run (tasks/verify/extract). The query ranker
(traversal_ranker / memory_refiner) is the trainable upgrade of make_query — same interface.

  selftest (no model):  python -m v5.runtime.algo_graph_reason --selftest
  run (GPU):            V5_LM_TRUST_REMOTE_CODE=1 python -m v5.runtime.algo_graph_reason --run
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from graph_core import MemoryGraph
from v5.runtime.algo_graph_edits import grow, propose_edits
from v5.runtime.algo_graph_mg import MGRetriever, _fn_name, seed_graph
from v5.runtime.algo_graph_run import _sig, _task_verify, _task_verify_detail, _top_defs
from v5.runtime.derive_reward import _def_names
from v5.runtime.tool_memory import _extract_code


@dataclass
class Query:
    """What the task NEEDS — the shared thread: drives retrieval (vec), features the realizer (text),
    is the trainable target for the ranker (task -> the atoms that verified)."""
    text: str
    vec: np.ndarray


@dataclass
class Budget:
    """GRR-2: a real generation budget for ONE solve, so 'benefit under budget' has something to push
    against. Composing (short glue) is cheap -> more attempts fit; re-deriving inline (long) is
    expensive -> the budget runs out first. This is the amortization win made a first-class constraint."""
    total_tokens: int
    spent: int = 0

    def charge(self, n: int):
        self.spent += int(n)

    def exhausted(self) -> bool:
        return self.spent >= self.total_tokens


def _char_cost(gen: str) -> int:
    """Cheap token proxy (~4 chars/token) for no-GPU budget accounting; the real run passes a
    tokenizer-based cost_fn."""
    return max(1, len(gen) // 4)


def make_query(task, embed_fn, extra: str = "") -> Query:
    """v0: query = the task's own need statement. `extra` folds an obs-informed failure signal into the
    query so RE-retrieval (and the realizer feature) reflect what went wrong last round — the iterative
    loop's re-query. The trainable upgrade is traversal_ranker/memory_refiner, same (text, vec)."""
    text = task.text if not extra else f"{task.text}\n(retry — last attempt failed: {extra})"
    vec = np.asarray(list(embed_fn({"q": text}).values())[0], dtype=np.float32)
    return Query(text=text, vec=vec)


def _realize_prompt(task, query_text: str, candidates: list[tuple[str, str]], feedback: str = "") -> str:
    """Glue-realizer prompt: the QUERY is a FEATURE (the LM is TOLD what it needs), the atoms are the
    building blocks. `feedback` shows the previous attempt's failure so the retry can fix it."""
    parts = [task.text, f"\nWhat this needs: {query_text}"]
    if feedback:
        parts.append(f"Your PREVIOUS attempt failed with: {feedback}\nFix that in this attempt.")
    if candidates:
        parts.append("Building blocks already available (CALL them by name — do NOT re-implement):")
        for name, code in candidates:
            parts.append(f"  {_sig(code) or name}")
        parts.append("COMPOSE these to solve it (wire their outputs together).")
    parts.append(f"Write `{task.name}(...)` in ONE Python code block.")
    return "\n\n".join(parts)


def solve_iterative(task, retriever: MGRetriever, gen_fn, embed_fn, max_rounds: int = 3,
                    samples: int = 4, k: int = 6, min_cos: float = 0.25, extra_deps: str = "",
                    meta_retriever=None, budget: "Budget" = None, cost_fn=None):
    """Q1: the ITERATIVE loop — retrieve -> attempt (best-of-N) -> verify -> on failure re-query memory
    with the failure signal + retry, until solved or max_rounds. Unlike 1-shot search_compose, the
    failure OBSERVATION steers the next retrieval and prompt (obs-informed retry). `extra_deps` is code
    always in scope at verify time (multi-step: prior temp nodes + their transitive atoms).
    `meta_retriever` (#54) injects POLICY hints. `budget` (GRR-2) caps total generation — each attempt
    is charged cost_fn(gen); the solve stops when the budget runs out, so cheap (composed) solutions win
    under pressure. Returns solved, rounds, code, used, used_policies, tokens, history."""
    from v5.runtime.algo_meta import inject_policies
    cost_fn = cost_fn or _char_cost
    feedback, last, history = "", ("", [], False), []
    used_policies = []
    rnd = 0
    for rnd in range(max_rounds):
        if budget is not None and budget.exhausted():               # out of budget -> stop
            break
        query = make_query(task, embed_fn, extra=feedback)          # re-query reflects the failure
        ranked = retriever.retrieve_vec(query.vec, k=k, min_cos=min_cos)
        names = [n for n, _ in ranked]
        prompt = _realize_prompt(task, query.text, ranked, feedback=feedback)
        if meta_retriever is not None:                              # inject policy know-how (#54)
            pols = meta_retriever.retrieve(task.text, k=2)
            if pols:
                used_policies = [pid for pid, _ in pols]
                prompt = inject_policies(prompt, [t for _, t in pols])
        round_err = "no attempt verified"
        for gen in gen_fn([prompt] * samples):
            if budget is not None:
                budget.charge(cost_fn(gen))                         # pay for the generation
            code = _extract_code(gen)
            called = [n for n in names if n not in _def_names(code)
                      and re.search(rf"\b{re.escape(n)}\s*\(", code)]
            deps = ((extra_deps + "\n\n") if extra_deps else "") + \
                   "\n\n".join(c for n, c in ranked if n in called)
            ok, err = _task_verify_detail(task, code, deps)
            if ok:
                history.append((rnd, True, ""))
                return dict(solved=True, rounds=rnd + 1, code=code, used=called,
                            used_policies=used_policies, tokens=(budget.spent if budget else None),
                            history=history)
            last, round_err = (code, called, False), err
            if budget is not None and budget.exhausted():
                break
        history.append((rnd, False, round_err))
        feedback = round_err                                         # obs-informed: steer next round
    return dict(solved=False, rounds=rnd + 1, code=last[0], used=last[1],
                used_policies=used_policies, tokens=(budget.spent if budget else None), history=history)


def search_compose(task, retriever: MGRetriever, gen_fn, query: Query, k: int = 6, samples: int = 8,
                   min_cos: float = 0.2):
    """Guided SEARCH + VERIFY: retrieve ranked atoms (by the query vec), glue-realize best-of-N
    (query text as a feature), verify each by execution, keep a VERIFIED one — preferring one that
    actually composed a retrieved atom. Returns (code, used_atoms, verified, trace) where trace =
    per-sample [(called_atoms, verified)] for diagnosis (why inline vs compose vs fail)."""
    ranked = retriever.retrieve_vec(query.vec, k=k, min_cos=min_cos)   # [(fn_name, code)]
    names = [n for n, _ in ranked]
    prompt = _realize_prompt(task, query.text, ranked)
    pick, fallback, trace = None, ("", [], False), []
    for gen in gen_fn([prompt] * samples):
        code = _extract_code(gen)
        called = [n for n in names if n not in _def_names(code)
                  and re.search(rf"\b{re.escape(n)}\s*\(", code)]
        deps = "\n\n".join(c for n, c in ranked if n in called)
        ok = _task_verify(task, code, deps)                  # VERIFY grounds the search
        trace.append((called, ok, code))
        if ok:
            if called and (pick is None):                    # prefer a composing solution
                pick = (code, called, True)
            elif fallback[0] == "" or not fallback[2]:
                fallback = (code, called, True)
        elif fallback[0] == "":
            fallback = (code, called, False)
    c, u, v = pick or fallback
    return c, u, v, trace


def reason_grow(task, graph_path: str, out_path: str, retriever: MGRetriever, gen_fn,
                embed_fn, concept: str = "concept_algorithms", k: int = 6, samples: int = 8,
                min_cos: float = 0.2):
    """One reasoning step: query -> search+verify -> if it composed, GROW edges (the composition
    memory) so the next search is biased by what worked."""
    query = make_query(task, embed_fn)
    code, used, verified, trace = search_compose(task, retriever, gen_fn, query, k=k, samples=samples,
                                                 min_cos=min_cos)
    grown = None
    if verified and used:                                    # verified COMPOSITION -> edges
        sol_id = f"impl_{task.name}"
        stores = [(sol_id, code, f"solves: {task.text[:60]}")]
        edges = [(sol_id, concept, "part_of")] + [(sol_id, f"impl_{a}", "depend") for a in used]
        grown = grow(graph_path, out_path, propose_edits(f"reason_{task.name}", stores, edges))
    return dict(name=task.name, verified=verified, used=used, code=code, grown=grown, trace=trace)


def solve_multistep(mtask, base_graph_path: str, gen_fn, embed_fn, promote_to: str = None,
                    max_rounds: int = 2, samples: int = 4, k: int = 8, min_cos: float = 0.2):
    """#50 + #51: solve ORDERED sub-goals; each verified sub-result becomes a TEMP node in an
    in-session scratch graph (working memory the next step RETRIEVES = the progression lives in the
    graph, not the context window -> anti-forget). PROMOTE the whole chain to permanent memory
    (graph_grower) only if the FINAL verifies; else the scratch is discarded. Reuses solve_iterative
    per step, so each sub-goal gets the obs-informed retry loop (#49)."""
    import json
    import tempfile
    from pathlib import Path
    from v5.runtime.algo_graph_run import MBPPTask
    base = json.loads(Path(base_graph_path).read_text(encoding="utf-8"))
    scratch = {"metadata": base.get("metadata", {}), "nodes": list(base["nodes"]),
               "edges": list(base.get("edges", []))}
    sp = str(Path(tempfile.mkdtemp()) / "scratch.json")

    def _reindex():
        Path(sp).write_text(json.dumps(scratch), encoding="utf-8")
        return MGRetriever(MemoryGraph.load_json(sp), embed_fn)

    def _scratch_code():                                     # all impl code in scope (transitive deps)
        return "\n\n".join(n["metadata"]["code"] for n in scratch["nodes"]
                           if n.get("node_type") == "implementation" and n.get("metadata", {}).get("code"))

    concept = next((n["id"] for n in base["nodes"] if n.get("node_type") == "concept"), None)
    retr = _reindex()
    temp_nodes, progress = [], []
    for i, st in enumerate(mtask.steps):
        sub = MBPPTask(st.name, st.text, st.tests)
        res = solve_iterative(sub, retr, gen_fn, embed_fn, max_rounds=max_rounds, samples=samples,
                              k=k, min_cos=min_cos, extra_deps=_scratch_code())
        if not res["solved"]:                                # progression so far is IN the scratch graph
            return dict(solved=False, failed_step=i, progress=progress,
                        temp_nodes=[t[0] for t in temp_nodes])
        nid = f"impl_{st.name}"
        scratch["nodes"].append({"id": nid, "text": st.text, "node_type": "implementation",
                                 "metadata": {"code": res["code"]}})       # TEMP node (in-session only)
        temp_nodes.append((nid, st.name, res["code"], set(st.needs)))
        progress.append(st.name)
        retr = _reindex()                                    # re-index -> next step retrieves it
    ftask = MBPPTask(mtask.final_name, mtask.text + " — call the step functions you built",
                     mtask.final_tests)
    fres = solve_iterative(ftask, retr, gen_fn, embed_fn, max_rounds=max_rounds, samples=samples, k=k,
                           min_cos=min_cos, extra_deps=_scratch_code())
    promoted = None
    if fres["solved"] and promote_to:                        # verified chain -> PROMOTE to permanent
        fid = f"impl_{mtask.final_name}"
        stores = [(nid, code, f"multistep sub-atom: {name}") for nid, name, code, _ in temp_nodes]
        stores.append((fid, fres["code"], f"solves: {mtask.name}"))
        edges = [(fid, nid, "depend") for nid, _, _, _ in temp_nodes]        # final depends on steps
        for nid, _, _, needs in temp_nodes:                                  # steps tie into base atoms
            edges += [(nid, f"impl_{a}", "depend") for a in needs]
            if concept:
                edges.append((nid, concept, "part_of"))                      # ...and the concept
        promoted = grow(base_graph_path, promote_to, propose_edits(f"ms_{mtask.name}", stores, edges))
    return dict(solved=fres["solved"], progress=progress, temp_nodes=[t[0] for t in temp_nodes],
                final_code=fres["code"], promoted=promoted)


# ═══════════════════════════════════════════════════════════════════════════════
# REAL-LM RUN — the honest test: search+verify needs NO training (glue is trivial, so a FROZEN model
# can realize it). Seed atoms -> reason_grow each compose-necessary task -> does reuse move? The pure-
# LM approaches got reuse ~8% because compose was a rare spontaneous decision; here it's a SEARCH.
# ═══════════════════════════════════════════════════════════════════════════════

def _build_lm(model_name: str, max_new_tokens: int = 256, temperature: float = 0.8):
    """(gen_fn, embed_fn) from a frozen LM — gen_fn: list[str] prompts -> list[str] gens (best-of-N by
    sampling). The glue-realizer's only job is to call named atoms, so a frozen model suffices."""
    import os
    import torch
    from transformers import AutoTokenizer
    from v5.lm_loader import load_frozen_lm
    from v5.memory.store import make_mpnet_embedder

    trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
    model = load_frozen_lm(model_name)
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
    dev = next(model.parameters()).device

    def _encode(prompt):
        m = [{"role": "user", "content": prompt}]
        kw = dict(add_generation_prompt=True, return_tensors="pt", return_dict=True)
        try:
            return tok.apply_chat_template(m, enable_thinking=False, **kw)["input_ids"].to(dev)
        except TypeError:
            return tok.apply_chat_template(m, **kw)["input_ids"].to(dev)

    @torch.no_grad()
    def gen_fn(prompts):
        # search_compose passes [prompt]*samples (identical) -> one encode, num_return_sequences
        out = []
        uniq = list(dict.fromkeys(prompts))
        for p in uniq:
            n = prompts.count(p)
            pids = _encode(p)
            seqs = model.generate(pids, do_sample=(n > 1 or temperature > 0), temperature=temperature,
                                  top_p=0.95, num_return_sequences=n, max_new_tokens=max_new_tokens,
                                  pad_token_id=tok.eos_token_id)
            out += [tok.decode(seqs[j, pids.shape[1]:], skip_special_tokens=True) for j in range(n)]
        return out

    return gen_fn, make_mpnet_embedder()


def run_reason(model_name: str, graph_path: str, tasks=None, concept: str = "concept_algorithms",
               k: int = 6, samples: int = 8, min_cos: float = 0.25, seed_atoms: bool = True,
               dump: str = "artifacts/reason_dump.txt"):
    """Seed atoms (bootstrap the candidate set) -> for each compose-necessary task: query -> search +
    verify + glue -> grow edges. Reports solve% and REUSE% (composed a stored atom) — the metric the
    pure-LM runs couldn't move — plus the edges grown (composition memory). `dump` writes each task's
    per-sample trace (how many of N composed / verified / composed+verified) + picked code, so an
    inline/FAILED case is diagnosable (survives molab reset via stdout too)."""
    from v5.runtime.algo_compose_tasks import COMPOSE, seed_atom_graph
    if tasks is None:
        tasks = COMPOSE
    if seed_atoms or not Path(graph_path).exists():
        seed_atom_graph(graph_path, concept)                 # bootstrap: verified atoms in the graph
    gen_fn, embed = _build_lm(model_name)
    retr = MGRetriever(MemoryGraph.load_json(graph_path), embed)
    n_atoms = len(retr.ids)
    print(f"reason/run: {model_name} | graph {graph_path} | {n_atoms} atoms seeded | {len(tasks)} "
          f"compose-necessary tasks | k={k} samples={samples}", flush=True)

    solved = composed = 0
    dlines = []
    for task in tasks:
        tmp = graph_path + ".grown"
        res = reason_grow(task, graph_path, tmp, retr, gen_fn, embed, concept=concept, k=k,
                          samples=samples, min_cos=min_cos)
        solved += int(res["verified"]); composed += int(bool(res["used"]))
        tag = "COMPOSED " + "+".join(res["used"]) if res["used"] else (
              "verified (inline)" if res["verified"] else "FAILED")
        # per-sample trace: of N, how many composed an atom / verified / BOTH (the diagnosis)
        tr = res.get("trace", [])
        n_comp = sum(1 for c, _, _ in tr if c)
        n_ver = sum(1 for _, v, _ in tr if v)
        n_both = sum(1 for c, v, _ in tr if c and v)
        print(f"  {task.name:22s} {tag:28s} [of {len(tr)}: {n_comp} composed, {n_ver} verified, "
              f"{n_both} composed+verified]", flush=True)
        dlines.append(f"### {task.name}  ->  {tag}   "
                      f"(composed {n_comp}/{len(tr)}, verified {n_ver}, both {n_both})\n"
                      f"--- picked code ---\n{res['code']}\n"
                      + "".join(f"--- sample {i} (called={c or '[]'}, verified={v}) ---\n{code}\n"
                               for i, (c, v, code) in enumerate(tr)) + "\n")
        if res.get("grown") and res["grown"].get("persisted"):  # composition memory grew -> chain it
            Path(tmp).replace(graph_path)
            retr = MGRetriever(MemoryGraph.load_json(graph_path), embed)
    g = MemoryGraph.load_json(graph_path)
    n_edges = sum(1 for e in g.edges if e.relation in ("depend", "part_of")
                  and e.src.startswith("impl_") and e.dst.startswith("impl_"))
    N = len(tasks)
    print(f"\n  solve={solved}/{N} ({solved/N:.0%})  REUSE={composed}/{N} ({composed/N:.0%})  "
          f"| graph {len(g.nodes)} nodes, {len(g.edges)} edges ({n_edges} atom->atom composition edges)",
          flush=True)
    print("  (reuse% is the number the pure-LM GRPO/STaR runs left at ~8% — search+verify targets it)")
    if dump:
        Path(dump).parent.mkdir(parents=True, exist_ok=True)
        Path(dump).write_text("\n".join(dlines), encoding="utf-8")
        print(f"  per-sample trace + code -> {dump}", flush=True)
    return dict(solve=solved / N, reuse=composed / N, nodes=len(g.nodes), edges=len(g.edges))


def run_multistep(model_name: str, graph_path: str = "graphs/algo_multistep.json", n_tasks: int = 0,
                  seed: int = 7, hard: bool = True, promote: bool = True, max_rounds: int = 2,
                  samples: int = 6, k: int = 8, min_cos: float = 0.25):
    """Real-LM multi-step run: solve decomposable tasks step-by-step with the graph as working memory;
    promote verified chains to permanent atoms. Reports solved% + how the graph GREW (compounding)."""
    from v5.runtime.algo_compose_tasks import seed_atom_graph
    from v5.runtime.algo_multistep_tasks import MULTISTEP, gen_multistep_tasks
    if not Path(graph_path).exists():
        seed_atom_graph(graph_path, hard=hard)
    gen_fn, embed = _build_lm(model_name)
    tasks = gen_multistep_tasks(n_tasks, seed) if n_tasks else MULTISTEP
    n0 = len(MemoryGraph.load_json(graph_path).nodes)
    print(f"multistep/run: {model_name} | {graph_path} | {len(tasks)} tasks | promote={promote}", flush=True)
    solved = promoted = 0
    for t in tasks:
        tmp = graph_path + ".grown"
        res = solve_multistep(t, graph_path, gen_fn, embed, promote_to=(tmp if promote else None),
                              max_rounds=max_rounds, samples=samples, k=k, min_cos=min_cos)
        solved += int(res["solved"])
        did = bool(res.get("promoted") and res["promoted"].get("persisted"))
        promoted += int(did)
        tag = "SOLVED " + "->".join(res["progress"]) if res["solved"] else f"FAILED@step{res.get('failed_step')}"
        print(f"  {t.name:24s} {tag}{'  [+promoted]' if did else ''}", flush=True)
        if did:
            Path(tmp).replace(graph_path)                    # compound: promoted atoms persist
    g = MemoryGraph.load_json(graph_path)
    print(f"\n  solved={solved}/{len(tasks)} ({solved/len(tasks):.0%})  promoted={promoted}  | graph "
          f"{n0} -> {len(g.nodes)} nodes (compounding: multi-step atoms became reusable)", flush=True)
    return dict(solved=solved / len(tasks), promoted=promoted, nodes=len(g.nodes))


def budget_curve(model_name: str, graph_path: str, budgets=(150, 300, 600, 1200), n_tasks: int = 30,
                 hard: bool = True, seed: int = 99, samples: int = 4, k: int = 6, min_cos: float = 0.25):
    """GRR-2: the solve-vs-budget curve — solve held-out tasks under each total-generation budget. This
    is 'benefit under budget' made first-class (the 200-vs-400 sweep, generalized). GRR-3 measures the
    same curve WITH vs WITHOUT an atom to get its Δcapability."""
    from v5.runtime.algo_compose_tasks import gen_compose_tasks, seed_atom_graph
    if not Path(graph_path).exists():
        seed_atom_graph(graph_path, hard=hard)
    gen_fn, embed = _build_lm(model_name)
    retr = MGRetriever(MemoryGraph.load_json(graph_path), embed)
    tasks = gen_compose_tasks(n_tasks, seed, hard=hard)
    print(f"budget_curve: {model_name} | {'hard' if hard else 'easy'} | {n_tasks} tasks | "
          f"budgets(~tokens) {list(budgets)}\n  budget   solve            avg tokens/solve", flush=True)
    out = {}
    for B in budgets:
        solved, toks = 0, []
        for task in tasks:
            bud = Budget(B)
            r = solve_iterative(task, retr, gen_fn, embed, max_rounds=12, samples=samples, k=k,
                                min_cos=min_cos, budget=bud)
            solved += int(r["solved"])
            if r["solved"]:
                toks.append(r["tokens"])
        rate = solved / len(tasks)
        avg = f"{sum(toks)/len(toks):.0f}" if toks else "-"
        print(f"  {B:6d}   {solved}/{len(tasks)} ({rate:.0%})       {avg}", flush=True)
        out[B] = rate
    print("  (rising solve% with budget = amortization has room to pay; GRR-3 measures per-atom Δ here)")
    return out


def harvest(model_name: str, graph_path: str, out: str = "artifacts/reason_harvest.jsonl",
            n_tasks: int = 200, samples: int = 8, k: int = 6, min_cos: float = 0.25, seed: int = 0,
            concept: str = "concept_algorithms", composed_only: bool = True, hard: bool = False,
            fuzz_gate: bool = True):
    """Run the WORKING search+verify over many generated compose tasks; save every COMPOSED+verified
    sample as {task, prompt, code, used}. hard=True: inline-FAILING DP families + bigger seeded graph.
    fuzz_gate (GRR-1): also require the solution to be GENERAL (correct on random inputs vs oracle), not
    just pass the 2 benchmark asserts — kills the ~31% overfit/buggy solutions verify alone let through."""
    import json
    import time
    from v5.runtime.algo_compose_tasks import gen_compose_tasks, seed_atom_graph
    from v5.runtime.algo_quality import fuzz
    if not Path(graph_path).exists():
        seed_atom_graph(graph_path, concept, hard=hard)
    gen_fn, embed = _build_lm(model_name)
    retr = MGRetriever(MemoryGraph.load_json(graph_path), embed)
    tasks = gen_compose_tasks(n_tasks, seed, hard=hard)
    rows, n_ver, n_comp, n_overfit = [], 0, 0, 0
    t_start = time.perf_counter()
    for ti, task in enumerate(tasks):
        q = make_query(task, embed)
        ranked = retr.retrieve_vec(q.vec, k=k, min_cos=min_cos)
        names = [n for n, _ in ranked]
        prompt = _realize_prompt(task, q.text, ranked)
        for gen in gen_fn([prompt] * samples):
            code = _extract_code(gen)
            called = [n for n in names if n not in _def_names(code)
                      and re.search(rf"\b{re.escape(n)}\s*\(", code)]
            deps = "\n\n".join(c for n, c in ranked if n in called)
            if _task_verify(task, code, deps):
                n_ver += 1
                n_comp += bool(called)
                if called or not composed_only:
                    if fuzz_gate:                        # GRR-1: general, not just benchmark-verified
                        p, t = fuzz(code, task.name, deps, n=40)
                        if t and p < t:                  # overfit -> reject (verify alone accepted it)
                            n_overfit += 1
                            continue
                    rows.append({"task": task.name, "prompt": prompt, "code": code, "used": called})
        if (ti + 1) % 25 == 0:
            print(f"  ..{ti+1}/{len(tasks)} tasks | {n_ver} verified, {n_comp} composed", flush=True)
    seen, uniq = set(), []
    for r in rows:                                           # dedup identical (task, code)
        key = (r["task"], r["code"])
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(json.dumps(r) for r in uniq), encoding="utf-8")
    dt = time.perf_counter() - t_start
    print(f"\nharvest: {len(tasks)} tasks x{samples} = {len(tasks)*samples} samples -> {n_ver} verified "
          f"({n_comp} composed, {n_overfit} REJECTED as overfit by fuzz-gate) -> {len(uniq)} unique "
          f"general rows -> {out}", flush=True)
    print(f"  wall {dt:.0f}s ({dt/60:.1f} min, {len(tasks)*samples/max(dt,1):.1f} samples/s)", flush=True)
    return uniq


def retrieval_eval(graph_path: str, n_tasks: int = 200, k: int = 6, min_cos: float = 0.25,
                   seed: int = 99, hard: bool = False, embed=None):
    """Retrieval QUALITY (no LM): for each task, is the NEEDED atom in the top-k retrieved? This caps
    compose — at a 9-node graph with k=6, 2 atoms are dropped per query, so a miss means the model is
    never offered the atom it needs. Reports recall@k, the ceiling recall@all (min_cos only), avg
    retrieved, and per-family recall."""
    from v5.runtime.algo_compose_tasks import _NEEDS, gen_compose_tasks, seed_atom_graph
    if not Path(graph_path).exists():
        seed_atom_graph(graph_path, hard=hard)
    if embed is None:
        from v5.memory.store import make_mpnet_embedder
        embed = make_mpnet_embedder()
    retr = MGRetriever(MemoryGraph.load_json(graph_path), embed)
    n_atoms = len(retr.ids)
    tasks = gen_compose_tasks(n_tasks, seed, hard=hard)
    hitk = hitall = tot = 0
    ret_sizes = []
    per_fam: dict[str, list[int]] = {}
    for task in tasks:
        q = make_query(task, embed)
        gk = {n for n, _ in retr.retrieve_vec(q.vec, k=k, min_cos=min_cos)}
        gall = {n for n, _ in retr.retrieve_vec(q.vec, k=n_atoms, min_cos=min_cos)}  # ceiling: min_cos only
        need = _NEEDS[task.name]
        ok = need <= gk
        hitk += ok; hitall += (need <= gall); tot += 1
        ret_sizes.append(len(gk))
        pf = per_fam.setdefault(task.name, [0, 0]); pf[0] += ok; pf[1] += 1
    print(f"retrieval_eval: {graph_path} | {n_atoms} atoms | k={k} min_cos={min_cos} | {tot} tasks", flush=True)
    print(f"  recall@k = {hitk}/{tot} ({hitk/tot:.0%})   ceiling recall@all(min_cos only) = "
          f"{hitall}/{tot} ({hitall/tot:.0%})   avg retrieved = {sum(ret_sizes)/len(ret_sizes):.1f}/{k}",
          flush=True)
    for fam, (h, t) in sorted(per_fam.items()):
        print(f"    {fam:24s} recall@k {h}/{t} ({h/t:.0%})", flush=True)
    gap = hitall - hitk
    if gap:
        print(f"  NOTE: k={k} drops the needed atom on {gap} tasks that min_cos alone would keep -> "
              f"raise --k (>= n_atoms={n_atoms} removes the drop) to lift the compose ceiling", flush=True)
    return dict(recall_k=hitk / tot, recall_all=hitall / tot)


# ═══════════════════════════════════════════════════════════════════════════════
# SELFTEST (no model) — the SYNERGY: one query -> retrieve + realizer-feature -> search+verify ->
# a COMPOSE-NECESSARY task solved by composing TWO atoms -> edges grow (composition memory)
# ═══════════════════════════════════════════════════════════════════════════════

def _selftest() -> bool:
    import json
    import tempfile
    from pathlib import Path
    from v5.memory.store import make_fake_embedder
    from v5.runtime.algo_graph_run import MBPPTask
    print("algo_graph_reason --selftest: query -> retrieve+feature -> search+verify(compose 2 atoms) "
          "-> grow edges\n")
    embed = make_fake_embedder()

    IS_PRIME = ("def is_prime(n):\n    if n < 2:\n        return False\n    i = 2\n"
                "    while i * i <= n:\n        if n % i == 0:\n            return False\n"
                "        i += 1\n    return True")
    DIGIT_SUM = "def digit_sum(n):\n    return sum(int(c) for c in str(abs(n)))"

    with tempfile.TemporaryDirectory() as td:
        # seed a graph that ALREADY has the two atoms (the search needs a candidate set)
        gp = str(Path(td) / "g.json")
        nodes = [{"id": "concept_number_theory", "text": "number theory", "node_type": "concept"},
                 {"id": "impl_is_prime", "text": "prime test — is n prime", "node_type": "implementation",
                  "metadata": {"code": IS_PRIME}},
                 {"id": "impl_digit_sum", "text": "digit sum of an integer", "node_type": "implementation",
                  "metadata": {"code": DIGIT_SUM}}]
        Path(gp).write_text(json.dumps({"metadata": {}, "nodes": nodes, "edges": []}))
        g = MemoryGraph.load_json(gp)
        retr = MGRetriever(g, embed)

        # a COMPOSE-NECESSARY task: sum of digit-sums of the primes in a list (needs BOTH atoms)
        task = MBPPTask(
            "sum_digitsum_primes",
            "sum of digit-sums of the prime numbers in a list. needs: prime test and digit sum",
            ["assert sum_digitsum_primes([11, 4, 23]) == (1+1) + (2+3)",   # primes 11,23 -> 2 + 5 = 7
             "assert sum_digitsum_primes([8, 9, 10]) == 0"])               # no primes -> 0

        # the QUERY threads the loop: it retrieves the atoms AND features the realizer prompt
        q = make_query(task, embed)
        ranked = retr.retrieve_vec(q.vec, k=6, min_cos=-1.0)   # fake embedder: top-k regardless of score
        assert {n for n, _ in ranked} >= {"is_prime", "digit_sum"}, f"query retrieved atoms: {ranked}"
        p = _realize_prompt(task, q.text, ranked)
        assert q.text in p and "is_prime" in p and "digit_sum" in p, "query is a realizer FEATURE"
        print("  [1] one query -> retrieves BOTH atoms + is a realizer feature -> PASS")

        # glue-realizer stub: composes the two named atoms (the LM only writes glue)
        def _glue(prompts):
            return ["```python\ndef sum_digitsum_primes(lst):\n"
                    "    return sum(digit_sum(x) for x in lst if is_prime(x))\n```"] * len(prompts)

        code, used, verified, _tr = search_compose(task, retr, _glue, q, min_cos=-1.0)
        assert verified and set(used) == {"is_prime", "digit_sum"}, f"composed BOTH atoms: {used}"
        print(f"  [2] search+verify: compose-necessary task solved by GLUING 2 atoms {sorted(used)} -> PASS")

        # verified composition -> GROW edges (the composition memory): sol depend is_prime + digit_sum
        out = str(Path(td) / "g2.json")
        res = reason_grow(task, gp, out, retr, _glue, embed, concept="concept_number_theory", min_cos=-1.0)
        assert res["verified"] and res["grown"]["gate_passed"], res
        g2 = MemoryGraph.load_json(out)
        assert g2.edge_between("impl_sum_digitsum_primes", "impl_is_prime") is not None
        assert g2.edge_between("impl_sum_digitsum_primes", "impl_digit_sum") is not None
        print(f"  [3] verified composition -> {res['grown']['edit_stats']['edge_edits']} depend/part_of "
              f"EDGES grown (composition memory) -> PASS")

        # [4] ITERATIVE loop: round-1 attempt FAILS, the failure is fed back, round-2 SOLVES
        retr2 = MGRetriever(MemoryGraph.load_json(gp), embed)     # graph irrelevant here; task inlines
        ftask = MBPPTask("add_one", "return n + 1", ["assert add_one(3) == 4"])

        def _retry_stub(prompts):
            # buggy until the prompt carries the failure feedback, then correct (obs-informed retry)
            good = "```python\ndef add_one(n):\n    return n + 1\n```"
            bad = "```python\ndef add_one(n):\n    return n - 1\n```"
            return [good if "PREVIOUS attempt failed" in prompts[0] else bad] * len(prompts)

        it = solve_iterative(ftask, retr2, _retry_stub, embed, max_rounds=3, samples=2, min_cos=-1.0)
        assert it["solved"] and it["rounds"] == 2, it
        assert it["history"][0][1] is False and "add_one(3) == 4" not in it["history"][0][2], it["history"]
        print(f"  [4] iterative loop: round 1 FAILED ({it['history'][0][2][:34]}...) -> fed back -> "
              f"round {it['rounds']} SOLVED -> PASS")

        # [5] MULTI-STEP: solve ordered sub-goals -> temp nodes (working memory) -> promote verified chain
        from v5.runtime.algo_compose_tasks import seed_atom_graph
        from v5.runtime.algo_multistep_tasks import MULTISTEP, _REF_FINAL, _REF_STEP
        msg = str(Path(td) / "ms_base.json"); seed_atom_graph(msg, hard=True)   # atoms: is_prime, digit_sum...
        allref = {**_REF_STEP, **_REF_FINAL}

        def _ms_stub(prompts):                                # solves each step/final by fn name
            p = prompts[0]
            for nm, cd in allref.items():
                if f"Write `{nm}(" in p:
                    return [f"```python\n{cd}\n```"] * len(prompts)
            return ["```python\ndef _x():\n    return 0\n```"] * len(prompts)

        mt = MULTISTEP[0]            # primes_then_digitsum: keep_primes -> total_ds -> final
        out5 = str(Path(td) / "ms_grown.json")
        ms = solve_multistep(mt, msg, _ms_stub, embed, promote_to=out5, max_rounds=1, samples=1, min_cos=-1.0)
        assert ms["solved"] and ms["progress"] == ["keep_primes", "total_ds"], ms
        assert ms["promoted"] and ms["promoted"]["persisted"], ms
        g5 = MemoryGraph.load_json(out5)
        assert "impl_keep_primes" in g5.nodes and "impl_primes_then_digitsum" in g5.nodes
        assert g5.edge_between("impl_primes_then_digitsum", "impl_keep_primes") is not None
        print(f"  [5] multi-step: solved {'->'.join(ms['progress'])} via temp nodes -> verified chain "
              f"PROMOTED (+{len(ms['temp_nodes'])} atoms + depend edges) -> PASS")

        # [6] BUDGET bites: same task+stub, tight budget runs out before the retry -> unsolved;
        # loose budget lets the obs-informed retry finish -> solved
        btask = MBPPTask("add_one", "return n + 1", ["assert add_one(3) == 4"])

        def _bstub(prompts):
            good = "```python\ndef add_one(n):\n    return n + 1\n```"
            bad = "```python\ndef add_one(n):\n    return n - 1\n```"
            return [good if "PREVIOUS attempt failed" in prompts[0] else bad] * len(prompts)

        one_cost = _char_cost("```python\ndef add_one(n):\n    return n - 1\n```")
        tight = solve_iterative(btask, retr2, _bstub, embed, max_rounds=5, samples=1, min_cos=-1.0,
                                budget=Budget(one_cost))          # only affords the first (wrong) attempt
        loose = solve_iterative(btask, retr2, _bstub, embed, max_rounds=5, samples=1, min_cos=-1.0,
                                budget=Budget(10000))
        assert not tight["solved"] and tight["tokens"] >= one_cost, tight
        assert loose["solved"] and loose["tokens"] > 0, loose
        print(f"  [6] budget bites: tight (~{one_cost} tok) -> UNSOLVED (spent {tight['tokens']}); "
              f"loose -> SOLVED (spent {loose['tokens']}) -> PASS")

    print("\n  ALGO_GRAPH_REASON SELFTEST -> PASS")
    return True


def main():
    ap = argparse.ArgumentParser(description="Reasoning-over-graph: guided search + verify + glue-realizer.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true", help="real-LM run on the compose-necessary tasks")
    ap.add_argument("--multistep", action="store_true", help="real-LM multi-step run (working memory + temp nodes)")
    ap.add_argument("--budget-curve", action="store_true", help="GRR-2: solve% vs generation budget")
    ap.add_argument("--harvest", action="store_true", help="harvest verified compositions -> training corpus")
    ap.add_argument("--retrieval-eval", action="store_true", help="retrieval recall@k (no LM) — does the query deliver the needed atom?")
    ap.add_argument("--hard", action="store_true", help="hard suite (inline-failing DP atoms + bigger graph)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--graph", default="graphs/algo_reason.json", help="MemoryGraph (seeded with atoms)")
    ap.add_argument("--out", default="artifacts/reason_harvest.jsonl", help="harvest output jsonl")
    ap.add_argument("--n-tasks", type=int, default=200, help="generated compose instances to harvest over")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--samples", type=int, default=8, help="best-of-N glue realizations per task")
    ap.add_argument("--min-cos", type=float, default=0.25)
    ap.add_argument("--no-seed", action="store_true", help="do NOT re-seed atoms (use existing graph)")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    if a.multistep:
        gpath = a.graph if a.graph != "graphs/algo_reason.json" else "graphs/algo_multistep.json"
        run_multistep(a.model, gpath, n_tasks=a.n_tasks, hard=a.hard, samples=a.samples, k=a.k,
                      min_cos=a.min_cos)
        return
    if a.budget_curve:
        budget_curve(a.model, a.graph, n_tasks=a.n_tasks, hard=a.hard, samples=a.samples, k=a.k,
                     min_cos=a.min_cos)
        return
    if a.retrieval_eval:
        retrieval_eval(a.graph, n_tasks=a.n_tasks, k=a.k, min_cos=a.min_cos, hard=a.hard)
        return
    if a.harvest:
        harvest(a.model, a.graph, out=a.out, n_tasks=a.n_tasks, samples=a.samples, k=a.k,
                min_cos=a.min_cos, hard=a.hard)
        return
    if a.run:
        run_reason(a.model, a.graph, k=a.k, samples=a.samples, min_cos=a.min_cos,
                   seed_atoms=not a.no_seed)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
