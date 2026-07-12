"""GRR-6 steps 2-3: the TRM POLICY drives the compose-forced solver (phase A, templated).

The compose-forcing comes from the REALIZER (fills atom slots in a fixed family template — it CANNOT
inline). The TRM is the reasoner that PICKS which atom fills each slot. Here, end to end and tiny:

  trace   = (task_vec, target atom idx)   from gen_compose_tasks + _NEEDS   (oracle labels, unlimited)
  train   = TRMReasoner (algo_trm) — pointer over atoms, deep supervision
  solve   = TRM scores atoms -> top pick -> realize(family, pick) -> resolve_deps -> fuzz-verify

So the "model" is a ~9k-param recurrent reasoner that learns task->atom and drives a solver that must
compose. Phase A uses single-slot HARD families (each needs exactly one atom: sum_edit_distance->
edit_distance, sum_lcs->lcs_length, count_makeable->coin_change, max_lis->lis_length). The DSL for
multi-slot / novel structure is phase B (step 5b).

  selftest (no LM, synthetic embed):  python -m v5.runtime.algo_trm_compose --selftest
"""
from __future__ import annotations

import argparse
import re
import sys

import numpy as np

from v5.runtime.algo_compose_tasks import _NEEDS, _REF

# the single-slot families (exactly one atom each) — phase A
SINGLE_SLOT = [f for f in ("sum_edit_distance", "sum_lcs", "count_makeable", "max_lis")]


def family_template(family: str):
    """(_REF wiring with the atom name replaced by a slot, the correct atom). Word-boundary sub so the
    fn name (sum_edit_distance) is NOT touched, only the called atom (edit_distance)."""
    atom = sorted(_NEEDS[family])[0]
    tmpl = re.sub(rf"\b{re.escape(atom)}\b", "__ATOM__", _REF[family])
    return tmpl, atom


def realize(family: str, chosen_atom: str) -> str:
    """Fill the family's single slot with the TRM's chosen atom -> runnable code that CALLS it."""
    tmpl, _ = family_template(family)
    return tmpl.replace("__ATOM__", chosen_atom)


def gen_traces(n: int, atom_names, embed_fn, seed: int = 0):
    """[(task_vec, target_atom_idx)] from generated single-slot tasks + the oracle _NEEDS labels."""
    from v5.runtime.algo_compose_tasks import gen_compose_tasks
    idx = {a: i for i, a in enumerate(atom_names)}
    tasks = [t for t in gen_compose_tasks(n * 3, seed=seed, hard=True) if t.name in SINGLE_SLOT][:n]
    traces = []
    for t in tasks:
        atom = sorted(_NEEDS[t.name])[0]
        if atom not in idx:
            continue
        tv = np.asarray(list(embed_fn({"q": t.text}).values())[0], dtype=np.float32)
        traces.append((tv, {idx[atom]}, t))
    return traces


def train_policy(atom_names, atom_vecs, embed_fn, n_train: int = 240, d: int = 48, T: int = 3,
                 steps: int = 500, lr: float = 5e-3, seed: int = 0):
    """Train the TRM pointer to map task_vec -> the correct atom (deep supervision, BCE)."""
    import torch
    import torch.nn as nn
    from v5.runtime.algo_trm import _build
    _, _, TRMReasoner = _build()
    torch.manual_seed(seed)
    d_in = atom_vecs.shape[1]
    model = TRMReasoner(d_in=d_in, d=d, T=T)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    A = torch.as_tensor(atom_vecs, dtype=torch.float32)
    traces = gen_traces(n_train, atom_names, embed_fn, seed=seed)
    import random
    rng = random.Random(seed)
    for _ in range(steps):
        tv, tgt, _t = traces[rng.randrange(len(traces))]
        x = torch.as_tensor(tv, dtype=torch.float32)
        target = torch.zeros(len(atom_names))
        for i in tgt:
            target[i] = 1.0
        outs = model(x, A, return_all=True)
        loss = sum(bce(o, target) for o in outs) / len(outs)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def trm_solve(model, task, atom_names, atom_vecs, graph_path, embed_fn):
    """TRM picks the top atom for the task -> realize -> resolve deps through the graph -> fuzz-verify.
    Returns (solved, chosen_atom)."""
    import torch
    from v5.runtime.algo_graph_mg import MGRetriever
    from v5.runtime.algo_quality import fuzz
    from graph_core import MemoryGraph
    with torch.no_grad():
        y = model(torch.as_tensor(np.asarray(list(embed_fn({"q": task.text}).values())[0], dtype=np.float32)),
                  torch.as_tensor(atom_vecs, dtype=torch.float32))
    chosen = atom_names[int(torch.argmax(y))]
    code = realize(task.name, chosen)
    retr = MGRetriever(MemoryGraph.load_json(graph_path), embed_fn)
    deps = retr.resolve_deps({chosen})
    passed, total = fuzz(code, task.name, deps, n=40)
    return (bool(total and passed == total), chosen)


def eval_policy(model, atom_names, atom_vecs, graph_path, embed_fn, n_eval: int = 40, seed: int = 999):
    from v5.runtime.algo_compose_tasks import gen_compose_tasks
    tasks = [t for t in gen_compose_tasks(n_eval * 3, seed=seed, hard=True) if t.name in SINGLE_SLOT][:n_eval]
    solved = correct = 0
    for t in tasks:
        ok, chosen = trm_solve(model, t, atom_names, atom_vecs, graph_path, embed_fn)
        solved += int(ok)
        correct += int(chosen == sorted(_NEEDS[t.name])[0])
    return dict(solved=solved / max(1, len(tasks)), atom_acc=correct / max(1, len(tasks)), n=len(tasks))


# ═══════════════════════════════════════════════════════════════════════════════
# SELFTEST — the TRM learns task->atom and DRIVES the compose-forced solver (no LM). Synthetic embedder
# gives each family a recoverable signal; the TRM must map it to the right atom among all candidates.
# ═══════════════════════════════════════════════════════════════════════════════

def _selftest() -> bool:
    import json
    import tempfile
    from pathlib import Path
    from v5.runtime.algo_compose_tasks import ALL_ATOMS
    print("algo_trm_compose --selftest: TRM policy learns task->atom + drives the compose-forced solver\n")

    atom_names = ["is_prime", "digit_sum", "build_adj", "dijkstra", "edit_distance", "lcs_length",
                  "coin_change", "lis_length"]
    rng = np.random.default_rng(0)
    d_in = 64
    atom_vecs = rng.standard_normal((len(atom_names), d_in)).astype("float32")
    # synthetic embedder: family name -> a fixed random base + noise (family recoverable, learnable)
    fam_base = {f: rng.standard_normal(d_in).astype("float32") for f in SINGLE_SLOT}

    # keyword -> family, proxy for what mpnet captures semantically (the task text is a DESCRIPTION,
    # not the fn name — so key on distinctive words, like a real embedder would cluster them)
    kw = [("edit", "sum_edit_distance"), ("subsequence", "sum_lcs"), ("coin", "count_makeable"),
          ("increasing", "max_lis")]

    def embed(d):                                        # handle a dict of MANY {id: text}
        out = {}
        for k, text in d.items():
            fam = next((f for w, f in kw if w in text.lower()), None)
            base = fam_base[fam] if fam else np.zeros(d_in, "float32")
            out[k] = (base + 0.35 * rng.standard_normal(d_in)).astype("float32")
        return out

    with tempfile.TemporaryDirectory() as td:
        gp = str(Path(td) / "g.json")
        nodes = [{"id": "concept_algorithms", "text": "algorithms", "node_type": "concept"}]
        for a in atom_names:
            nodes.append({"id": f"impl_{a}", "text": ALL_ATOMS[a][0], "node_type": "implementation",
                          "metadata": {"code": ALL_ATOMS[a][1]}})
        Path(gp).write_text(json.dumps({"metadata": {}, "nodes": nodes, "edges": []}))

        # [1] realizer is compose-forced: fills the slot, calls the atom, never inlines
        code = realize("sum_edit_distance", "edit_distance")
        assert "edit_distance(a, b)" in code and "def sum_edit_distance" in code and "dp" not in code
        print("  [1] templated realizer: sum_edit_distance -> calls edit_distance (no inline DP) -> PASS")

        # [2] train the TRM policy, eval held-out: does it pick the right atom + solve?
        model = train_policy(atom_names, atom_vecs, embed, n_train=300, steps=900)
        ev = eval_policy(model, atom_names, atom_vecs, gp, embed, n_eval=40)
        # synthetic-embed proof (real mpnet + molab does better); 3x the 4-family random baseline
        assert ev["atom_acc"] >= 0.7 and ev["solved"] == ev["atom_acc"], ev
        print(f"  [2] TRM policy (trained from scratch): held-out atom-pick acc={ev['atom_acc']:.0%}, "
              f"compose-forced solve={ev['solved']:.0%} over {ev['n']} tasks -> PASS")

        # [3] the TRM genuinely DECIDES (a random policy over 8 atoms would pick right ~1/8)
        from v5.runtime.algo_compose_tasks import gen_compose_tasks
        rtasks = [t for t in gen_compose_tasks(120, seed=7, hard=True) if t.name in SINGLE_SLOT][:40]
        rand = sum(int(atom_names[int(rng.integers(0, len(atom_names)))] == sorted(_NEEDS[t.name])[0])
                   for t in rtasks)
        assert ev["atom_acc"] > 3 * (rand / len(rtasks)), (ev["atom_acc"], rand / len(rtasks))
        print(f"  [3] TRM acc {ev['atom_acc']:.0%} >> random-over-8 baseline {rand/len(rtasks):.0%} "
              f"(the reasoner learned the mapping) -> PASS")

    print("\n  ALGO_TRM_COMPOSE SELFTEST -> PASS  (tiny reasoner picks atoms; realizer forces compose)")
    return True


def main():
    ap = argparse.ArgumentParser(description="GRR-6 steps 2-3: TRM policy + templated realizer (phase A).")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
