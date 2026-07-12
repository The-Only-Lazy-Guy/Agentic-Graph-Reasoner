"""Anti-cheat — stop the model storing literal ANSWERS instead of reusable atoms (Q2). Two measured,
credit-enforced rules (turns "don't cheat" from hope into a number):

  1. generalization gap  = solve(train tasks) - solve(HELD-OUT tasks). A node that memorizes an answer
     helps its own train instance but NOT held-out instances of the same family -> the gap spikes. The
     fingerprint store-gate + held-out eval are partial defenses; this MEASURES the leak.

  2. single-use penalty  = a node reused by <=1 distinct task family is over-specific (a stored answer,
     or a structurally-memorized artifact) -> PUNISH it; reward reuse across DIVERSE families. This is
     the original curriculum requirement: "reward diverse amortized reuse, punish the non-reusable."
     Feeds graph_edits confidence (STRENGTHEN diverse, WEAKEN/RETIRE single-use).

  selftest (no model):  python -m v5.runtime.algo_anticheat --selftest
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict


def generalization_gap(solve_fn, train_tasks, held_tasks, thresh: float = 0.2) -> dict:
    """solve_fn(task) -> bool. Gap = train_solve - held_solve. A large gap means the graph is helping
    on SEEN instances but not GENERALIZING -> memorization suspected (the honest cheat-detector)."""
    tr = [bool(solve_fn(t)) for t in train_tasks]
    he = [bool(solve_fn(t)) for t in held_tasks]
    train = sum(tr) / max(1, len(tr))
    held = sum(he) / max(1, len(he))
    gap = train - held
    return dict(train=round(train, 3), held=round(held, 3), gap=round(gap, 3),
                verdict=("MEMORIZATION SUSPECTED" if gap > thresh else "generalizes"))


def reuse_audit(usage) -> dict:
    """usage = [(task_family, [atoms_used]), ...] over SOLVED tasks. Returns each atom's diversity =
    the number of DISTINCT task families that reused it, and the single-use set (diversity <= 1)."""
    fams = defaultdict(set)
    for fam, atoms in usage:
        for a in atoms:
            fams[a].add(fam)
    diversity = {a: len(f) for a, f in fams.items()}
    single_use = sorted(a for a, n in diversity.items() if n <= 1)
    return dict(diversity=diversity, single_use=single_use)


def reuse_reward(diversity: int) -> float:
    """Node credit from its reuse diversity (the §4 shape): diverse reuse -> reward (saturating);
    single-use -> PUNISH (over-specific / memorized); never reused -> mild decay (dead weight)."""
    if diversity <= 0:
        return -0.10                       # dead node — decay
    if diversity == 1:
        return -0.25                       # single-use / over-specific — punish
    return min(1.0, 0.20 * diversity)      # reused across D diverse families — reward


def audit_graph_nodes(usage, keep_thresh: int = 2) -> dict:
    """Roll audit + reward into a prune/keep verdict per node: keep if reused by >= keep_thresh
    families, else flag for WEAKEN/RETIRE (single-use). Ready to feed graph_edits confidence."""
    au = reuse_audit(usage)
    verdicts = {a: {"diversity": d, "reward": round(reuse_reward(d), 3),
                    "verdict": ("keep" if d >= keep_thresh else "weaken/retire")}
                for a, d in au["diversity"].items()}
    return dict(nodes=verdicts, single_use=au["single_use"],
                keep=[a for a, v in verdicts.items() if v["verdict"] == "keep"])


def anticheat_eval(model_name: str, graph_path: str, n_tasks: int = 40, hard: bool = False, k: int = 6,
                   samples: int = 4, max_rounds: int = 2, min_cos: float = 0.25):
    """Runnable check (GPU): solve TRAIN (seed 0) vs HELD-OUT (seed 99) with the iterative loop (#49),
    report the generalization gap + audit which atoms are single-use. Reuses solve_iterative."""
    from pathlib import Path
    from graph_core import MemoryGraph
    from v5.runtime.algo_compose_tasks import gen_compose_tasks, seed_atom_graph
    from v5.runtime.algo_graph_mg import MGRetriever
    from v5.runtime.algo_graph_reason import _build_lm, solve_iterative
    if not Path(graph_path).exists():
        seed_atom_graph(graph_path, hard=hard)
    gen_fn, embed = _build_lm(model_name)
    retr = MGRetriever(MemoryGraph.load_json(graph_path), embed)

    def run(tasks):
        solved, usage = [], []
        for t in tasks:
            r = solve_iterative(t, retr, gen_fn, embed, max_rounds=max_rounds, samples=samples, k=k,
                                min_cos=min_cos)
            solved.append(r["solved"])
            if r["solved"] and r["used"]:
                usage.append((t.name, r["used"]))
        return solved, usage

    print(f"anticheat_eval: {model_name} | {'hard' if hard else 'easy'} | {n_tasks} train(seed0) vs "
          f"{n_tasks} held(seed99)", flush=True)
    tr_solved, tr_usage = run(gen_compose_tasks(n_tasks, seed=0, hard=hard))
    he_solved, _ = run(gen_compose_tasks(n_tasks, seed=99, hard=hard))
    train = sum(tr_solved) / max(1, len(tr_solved))
    held = sum(he_solved) / max(1, len(he_solved))
    gap = train - held
    verdict = "MEMORIZATION SUSPECTED" if gap > 0.2 else "generalizes"
    print(f"  solve train={train:.0%}  held-out={held:.0%}  gap={gap:+.0%}  -> {verdict}", flush=True)
    audit = audit_graph_nodes(tr_usage, keep_thresh=2)
    print(f"  reuse: keep={audit['keep']}  single-use(weaken/retire)={audit['single_use']}", flush=True)
    return dict(train=train, held=held, gap=gap, audit=audit)


# ═══════════════════════════════════════════════════════════════════════════════
# SELFTEST (no model) — the memorizer is CAUGHT (gap spikes) + single-use nodes are punished
# ═══════════════════════════════════════════════════════════════════════════════

def _selftest() -> bool:
    print("algo_anticheat --selftest: gen-gap catches memorization + single-use nodes punished\n")
    train = ["famA#0", "famA#1", "famB#0", "famB#1"]
    held = ["famA#2", "famA#3", "famB#2", "famB#3"]

    # [1] a MEMORIZER solves only what it saw (train) -> gap spikes -> flagged
    memo = generalization_gap(lambda t: t in train, train, held)
    assert memo["train"] == 1.0 and memo["held"] == 0.0 and memo["gap"] == 1.0, memo
    assert "MEMORIZATION" in memo["verdict"], memo
    # a GENERALIZER solves both -> no gap -> clean
    gen = generalization_gap(lambda t: True, train, held)
    assert gen["gap"] == 0.0 and gen["verdict"] == "generalizes", gen
    print(f"  [1] gen-gap: memorizer train={memo['train']} held={memo['held']} gap={memo['gap']} "
          f"-> {memo['verdict']}; generalizer gap={gen['gap']} -> {gen['verdict']} -> PASS")

    # [2] reuse audit: atom x reused by 3 families (diverse), y by 1 (single-use answer)
    usage = [("sum_edit_distance", ["edit_distance"]),
             ("count_makeable", ["coin_change"]),
             ("max_lis", ["lis_length"]),
             ("sum_lcs", ["lcs_length"]),
             ("edit_distance", ["edit_distance"]),      # edit_distance reused across 2 families
             ("weird_one_off", ["answer_42"])]          # answer_42 used by exactly ONE family
    au = reuse_audit(usage)
    assert au["diversity"]["edit_distance"] == 2 and au["diversity"]["answer_42"] == 1, au
    assert "answer_42" in au["single_use"] and "edit_distance" not in au["single_use"], au
    print(f"  [2] reuse audit: edit_distance diversity={au['diversity']['edit_distance']} (kept), "
          f"answer_42 diversity=1 -> single-use {au['single_use']} -> PASS")

    # [3] reward shape: diverse rewarded, single-use PUNISHED, dead decays
    assert reuse_reward(3) > 0 and reuse_reward(1) < 0 and reuse_reward(0) < 0, "reward shape"
    verd = audit_graph_nodes(usage, keep_thresh=2)
    assert "answer_42" in verd["single_use"] and verd["nodes"]["answer_42"]["verdict"] == "weaken/retire"
    assert verd["nodes"]["edit_distance"]["verdict"] == "keep"
    print(f"  [3] reward: diverse={reuse_reward(3):+.2f} single-use={reuse_reward(1):+.2f} "
          f"dead={reuse_reward(0):+.2f}; answer_42 -> weaken/retire, edit_distance -> keep -> PASS")

    print("\n  ALGO_ANTICHEAT SELFTEST -> PASS")
    return True


def main():
    ap = argparse.ArgumentParser(description="Anti-cheat: generalization gap + single-use node penalty.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true", help="GPU: train-vs-held gen-gap + reuse audit")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--graph", default="graphs/algo_reason_hard.json")
    ap.add_argument("--n-tasks", type=int, default=40)
    ap.add_argument("--hard", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    if a.run:
        anticheat_eval(a.model, a.graph, n_tasks=a.n_tasks, hard=a.hard)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
