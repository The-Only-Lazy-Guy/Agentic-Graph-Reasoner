"""Edge-traversal retrieval — does graph STRUCTURE (leveraged edges) beat flat similarity?

The honest "why a graph, not an array" test. Flat retrieval ranks symbols by query↔symbol
cosine — but the issue (bug symptom) and a bare `def` signature barely overlap (the measured
query↔symbol semantic gap; symbol-recall ~0.45). STRATEGY nodes are behavior-level text that
matches the ISSUE far better, and each `leveraged` edge points from a strategy to the symbols it
actually used. So:

  query -> rank STRATEGY nodes (behavior-level, closes the gap)
        -> traverse `leveraged` -> boost the symbols those strategies used
        -> re-rank symbols

This is NOT the failed naive bidirectional neighbor-boost (READ_THIS 2026-06-03c, Hit@1 -45%):
it is DIRECTED (strategy->symbol only), TYPED (`leveraged` only), and LEAKAGE-GATED (exclude any
strategy minted from the query's OWN instance — else it trivially surfaces the gold).

Measures symbol recall@k: FLAT vs FLAT+TRAVERSAL. If traversal lifts recall leakage-free, the
edges are load-bearing -> structure earns its place. Embedding runs on a GPU box (st-embed
segfaults on Windows); `--selftest` validates the traversal logic locally with no embeddings.

  GPU: python -m v5.graph_grower.traversal_eval --model models/ranker-code --alpha 0.5
  local: python -m v5.graph_grower.traversal_eval --selftest
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Dict, List

import numpy as np


# ── load graph (symbols + strategies + leveraged edges) ──────────────────────────
def load_symbols(path: str) -> Dict[str, str]:
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("op") == "add_node" and r.get("node_type") == "symbol":
            t = (r.get("text") or "").strip()
            if t:
                out[r["node_id"]] = t
    return out


def load_strategies(path: str):
    """-> (strat_text {kb_id:text}, strat_inst {kb_id:session_id}, leveraged {kb_id:[sym_ids]})."""
    text, inst = {}, {}
    leveraged = defaultdict(list)
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        sess = r.get("session_id")
        e = r.get("raw_edit") or {}
        op = e.get("op")
        if op == "add_node":
            nid = e.get("node_id") or r.get("target_id")
            t = (r.get("text") or e.get("text") or "").strip()
            if nid and nid.startswith("kb_cot") and t:
                text[nid] = t
                inst[nid] = sess
        elif op == "add_edge" and e.get("relation") == "leveraged":
            src, dst = e.get("src"), e.get("dst")
            if src and dst and dst.startswith("sym_"):
                leveraged[src].append(dst)
                inst.setdefault(src, sess)
    return text, inst, dict(leveraged)


# ── the traversal (directed, typed, leakage-gated) — unit-testable, no embeddings ──
def expand(flat_sym: Dict[str, float], strat_score: Dict[str, float],
           leveraged: Dict[str, List[str]], strat_inst: Dict[str, str],
           query_inst: str, alpha: float, exclude_self: bool) -> Dict[str, float]:
    """expanded[sym] = flat[sym] + alpha * max strategy-score over strategies that leverage `sym`
    (excluding the query's own instance). A symbol the flat ranker scored low gets pulled up if a
    strategy that strongly matches the issue used it."""
    boost: Dict[str, float] = {}
    for kb, s in strat_score.items():
        if exclude_self and query_inst is not None and strat_inst.get(kb) == query_inst:
            continue
        for sym in leveraged.get(kb, []):
            if s > boost.get(sym, -1e9):
                boost[sym] = s
    out = dict(flat_sym)
    for sym, b in boost.items():
        out[sym] = flat_sym.get(sym, 0.0) + alpha * b
    return out


def recall_at_k(ranked: List[str], gold: set, ks=(1, 5, 10, 20)) -> Dict[int, float]:
    return {k: (len(set(ranked[:k]) & gold) / len(gold) if gold else 0.0) for k in ks}


def _selftest():
    # symbols s1..s4; strategies A(inst X, lev s3), B(inst Y, lev s4)
    flat = {"s1": 0.9, "s2": 0.5, "s3": 0.2, "s4": 0.1}      # gold s3 ranked LOW by flat
    strat = {"A": 0.95, "B": 0.3}
    lev = {"A": ["s3"], "B": ["s4"]}
    inst = {"A": "X", "B": "Y"}
    gold = {"s3"}
    flat_rank = sorted(flat, key=lambda s: -flat[s])
    exp = expand(flat, strat, lev, inst, query_inst="Z", alpha=0.5, exclude_self=True)
    exp_rank = sorted(exp, key=lambda s: -exp[s])
    ok = True
    fr = recall_at_k(flat_rank, gold, (1, 2)); er = recall_at_k(exp_rank, gold, (1, 2))
    print(f"flat rank {flat_rank} -> recall@1 {fr[1]}  | s3 boosted to {exp['s3']:.3f}")
    print(f"expanded rank {exp_rank} -> recall@2 {er[2]} (flat recall@2 {fr[2]})")
    ok &= er[2] > fr[2]                              # traversal pulls the gold symbol into top-k
    # leakage gate: if query is instance X, strategy A is excluded -> no boost
    exp_leak = expand(flat, strat, lev, inst, query_inst="X", alpha=0.5, exclude_self=True)
    ok &= abs(exp_leak["s3"] - flat["s3"]) < 1e-9   # A excluded -> s3 unboosted
    print(f"leakage gate (query=X): s3 stays {exp_leak['s3']:.3f} (A excluded) -> {'OK' if abs(exp_leak['s3']-0.2)<1e-9 else 'FAIL'}")
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


def run(a):
    from sentence_transformers import SentenceTransformer

    syms = load_symbols(a.nodes)
    s_text, s_inst, leveraged = load_strategies(a.strategies)
    sym_ids = list(syms)
    sym_set = set(sym_ids)
    print(f"symbols {len(sym_ids)} | strategies {len(s_text)} | leveraged edges "
          f"{sum(len(v) for v in leveraged.values())}", flush=True)

    # instance_id reference (the heldout gold omits it; the full gold has it) -> question->iid
    inst_ref = {}
    for refp in (a.inst_ref or []):
        try:
            for line in open(refp, encoding="utf-8"):
                rr = json.loads(line)
                if rr.get("question") and rr.get("instance_id"):
                    inst_ref[rr["question"]] = rr["instance_id"]
        except FileNotFoundError:
            pass

    gold, q_inst = {}, {}
    for line in open(a.gold, encoding="utf-8"):
        r = json.loads(line)
        q = r.get("question")
        sup = [s for s in (r.get("support_ids") or []) if s in sym_set]
        if q and sup:
            gold[q] = sup
            q_inst[q] = r.get("instance_id") or inst_ref.get(q)
    mapped = sum(1 for v in q_inst.values() if v)
    # how many strategies WILL be excluded by the gate (visibility — the bug last time was 0)?
    excl = sum(1 for q in gold for kb in s_inst
               if q_inst.get(q) and s_inst.get(kb) == q_inst[q])
    print(f"queries {len(gold)} | with instance_id {mapped} | total same-instance strategy "
          f"exclusions {excl} (gate {'FIRES' if excl else 'DEAD — check'})", flush=True)

    mdl = SentenceTransformer(a.model, trust_remote_code=True)
    enc = lambda xs: mdl.encode(xs, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    sym_emb = np.asarray(enc([syms[s] for s in sym_ids]))
    strat_ids = list(s_text)
    strat_emb = np.asarray(enc([s_text[k] for k in strat_ids]))
    qs = list(gold)
    q_emb = np.asarray(enc([q for q in qs]))

    ks = (1, 5, 10, 20)
    flat_R = {k: [] for k in ks}; exp_R = {k: [] for k in ks}
    for qi, q in enumerate(qs):
        g = set(gold[q])
        if not g:
            continue
        fsim = sym_emb @ q_emb[qi]
        flat_sym = {sym_ids[i]: float(fsim[i]) for i in range(len(sym_ids))}
        ssim = strat_emb @ q_emb[qi]
        strat_score = {strat_ids[i]: float(ssim[i]) for i in range(len(strat_ids))}
        exp_sym = expand(flat_sym, strat_score, leveraged, s_inst,
                         q_inst.get(q), a.alpha, exclude_self=not a.no_exclude)
        fr = sorted(flat_sym, key=lambda s: -flat_sym[s])
        er = sorted(exp_sym, key=lambda s: -exp_sym[s])
        for k in ks:
            flat_R[k].append(recall_at_k(fr, g, (k,))[k])
            exp_R[k].append(recall_at_k(er, g, (k,))[k])
    n = len(flat_R[1])
    print(f"\n=== symbol recall@k over {n} queries (leakage-gated={not a.no_exclude}, alpha={a.alpha}) ===")
    print(f"  {'k':>3}  {'flat':>8}  {'+traversal':>11}  {'delta':>8}")
    for k in ks:
        f = np.mean(flat_R[k]); e = np.mean(exp_R[k])
        print(f"  {k:>3}  {f:8.4f}  {e:11.4f}  {e-f:+8.4f}")
    print("\n  delta>0 at any k => leveraged edges (STRUCTURE) reach symbols flat similarity misses")
    print("  => the graph beats an array. delta~0 => structure decorative here too (honest).")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", default="artifacts/graph_growth/swe_code_candidates.jsonl")
    ap.add_argument("--strategies", default="artifacts/graph_growth/swe_strategy_candidates_clean.jsonl")
    ap.add_argument("--gold", default="data/swe/retrieval_gold_heldout.jsonl")
    ap.add_argument("--inst-ref", nargs="*",
                    default=["data/swe/retrieval_gold_code.jsonl",
                             "data/swe/retrieval_gold_code_verified.jsonl"],
                    help="gold(s) carrying instance_id -> question->instance map for the leakage gate")
    ap.add_argument("--model", default="models/ranker-code")
    ap.add_argument("--alpha", type=float, default=0.5, help="strategy-boost weight")
    ap.add_argument("--no-exclude", action="store_true", help="DISABLE leakage gate (debug; inflates)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        raise SystemExit(0 if _selftest() else 1)
    run(a)


if __name__ == "__main__":
    raise SystemExit(main())
