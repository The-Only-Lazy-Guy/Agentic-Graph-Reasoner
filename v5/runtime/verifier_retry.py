"""Verifier-retry loop (V4's loop, for code) — TIER 1: apply-feedback (no verifier).

The single-shot 4B writes SR edits whose SEARCH often doesn't match the file (search_in_file
~0.2-0.33) -> unapplyable. The loop fixes that by ITERATING: generate -> check the SEARCH against
the real file -> if it doesn't match, hand back the exact code and retry. The model's #1 failure
(inexact SEARCH) is exactly what a tiny feedback step repairs. No tests, no Docker -- pure
file-matching. Measures applyable@1 vs applyable@k (does iteration lift the applyable rate).

TIER 2 (test-pass) is pluggable: once a SEARCH-matching patch exists, apply_sr + run the tests
(swe_verify / sb-cli) and feed the failure back. Stubbed here behind --verify.

This is V4's reasoning_loop shape (retrieve -> act -> verify -> retry), code action-set.

  V5_LM_TRUST_REMOTE_CODE=1 python -m v5.runtime.verifier_retry \
    --adapter-ckpt artifacts/stage_cache/adapter_code_s3.pt --n-eval 15 --max-retries 3 \
    --dump artifacts/retry_dump.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

# node types that route to the L8 PLANNING pool (subgraph.PLANNING_NODE_TYPES)
_PLANNING_TYPES = {"strategy", "reasoning_atom", "reasoning_chain", "failure_pattern", "control_rule"}


def _unit(v):
    a = np.asarray(v, dtype=np.float32)
    return a / (np.linalg.norm(a) + 1e-9)


def _recall(ret_meta, gold_meta):
    """Fraction of gold-touched symbols a retrieved symbol overlaps (same file + lineno span)."""
    if not gold_meta:
        return 1.0
    found = 0
    for g in gold_meta.values():
        gf, (glo, ghi) = g["file"], (g["lineno"][0], g["lineno"][-1])
        hit = any(r["file"] == gf and not (r["lineno"][-1] < glo or r["lineno"][0] > ghi)
                  for r in ret_meta.values())
        found += 1 if hit else 0
    return found / len(gold_meta)


def load_strategy_meta(paths):
    """{instance_id: [(node_id, text, node_type)]} for planning-pool strategy nodes."""
    out = {}
    for p in (paths or []):
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            e = r.get("raw_edit") or r
            if e.get("op") != "add_node" or e.get("node_type") not in _PLANNING_TYPES:
                continue
            iid = (e.get("metadata") or {}).get("instance_id")
            if iid:
                out.setdefault(iid, []).append((e["node_id"], e.get("text", "") or "", e["node_type"]))
    return out

from v5.adapter import GraphAttentionInjector
from v5.cross_attention import V5AttentionAdapter
from v5.gnn_encoder import RGCNEncoder
from v5.goal_encoder import GoalEncoder
from v5.training.providers import RealEmbedder, FrozenQwenHInitProvider
from v5.training.stage4_generate import _gen, _stub_graph
from v5.graph_grower.swe_load import load_instances, patch_files, checkout_repo
from v5.graph_grower.swe_probe import load_traces, _symbol_name
from v5.runtime.search_replace import SR_SYS, parse_sr, apply_sr
from v5.runtime.sr_withcode import load_symbol_meta, read_body, _file_text
from v5.graph_grower.swe_verify import write_predictions

import subprocess


def _user(issue, src_ctx, feedback="", exemplar_diff=""):
    s = f"ISSUE:\n{issue[:1400]}\n\n"
    if exemplar_diff:                        # a similar PAST fix to adapt (learning loop)
        s += f"A SIMILAR resolved bug was fixed like this — ADAPT the pattern:\n{exemplar_diff[:1200]}\n\n"
    if src_ctx:                              # empty under --no-graph (cold baseline)
        s += f"RELEVANT SOURCE (the bug is in here):\n{src_ctx}\n\n"
    if feedback:
        s += f"PREVIOUS ATTEMPT FAILED:\n{feedback}\n\n"
    return (s + "Find the exact line(s) causing the bug and fix them. Output ONLY search/replace "
            "blocks: SEARCH must copy the source EXACTLY (character-for-character); REPLACE must "
            "DIFFER from SEARCH. Keep it minimal.")


def _unmatched(blocks, dest):
    out = []
    for b in blocks:
        s = (b.get("search") or "").strip()
        if s and s not in _file_text(dest, b.get("file")):
            out.append(b)
    return out


def solve(model, tok, injector, issue, src_ctx, dest, max_retries, max_new, inject_on=True,
          exemplar_diff=""):
    """Iterate generate -> check SEARCH matches file -> feedback -> retry. Returns
    (applyable, attempts_used, blocks, history). inject_on=False = cold baseline (--no-graph).
    exemplar_diff = a similar past fix to adapt (learning loop)."""
    feedback = ""
    history = []
    blocks = []
    for attempt in range(max_retries):
        gen = _gen(model, tok, [{"role": "system", "content": SR_SYS},
                                {"role": "user", "content": _user(issue, src_ctx, feedback, exemplar_diff)}],
                   model.device if hasattr(model, "device") else next(model.parameters()).device,
                   injector, inject_on, max_new)
        blocks = parse_sr(gen)
        um = _unmatched(blocks, dest)
        applyable = bool(blocks) and not um
        history.append({"attempt": attempt + 1, "n_blocks": len(blocks), "applyable": applyable})
        if applyable:
            return True, attempt + 1, blocks, history
        # apply-feedback: tell it exactly what failed
        if not blocks:
            feedback = "You produced NO valid search/replace block. Output ONLY blocks in the exact format."
        else:
            b = um[0]
            feedback = (f"Your SEARCH for file '{b['file']}' was NOT found in the source. This text "
                        f"does not exist there:\n{(b.get('search') or '')[:200]}\n"
                        "Copy the lines from the SOURCE above EXACTLY (same indentation/spacing).")
    return False, max_retries, blocks, history


def solve_vote(model, tok, injector, issue, src_ctx, dest, k, temp, max_new):
    """SELF-CONSISTENCY: sample K patches from the SAME prompt (no perturbation), keep the
    applyable ones, VOTE on the canonical edit (file+search+replace block-set). Errors scatter,
    the right minimal edit repeats -> the mode beats greedy. Returns (blocks|None, votes, n_app)."""
    from collections import Counter
    ballots = {}
    counts = Counter()
    n_app = 0
    msgs_user = _user(issue, src_ctx)
    for _ in range(k):
        gen = _gen(model, tok, [{"role": "system", "content": SR_SYS},
                                {"role": "user", "content": msgs_user}],
                   next(model.parameters()).device, injector, True, max_new, temperature=temp)
        blocks = parse_sr(gen)
        if not blocks or _unmatched(blocks, dest):
            continue
        n_app += 1
        key = tuple(sorted((b.get("file", ""), (b.get("search") or "").strip(),
                            (b.get("replace") or "").strip()) for b in blocks))
        ballots[key] = blocks
        counts[key] += 1
    if not counts:
        return None, 0, 0, []
    ranked = [ballots[key] for key, _ in counts.most_common()]   # distinct candidates, most-voted first
    best, votes = counts.most_common(1)[0]
    return ballots[best], votes, n_app, ranked


def run(model_name, traces_p, nodes_p, adapter_ckpt, dataset, split, n_eval, max_new,
        repo_root, max_retries, dump="", emit_predictions="", no_graph=False,
        strategy="off", strategy_nodes=None, strat_topk=3,
        src_bodies=6, src_lines=70, best_of_k=0, temp=0.7, emit_candidates=0,
        retrieve=0, retriever_ckpt="", retriever_st="", retr_max_files=40, retr_max_syms=1500,
        device_str=None):
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device}  max_retries={max_retries}", flush=True)
    provider = FrozenQwenHInitProvider(model_name, device=device)
    model, tok = provider.model, provider.tok
    lm_dim = provider.hidden_size
    embedder = RealEmbedder(device)
    gnn = RGCNEncoder().to(device).eval()
    goal_enc = GoalEncoder().to(device).eval()
    adapter = V5AttentionAdapter(r_plan=3, r_evidence=4, lm_hidden_dim=lm_dim).to(device)
    adapter.load_state_dict(torch.load(adapter_ckpt, map_location=device))
    adapter.eval()
    injector = GraphAttentionInjector(adapter, gnn, goal_enc, device=device)

    traces = load_traces(traces_p)
    meta = load_symbol_meta(nodes_p)
    insts = {t["instance_id"]: t for t in load_instances(dataset, split, limit=0)}
    ids = [i for i in traces if i in insts][:n_eval]
    strat_meta = load_strategy_meta(strategy_nodes) if strategy != "off" else {}
    strat_vecs, strat_syms = {}, {}
    if strategy == "retrieved":
        for siid in strat_meta:
            if siid in traces:
                strat_vecs[siid] = _unit(embedder.embed_nodes({"q": traces[siid]["issue"]})["q"])
                strat_syms[siid] = {n for s in (traces[siid].get("support_ids") or [])
                                    if s in meta and (n := _symbol_name(meta[s]["text"]))}
    print(f"eval instances={len(ids)} | symbol meta={len(meta)} | strategy={strategy} "
          f"(instances={len(strat_meta)})", flush=True)

    _retr_delta = _retr_st = None
    if retrieve > 0 and retriever_st:        # the DESIGN's trained bi-encoder (preferred)
        from v5.runtime.code_retrieve import load_st_ranker
        _retr_st = load_st_ranker(retriever_st)
        print(f"loaded TRAINED bi-encoder ranker {retriever_st}", flush=True)
    elif retrieve > 0 and retriever_ckpt:    # fallback: zero-init delta on the frozen embedder
        from v5.runtime.code_retrieve import load_delta
        dim = len(embedder.embed_nodes({"q": "x"})["q"])
        _retr_delta = load_delta(retriever_ckpt, dim, device)
        print(f"loaded delta retriever {retriever_ckpt} (dim {dim})", flush=True)

    app1 = appk = scored = 0
    records = []
    recalls = []                            # retrieve mode: SYMBOL recall@K vs gold-touched symbols
    file_recalls = []                       # STAGE-1: gold FILE in the lexical pool? (splits the wall)
    preds = {}                              # instance_id -> git diff (swebench prediction)
    cand_preds = {}                         # rank -> {instance_id: patch} for --emit-candidates (oracle best-of-K)
    tf = {"task_family": "code_fix", "required_slots": []}
    for k, iid in enumerate(ids):
        t = traces[iid]; inst = insts[iid]
        # per-REPO dest (clone once, fetch+checkout each commit) -> NOT a fresh clone per commit.
        dest = Path(repo_root) / inst["repo"].replace("/", "__")
        ok, msg = checkout_repo(inst["repo"], inst["base_commit"], dest, timeout=1800)
        if not ok:
            print(f"  [{k+1}] {iid} checkout FAILED"); continue
        if retrieve > 0:                     # REAL localization: rank repo symbols by the ISSUE (no gold)
            from v5.runtime.code_retrieve import retrieve_support, to_meta, _rank_files
            gold_m = {s: meta[s] for s in t["support_ids"] if s in meta}
            gold_files = {g["file"] for g in gold_m.values()}
            s1 = set(_rank_files(str(dest), t["issue"], retr_max_files))   # STAGE-1 diagnostic
            file_recalls.append(1.0 if (gold_files & s1) else 0.0)
            inst_meta = to_meta(retrieve_support(str(dest), t["issue"], embedder, retrieve,
                                                 max_files=retr_max_files, max_syms=retr_max_syms,
                                                 delta=_retr_delta, st_model=_retr_st))
            support = list(inst_meta)
            recalls.append(_recall(inst_meta, gold_m))
        else:                                # ORACLE support (gold-AST-mapped) — the default
            inst_meta = meta
            support = [s for s in t["support_ids"] if s in meta]
        if not support:
            continue
        src_parts = []
        if not no_graph:                     # cold baseline (--no-graph): no source, no injection
            for s in support[:src_bodies]:
                body = read_body(str(dest), inst_meta[s]["file"], inst_meta[s]["lineno"], max_lines=src_lines)
                if body:
                    src_parts.append(f"# {inst_meta[s]['file']}\n{body}")
        src_ctx = "\n\n".join(src_parts)
        sym_ids = support[:24]
        id2text = {s: inst_meta[s]["text"] for s in sym_ids}
        ntypes = {s: "fact" for s in sym_ids}          # symbols -> L20 EVIDENCE
        # strategy nodes -> L8 PLANNING (off / own=upper-bound leakage / retrieved=real, exclude self)
        strat_pick = []
        if strategy == "own":
            strat_pick = strat_meta.get(iid, [])[:4]
        elif strategy == "retrieved" and strat_vecs:
            qv = _unit(embedder.embed_nodes({"q": t["issue"]})["q"])
            tsyms = {n for s in support if (n := _symbol_name(inst_meta[s]["text"]))}

            def _score(s):                       # issue similarity + symbol-overlap (same buggy fns -> shared HOW)
                ov = len(tsyms & strat_syms.get(s, set())) / (len(tsyms | strat_syms.get(s, set())) + 1e-9)
                return float(np.dot(qv, strat_vecs[s])) + 0.5 * ov

            top = sorted((s for s in strat_vecs if s != iid), key=_score, reverse=True)[:strat_topk]
            pool = []
            for s in top:
                pool.extend(strat_meta[s])
            strat_pick = pool[:6]
        for sid, stext, stype in strat_pick:
            id2text[sid] = stext; ntypes[sid] = stype
        all_ids = list(id2text.keys())
        text_emb = embedder.embed_nodes(id2text)
        # r_plan = loop DEPTH (match training=3); the pooled strategy nodes are all attended in
        # one pass, so more nodes cost ~nothing — don't raise r_plan (slower + train mismatch).
        injector.prepare_session(_stub_graph(all_ids, id2text, ntypes),
                                 all_ids, text_emb, tf, r_plan=3, r_evidence=4)

        if best_of_k > 1:
            blocks, votes, n_app, ranked = solve_vote(model, tok, injector, t["issue"], src_ctx,
                                                      str(dest), best_of_k, temp, max_new)
            applyable = blocks is not None
            attempts = best_of_k
            hist = [{"attempt": 1, "n_blocks": len(blocks or []), "applyable": applyable,
                     "votes": votes, "applyable_samples": n_app, "n_distinct": len(ranked)}]
            if emit_candidates:                  # ORACLE best-of-K: every distinct candidate -> its own file
                for r, cb in enumerate(ranked[:emit_candidates]):
                    _, cp = apply_sr(str(dest), cb)
                    subprocess.run(["git", "-C", str(dest), "checkout", "--", "."], capture_output=True)
                    if cp.strip():
                        cand_preds.setdefault(r, {})[iid] = cp
        else:
            applyable, attempts, blocks, hist = solve(model, tok, injector, t["issue"], src_ctx,
                                                      str(dest), max_retries, max_new, inject_on=not no_graph)
        scored += 1
        app1 += 1 if (hist and hist[0]["applyable"]) else 0
        appk += 1 if applyable else 0
        # turn the applyable SR blocks into a real git diff (the swebench prediction), then
        # restore the checkout so it's clean for the next instance.
        model_patch = ""
        if applyable:
            _, model_patch = apply_sr(str(dest), blocks)
            subprocess.run(["git", "-C", str(dest), "checkout", "--", "."], capture_output=True)
        preds[iid] = model_patch
        records.append({"id": iid, "applyable": applyable, "attempts": attempts, "history": hist,
                        "gold_patch": inst.get("patch", ""), "final_blocks": blocks})
        if best_of_k > 1:
            print(f"  [{k+1}/{len(ids)}] {iid:24} applyable={applyable} "
                  f"votes={hist[0].get('votes', 0)}/{hist[0].get('applyable_samples', 0)} of {best_of_k}", flush=True)
        else:
            print(f"  [{k+1}/{len(ids)}] {iid:24} applyable={applyable} in {attempts} "
                  f"(attempt1={'Y' if hist and hist[0]['applyable'] else 'N'})", flush=True)

    if dump:
        Path(dump).parent.mkdir(parents=True, exist_ok=True)
        with open(dump, "w", encoding="utf-8") as w:
            for r in records:
                w.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\ndumped {len(records)} -> {dump}", flush=True)

    if emit_predictions:
        n = write_predictions({i: p for i, p in preds.items() if p.strip()}, emit_predictions, "v5_retry")
        print(f"emitted {n} applyable predictions -> {emit_predictions}  (TIER-2: verify with\n"
              f"  python -m v5.graph_grower.swe_verify --predictions {emit_predictions} --dataset {dataset})", flush=True)

    if emit_candidates and cand_preds:
        base = Path(emit_predictions or "artifacts/cand").with_suffix("")
        for r in range(emit_candidates):
            cp = cand_preds.get(r, {})
            if cp:
                path = f"{base}_cand{r+1}.jsonl"
                write_predictions(cp, path, "v5_cand")
                print(f"  candidate-{r+1}: {len(cp)} patches -> {path}", flush=True)
        print("\nORACLE best-of-K: verify EACH cand*.jsonl, UNION the resolved instance_ids. If the"
              "\nunion >> the mode's resolves, the TRUTH is in the candidate set -> verifier-selection"
              "\n(test candidates, keep any pass) is the lever. If ~equal, the model can't synthesize it.", flush=True)

    print(f"\n=== RETRY LOOP (apply-feedback, no verifier) ===")
    print(f"  applyable@1   {app1}/{scored} ({100*app1/max(1,scored):.0f}%)")
    print(f"  applyable@{max_retries}   {appk}/{scored} ({100*appk/max(1,scored):.0f}%)")
    if retrieve > 0 and recalls:
        full = sum(1 for r in recalls if r >= 0.999)
        fr = sum(file_recalls) / max(1, len(file_recalls))
        print(f"  STAGE-1 FILE recall (gold file in lexical pool): {fr:.2f}  <- the ceiling")
        print(f"  STAGE-2 SYMBOL recall@{retrieve}: mean {sum(recalls)/len(recalls):.2f} | all-gold {full}/{len(recalls)}")
        print(f"  -> if file recall >> symbol recall: the RANKER/representation is the wall;"
              f"\n     if file recall ~ symbol recall (~0.34): the lexical FILE-finder is the wall (parse tracebacks).")
    print("\napplyable@k > applyable@1 = iteration repairs unmatched SEARCH -> the loop lifts the"
          "\napplyable rate with zero infra. Tier-2 (run tests) plugs in when a verifier is available.")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Verifier-retry loop tier-1 (apply-feedback).")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--traces", nargs="+", default=["data/swe/grounded_traces.jsonl"])
    ap.add_argument("--nodes", nargs="+", default=["artifacts/graph_growth/swe_code_candidates.jsonl"])
    ap.add_argument("--adapter-ckpt", default="artifacts/stage_cache/adapter_code_s3.pt")
    ap.add_argument("--dataset", default="lite")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-eval", type=int, default=15)
    ap.add_argument("--max-new", type=int, default=500)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--repo-root", default="data/swe_repos")
    ap.add_argument("--dump", default="")
    ap.add_argument("--emit-predictions", default="",
                    help="write the loop's applyable patches as a swebench predictions jsonl -> tier-2 verify")
    ap.add_argument("--no-graph", action="store_true",
                    help="COLD baseline: no source-read, no injection -> ablation for the grounding lift")
    ap.add_argument("--strategy", choices=["off", "own", "retrieved"], default="off",
                    help="strategy@L8 planning injection: off=baseline, own=upper-bound (leakage), "
                         "retrieved=nearest other task's strategy (real, exclude self)")
    ap.add_argument("--strategy-nodes", nargs="+",
                    default=["artifacts/graph_growth/swe_strategy_candidates_clean.jsonl"])
    ap.add_argument("--strat-topk", type=int, default=3,
                    help="retrieved: pool strategies from top-K nearest tasks (issue+symbol-overlap)")
    ap.add_argument("--src-bodies", type=int, default=6, help="read-source: top-K support bodies (MATCH training)")
    ap.add_argument("--src-lines", type=int, default=70, help="read-source: max lines/body (MATCH training)")
    ap.add_argument("--best-of-k", type=int, default=0,
                    help=">1 = SELF-CONSISTENCY: sample K patches from the same prompt, vote on the modal edit")
    ap.add_argument("--temp", type=float, default=0.7, help="sampling temperature for --best-of-k")
    ap.add_argument("--emit-candidates", type=int, default=0,
                    help="with --best-of-k: write top-N distinct candidates to <preds>_candR.jsonl (ORACLE best-of-K)")
    ap.add_argument("--retrieve", type=int, default=0,
                    help=">0 = REAL localization: rank repo symbols by the issue, top-K as support (NOT gold)")
    ap.add_argument("--retriever-ckpt", default="",
                    help="zero-init delta ckpt (fallback retriever on the frozen embedder)")
    ap.add_argument("--retriever-st", default="",
                    help="trained sentence-transformers bi-encoder (models/ranker-code) — the DESIGN ranker")
    ap.add_argument("--retr-max-files", type=int, default=40, help="STAGE-1 file-filter: top-K files by issue keywords")
    ap.add_argument("--retr-max-syms", type=int, default=1500, help="STAGE-2: max symbols to embed+rank")
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    run(a.model, a.traces, a.nodes, a.adapter_ckpt, a.dataset, a.split, a.n_eval, a.max_new,
        a.repo_root, a.max_retries, dump=a.dump, emit_predictions=a.emit_predictions,
        no_graph=a.no_graph, strategy=a.strategy, strategy_nodes=a.strategy_nodes,
        strat_topk=a.strat_topk, src_bodies=a.src_bodies, src_lines=a.src_lines,
        best_of_k=a.best_of_k, temp=a.temp, emit_candidates=a.emit_candidates,
        retrieve=a.retrieve, retriever_ckpt=a.retriever_ckpt, retriever_st=a.retriever_st,
        retr_max_files=a.retr_max_files, retr_max_syms=a.retr_max_syms, device_str=a.device)


if __name__ == "__main__":
    raise SystemExit(main())
