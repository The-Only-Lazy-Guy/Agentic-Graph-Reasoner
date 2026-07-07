"""v3-B PROJECT LOOP — session chains on an agent-owned repo, memory across sessions.

Per instance (a project chain from project_gen): the agent works the ordered sessions;
its OWN repo state persists between them; after each session the outcome is written to a
PER-CHAIN TotalMemory (L1 episodic) and the repo is scanned into L0 (symbols carry the
conventions: inline format strings, seeded names). Dependency sessions withhold those
conventions from the spec — the arms differ ONLY in what fills the slot:

  off      spec + current target file            (the stateless agent)
  memory   + TotalMemory payload (L0 symbols + L1 impl, relevance-gated)
  ceiling  + the WHOLE repo dumped in the prompt (what memory approximates)

Gates:
  GB1  memory > off on DEPENDENCY sessions (off lacks the information by construction)
  GB2  memory reaches >= 90% of ceiling's dependency solve-rate at a fraction of its
       payload tokens (the scale/speed claim: repos won't fit in prompts; memory must)

Chain healing (default ON): after a failed session the repo file is restored to gold so
later sessions are measured from a sane prefix (memory still stores the agent's real
attempt). --no-heal = fully agent-owned state (deployment realism, entangled metrics).
DEBUG sessions overwrite the target with the generator's buggy variant (canonicalizes
that file for the session; noted limitation).

  python -m v5.runtime.project_loop --selftest              # no model
  python -m v5.runtime.project_loop --smoke                 # 0.5B, 2 chains, local
  python -m v5.runtime.project_loop --train-lora            # gold-chain proposer (molab)
  python -m v5.runtime.project_loop --run --arm off|memory|ceiling
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from v5.runtime.lggn_realizer import (SEP_N, SEP_T, SEP_W, TRIPLES, RawLM, load_triples,
                                      why_pairs, why_prompt)
from v5.runtime.project_gen import gold_state_after, make_split
from v5.runtime.sandbox import obs_text, run_project

LORA_DIR = "artifacts/project_lora"
RESULTS_PATH = "artifacts/project_results.json"
CHAINS_ROOT = "data/memory_chains"
PAYLOAD_CAP = 1400
CEILING_CAP = 4000
WHY_MAX_NEW = 64             # Call A completion budget — short "why" statement, not code

TRAIN_SEEDS = range(100, 130)
EVAL_SEEDS = range(0, 20)


def session_data(spec: str, current: str) -> str:
    return spec + "\n" + (current or "")


def build_prompt(spec: str, current: str, payload: str) -> str:
    return session_data(spec, current) + SEP_T + (payload or "")[:PAYLOAD_CAP] + SEP_N


def repo_dump(repo: dict[str, str], cap: int = CEILING_CAP) -> str:
    parts = [f"## {name}\n{body}" for name, body in sorted(repo.items())]
    return "\n".join(parts)[:cap]


def _save_results(update: dict, path: str = RESULTS_PATH) -> dict:
    p = Path(path)
    merged = {}
    if p.exists():
        try:
            merged = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            merged = {}
    merged.update(update)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    return merged


# ── proposer training (gold chains, all three slot distributions) ────────────────

def _memoryish_payload(inst: dict, upto: int) -> str:
    """What a good memory read WOULD deliver before session `upto`: the prior sessions'
    gold bodies (symbol-flavored, capped) — trains the model to exploit the slot."""
    parts = []
    for s in inst["sessions"][:upto]:
        for body in s["gold"].values():
            parts.append(body[:280])
    return "\n".join(parts[-4:])[:PAYLOAD_CAP]


def _oversample(pairs: list, factor: float, seed: int = 0) -> list:
    """Repeat `pairs` `factor` times (fractional part = a deterministic random sample of
    that fraction, no duplicates within the fractional slice). factor=1.0 -> pairs unchanged
    (new list, same contents)."""
    if factor <= 0:
        return []
    n_reps, frac = int(factor), factor - int(factor)
    out = list(pairs) * n_reps
    if frac > 0 and pairs:
        import random
        out += random.Random(seed).sample(pairs, min(len(pairs), round(len(pairs) * frac)))
    return out


def train_lora(model_name: str, out_dir: str = LORA_DIR, epochs: int = 2,
               batch_size: int = 8, max_tokens: int = 1600, fable5_triples: str = TRIPLES,
               why_oversample: float = 1.0, seed: int = 0, log=print) -> None:
    """why_oversample: repeat the Fable-5 why-pairs this many times (default 1.0 = off,
    unchanged behavior). Diagnosed need (2026-07-07 molab run): pooled loss plateaued at
    ~1.2 vs the code-only baseline's 0.041 at the same 2 epochs -- the repetitive synthetic
    code pairs converge almost immediately, so most of that 2-epoch budget's useful gradient
    was ALREADY going to the harder, more diverse real why-pairs; they just hadn't converged
    in 2 passes. This wasn't a mix-RATIO problem (why-pairs were already the majority by
    count, 936 vs 690) -- it's a converged-vs-not gap. Oversampling gives the still-learning
    class more full passes without proportionally slowing the already-converged one; combine
    with a higher --epochs on the actual retrain (already an exposed CLI flag)."""
    insts = make_split(seeds=TRAIN_SEEDS)
    pairs = []
    for inst in insts:
        repo: dict[str, str] = {}
        for k, s in enumerate(inst["sessions"]):
            current = s["buggy"][s["target_file"]] if s.get("buggy") else \
                repo.get(s["target_file"], "")
            gold = s["gold"][s["target_file"]]
            pairs.append((build_prompt(s["spec"], current, ""), gold))
            if k > 0:
                pairs.append((build_prompt(s["spec"], current,
                                           _memoryish_payload(inst, k)), gold))
                pairs.append((build_prompt(s["spec"], current,
                                           repo_dump(gold_state_after(inst, k - 1))), gold))
            repo.update(s["gold"])
    log(f"  [lora] {len(pairs)} code pairs from {len(insts)} gold chains")
    # v3 Stage 1: mix in Call-A (SEP_W) supervision from REAL Fable-5 (goal,old,trace) triples
    # -- one shared LoRA, same mechanism already used above (multiple slot distributions in
    # one pairs list) extended to a second job (query formation) via a distinct separator.
    triples = load_triples(fable5_triples, log=log)
    why_p = why_pairs(triples)
    oversampled = _oversample(why_p, why_oversample, seed=seed)
    pairs += oversampled
    log(f"  [lora] +{len(oversampled)} Call-A why-pairs from Fable-5 ({fable5_triples}), "
        f"oversample={why_oversample}x (base {len(why_p)})")
    lm = RawLM(model_name)
    lm.train_on(pairs, epochs=epochs, batch_size=batch_size, max_tokens=max_tokens, log=log)
    lm.save_checkpoint(out_dir)
    log(f"  [lora] checkpoint -> {out_dir}")
    lm.cleanup()


# ── chain runner ─────────────────────────────────────────────────────────────────

def run_chain(lm, inst: dict, arm: str, budget: int = 2, max_new: int = 512,
              heal: bool = True, chains_root: str = CHAINS_ROOT, embed_fn=None,
              query_mode: str = "spec", ranker=None, log=print) -> list[dict]:
    """query_mode (only matters when arm == "memory"):
      "spec"    (default, unchanged) — memory.read(goal=spec, ...), current GB1-validated path.
      "why"     — v3 Stage 1: Call A first (spec+current -> why_text via SEP_W), then
                  memory.read(goal=why_text, ...). why_text is captured/logged (the "model
                  explains its reasoning to the user" requirement) and costs one extra short
                  LM call per session.
      "refiner" — v3 Stage 2 (gated on Stage 1): same Call A, but query_fn (built from
                  `ranker`) overrides TotalMemory's flat-embed query with a K-step refined one.
    """
    memory = None
    chain_dir = Path(chains_root) / inst["instance_id"]
    if arm == "memory":
        from v5.memory.memory import TotalMemory
        if chain_dir.exists():
            shutil.rmtree(chain_dir)
        memory = TotalMemory(chain_dir / "mem", mode="concept", embed_fn=embed_fn)
        # query_fn is attached AFTER construction, not passed into __init__: make_query_fn
        # (memory_refiner.py, task #20) needs THIS chain's ImplStore to do its own pool
        # search before refining -- that store doesn't exist until TotalMemory is built.
        if query_mode == "refiner" and ranker is not None:
            from v5.runtime.memory_refiner import make_query_fn   # deferred: avoids a
            net, ops, K_r = ranker                                 # module cycle (memory_
            memory.query_fn = make_query_fn(net, ops, embed_fn,     # refiner imports nothing
                                            memory.impls, K=K_r)     # from here at module level)
    repo: dict[str, str] = {}
    rows = []
    for s in inst["sessions"]:
        target = s["target_file"]
        if s.get("buggy"):
            repo[target] = s["buggy"][target]
        current = repo.get(target, "")
        why_text, why_tok = "", 0
        if arm == "memory" and query_mode in ("why", "refiner"):
            wp = why_prompt(s["spec"], current)
            why_text = lm.generate_raw_batch([wp], max_new_tokens=WHY_MAX_NEW)[0].strip()
            why_tok = (len(wp) + len(why_text)) // 4
            if not why_text:                                  # defensive: degenerate generation
                why_text = s["spec"]
            log(f"    [why] {inst['instance_id']}/{s['sid']}: {why_text[:160]}")
        goal_for_query = why_text if why_text else s["spec"]
        if arm == "ceiling":
            others = {f: b for f, b in repo.items() if f != target}
            payload = repo_dump(others)
            mem_tok = len(payload) // 4
        elif arm != "memory":
            payload, mem_tok = "", 0
        # arm == "memory": payload is read PER ATTEMPT below (obs-informed retry, not a
        # single pre-loop read) -- TotalMemory.read() already threads `obs` into its query
        # embedding (memory.py: q = embed(goal + obs[-200:])), it was just never called with
        # real obs. Call A (why_text) stays a single generation per session -- re-running it
        # per retry would cost an extra LM call; the cheap win is re-QUERYING memory with the
        # obs signal, not re-authoring the intent.
        obs = ""
        passed, gen, attempts, tok = False, "", 0, 0
        delivered_task_id, delivered_task_ids = None, []
        for _ in range(budget):
            if arm == "memory" and memory is not None:
                hit = memory.read(goal=goal_for_query, span=session_data(s["spec"], current),
                                  obs=obs, file_path=target)
                payload, mem_tok = hit.trace_text, hit.tokens_est
                delivered_task_id = hit.impls[0].get("task_id") if hit.impls else None
                delivered_task_ids = [i.get("task_id") for i in hit.impls]
                if s.get("withheld"):     # dependency session -- which record actually landed?
                    delivered = hit.impls[0].get("file_path") if hit.impls else None
                    log(f"      [mem] {inst['instance_id']}/{s['sid']} kind={s['kind']} "
                        f"target={target!r} delivered={delivered!r} n_impls={len(hit.impls)}")
            slot = (payload + ("\n" + obs if obs else "")).strip()
            prompt = build_prompt(s["spec"], current, slot)
            gen = lm.generate_raw_batch([prompt], max_new_tokens=max_new)[0]
            attempts += 1
            tok += len(prompt) // 4
            res = run_project({**repo, target: gen}, s["tests"])
            passed = res["passed"]
            if passed:
                break
            obs = obs_text(res)
        src_idx = s.get("source_session_idx")
        source_sid = inst["sessions"][src_idx]["sid"] if src_idx is not None else None
        rows.append({"iid": inst["instance_id"], "sid": s["sid"], "kind": s["kind"],
                     "depth": s["depth"], "dependency": bool(s.get("withheld")),
                     "passed": passed, "attempts": attempts, "prompt_tokens": tok,
                     "mem_tokens": mem_tok, "why_tokens": why_tok,
                     "why_text": why_text[:200] if why_text else "",
                     "source_sid": source_sid, "delivered_task_id": delivered_task_id,
                     "delivered_task_ids": delivered_task_ids})
        if memory is not None:
            memory.write(goal=s["spec"], old=current, new=gen,
                         trace=s["spec"][:400], verified=passed,
                         file_path=target, task_id=s["sid"], kind=s["kind"])
        repo[target] = s["gold"][target] if (heal and not passed) else gen
        if not passed and not heal:
            pass                                            # broken state propagates (realism)
        if memory is not None:                              # L0 learns the repo truth
            rdir = chain_dir / "repo"
            rdir.mkdir(parents=True, exist_ok=True)
            for f, b in repo.items():
                (rdir / f).write_text(b, encoding="utf-8")
            memory.syntax.scan_files(str(rdir), list(repo.keys()), repo=inst["instance_id"])
    return rows


def _gen_chunked(lm, prompts: list[str], max_new_tokens: int, gen_batch: int) -> list[str]:
    """generate_raw_batch, chunked to bound peak VRAM (KV cache scales with batch x context) --
    same purpose as train_lora's batch_size, just at inference time."""
    if not prompts:
        return []
    if gen_batch <= 0 or len(prompts) <= gen_batch:
        return lm.generate_raw_batch(prompts, max_new_tokens=max_new_tokens)
    outs = []
    for i in range(0, len(prompts), gen_batch):
        outs.extend(lm.generate_raw_batch(prompts[i:i + gen_batch], max_new_tokens=max_new_tokens))
    return outs


def run_arm(lm, insts: list[dict], arm: str, budget: int, max_new: int, heal: bool,
            embed_fn=None, query_mode: str = "spec", ranker=None,
            chains_root: str = CHAINS_ROOT, gen_batch: int = 16, log=print) -> dict:
    """Batches generate_raw_batch calls ACROSS chains at each session depth, instead of
    running run_chain per instance (one prompt per generate call, 40x more forward passes
    than necessary). Chains are independent of each other -- only sessions WITHIN one chain
    are sequential (session N's repo/memory state depends on session N-1's outcome) -- so
    depth d's Call A / Call B / retry-attempt prompts across ALL still-active chains go into
    ONE generate_raw_batch call each (chunked by gen_batch to bound peak VRAM). Per-chain
    state lives in `states`; each chain's own session order and memory instance stay exactly
    as isolated as run_chain's, only the LM calls are batched. Same row schema/semantics as
    run_chain (verified: --selftest exercises both paths)."""
    t0 = time.time()
    make_qfn = None
    if arm == "memory" and query_mode == "refiner" and ranker is not None:
        from v5.runtime.memory_refiner import make_query_fn
        net, ops, K_r = ranker
        make_qfn = lambda impls: make_query_fn(net, ops, embed_fn, impls, K=K_r)
    memory_cls = None
    if arm == "memory":
        from v5.memory.memory import TotalMemory
        memory_cls = TotalMemory

    states = []
    for inst in insts:
        chain_dir = Path(chains_root) / inst["instance_id"]
        memory = None
        if memory_cls is not None:
            if chain_dir.exists():
                shutil.rmtree(chain_dir)
            memory = memory_cls(chain_dir / "mem", mode="concept", embed_fn=embed_fn)
            # query_fn is bound PER CHAIN, after construction: each chain has its own
            # isolated ImplStore (fresh rmtree'd above), and make_query_fn's pool search
            # needs THAT specific store, not one shared across all 40 chains.
            if make_qfn is not None:
                memory.query_fn = make_qfn(memory.impls)
        states.append({"inst": inst, "memory": memory, "repo": {}, "rows": [],
                       "chain_dir": chain_dir})

    max_depth = max((len(st["inst"]["sessions"]) for st in states), default=0)
    for depth in range(max_depth):
        active = [st for st in states if depth < len(st["inst"]["sessions"])]
        if not active:
            continue
        for st in active:
            s = st["inst"]["sessions"][depth]
            target = s["target_file"]
            if s.get("buggy"):
                st["repo"][target] = s["buggy"][target]
            st["s"], st["target"] = s, target
            st["current"] = st["repo"].get(target, "")

        # Call A: one batched why-text generation across every active chain at this depth
        if arm == "memory" and query_mode in ("why", "refiner"):
            wps = [why_prompt(st["s"]["spec"], st["current"]) for st in active]
            outs = _gen_chunked(lm, wps, WHY_MAX_NEW, gen_batch)
            for st, wp, out in zip(active, wps, outs):
                why_text = (out or "").strip() or st["s"]["spec"]
                st["why_text"], st["why_tok"] = why_text, (len(wp) + len(why_text)) // 4
                log(f"    [why] {st['inst']['instance_id']}/{st['s']['sid']}: {why_text[:160]}")
        else:
            for st in active:
                st["why_text"], st["why_tok"] = "", 0

        for st in active:
            s, target = st["s"], st["target"]
            st["goal_for_query"] = st["why_text"] or s["spec"]
            if arm == "ceiling":
                others = {f: b for f, b in st["repo"].items() if f != target}
                st["payload"] = repo_dump(others)
                st["mem_tok"] = len(st["payload"]) // 4
            elif arm != "memory":
                st["payload"], st["mem_tok"] = "", 0
            st["obs"], st["passed"], st["gen"] = "", False, ""
            st["attempts"], st["tok"], st["_done"] = 0, 0, False
            st["delivered_task_id"], st["delivered_task_ids"] = None, []

        for _attempt in range(budget):
            pending = [st for st in active if not st["_done"]]
            if not pending:
                break
            for st in pending:
                s, target = st["s"], st["target"]
                if arm == "memory" and st["memory"] is not None:
                    hit = st["memory"].read(goal=st["goal_for_query"],
                                            span=session_data(s["spec"], st["current"]),
                                            obs=st["obs"], file_path=target)
                    st["payload"], st["mem_tok"] = hit.trace_text, hit.tokens_est
                    st["delivered_task_id"] = hit.impls[0].get("task_id") if hit.impls else None
                    st["delivered_task_ids"] = [i.get("task_id") for i in hit.impls]
                    if s.get("withheld"):
                        delivered = hit.impls[0].get("file_path") if hit.impls else None
                        log(f"      [mem] {st['inst']['instance_id']}/{s['sid']} "
                            f"kind={s['kind']} target={target!r} delivered={delivered!r} "
                            f"n_impls={len(hit.impls)}")
                slot = (st["payload"] + ("\n" + st["obs"] if st["obs"] else "")).strip()
                st["prompt"] = build_prompt(s["spec"], st["current"], slot)

            prompts = [st["prompt"] for st in pending]
            outs = _gen_chunked(lm, prompts, max_new, gen_batch)
            for st, prompt, gen in zip(pending, prompts, outs):
                s, target = st["s"], st["target"]
                st["gen"] = gen
                st["attempts"] += 1
                st["tok"] += len(prompt) // 4
                res = run_project({**st["repo"], target: gen}, s["tests"])
                st["passed"] = res["passed"]
                if st["passed"]:
                    st["_done"] = True
                else:
                    st["obs"] = obs_text(res)

        for st in active:
            s, target = st["s"], st["target"]
            src_idx = s.get("source_session_idx")
            source_sid = st["inst"]["sessions"][src_idx]["sid"] if src_idx is not None else None
            st["rows"].append({"iid": st["inst"]["instance_id"], "sid": s["sid"],
                               "kind": s["kind"], "depth": s["depth"],
                               "dependency": bool(s.get("withheld")),
                               "passed": st["passed"], "attempts": st["attempts"],
                               "prompt_tokens": st["tok"], "mem_tokens": st["mem_tok"],
                               "why_tokens": st["why_tok"],
                               "why_text": st["why_text"][:200] if st["why_text"] else "",
                               "source_sid": source_sid,
                               "delivered_task_id": st.get("delivered_task_id"),
                               "delivered_task_ids": st.get("delivered_task_ids", [])})
            if st["memory"] is not None:
                st["memory"].write(goal=s["spec"], old=st["current"], new=st["gen"],
                                   trace=s["spec"][:400], verified=st["passed"],
                                   file_path=target, task_id=s["sid"], kind=s["kind"])
            st["repo"][target] = s["gold"][target] if (heal and not st["passed"]) else st["gen"]
            if st["memory"] is not None:                      # L0 learns the repo truth
                rdir = st["chain_dir"] / "repo"
                rdir.mkdir(parents=True, exist_ok=True)
                for f, b in st["repo"].items():
                    (rdir / f).write_text(b, encoding="utf-8")
                st["memory"].syntax.scan_files(str(rdir), list(st["repo"].keys()),
                                               repo=st["inst"]["instance_id"])
        done_ct = sum(1 for st in states for r in st["rows"] if r["passed"])
        total_ct = sum(len(st["rows"]) for st in states)
        log(f"    [{arm}] depth {depth+1}/{max_depth} ({len(active)} chains): "
            f"cum {done_ct}/{total_ct}")

    rows = []
    for st in states:
        rows.extend(st["rows"])
    dep = [r for r in rows if r["dependency"]]
    ind = [r for r in rows if not r["dependency"]]

    def rate(rs):
        return sum(r["passed"] for r in rs) / max(1, len(rs))

    return {"n": len(rows), "solved": sum(r["passed"] for r in rows),
            "solve_rate": rate(rows),
            "dep_rate": rate(dep), "dep_n": len(dep),
            "indep_rate": rate(ind), "indep_n": len(ind),
            "by_kind": {k: [sum(1 for r in rows if r["kind"] == k and r["passed"]),
                            sum(1 for r in rows if r["kind"] == k)]
                        for k in ("create", "cross", "debug", "extend")},
            "mean_mem_tokens": sum(r["mem_tokens"] for r in rows) / max(1, len(rows)),
            "mean_prompt_tokens": sum(r["prompt_tokens"] for r in rows) / max(1, len(rows)),
            "mean_why_tokens": sum(r.get("why_tokens", 0) for r in rows) / max(1, len(rows)),
            "wall_s": round(time.time() - t0, 1), "rows": rows}


def _report(results: dict, log=print) -> None:
    log("\n=== PROJECT LOOP (repo-continuity; slot content is the only variable) ===")
    for key in sorted(k for k in results if isinstance(results[k], dict) and "solve_rate" in results[k]):
        r = results[key]
        log(f"  {key:12} solve {r['solved']}/{r['n']} = {r['solve_rate']:.3f}  "
            f"DEP {r['dep_rate']:.3f} (n={r['dep_n']})  indep {r['indep_rate']:.3f}  "
            f"mem_tok {r['mean_mem_tokens']:.0f}  why_tok {r.get('mean_why_tokens', 0):.0f}  "
            f"wall {r['wall_s']}s  "
            + " ".join(f"{k}:{v[0]}/{v[1]}" for k, v in r["by_kind"].items()))
    off, mem, ceil = (results.get(a) for a in ("off", "memory", "ceiling"))
    if off and mem:
        d = mem["dep_rate"] - off["dep_rate"]
        log(f"\n  GB1 memory - off on DEPENDENCY sessions: {d:+.3f}  -> "
            f"{'PASS' if d >= 0.10 else 'FAIL'}")
    if mem and ceil:
        frac = mem["dep_rate"] / ceil["dep_rate"] if ceil["dep_rate"] > 0 else 0.0
        tok_ratio = (mem["mean_mem_tokens"] / ceil["mean_mem_tokens"]
                     if ceil["mean_mem_tokens"] > 0 else 0.0)
        log(f"  GB2 memory/ceiling DEP rate = {frac:.2f} at {tok_ratio:.2f}x ceiling tokens "
            f"-> {'PASS' if frac >= 0.90 and tok_ratio <= 0.6 else 'FAIL'}")
    _report_gb3(results, log)
    _report_gb4(results, log)


def _report_gb3(results: dict, log=print) -> None:
    """v3 Stage 1: does a self-authored (why_text) query beat the raw-spec query on the SAME
    dependency sessions? results["memory"] (query_mode="spec", the default/historical key —
    the already-validated GB1 result reused as baseline, no re-run needed) vs
    results["memory_why"] (query_mode="why") — both arm="memory", same EVAL_SEEDS chains."""
    spec, why = results.get("memory"), results.get("memory_why")
    if not (spec and why):
        return
    d = why["dep_rate"] - spec["dep_rate"]
    verdict = "PASS" if d >= 0.03 else ("NO-REGRESSION" if d >= -0.02 else "FAIL")
    log(f"  GB3 why-query - spec-query DEP rate: {d:+.3f}  -> {verdict}")


def _hit_rate_from_rows(rows: list[dict], rank1: bool = False) -> float:
    """Fraction of dependency rows where the correct source record (source_sid, task #18) was
    delivered -- did memory find the RIGHT record, not just solve the session (a session can
    pass without the right record, or fail with it).

    rank1=False (default): correct record ANYWHERE in what got delivered. k_impl=2 means a
    query's dominant topic can legitimately win rank-1 while the actually-needed record rides
    at rank-2 and still reaches the prompt -- a rank1-only check reads 0.000 even on a
    provably-correct, ceiling-solving run (confirmed: why-query's 40/40 extend run scored
    0.000 rank-1 hit-rate, because orders.py's own record -- not pricing.py, the true source
    -- consistently wins rank-1 there). rank1=True is the stricter, narrower question."""
    dep = [r for r in rows if r.get("source_sid") is not None]
    if not dep:
        return 0.0
    if rank1:
        return sum(1 for r in dep if r.get("delivered_task_id") == r["source_sid"]) / len(dep)
    return sum(1 for r in dep if r["source_sid"] in (r.get("delivered_task_ids") or [])) / len(dep)


def _report_gb4(results: dict, log=print) -> None:
    """v3 Stage 2: does the refiner-ranked query (query_mode="refiner") beat the flat why-
    query at finding the RIGHT source record (GB4a) and at solving dependency sessions
    (GB4b)? results["memory_why"] (baseline) vs results["memory_refiner"] — same EVAL_SEEDS
    chains as GB1/GB3."""
    why, ref = results.get("memory_why"), results.get("memory_refiner")
    if not (why and ref):
        return
    why_rows, ref_rows = why.get("rows", []), ref.get("rows", [])
    hit_why, hit_ref = _hit_rate_from_rows(why_rows), _hit_rate_from_rows(ref_rows)
    d_hit = hit_ref - hit_why
    log(f"  GB4a refiner hit-rate {hit_ref:.3f} vs why-query {hit_why:.3f}: {d_hit:+.3f}  -> "
        f"{'PASS' if d_hit >= 0.05 else 'FAIL'}")
    r1_why = _hit_rate_from_rows(why_rows, rank1=True)
    r1_ref = _hit_rate_from_rows(ref_rows, rank1=True)
    log(f"    (rank-1-only, stricter: refiner {r1_ref:.3f} vs why {r1_why:.3f})")
    d_dep = ref["dep_rate"] - why["dep_rate"]
    log(f"  GB4b refiner-query - why-query DEP rate: {d_dep:+.3f}  -> "
        f"{'PASS' if d_dep >= 0.03 else 'FAIL'}")


# ── selftest (no model) ─────────────────────────────────────────────────────────

def _shares_code(payload: str, earlier_src: str, min_len: int = 20, step: int = 5) -> bool:
    """Does payload contain a verbatim chunk of earlier source? (proxy for 'the convention
    is actually visible', independent of which literal test VALUE that convention produces —
    payloads carry code templates, not evaluated outputs)."""
    if not payload or not earlier_src:
        return False
    for i in range(0, max(1, len(earlier_src) - min_len), step):
        if earlier_src[i:i + min_len] in payload:
            return True
    return False


class _GoldLM:
    """Answers with the session's gold when the payload shares real code with the earlier
    sessions that established the withheld convention (simulating memory doing its job);
    otherwise emits a plausible-but-wrong guess."""

    def __init__(self, insts):
        self.entries = []                                  # (spec_prefix, inst, k)
        for inst in insts:
            for k, s in enumerate(inst["sessions"]):
                self.entries.append((s["spec"][:80], inst, k))

    def generate_raw_batch(self, prompts, max_new_tokens=0, **kw):
        outs = []
        for p in prompts:
            if SEP_T not in p and SEP_W in p:
                # Call A (why-prompt, ends in SEP_W, no SEP_T): echo a plausible why_text
                # stand-in (the spec's own head) — good enough for the plumbing this proves.
                data = p.split(SEP_W, 1)[0]
                hit = next(((inst, k) for pre, inst, k in self.entries if pre in data), None)
                outs.append(hit[0]["sessions"][hit[1]]["spec"][:80] if hit else "need context")
                continue
            data, slot = p.split(SEP_T, 1)
            hit = next(((inst, k) for pre, inst, k in self.entries if pre in data), None)
            if hit is None:
                outs.append("def broken(:")
                continue
            inst, k = hit
            s = inst["sessions"][k]
            gold = s["gold"][s["target_file"]]
            earlier_src = "".join(b for j in range(k) for b in inst["sessions"][j]["gold"].values())
            if not s.get("withheld") or _shares_code(slot, earlier_src):
                outs.append(gold)                          # convention visible -> solve
            else:
                import re
                names = re.findall(r"^def (\w+)\(([^)]*)\)", gold, re.M)
                outs.append("\n".join(f"def {n}({a}):\n    return None" for n, a in names))
        return outs


def _selftest() -> bool:
    import tempfile
    from v5.memory.store import make_fake_embedder
    from v5.runtime.project_gen import make_instance
    print("project_loop --selftest: chain plumbing, arms, healing, GB accounting (no model)\n")
    insts = [make_instance("inventory", 0), make_instance("logparse", 0)]
    lm = _GoldLM(insts)

    with tempfile.TemporaryDirectory() as td:
        r_off = run_arm(lm, insts, "off", budget=1, max_new=0, heal=True,
                        log=lambda *a: None)
        r_ceil = run_arm(lm, insts, "ceiling", budget=1, max_new=0, heal=True,
                         log=lambda *a: None)
        assert r_off["indep_rate"] == 1.0, r_off          # non-dependency solvable without info
        assert r_off["dep_rate"] < 0.5, "off must fail withheld sessions"
        assert r_ceil["dep_rate"] == 1.0, "ceiling (repo in prompt) carries the conventions"
        print(f"  [1] off dep={r_off['dep_rate']:.2f} vs ceiling dep=1.0 -> PASS")

        # memory arm with FAKE embedder: L0/L1 payload text still contains the gold bodies
        # (hash embeddings make retrieval arbitrary, but the file-mention boost fires on
        # spec text like 'inventory'), so at least some dependency sessions get the payload
        r_mem = run_arm(lm, insts, "memory", budget=1, max_new=0, heal=True,
                        embed_fn=make_fake_embedder(),
                        log=lambda *a: None)
        assert r_mem["dep_rate"] >= r_off["dep_rate"], \
            (r_mem["dep_rate"], r_off["dep_rate"])
        assert r_mem["mean_mem_tokens"] > 0, "memory arm delivered payloads"
        print(f"  [2] memory dep={r_mem['dep_rate']:.2f} >= off, mem_tok "
              f"{r_mem['mean_mem_tokens']:.0f} -> PASS")

        # gen_batch is a VRAM cap, not a semantic knob: forcing full chunk-serialization
        # (gen_batch=1, one prompt per generate call, same as pre-batching behavior) across
        # MULTIPLE real chains must reproduce gen_batch=0's (unbounded, all-active-in-one-call)
        # rows exactly -- proves cross-chain batching didn't change per-chain outcomes.
        r_mem_chunked = run_arm(lm, insts, "memory", budget=1, max_new=0, heal=True,
                                embed_fn=make_fake_embedder(), gen_batch=1,
                                log=lambda *a: None)
        key = lambda r: (r["iid"], r["sid"])
        assert sorted(r_mem["rows"], key=key) == sorted(r_mem_chunked["rows"], key=key), \
            "gen_batch=1 (chunked) must match gen_batch=0 (unbounded) row-for-row"
        print("  [2g] gen_batch is VRAM-cap only, chunked == unbounded -> PASS")

        rep = _save_results({"off": r_off, "memory": r_mem, "ceiling": r_ceil},
                            path=str(Path(td) / "r.json"))
        _report(rep, log=lambda *a: None)
        print("  [3] report + persistence -> PASS")

        # v3 Stage 1: query_mode="spec" (default) never invokes Call A; "why" does, and only
        # on arm="memory" rows (off/ceiling never build a memory query at all).
        r_off_spec = run_arm(lm, insts, "off", budget=1, max_new=0, heal=True,
                             query_mode="spec", log=lambda *a: None)
        assert all(r["why_tokens"] == 0 for r in r_off_spec["rows"]), "off never calls Call A"
        r_mem_spec = run_arm(lm, insts, "memory", budget=1, max_new=0, heal=True,
                             embed_fn=make_fake_embedder(), query_mode="spec",
                             log=lambda *a: None)
        assert all(r["why_tokens"] == 0 for r in r_mem_spec["rows"]), \
            "query_mode=spec: Call A skipped -> behavior-unchanged path"
        r_mem_why = run_arm(lm, insts, "memory", budget=1, max_new=0, heal=True,
                            embed_fn=make_fake_embedder(), query_mode="why",
                            log=lambda *a: None)
        assert all(r["why_tokens"] > 0 for r in r_mem_why["rows"]), \
            "query_mode=why: Call A runs on every memory-arm row"
        assert all(r["why_text"] for r in r_mem_why["rows"]), "why_text captured"
        print("  [4] query_mode spec (unchanged) vs why (Call A wired) -> PASS")

        rep2 = _save_results({"memory": r_mem_spec, "memory_why": r_mem_why},
                             path=str(Path(td) / "r2.json"))
        _report_gb3(rep2, log=lambda *a: None)                # must not crash; keys present
        _report_gb3({}, log=lambda *a: None)                  # must not crash; keys absent
        print("  [5] GB3 report -> PASS")

        # GB4 (Stage 2): source_sid (task #18's ground-truth label) must reach every
        # dependency row, hit-rate is computable, report doesn't crash on missing/present
        # keys. Real refiner-vs-why deltas only come from a real molab run (query_mode=
        # "refiner" needs a trained ranker); here just prove the plumbing, same technique as
        # GB3's smoke test above (comparing a result against itself -> deltas should be ~0).
        dep_rows = [r for r in r_mem_why["rows"] if r["dependency"]]
        assert dep_rows and all(r["source_sid"] is not None for r in dep_rows), \
            "every dependency row must carry a ground-truth source_sid (task #18)"
        hr = _hit_rate_from_rows(r_mem_why["rows"])
        assert 0.0 <= hr <= 1.0

        # rank1=False (default) vs rank1=True: real molab bug this reproduces -- k_impl=2 can
        # legitimately put the correct source record at rank 2 (the query's dominant topic
        # wins rank 1), still reaches the prompt, session still solves -- but a rank1-only
        # check reads that as a MISS. Confirmed on a real run: why-query's 40/40-solving
        # extend chains scored 0.000 rank-1 hit-rate for exactly this reason.
        fake_rows = [{"source_sid": "X", "delivered_task_id": "A",
                     "delivered_task_ids": ["A", "X"]}]      # correct record at rank 2
        assert _hit_rate_from_rows(fake_rows, rank1=False) == 1.0, \
            "membership check must count a rank-2 hit"
        assert _hit_rate_from_rows(fake_rows, rank1=True) == 0.0, \
            "rank1 check must NOT count a rank-2 hit -- this is the exact distinction"
        print("  [5c] hit-rate: membership (any rank) vs rank1-only give DIFFERENT answers -> PASS")
        rep3 = _save_results({"memory_why": r_mem_why, "memory_refiner": r_mem_why},
                             path=str(Path(td) / "r3.json"))
        _report_gb4(rep3, log=lambda *a: None)                # must not crash; keys present
        _report_gb4({}, log=lambda *a: None)                  # must not crash; keys absent
        print("  [5b] GB4 report + source_sid plumbing -> PASS")

        # _oversample: diagnosed fix for the ~1.2 loss plateau (harder Fable-5 why-pairs
        # under-converged relative to the near-deterministic synthetic code pairs)
        base = list(range(10))
        assert _oversample(base, 1.0) == base and _oversample(base, 1.0) is not base
        assert _oversample(base, 2.0) == base * 2
        assert _oversample(base, 0.0) == []
        one_five = _oversample(base, 1.5, seed=0)
        assert len(one_five) == 15 and one_five[:10] == base, "1.5x = full copy + half sample"
        assert _oversample(base, 1.5, seed=0) == _oversample(base, 1.5, seed=0), "deterministic"
        print("  [6] _oversample -> PASS")

        # _gen_chunked: cross-chain batching (run_arm's speedup) must be order-preserving and
        # content-identical to one big unchunked call -- chunking is a VRAM cap, not a
        # semantic change (greedy decoding: same prompt -> same output regardless of what
        # else shares its batch).
        class _EchoLM:
            def generate_raw_batch(self, prompts, max_new_tokens=0, **kw):
                return [f"out:{p}" for p in prompts]

        echo = _EchoLM()
        ps = [f"p{i}" for i in range(10)]
        whole = _gen_chunked(echo, ps, 8, gen_batch=0)
        chunked = _gen_chunked(echo, ps, 8, gen_batch=3)
        assert whole == chunked == [f"out:p{i}" for i in range(10)], (whole, chunked)
        assert _gen_chunked(echo, [], 8, gen_batch=3) == []
        print("  [6b] _gen_chunked matches unchunked, order-preserving -> PASS")

        # query_mode="refiner" wiring (task #21): each chain must get its OWN query_fn, bound
        # to that chain's isolated ImplStore -- not one shared closure built once outside the
        # per-chain loop (the bug this fix targeted: chains are rmtree'd/rebuilt individually,
        # a shared closure would silently bind to whichever chain constructed it last).
        from v5.runtime import memory_refiner as _mr
        seen_stores = []
        orig_make_query_fn = _mr.make_query_fn

        def _spy_make_query_fn(net, ops, embed_fn, impl_store, K=3, pool_k=16):
            seen_stores.append(impl_store)
            return orig_make_query_fn(net, ops, embed_fn, impl_store, K=K, pool_k=pool_k)

        _mr.make_query_fn = _spy_make_query_fn
        try:
            import numpy as np

            from v5.runtime.lggn_refine import Refiner
            net = Refiner.Net(768, r=32, n_op=4, max_K=4)
            ops = np.random.RandomState(0).randn(4, 768).astype("float32")
            r_refiner = run_arm(lm, insts, "memory", budget=1, max_new=0, heal=True,
                                embed_fn=make_fake_embedder(), query_mode="refiner",
                                ranker=(net, ops, 3), log=lambda *a: None)
            assert r_refiner["n"] > 0, "refiner arm produced no rows"
            assert len(seen_stores) == len(insts), \
                f"expected one query_fn built per chain ({len(insts)}), got {len(seen_stores)}"
            assert len(set(id(s) for s in seen_stores)) == len(insts), \
                "chains shared the SAME ImplStore -- per-chain binding regressed"
        finally:
            _mr.make_query_fn = orig_make_query_fn
        print("  [6c] query_mode=refiner: query_fn bound per-chain, not shared -> PASS")

        # obs-informed retry: memory.read() must be re-invoked EACH attempt with the growing
        # obs (not a single pre-loop read whose payload is frozen for the whole session) --
        # spy on TotalMemory.read to capture the per-call obs kwarg.
        from v5.memory.memory import TotalMemory
        calls: list[str] = []
        orig_read = TotalMemory.read

        def _spy_read(self, *a, **kw):
            calls.append(kw.get("obs", ""))
            return orig_read(self, *a, **kw)

        TotalMemory.read = _spy_read
        try:
            mini = {"instance_id": "spy_0", "seed": 0, "archetype": "spy", "sessions": [{
                "sid": "spy_0_s0", "kind": "create", "depth": 0, "target_file": "m.py",
                "spec": "spy retry test", "tests": ["import m\nassert m.f() == 1"], "setup": "",
                "gold": {"m.py": "def f(): return 1"}, "withheld": [],
            }]}

            class _FailOnceLM:
                def __init__(self):
                    self.n = 0

                def generate_raw_batch(self, prompts, max_new_tokens=0, **kw):
                    self.n += 1
                    return ["def f(): return 1"] if self.n > 1 else ["def f(): return 2"]

            rows = run_chain(_FailOnceLM(), mini, "memory", budget=2, max_new=0, heal=True,
                             embed_fn=make_fake_embedder(), log=lambda *a: None)
            assert rows[0]["passed"] and rows[0]["attempts"] == 2, rows[0]
            assert len(calls) == 2, f"expected 2 read() calls (one per attempt), got {len(calls)}"
            assert calls[0] == "", "attempt 1: no obs yet"
            assert calls[1] != "" and "assert" in calls[1].lower() or "error" in calls[1].lower() \
                or calls[1], f"attempt 2: obs from attempt 1's failure, got {calls[1]!r}"
        finally:
            TotalMemory.read = orig_read                        # always restore, even on failure
        print("  [7] obs-informed retry: memory re-queried per attempt -> PASS")
    print("\n  PROJECT_LOOP SELFTEST -> PASS")
    return True


# ── CLI ─────────────────────────────────────────────────────────────────────────

def main():
    import sys
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="v3-B project loop — repo-continuity chains.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="0.5B, 2 chains, arms off+memory")
    ap.add_argument("--train-lora", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--lora", default=LORA_DIR)
    ap.add_argument("--arm", choices=["off", "memory", "ceiling"], default="off")
    ap.add_argument("--n-chains", type=int, default=0, help="cap eval chains (0=all 40)")
    ap.add_argument("--budget", type=int, default=2)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--gen-batch-size", type=int, default=16,
                    help="max chains' prompts per generate_raw_batch call during --run/--smoke "
                         "(0=unbounded -- all active chains in one call). Chains are batched "
                         "ACROSS at each session depth; this caps peak VRAM, doesn't change "
                         "results (greedy decoding).")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--why-oversample", type=float, default=1.0,
                    help="repeat Fable-5 why-pairs Nx during --train-lora (see train_lora "
                         "docstring; diagnosed fix for the ~1.2 loss plateau)")
    ap.add_argument("--no-heal", action="store_true")
    ap.add_argument("--query-mode", choices=["spec", "why", "refiner"], default="spec",
                    help="v3 Stage 1/2: 'spec' = current GB1-validated path (unchanged), "
                         "'why' = self-authored query (Call A + SEP_W, GB3), "
                         "'refiner' = ranker-sourced query (Stage 2, GB4)")
    ap.add_argument("--ranker", default="", help="Stage 2: memory_refiner checkpoint dir "
                    "(required when --query-mode refiner)")
    ap.add_argument("--result-key", default="", help="override the results.json key "
                    "(default: arm, or f'{arm}_{query_mode}' when arm=memory)")
    a = ap.parse_args()

    if a.selftest:
        raise SystemExit(0 if _selftest() else 1)
    if a.train_lora:
        train_lora(a.model, out_dir=a.lora, epochs=a.epochs, batch_size=a.batch_size,
                  why_oversample=a.why_oversample)
        return
    if a.smoke:
        a.model = "Qwen/Qwen2.5-0.5B" if a.model == "Qwen/Qwen2.5-3B" else a.model
        insts = make_split(seeds=range(0, 1))              # 2 chains (one per archetype)
        print(f"[SMOKE] model={a.model} chains={len(insts)}")
        lm = RawLM(a.model)
        gold_pairs = []
        for inst in make_split(seeds=range(100, 103)):
            repo = {}
            for s in inst["sessions"]:
                cur = s["buggy"][s["target_file"]] if s.get("buggy") else repo.get(s["target_file"], "")
                gold_pairs.append((build_prompt(s["spec"], cur, ""), s["gold"][s["target_file"]]))
                repo.update(s["gold"])
        lm.train_on(gold_pairs, epochs=1, batch_size=2, max_tokens=1600, log=print)
        from v5.memory.store import make_mpnet_embedder
        embed = make_mpnet_embedder()
        for arm, qm in (("off", "spec"), ("memory", "spec"), ("memory", "why")):
            r = run_arm(lm, insts, arm, budget=1, max_new=384, heal=True,
                        embed_fn=embed if arm == "memory" else None, query_mode=qm,
                        gen_batch=a.gen_batch_size, log=print)
            print(f"  [smoke:{arm}/{qm}] {r['solved']}/{r['n']} dep={r['dep_rate']:.2f} "
                  f"mem_tok={r['mean_mem_tokens']:.0f} why_tok={r['mean_why_tokens']:.0f}")
        lm.cleanup()
        return
    if a.run:
        if a.query_mode == "refiner" and not a.ranker:
            raise SystemExit("--query-mode refiner needs --ranker <memory_refiner checkpoint dir>")
        insts = make_split(seeds=EVAL_SEEDS)
        if a.n_chains:
            insts = insts[:a.n_chains]
        print(f"[project-loop] model={a.model} arm={a.arm} query_mode={a.query_mode} "
              f"chains={len(insts)} budget={a.budget} heal={not a.no_heal}")
        lm = RawLM.load_checkpoint(a.model, a.lora)
        embed_fn = None
        if a.arm == "memory":
            from v5.memory.store import make_mpnet_embedder
            embed_fn = make_mpnet_embedder()
        ranker = None
        if a.query_mode == "refiner":
            from v5.runtime.memory_refiner import load_ranker
            ranker = load_ranker(a.ranker)
            print(f"  [ranker] loaded <- {a.ranker}")
        res = run_arm(lm, insts, a.arm, budget=a.budget, max_new=a.max_new,
                      heal=not a.no_heal, embed_fn=embed_fn, query_mode=a.query_mode,
                      ranker=ranker, gen_batch=a.gen_batch_size, log=print)
        res_slim = {k: v for k, v in res.items() if k != "rows"}
        # key "memory" (query_mode=spec, the default) stays literal — backward compatible with
        # the already-validated GB1/GB2 molab result, and lets GB3 reuse it as the baseline
        # without a re-run. Only "why"/"refiner" get namespaced (GB3/GB4 read these + "memory").
        key = a.result_key or (a.arm if a.query_mode == "spec" else f"{a.arm}_{a.query_mode}")
        merged = _save_results({key: res_slim})
        Path("artifacts/project_rows").mkdir(parents=True, exist_ok=True)
        with open(f"artifacts/project_rows/{key}.jsonl", "w", encoding="utf-8") as w:
            for r in res["rows"]:
                w.write(json.dumps(r) + "\n")
        _report(merged, log=print)
        lm.cleanup()
        return
    ap.print_help()


if __name__ == "__main__":
    main()
