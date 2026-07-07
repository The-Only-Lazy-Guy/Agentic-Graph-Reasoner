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


def train_lora(model_name: str, out_dir: str = LORA_DIR, epochs: int = 2,
               batch_size: int = 8, max_tokens: int = 1600, fable5_triples: str = TRIPLES,
               log=print) -> None:
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
    pairs += why_p
    log(f"  [lora] +{len(why_p)} Call-A why-pairs from Fable-5 ({fable5_triples})")
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
        query_fn = None
        if query_mode == "refiner" and ranker is not None:
            from v5.runtime.memory_refiner import make_query_fn   # deferred: mirrors the
            net, ops, K_r = ranker                                 # TotalMemory import below;
            query_fn = make_query_fn(net, ops, embed_fn, K_r)       # avoids a module cycle
        from v5.memory.memory import TotalMemory
        if chain_dir.exists():
            shutil.rmtree(chain_dir)
        # query_fn only passed when set (Stage 2/query_mode="refiner"): TotalMemory.__init__
        # doesn't accept it yet (that's task #19, gated on Stage 1's boundary checklist) --
        # Stage 1 (spec/why) must call TotalMemory exactly as it does today.
        mem_kwargs = {"query_fn": query_fn} if query_fn is not None else {}
        memory = TotalMemory(chain_dir / "mem", mode="concept", embed_fn=embed_fn, **mem_kwargs)
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
        if arm == "memory" and memory is not None:
            hit = memory.read(goal=goal_for_query, span=session_data(s["spec"], current),
                              file_path=target)
            payload, mem_tok = hit.trace_text, hit.tokens_est
        elif arm == "ceiling":
            others = {f: b for f, b in repo.items() if f != target}
            payload = repo_dump(others)
            mem_tok = len(payload) // 4
        else:
            payload, mem_tok = "", 0
        obs = ""
        passed, gen, attempts, tok = False, "", 0, 0
        for _ in range(budget):
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
        rows.append({"iid": inst["instance_id"], "sid": s["sid"], "kind": s["kind"],
                     "depth": s["depth"], "dependency": bool(s.get("withheld")),
                     "passed": passed, "attempts": attempts, "prompt_tokens": tok,
                     "mem_tokens": mem_tok, "why_tokens": why_tok,
                     "why_text": why_text[:200] if why_text else ""})
        if memory is not None:
            memory.write(goal=s["spec"], old=current, new=gen,
                         trace=s["spec"][:400], verified=passed,
                         file_path=target, task_id=s["sid"])
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


def run_arm(lm, insts: list[dict], arm: str, budget: int, max_new: int, heal: bool,
            embed_fn=None, query_mode: str = "spec", ranker=None, log=print) -> dict:
    t0 = time.time()
    rows = []
    for i, inst in enumerate(insts):
        rows.extend(run_chain(lm, inst, arm, budget=budget, max_new=max_new, heal=heal,
                              embed_fn=embed_fn, query_mode=query_mode, ranker=ranker, log=log))
        done = [r for r in rows if r["passed"]]
        log(f"    [{arm}] chain {i+1}/{len(insts)} ({inst['instance_id']}): "
            f"cum {len(done)}/{len(rows)}")
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
    ap.add_argument("--epochs", type=int, default=2)
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
        train_lora(a.model, out_dir=a.lora, epochs=a.epochs, batch_size=a.batch_size)
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
                        embed_fn=embed if arm == "memory" else None, query_mode=qm, log=print)
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
                      ranker=ranker, log=print)
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
