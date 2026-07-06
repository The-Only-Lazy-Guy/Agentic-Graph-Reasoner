"""LGGN v2 M2 — the TRACER: graph latent h_K -> ~100-token reasoning trace -> frozen M1 realizer.

M1 PASSED all gates (trace 0.210 vs notrace 0.004 vs shuffled 0.003, 3 seeds): real traces carry
essentially ALL realization signal. M2 asks: can the GRAPH supply the trace? The refiner is
RETARGETED to trace-repr space (f = repr of the gold trace, not the gold fix), and a MoLoRA-
conditioned tracer LM decodes z into trace text. The tracer input is the old span ONLY — no goal
text — so z is the single channel carrying task intent: any latent-over-baseline delta is
attributable to z content.

Arms (fresh tracer each): baseline (span only, pure LoRA) / constant (mean h_K broadcast) /
latent (h_K per instance, z_dropout) / ceiling (z = gold trace repr).
End-to-end: generated trace -> frozen seed-matched M1 realizer -> added_recall vs gold new span.
Anchors: gold-trace-through-realizer (upper), M1 notrace (lower, from M1 results json).

Gates: G2 refiner cos(h_K, f_trace) >= 0.45 and >= raw cos(g,f)+0.05 (before tracer training) |
G3 ceiling-first: e2e ceiling-baseline >= +0.05 (1 seed, else the z channel is dead -> stop) |
G4 e2e latent-baseline >= +0.03 every seed, and trace-cos latent > constant.

  wiring  : python -m v5.runtime.lggn_tracer --selftest        # no GPU, no model
  smoke   : python -m v5.runtime.lggn_tracer --smoke           # 0.5B, real loop, local GPU
  G3 first: V5_LM_QUANT=4bit python -m v5.runtime.lggn_tracer --seed-list 0 --arms ceiling,baseline
  full    : V5_LM_QUANT=4bit python -m v5.runtime.lggn_tracer --seed-list 0   (then 1, then 2)
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from v5.runtime.lggn_realizer import (TRIPLES, RawLM, added_recall, load_triples,
                                      realizer_prompt, split_by_goal, tracer_prompt)

RESULTS_PATH = "artifacts/lggn_tracer_results.json"
M1_RESULTS_PATH = "artifacts/lggn_realizer_results.json"
ARM_ORDER = ("baseline", "constant", "latent", "ceiling")


def _trace_texts(triples: list[dict]) -> list[dict]:
    """Repr-extraction texts. The ONLY change vs the v1 pipeline: `added` = the GOLD TRACE
    (reasoning), not the fix text — h_K is trained to live in reasoning space."""
    return [{"issue": t["goal"][:700], "code": t["old"][:900], "added": t["trace"]}
            for t in triples]


def _cache_key(model_name: str, n: int, layer_frac: float, t_ctx: int) -> str:
    tail = model_name.split("/")[-1]
    return f"artifacts/lggn_reprs_fable5trace_{tail}_n{n}_L{layer_frac}_T{t_ctx}.npz"


def _arm_latents(arm: str, idx: list[int], h_all, f, h_mean):
    """Per-instance z for an arm, aligned to idx order. baseline -> None (pure LoRA, no hooks)."""
    if arm == "baseline":
        return None
    if arm == "constant":
        return [h_mean for _ in idx]
    if arm == "latent":
        return [h_all[i] for i in idx]
    if arm == "ceiling":
        return [f[i] for i in idx]
    raise ValueError(f"unknown arm {arm!r}")


def _save_results(results: dict, path: str = RESULTS_PATH) -> dict:
    """Merge-write after every arm (same walltime-kill protection as M1)."""
    p = Path(path)
    merged: dict = {}
    if p.exists():
        try:
            merged = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            merged = {}
    for s, arms_d in results.items():
        merged.setdefault(s, {}).update(arms_d)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    return merged


def _embed_texts(model_name: str, texts: list[str], layer_frac: float = 0.6):
    """Frozen-LM last-token reprs of generated traces (for the trace-cos diagnostic).
    One load, one pass, freed after."""
    import os
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from v5.lm_loader import load_frozen_lm
    trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
    lm = load_frozen_lm(model_name)
    lm.eval()
    dev = next(lm.parameters()).device
    L = max(1, int(layer_frac * lm.config.num_hidden_layers))
    out = []
    with torch.no_grad():
        for t in texts:
            enc = tok((t or " ")[:2000], return_tensors="pt", truncation=True,
                      max_length=256).to(dev)
            hs = lm(**enc, output_hidden_states=True).hidden_states[L][0]
            out.append(hs[-1].float().cpu().numpy())
    del lm, tok
    torch.cuda.empty_cache()
    return np.asarray(out)


def _cos_rows(a, b) -> float:
    import numpy as np
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return float((an * bn).sum(1).mean())


def _m1_notrace_anchor(seed: int) -> float | None:
    p = Path(M1_RESULTS_PATH)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d[str(seed)]["notrace"]["added_recall"]
    except Exception:
        return None


# ── the experiment ──────────────────────────────────────────────────────────────

def run_m2(model_name: str, triples: list[dict], seeds: list[int], arms: list[str],
           K: int, r: int, n_op: int, refiner_epochs: int, tracer_epochs: int,
           z_dropout: float, eval_n: int, held_frac: float, max_new_trace: int,
           max_new_realize: int, max_tokens: int, batch_size: int, eval_batch: int,
           realizer_dir: str, layer_frac: float = 0.6, t_ctx: int = 128,
           warmup: int = 2, trace_cos: bool = True, smoke_realizer: bool = False,
           out_dir: str = "artifacts/lggn_tracer", log=print):
    import numpy as np
    from v5.runtime.lggn_decode import _train_refiner
    from v5.runtime.lggn_refine import _reprs_from_texts

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    texts = _trace_texts(triples)
    g, f, ctx, cmask = _reprs_from_texts(
        model_name, texts, layer_frac=layer_frac, t_ctx=t_ctx,
        cache_key=_cache_key(model_name, len(texts), layer_frac, t_ctx))
    d_latent = g.shape[1]
    log(f"  {len(g)} instances, d={d_latent}")

    results: dict[str, dict] = {}
    for seed in seeds:
        tr, he = split_by_goal(triples, seed, held_frac)     # SAME split as M1 seed k
        he_eval = he[:eval_n]
        seed_res = results.setdefault(str(seed), {})
        log(f"\n--- seed {seed}: train {len(tr)} / held {len(he)} (eval {len(he_eval)}) ---")

        # GATE 2 — refiner retargeted to trace space
        raw_cos = _cos_rows(g[he], f[he])
        h_all, ops, net, cos_he = _train_refiner(
            g, f, ctx, cmask, np.asarray(tr), K=K, r=r, n_op=n_op,
            epochs=refiner_epochs, seed=seed, log=log)
        g2 = cos_he >= 0.45 and cos_he >= raw_cos + 0.05
        log(f"  G2 refiner->trace: cos={cos_he:.3f} raw={raw_cos:.3f} -> {'PASS' if g2 else 'FAIL'}"
            + ("" if g2 else "  (fallbacks: --contrastive 0.1 / composite g'=hid(goal+old))"))
        seed_res["refiner"] = {"cos_he": cos_he, "raw_cos": raw_cos, "g2": g2}
        _save_results(results)
        h_mean = h_all[np.asarray(tr)].mean(0)

        # tracer arms
        gen_traces: dict[str, list[str]] = {}
        for arm in arms:
            log(f"  [tracer/{arm}] training ({tracer_epochs} epochs, {len(tr)} pairs)...")
            lm = RawLM(model_name, d_latent=d_latent, use_molora=(arm != "baseline"))
            pairs = [(tracer_prompt(triples[i]["old"]), triples[i]["trace"]) for i in tr]
            lm.train_on(pairs, latents=_arm_latents(arm, tr, h_all, f, h_mean),
                        epochs=tracer_epochs, warmup=warmup,
                        z_dropout=(z_dropout if arm == "latent" else 0.0),
                        max_tokens=max_tokens, batch_size=batch_size, log=log)
            outs: list[str] = []
            t0 = time.time()
            for b0 in range(0, len(he_eval), eval_batch):
                chunk = he_eval[b0:b0 + eval_batch]
                prompts = [tracer_prompt(triples[i]["old"]) for i in chunk]
                zs = _arm_latents(arm, chunk, h_all, f, h_mean)
                outs.extend(lm.generate_raw_batch(prompts, zs=zs, max_new_tokens=max_new_trace))
                log(f"      {b0+len(chunk)}/{len(he_eval)} traces gen'd "
                    f"({(time.time()-t0)/(b0+len(chunk)):.1f}s/inst)")
            gen_traces[arm] = outs
            with open(f"{out_dir}/traces_seed{seed}_{arm}.jsonl", "w", encoding="utf-8") as w:
                for j, i in enumerate(he_eval):
                    w.write(json.dumps({"idx": i, "gen_trace": outs[j][:600],
                                        "gold_trace": triples[i]["trace"][:400]},
                                       ensure_ascii=False) + "\n")
            lm.cleanup()

        # e2e through the frozen seed-matched M1 realizer (one load scores all arms + anchor)
        ck = f"{realizer_dir}/seed{seed}_trace"
        if smoke_realizer and not Path(ck).exists():
            log(f"  [smoke] no realizer ckpt at {ck} -> training a throwaway stand-in...")
            rlz = RawLM(model_name)
            rp = [(realizer_prompt(triples[i]["old"], triples[i]["trace"]), triples[i]["new"])
                  for i in tr]
            rlz.train_on(rp, epochs=1, max_tokens=max_tokens, batch_size=batch_size, log=log)
        else:
            log(f"  [e2e] loading frozen M1 realizer <- {ck}")
            rlz = RawLM.load_checkpoint(model_name, ck)

        def realize_score(traces: list[str]) -> tuple[float, int]:
            recs = []
            for b0 in range(0, len(he_eval), eval_batch):
                chunk = he_eval[b0:b0 + eval_batch]
                prompts = [realizer_prompt(triples[i]["old"], traces[b0 + j])
                           for j, i in enumerate(chunk)]
                for gen, i in zip(rlz.generate_raw_batch(prompts, max_new_tokens=max_new_realize),
                                  chunk):
                    rec = added_recall(gen, triples[i]["old"], triples[i]["new"])
                    if rec is not None:
                        recs.append(rec)
            import numpy as _np
            return (float(_np.mean(recs)) if recs else 0.0), len(recs)

        gold_anchor, n_anchor = realize_score([triples[i]["trace"] for i in he_eval])
        log(f"  [e2e] gold-trace anchor = {gold_anchor:.3f} (n={n_anchor})")
        seed_res["gold_anchor"] = {"e2e_added_recall": gold_anchor, "n": n_anchor}
        for arm in arms:
            e2e, n_scored = realize_score(gen_traces[arm])
            log(f"  [e2e/{arm}] added_recall = {e2e:.3f} (n={n_scored})")
            seed_res[arm] = {"e2e_added_recall": e2e, "n": n_scored}
            _save_results(results)
        rlz.cleanup()

        # trace-cos diagnostic: cos(repr(gen trace), f_trace) per arm
        if trace_cos:
            flat, spans = [], {}
            for arm in arms:
                spans[arm] = (len(flat), len(flat) + len(gen_traces[arm]))
                flat.extend(gen_traces[arm])
            embs = _embed_texts(model_name, flat, layer_frac)
            f_he = f[np.asarray(he_eval)]
            for arm in arms:
                a0, a1 = spans[arm]
                c = _cos_rows(embs[a0:a1], f_he)
                seed_res[arm]["trace_cos"] = c
                log(f"  [trace-cos/{arm}] cos(gen, gold_trace_repr) = {c:.3f}")
            _save_results(results)

        anchor = _m1_notrace_anchor(seed)
        if anchor is not None:
            log(f"  [anchor] M1 notrace (lower bound) seed {seed} = {anchor:.3f}")

    merged = _save_results(results)
    _report_m2(merged, log)
    log(f"  results -> {RESULTS_PATH}")
    return merged


def _report_m2(results: dict, log=print):
    import numpy as np
    log("\n=== M2 TRACER (z -> trace -> frozen realizer; e2e added-line recall) ===")
    arms = [a for a in ARM_ORDER if any(a in s for s in results.values())]

    def vals(arm, key="e2e_added_recall"):
        return [s[arm][key] for s in results.values() if arm in s and key in s[arm]]

    for arm in arms + ["gold_anchor"]:
        v = vals(arm)
        if v:
            tc = vals(arm, "trace_cos")
            tcs = f"  trace_cos={np.mean(tc):.3f}" if tc else ""
            log(f"  {arm:12}: e2e = {np.mean(v):.3f} +/- {np.std(v):.3f}  ({len(v)} seeds){tcs}")
    rcos = [s["refiner"]["cos_he"] for s in results.values() if "refiner" in s]
    if rcos:
        log(f"  refiner     : cos(h_K, f_trace) = {np.mean(rcos):.3f} +/- {np.std(rcos):.3f}")

    def delta(a, b):
        d = [s[a]["e2e_added_recall"] - s[b]["e2e_added_recall"]
             for s in results.values() if a in s and b in s]
        return (float(np.mean(d)), float(np.std(d)), all(x > 0 for x in d)) if d else None

    log("")
    d = delta("ceiling", "baseline")
    if d:
        log(f"  G3 ceiling - baseline      : {d[0]:+.3f} +/- {d[1]:.3f}  "
            f"-> {'PASS' if d[0] >= 0.05 else 'FAIL (z channel dead for traces -> stop, rethink bridge)'}")
    d = delta("latent", "baseline")
    if d:
        log(f"  G4 latent - baseline       : {d[0]:+.3f} +/- {d[1]:.3f} all_pos={d[2]}  "
            f"-> {'PASS' if d[0] >= 0.03 and d[2] else 'FAIL'}")
    d = delta("ceiling", "latent")
    if d:
        log(f"     ceiling - latent        : {d[0]:+.3f}  (refiner headroom)")
    tc_l, tc_c = vals("latent", "trace_cos"), vals("constant", "trace_cos")
    if tc_l and tc_c:
        log(f"     trace-cos latent>const  : {np.mean(tc_l):.3f} vs {np.mean(tc_c):.3f}  "
            f"-> {'PASS' if np.mean(tc_l) > np.mean(tc_c) else 'FAIL'}")


# ── selftest (no GPU, no model) ─────────────────────────────────────────────────

def _selftest() -> bool:
    import numpy as np
    print("lggn_tracer --selftest: texts, arm z-matrix, refiner wiring, e2e plumbing (no model)\n")

    trips = [{"goal": f"goal {k}", "trace": f"trace {k}", "old": f"old {k}", "new": f"new {k}",
              "session_id": ""} for k in range(6)]
    texts = _trace_texts(trips)
    assert texts[0]["issue"] == "goal 0" and texts[0]["code"] == "old 0"
    assert texts[0]["added"] == "trace 0", "repr target must be the TRACE, not the fix"
    assert "fable5trace" in _cache_key("Qwen/Qwen2.5-3B", 10, 0.6, 128), "new cache namespace"
    print("  [1] trace texts + cache key -> PASS")

    h_all = np.arange(12, dtype=np.float32).reshape(6, 2)
    fr = h_all + 100
    h_mean = h_all.mean(0)
    idx = [1, 3]
    assert _arm_latents("baseline", idx, h_all, fr, h_mean) is None
    const = _arm_latents("constant", idx, h_all, fr, h_mean)
    assert np.allclose(const[0], h_mean) and np.allclose(const[1], h_mean)
    lat = _arm_latents("latent", idx, h_all, fr, h_mean)
    assert np.allclose(lat[0], h_all[1]) and np.allclose(lat[1], h_all[3])
    ceil = _arm_latents("ceiling", idx, h_all, fr, h_mean)
    assert np.allclose(ceil[1], fr[3])
    print("  [2] arm z-matrix -> PASS")

    from v5.runtime.lggn_decode import _train_refiner
    rng = np.random.RandomState(0)
    N, d, t = 40, 24, 6
    g = rng.randn(N, d).astype(np.float32)
    delta = rng.randn(1, d).astype(np.float32)
    f2 = g + delta                                          # one shared displacement -> learnable
    ctx = rng.randn(N, t, d).astype(np.float32)
    cmask = np.ones((N, t), dtype=bool)
    h, ops, net, cos_he = _train_refiner(g, f2, ctx, cmask, np.arange(30), K=2, r=32, n_op=8,
                                         epochs=60, seed=0, log=lambda *a, **k: None)
    assert h.shape == (N, d) and cos_he > 0.5, f"refiner failed to learn shared shift (cos={cos_he:.2f})"
    print(f"  [3] refiner retarget wiring (synthetic cos={cos_he:.2f}) -> PASS")

    tp = tracer_prompt("OLD SPAN")
    rp = realizer_prompt("OLD SPAN", "GEN TRACE")
    assert rp.startswith(tp), "tracer prompt must be a strict prefix of the realizer prompt"
    fake_traces = [f"t{k}" for k in range(3)]
    prompts = [realizer_prompt(trips[i]["old"], fake_traces[j]) for j, i in enumerate([0, 2, 4])]
    assert all(fake_traces[j] in prompts[j] for j in range(3)), "gen traces flow into realizer prompts"
    print("  [4] e2e plumbing + prefix property -> PASS")

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = str(Path(td) / "r.json")
        _save_results({"0": {"baseline": {"e2e_added_recall": 0.1, "n": 5}}}, p)
        m = _save_results({"0": {"latent": {"e2e_added_recall": 0.2, "n": 5}},
                           "1": {"baseline": {"e2e_added_recall": 0.15, "n": 5}}}, p)
        assert set(m["0"]) == {"baseline", "latent"} and "1" in m, "merge keeps earlier arms"
    _report_m2({"0": {"refiner": {"cos_he": 0.5, "raw_cos": 0.3, "g2": True},
                      "baseline": {"e2e_added_recall": 0.05, "n": 9, "trace_cos": 0.3},
                      "constant": {"e2e_added_recall": 0.06, "n": 9, "trace_cos": 0.31},
                      "latent": {"e2e_added_recall": 0.12, "n": 9, "trace_cos": 0.4},
                      "ceiling": {"e2e_added_recall": 0.15, "n": 9, "trace_cos": 0.5},
                      "gold_anchor": {"e2e_added_recall": 0.2, "n": 9}}},
                log=lambda *a, **k: None)
    print("  [5] merge-write + report -> PASS")

    print("\n  LGGN_TRACER SELFTEST -> PASS")
    return True


# ── CLI ─────────────────────────────────────────────────────────────────────────

def main():
    import sys
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="LGGN v2 M2 — tracer: z -> trace -> frozen realizer.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny local run: 0.5B, 32 triples, refiner 60ep, arms baseline+latent, "
                         "throwaway realizer if no ckpt")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--triples", default=TRIPLES)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed-list", default="", help="exact seeds, e.g. '0' (walltime-safe jobs)")
    ap.add_argument("--arms", default="baseline,constant,latent,ceiling")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--r", type=int, default=512)
    ap.add_argument("--n-op", type=int, default=48)
    ap.add_argument("--refiner-epochs", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=8,
                    help="tracer LM epochs. MoLoRA needs >=6 EFFECTIVE epochs (v1 finding: "
                         "2ep/1eff = unspecialized experts, loss identical to baseline)")
    ap.add_argument("--warmup", type=int, default=2, help="conditioner freeze epochs (v1 recipe: 2 of 8)")
    ap.add_argument("--z-dropout", type=float, default=0.1)
    ap.add_argument("--eval-n", type=int, default=150)
    ap.add_argument("--held-frac", type=float, default=0.2)
    ap.add_argument("--trace-chars", type=int, default=400)
    ap.add_argument("--max-new-trace", type=int, default=128)
    ap.add_argument("--max-new-realize", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--eval-batch", type=int, default=8)
    ap.add_argument("--realizer-dir", default="artifacts/lggn_realizer")
    ap.add_argument("--layer-frac", type=float, default=0.6)
    ap.add_argument("--t-ctx", type=int, default=128)
    ap.add_argument("--no-trace-cos", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-fresh-only", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        raise SystemExit(0 if _selftest() else 1)

    triples = load_triples(a.triples, trace_chars=a.trace_chars, fresh_only=not a.no_fresh_only,
                           limit=a.limit, stats=True)
    print(f"  {len(triples)} usable triples")
    seeds = ([int(s) for s in a.seed_list.split(",") if s.strip() != ""]
             if a.seed_list.strip() else list(range(a.seeds)))
    if a.smoke:
        a.model = "Qwen/Qwen2.5-0.5B" if a.model == "Qwen/Qwen2.5-3B" else a.model
        triples = triples[:32]
        print(f"[SMOKE] model={a.model} n={len(triples)} seed 0, arms baseline+latent")
        run_m2(a.model, triples, [0], ["baseline", "latent"], K=2, r=64, n_op=8,
               refiner_epochs=60, tracer_epochs=1, z_dropout=0.1, eval_n=6, held_frac=0.25,
               max_new_trace=96, max_new_realize=192, max_tokens=a.max_tokens,
               batch_size=2, eval_batch=2, realizer_dir=a.realizer_dir,
               layer_frac=a.layer_frac, t_ctx=32, warmup=0, trace_cos=True, smoke_realizer=True)
        return
    arms = [x for x in ARM_ORDER if x in set(a.arms.split(","))]
    print(f"[lggn-tracer M2] model={a.model} seeds={seeds} arms={arms} K={a.K} r={a.r} "
          f"ops={a.n_op} ref_ep={a.refiner_epochs} tracer_ep={a.epochs} warmup={a.warmup} "
          f"zdrop={a.z_dropout} eval_n={a.eval_n} batch={a.batch_size}/{a.eval_batch}")
    if a.epochs - a.warmup < 6:
        print(f"  WARN: only {a.epochs - a.warmup} effective MoLoRA epochs (<6) — v1 showed "
              "experts stay unspecialized; conditioned arms will look like baseline.")
    run_m2(a.model, triples, seeds, arms, K=a.K, r=a.r, n_op=a.n_op,
           refiner_epochs=a.refiner_epochs, tracer_epochs=a.epochs, z_dropout=a.z_dropout,
           eval_n=a.eval_n, held_frac=a.held_frac, max_new_trace=a.max_new_trace,
           max_new_realize=a.max_new_realize, max_tokens=a.max_tokens,
           batch_size=a.batch_size, eval_batch=a.eval_batch, realizer_dir=a.realizer_dir,
           layer_frac=a.layer_frac, t_ctx=a.t_ctx, warmup=a.warmup,
           trace_cos=not a.no_trace_cos)


if __name__ == "__main__":
    main()
