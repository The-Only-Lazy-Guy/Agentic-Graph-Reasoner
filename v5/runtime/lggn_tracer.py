"""LGGN v2 M2 — the TRACER: graph latent h_K + goal text -> reasoning trace -> frozen M1 realizer.

M1 PASSED all gates (trace 0.210 vs notrace 0.004 vs shuffled 0.003, 3 seeds): real traces carry
essentially ALL realization signal. M2 asks: can the GRAPH supply the trace?

BRIDGE v2 (after the span-only G3 verdict, 2026-07-06): the tracer sees GOAL TEXT + old span,
and z STEERS. The span-only tracer proved z cannot RECONSTRUCT trace content (that is embedding
inversion — vec2text needs millions of pairs, we have 936): with z = the gold trace's own repr,
generation reached only trace_cos 0.541 (baseline 0.493) and e2e stayed at the notrace floor
(0.002-0.005 vs gold-anchor 0.165) — while training loss DID diverge (1.038 vs 1.158) and
trace_cos DID move, i.e. z steers but cannot dictate. So give the tracer the content basis that
exists at inference anyway (the issue/goal is DATA, not prompt engineering — Fable-5's goal is
the whole-session request, the trace is the edit-specific plan, so goal->trace remains a real
derivation task) and let z do the one job it demonstrably can: disambiguate/steer.

Arms (fresh tracer each): baseline (goal+span, pure LoRA) / constant (mean h_K broadcast) /
latent (h_K per instance, z_dropout) / ceiling (z = gold trace repr) / retrieval (NO LM:
h_K nearest-neighbor over TRAIN gold-trace reprs -> that train instance's trace text; full text
fidelity, wrong instance — measures specificity-vs-generality of traces).
End-to-end: generated trace -> frozen seed-matched M1 realizer -> added_recall vs gold new span.
Anchors: gold-trace-through-realizer (upper), M1 notrace (lower, from M1 results json).

Gates: G2 refiner cos(h_K, f_trace) >= 0.45 and >= raw cos(g,f)+0.05 (before tracer training) |
G3 ceiling-first: e2e ceiling-baseline >= +0.05 (1 seed, else the z channel adds nothing over
goal text -> stop) | G4 e2e latent-baseline >= +0.03 every seed, and trace-cos latent > constant.

  wiring  : python -m v5.runtime.lggn_tracer --selftest        # no GPU, no model
  smoke   : python -m v5.runtime.lggn_tracer --smoke           # 0.5B, real loop, local GPU
  G3 first: V5_LM_QUANT=4bit python -m v5.runtime.lggn_tracer --seed-list 0 --arms ceiling,baseline,retrieval
  full    : V5_LM_QUANT=4bit python -m v5.runtime.lggn_tracer --seed-list 0   (then 1, then 2)
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from v5.runtime.lggn_realizer import (SEP_T, TRIPLES, RawLM, added_recall, load_triples,
                                      realizer_prompt, split_by_goal, tracer_prompt)

RESULTS_PATH = "artifacts/lggn_tracer_results.json"
M1_RESULTS_PATH = "artifacts/lggn_realizer_results.json"
ARM_ORDER = ("baseline", "constant", "latent", "ceiling", "retrieval")

SEP_O = "\n###O\n"          # old span follows (tracer-side only; realizer format unchanged)


def tracer_goal_prompt(goal: str, old: str) -> str:
    """Bridge v2 tracer input: goal text (content basis, available at inference) + old span.
    Still zero instruction text — both fields are data. NOTE: no longer a prefix of the
    realizer prompt (that property belonged to the span-only tracer)."""
    return goal[:700] + SEP_O + old + SEP_T


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


def _get_realizer(model_name: str, ck: str, triples: list[dict], tr: list[int],
                  max_tokens: int, batch_size: int, smoke: bool = False, log=print) -> "RawLM":
    """Load the frozen seed-matched M1 realizer; if the checkpoint is missing (fresh box —
    artifacts/ is gitignored), retrain it inline with the exact M1 recipe (trace arm, same
    goal-split) and SAVE it, so E1/M2-A/M2 are self-sufficient anywhere."""
    if Path(ck, "lora").exists():
        log(f"  [e2e] loading frozen M1 realizer <- {ck}")
        return RawLM.load_checkpoint(model_name, ck)
    log(f"  [e2e] realizer ckpt missing at {ck} -> training inline "
        f"({'1 epoch throwaway' if smoke else 'M1 recipe, 2 epochs, saved'})...")
    rlz = RawLM(model_name)
    pairs = [(realizer_prompt(triples[i]["old"], triples[i]["trace"]), triples[i]["new"])
             for i in tr]
    rlz.train_on(pairs, epochs=1 if smoke else 2, max_tokens=max_tokens,
                 batch_size=batch_size, log=log)
    if not smoke:
        rlz.save_checkpoint(ck)
        log(f"  [e2e] realizer checkpoint saved -> {ck}")
    return rlz


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
            if arm == "retrieval":
                # NO LM: h_K of the held instance -> nearest TRAIN gold-trace repr -> that
                # train instance's trace TEXT. Full text fidelity, wrong instance.
                # No leakage: h_K uses only (g, ctx); the pool is train-only.
                f_tr = f[np.asarray(tr)]
                f_tr_n = f_tr / (np.linalg.norm(f_tr, axis=1, keepdims=True) + 1e-9)
                outs = []
                for i in he_eval:
                    q = h_all[i] / (np.linalg.norm(h_all[i]) + 1e-9)
                    outs.append(triples[tr[int(np.argmax(f_tr_n @ q))]]["trace"])
                log(f"  [tracer/retrieval] {len(outs)} traces retrieved (no training)")
            else:
                log(f"  [tracer/{arm}] training ({tracer_epochs} epochs, {len(tr)} pairs)...")
                lm = RawLM(model_name, d_latent=d_latent, use_molora=(arm != "baseline"))
                pairs = [(tracer_goal_prompt(triples[i]["goal"], triples[i]["old"]),
                          triples[i]["trace"]) for i in tr]
                lm.train_on(pairs, latents=_arm_latents(arm, tr, h_all, f, h_mean),
                            epochs=tracer_epochs, warmup=warmup,
                            z_dropout=(z_dropout if arm == "latent" else 0.0),
                            max_tokens=max_tokens, batch_size=batch_size, log=log)
                outs = []
                t0 = time.time()
                for b0 in range(0, len(he_eval), eval_batch):
                    chunk = he_eval[b0:b0 + eval_batch]
                    prompts = [tracer_goal_prompt(triples[i]["goal"], triples[i]["old"])
                               for i in chunk]
                    zs = _arm_latents(arm, chunk, h_all, f, h_mean)
                    outs.extend(lm.generate_raw_batch(prompts, zs=zs, max_new_tokens=max_new_trace))
                    log(f"      {b0+len(chunk)}/{len(he_eval)} traces gen'd "
                        f"({(time.time()-t0)/(b0+len(chunk)):.1f}s/inst)")
                lm.cleanup()
            gen_traces[arm] = outs
            with open(f"{out_dir}/traces_seed{seed}_{arm}.jsonl", "w", encoding="utf-8") as w:
                for j, i in enumerate(he_eval):
                    w.write(json.dumps({"idx": i, "gen_trace": outs[j][:600],
                                        "gold_trace": triples[i]["trace"][:400]},
                                       ensure_ascii=False) + "\n")

        # e2e through the frozen seed-matched M1 realizer (one load scores all arms + anchor)
        rlz = _get_realizer(model_name, f"{realizer_dir}/seed{seed}_trace", triples, tr,
                            max_tokens, batch_size, smoke=smoke_realizer, log=log)

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


# ── E1: trace decomposition — where does the trace's information live? ──────────

_IDENT = None  # compiled lazily


def _ident_pattern():
    global _IDENT
    if _IDENT is None:
        import re
        # code-ish tokens: backticked spans, dotted.paths, snake_case, CamelCase, CONSTS,
        # quoted short strings — the BINDINGS of a trace
        _IDENT = re.compile(
            r"`[^`]+`"
            r"|\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b"   # dotted.path
            r"|\b[a-z0-9]+(?:_[a-z0-9]+)+\b"                               # snake_case
            r"|\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+\b"                      # CamelCase
            r"|\b[A-Z][A-Z0-9_]{2,}\b"                                     # CONSTS
            r"|\"[^\"\n]{1,40}\"|'[^'\n]{1,40}'")
    return _IDENT


def trace_skeleton(trace: str) -> str:
    """Strategy skeleton: identifiers/bindings masked out."""
    return _ident_pattern().sub("<X>", trace or "")


def trace_idents(trace: str) -> list[str]:
    return _ident_pattern().findall(trace or "")


def bind_skeleton(skeleton: str, idents: list[str]) -> str:
    """v0 binder: fill <X> slots round-robin with the given identifiers (naive; measures
    whether skeleton+bindings carries the signal before building a learned binder)."""
    out, k = [], 0
    parts = (skeleton or "").split("<X>")
    for i, part in enumerate(parts):
        out.append(part)
        if i < len(parts) - 1:
            out.append(idents[k % len(idents)] if idents else "<X>")
            k += 1
    return "".join(out)


def run_e1_decompose(model_name: str, triples: list[dict], seed: int, eval_n: int,
                     held_frac: float, max_new_realize: int, max_tokens: int,
                     batch_size: int, eval_batch: int, realizer_dir: str,
                     layer_frac: float = 0.6, t_ctx: int = 128,
                     smoke_realizer: bool = False,
                     out_dir: str = "artifacts/lggn_tracer", log=print):
    """Feed the frozen realizer VARIANTS of the gold trace (no training anywhere):
      gold        : as-is (anchor, ~0.165)
      skeleton    : identifiers masked            -> value of strategy alone
      idents_only : identifiers without prose     -> value of bindings alone
      nbr_bound   : NEIGHBOR skeleton (h_K-retrieved from TRAIN) + THIS instance's gold
                    identifiers (v0 round-robin binder) -> viability of the operator-library
                    path (graph retrieves skeleton, binder fills, LM realizes)
    Decomposes the trace's information into strategy vs bindings."""
    import numpy as np
    from v5.runtime.lggn_decode import _train_refiner
    from v5.runtime.lggn_refine import _reprs_from_texts

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    texts = _trace_texts(triples)
    g, f, ctx, cmask = _reprs_from_texts(
        model_name, texts, layer_frac=layer_frac, t_ctx=t_ctx,
        cache_key=_cache_key(model_name, len(texts), layer_frac, t_ctx))
    tr, he = split_by_goal(triples, seed, held_frac)
    he_eval = he[:eval_n]
    log(f"\n--- E1 decompose, seed {seed}: eval {len(he_eval)} held (no training) ---")
    # h_K only for neighbor retrieval (train-pool skeletons)
    h_all, _ops, _net, cos_he = _train_refiner(
        g, f, ctx, cmask, np.asarray(tr), K=4, r=512, n_op=48, epochs=400, seed=seed,
        log=lambda *a, **k: None)
    log(f"  refiner cos = {cos_he:.3f} (used for neighbor skeleton retrieval only)")
    f_tr = f[np.asarray(tr)]
    f_tr_n = f_tr / (np.linalg.norm(f_tr, axis=1, keepdims=True) + 1e-9)

    variants: dict[str, list[str]] = {"gold": [], "skeleton": [], "idents_only": [], "nbr_bound": []}
    for i in he_eval:
        gold = triples[i]["trace"]
        ids_ = trace_idents(gold)
        q = h_all[i] / (np.linalg.norm(h_all[i]) + 1e-9)
        nbr = triples[tr[int(np.argmax(f_tr_n @ q))]]["trace"]
        variants["gold"].append(gold)
        variants["skeleton"].append(trace_skeleton(gold))
        variants["idents_only"].append(" ".join(ids_) if ids_ else gold)
        variants["nbr_bound"].append(bind_skeleton(trace_skeleton(nbr), ids_))

    rlz = _get_realizer(model_name, f"{realizer_dir}/seed{seed}_trace", triples, tr,
                        max_tokens, batch_size, smoke=smoke_realizer, log=log)
    res = {"refiner_cos": cos_he, "n": 0}
    for name, tr_list in variants.items():
        recs = []
        for b0 in range(0, len(he_eval), eval_batch):
            chunk = he_eval[b0:b0 + eval_batch]
            outs = rlz.generate_raw_batch(
                [realizer_prompt(triples[i]["old"], tr_list[b0 + j])
                 for j, i in enumerate(chunk)],
                max_new_tokens=max_new_realize)
            for gen, i in zip(outs, chunk):
                rec = added_recall(gen, triples[i]["old"], triples[i]["new"])
                if rec is not None:
                    recs.append(rec)
        res[name] = float(np.mean(recs)) if recs else 0.0
        res["n"] = len(recs)
        log(f"  [{name:12}] added_recall = {res[name]:.3f} (n={len(recs)})")
    rlz.cleanup()
    with open(f"{out_dir}/decompose_seed{seed}.jsonl", "w", encoding="utf-8") as w:
        for j, i in enumerate(he_eval[:30]):
            w.write(json.dumps({
                "idx": int(i),
                "gold": variants["gold"][j][:250], "skeleton": variants["skeleton"][j][:250],
                "idents_only": variants["idents_only"][j][:250],
                "nbr_bound": variants["nbr_bound"][j][:250],
            }, ensure_ascii=False) + "\n")
    _save_results({f"decompose_seed{seed}": res})
    log(f"\n=== E1 TRACE DECOMPOSITION (seed {seed}, n={res['n']}) ===")
    log(f"  gold         {res['gold']:.3f}   (anchor)")
    log(f"  skeleton     {res['skeleton']:.3f}   <- strategy alone")
    log(f"  idents_only  {res['idents_only']:.3f}   <- bindings alone")
    log(f"  nbr_bound    {res['nbr_bound']:.3f}   <- retrieved skeleton + my bindings (the architecture)")
    log(f"  reading: skeleton+bindings both needed if each alone << gold; nbr_bound >> retrieval "
        f"floor (0.002) = operator-library path viable")
    log(f"  results -> {RESULTS_PATH} | samples -> {out_dir}/decompose_seed{seed}.jsonl")
    return res


# ── M2-A: sampling + selection (generation for COVERAGE, latent for SELECTION) ──

def _rank_candidates(cand_reprs, query, n_samples: int) -> list[int]:
    """Per instance: argmax cosine(candidate repr, query repr). cand_reprs [N*S, d] grouped by
    instance (S consecutive rows each), query [N, d]. Returns picked candidate index per instance."""
    import numpy as np
    picks = []
    for i in range(len(query)):
        c = cand_reprs[i * n_samples:(i + 1) * n_samples]
        cn = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-9)
        q = query[i] / (np.linalg.norm(query[i]) + 1e-9)
        picks.append(int(np.argmax(cn @ q)))
    return picks


def run_m2a_selection(model_name: str, triples: list[dict], seed: int, n_samples: int,
                      temperature: float, K: int, r: int, n_op: int, refiner_epochs: int,
                      tracer_epochs: int, eval_n: int, held_frac: float, max_new_trace: int,
                      max_new_realize: int, max_tokens: int, batch_size: int, eval_batch: int,
                      realizer_dir: str, layer_frac: float = 0.6, t_ctx: int = 128,
                      warmup: int = 2, smoke_realizer: bool = False,
                      out_dir: str = "artifacts/lggn_tracer", log=print):
    """Sample N traces per held instance, realize ALL, compare pickers on the same candidates:
      oracle   : max added_recall over candidates  (coverage — does a good trace even exist?)
      random   : mean over candidates              (expected single sample)
      z_rank   : candidate nearest h_K             (THE LGGN selection mechanism)
      gold_rank: candidate nearest gold-trace repr (selection ceiling given perfect z)
    Gates: GA1 oracle >= 0.05 (else generation coverage dead on this corpus) |
           GA2 z_rank - random >= +0.02 and gold_rank > random (selection signal exists)."""
    import numpy as np
    import torch as _torch
    from v5.runtime.lggn_decode import _train_refiner
    from v5.runtime.lggn_refine import _reprs_from_texts

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    texts = _trace_texts(triples)
    g, f, ctx, cmask = _reprs_from_texts(
        model_name, texts, layer_frac=layer_frac, t_ctx=t_ctx,
        cache_key=_cache_key(model_name, len(texts), layer_frac, t_ctx))
    d_latent = g.shape[1]
    tr, he = split_by_goal(triples, seed, held_frac)
    he_eval = he[:eval_n]
    log(f"\n--- M2-A seed {seed}: train {len(tr)} / eval {len(he_eval)} x {n_samples} samples "
        f"(temp={temperature}) ---")
    h_all, ops, net, cos_he = _train_refiner(
        g, f, ctx, cmask, np.asarray(tr), K=K, r=r, n_op=n_op,
        epochs=refiner_epochs, seed=seed, log=log)
    log(f"  refiner cos(h_K, f_trace) = {cos_he:.3f}")

    log(f"  [tracer] training ({tracer_epochs} epochs, {len(tr)} pairs, pure LoRA)...")
    lm = RawLM(model_name)
    pairs = [(tracer_goal_prompt(triples[i]["goal"], triples[i]["old"]), triples[i]["trace"])
             for i in tr]
    lm.train_on(pairs, epochs=tracer_epochs, warmup=warmup, max_tokens=max_tokens,
                batch_size=batch_size, log=log)
    cands: list[str] = []                                  # instance-major, S per instance
    t0 = time.time()
    for j, i in enumerate(he_eval):
        _torch.manual_seed(10_000 * seed + j)              # reproducible sampling
        prompts = [tracer_goal_prompt(triples[i]["goal"], triples[i]["old"])] * n_samples
        cands.extend(lm.generate_raw_batch(prompts, max_new_tokens=max_new_trace,
                                           temperature=temperature))
        if (j + 1) % 10 == 0:
            log(f"      {j+1}/{len(he_eval)} instances sampled "
                f"({(time.time()-t0)/(j+1):.1f}s/inst)")
    lm.cleanup()

    rlz = _get_realizer(model_name, f"{realizer_dir}/seed{seed}_trace", triples, tr,
                        max_tokens, batch_size, smoke=smoke_realizer, log=log)
    recalls = np.zeros((len(he_eval), n_samples))          # None -> treated as 0 for ranking
    scoreable = np.zeros((len(he_eval), n_samples), dtype=bool)
    t0 = time.time()
    flat_prompts = [realizer_prompt(triples[i]["old"], cands[j * n_samples + s])
                    for j, i in enumerate(he_eval) for s in range(n_samples)]
    done = 0
    for b0 in range(0, len(flat_prompts), eval_batch):
        outs = rlz.generate_raw_batch(flat_prompts[b0:b0 + eval_batch],
                                      max_new_tokens=max_new_realize)
        for off, gen in enumerate(outs):
            idx = b0 + off
            j, s = divmod(idx, n_samples)
            rec = added_recall(gen, triples[he_eval[j]]["old"], triples[he_eval[j]]["new"])
            if rec is not None:
                recalls[j, s] = rec
                scoreable[j, s] = True
        done += len(outs)
        if done % (eval_batch * 8) < eval_batch:
            log(f"      {done}/{len(flat_prompts)} realized ({(time.time()-t0)/done:.1f}s/cand)")
    gold_anchor_recs = []
    for b0 in range(0, len(he_eval), eval_batch):
        chunk = he_eval[b0:b0 + eval_batch]
        outs = rlz.generate_raw_batch(
            [realizer_prompt(triples[i]["old"], triples[i]["trace"]) for i in chunk],
            max_new_tokens=max_new_realize)
        for gen, i in zip(outs, chunk):
            rec = added_recall(gen, triples[i]["old"], triples[i]["new"])
            if rec is not None:
                gold_anchor_recs.append(rec)
    rlz.cleanup()

    log("  embedding candidates for ranking...")
    cand_reprs = _embed_texts(model_name, cands, layer_frac)
    he_arr = np.asarray(he_eval)
    keep = scoreable.any(1)                                # instances with at least one scoreable cand
    z_picks = _rank_candidates(cand_reprs, h_all[he_arr], n_samples)
    gold_picks = _rank_candidates(cand_reprs, f[he_arr], n_samples)
    res = {
        "n": int(keep.sum()), "n_samples": n_samples, "temperature": temperature,
        "refiner_cos": cos_he,
        "oracle": float(recalls[keep].max(1).mean()),
        "random": float(recalls[keep].mean(1).mean()),
        "z_rank": float(np.mean([recalls[j, z_picks[j]] for j in range(len(he_eval)) if keep[j]])),
        "gold_rank": float(np.mean([recalls[j, gold_picks[j]] for j in range(len(he_eval)) if keep[j]])),
        "gold_anchor": float(np.mean(gold_anchor_recs)) if gold_anchor_recs else 0.0,
        "cand_nonempty": float(np.mean([bool(c.strip()) for c in cands])),
    }
    with open(f"{out_dir}/selection_seed{seed}.jsonl", "w", encoding="utf-8") as w:
        for j, i in enumerate(he_eval[:40]):
            w.write(json.dumps({
                "idx": int(i), "gold_trace": triples[i]["trace"][:300],
                "recalls": [round(float(x), 3) for x in recalls[j]],
                "z_pick": z_picks[j], "gold_pick": gold_picks[j],
                "cands": [c[:200] for c in cands[j * n_samples:(j + 1) * n_samples]],
            }, ensure_ascii=False) + "\n")
    merged = _save_results({f"selection_seed{seed}": res})
    log(f"\n=== M2-A SELECTION (seed {seed}, N={n_samples}, temp={temperature}, "
        f"n={res['n']}) ===")
    log(f"  oracle best-of-{n_samples}: {res['oracle']:.3f}   <- coverage")
    log(f"  random pick          : {res['random']:.3f}")
    log(f"  z-ranked (h_K)       : {res['z_rank']:.3f}   <- the LGGN selector")
    log(f"  gold-ranked (ceiling): {res['gold_rank']:.3f}")
    log(f"  gold-trace anchor    : {res['gold_anchor']:.3f}")
    ga1 = res["oracle"] >= 0.05
    ga2 = (res["z_rank"] - res["random"] >= 0.02) and (res["gold_rank"] > res["random"])
    log(f"\n  GA1 oracle >= 0.05           : {res['oracle']:.3f} -> "
        f"{'PASS' if ga1 else 'FAIL (generation coverage dead on this corpus)'}")
    log(f"  GA2 z_rank - random >= +0.02 : {res['z_rank'] - res['random']:+.3f} "
        f"(gold_rank - random {res['gold_rank'] - res['random']:+.3f}) -> "
        f"{'PASS' if ga2 else 'FAIL'}")
    log(f"  results -> {RESULTS_PATH} | candidates -> {out_dir}/selection_seed{seed}.jsonl")
    return res


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

    tp = tracer_goal_prompt("THE GOAL", "OLD SPAN")
    assert tp == "THE GOAL" + SEP_O + "OLD SPAN" + SEP_T, "bridge-v2 tracer format"
    assert tracer_goal_prompt("g" * 2000, "o").startswith("g" * 700 + SEP_O), "goal capped at 700"
    fake_traces = [f"t{k}" for k in range(3)]
    prompts = [realizer_prompt(trips[i]["old"], fake_traces[j]) for j, i in enumerate([0, 2, 4])]
    assert all(fake_traces[j] in prompts[j] for j in range(3)), "gen traces flow into realizer prompts"
    print("  [4] e2e plumbing + bridge-v2 format -> PASS")

    # retrieval arm: h_K nearest TRAIN gold-trace repr -> that train instance's trace
    f6 = np.eye(6, 4, dtype=np.float32)                     # instance i points along axis i (i<4)
    h6 = np.zeros((6, 4), dtype=np.float32)
    h6[4] = [0, 1, 0.1, 0]                                  # held 4 nearest to train 1
    h6[5] = [0.1, 0, 0, 1]                                  # held 5 nearest to train 3
    tr6, he6 = [0, 1, 2, 3], [4, 5]
    f_tr = f6[np.asarray(tr6)]
    f_tr_n = f_tr / (np.linalg.norm(f_tr, axis=1, keepdims=True) + 1e-9)
    picked = [tr6[int(np.argmax(f_tr_n @ (h6[i] / (np.linalg.norm(h6[i]) + 1e-9)))) ] for i in he6]
    assert picked == [1, 3], f"retrieval NN picked {picked}"
    print("  [6] retrieval nearest-neighbor -> PASS")

    # E1 decomposition: skeleton masks identifiers, binder refills
    t = "Wrap `get_url` in a try/except and log conn_err via logger.warning"
    sk = trace_skeleton(t)
    assert "`get_url`" not in sk and "conn_err" not in sk and "logger.warning" not in sk, sk
    assert "<X>" in sk and "try/except" in sk, "prose structure survives masking"
    ids_ = trace_idents(t)
    assert "`get_url`" in ids_ and "conn_err" in ids_ and "logger.warning" in ids_, ids_
    rebound = bind_skeleton(sk, ids_)
    assert "`get_url`" in rebound and "<X>" not in rebound, "binder fills all slots"
    assert bind_skeleton("a <X> b", []) == "a <X> b", "no idents -> slots stay"

    # M2-A ranking: z picks the candidate nearest the query
    cands = np.array([[1, 0], [0, 1], [0.9, 0.1], [0, 1]], dtype=np.float32)  # 2 inst x 2 cands
    queries = np.array([[1, 0], [0, 1]], dtype=np.float32)
    picks = _rank_candidates(cands, queries, 2)
    assert picks == [0, 1], f"rank picks {picks}"
    print("  [7] skeleton/binder + candidate ranking -> PASS")

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
    ap.add_argument("--arms", default="baseline,constant,latent,ceiling,retrieval")
    ap.add_argument("--select", action="store_true",
                    help="M2-A: sample N traces, realize all, compare pickers "
                         "(oracle/random/z-rank/gold-rank)")
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--decompose", action="store_true",
                    help="E1: gold trace variants (skeleton/idents/nbr_bound) through the "
                         "frozen realizer — where does the trace's information live? No training.")
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
        if a.select:
            print(f"[SMOKE/select] model={a.model} n={len(triples)} seed 0, N=3 samples")
            run_m2a_selection(a.model, triples, 0, n_samples=3, temperature=0.8, K=2, r=64,
                              n_op=8, refiner_epochs=60, tracer_epochs=1, eval_n=6,
                              held_frac=0.25, max_new_trace=96, max_new_realize=192,
                              max_tokens=a.max_tokens, batch_size=2, eval_batch=2,
                              realizer_dir=a.realizer_dir, layer_frac=a.layer_frac,
                              t_ctx=32, warmup=0, smoke_realizer=True)
            return
        if a.decompose:
            print(f"[SMOKE/decompose] model={a.model} n={len(triples)} seed 0")
            run_e1_decompose(a.model, triples, 0, eval_n=6, held_frac=0.25,
                             max_new_realize=192, max_tokens=a.max_tokens, batch_size=2,
                             eval_batch=2, realizer_dir=a.realizer_dir,
                             layer_frac=a.layer_frac, t_ctx=32, smoke_realizer=True)
            return
        print(f"[SMOKE] model={a.model} n={len(triples)} seed 0, arms baseline+latent")
        run_m2(a.model, triples, [0], ["baseline", "latent", "retrieval"], K=2, r=64, n_op=8,
               refiner_epochs=60, tracer_epochs=1, z_dropout=0.1, eval_n=6, held_frac=0.25,
               max_new_trace=96, max_new_realize=192, max_tokens=a.max_tokens,
               batch_size=2, eval_batch=2, realizer_dir=a.realizer_dir,
               layer_frac=a.layer_frac, t_ctx=32, warmup=0, trace_cos=True, smoke_realizer=True)
        return
    if a.decompose:
        print(f"[lggn-tracer E1] model={a.model} seeds={seeds} eval_n={a.eval_n}")
        for seed in seeds:
            run_e1_decompose(a.model, triples, seed, eval_n=a.eval_n, held_frac=a.held_frac,
                             max_new_realize=a.max_new_realize, max_tokens=a.max_tokens,
                             batch_size=a.batch_size, eval_batch=a.eval_batch,
                             realizer_dir=a.realizer_dir, layer_frac=a.layer_frac,
                             t_ctx=a.t_ctx)
        return
    if a.select:
        print(f"[lggn-tracer M2-A] model={a.model} seeds={seeds} N={a.n_samples} "
              f"temp={a.temperature} tracer_ep={a.epochs} eval_n={a.eval_n} "
              f"batch={a.batch_size}/{a.eval_batch}")
        for seed in seeds:
            run_m2a_selection(a.model, triples, seed, n_samples=a.n_samples,
                              temperature=a.temperature, K=a.K, r=a.r, n_op=a.n_op,
                              refiner_epochs=a.refiner_epochs, tracer_epochs=a.epochs,
                              eval_n=a.eval_n, held_frac=a.held_frac,
                              max_new_trace=a.max_new_trace, max_new_realize=a.max_new_realize,
                              max_tokens=a.max_tokens, batch_size=a.batch_size,
                              eval_batch=a.eval_batch, realizer_dir=a.realizer_dir,
                              layer_frac=a.layer_frac, t_ctx=a.t_ctx, warmup=a.warmup)
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
