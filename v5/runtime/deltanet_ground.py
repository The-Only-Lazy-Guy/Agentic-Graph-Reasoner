"""Phase 2 — train the node->(k,v) projector + run the killer unlearn test (needs the 4B, A40).

The make-or-break for the DeltaNet graph-memory bet. Plumbing is validated
(v5/deltanet_layer_test.py, v5/runtime/deltanet_inject.py). The two open questions:
  (1) Can a trained projector make the FROZEN 4B actually READ the written nodes (lower gold NLL)?
  (2) Is node add/remove clean+selective (the unlearn that cross-attn injection failed)?

Train mode: only the projector trains (LM frozen, NO per-node LoRA). Loss = gold SR-NLL with nodes
written + ortho_weight * key-orthogonality (T3b: selectivity needs ~orthogonal keys).

Killer mode (--killer): two disjoint-skill instances A,B. Measure gold-NLL under:
  cold (no graph) / both (A+B written) / drop-A (A removed).
  PASS = A grounded (both<cold) AND A unlearned (drop-A -> back toward cold) AND B retained
         (drop-A ~ both for B).

  Train : V5_LM_TRUST_REMOTE_CODE=1 V5_LM_QUANT=4bit python -m v5.runtime.deltanet_ground \
            --layers auto2 --n-train 120 --n-eval 40 --epochs 2 --out artifacts/stage_cache/dn_proj.pt
  Killer: ... --killer --proj artifacts/stage_cache/dn_proj.pt
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext

import torch

from v5.training.providers import RealEmbedder, FrozenQwenHInitProvider
from v5.training.stage_sr_sft import _build_rows, patch_to_sr, _user
from v5.runtime.search_replace import SR_SYS
from v5.runtime.sr_withcode import load_symbol_meta
from v5.graph_grower.swe_load import load_instances
from v5.graph_grower.swe_probe import load_traces
from v5.runtime.deltanet_inject import DeltaNetGraphInjector, key_orthogonality_loss


def _node_emb(embedder, texts: dict, order):
    """texts {id:str}, order [ids] -> [N, E] tensor (frozen embedder)."""
    e = embedder.embed_nodes(texts)
    import numpy as np
    return torch.tensor(np.stack([e[i] for i in order]), dtype=torch.float32)


def dn_nll(model, tok, issue, src, sr_text, device, max_new=400):
    """Gold SR-NLL under the model AS CURRENTLY CONFIGURED (injector S_graph set or cleared)."""
    if not sr_text.strip():
        return None
    msgs = [{"role": "system", "content": SR_SYS}, {"role": "user", "content": _user(issue, src)}]
    p = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                return_dict=True)["input_ids"].to(device)
    a = tok(sr_text, add_special_tokens=False, return_tensors="pt",
            truncation=True, max_length=max_new).input_ids.to(device)
    if a.shape[1] == 0:
        return None
    full = torch.cat([p, a], dim=1)
    if full.shape[1] > 1600:
        return None
    logits = model(full).logits
    lp = torch.log_softmax(logits[0, p.shape[1] - 1:-1].float(), dim=-1)
    idx = torch.arange(a.shape[1], device=lp.device)
    return -lp[idx, a[0]].mean()


def _pick_layers(model, spec):
    gdn = DeltaNetGraphInjector.gdn_layer_indices(model)
    if not gdn:
        raise ValueError("model has no GatedDeltaNet layers")
    if spec == "auto2":
        return [gdn[len(gdn) // 2], gdn[-1]]            # one mid, one late linear-attn layer
    if spec == "auto1":
        return [gdn[-1]]
    return [int(x) for x in spec.split(",")]


def _load_rows(traces_p, nodes_p, dataset, split, repo_root, skip, n, src_bodies, src_lines):
    traces = load_traces([traces_p]); meta = load_symbol_meta([nodes_p])
    insts = {t["instance_id"]: t for t in load_instances(dataset, split, limit=0)}
    ids = [i for i in traces if i in insts][skip: skip + n]
    return _build_rows(ids, traces, insts, meta, repo_root, src_bodies, src_lines)


def run(a):
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    provider = FrozenQwenHInitProvider(a.model, device=device)
    model, tok = provider.model, provider.tok
    for p in model.parameters():
        p.requires_grad_(False)
    embedder = RealEmbedder(device)
    emb_dim = len(embedder.embed_nodes({"q": "x"})["q"])
    layers = _pick_layers(model, a.layers)
    print(f"inject GatedDeltaNet layers {layers} | emb_dim {emb_dim}", flush=True)
    inj = DeltaNetGraphInjector(model, layers, emb_dim=emb_dim, device=device)
    if a.proj:
        inj.projectors.load_state_dict(torch.load(a.proj, map_location=device))
        print(f"loaded projector <- {a.proj}", flush=True)

    if a.killer:
        return _killer(a, model, tok, embedder, inj, device)

    rows = _load_rows(a.traces, a.nodes, a.dataset, a.split, a.repo_root, a.skip,
                      a.n_train + a.n_eval, a.src_bodies, a.src_lines)
    train_r, eval_r = rows[:a.n_train], rows[a.n_train:a.n_train + a.n_eval]
    print(f"rows: {len(train_r)} train / {len(eval_r)} eval", flush=True)
    opt = torch.optim.AdamW(inj.projectors.parameters(), lr=a.lr)
    for ep in range(a.epochs):
        inj.projectors.train(); tot = n = 0
        for r in train_r:
            emb = _node_emb(embedder, r["texts"], r["node_ids"]).to(device)
            inj.set_nodes(emb)
            nll = dn_nll(model, tok, r["issue"], r["src"], r["sr"], device)
            if nll is None or not torch.isfinite(nll):
                inj.clear(); continue
            ortho = sum(key_orthogonality_loss(k) for k in inj.keys_for_ortho_loss(emb).values())
            loss = nll + a.ortho_weight * ortho
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(inj.projectors.parameters(), 1.0)
            opt.step(); inj.clear()
            tot += float(nll); n += 1
            if n % 25 == 0:
                print(f"    ep{ep+1} {n}/{len(train_r)} NLL {tot/max(1,n):.4f}", flush=True)
        print(f"  [dn] epoch {ep+1} mean grounded-NLL {tot/max(1,n):.4f} ({n})", flush=True)

    torch.save(inj.projectors.state_dict(), a.out)
    print(f"saved projector -> {a.out}", flush=True)

    inj.projectors.eval(); cold, grd = [], []
    with torch.no_grad():
        for r in eval_r:
            inj.clear()
            c = dn_nll(model, tok, r["issue"], r["src"], r["sr"], device)
            inj.set_nodes(_node_emb(embedder, r["texts"], r["node_ids"]).to(device))
            g = dn_nll(model, tok, r["issue"], r["src"], r["sr"], device)
            inj.clear()
            if c is not None and g is not None and torch.isfinite(c) and torch.isfinite(g):
                cold.append(float(c)); grd.append(float(g))
    if cold:
        import statistics as st
        win = sum(1 for c, g in zip(cold, grd) if g < c)
        print(f"\n=== DeltaNet grounding held-out NLL ===")
        print(f"  cold {st.mean(cold):.4f} -> grounded {st.mean(grd):.4f} "
              f"(delta {st.mean(cold)-st.mean(grd):+.4f}) | grounded better {win}/{len(cold)}")
        print("  delta>0 = the frozen LM READS the DeltaNet-written nodes (the open question #1).")


def _killer(a, model, tok, embedder, inj, device):
    """Two disjoint skills A,B: cold / both / drop-A. PASS = A grounded+unlearned, B retained."""
    rows = _load_rows(a.traces, a.nodes, a.dataset, a.split, a.repo_root, 0, a.max_ids,
                      a.src_bodies, a.src_lines)
    seen = {}
    for r in rows:
        seen.setdefault(r["iid"].split("__")[0], r)
    pair = list(seen.values())[:2]
    if len(pair) < 2:
        print("need 2 different-repo instances"); return
    A, B = pair
    setA = set(A["node_ids"])
    union_ids = A["node_ids"] + [n for n in B["node_ids"] if n not in setA]
    union_texts = {**A["texts"], **B["texts"]}
    print(f"A={A['iid']} ({len(A['node_ids'])}n)  B={B['iid']} ({len(B['node_ids'])}n)", flush=True)

    @torch.no_grad()
    def probe(row):
        out = {}
        inj.clear(); out["cold"] = float(dn_nll(model, tok, row["issue"], row["src"], row["sr"], device))
        inj.set_nodes(_node_emb(embedder, union_texts, union_ids).to(device))
        out["both"] = float(dn_nll(model, tok, row["issue"], row["src"], row["sr"], device))
        inj.set_nodes(_node_emb(embedder, union_texts, union_ids).to(device))
        for ai in range(len(A["node_ids"])):     # drop all of A's nodes (A is first in union)
            inj.drop_node(0)
        out["drop-A"] = float(dn_nll(model, tok, row["issue"], row["src"], row["sr"], device))
        inj.clear()
        return out

    rA, rB = probe(A), probe(B)
    print(f"\n  skill  {'cold':>8} {'both':>8} {'drop-A':>8}")
    for nm, r in (("A", rA), ("B", rB)):
        print(f"  {nm:5}  {r['cold']:8.4f} {r['both']:8.4f} {r['drop-A']:8.4f}")
    liftA = rA["cold"] - rA["both"]; revertA = rA["drop-A"] - rA["both"]
    fracA = revertA / liftA if abs(liftA) > 1e-6 else 0.0
    driftB = abs(rB["drop-A"] - rB["both"])
    ok = liftA > 0 and fracA > 0.5 and driftB < 0.3 * abs(liftA)
    print(f"\n  A lift {liftA:+.3f} | A revert {revertA:+.3f} ({fracA*100:.0f}%) | B drift {driftB:.3f}")
    print(f"  KILLER TEST: {'PASS — DeltaNet gives node-specific grounding + selective unlearn' if ok else 'FAIL'}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--traces", default="data/swe/grounded_traces.jsonl")
    ap.add_argument("--nodes", default="artifacts/graph_growth/swe_code_candidates.jsonl")
    ap.add_argument("--layers", default="auto2", help="auto1|auto2|comma idxs of GatedDeltaNet layers")
    ap.add_argument("--dataset", default="lite"); ap.add_argument("--split", default="test")
    ap.add_argument("--repo-root", default="data/swe_repos")
    ap.add_argument("--skip", type=int, default=30)
    ap.add_argument("--n-train", type=int, default=120); ap.add_argument("--n-eval", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ortho-weight", type=float, default=0.1)
    ap.add_argument("--src-bodies", type=int, default=4); ap.add_argument("--src-lines", type=int, default=55)
    ap.add_argument("--out", default="artifacts/stage_cache/dn_proj.pt")
    ap.add_argument("--proj", default="", help="load a trained projector (killer mode / warm start)")
    ap.add_argument("--killer", action="store_true"); ap.add_argument("--max-ids", type=int, default=40)
    ap.add_argument("--device", default=None)
    run(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
