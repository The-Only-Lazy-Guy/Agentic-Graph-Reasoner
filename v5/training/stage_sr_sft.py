"""Stage SR-SFT — distill SYNTHESIS into the adapter (frozen LM).

Stages 1/2/3 taught the adapter to ATTEND + make the gold ANSWER more likely. But the runtime
asks for a SEARCH/REPLACE edit given (issue + REAL source + injection) — a different, harder
target. This stage trains exactly that: forward the FROZEN LM on the real runtime prompt
(SR system + issue + read-source) followed by the GOLD patch rendered as SR blocks, take the
NLL on the SR tokens (teacher-forced, injection on all positions), backprop the ADAPTER.

Why: the bottleneck is SYNTHESIS, not localization. Grounding gets the right file; the weak 4B
still mis-writes the fix. Distilling gold SR into the adapter raises SINGLE-SHOT synthesis ->
fewer agentic retries -> faster AND more accurate (money-for-speed, the time-fear fix). Frozen
base keeps the v2 thesis; the adapter learns to steer the edit. Next: RFT/STaR on verified
self-resolves.

  V5_LM_TRUST_REMOTE_CODE=1 python -m v5.training.stage_sr_sft \
    --adapter-ckpt artifacts/stage_cache/adapter_code_s3.pt --epochs 3 --n-train 200 --n-eval 40
"""
from __future__ import annotations

import argparse
import os
import statistics as st
from contextlib import nullcontext
from pathlib import Path

import torch

from v5.adapter import GraphAttentionInjector
from v5.cross_attention import V5AttentionAdapter
from v5.gnn_encoder import RGCNEncoder
from v5.goal_encoder import GoalEncoder
from v5.training.providers import RealEmbedder, FrozenQwenHInitProvider
from v5.training.stage4_generate import _stub_graph
from v5.graph_grower.swe_load import load_instances, checkout_repo
from v5.graph_grower.swe_probe import load_traces
from v5.runtime.search_replace import SR_SYS
from v5.runtime.sr_withcode import load_symbol_meta, read_body


def patch_to_sr(patch: str) -> str:
    """Unified-diff gold patch -> SEARCH/REPLACE text (the format the model emits)."""
    blocks, cur = [], None
    lines = patch.splitlines()
    i = 0
    while i < len(lines):
        l = lines[i]
        if l.startswith("+++ b/"):
            cur = l[6:]
        elif l.startswith("+++ "):
            cur = l[4:].lstrip()
        elif l.startswith("@@"):
            i += 1
            search, replace = [], []
            while i < len(lines) and not lines[i].startswith(("@@", "diff --git", "--- ", "+++ ")):
                h = lines[i]
                if h.startswith("-"):
                    search.append(h[1:])
                elif h.startswith("+"):
                    replace.append(h[1:])
                elif h.startswith(" "):
                    search.append(h[1:]); replace.append(h[1:])
                i += 1
            if cur and (search or replace):
                blocks.append((cur, "\n".join(search), "\n".join(replace)))
            continue
        i += 1
    out = []
    for f, s, r in blocks:
        out.append(f"{f}\n<<<<<<< SEARCH\n{s}\n=======\n{r}\n>>>>>>> REPLACE")
    return "\n\n".join(out)


def _user(issue, src):
    return (f"ISSUE:\n{issue[:1000]}\n\nRELEVANT SOURCE (the bug is in here):\n{src}\n\n"
            "Find the exact line(s) causing the bug and fix them. Output ONLY search/replace "
            "blocks: SEARCH must copy the source EXACTLY; REPLACE must DIFFER from SEARCH.")


def sr_nll(model, tok, injector, issue, src, sr_text, device, inject: bool, max_new=400):
    """NLL of the gold SR tokens under the (optionally injected) frozen LM, teacher-forced."""
    if not sr_text.strip():
        return None
    msgs = [{"role": "system", "content": SR_SYS}, {"role": "user", "content": _user(issue, src)}]
    p_ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                    return_dict=True)["input_ids"].to(device)
    a_ids = tok(sr_text, add_special_tokens=False, return_tensors="pt",
                truncation=True, max_length=max_new).input_ids.to(device)
    if a_ids.shape[1] == 0:
        return None
    full = torch.cat([p_ids, a_ids], dim=1)
    if full.shape[1] > 1600:                 # backstop: skip pathologically long rows -> no OOM
        return None
    ctx = injector.inject(model) if inject else nullcontext()
    with ctx:
        logits = model(full).logits
    plen = p_ids.shape[1]
    lp = torch.log_softmax(logits[0, plen - 1:-1].float(), dim=-1)
    idx = torch.arange(a_ids.shape[1], device=lp.device)
    return -lp[idx, a_ids[0]].mean()


import subprocess


def _ensure_repo(repo, commit, dest):
    """Blobless clone (once/repo) + fetch the commit object. NO working-tree checkout."""
    dest = Path(dest)
    if not (dest / ".git").exists():
        dest.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout",
                            f"https://github.com/{repo}.git", str(dest)],
                           capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            return False
    f = subprocess.run(["git", "fetch", "--depth", "1", "origin", commit],
                       cwd=str(dest), capture_output=True, text=True, timeout=900)
    return f.returncode == 0


def _body_at(dest, commit, file, lineno, max_lines):
    """Read ONE file at <commit> via `git show` (fetches just that blob) -> slice the symbol body.
    No full checkout = ~10-50x faster than materializing the whole repo per instance."""
    r = subprocess.run(["git", "-C", str(dest), "show", f"{commit}:{file}"],
                       capture_output=True, text=True, timeout=120)
    if r.returncode != 0 or not r.stdout:
        return ""
    lines = r.stdout.splitlines()
    lo = lineno[0] if lineno else 1
    hi = lineno[1] if (lineno and len(lineno) > 1) else lo
    return "\n".join(lines[max(0, lo - 1):hi][:max_lines])


def _build_rows(ids, traces, insts, meta, repo_root):
    """For each instance: (issue, read-source via git-show, support node_ids/texts, gold SR)."""
    rows = []
    for k, iid in enumerate(ids):
        if k % 25 == 0:
            print(f"    build-rows {k}/{len(ids)} (kept {len(rows)})...", flush=True)
        t, inst = traces[iid], insts[iid]
        support = [s for s in t["support_ids"] if s in meta]
        sr = patch_to_sr(inst.get("patch", "") or "")
        if not support or not sr.strip():
            continue
        # per-REPO blobless clone (once) + git-show each needed file at the commit. NO full
        # checkout -> read 4 blobs, not materialize the whole repo. ~10-50x faster.
        dest = Path(repo_root) / inst["repo"].replace("/", "__")
        commit = inst["base_commit"]
        if not _ensure_repo(inst["repo"], commit, dest):
            continue
        parts = []
        for s in support[:3]:                # 3x45 (was 4x55) -> shorter seq, fits 48GB on gym
            body = _body_at(dest, commit, meta[s]["file"], meta[s]["lineno"], 45)
            if body:
                parts.append(f"# {meta[s]['file']}\n{body}")
        src = "\n\n".join(parts)
        if not src.strip():
            continue
        rows.append({"iid": iid, "issue": t["issue"], "src": src, "sr": sr,
                     "node_ids": support[:24], "texts": {s: meta[s]["text"] for s in support[:24]}})
    return rows


def run(model_name, traces_p, nodes_p, adapter_ckpt, dataset, split, epochs, lr,
        n_train, n_eval, repo_root, out_ckpt, skip=30, device_str=None):
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device}  lm={model_name}  SR-SFT from {adapter_ckpt}", flush=True)
    provider = FrozenQwenHInitProvider(model_name, device=device)
    model, tok = provider.model, provider.tok
    # NO gradient checkpointing: it re-fires the injection hooks on every recompute segment ->
    # re-runs the all-positions adapter many times -> memory EXPLODES (24->90GB). Cap the
    # sequence instead (sr_nll truncates) so the retained activations stay bounded.
    lm_dim = provider.hidden_size
    embedder = RealEmbedder(device)
    gnn = RGCNEncoder().to(device).eval()
    for p in gnn.parameters():
        p.requires_grad_(False)
    goal_enc = GoalEncoder().to(device).eval()
    adapter = V5AttentionAdapter(r_plan=3, r_evidence=4, lm_hidden_dim=lm_dim).to(device)
    adapter.load_state_dict(torch.load(adapter_ckpt, map_location=device))
    injector = GraphAttentionInjector(adapter, gnn, goal_enc, device=device)
    injector.inject_all_positions = True       # teacher-forced: condition ALL SR tokens

    traces = load_traces(traces_p)
    meta = load_symbol_meta(nodes_p)
    insts = {t["instance_id"]: t for t in load_instances(dataset, split, limit=0)}
    # NO LEAKAGE: skip the first `skip` instances (the verifier_retry resolve-eval set) so SR-SFT
    # never trains on what we benchmark.
    ids = [i for i in traces if i in insts][skip: skip + n_train + n_eval]
    print(f"building rows (checkout + read-source) for {len(ids)} instances (skip first {skip})...", flush=True)
    rows = _build_rows(ids, traces, insts, meta, repo_root)
    train_r, eval_r = rows[:n_train], rows[n_train:n_train + n_eval]
    print(f"rows: {len(train_r)} train / {len(eval_r)} eval", flush=True)
    tf = {"task_family": "code_fix", "required_slots": []}

    def prep(r):
        injector.prepare_session(_stub_graph(r["node_ids"], r["texts"], {s: "fact" for s in r["node_ids"]}),
                                 r["node_ids"], embedder.embed_nodes(r["texts"]), tf, r_plan=3, r_evidence=4)

    opt = torch.optim.AdamW([p for p in adapter.parameters() if p.requires_grad], lr=lr)
    for ep in range(epochs):
        adapter.train()
        tot = n = 0
        for r in train_r:
            prep(r)
            nll = sr_nll(model, tok, injector, r["issue"], r["src"], r["sr"], device, inject=True)
            if nll is None or not torch.isfinite(nll):
                continue
            opt.zero_grad(); nll.backward()
            torch.nn.utils.clip_grad_norm_([p for p in adapter.parameters() if p.requires_grad], 1.0)
            opt.step()
            tot += float(nll.item()); n += 1
            del nll
            if n % 25 == 0 and device.type == "cuda":
                torch.cuda.empty_cache()
            if n % 100 == 0:
                print(f"    ep{ep+1} {n}/{len(train_r)}  running SR-NLL {tot/max(1,n):.4f}", flush=True)
        print(f"  [SR-SFT] epoch {ep+1}  mean SR-NLL(inject) {tot/max(1,n):.4f}  ({n} rows)", flush=True)

    torch.save(adapter.state_dict(), out_ckpt)
    print(f"saved SR-SFT adapter -> {out_ckpt}", flush=True)

    adapter.eval()
    cold, inj = [], []
    with torch.no_grad():
        for r in eval_r:
            prep(r)
            c = sr_nll(model, tok, injector, r["issue"], r["src"], r["sr"], device, inject=False)
            i = sr_nll(model, tok, injector, r["issue"], r["src"], r["sr"], device, inject=True)
            if c is not None and i is not None and torch.isfinite(c) and torch.isfinite(i):
                cold.append(float(c)); inj.append(float(i))
    if cold:
        mc, mi = st.mean(cold), st.mean(inj)
        win = sum(1 for c, i in zip(cold, inj) if i < c)
        print(f"\n=== SR-SFT held-out SR-NLL (lower=better) ===")
        print(f"  cold {mc:.4f} -> injected {mi:.4f}  (delta {mc-mi:+.4f}) | injection better {win}/{len(cold)}")
        print("  lower injected SR-NLL = the trained adapter makes the gold EDIT more likely ->"
              "\n  better single-shot synthesis -> fewer retries. Verify the real lift with verifier_retry.")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stage SR-SFT: distill gold SEARCH/REPLACE synthesis into the adapter.")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--traces", nargs="+", default=["data/swe/grounded_traces.jsonl"])
    ap.add_argument("--nodes", nargs="+", default=["artifacts/graph_growth/swe_code_candidates.jsonl"])
    ap.add_argument("--adapter-ckpt", default="artifacts/stage_cache/adapter_code_s3.pt")
    ap.add_argument("--out-ckpt", default="artifacts/stage_cache/adapter_code_sr.pt")
    ap.add_argument("--dataset", default="lite")
    ap.add_argument("--split", default="test")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--n-eval", type=int, default=40)
    ap.add_argument("--repo-root", default="data/swe_repos")
    ap.add_argument("--skip", type=int, default=30,
                    help="skip the first N instances (the verifier_retry resolve-eval set) -> no leakage")
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    run(a.model, a.traces, a.nodes, a.adapter_ckpt, a.dataset, a.split, a.epochs, a.lr,
        a.n_train, a.n_eval, a.repo_root, a.out_ckpt, skip=a.skip, device_str=a.device)


if __name__ == "__main__":
    raise SystemExit(main())
