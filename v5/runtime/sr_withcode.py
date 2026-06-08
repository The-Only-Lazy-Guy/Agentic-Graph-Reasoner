"""Read-the-source step — feed the localized symbol's ACTUAL code, then emit SR edits.

Diagnosis: the brief gives symbol SIGNATURES (where), not the function BODY (what to edit), so
the 4B writes SEARCH blocks for code it never saw -> no-op / wrong-function / unmatchable edits.
Fix (no graph-search tool, no verifier): checkout repo@base_commit, read the real source of each
support symbol (metadata.file + lineno), put it in context, and let the model edit what it can see.

Compares INJECT (signatures only, current) vs INJECT+SOURCE (real code in prompt). The decisive
metric: `search_in_file` = does the emitted SEARCH block EXACTLY match the checked-out file
(= the edit can actually apply). If that jumps with source -> the bottleneck was missing code,
not a model that ignores grounding.

  V5_LM_TRUST_REMOTE_CODE=1 python -m v5.runtime.sr_withcode \
    --adapter-ckpt artifacts/stage_cache/adapter_code_s3.pt --n-eval 15 --dump artifacts/sr_withcode_dump.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from v5.adapter import GraphAttentionInjector
from v5.cross_attention import V5AttentionAdapter
from v5.gnn_encoder import RGCNEncoder
from v5.goal_encoder import GoalEncoder
from v5.training.providers import RealEmbedder, FrozenQwenHInitProvider
from v5.training.stage4_generate import _gen, _stub_graph
from v5.graph_grower.swe_load import load_instances, patch_files, checkout_repo
from v5.graph_grower.swe_probe import load_traces, _symbol_name
from v5.runtime.search_replace import SR_SYS, parse_sr, sr_metrics


def load_symbol_meta(paths):
    """node_id -> {file, lineno, text} for symbol nodes."""
    meta = {}
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            e = r.get("raw_edit") or r          # candidates are flat (no raw_edit nesting)
            if e.get("op") == "add_node" and e.get("node_type") == "symbol":
                m = e.get("metadata", {})
                meta[e["node_id"]] = {"file": m.get("file"), "lineno": m.get("lineno"),
                                      "text": e.get("text", "")}
    return meta


def read_body(repo_dir, file_rel, lineno, max_lines=70) -> str:
    fp = Path(repo_dir) / (file_rel or "")
    if not fp.exists() or not lineno:
        return ""
    lines = fp.read_text(encoding="utf-8", errors="ignore").splitlines()
    lo, hi = lineno[0], lineno[1]
    body = lines[max(0, lo - 1): min(len(lines), hi)]
    return "\n".join(body[:max_lines])


def _file_text(repo_dir, rel):
    if not rel:                              # empty/missing file in an SR block -> no text (not the repo dir)
        return ""
    fp = Path(repo_dir) / rel
    return fp.read_text(encoding="utf-8", errors="ignore") if fp.is_file() else ""


def _search_in_file(repo_dir, blocks) -> float:
    """fraction of SR blocks whose SEARCH text exactly occurs in its target file (applyable)."""
    if not blocks:
        return 0.0
    ok = 0
    for b in blocks:
        s = (b.get("search") or "").strip()
        if s and s in _file_text(repo_dir, b.get("file")):
            ok += 1
    return ok / len(blocks)


def _user(issue, src_ctx=""):
    s = f"ISSUE:\n{issue[:1400]}\n\n"
    if src_ctx:
        return (s + f"RELEVANT SOURCE (the bug is in here):\n{src_ctx}\n\n"
                "Find the exact line(s) causing the bug and fix them. Output ONLY search/replace "
                "blocks: SEARCH must copy the source EXACTLY; REPLACE must DIFFER from SEARCH "
                "(make the real change — do NOT echo unchanged code). Keep it minimal.")
    return s + "Output ONLY search/replace blocks (no prose)."


def run(model_name, traces_p, nodes_p, adapter_ckpt, dataset, split, n_eval, max_new,
        repo_root, max_syms=6, max_body_lines=70, dump="", device_str=None):
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device}  lm={model_name}", flush=True)
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
    print(f"eval instances={len(ids)} | symbol meta={len(meta)}", flush=True)

    rows = {"inject": [], "withcode": []}
    records = []
    tf = {"task_family": "code_fix", "required_slots": []}
    for k, iid in enumerate(ids):
        t = traces[iid]; inst = insts[iid]
        support = [s for s in t["support_ids"] if s in meta]
        if not support:
            continue
        # checkout repo@base_commit (cached) and read the real source of each support symbol
        dest = Path(repo_root) / f"{inst['repo'].replace('/', '__')}__{inst['base_commit'][:8]}"
        ok, msg = checkout_repo(inst["repo"], inst["base_commit"], dest)
        if not ok:
            print(f"  [{k+1}] {iid} checkout FAILED: {msg[:60]}"); continue
        src_parts = []
        for s in support[:max_syms]:        # top-K support only -> shorter prompt, faster, less over-reasoning
            body = read_body(str(dest), meta[s]["file"], meta[s]["lineno"], max_lines=max_body_lines)
            if body:
                src_parts.append(f"# {meta[s]['file']}\n{body}")
        src_ctx = "\n\n".join(src_parts)
        gold_files = patch_files(inst.get("patch", ""))
        gold_syms = [n for s in support if (n := _symbol_name(meta[s]["text"]))]
        node_ids = support[:24]
        text_emb = embedder.embed_nodes({s: meta[s]["text"] for s in node_ids})
        injector.prepare_session(_stub_graph(node_ids, {s: meta[s]["text"] for s in node_ids},
                                             {s: "fact" for s in node_ids}),
                                 node_ids, text_emb, tf, r_plan=3, r_evidence=4)

        inj = _gen(model, tok, [{"role": "system", "content": SR_SYS},
                                {"role": "user", "content": _user(t["issue"])}], device, injector, True, max_new)
        wc = _gen(model, tok, [{"role": "system", "content": SR_SYS},
                               {"role": "user", "content": _user(t["issue"], src_ctx)}], device, injector, True, max_new)
        ib, wb = parse_sr(inj), parse_sr(wc)
        im = sr_metrics(ib, gold_files, gold_syms); im["search_in_file"] = _search_in_file(str(dest), ib)
        wm = sr_metrics(wb, gold_files, gold_syms); wm["search_in_file"] = _search_in_file(str(dest), wb)
        rows["inject"].append(im); rows["withcode"].append(wm)
        records.append({"id": iid, "gold_symbols": gold_syms, "gold_patch": inst.get("patch", ""),
                        "src_chars": len(src_ctx), "inject_blocks": ib, "withcode_blocks": wb,
                        "withcode": wc})
        print(f"  [{k+1}/{len(ids)}] {iid:24} wf {im['well_formed']:.0f}->{wm['well_formed']:.0f}  "
              f"search_in_file {im['search_in_file']:.2f}->{wm['search_in_file']:.2f}  "
              f"edit {im['edit_cov']:.2f}->{wm['edit_cov']:.2f}", flush=True)

    if dump:
        Path(dump).parent.mkdir(parents=True, exist_ok=True)
        with open(dump, "w", encoding="utf-8") as w:
            for r in records:
                w.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\ndumped {len(records)} -> {dump}", flush=True)

    def _m(c, kk):
        return round(sum(r[kk] for r in rows[c]) / max(1, len(rows[c])), 4)
    print("\n=== READ-THE-SOURCE: SR emission (inject=signatures -> withcode=real source) ===")
    for m in ("well_formed", "search_in_file", "file_cov", "edit_cov"):
        print(f"  {m:14} {_m('inject', m)} -> {_m('withcode', m)}")
    print("\nsearch_in_file jump = giving the real code lets the 4B write APPLYABLE edits -> the"
          "\nbottleneck was missing CODE (graph gave location, not body), not the model ignoring grounding.")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Read-the-source SR emission eval.")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--traces", nargs="+", default=["data/swe/grounded_traces.jsonl"])
    ap.add_argument("--nodes", nargs="+", default=["artifacts/graph_growth/swe_code_candidates.jsonl"])
    ap.add_argument("--adapter-ckpt", default="artifacts/stage_cache/adapter_code_s3.pt")
    ap.add_argument("--dataset", default="lite")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-eval", type=int, default=15)
    ap.add_argument("--max-new", type=int, default=500)
    ap.add_argument("--repo-root", default="data/swe_repos")
    ap.add_argument("--max-syms", type=int, default=6,
                    help="source bodies for top-K support. Trimming to 2 HURT — all support "
                         "symbols are patch-relevant (AST-mapped), don't drop the buggy function.")
    ap.add_argument("--max-body-lines", type=int, default=70)
    ap.add_argument("--dump", default="")
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    run(a.model, a.traces, a.nodes, a.adapter_ckpt, a.dataset, a.split, a.n_eval, a.max_new,
        a.repo_root, max_syms=a.max_syms, max_body_lines=a.max_body_lines, dump=a.dump, device_str=a.device)


if __name__ == "__main__":
    raise SystemExit(main())
