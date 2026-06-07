"""Grounded coder — the capstone runtime: one self-improving loop over a task stream.

Ties every validated piece into a single agent that gets better as it goes:

  for each task:
    retrieve nearest EXEMPLAR from memory (a past solved fix, exclude self)   [learning]
    -> checkout repo + READ the real source of the localized symbols           [grounding]
    -> INJECT the support subgraph (trained adapter, frozen LM)                 [reasoning]
    -> emit SEARCH/REPLACE edit, APPLY-FEEDBACK RETRY until it matches the file [emission+loop]
    -> GATE (applyable now; swap to the verifier/tests at the benchmark stage)
    -> on pass: CAPTURE the working patch as a new exemplar -> memory grows     [post_processing]

The memory (exemplar store) starts empty and grows with the agent's OWN solved patches, so
later tasks retrieve + adapt earlier solutions. The gate is pluggable: `applyable` (no infra,
demo of the closed loop) or `verify` (real test-pass — wire at the Docker stage). The rigorous
"does memory lift resolve-rate" is the benchmark; this proves the loop CLOSES + memory grows.

  V5_LM_TRUST_REMOTE_CODE=1 python -m v5.runtime.grounded_coder \
    --adapter-ckpt artifacts/stage_cache/adapter_code_s3.pt --n-eval 30 --max-retries 3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from v5.adapter import GraphAttentionInjector
from v5.cross_attention import V5AttentionAdapter
from v5.gnn_encoder import RGCNEncoder
from v5.goal_encoder import GoalEncoder
from v5.training.providers import RealEmbedder, FrozenQwenHInitProvider
from v5.training.stage4_generate import _stub_graph
from v5.graph_grower.swe_load import load_instances, checkout_repo
from v5.graph_grower.swe_probe import load_traces, _symbol_name
from v5.runtime.search_replace import apply_sr
from v5.runtime.sr_withcode import load_symbol_meta, read_body
from v5.runtime.verifier_retry import solve
import subprocess


def _unit(v):
    a = np.asarray(v, dtype=np.float32)
    return a / (np.linalg.norm(a) + 1e-9)


def run(model_name, traces_p, nodes_p, adapter_ckpt, dataset, split, n_eval, max_new,
        repo_root, max_retries, gate, device_str=None):
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device}  gate={gate}  max_retries={max_retries}", flush=True)
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
    print(f"tasks={len(ids)} | symbol meta={len(meta)}", flush=True)

    store: list = []                          # growing memory: [{iid, vec, diff, symbols}]
    tf = {"task_family": "code_fix", "required_slots": []}
    solved = used_ex = 0

    def nearest_exemplar(iid, vec):
        best, bd = None, -1.0
        for e in store:
            if e["iid"] == iid:
                continue
            s = float(np.dot(vec, e["vec"]))
            if s > bd:
                bd, best = s, e
        return (best["diff"] if best else ""), (best["iid"] if best else "")

    for k, iid in enumerate(ids):
        t = traces[iid]; inst = insts[iid]
        support = [s for s in t["support_ids"] if s in meta]
        if not support:
            continue
        vec = _unit(embedder.embed_nodes({"q": t["issue"]})["q"])
        ex_diff, ex_id = nearest_exemplar(iid, vec)
        used_ex += 1 if ex_diff else 0

        dest = Path(repo_root) / f"{inst['repo'].replace('/', '__')}__{inst['base_commit'][:8]}"
        ok, _ = checkout_repo(inst["repo"], inst["base_commit"], dest)
        if not ok:
            print(f"  [{k+1}] {iid} checkout FAILED"); continue
        src = "\n\n".join(f"# {meta[s]['file']}\n{read_body(str(dest), meta[s]['file'], meta[s]['lineno'], 70)}"
                          for s in support[:6] if read_body(str(dest), meta[s]['file'], meta[s]['lineno'], 70))
        node_ids = support[:24]
        text_emb = embedder.embed_nodes({s: meta[s]["text"] for s in node_ids})
        injector.prepare_session(_stub_graph(node_ids, {s: meta[s]["text"] for s in node_ids},
                                             {s: "fact" for s in node_ids}),
                                 node_ids, text_emb, tf, r_plan=3, r_evidence=4)

        applyable, attempts, blocks, _ = solve(model, tok, injector, t["issue"], src, str(dest),
                                              max_retries, max_new, exemplar_diff=ex_diff)
        patch = ""
        if applyable:
            _, patch = apply_sr(str(dest), blocks)
            subprocess.run(["git", "-C", str(dest), "checkout", "--", "."], capture_output=True)

        passed = applyable if gate == "applyable" else False   # gate=='verify' -> wire swe_verify here
        if passed and patch.strip():
            solved += 1
            syms = [n for s in support if (n := _symbol_name(meta[s]["text"]))]
            store.append({"iid": iid, "vec": vec, "diff": patch, "symbols": syms})   # MEMORY GROWS
        print(f"  [{k+1}/{len(ids)}] {iid:24} ex={'Y('+ex_id[:18]+')' if ex_diff else 'N':22} "
              f"applyable={applyable} solved={passed} | memory={len(store)}", flush=True)

    print(f"\n=== GROUNDED CODER (closed loop, gate={gate}) ===")
    print(f"  tasks {len(ids)} | solved {solved} | memory grew 0 -> {len(store)} exemplars | "
          f"tasks-with-retrieved-exemplar {used_ex}")
    print("\nmemory grows with the agent's OWN solved patches -> later tasks retrieve + adapt them."
          "\nThe loop CLOSES: solve -> capture -> grow -> next task uses it. (gate=verify + the"
          "\nbenchmark = the rigorous resolve-rate + the memory lift.)")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Grounded coder — the self-improving capstone loop.")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--traces", nargs="+", default=["data/swe/grounded_traces.jsonl"])
    ap.add_argument("--nodes", nargs="+", default=["artifacts/graph_growth/swe_code_candidates.jsonl"])
    ap.add_argument("--adapter-ckpt", default="artifacts/stage_cache/adapter_code_s3.pt")
    ap.add_argument("--dataset", default="lite")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-eval", type=int, default=30)
    ap.add_argument("--max-new", type=int, default=500)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--repo-root", default="data/swe_repos")
    ap.add_argument("--gate", choices=["applyable", "verify"], default="applyable",
                    help="applyable = closed-loop demo (no infra); verify = real test-pass (Docker)")
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    run(a.model, a.traces, a.nodes, a.adapter_ckpt, a.dataset, a.split, a.n_eval, a.max_new,
        a.repo_root, a.max_retries, a.gate, device_str=a.device)


if __name__ == "__main__":
    raise SystemExit(main())
