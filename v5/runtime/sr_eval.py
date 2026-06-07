"""Search/Replace generation eval — does the SR format fix emission for the weak 4B?

Compares cold vs inject, but asks for SEARCH/REPLACE blocks (SR_SYS) instead of a unified
diff. Metric: well_formed (emitted >=1 parseable block) should jump well above the ~15/20
applyable-diff rate, because SR drops the line-number/hunk math the 4B fails at. Also n_blocks
/ file_cov / edit_cov, plus a dump for manual inspection. No verifier.

  V5_LM_TRUST_REMOTE_CODE=1 python -m v5.runtime.sr_eval \
    --adapter-ckpt artifacts/stage_cache/adapter_code_s3.pt --n-eval 20 --dump artifacts/sr_dump.jsonl
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
from v5.graph_grower.swe_load import load_instances, patch_files
from v5.graph_grower.swe_probe import load_traces, load_node_texts, _symbol_name
from v5.runtime.search_replace import SR_SYS, parse_sr, sr_metrics


def run(model_name, traces_p, nodes_p, adapter_ckpt, dataset, split, n_eval, max_new,
        dump="", device_str=None):
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device}  lm={model_name}  adapter={adapter_ckpt}")
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
    id2text = load_node_texts(nodes_p)
    insts = {t["instance_id"]: t for t in load_instances(dataset, split, limit=0)}
    ids = [i for i in traces if i in insts][:n_eval]
    print(f"eval instances={len(ids)}", flush=True)

    rows = {"cold": [], "inject": []}
    records = []
    tf = {"task_family": "code_fix", "required_slots": []}
    for k, iid in enumerate(ids):
        t = traces[iid]; inst = insts[iid]
        support = [s for s in t["support_ids"] if s in id2text]
        if not support:
            continue
        gold_files = patch_files(inst.get("patch", ""))
        gold_syms = [n for s in support if (n := _symbol_name(id2text[s]))]
        node_ids = support[:24]
        text_emb = embedder.embed_nodes({s: id2text[s] for s in node_ids})
        injector.prepare_session(_stub_graph(node_ids, id2text, {s: "fact" for s in node_ids}),
                                 node_ids, text_emb, tf, r_plan=3, r_evidence=4)
        msgs = [{"role": "system", "content": SR_SYS},
                {"role": "user", "content": f"ISSUE:\n{t['issue'][:2000]}\n\n"
                                            "Output ONLY search/replace blocks (no prose)."}]
        cold = _gen(model, tok, msgs, device, injector, False, max_new)
        inj = _gen(model, tok, msgs, device, injector, True, max_new)
        cb, ib = parse_sr(cold), parse_sr(inj)
        cm = sr_metrics(cb, gold_files, gold_syms); im = sr_metrics(ib, gold_files, gold_syms)
        rows["cold"].append(cm); rows["inject"].append(im)
        records.append({"id": iid, "gold_symbols": gold_syms, "gold_patch": inst.get("patch", ""),
                        "cold": cold, "inject": inj, "cold_blocks": cb, "inject_blocks": ib})
        print(f"  [{k+1}/{len(ids)}] {iid:26} well-formed {cm['well_formed']:.0f}->{im['well_formed']:.0f}  "
              f"blocks {cm['n_blocks']}->{im['n_blocks']}  file {cm['file_cov']:.0f}->{im['file_cov']:.0f}  "
              f"edit {cm['edit_cov']:.2f}->{im['edit_cov']:.2f}", flush=True)

    if dump:
        Path(dump).parent.mkdir(parents=True, exist_ok=True)
        with open(dump, "w", encoding="utf-8") as w:
            for r in records:
                w.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\ndumped {len(records)} -> {dump}", flush=True)

    def _m(c, kk):
        return round(sum(r[kk] for r in rows[c]) / max(1, len(rows[c])), 4)
    print("\n=== SEARCH/REPLACE emission (cold -> inject) ===")
    for m in ("well_formed", "n_blocks", "file_cov", "edit_cov"):
        print(f"  {m:11} {_m('cold', m)} -> {_m('inject', m)}")
    print("\nwell_formed >> the ~0.75 unified-diff applyable rate = the SR format fixed emission."
          "\nThese blocks deterministically apply -> ready for the verifier (real test-pass).")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Search/Replace emission eval.")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--traces", nargs="+", default=["data/swe/grounded_traces.jsonl"])
    ap.add_argument("--nodes", nargs="+", default=["artifacts/graph_growth/swe_code_candidates.jsonl"])
    ap.add_argument("--adapter-ckpt", default="artifacts/stage_cache/adapter_code_s3.pt")
    ap.add_argument("--dataset", default="lite")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-eval", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=500)
    ap.add_argument("--dump", default="")
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    run(a.model, a.traces, a.nodes, a.adapter_ckpt, a.dataset, a.split, a.n_eval, a.max_new,
        dump=a.dump, device_str=a.device)


if __name__ == "__main__":
    raise SystemExit(main())
