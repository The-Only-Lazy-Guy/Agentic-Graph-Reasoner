"""Retrieve-and-adapt generation (plan Phase A) — attack the execution gap.

The weak 4B can't SYNTHESIZE the correct fix (stage-4 manual read: right file, wrong/garbage
content). Here it ADAPTS instead: for a held-out issue we retrieve the NEAREST OTHER exemplar
(a resolved bug's gold diff; exclude self = no leakage) and give it as a template, while the
symbol brief is INJECTED via the trained adapter (frozen LM). Three conditions compared:
  cold            : issue only
  inject          : issue + symbol brief injected (stage-4 baseline)
  retrieve-adapt  : inject + a similar resolved diff in the prompt to adapt
Measured by adherence (gold symbols/files, echo-free) + a manual patch dump. No verifier.

  V5_LM_TRUST_REMOTE_CODE=1 python -m v5.runtime.retrieve_adapt \
    --adapter-ckpt artifacts/stage_cache/adapter_code_s3.pt \
    --exemplars data/swe/exemplars.jsonl --n-eval 20 --dump artifacts/retrieve_adapt_dump.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from v5.adapter import GraphAttentionInjector
from v5.cross_attention import V5AttentionAdapter
from v5.gnn_encoder import RGCNEncoder
from v5.goal_encoder import GoalEncoder
from v5.training.providers import RealEmbedder, FrozenQwenHInitProvider
from v5.training.stage4_generate import _gen, _stub_graph, _adherence, GEN_SYS
from v5.graph_grower.swe_load import load_instances, patch_files
from v5.graph_grower.swe_probe import load_traces, load_node_texts, _symbol_name


def _unit(v):
    a = np.asarray(v, dtype=np.float32)
    return a / (np.linalg.norm(a) + 1e-9)


def _user(issue: str, exemplar_diff: str = "") -> str:
    s = f"ISSUE:\n{issue[:2000]}\n\n"
    if exemplar_diff:
        s += ("A SIMILAR resolved bug was fixed with this diff — ADAPT the same pattern to THIS "
              f"issue (do not copy it verbatim):\n{exemplar_diff[:1500]}\n\n")
    return s + "Output ONLY the unified diff that fixes THIS issue."


def run(model_name, traces_p, nodes_p, exemplars_p, adapter_ckpt, dataset, split,
        n_eval, max_new, dump="", device_str=None):
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

    exemplars = [json.loads(l) for l in Path(exemplars_p).read_text(encoding="utf-8").splitlines() if l.strip()]
    ex_ids = [e["instance_id"] for e in exemplars]
    ex_diff = {e["instance_id"]: e["diff"] for e in exemplars}
    print(f"exemplars={len(exemplars)} — embedding issues for retrieval...", flush=True)
    ev = embedder.embed_nodes({e["instance_id"]: e["issue"] for e in exemplars})
    ex_mat = np.stack([_unit(ev[i]) for i in ex_ids])           # [E, d]

    def nearest(q_iid, q_issue):
        qv = _unit(embedder.embed_nodes({"q": q_issue})["q"])
        order = np.argsort(-(ex_mat @ qv))
        for j in order:
            if ex_ids[j] != q_iid:                              # exclude self -> no leakage
                return ex_ids[j], ex_diff[ex_ids[j]]
        return None, ""

    traces = load_traces(traces_p)
    id2text = load_node_texts(nodes_p)
    insts = {t["instance_id"]: t for t in load_instances(dataset, split, limit=0)}
    ids = [i for i in traces if i in insts][:n_eval]
    print(f"eval instances={len(ids)}", flush=True)

    rows = {"cold": [], "inject": [], "adapt": []}
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
        ex_id, ex_d = nearest(iid, t["issue"])

        cold = _gen(model, tok, [{"role": "system", "content": GEN_SYS},
                                 {"role": "user", "content": _user(t["issue"])}], device, injector, False, max_new)
        inj = _gen(model, tok, [{"role": "system", "content": GEN_SYS},
                                {"role": "user", "content": _user(t["issue"])}], device, injector, True, max_new)
        adapt = _gen(model, tok, [{"role": "system", "content": GEN_SYS},
                                  {"role": "user", "content": _user(t["issue"], ex_d)}], device, injector, True, max_new)
        for cond, txt in (("cold", cold), ("inject", inj), ("adapt", adapt)):
            rows[cond].append(_adherence(txt, gold_files, gold_syms))
        records.append({"id": iid, "exemplar_id": ex_id, "issue": t["issue"][:800],
                        "gold_symbols": gold_syms, "gold_patch": inst.get("patch", ""),
                        "cold": cold, "inject": inj, "adapt": adapt})
        a = rows["cold"][-1]; b = rows["inject"][-1]; c = rows["adapt"][-1]
        print(f"  [{k+1}/{len(ids)}] {iid:24} ex={ex_id:24} edit {a['edit_cov']:.2f}/{b['edit_cov']:.2f}/{c['edit_cov']:.2f}"
              f"  file {a['file_cov']:.2f}/{b['file_cov']:.2f}/{c['file_cov']:.2f}"
              f"  diff {a['is_diff']:.0f}/{b['is_diff']:.0f}/{c['is_diff']:.0f}", flush=True)

    if dump:
        Path(dump).parent.mkdir(parents=True, exist_ok=True)
        with open(dump, "w", encoding="utf-8") as w:
            for r in records:
                w.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\ndumped {len(records)} -> {dump}", flush=True)

    def _m(cond, kk):
        return round(sum(r[kk] for r in rows[cond]) / max(1, len(rows[cond])), 4)
    print("\n=== RETRIEVE-AND-ADAPT adherence (cold / inject / retrieve-adapt) ===")
    for m in ("edit_cov", "file_cov", "is_diff"):
        print(f"  {m:9} {_m('cold', m)} / {_m('inject', m)} / {_m('adapt', m)}")
    print("\nadapt > inject on edit_cov = giving a similar resolved diff to ADAPT moves the 4B toward"
          "\nthe right edit (the synthesis gap). Read the dump for whether patches are real, not garbage.")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Retrieve-and-adapt generation (Phase A).")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--traces", nargs="+", default=["data/swe/grounded_traces.jsonl"])
    ap.add_argument("--nodes", nargs="+", default=["artifacts/graph_growth/swe_code_candidates.jsonl"])
    ap.add_argument("--exemplars", default="data/swe/exemplars.jsonl")
    ap.add_argument("--adapter-ckpt", default="artifacts/stage_cache/adapter_code_s3.pt")
    ap.add_argument("--dataset", default="lite")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-eval", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--dump", default="")
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    run(a.model, a.traces, a.nodes, a.exemplars, a.adapter_ckpt, a.dataset, a.split,
        a.n_eval, a.max_new, dump=a.dump, device_str=a.device)


if __name__ == "__main__":
    raise SystemExit(main())
