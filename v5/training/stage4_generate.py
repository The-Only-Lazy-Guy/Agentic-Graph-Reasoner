"""Stage 4 (core) — grounded generation runtime + the actual-generation proof.

Stage 3 showed injection lowers the gold-patch NLL. Stage 4 closes the loop: actually
GENERATE a patch with the trained adapter injecting the task subgraph, vs cold (same
prompt, injection off), and measure whether grounded generation emits the right symbols/
files more. This is echo-free: the support symbols enter via the GNN/adapter (hidden-state
injection), NOT the prompt text, so the model can't copy them. It's also the deployable
inference path (assemble subgraph -> inject -> generate) that the verifier-retry loop
(needs Docker) will wrap.

  V5_LM_TRUST_REMOTE_CODE=1 python -m v5.training.stage4_generate \
    --model Qwen/Qwen3.5-4B --adapter-ckpt artifacts/stage_cache/adapter_code_s3.pt \
    --n-eval 30
"""
from __future__ import annotations

import argparse
import re
from contextlib import nullcontext
from pathlib import Path

import torch

from v5.adapter import GraphAttentionInjector
from v5.cross_attention import V5AttentionAdapter
from v5.gnn_encoder import RGCNEncoder
from v5.goal_encoder import GoalEncoder
from v5.perturbation_baseline import _StubNode
from v5.training.providers import RealEmbedder, FrozenQwenHInitProvider
from v5.graph_grower.swe_load import load_instances, patch_files
from v5.graph_grower.swe_probe import load_traces, load_node_texts, _symbol_name, _diff_lines

GEN_SYS = ("You are fixing a bug in a Python project. Produce a minimal unified diff patch "
           "that fixes the reported issue. Output ONLY the diff, in this format:\n"
           "--- a/path/file.py\n+++ b/path/file.py\n@@ -10,7 +10,7 @@\n-old\n+new")


def _stub_graph(node_ids, id2text, ntypes):
    class _G:
        nodes = {nid: _StubNode(id2text.get(nid, ""), ntypes.get(nid, "fact")) for nid in node_ids}
        edges = []
    return _G()


def _gen(model, tok, msgs, device, injector, inject: bool, max_new: int,
         constrain_symbols=None) -> str:
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                  return_dict=True).to(device)
    procs = None
    if constrain_symbols:
        from transformers import LogitsProcessorList
        from v5.graph_grower.constrained_decode import make_inpatch_processor
        plen = enc["input_ids"].shape[1]
        procs = LogitsProcessorList([make_inpatch_processor(tok, constrain_symbols, plen)])
    ctx = injector.inject(model) if inject else nullcontext()
    with ctx, torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             logits_processor=procs, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


def _adherence(gen: str, gold_files, gold_syms) -> dict:
    g = gen or ""; dl = _diff_lines(g)
    fc = sum(1 for f in gold_files if Path(f).name and Path(f).name in g) / max(1, len(gold_files))
    sc = sum(1 for s in gold_syms if s and re.search(rf"\b{re.escape(s)}\b", dl)) / max(1, len(gold_syms))
    return {"file_cov": fc, "edit_cov": sc,
            "is_diff": 1.0 if re.search(r"^@@ |^\+\+\+ ", g, re.M) else 0.0}


def run(model_name, traces_p, nodes_p, adapter_ckpt, dataset, split, n_eval, max_new,
        constrain=False, device_str=None):
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
    injector = GraphAttentionInjector(adapter, gnn, goal_enc, device=device)  # last-token (generation)

    traces = load_traces(traces_p)
    id2text = load_node_texts(nodes_p)
    insts = {t["instance_id"]: t for t in load_instances(dataset, split, limit=0)}
    ids = [i for i in traces if i in insts][:n_eval]
    print(f"eval instances={len(ids)}", flush=True)

    cold_rows, inj_rows, con_rows = [], [], []
    tf = {"task_family": "code_fix", "required_slots": []}
    for k, iid in enumerate(ids):
        t = traces[iid]; inst = insts[iid]
        support = [s for s in t["support_ids"] if s in id2text]
        if not support:
            continue
        gold_files = patch_files(inst.get("patch", ""))
        gold_syms = [n for s in support if (n := _symbol_name(id2text[s]))]
        node_ids = support[:24]
        ntypes = {s: "fact" for s in node_ids}
        text_emb = embedder.embed_nodes({s: id2text[s] for s in node_ids})
        injector.prepare_session(_stub_graph(node_ids, id2text, ntypes), node_ids, text_emb, tf,
                                 r_plan=3, r_evidence=4)
        msgs = [{"role": "system", "content": GEN_SYS},
                {"role": "user", "content": f"ISSUE:\n{t['issue'][:2000]}\n\nOutput ONLY the diff."}]
        cold = _gen(model, tok, msgs, device, injector, False, max_new)
        inj = _gen(model, tok, msgs, device, injector, True, max_new)
        cr, ir = _adherence(cold, gold_files, gold_syms), _adherence(inj, gold_files, gold_syms)
        cold_rows.append(cr); inj_rows.append(ir)
        line = (f"  [{k+1}/{len(ids)}] {iid:26} edit {cr['edit_cov']:.2f}->{ir['edit_cov']:.2f}  "
                f"file {cr['file_cov']:.2f}->{ir['file_cov']:.2f}  diff {cr['is_diff']:.0f}->{ir['is_diff']:.0f}")
        if constrain:
            con = _gen(model, tok, msgs, device, injector, True, max_new, constrain_symbols=gold_syms)
            xr = _adherence(con, gold_files, gold_syms); con_rows.append(xr)
            line += f"  |inj+con edit {xr['edit_cov']:.2f} diff {xr['is_diff']:.0f}"
        print(line, flush=True)

    def _m(rows, k):
        return round(sum(r[k] for r in rows) / max(1, len(rows)), 4)
    hdr = "cold -> injected" + (" -> inj+constrain" if constrain else "")
    print(f"\n=== STAGE 4 grounded GENERATION adherence ({hdr}) ===")
    for m in ("edit_cov", "file_cov", "is_diff"):
        c, i = _m(cold_rows, m), _m(inj_rows, m)
        extra = f" -> {_m(con_rows, m)}" if constrain else ""
        print(f"  {m:9} {c} -> {i}{extra}  (inj lift {i - c:+.4f})")
    print("\nedit_cov/file_cov lift = injection STEERS generation toward the gold symbols/files"
          "\n(echo-free: symbols enter via the graph, not the prompt). The generation-level"
          "\nconfirmation of the Stage-3 NLL win + the deployable runtime core.")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Stage 4: grounded generation runtime + adherence.")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--traces", nargs="+", default=["data/swe/grounded_traces.jsonl"])
    ap.add_argument("--nodes", nargs="+", default=["artifacts/graph_growth/swe_code_candidates.jsonl"])
    ap.add_argument("--adapter-ckpt", default="artifacts/stage_cache/adapter_code_s3.pt")
    ap.add_argument("--dataset", default="lite")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-eval", type=int, default=30)
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--constrain", action="store_true",
                    help="add a 3rd condition: injection + in-patch constrained decode (force exact symbols)")
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    run(a.model, a.traces, a.nodes, a.adapter_ckpt, a.dataset, a.split, a.n_eval, a.max_new,
        constrain=a.constrain, device_str=a.device)


if __name__ == "__main__":
    raise SystemExit(main())
