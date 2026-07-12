"""Full trained cross-attend REFINER for the reason loop — the user's HRM idea on the validated LGGN
machinery (V5AttentionAdapter + GraphAttentionInjector + RGCNEncoder + GoalEncoder; QA-validated
refiner cos 0.70-0.74, latent traversal composition +0.12).

    retrieved subgraph --GNN--> GraphMemoryKV (node K/V)
        --> V5AttentionAdapter: L8 planning + L20 evidence recurrent blocks (R iters + exit,
            goal-conditioned)  == HRM-style recursive refinement of the retrieved subgraph
        --> refined LM hidden injected via forward hook (LM FROZEN, only adapter/GNN train)

z-wall (validated): the latent carries STRATEGY, not literal content -> the literal atom CODE STILL
goes to the LM as text (grounds the glue); this adds a STRATEGY channel the LM cross-attends to. The
adapter is trained on VERIFIED compositions (teacher-forced CE on the glue, hooks active, all
positions). This is the "graph reasons (recurrent refiner), LM realizes" half, wired to the reason
loop's retrieved subgraph.

  local smoke (no net, random tiny LM):  python -m v5.runtime.algo_reason_adapter --smoke
  train (GPU, 4B, on a harvest):         python -m v5.runtime.algo_reason_adapter --train --harvest artifacts/reason_harvest.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from graph_core import MemoryGraph


# ── reason-context -> injector inputs ─────────────────────────────────────────────

def reason_task_frame(task_name: str, atom_names) -> dict:
    """The goal the adapter conditions on. required_slots = what the composition needs (the task +
    the atoms it should reach for) -> the GoalEncoder turns this into the goal vector."""
    return {"task_family": "code_compose", "question_mode": "author",
            "required_slots": [task_name] + list(atom_names)[:8]}


def reason_subgraph(graph: MemoryGraph, node_ids, embed_fn):
    """text_embeddings {nid: [768]} for the subgraph nodes (mpnet node.text). The GNN consumes these +
    the graph topology (part_of/depend edges) to produce the node K/V the adapter attends to."""
    texts = {nid: (graph.nodes[nid].text or nid) for nid in node_ids}
    vecs = embed_fn(texts)
    return {nid: [float(x) for x in vecs[nid]] for nid in node_ids}


def subgraph_node_ids(graph: MemoryGraph, retriever=None):
    """Nodes the adapter sees: the impl atoms (retrieved candidate set) + concept nodes. At bootstrap
    scale this is the whole graph; at scale, narrow to the retrieved neighborhood."""
    ids = list(retriever.ids) if retriever is not None else \
        [nid for nid, n in graph.nodes.items() if n.node_type == "implementation"]
    ids += [nid for nid, n in graph.nodes.items() if n.node_type == "concept" and nid not in ids]
    return ids


# ── build the injector (fresh adapter/GNN/goal for the code-compose domain) ────────

def build_injector(model, device, r_plan: int = 4, r_evidence: int = 6, train_gnn: bool = False,
                   seed: int = 0):
    from v5.adapter import GraphAttentionInjector
    from v5.cross_attention import V5AttentionAdapter
    from v5.gnn_encoder import RGCNEncoder
    from v5.goal_encoder import GoalEncoder
    torch.manual_seed(seed)                    # deterministic init: a frozen GNN must reproduce at eval
    hid = model.config.hidden_size
    adapter = V5AttentionAdapter(r_plan=r_plan, r_evidence=r_evidence, lm_hidden_dim=hid).to(device)
    gnn = RGCNEncoder().to(device)
    goal = GoalEncoder().to(device)
    inj = GraphAttentionInjector(adapter, gnn, goal, device=device)
    inj.train_gnn = train_gnn
    inj.inject_all_positions = True    # TRAIN: refined strategy shifts ALL positions (teacher forcing)
    return inj, adapter, gnn


# ── one training step: teacher-forced CE on the verified glue, adapter injecting ───

def forward_ce(model, inj, graph, node_ids, text_embeddings, task_frame,
               input_ids: torch.Tensor, prompt_len: int, device) -> torch.Tensor:
    """prepare_session (GNN K/V + goal + masks for this subgraph) -> forward [prompt+code] with the
    adapter injecting at L8/L20 -> CE over the CODE tokens only. Grad flows through the frozen LM into
    the adapter (+GNN if train_gnn)."""
    inj.prepare_session(graph, node_ids, text_embeddings, task_frame,
                        r_plan=inj._r_plan, r_evidence=inj._r_evidence)
    with inj.inject(model):
        logits = model(input_ids).logits                       # [1, L, V]
    L = input_ids.shape[1]
    # next-token CE on the code span [prompt_len, L): predict token t from position t-1
    pred = logits[:, prompt_len - 1:L - 1, :].reshape(-1, logits.shape[-1]).float()
    tgt = input_ids[:, prompt_len:L].reshape(-1)
    return nn.CrossEntropyLoss()(pred, tgt)


def train_adapter(model_name: str, harvest_path: str, graph_path: str, epochs: int = 3, lr: float = 1e-4,
                  r_plan: int = 4, r_evidence: int = 6, train_gnn: bool = False,
                  out: str = "artifacts/reason_adapter.pt"):
    """Train the cross-attend adapter on a harvest of VERIFIED compositions (jsonl:
    {task, prompt, code, node_ids?}). LM frozen; adapter (+GNN) trains to inject strategy that makes
    the LM produce the composing glue."""
    import json
    import os
    from transformers import AutoTokenizer
    from v5.lm_loader import load_frozen_lm
    from v5.memory.store import make_mpnet_embedder

    trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_frozen_lm(model_name)
    for p in model.parameters():
        p.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
    embed = make_mpnet_embedder()
    graph = MemoryGraph.load_json(graph_path)
    node_ids = subgraph_node_ids(graph)
    text_embeddings = reason_subgraph(graph, node_ids, embed)

    inj, adapter, gnn = build_injector(model, dev, r_plan, r_evidence, train_gnn)
    params = list(adapter.parameters()) + (list(gnn.parameters()) if train_gnn else [])
    opt = torch.optim.AdamW(params, lr=lr)
    rows = [json.loads(l) for l in Path(harvest_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"train_adapter: {model_name} h={model.config.hidden_size} | {len(rows)} verified "
          f"compositions | {len(node_ids)} subgraph nodes | train_gnn={train_gnn}", flush=True)

    def encode(prompt):
        m = [{"role": "user", "content": prompt}]
        kw = dict(add_generation_prompt=True, return_tensors="pt", return_dict=True)
        try:
            return tok.apply_chat_template(m, enable_thinking=False, **kw)["input_ids"].to(dev)
        except TypeError:
            return tok.apply_chat_template(m, **kw)["input_ids"].to(dev)

    for ep in range(1, epochs + 1):
        losses = []
        for r in rows:
            tf = reason_task_frame(r.get("task", "compose"), r.get("used", []))
            pids = encode(r["prompt"])
            cids = tok(r["code"] + tok.eos_token, return_tensors="pt",
                       add_special_tokens=False).input_ids.to(dev)
            ids = torch.cat([pids, cids], dim=1)
            loss = forward_ce(model, inj, graph, node_ids, text_embeddings, tf, ids, pids.shape[1], dev)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad()
            losses.append(float(loss.detach()))
        print(f"[epoch {ep}] ce={sum(losses)/max(1,len(losses)):.3f}", flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"adapter": adapter.state_dict(), "gnn": gnn.state_dict()}, out)   # save GNN always
    print(f"  adapter -> {out}", flush=True)


# ── STEP 4: the payoff eval — does the STRATEGY channel add lift BEYOND the literal channel? ──────

def eval_adapter(model_name: str, graph_path: str, adapter_path: str, n_tasks: int = 42, samples: int = 8,
                 k: int = 6, min_cos: float = 0.25, seed: int = 99, r_plan: int = 4, r_evidence: int = 6,
                 hard: bool = False):
    """Matched-seed A/B on HELD-OUT compose tasks: literal-only (no hooks) vs +adapter (cross-attend
    strategy injected during generation). Same sampling seed per task per arm, so the only difference
    is the adapter -> a clean read on whether the strategy channel lifts compose/solve beyond the
    literal atom code (the z-wall question). 3-atom family called out (where strategy matters most)."""
    import contextlib
    import os
    from transformers import AutoTokenizer
    from v5.lm_loader import load_frozen_lm
    from v5.memory.store import make_mpnet_embedder
    from v5.runtime.algo_compose_tasks import _NEEDS, gen_compose_tasks, seed_atom_graph
    from v5.runtime.algo_graph_mg import MGRetriever
    from v5.runtime.algo_graph_reason import make_query, search_compose

    trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    focus = "sum_edit_distance" if hard else "sum_digitsum_prime_dists"   # hardest family (callout col)
    if not Path(graph_path).exists():
        seed_atom_graph(graph_path, hard=hard)
    model = load_frozen_lm(model_name)
    for p in model.parameters():
        p.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
    embed = make_mpnet_embedder()
    graph = MemoryGraph.load_json(graph_path)
    retr = MGRetriever(graph, embed)
    node_ids = subgraph_node_ids(graph)
    text_embeddings = reason_subgraph(graph, node_ids, embed)

    inj, adapter, gnn = build_injector(model, dev, r_plan, r_evidence, train_gnn=False)
    inj.inject_all_positions = False                 # GENERATION: inject at the anchor, not all positions
    state = torch.load(adapter_path, map_location=dev)
    adapter.load_state_dict(state["adapter"])
    if state.get("gnn"):
        gnn.load_state_dict(state["gnn"])            # matched frozen GNN the adapter trained against
    adapter.eval()
    gnn.eval()

    def encode(prompt):
        m = [{"role": "user", "content": prompt}]
        kw = dict(add_generation_prompt=True, return_tensors="pt", return_dict=True)
        try:
            return tok.apply_chat_template(m, enable_thinking=False, **kw)["input_ids"].to(dev)
        except TypeError:
            return tok.apply_chat_template(m, **kw)["input_ids"].to(dev)

    @torch.no_grad()
    def _gen(prompts, hooked):
        # ONE sequence per generate (batch=1): the adapter's goal is [1, GOAL_DIM] and does not
        # broadcast across num_return_sequences. Re-enter inject() per generate so each is grounded
        # (run-once resets on enter). Same per-sample RNG use in both arms -> matched A/B.
        out = []
        for p in prompts:
            pids = encode(p)
            ctx = inj.inject(model) if hooked else contextlib.nullcontext()
            with ctx:
                seqs = model.generate(pids, do_sample=True, temperature=0.8, top_p=0.95,
                                      num_return_sequences=1, max_new_tokens=400,
                                      pad_token_id=tok.eos_token_id)
            out.append(tok.decode(seqs[0, pids.shape[1]:], skip_special_tokens=True))
        return out

    tasks = gen_compose_tasks(n_tasks, seed, hard=hard)
    print(f"eval_adapter: {model_name} | {'HARD' if hard else 'easy'} suite | {len(tasks)} held-out "
          f"tasks (seed {seed}) | adapter {adapter_path}\n"
          f"  arm            solve            compose          {focus[:16]}", flush=True)
    agg = {"literal-only": [0, 0, 0, 0], "+adapter": [0, 0, 0, 0]}   # solve, compose, deep_comp, deep_tot
    for ti, task in enumerate(tasks):
        q = make_query(task, embed)
        deep = task.name == focus
        for arm, hooked in (("literal-only", False), ("+adapter", True)):
            if hooked:
                inj.prepare_session(graph, node_ids, text_embeddings,
                                    reason_task_frame(task.name, _NEEDS[task.name]),
                                    r_plan=r_plan, r_evidence=r_evidence)
            torch.manual_seed(2000 + ti)             # matched draws: only the adapter differs
            _c, used, ver, _tr = search_compose(task, retr, (lambda pr, h=hooked: _gen(pr, h)), q,
                                                k=k, samples=samples, min_cos=min_cos)
            a = agg[arm]
            a[0] += int(ver); a[1] += int(bool(used)); a[2] += int(deep and bool(used)); a[3] += int(deep)
    N = len(tasks)
    for arm in ("literal-only", "+adapter"):
        s, c, dc, dt = agg[arm]
        print(f"  {arm:14s} {s}/{N} ({s/N:.0%})      {c}/{N} ({c/N:.0%})      {dc}/{dt}", flush=True)
    dl = agg["+adapter"][1] / N - agg["literal-only"][1] / N
    ds = agg["+adapter"][0] / N - agg["literal-only"][0] / N
    verdict = "strategy channel HELPS" if dl > 0.02 or ds > 0.02 else \
              "no lift beyond literal (z-wall / needs train_gnn or more data)"
    print(f"\n  adapter delta: compose {dl:+.0%}, solve {ds:+.0%}  ->  {verdict}", flush=True)
    return agg


# ═══════════════════════════════════════════════════════════════════════════════
# LOCAL SMOKE (no network) — random tiny 22-layer LM + fake ids: prove prepare_session -> hooks fire
# at L8/L20 -> adapter injects -> CE backprops INTO the adapter (shapes/dtypes/grad all green)
# ═══════════════════════════════════════════════════════════════════════════════

def _smoke() -> bool:
    import tempfile
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from v5.runtime.algo_compose_tasks import seed_atom_graph
    print("algo_reason_adapter --smoke: prepare_session -> L8/L20 hooks -> adapter inject -> CE "
          "backprop into adapter (random tiny LM, no network)\n")
    dev = torch.device("cpu")
    torch.manual_seed(0)

    # tiny random LM with >20 layers (so PLANNING_LAYER=8 / EVIDENCE_LAYER=20 both exist)
    cfg = Qwen2Config(vocab_size=256, hidden_size=128, num_hidden_layers=22, num_attention_heads=4,
                      num_key_value_heads=4, intermediate_size=256, max_position_embeddings=64)
    model = Qwen2ForCausalLM(cfg).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"  tiny LM: {cfg.num_hidden_layers} layers, hidden {cfg.hidden_size} (frozen) -> PASS")

    with tempfile.TemporaryDirectory() as td:
        gp = str(Path(td) / "g.json")
        seed_atom_graph(gp)
        graph = MemoryGraph.load_json(gp)
        node_ids = subgraph_node_ids(graph)
        # 768-dim fake text embeddings (TEXT_EMBED_DIM) per node
        rng = np.random.default_rng(0)
        text_embeddings = {nid: rng.standard_normal(768).tolist() for nid in node_ids}
        tf = reason_task_frame("count_reachable", ["dijkstra"])
        assert len(node_ids) >= 4, node_ids
        print(f"  subgraph: {len(node_ids)} nodes (atoms+concept), task_frame slots="
              f"{tf['required_slots'][:3]}... -> PASS")

        inj, adapter, gnn = build_injector(model, dev, train_gnn=False)
        inj._r_plan, inj._r_evidence = 4, 6
        before = {n: p.detach().clone() for n, p in adapter.named_parameters()}

        # fake [prompt|code] ids; CE on the code half
        L, plen = 24, 12
        ids = torch.randint(0, 256, (1, L), device=dev)
        loss = forward_ce(model, inj, graph, node_ids, text_embeddings, tf, ids, plen, dev)
        assert torch.isfinite(loss), f"loss not finite: {loss}"
        print(f"  forward+inject: hooks fired {inj.get_hook_call_counts()} | CE loss={float(loss):.3f} "
              f"(finite) -> PASS")

        loss.backward()
        gnorm = sum(float(p.grad.norm()) for p in adapter.parameters() if p.grad is not None)
        n_grad = sum(1 for p in adapter.parameters() if p.grad is not None and p.grad.norm() > 0)
        assert n_grad > 0 and gnorm > 0, f"no grad reached adapter: {n_grad} params, norm {gnorm}"
        print(f"  backprop: grad reached {n_grad} adapter tensors, total grad-norm {gnorm:.3f} "
              f"(the LM->adapter path is differentiable) -> PASS")

        # one opt step actually MOVES adapter params (the training loop will learn)
        opt = torch.optim.AdamW(adapter.parameters(), lr=1e-3)
        opt.step()
        moved = sum(1 for n, p in adapter.named_parameters()
                    if p.grad is not None and not torch.equal(p, before[n]))
        assert moved > 0, "opt.step did not move any adapter param"
        print(f"  opt.step moved {moved} adapter tensors -> PASS")

    print("\n  ALGO_REASON_ADAPTER SMOKE -> PASS  (plumbing green; molab train next on a harvest)")
    return True


def main():
    ap = argparse.ArgumentParser(description="Trained cross-attend refiner for the reason loop (LGGN adapter).")
    ap.add_argument("--smoke", action="store_true", help="local no-network plumbing smoke")
    ap.add_argument("--train", action="store_true", help="train the adapter on a harvest (GPU)")
    ap.add_argument("--eval", action="store_true", help="A/B: literal-only vs +adapter on held-out tasks")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--graph", default="graphs/algo_reason.json")
    ap.add_argument("--harvest", default="artifacts/reason_harvest.jsonl")
    ap.add_argument("--adapter", default="artifacts/reason_adapter.pt")
    ap.add_argument("--n-tasks", type=int, default=42, help="held-out tasks for --eval")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--train-gnn", action="store_true")
    ap.add_argument("--hard", action="store_true", help="hard suite (inline-failing DP atoms) for --eval")
    a = ap.parse_args()
    if a.smoke:
        sys.exit(0 if _smoke() else 1)
    if a.train:
        train_adapter(a.model, a.harvest, a.graph, epochs=a.epochs, lr=a.lr, train_gnn=a.train_gnn,
                      out=a.adapter)
        return
    if a.eval:
        eval_adapter(a.model, a.graph, a.adapter, n_tasks=a.n_tasks, samples=a.samples, hard=a.hard)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
