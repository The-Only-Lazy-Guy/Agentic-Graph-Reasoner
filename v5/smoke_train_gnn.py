"""Cheap grad-flow check for the --train-gnn path (NO 4B LM, fits any box).

Validates that with injector.train_gnn=True, a loss on the adapter's injected output
backpropagates all the way into the GNN parameters (i.e. prepare_session kept the GNN
K/V in the autograd graph). This de-risks the change before spending big-box GPU time.
"""
from __future__ import annotations

import torch

from v5.adapter import GraphAttentionInjector
from v5.cross_attention import V5AttentionAdapter
from v5.gnn_encoder import RGCNEncoder
from v5.goal_encoder import GoalEncoder
from v5.training.stage4_generate import _stub_graph


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    lm_dim = 2560

    gnn = RGCNEncoder().to(dev)
    for p in gnn.parameters():
        p.requires_grad_(True)
    adapter = V5AttentionAdapter(r_plan=3, r_evidence=4, lm_hidden_dim=lm_dim).to(dev)
    goal_enc = GoalEncoder().to(dev).eval()
    inj = GraphAttentionInjector(adapter, gnn, goal_enc, device=torch.device(dev))
    inj.train_gnn = True

    node_ids = [f"n{i}" for i in range(12)]
    texts = {n: f"def f{i}(x): return x+{i}" for i, n in enumerate(node_ids)}
    text_emb = {n: torch.randn(768).tolist() for n in node_ids}
    tf = {"task_family": "code_fix", "required_slots": []}
    graph = _stub_graph(node_ids, texts, {n: "fact" for n in node_ids})

    inj.prepare_session(graph, node_ids, text_emb, tf, r_plan=3, r_evidence=4)

    print("graph_kv requires_grad:", inj._graph_kv.node_embeddings.requires_grad)

    h = torch.randn(1, lm_dim, device=dev, requires_grad=False) * 0.02
    goal = inj._goal
    hp, _, _ = adapter.run_planning(h, goal, inj._graph_kv, node_ids, r_max=3, task_frame=tf)
    he, _, _ = adapter.run_evidence(hp, goal, inj._graph_kv, node_ids, r_max=4, task_frame=tf)
    loss = he.float().pow(2).mean()
    loss.backward()

    gnn_grads = [(k, p.grad) for k, p in gnn.named_parameters()]
    have = [k for k, g in gnn_grads if g is not None and g.abs().sum().item() > 0]
    none = [k for k, g in gnn_grads if g is None]
    conv = [k for k in have if "conv" in k]
    print(f"loss={loss.item():.4f}")
    print(f"GNN params WITH nonzero grad: {len(have)}/{len(gnn_grads)}")
    print(f"  conv (message-passing) params with grad: {len(conv)}  -> {'EDGES TRAINABLE' if conv else 'NO EDGE GRAD!'}")
    if none:
        print(f"  params with grad=None: {none[:5]}{'...' if len(none) > 5 else ''}")
    ok = len(conv) > 0 and len(have) >= len(gnn_grads) - 2
    print("RESULT:", "PASS — gradients reach the GNN (incl. conv/edge weights)" if ok
          else "FAIL — GNN not in the grad path")


if __name__ == "__main__":
    main()
