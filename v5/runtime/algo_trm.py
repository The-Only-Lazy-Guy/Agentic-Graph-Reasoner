"""TRM-over-graph (#55) — a TINY Recursive Model as the OWNED reasoner, trained from scratch, with the
graph giving it exactly what a tiny net needs (Q5). TRM (Tiny Recursive Model, ~7M, beats HRM on
ARC/Sudoku/Maze) recursively refines a latent answer + a scratchpad.

V2 (2026-07-15) — GRR-Tool expansion:
  - TRMReasoner now accepts tool_feedback vectors per step
  - Larger d (256), more steps (T=5)
  - Auxiliary prediction heads: confidence, usefulness, stop gate
  - ToolHead base class with 3-layer residual MLP
  - RetrievalHead: query vector + stop gate for iterative retrieval
  - WriteHead: latent for LM code generation
  - EdgeHead: src/dst pointers + relation logits
  - TRMWithTools orchestrator: runs TRM + tools for T steps, collects trace

  selftest (no torch-GPU):  python -m v5.runtime.algo_trm --selftest
"""
from __future__ import annotations

import argparse
import sys


def _build():
    import torch
    import torch.nn as nn

    # ═══════════════════════════════════════════════════════════════════════════
    # TRMReasoner V2 — with tool-feedback and auxiliary heads
    # ═══════════════════════════════════════════════════════════════════════════
    class TRMReasoner(nn.Module):
        """Tiny recursive planner. Point-attention over atoms; recursion refines a
        scratchpad z that re-scores the atoms each step. V2 adds tool_feedback
        injection per step and auxiliary prediction heads (confidence, usefulness,
        stop) for deep supervision on the reasoning process itself.

        d:      hidden dim of scratchpad / projections (256 recommended)
        T:      number of recursion steps (5 recommended)
        d_feedback: dimension of tool-feedback vector injected at each step
        """
        def __init__(self, d_in: int = 768, d: int = 256, T: int = 5, d_feedback: int = 64):
            super().__init__()
            self.T = T
            self.d = d
            self.d_feedback = d_feedback
            self.task_proj = nn.Linear(d_in, d)
            self.atom_proj = nn.Linear(d_in, d)
            self.z0 = nn.Parameter(torch.zeros(d))
            self.f = nn.Sequential(                     # scratchpad update: [x, ysum, z, fb]
                nn.Linear(3 * d + d_feedback, d), nn.GELU(),
                nn.Linear(d, d),
            )
            self.q = nn.Sequential(                     # query to score atoms: [x, z]
                nn.Linear(2 * d, d), nn.GELU(),
                nn.Linear(d, d),
            )
            self.scale = d ** 0.5
            # auxiliary prediction heads (deep supervision targets)
            self.conf_head = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))
            self.use_head  = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))
            self.stop_head = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))

        def forward(self, x_vec, atom_vecs, tool_feedback=None, return_all=False):
            """Forward pass.

            Args:
                x_vec: [d_in] task embedding
                atom_vecs: [n, d_in] atom embeddings
                tool_feedback: [T, d_feedback] or None — per-step feedback injected
                    into the scratchpad update. If None, zeros are used.
                return_all: if True returns (outs, zs, aux) where:
                    outs: [T, n] per-step atom logits
                    zs: [T, d] per-step scratchpad states
                    aux: (conf [T], use [T], stop [T]) auxiliary predictions
                    if False (default): returns final atom logits [n] (backward compat)

            For backward compatibility: when tool_feedback is None, return_all=True
            returns outs as a list of [n] tensors (old format).
            """
            x = self.task_proj(x_vec)
            A = self.atom_proj(atom_vecs)
            z = self.z0
            y = torch.zeros(A.shape[0], device=A.device)
            outs, zs = [], []
            confs, uses, stops = [], [], []
            fb_device = A.device

            for t in range(self.T):
                fb = tool_feedback[t] if tool_feedback is not None else torch.zeros(self.d_feedback, device=fb_device)
                ysoft = torch.softmax(y, dim=0)
                ysum = ysoft @ A
                z = self.f(torch.cat([x, ysum, z, fb]))
                query = self.q(torch.cat([x, z]))
                y = (A @ query) / self.scale
                outs.append(y)
                zs.append(z)
                confs.append(self.conf_head(z))
                uses.append(self.use_head(z))
                stops.append(self.stop_head(z))

            if not return_all:
                return outs[-1]
            # return_all=True: always return full trace
            return torch.stack(outs), torch.stack(zs), (
                torch.stack(confs).squeeze(-1),
                torch.stack(uses).squeeze(-1),
                torch.stack(stops).squeeze(-1),
            )

    # ═══════════════════════════════════════════════════════════════════════════
    # Tool Head — base class with 3-layer residual MLP
    # ═══════════════════════════════════════════════════════════════════════════
    class ToolHead(nn.Module):
        """3-layer residual MLP tool head. Maps a state vector (typically the
        concatenation of [x_task, z_scratchpad, ysum_atom_summary]) to:

          action_vec:  the tool-specific output (query, write latent, edge logits, …)
          feedback_vec: fed back into the next TRM recursion step's z update

        Subclasses add dedicated output heads on top.
        """
        def __init__(self, d_state: int, d_action: int, d_feedback: int, hidden: int | None = None):
            super().__init__()
            h = hidden or max(d_state, d_action + d_feedback)
            self.shared = nn.Sequential(
                nn.Linear(d_state, h), nn.GELU(),
                nn.Linear(h, h), nn.GELU(),
            )
            self.action_out = nn.Linear(h, d_action)
            self.feedback_out = nn.Linear(h, d_feedback)

        def forward(self, state):
            h = self.shared(state)
            return self.action_out(h), self.feedback_out(h)

    # ═══════════════════════════════════════════════════════════════════════════
    # Retrieval Head — drives iterative retrieval
    # ═══════════════════════════════════════════════════════════════════════════
    class RetrievalHead(nn.Module):
        """Produces a retrieval query vector + a binary stop gate + feedback.

        The stop gate decides whether to continue retrieving (another hop) or
        proceed to the next reasoning step. During training the stop target is
        'did the retrieved atoms help solve the task?' — forcing z to encode
        whether the current atom set is sufficient.

        Input state dim = 3*d (x + z + ysum).
        d_emb = query embedding dim (typically matches mpnet=768).
        """
        def __init__(self, d_state: int, d_emb: int = 256, d_feedback: int = 64):
            super().__init__()
            h = d_state
            self.shared = nn.Sequential(
                nn.Linear(d_state, h), nn.GELU(),
                nn.Linear(h, h), nn.GELU(),
            )
            self.query_out = nn.Linear(h, d_emb)
            self.stop_out = nn.Linear(h, 1)
            self.feedback_out = nn.Linear(h, d_feedback)

        def forward(self, state):
            h = self.shared(state)
            return (
                self.query_out(h),
                self.stop_out(h),
                self.feedback_out(h),
            )

    # ═══════════════════════════════════════════════════════════════════════════
    # Write Head — latent for LM code generation
    # ═══════════════════════════════════════════════════════════════════════════
    class WriteHead(nn.Module):
        """Produces a write latent (fed to LM via prompt to generate node code) +
        a node pointer (which atom to write to, if updating an existing node) +
        feedback.

        Input state dim = 3*d (x + z + ysum).
        d_emb = write latent dim (fed into LM prompt).
        """
        def __init__(self, d_state: int, d_emb: int = 256, d_feedback: int = 64):
            super().__init__()
            h = d_state
            self.shared = nn.Sequential(
                nn.Linear(d_state, h), nn.GELU(),
                nn.Linear(h, h), nn.GELU(),
            )
            self.latent_out = nn.Linear(h, d_emb)
            self.pointer_out = nn.Linear(h, 1)     # logit: write new vs update existing
            self.feedback_out = nn.Linear(h, d_feedback)

        def forward(self, state):
            h = self.shared(state)
            return (
                self.latent_out(h),
                self.pointer_out(h),
                self.feedback_out(h),
            )

    # ═══════════════════════════════════════════════════════════════════════════
    # Edge Head — produces edge proposals (src, dst, relation)
    # ═══════════════════════════════════════════════════════════════════════════
    class EdgeHead(nn.Module):
        """Produces edge logits: source atom pointer, destination atom pointer,
        and relation type logits (part_of, depend, related) + feedback.

        Input state dim = 3*d (x + z + ysum).
        n_relations = relation types (default 3: part_of, depend, related).
        During training, targets are the known correct edges for the task.
        """
        def __init__(self, d_state: int, n_atoms_emb: int = 256, n_relations: int = 3, d_feedback: int = 64):
            super().__init__()
            h = d_state
            self.shared = nn.Sequential(
                nn.Linear(d_state, h), nn.GELU(),
                nn.Linear(h, h), nn.GELU(),
            )
            self.src_out = nn.Linear(h, n_atoms_emb)       # pointer to source atom
            self.dst_out = nn.Linear(h, n_atoms_emb)       # pointer to destination atom
            self.rel_out = nn.Linear(h, n_relations)
            self.feedback_out = nn.Linear(h, d_feedback)

        def forward(self, state):
            h = self.shared(state)
            return (
                self.src_out(h),
                self.dst_out(h),
                self.rel_out(h),
                self.feedback_out(h),
            )

    # ═══════════════════════════════════════════════════════════════════════════
    # TRMWithTools — orchestrator
    # ═══════════════════════════════════════════════════════════════════════════
    class TRMWithTools(nn.Module):
        """Orchestrates TRMReasoner + optional tool heads for T steps.

        At each step t:
          1. TRM reasons: atom-pointer + scratchpad update (+ feedback from step t-1)
          2. Auxiliary heads: confidence, usefulness, stop predictions
          3. Each tool head produces tool-specific action + feedback vector
          4. Feedback vectors sum and flow into next TRM step

        Returns: (final_atom_logits, trace) where trace is a list of per-step dicts.

        For training, deep supervision applies at every step:
          - BCE on atom-pointer logits (supervised: which atoms to use)
          - BCE on stop gate (supervised: continue vs halt)
          - MSE on write latent / edge logits (supervised: ground-truth edges)
          - BCE on confidence / usefulness (supervised: did the step help?)
        """
        def __init__(self, trm: TRMReasoner,
                     ret_head: RetrievalHead | None = None,
                     write_head: WriteHead | None = None,
                     edge_head: EdgeHead | None = None,
                     retrieve_fn: callable | None = None):
            """Orchestrator.

            Args:
                trm: TRMReasoner instance
                ret_head: optional RetrievalHead
                write_head: optional WriteHead
                edge_head: optional EdgeHead
                retrieve_fn: optional callable(query_vec_np) -> feedback_vec_np.
                    Called when ret_head produces a query. The returned feedback
                    vector is added to the tool feedback for the next TRM step.
                    Expected signature: numpy array [d_emb] -> numpy array [d_feedback]
            """
            super().__init__()
            self.trm = trm
            self.ret_head = ret_head
            self.write_head = write_head
            self.edge_head = edge_head
            self.retrieve_fn = retrieve_fn
            self.d_fb = trm.d_feedback

        def forward(self, x_vec, atom_vecs, return_all=False):
            x = self.trm.task_proj(x_vec)
            A = self.trm.atom_proj(atom_vecs)
            z = self.trm.z0
            y = torch.zeros(A.shape[0], device=A.device)
            fb = torch.zeros(self.d_fb, device=A.device)

            trace = []
            for t in range(self.trm.T):
                # ── TRM step ──────────────────────────────────────────────
                ysoft = torch.softmax(y, dim=0)
                ysum = ysoft @ A
                z = self.trm.f(torch.cat([x, ysum, z, fb]))
                query = self.trm.q(torch.cat([x, z]))
                y = (A @ query) / self.trm.scale

                # auxiliary predictions
                conf = self.trm.conf_head(z).squeeze(-1)
                use = self.trm.use_head(z).squeeze(-1)
                stop_gate = self.trm.stop_head(z).squeeze(-1)

                step = {"z": z.clone(), "y": y.clone(), "conf": conf, "use": use, "stop": stop_gate}

                # ── tool heads ────────────────────────────────────────────
                fb = torch.zeros(self.d_fb, device=A.device)
                state = torch.cat([x, z, ysum])

                if self.ret_head is not None:
                    q_vec, s_logit, fb_r = self.ret_head(state)
                    step["query"] = q_vec
                    step["ret_stop"] = s_logit.squeeze(-1)
                    # Execute retrieval if a retrieve_fn is provided
                    if self.retrieve_fn is not None:
                        q_np = q_vec.cpu().detach().numpy()
                        fb_extra = self.retrieve_fn(q_np)
                        fb_extra_t = torch.as_tensor(fb_extra, dtype=z.dtype, device=z.device)
                        step["retrieve_feedback"] = fb_extra_t
                        fb = fb + fb_extra_t
                    fb = fb + fb_r

                if self.write_head is not None:
                    w_latent, w_ptr, fb_w = self.write_head(state)
                    step["write_latent"] = w_latent
                    step["write_pointer"] = w_ptr
                    fb = fb + fb_w

                if self.edge_head is not None:
                    src_l, dst_l, rel_l, fb_e = self.edge_head(state)
                    step["edge_src"] = src_l
                    step["edge_dst"] = dst_l
                    step["edge_rel"] = rel_l
                    fb = fb + fb_e

                step["feedback"] = fb.clone()
                trace.append(step)

            if return_all:
                return y, trace
            return y

    return torch, nn, TRMReasoner, ToolHead, RetrievalHead, WriteHead, EdgeHead, TRMWithTools


def train_trm(traces, atom_vecs, n_atoms, d_in=768, d=64, T=3, steps=300, lr=5e-3, seed=0):
    """Supervised from scratch: trace = (task_vec, target_atom_idx_set). Deep supervision — BCE on the
    atom-selection logits at EVERY recursion step (TRM's recipe). Returns the trained planner.

    Uses the V1-compatible TRMReasoner (no tool_feedback).
    """
    import torch.nn as nn
    torch, _nnu, TRMReasoner, *_ = _build()
    torch.manual_seed(seed)
    model = TRMReasoner(d_in=d_in, d=d, T=T)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    A = torch.as_tensor(atom_vecs, dtype=torch.float32)
    import random
    rng = random.Random(seed)
    for step in range(steps):
        tv, tgt = traces[rng.randrange(len(traces))]
        x = torch.as_tensor(tv, dtype=torch.float32)
        target = torch.zeros(n_atoms)
        for i in tgt:
            target[i] = 1.0
        outs, _zs, _aux = model(x, A, return_all=True)
        loss = sum(bce(o, target) for o in outs) / len(outs)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# SELFTEST
# ═══════════════════════════════════════════════════════════════════════════════

def _selftest() -> bool:
    torch, _nn, TRMReasoner, ToolHead, RetrievalHead, WriteHead, EdgeHead, TRMWithTools = _build()
    import numpy as np
    print("algo_trm --selftest: V2 TRM + tool heads + orchestrator\n")

    d_in, d, T, d_fb = 32, 48, 4, 16
    n_atoms = 6

    # ── [1] TRMReasoner V2 builds ──────────────────────────────────────────
    trm = TRMReasoner(d_in=d_in, d=d, T=T, d_feedback=d_fb)
    n_trm = sum(p.numel() for p in trm.parameters())
    print(f"  [1] TRMReasoner V2: {n_trm} params (d={d}, T={T}, d_fb={d_fb}) -> PASS")

    # ── [2] forward pass with tool_feedback ────────────────────────────────
    atom_vecs = torch.randn(n_atoms, d_in)
    x_vec = torch.randn(d_in)
    fb = torch.randn(T, d_fb)
    outs, zs, (confs, uses, stops) = trm(x_vec, atom_vecs, tool_feedback=fb, return_all=True)
    assert outs.shape == (T, n_atoms), f"outs shape {outs.shape}"
    assert zs.shape == (T, d), f"zs shape {zs.shape}"
    assert confs.shape == (T,) and uses.shape == (T,) and stops.shape == (T,)
    print(f"  [2] forward with tool_feedback: outs {list(outs.shape)} zs {list(zs.shape)} "
          f"aux ({list(confs.shape)},...) -> PASS")

    # ── [3] forward without tool_feedback ──────────────────────────────────
    outs_no, zs_no, aux_no = trm(x_vec, atom_vecs, return_all=True)
    assert len(outs_no) == T, f"outs len {len(outs_no)}"
    assert zs_no.shape == (T, d), f"zs shape {zs_no.shape}"
    print(f"  [3] forward without tool_feedback: {len(outs_no)} outs, "
          f"zs {list(zs_no.shape)}, aux present -> PASS")

    # ── [4] ToolHead base ──────────────────────────────────────────────────
    d_state = 3 * d
    th = ToolHead(d_state, d_action=16, d_feedback=d_fb)
    state = torch.randn(d_state)
    action, fb_out = th(state)
    assert action.shape == (16,), f"action shape {action.shape}"
    assert fb_out.shape == (d_fb,), f"feedback shape {fb_out.shape}"
    print(f"  [4] ToolHead: action {list(action.shape)} feedback {list(fb_out.shape)} -> PASS")

    # ── [5] RetrievalHead ──────────────────────────────────────────────────
    rh = RetrievalHead(d_state, d_emb=8, d_feedback=d_fb)
    q_vec, s_logit, fb_r = rh(state)
    assert q_vec.shape == (8,), f"query shape {q_vec.shape}"
    assert s_logit.shape == (1,), f"stop shape {s_logit.shape}"
    assert fb_r.shape == (d_fb,), f"feedback shape {fb_r.shape}"
    print(f"  [5] RetrievalHead: query {list(q_vec.shape)} stop {list(s_logit.shape)} -> PASS")

    # ── [6] WriteHead ──────────────────────────────────────────────────────
    wh = WriteHead(d_state, d_emb=16, d_feedback=d_fb)
    w_latent, w_ptr, fb_w = wh(state)
    assert w_latent.shape == (16,), f"write latent shape {w_latent.shape}"
    assert w_ptr.shape == (1,), f"write pointer shape {w_ptr.shape}"
    print(f"  [6] WriteHead: latent {list(w_latent.shape)} pointer {list(w_ptr.shape)} -> PASS")

    # ── [7] EdgeHead ───────────────────────────────────────────────────────
    eh = EdgeHead(d_state, n_atoms_emb=8, n_relations=3, d_feedback=d_fb)
    src_l, dst_l, rel_l, fb_e = eh(state)
    assert src_l.shape == (8,), f"src shape {src_l.shape}"
    assert dst_l.shape == (8,), f"dst shape {dst_l.shape}"
    assert rel_l.shape == (3,), f"rel shape {rel_l.shape}"
    print(f"  [7] EdgeHead: src {list(src_l.shape)} dst {list(dst_l.shape)} rel {list(rel_l.shape)} -> PASS")

    # ── [8] TRMWithTools orchestrator ──────────────────────────────────────
    tool_trm = TRMWithTools(trm, ret_head=rh, write_head=wh, edge_head=eh)
    y_final, trace = tool_trm(x_vec, atom_vecs, return_all=True)
    assert y_final.shape == (n_atoms,), f"final y shape {y_final.shape}"
    assert len(trace) == T, f"trace length {len(trace)}"
    for i, step in enumerate(trace):
        assert "z" in step and "y" in step and "conf" in step
        assert "query" in step and "ret_stop" in step
        assert "write_latent" in step and "write_pointer" in step
        assert "edge_src" in step and "edge_dst" in step and "edge_rel" in step
        assert "feedback" in step
    print(f"  [8] TRMWithTools: {len(trace)} steps, {len(step)} keys per step -> PASS")

    # ── [9] TRMWithTools without tool heads (graceful fallback) ────────────
    bare_trm = TRMWithTools(TRMReasoner(d_in=d_in, d=d, T=2, d_feedback=d_fb))
    y_bare, trace_bare = bare_trm(x_vec, atom_vecs, return_all=True)
    assert len(trace_bare) == 2
    assert all(k in trace_bare[0] for k in ("z", "y", "conf", "use", "stop", "feedback"))
    assert "query" not in trace_bare[0]
    print(f"  [9] bare TRMWithTools (no heads): {len(trace_bare)} steps, "
          f"keys = {list(trace_bare[0].keys())} -> PASS")

    # ── [10] training still works (V1 backward compat) ─────────────────────
    rng = np.random.default_rng(0)
    av = rng.standard_normal((n_atoms, d_in)).astype("float32")

    def make(n):
        data = []
        for _ in range(n):
            tgt = sorted(rng.choice(n_atoms, size=2, replace=False).tolist())
            tv = av[tgt].mean(0) + 0.6 * rng.standard_normal(d_in).astype("float32")
            data.append((tv, tgt))
        return data
    train, test = make(80), make(30)

    def recall_at2(model):
        A = torch.as_tensor(av)
        hit = tot = 0
        with torch.no_grad():
            for tv, tgt in test:
                y = model(torch.as_tensor(tv), A)
                top2 = set(torch.topk(y, 2).indices.tolist())
                hit += len(top2 & set(tgt)); tot += len(tgt)
        return hit / tot

    m = train_trm(train, av, n_atoms, d_in=d_in, d=24, T=3, steps=200)
    acc = recall_at2(m)
    assert acc > 0.70, f"planner should learn atom-selection, got recall@2={acc:.2f}"
    print(f"  [10] train_trm (V1 compat): recall@2 = {acc:.0%} on held-out -> PASS")

    print("\n  ALGO_TRM SELFTEST -> PASS  (V2 TRM + tool heads + orchestrator)")
    return True


def main():
    ap = argparse.ArgumentParser(description="TRM-over-graph: tiny recursive planner over the atom action space.")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
