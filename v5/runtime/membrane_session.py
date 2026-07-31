"""membrane_session.py — the missing middle tier: long-term graph (persists) <-> session focus (this file,
ephemeral, rebuilt each run) <-> working memory (per-task slots, trm_wm.py). Wraps membrane_edits'
glowing_subgraph into a stateful per-session view that BOOSTS retrieval toward what the current
conversation has actually touched, without ever hard-filtering — a genuinely new topic mid-session must
still be answerable from the full long-term graph.

Two classes, and the distinction is the whole point of this file:
  SessionFocus  — a READ-ONLY view. Re-ranks long-term retrieval toward what the session touched. It holds
                  no content of its own, so it cannot answer "what did I say 3000 tokens ago".
  SessionGraph  — real STORAGE for this session's own content. What the KV cache evicts is written here as
                  a node and recalled by meaning later. This is what makes context growth a CPU cost
                  instead of a VRAM cost, which is the only reason the 6 GB ceiling is survivable:
                  a KV token costs ~144 KiB of VRAM (measured, scripts/kv eviction work), so a 16k context
                  is ~2.3 GB of cache alone. A session node is one 384-d fp32 embedding plus its text,
                  ~1.5 KiB, on the CPU. Same content, ~100x cheaper, and in the one place that isn't
                  scarce. Cap the cache, spill to the graph, recall on demand: VRAM goes FLAT in session
                  length while what the model can still reach keeps growing.
"""
from __future__ import annotations

import numpy as np


class SessionFocus:
    """A session-scoped activation state over a long-term AtomGraph. `.update(query_text)` re-seeds
    activation from the current turn; `.boost_sims(order, sims)` nudges a cosine-similarity ranking toward
    the currently-activated subgraph. Never removes a candidate — only reorders/boosts."""

    def __init__(self, graph, steps: int = 3, threshold: float = 0.1, boost: float = 0.25):
        self.g = graph
        self.steps = steps
        self.threshold = threshold
        self.boost = boost
        self.focus: set[str] = set()

    def update(self, query_text: str) -> set[str]:
        from v5.runtime.membrane_edits import glowing_subgraph
        self.focus = set(glowing_subgraph(self.g, query_text, steps=self.steps, threshold=self.threshold))
        return self.focus

    def boost_sims(self, order: list[str], sims) -> np.ndarray:
        boosted = np.asarray(sims, dtype=np.float32).copy()
        if not self.focus:
            return boosted
        for i, n in enumerate(order):
            if n in self.focus:
                boosted[i] += self.boost
        return boosted


class _SessionRanker:
    """Lightweight learned re-ranker for session-graph recall. Trained on the GSM8K stream itself:
    the correct span for a query about problem[i] is the span that was evicted containing problem[i]'s
    text. Trained once at session start, then used to re-score recall candidates beyond plain cosine.

    The interface matches what SessionGraph.recall() expects:
        rank(q_vec: Tensor[384], E_mat: Tensor[N,384]) -> Tensor[N]
    which returns per-candidate relevance weights."""

    def __init__(self, d: int = 384, hidden: int = 64):
        import torch.nn as nn
        self.net = nn.Sequential(
            nn.Linear(d * 2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self._trained = False

    def to(self, device):
        self.net = self.net.to(device)
        return self

    @property
    def device(self):
        return next(self.net.parameters()).device

    def rank(self, q, E, use_no_grad: bool = True):
        import torch
        q = q.to(self.device)
        E = E.to(self.device)
        N = E.shape[0]
        pairs = torch.cat([q.unsqueeze(0).expand(N, -1), E], dim=-1)
        if use_no_grad:
            with torch.no_grad():
                w = self.net(pairs).squeeze(-1)
        else:
            w = self.net(pairs).squeeze(-1)
        if not self._trained and use_no_grad:
            return torch.full_like(w, 0.5)
        return w

    def train(self, queries, positives, negatives, epochs: int = 30, lr: float = 3e-4):
        """Train on triplets: each query has one positive span and k negative spans.
        queries: list of [384] numpy arrays
        positives: list of [384] numpy arrays (the correct span for each query)
        negatives: list of list of [384] numpy arrays (wrong spans for each query)

        Uses a margin-based ranking loss: score(positive) > score(negative) + margin for all negatives.
        This avoids the multi-class shape issues of cross-entropy while learning the same ordering."""
        import torch
        import torch.nn as nn
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        last = float("nan")
        margin = 0.3
        for ep in range(epochs):
            tot = 0.0
            for q, pos, negs in zip(queries, positives, negatives):
                q_t = torch.from_numpy(q).to(self.device)
                p_t = torch.from_numpy(pos).to(self.device).unsqueeze(0)
                n_t = torch.from_numpy(np.stack(negs)).to(self.device)
                pos_score = self.rank(q_t, p_t, use_no_grad=False)
                neg_scores = self.rank(q_t, n_t, use_no_grad=False)
                loss = torch.relu(margin - pos_score + neg_scores).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss.detach())
            last = tot / max(1, len(queries))
        self._trained = True
        return last


class SessionGraph:
    """This session's own content, stored as a real graph on the CPU. Built for exactly one job: absorb
    what the KV cache has to evict, and give it back later when it becomes relevant again.

    Why a graph and not the flat list that was here before (trm_wm's `top_memory_list`, capped at 16
    pooled vectors): a capped sliding list is still forgetting, just slower — span 17 silently deletes
    span 1, and the only retrieval key is recency, so a fact from early in the session is unreachable no
    matter how relevant it becomes. Nodes are addressable BY MEANING and never evicted, so the horizon
    stops being "the last k spans" and becomes "everything, ranked".

    It is deliberately the SAME AtomGraph the long-term tier uses. Not for code reuse — because it makes
    the session tier and the permanent tier the same kind of object, so a span that proves worth keeping
    is promoted by copying a node between graphs rather than by converting between two memory formats.
    """

    def __init__(self, long_term=None, recency: float = 0.05, link_lt: float = 0.35, ranker=None,
                 rank_prefilter: int = 64):
        from v5.runtime.membrane import AtomGraph
        self.g = AtomGraph()
        self.lt = long_term
        self.recency = recency
        self.link_lt = link_lt
        self.n = 0
        self.order: list[str] = []
        self.ranker = ranker
        self.rank_prefilter = rank_prefilter

    def write(self, text: str, kind: str = "span", name: str | None = None,
              keep: bool = True) -> str | None:
        """Add an evicted span as a session node. `keep=False` DROPS it — the TRM controller's
        evict decision happens here, before anything costs memory: a span the controller rejects
        never enters the graph at all (the write is the activity it controls)."""
        from v5.runtime.membrane import Atom
        if not keep:
            return None
        text = (text or "").strip()
        if not text:
            return None
        nm = name or f"_s{self.n:04d}"
        self.g.add(Atom(name=nm, code=text, description=text, kind=kind, provenance="session"))
        if self.order:
            self.g.link(self.order[-1], nm, "follows")
        self.order.append(nm)
        self.n += 1
        if self.lt is not None and len(self.lt) > 0:
            try:
                M, order = self.lt.matrix()
                if len(order):
                    q = self.g.atoms[nm].emb
                    sims = M @ q
                    j = int(np.argmax(sims))
                    if float(sims[j]) >= self.link_lt:
                        self.g.add(self.lt.atoms[order[j]])
                        self.g.link(nm, order[j], "grounds")
            except Exception:
                pass
        return nm

    def recall(self, query: str, k: int = 3) -> list:
        from embedder import encode_batch
        names = [n for n in self.order if n in self.g.atoms]
        if not names or k <= 0:
            return []
        q = encode_batch([query])[0]
        E = np.stack([self.g.atoms[n].emb for n in names])
        cos = E @ q
        if self.ranker is not None:
            import torch
            dev = self.ranker.device
            sub = names if len(names) <= self.rank_prefilter else \
                [names[i] for i in np.argsort(-cos)[:self.rank_prefilter]]
            E_sub = np.stack([self.g.atoms[n].emb for n in sub])
            with torch.no_grad():
                w = self.ranker.rank(torch.from_numpy(q).to(dev), torch.from_numpy(E_sub).to(dev))
            sims = np.full(len(names), -1e9, dtype=np.float32)
            pos = {n: i for i, n in enumerate(names)}
            for j, n in enumerate(sub):
                sims[pos[n]] = float(w[j])
        else:
            sims = cos
        if self.recency and len(names) > 1:
            sims = sims + self.recency * np.linspace(0.0, 1.0, len(names), dtype=np.float32)
        idx = np.argsort(-sims)[:k]
        return [self.g.atoms[names[i]] for i in sorted(idx)]

    def recall_text(self, query: str, k: int = 3) -> str:
        return "\n".join(a.code for a in self.recall(query, k))

    def stats(self) -> dict:
        return {"spans": self.n, "nodes": len(self.g), "edges": len(self.g.edges),
                "chars": sum(len(a.code) for a in self.g.atoms.values())}
