"""Disk substrate for the total-memory graph: append-only JSONL WALs + float16 embedding
shards with cosine search. v1 is pragmatic (linear matmul search, whole-shard rewrite on
flush) but the INTERFACES are disk-first and stable: when a layer outgrows ~50k rows, the
same EmbStore API gets an hnswlib backend behind a flag — no caller changes.

Embedders are INJECTED as `embed_fn: dict[id->text] -> dict[id->list[float]]` so every
selftest runs without GPU/network (FakeEmbedder) while production uses mpnet-768
(v5.training.providers.RealEmbedder, the repo-wide node embedder).

  python -m v5.memory.store --selftest
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

EMB_DIM = 768                      # mpnet space — matches RealEmbedder + all node stores


def stable_id(prefix: str, text: str) -> str:
    """Content-addressed id, same convention as code_extract's sym_<sha1[:12]>."""
    return f"{prefix}_{hashlib.sha1(text.encode('utf-8', 'ignore')).hexdigest()[:12]}"


class JsonlWal:
    """Append-only record log. One json per line; read_all() replays."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return out

    def count(self) -> int:
        return len(self.read_all())


class EmbStore:
    """id-keyed embedding matrix on disk: {name}_ids.json + {name}_emb.npy (float16,
    L2-normalized rows). search = cosine via one matmul (v1 linear; ANN later)."""

    def __init__(self, root: str | Path, name: str, dim: int = EMB_DIM):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.dim = dim
        self._ids: list[str] = []
        self._pos: dict[str, int] = {}
        self._mat: np.ndarray | None = None            # float16 [N, dim], normalized
        self._load()

    @property
    def _ids_path(self) -> Path:
        return self.root / f"{self.name}_ids.json"

    @property
    def _emb_path(self) -> Path:
        return self.root / f"{self.name}_emb.npy"

    def _load(self) -> None:
        # eager load, file kept CLOSED: Windows cannot replace/unlink an open mmap, and v1
        # shard sizes are RAM-trivial. mmap returns with the hnswlib/shard upgrade (>50k).
        if self._ids_path.exists() and self._emb_path.exists():
            self._ids = json.loads(self._ids_path.read_text(encoding="utf-8"))
            self._mat = np.load(self._emb_path)
            if len(self._ids) != len(self._mat):       # torn write -> trust the shorter
                n = min(len(self._ids), len(self._mat))
                self._ids, self._mat = self._ids[:n], np.asarray(self._mat[:n])
            self._pos = {i: k for k, i in enumerate(self._ids)}

    def __len__(self) -> int:
        return len(self._ids)

    def __contains__(self, id_: str) -> bool:
        return id_ in self._pos

    @staticmethod
    def _norm(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32)
        if v.ndim == 1:
            v = v[None, :]
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

    def add(self, ids: list[str], vecs) -> int:
        """Append new ids (existing ids skipped). Flushes to disk. Returns n added."""
        vecs = self._norm(np.asarray(vecs, dtype=np.float32))
        keep = [(i, v) for i, v in zip(ids, vecs) if i not in self._pos]
        if not keep:
            return 0
        new_ids = [i for i, _ in keep]
        new_mat = np.stack([v for _, v in keep]).astype(np.float16)
        old = np.asarray(self._mat) if self._mat is not None and len(self._mat) else \
            np.zeros((0, self.dim), dtype=np.float16)
        self._mat = np.concatenate([old, new_mat], axis=0)
        self._ids.extend(new_ids)
        self._pos = {i: k for k, i in enumerate(self._ids)}
        self.flush()
        return len(new_ids)

    def flush(self) -> None:
        # atomic-ish: write tmp then replace (whole-shard rewrite is the v1 tradeoff).
        # np.save via handle — passing a path makes numpy append a second ".npy".
        tmp_e = self._emb_path.with_suffix(".npy.tmp")
        with open(tmp_e, "wb") as f:
            np.save(f, np.asarray(self._mat, dtype=np.float16))
        tmp_e.replace(self._emb_path)
        tmp_i = self._ids_path.with_suffix(".json.tmp")
        tmp_i.write_text(json.dumps(self._ids), encoding="utf-8")
        tmp_i.replace(self._ids_path)

    def get(self, ids: list[str]) -> np.ndarray:
        return np.asarray([self._mat[self._pos[i]] for i in ids], dtype=np.float32)

    def search(self, query, k: int = 8, within: list[str] | None = None) -> list[tuple[str, float]]:
        """Top-k (id, cosine). `within` restricts to a candidate id subset (concept members)."""
        if not self._ids:
            return []
        q = self._norm(query)[0]
        if within is not None:
            within = [i for i in within if i in self._pos]
            if not within:
                return []
            sub = self.get(within)
            scores = sub @ q
            order = np.argsort(-scores)[:k]
            return [(within[j], float(scores[j])) for j in order]
        scores = np.asarray(self._mat, dtype=np.float32) @ q
        order = np.argsort(-scores)[:k]
        return [(self._ids[j], float(scores[j])) for j in order]


# ── embedders (injected) ─────────────────────────────────────────────────────────

def make_fake_embedder(dim: int = EMB_DIM):
    """Deterministic hash-seeded vectors — selftests run with zero GPU/network.
    Similar texts do NOT get similar vectors (it's a hash); tests that need controlled
    geometry construct vectors directly."""
    def embed(texts: dict[str, str]) -> dict[str, list[float]]:
        out = {}
        for i, t in texts.items():
            seed = int(hashlib.sha1((t or " ").encode("utf-8", "ignore")).hexdigest()[:8], 16)
            v = np.random.RandomState(seed).randn(dim).astype(np.float32)
            out[i] = (v / (np.linalg.norm(v) + 1e-9)).tolist()
        return out
    return embed


def make_mpnet_embedder(device: str | None = None, batch: int = 64):
    """Production embedder: RealEmbedder (mpnet-768), batched."""
    import torch
    from v5.training.providers import RealEmbedder
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    emb = RealEmbedder(dev)

    def embed(texts: dict[str, str]) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        items = list(texts.items())
        for b0 in range(0, len(items), batch):
            out.update(emb.embed_nodes(dict(items[b0:b0 + batch])))
        return out
    return embed


# ── selftest ────────────────────────────────────────────────────────────────────

def _selftest() -> bool:
    import tempfile
    print("memory.store --selftest: WAL, EmbStore add/get/search/persist (no model)\n")
    with tempfile.TemporaryDirectory() as td:
        wal = JsonlWal(Path(td) / "w.jsonl")
        wal.append({"a": 1})
        wal.append({"b": "x", "u": "ünïcode"})
        rows = wal.read_all()
        assert len(rows) == 2 and rows[1]["u"] == "ünïcode", "WAL roundtrip"
        print("  [1] JsonlWal -> PASS")

        es = EmbStore(td, "t", dim=4)
        v = np.eye(3, 4, dtype=np.float32)
        assert es.add(["a", "b", "c"], v) == 3
        assert es.add(["b", "d"], np.array([[0, 1, 0, 0], [0, 0, 0, 1.0]])) == 1, "dedup"
        assert len(es) == 4 and "d" in es
        got = es.get(["b"])
        assert np.allclose(got[0, 1], 1.0, atol=1e-3), "get returns stored vec"
        top = es.search([1, 0, 0, 0], k=2)
        assert top[0][0] == "a" and top[0][1] > 0.99, f"search top {top}"
        sub = es.search([1, 0, 0, 0], k=2, within=["c", "d"])
        assert {s[0] for s in sub} == {"c", "d"} and sub[0][1] < 0.1, "within-subset search"
        es2 = EmbStore(td, "t", dim=4)                 # reload from disk
        assert len(es2) == 4 and es2.search([0, 0, 0, 1], k=1)[0][0] == "d", "persistence"
        print("  [2] EmbStore add/dedup/get/search/within/reload -> PASS")

        fe = make_fake_embedder(8)
        e1, e2 = fe({"x": "hello"}), fe({"x": "hello"})
        assert e1["x"] == e2["x"], "fake embedder deterministic"
        assert abs(float(np.linalg.norm(e1["x"])) - 1.0) < 1e-5, "normalized"
        print("  [3] fake embedder -> PASS")

        assert stable_id("impl", "abc") == stable_id("impl", "abc") != stable_id("impl", "abd")
        print("  [4] stable ids -> PASS")
    print("\n  MEMORY.STORE SELFTEST -> PASS")
    return True


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        raise SystemExit(0 if _selftest() else 1)
    ap.print_help()
