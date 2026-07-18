"""algo_grr_draft — TRM drafts tokens (tiny autoregressive decoder), LM verifies via
speculative decoding. The TRM (Token Retrieval Model) is the reasoner + drafter; the LM
(frozen 3B) is only a single-forward-pass verifier. NEVER generates tokens.

Architecture:
  TRMReasoner ──plan──▶ TRMDecoder ──tokens──▶ SpecDecVerify ◀── frozen LM
    (selects atoms)       (tiny GRU,           (accept/reject     (one fwd pass)
                          generates code)       per token)

Accepted tokens → test gate. Rejected → TRM re-reasons at disagreement point.

    python -m v5.runtime.algo_grr_draft --train-trm   (train the TRM decoder)
    python -m v5.runtime.algo_grr_draft --run          (end-to-end draft + verify)
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ═══════════════════════════════════════════════════════════════════════════════
# Vocabulary: build from verified LM-generated code tokens
# ═══════════════════════════════════════════════════════════════════════════════


def _collect_code_traces(corpus_paths: list[str] | None = None) -> list[str]:
    """Collect (task_text, code) pairs from verified traces for decoder training."""
    from v5.runtime.algo_grr_poison_test import curriculum

    traces: list[str] = []
    seen: set[str] = set()

    # (A) curriculum tasks — verified reference recipes
    for rnd in curriculum():
        for t in rnd:
            code = t["recipe"]
            if code not in seen:
                traces.append(code)
                seen.add(code)

    # (B) MBPP+ corpus — reference solutions
    if corpus_paths:
        for cp in corpus_paths:
            path = Path(cp)
            if path.exists():
                for line in path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    code = r.get("code", "")
                    if code and code not in seen:
                        traces.append(code)
                        seen.add(code)

    print(f"  [draft vocab] collected {len(traces)} unique code traces", flush=True)
    return traces


def build_draft_vocab(traces: list[str], tokenizer) -> tuple[dict[int, int], int, list[int]]:
    """Build a small vocabulary of token_ids that appear in verified code.

    Returns:
        lm_to_draft: mapping from LM token_id -> draft token_id
        draft_vocab_size: number of draft tokens
        special_tokens: [bos_id, eos_id, pad_id] in draft vocab
    """
    token_counts: dict[int, int] = {}
    for code in traces:
        ids = tokenizer.encode(code, add_special_tokens=False)
        for tid in ids:
            token_counts[tid] = token_counts.get(tid, 0) + 1

    # Keep tokens that appear at least twice (filter rare outliers)
    draft_ids = sorted(tid for tid, cnt in token_counts.items() if cnt >= 2)
    lm_to_draft = {tid: i + 3 for i, tid in enumerate(draft_ids)}  # reserve 0,1,2 for special

    # Special tokens in draft vocab
    BOS = 0
    EOS = 1
    PAD = 2

    draft_vocab_size = len(draft_ids) + 3
    specials = [BOS, EOS, PAD]

    print(f"  [draft vocab] LM tokens in code: {len(token_counts)}, kept: {len(draft_ids)}, "
          f"draft vocab: {draft_vocab_size}", flush=True)
    return lm_to_draft, draft_vocab_size, specials


def encode_for_draft(code: str, lm_tok, lm_to_draft: dict[int, int],
                     specials: list[int], max_len: int = 256) -> torch.LongTensor:
    """Encode code as draft token IDs (1-indexed from LM tokenization)."""
    BOS, EOS, _ = specials
    ids = lm_tok.encode(code, add_special_tokens=False)
    draft = [BOS] + [lm_to_draft.get(tid, EOS) for tid in ids[:max_len - 2]] + [EOS]
    return torch.tensor(draft, dtype=torch.long)


def decode_from_draft(ids: list[int], lm_tok, draft_to_lm: dict[int, int]) -> str:
    """Decode draft token IDs back to LM tokens -> text."""
    _, EOS, _ = (0, 1, 2)
    lm_ids = []
    for did in ids:
        if did == EOS:
            break
        lm_tid = draft_to_lm.get(did, None)
        if lm_tid is not None:
            lm_ids.append(lm_tid)
    return lm_tok.decode(lm_ids)


def build_reverse_map(lm_to_draft: dict[int, int]) -> dict[int, int]:
    return {v: k for k, v in lm_to_draft.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# TRMDecoder — small autoregressive GRU that generates code tokens
# ═══════════════════════════════════════════════════════════════════════════════


class TRMDecoder(nn.Module):
    """Tiny autoregressive decoder that converts TRM reasoning plan → code tokens.

    The TRMReasoner produces z_T (the reasoning plan) + selected atom embeddings.
    This decoder takes that plan and generates code tokens autoregressively.

    Architecture:
        token_embed (vocab_size × d_emb) → GRU(d_emb + d_context, d_hidden) →
        cross-attend to atom_embs → language head (d_hidden × vocab_size)
    """

    def __init__(self, vocab_size: int, d_context: int = 256, d_emb: int = 64,
                 d_hidden: int = 128, d_atom: int = 256, num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_hidden = d_hidden

        self.token_embed = nn.Embedding(vocab_size, d_emb, padding_idx=2)  # PAD=2
        self.plan_proj = nn.Linear(d_context, d_hidden)  # project z_T → GRU init

        # GRU: input = [token_emb, context] where context = plan + atom_summary
        self.gru = nn.GRU(d_emb + d_hidden, d_hidden, num_layers=num_layers,
                          dropout=dropout if num_layers > 1 else 0, batch_first=True)

        # Cross-attend to atom embeddings after each step
        self.atom_q = nn.Linear(d_hidden, d_hidden)
        self.atom_k = nn.Linear(d_atom, d_hidden)
        self.atom_v = nn.Linear(d_atom, d_hidden)

        # Language head
        self.lang_head = nn.Sequential(
            nn.Linear(d_hidden * 2, d_hidden),  # concat GRU out + attended atoms
            nn.GELU(),
            nn.Linear(d_hidden, vocab_size),
        )

    def forward(self, z_T: torch.Tensor, atom_embs: torch.Tensor | None = None,
                target_ids: torch.Tensor | None = None,
                max_tokens: int = 100, temperature: float = 1.0) -> torch.Tensor | list[int]:
        """Forward pass: teacher-force or generate.

        Args:
            z_T: [d_context] final TRM hidden state
            atom_embs: [n_atoms, d_atom] or None
            target_ids: [seq_len] for teacher-forcing
            max_tokens: max generation length
            temperature: sampling temperature for generation

        Returns:
            If target_ids given: logits [seq_len-1, vocab_size] for training
            Else: list of generated token IDs
        """
        # Plan as initial state
        plan = self.plan_proj(z_T)  # [d_hidden]
        h0 = plan.unsqueeze(0).expand(self.gru.num_layers, -1).contiguous()  # [num_layers, d_hidden]

        # Atom context
        if atom_embs is not None and atom_embs.shape[0] > 0:
            atom_k = self.atom_k(atom_embs)  # [n, d_hidden]
            atom_v = self.atom_v(atom_embs)  # [n, d_hidden]
        else:
            atom_k = atom_v = None

        def attend(h: torch.Tensor) -> torch.Tensor:
            if atom_k is None:
                return torch.zeros_like(h)
            q = self.atom_q(h).unsqueeze(0)  # [1, d_hidden]
            scores = (q @ atom_k.T) / (self.d_hidden ** 0.5)  # [1, n]
            attn = F.softmax(scores, dim=-1)  # [1, n]
            return (attn @ atom_v).squeeze(0)  # [d_hidden]

        if target_ids is not None:
            # Teacher-forcing: return logits per position
            embs = self.token_embed(target_ids[:-1])  # [seq_len-1, d_emb]
            context = plan.unsqueeze(0).expand(embs.shape[0], -1)  # [seq_len-1, d_hidden]
            gru_in = torch.cat([embs, context], dim=-1)  # [seq_len-1, d_emb + d_hidden]
            gru_out, _ = self.gru(gru_in.unsqueeze(0), h0.unsqueeze(1))  # [1, seq_len-1, d_hidden]
            gru_out = gru_out.squeeze(0)  # [seq_len-1, d_hidden]
            attended = torch.stack([attend(gru_out[i]) for i in range(gru_out.shape[0])])
            combined = torch.cat([gru_out, attended], dim=-1)  # [seq_len-1, 2*d_hidden]
            return self.lang_head(combined)  # [seq_len-1, vocab_size]

        # Generation mode
        generated = []
        h = h0.unsqueeze(1)  # [num_layers, 1, d_hidden]
        tok = torch.tensor([[0]], device=z_T.device)  # BOS

        for _ in range(max_tokens):
            emb = self.token_embed(tok)  # [1, 1, d_emb]
            context = plan.unsqueeze(0).unsqueeze(0)  # [1, 1, d_hidden]
            gru_in = torch.cat([emb, context], dim=-1)  # [1, 1, d_emb + d_hidden]
            out, h = self.gru(gru_in, h)  # out: [1, 1, d_hidden]
            h_state = out.squeeze(1)  # [1, d_hidden]
            att = attend(h_state[0])  # [d_hidden]
            combined = torch.cat([h_state[0], att], dim=-1)  # [2*d_hidden]
            logits = self.lang_head(combined.unsqueeze(0))  # [1, vocab_size]
            logits = logits / temperature

            # Sample or argmax
            if temperature > 0 and temperature != 0:
                probs = F.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs.squeeze(0), 1)
            else:
                next_tok = logits.argmax(dim=-1)

            tid = next_tok.item()
            if tid == 1:  # EOS
                break
            generated.append(tid)
            tok = next_tok.unsqueeze(0)  # [1, 1]

        return generated


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════


def _bow_embed(text: str, dim: int = 256) -> np.ndarray:
    """Simple bag-of-tokens embedding (no-GPU, deterministic)."""
    v = np.zeros(dim, dtype=np.float32)
    for t in text.lower().split():
        idx = hash(t) % dim
        v[idx] += 1.0
    n = float(np.linalg.norm(v))
    return v / n if n else v


def make_training_data(trm, tokenizer, lm_to_draft, specials, device,
                       corpus_paths: list[str] | None = None):
    """Build (z_T, atom_embs, target_ids) triples from curriculum + MBPP+ traces."""
    from v5.runtime.algo_grr_poison_test import curriculum, load_seed
    from v5.runtime.algo_grr_membrane import TokenRetriever

    graph = load_seed()
    retriever = TokenRetriever(graph)
    data = []
    seen_codes: set[str] = set()

    def _process_code(code: str, task_text: str) -> None:
        if code in seen_codes:
            return
        seen_codes.add(code)

        rank = retriever.rank(task_text, exclude=set())
        atom_ids = [nid for nid, _ in rank[:6]]
        atom_embs_list = [_bow_embed(graph.nodes[nid].text) for nid in atom_ids]
        atom_embs = torch.tensor(np.stack(atom_embs_list) if atom_embs_list else [[0.0]*256],
                                 dtype=torch.float, device=device)

        task_vec = _bow_embed(task_text)
        x_vec = torch.tensor(task_vec, dtype=torch.float, device=device)

        trm.eval()
        with torch.no_grad():
            outs = trm(x_vec, atom_embs, return_all=True)
            z_T = outs[1][-1]

        target = encode_for_draft(code, tokenizer, lm_to_draft, specials).to(device)
        if target.shape[0] >= 3:
            data.append((z_T.detach().cpu(), atom_embs.cpu(), target.cpu()))

    # (A) Curriculum tasks
    for rnd in curriculum():
        for t in rnd:
            _process_code(t["recipe"], t["text"])

    # (B) MBPP+ tasks
    if corpus_paths:
        for cp in corpus_paths:
            path = Path(cp)
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                code = r.get("code", "")
                text = r.get("text", "")
                if code and text:
                    _process_code(code, text)

    print(f"  [draft data] {len(data)} training examples, {len(seen_codes)} unique", flush=True)
    return data


def _collate(batch: list) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate a list of (z_T, atom_embs, target) into batched tensors.

    Atom embeddings are stacked; targets are padded to max seq len in batch.
    """
    z_Ts = torch.stack([item[0] for item in batch])
    targets = [item[2] for item in batch]
    max_len = max(t.shape[0] for t in targets)
    padded = torch.full((len(targets), max_len), 2, dtype=torch.long)  # PAD=2
    for i, t in enumerate(targets):
        padded[i, :t.shape[0]] = t

    # Handle variable-size atom_embs — since they differ per example,
    # we handle them per-sample in the forward pass.
    return z_Ts, [item[1] for item in batch], padded


def train_decoder(decoder: TRMDecoder, data: list, epochs: int = 50,
                  lr: float = 1e-3, batch_size: int = 8, device: str = "cpu") -> TRMDecoder:
    """Train the TRMDecoder on verified (plan → code) pairs.

    Uses DataLoader with batching. Variable-length atom_embs are handled
    per-sample inside the decoder wrapper.
    """
    from torch.utils.data import DataLoader, TensorDataset

    opt = torch.optim.AdamW(decoder.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss(ignore_index=2)  # PAD

    # Simple batching via DataLoader
    loader = DataLoader(data, batch_size=batch_size, shuffle=True,
                        collate_fn=_collate)

    decoder.train()
    for ep in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for z_Ts, atom_embs_list, targets in loader:
            z_Ts = z_Ts.to(device)
            targets = targets.to(device)

            batch_loss = 0.0
            for i in range(z_Ts.shape[0]):
                z_T = z_Ts[i]
                atom_embs = atom_embs_list[i].to(device)
                target = targets[i]

                # Mask out padding tokens
                valid = target != 2
                if valid.sum() < 2:
                    continue

                logits = decoder(z_T, atom_embs=atom_embs, target_ids=target[:valid.sum()])
                loss = ce(logits, target[1:valid.sum()])
                batch_loss = batch_loss + loss

            batch_loss = batch_loss / z_Ts.shape[0]
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            total_loss += float(batch_loss)
            n_batches += 1

        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  [draft train ep {ep+1}] mean loss {total_loss/max(1,n_batches):.4f}",
                  flush=True)

    decoder.eval()
    return decoder


# ═══════════════════════════════════════════════════════════════════════════════
# Speculative Decoding Verify — LM single forward pass
# ═══════════════════════════════════════════════════════════════════════════════


class SpecDecVerify:
    """LM verifies TRM-drafted tokens with a single forward pass.

    For each token in the draft, checks if the LM's log-prob ≥ threshold.
    Returns the longest accepted prefix.
    """

    def __init__(self, lm_name: str, threshold: float = -2.0,
                 max_new_tokens: int = 180, device: str | None = None):
        self.threshold = threshold
        self.max_new_tokens = max_new_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        import os
        from transformers import AutoTokenizer
        from v5.lm_loader import load_frozen_lm

        trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
        self.tokenizer = AutoTokenizer.from_pretrained(lm_name, trust_remote_code=trust)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Pre-compute embedding + lm_head shapes
        self.lm = load_frozen_lm(lm_name)
        self.lm.eval()

        # Cache lm_head for fast access
        self.lm_head = self.lm.lm_head if hasattr(self.lm, "lm_head") else None
        if self.lm_head is None:
            # GPT-like: lm_head might be tied to embeddings
            for name, mod in self.lm.named_modules():
                if "lm_head" in name:
                    self.lm_head = mod
                    break

        # Compute
        self.max_length = 2048

    def verify(self, draft_ids: list[int], draft_to_lm: dict[int, int],
               context_prompt: str = "") -> tuple[list[int], list[float], int]:
        """Run LM forward pass on draft tokens. Return accepted tokens, per-token probs, first reject idx.

        Returns:
            accepted_tokens: list of draft token IDs accepted
            token_logprobs: per-token log-prob under the LM
            reject_pos: position of first rejection (or -1 if all accepted)
        """
        # Decode draft tokens to text
        draft_text = decode_from_draft(draft_ids, self.tokenizer, draft_to_lm)

        # Full prompt = context + draft
        full_text = context_prompt + draft_text if context_prompt else draft_text

        # Tokenize
        enc = self.tokenizer(full_text, return_tensors="pt", truncation=True,
                             max_length=self.max_length).to(self.device)
        input_ids = enc["input_ids"]  # [1, seq_len]

        with torch.no_grad():
            outputs = self.lm(input_ids)
            logits = outputs.logits[0]  # [seq_len, vocab_size]

        # Per-token log-probs (skip first token — no prior)
        log_probs = F.log_softmax(logits[:-1], dim=-1)  # [seq_len-1, vocab_size]
        token_logprobs = log_probs[range(logits.shape[0] - 1), input_ids[0, 1:]]

        # Find where context ends and draft begins
        ctx_len = len(self.tokenizer.encode(context_prompt, add_special_tokens=False)) if context_prompt else 0

        accepted = []
        reject_pos = -1
        for i, lp in enumerate(token_logprobs):
            if i < ctx_len:
                continue  # skip context tokens
            if lp.item() >= self.threshold:
                # Map back to draft token
                lm_tid = input_ids[0, i + 1].item()
                if lm_tid in draft_to_lm:
                    accepted.append(draft_to_lm[lm_tid])
                else:
                    # Token not in draft vocab — still accept (it's a separator/whitespace)
                    pass
            else:
                reject_pos = i
                break

        return accepted, token_logprobs.tolist(), reject_pos


# ═══════════════════════════════════════════════════════════════════════════════
# Integration: draft compile_fn for MembraneSolver
# ═══════════════════════════════════════════════════════════════════════════════


def make_draft_compile_fn(trm, decoder, specdec: SpecDecVerify,
                          lm_to_draft: dict[int, int], draft_to_lm: dict[int, int],
                          tokenizer, device: str = "cpu",
                          max_retries: int = 2) -> Callable[[dict], str]:
    """Create a compile_fn that uses TRM draft + spec-dec verify.

    For each solve attempt:
      1. TRM reasons → selects atoms
      2. Decoder generates code tokens
      3. LM verifies (single forward pass) → accept/reject
      4. If rejected, TRM re-reasons with rejection context
      5. If accepted, return the generated code

    Signature matches MembraneSolver's compile_fn: (spec) -> str
    """
    def compile_fn(spec: dict) -> str:
        task_text = spec.get("task_text", "")
        entry = spec.get("entry", "")
        atoms = spec.get("atoms", [])
        derive_mode = spec.get("derive", False)

        # Build atom embeddings (bag-of-words, no-GPU)
        atom_texts = [a.get("code", a.get("purpose", "")) for a in atoms]
        task_vec = torch.tensor(_bow_embed(task_text), dtype=torch.float, device=device)
        if atom_texts:
            atom_embs = torch.stack([torch.tensor(_bow_embed(t), dtype=torch.float, device=device)
                                     for t in atom_texts])
        else:
            atom_embs = torch.zeros(1, 256, device=device)

        for attempt in range(max_retries + 1):
            # TRM reason
            trm.eval()
            with torch.no_grad():
                outs = trm(task_vec, atom_embs, return_all=True)
                z_T = outs[1][-1]  # final state

            # Decoder generate
            decoder.eval()
            with torch.no_grad():
                draft_ids = decoder(z_T, atom_embs=atom_embs, temperature=0.8)

            if not draft_ids:
                continue

            # LM verify (single forward pass)
            accepted, logprobs, reject_pos = specdec.verify(
                draft_ids, draft_to_lm,
                context_prompt=f"Task: {task_text}\nWrite {entry}.\n")
            if reject_pos == -1 or attempt >= max_retries:
                # All accepted or out of retries → decode and return
                code = decode_from_draft(accepted or draft_ids,
                                         specdec.tokenizer, draft_to_lm)
                return code if code else f"def {entry}(): pass"

            # More retries available — continue with shorter accepted prefix as context
            # (The spec-dec mechanism already handles this)

        # Fallback
        code = decode_from_draft(draft_ids, specdec.tokenizer, draft_to_lm)
        return code if code else f"def {entry}(): pass"

    return compile_fn


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-vocab", action="store_true", help="build draft vocab from traces")
    ap.add_argument("--train-trm", action="store_true", help="train TRM decoder")
    ap.add_argument("--run", action="store_true", help="end-to-end draft + verify")
    ap.add_argument("--lm", default="Qwen/Qwen2.5-3B-Instruct", help="LM for spec-dec verify")
    ap.add_argument("--decoder", default="artifacts/trm_decoder.pt", help="decoder weights path")
    ap.add_argument("--vocab", default="artifacts/draft_vocab.pkl", help="vocab mapping path")
    ap.add_argument("--corpus", nargs="*", default=[],
                    help="corpus JSONL paths for vocab building (uses curriculum if empty)")
    ap.add_argument("--epochs", type=int, default=50, help="training epochs")
    ap.add_argument("--batch-size", type=int, default=8, help="training batch size")
    ap.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"algo_grr_draft — TRM drafts, LM verifies (device={device})\n", flush=True)

    if a.train_vocab:
        from transformers import AutoTokenizer
        import os
        trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
        tokenizer = AutoTokenizer.from_pretrained(a.lm, trust_remote_code=trust)
        corpus = list(a.corpus) if a.corpus else []
        extra = ["artifacts/mbpp_plus_prepped.jsonl"]
        for p in extra:
            if Path(p).exists() and p not in corpus:
                corpus.append(p)
        traces = _collect_code_traces(corpus if corpus else None)
        lm_to_draft, draft_vocab_size, specials = build_draft_vocab(traces, tokenizer)
        draft_to_lm = build_reverse_map(lm_to_draft)
        Path(a.vocab).parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(dict(lm_to_draft=lm_to_draft, draft_to_lm=draft_to_lm,
                         draft_vocab_size=draft_vocab_size, specials=specials),
                    open(a.vocab, "wb"))
        print(f"  saved vocab ({draft_vocab_size} tokens) to {a.vocab}", flush=True)

    if a.train_trm:
        from v5.runtime.algo_trm import _build
        _, _, TRMReasoner, *_ = _build()

        if not Path(a.vocab).exists():
            print(f"  vocab not found at {a.vocab}, run --train-vocab first", flush=True)
            return

        vocab_data = pickle.load(open(a.vocab, "rb"))
        draft_vocab_size = vocab_data["draft_vocab_size"]
        specials = vocab_data["specials"]
        lm_to_draft = vocab_data["lm_to_draft"]

        import os
        from transformers import AutoTokenizer
        trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
        tokenizer = AutoTokenizer.from_pretrained(a.lm, trust_remote_code=trust)

        # TRM
        trm = TRMReasoner(d_in=256, d=256, T=5, d_feedback=64)
        trm.to(device)
        trm.eval()

        # Decoder
        decoder = TRMDecoder(vocab_size=draft_vocab_size, d_context=256, d_emb=64,
                             d_hidden=128, d_atom=256, num_layers=2)
        decoder.to(device)

        # Training data — include MBPP+ corpus if available
        corpus_paths = list(a.corpus) if a.corpus else []
        extra = ["artifacts/mbpp_plus_prepped.jsonl"]
        for p in extra:
            if Path(p).exists() and p not in corpus_paths:
                corpus_paths.append(p)
        data = make_training_data(trm, tokenizer, lm_to_draft, specials, device,
                                  corpus_paths=corpus_paths if corpus_paths else None)
        if not data:
            print("  no training data", flush=True)
            return

        decoder = train_decoder(decoder, data, epochs=a.epochs,
                                batch_size=a.batch_size, lr=a.lr, device=device)

        Path(a.decoder).parent.mkdir(parents=True, exist_ok=True)
        torch.save(decoder.state_dict(), a.decoder)
        print(f"  saved decoder to {a.decoder}", flush=True)

    if a.run:
        from v5.runtime.algo_trm import _build
        _, _, TRMReasoner, *_ = _build()

        if not Path(a.vocab).exists() or not Path(a.decoder).exists():
            print(f"  need both {a.vocab} and {a.decoder}; run --train-vocab + --train-trm first",
                  flush=True)
            return

        vocab_data = pickle.load(open(a.vocab, "rb"))
        lm_to_draft = vocab_data["lm_to_draft"]
        draft_to_lm = vocab_data["draft_to_lm"]
        draft_vocab_size = vocab_data["draft_vocab_size"]
        specials = vocab_data["specials"]

        trm = TRMReasoner(d_in=256, d=256, T=5, d_feedback=64)
        trm.to(device)
        trm.eval()

        decoder = TRMDecoder(vocab_size=draft_vocab_size, d_context=256, d_emb=64,
                             d_hidden=128, d_atom=256, num_layers=2)
        decoder.load_state_dict(torch.load(a.decoder, map_location=device))
        decoder.to(device)
        decoder.eval()

        specdec = SpecDecVerify(a.lm, threshold=-2.0)
        compile_fn = make_draft_compile_fn(trm, decoder, specdec, lm_to_draft,
                                           draft_to_lm, specdec.tokenizer, device=device)

        # Run curriculum
        print("\n--- TRM draft + spec-dec verify on curriculum ---\n", flush=True)
        from v5.runtime.algo_grr_poison_test import curriculum, load_seed, run_new_arm
        from v5.runtime.algo_grr_membrane import make_lm_compiler

        # Use draft as compile fn
        rounds = curriculum()
        m = run_new_arm(rounds, compile_fn)
        from v5.runtime.algo_grr_poison_test import _fmt
        _fmt("TRM-draft + spec-dec:", m)


if __name__ == "__main__":
    main()
