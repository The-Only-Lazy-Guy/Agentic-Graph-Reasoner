"""SWE RL — SFT-then-GRPO over a LoRA, on real SWE-bench coding tasks, scored by swe_reward.

Same recipe that closed the arithmetic gap (derive_rl), now on code: per instance the frozen 4B emits
SEARCH/REPLACE patches; reward = swe_reward (applies x solves x real-edit; unapplyable hunks punished).
SFT warm-start on staged gold (DIAGNOSE-context + gold SR target), then GRPO refines the LoRA.

Architecture (NO prompt-engineering; no re-pasted megabyte prompts):
  1. Context (issue + source) is PREFILLED ONCE into a shared KV prefix (PrefixSession).
  2. DIAGNOSE runs as a terse continuation on that prefix — its tokens stay in the KV cache
     (latent state-carry; they are NOT re-serialized or re-fed as text).
  3. FIX runs as a second continuation on [context + DIAGNOSE cache] — sees the diagnosis
     latently via the KV cache, not via a re-pasted string.
  4. OperatorInjector (op-signed graph grounding) is active during every forward pass when
     --graph is set — injects task evidence via activation steering, not via prompt text.
  5. GRPO gradient is applied ONLY to FIX tokens (the emission leaf). DIAGNOSE tokens are
     latent state; they are cached greedy and not differentiated (no gradient through the
     reasoning prefix — keeps VRAM sane and isolates the policy to the code-emission step).

Two reward/eval modes are supported:
  - proxy    : cheap in-loop reward via gold-overlap (default; the practical A40 mode)
  - verifier : exact pass/fail via the real SWE verifier (Docker or hosted sb-cli)

  selftest (no model):  python -m v5.runtime.swe_rl --selftest
  proxy train (A40):    V5_LM_TRUST_REMOTE_CODE=1 python -m v5.runtime.swe_rl --sft-steps 120 --steps 120 --k 6
  staged train:         V5_LM_TRUST_REMOTE_CODE=1 python -m v5.runtime.swe_rl --staged --sft-steps 120 --steps 120 --k 6
  exact eval/reward:    ... --reward-mode verifier --verify-backend docker
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path
from contextlib import nullcontext

from v5.graph_grower.swe_verify import write_predictions
from v5.runtime.derive_rl import advantages       # reuse the proven GRPO advantage
from v5.runtime.swe_exact_verify import SWEExactVerifier
from v5.runtime.search_replace import SR_SYS, apply_sr, parse_sr
from v5.runtime.swe_reward import is_real_edit, solves_goldoverlap, swe_reward


# ── Staged KV rollout constants ───────────────────────────────────────────────
# Minimal single-line instructions — NOT megabyte re-pasted prompts.
# Context (issue + source) is prefilled ONCE; these are the glue tokens between stages.
_DIAG_SYS = (
    "You are a terse debugging assistant. Use only the retrieved source. "
    "Output only FILE/FOCUS/WHY/CHANGE with no fences."
)
_DIAG_INSTR = (
    "STEP 1 — DIAGNOSE. Output ONLY:\n"
    "FILE: <path from source>\nFOCUS: <exact function name>\n"
    "WHY: <one-sentence root cause>\nCHANGE: <one-sentence edit intent>"
)
_FIX_GLUE = (
    "<|im_end|>\n<|im_start|>user\n"
    "/no_think\nSTEP 2 — FIX. Output ONLY a search/replace block. "
    "No prose, no thinking:\npath/from/source.py\n"
    "<<<<<<< SEARCH\n<exact existing code>\n=======\n<replacement>\n>>>>>>> REPLACE"
    "<|im_end|>\n<|im_start|>assistant\n"
)


def _stub_graph(node_ids, id2text, ntypes):
    class _StubNode:
        __slots__ = ("text", "node_type", "metadata", "confidence")
        def __init__(self, text, ntype):
            self.text = text; self.node_type = ntype; self.metadata = {}; self.confidence = 0.5
    class _G:
        nodes = {nid: _StubNode(id2text.get(nid, ""), ntypes.get(nid, "fact")) for nid in node_ids}
        edges = []
    return _G()


def _trim_common_context(search: list[str], replace: list[str], keep: int = 1) -> tuple[list[str], list[str]]:
    """Trim the shared leading/trailing CONTEXT lines a hunk carries, keeping `keep` lines of anchor on
    each side. The whole-hunk SEARCH is long + hard to reproduce exactly (-> unapplyable -> reward -1)
    and bloats the SFT target past max_new; the minimal changed region + a small anchor is reproducible
    and applyable while staying unique enough to locate."""
    p = 0
    while p < len(search) and p < len(replace) and search[p] == replace[p]:
        p += 1
    s = 0
    while s < len(search) - p and s < len(replace) - p and search[-1 - s] == replace[-1 - s]:
        s += 1
    lead = max(0, p - keep)
    tail_s = len(search) - max(0, s - keep)
    tail_r = len(replace) - max(0, s - keep)
    ns, nr = search[lead:tail_s], replace[lead:tail_r]
    if not ns and not nr:
        return search, replace
    return ns, nr


def gold_to_sr(diff: str, minimal: bool = True) -> list[dict]:
    """unified diff -> SEARCH/REPLACE blocks. SEARCH = context+removed (the original), REPLACE =
    context+added (the new), per hunk; file = the +++ path. With minimal=True, trim shared context to
    the changed region + a 1-line anchor (reproducible/applyable + fits the SFT target)."""
    blocks, file, lines, i = [], None, diff.splitlines(), 0
    while i < len(lines):
        l = lines[i]
        if l.startswith("+++ "):
            file = l[4:].strip().split("\t")[0]
            if file.startswith("b/"):
                file = file[2:]
            i += 1
        elif l.startswith("@@"):
            search, replace = [], []
            i += 1
            while i < len(lines) and not lines[i].startswith(("@@", "--- ", "diff ")):
                ln = lines[i]
                if ln.startswith("+"):
                    replace.append(ln[1:])
                elif ln.startswith("-"):
                    search.append(ln[1:])
                elif ln.startswith(" "):
                    search.append(ln[1:]); replace.append(ln[1:])
                else:
                    break
                i += 1
            if file and (search or replace):
                if minimal:
                    search, replace = _trim_common_context(search, replace)
                blocks.append({"file": file, "search": "\n".join(search), "replace": "\n".join(replace)})
        else:
            i += 1
    return blocks


def sr_to_text(blocks: list[dict]) -> str:
    return "\n\n".join(
        f"{b['file']}\n<<<<<<< SEARCH\n{b['search']}\n=======\n{b['replace']}\n>>>>>>> REPLACE"
        for b in blocks
    )


def _restore_repo(repo_dir: str) -> None:
    subprocess.run(["git", "-C", repo_dir, "checkout", "--", "."], capture_output=True, text=True)


def materialize_patch(task: dict, blocks: list[dict]) -> tuple[int, str]:
    """Apply SR blocks to the checked-out repo, capture the git diff, then restore the checkout."""
    dest = task["dest"]
    blocks = [b for b in blocks if (b.get("file") or "").strip() and (b.get("search") or "").strip()]
    if not blocks:
        return 0, ""
    _restore_repo(dest)
    try:
        applied, patch = apply_sr(dest, blocks)
    finally:
        _restore_repo(dest)
    return applied, patch


def _file_text_cached(task: dict, rel: str) -> str:
    """Read+cache a repo file's text (LF-normalized) for the in-memory applyability check (no git)."""
    cache = task.setdefault("_filecache", {})
    if rel not in cache:
        fp = Path(task["dest"]) / (rel or "")
        cache[rel] = fp.read_text(encoding="utf-8", errors="ignore") if fp.is_file() else ""
    return cache[rel]


def _applyable_inmem(task: dict, blocks: list[dict]) -> int:
    """Applyability check in-memory (no git subprocess -> fast in-loop for proxy reward)."""
    applied = 0
    for b in blocks:
        f, s = (b.get("file") or "").strip(), (b.get("search") or "")
        if f and s.strip() and s in _file_text_cached(task, f):
            applied += 1
    return applied


def _score_task_blocks(task: dict, blocks: list[dict], reward_mode: str = "proxy",
                       verifier: SWEExactVerifier | None = None):
    """Score a parsed SR patch. PROXY mode is in-memory fast; VERIFIER mode runs real tests."""
    if reward_mode == "verifier":
        applied, patch = materialize_patch(task, blocks)
        grounded_ok = applied > 0 and bool(patch.strip())
    else:
        applied = _applyable_inmem(task, blocks)
        grounded_ok = applied > 0
        patch = ""
    if not grounded_ok:
        return -1.0, {"grounded": False, "why": "patch did not apply",
                      "verdict": "PUNISH (unapplyable)"}
    real = is_real_edit(blocks)
    if not real:
        return 0.0, {"grounded": True, "real_edit": False,
                     "why": "no-op patch", "verdict": "zero (NO-OP)"}
    if reward_mode == "verifier":
        if verifier is None:
            raise ValueError("reward_mode='verifier' requires an exact verifier")
        solved = verifier.verify_patch(task, patch, tag="reward")
    else:
        solved = solves_goldoverlap(blocks, task["gold"])
    if solved:
        return 1.5, {"grounded": True, "real_edit": True, "solved": True,
                     "verdict": "REWARD (applies + solves + real edit)"}
    return 0.1, {"grounded": True, "real_edit": True, "solved": False,
                 "verdict": "small (applies + real edit, does not solve)"}


def score(completion: str, task: dict, reward_mode: str = "proxy",
          verifier: SWEExactVerifier | None = None):
    blocks = parse_sr(completion)
    return _score_task_blocks(task, blocks, reward_mode=reward_mode, verifier=verifier)


def load_swe_tasks(n, traces_p, nodes_p, dataset, repo_root, src_bodies, src_lines):
    from v5.graph_grower.swe_load import load_instances, checkout_repo
    from v5.graph_grower.swe_probe import load_traces
    from v5.runtime.sr_withcode import load_symbol_meta, read_body
    traces = load_traces([traces_p]); meta = load_symbol_meta([nodes_p])
    insts = {t["instance_id"]: t for t in load_instances(dataset, "test", limit=0)}
    ids = [i for i in traces if i in insts and all(s in meta for s in traces[i]["support_ids"])]
    tasks = []
    for iid in ids[:n]:
        t = traces[iid]; inst = insts[iid]
        dest = Path(repo_root) / inst["repo"].replace("/", "__")
        ok, _ = checkout_repo(inst["repo"], inst["base_commit"], dest, timeout=1800)
        if not ok:
            continue
        support = [s for s in t["support_ids"] if s in meta]
        src = "\n\n".join(f"# {meta[s]['file']}\n{body}" for s in support[:src_bodies]
                          if (body := read_body(str(dest), meta[s]["file"], meta[s]["lineno"], src_lines)))
        gold_sr = gold_to_sr(inst.get("patch", ""))
        if not src.strip() or not gold_sr:
            continue
        gold_sr_lines = sum(
            len((b.get("search", "") or "").splitlines()) + len((b.get("replace", "") or "").splitlines())
            for b in gold_sr
        )
        tasks.append({
            "iid": iid, "issue": t["issue"], "src": src, "gold": inst.get("patch", ""),
            "gold_sr_text": sr_to_text(gold_sr), "dest": str(dest),
            "n_gold_blocks": len(gold_sr), "gold_sr_lines": gold_sr_lines,
            "node_ids": support[:24], "texts": {s: meta[s]["text"] for s in support[:24]},
            "node_types": {s: "fact" for s in support[:24]},
        })
    return tasks


def _selftest():
    print("swe_rl --selftest: gold-diff -> SR roundtrip + reward + GRPO advantage (no model).\n")
    SRC = "class A:\n    def deconstruct(self):\n        return handle_mask(self.mask)\n    def other(self):\n        pass\n"
    GOLD = (
        "--- a/x.py\n+++ b/x.py\n@@ -2,2 +2,3 @@\n     def deconstruct(self):\n"
        "-        return handle_mask(self.mask)\n"
        "+        if self.mask is None: return deepcopy(operand.mask)\n"
        "+        return handle_mask(self.mask, operand.mask)\n"
    )
    blocks = gold_to_sr(GOLD)
    print(f"  gold_to_sr -> {len(blocks)} block(s); file={blocks[0]['file'] if blocks else None}")
    text = sr_to_text(blocks)
    reparsed = parse_sr(text)
    roundtrip = len(reparsed) == len(blocks) and reparsed[0].get("search", "").strip() in SRC
    r, b = swe_reward(blocks, SRC, gold_patch=GOLD)
    advs = advantages([1.5, 1.5, -1.0, 0.0])
    ok = len(blocks) == 1 and roundtrip and r > 1.0 and abs(sum(advs)) < 1e-6
    print(f"  roundtrip (gold SR re-parses + SEARCH in source): {roundtrip}")
    print(f"  reward on the gold patch: {r:+.2f}  {b['verdict']}")
    print(f"  GRPO advantages([1.5,1.5,-1,0]) = {[round(x,2) for x in advs]} (centered={abs(sum(advs))<1e-6})")
    print(f"\n  SWE-RL SELFTEST -> {'PASS' if ok else 'FAIL'}")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# STAGED ROLLOUT HELPERS
# Context (issue + source) prefilled ONCE into KV cache via PrefixSession.
# DIAGNOSE runs greedy as a latent continuation — tokens stay in the KV.
# FIX runs K sampled continuations on [context + DIAGNOSE] — no re-paste.
# ─────────────────────────────────────────────────────────────────────────────

def _build_ctx_ids(tok, dev, issue: str, src: str):
    """Tokenize the heavy [issue + source + DIAGNOSE instruction] prefix.
    This is prefilled ONCE per rollout; all subsequent stages attend it via cache."""
    import torch
    ctx = (
        f"ISSUE:\n{issue[:1400]}\n\n"
        f"SOURCE (the bug is in here):\n{src}\n\n"
        f"{_DIAG_INSTR}"
    )
    msgs = [{"role": "system", "content": _DIAG_SYS}, {"role": "user", "content": ctx}]
    kw = dict(add_generation_prompt=True, return_tensors="pt", return_dict=True)
    try:
        enc = tok.apply_chat_template(msgs, enable_thinking=False, **kw)
    except TypeError:
        enc = tok.apply_chat_template(msgs, **kw)
    return enc["input_ids"].to(dev)


def _glue_ids(tok, dev):
    """Token IDs for the FIX-stage glue turn (the only instruction sent after DIAGNOSE).
    Short on purpose — no re-pasted source/issue, just the format instruction."""
    import torch
    return tok(_FIX_GLUE, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAIN FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def train(
    model_name, tasks, steps, K, lr, r_lora, seed, layers, eval_every, ent_coef, temperature,
    sft_steps, max_new,
    reward_mode: str = "proxy", verifier: SWEExactVerifier | None = None,
    verify_every: int = 0, verify_gold_sanity: int = 0,
    emit_preds_dir: str = "", emit_preds_every: int = 0,
    use_exemplar: bool = False, eff_coef: float = 0.0,
    graph: bool = False, adapter_ckpt: str = "", gnn_ckpt: str = "",
    train_gnn: bool = False, op_layer: int = 26, op_alpha: float = 4.0,
    staged: bool = False, diag_max_new: int = 200,
    distill: bool = False, distill_steps: int = 0, gap_coef: float = 0.0, op_mean: bool = True,
    rep_penalty: float = 1.15,
):
    """Train the SWE RL model.

    staged=True  (--staged flag):
      Context prefilled ONCE into KV cache. DIAGNOSE runs greedy as latent continuation
      (tokens in cache, not text-serialized). FIX runs K sampled continuations on [ctx+DIAGNOSE].
      GRPO gradient applied ONLY to FIX tokens. OperatorInjector active at every fwd pass.
      SFT warm-start: [ctx | gold_diag | fix_glue | gold_sr], loss on gold_sr tokens only.

    staged=False (default, --no-staged):
      Original one-shot GRPO path. Kept for A/B baseline comparison.
    """
    import random, torch, torch.nn as nn
    from transformers import AutoTokenizer
    from peft import LoraConfig, get_peft_model
    try:
        from peft import prepare_model_for_kbit_training
    except ImportError:
        prepare_model_for_kbit_training = None
    from v5.lm_loader import load_frozen_lm, resolve_dtype, resolve_quant
    from v5.runtime.swe_slot import fix_user
    if staged:
        from v5.runtime.prefix_session import PrefixSession, clone_past_key_values

    trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
    base = load_frozen_lm(model_name)
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
    dev = next(base.parameters()).device
    qmode = resolve_quant(None)
    dtype = resolve_dtype(dev)
    if qmode in ("4bit", "8bit") and prepare_model_for_kbit_training is not None:
        base = prepare_model_for_kbit_training(base)
    leaf = sorted({
        n.split(".")[-1] for n, m in base.named_modules()
        if isinstance(m, nn.Linear) and ".layers." in n
        and not any(x in n.lower() for x in ("lm_head", "embed"))
    })
    cfg = LoraConfig(
        r=r_lora, lora_alpha=2 * r_lora, lora_dropout=0.0,
        task_type="CAUSAL_LM", target_modules=leaf, layers_to_transform=layers,
    )
    model = get_peft_model(base, cfg); model.train()
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    # Gradient checkpointing is incompatible with manual KV-cache threading in staged mode.
    if not graph and not staged and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass
    trainable = [p for p in model.parameters() if p.requires_grad]

    # ── Graph cross-attention + OperatorInjector ──────────────────────────────
    # These inject task evidence via activation steering (cross-attention + op-signed vectors),
    # NOT via prompt text. Active during every forward pass (rollout + grad step).
    _make_fwd_ctx = lambda: nullcontext()
    _prep = lambda t: None
    if distill and graph:
        print("[distill] --graph ignored: distill uses the clean op-only latent channel "
              "(no random cross-attn adapter to confound the kill-test).", flush=True)
        graph = False
    if graph:
        from contextlib import ExitStack
        from v5.adapter import GraphAttentionInjector
        from v5.cross_attention import V5AttentionAdapter
        from v5.gnn_encoder import RGCNEncoder
        from v5.goal_encoder import GoalEncoder
        from v5.operator_injector import OperatorInjector, SIGN as OP_SIGN
        from v5.operator_schema import op_kind_for
        from v5.training.providers import RealEmbedder

        lm_dim = base.config.hidden_size
        embedder = RealEmbedder(torch.device("cpu"))
        gnn = RGCNEncoder().to(dev)
        if gnn_ckpt:
            gnn.load_state_dict(torch.load(gnn_ckpt, map_location=dev))
        gnn.train() if train_gnn else gnn.eval()
        for p in gnn.parameters(): p.requires_grad_(train_gnn)
        goal_enc = GoalEncoder().to(dev).eval()
        adapter = V5AttentionAdapter(r_plan=3, r_evidence=4, lm_hidden_dim=lm_dim).to(dev)
        if adapter_ckpt:
            adapter.load_state_dict(torch.load(adapter_ckpt, map_location=dev))
        adapter.train()
        injector = GraphAttentionInjector(adapter, gnn, goal_enc, device=dev)
        injector.inject_all_positions = True; injector.train_gnn = train_gnn
        inj_base = model.get_base_model() if hasattr(model, "get_base_model") else model
        op_injector = OperatorInjector(inj_base, tok, layer=op_layer, alpha=op_alpha)
        _op_v: torch.Tensor | None = None

        def _prep(t):
            injector.prepare_session(
                _stub_graph(t["node_ids"], t["texts"], t.get("node_types", {})),
                t["node_ids"], embedder.embed_nodes(t["texts"]),
                {"task_family": "code_fix", "required_slots": []},
            )
            nodes = [(t["texts"][nid], op_kind_for(t.get("node_types", {}).get(nid, "fact")))
                     for nid in t["node_ids"]]
            nonlocal _op_v; _op_v = op_injector.combine(nodes, t["issue"], normalize=True)

        def _make_fwd_ctx():
            stack = ExitStack()
            stack.enter_context(injector.inject(model))
            if _op_v is not None:
                stack.enter_context(op_injector.inject(_op_v))
            return stack

        trainable += [p for p in adapter.parameters() if p.requires_grad]
        if train_gnn:
            trainable += [p for p in gnn.parameters() if p.requires_grad]
        print(f"graph adapter: trainable {sum(p.numel() for p in adapter.parameters() if p.requires_grad):,}"
              f" | train_gnn={train_gnn}", flush=True)
        print(f"operator injector: L{op_injector.L} alpha={op_injector.alpha} | nodes={OP_SIGN}", flush=True)

    # ── Distillation latent channel (LGGN step 1) ─────────────────────────────
    # Clean latent edge-operator: OperatorInjector delta ONLY (no random cross-attn adapter to
    # confound). _op_v = op-signed ASSERT of the support nodes (content from node texts) added
    # to the residual at layer L (h <- h + delta). The frozen LM + LoRA learn to DECODE the gold
    # answer from [plain prompt + delta] to MATCH the text-grounded teacher -> "pay attention to
    # the injected context". Skipped if --graph already set up the (heavier) injector path.
    _opv = {"v": None}
    if distill and not graph:
        from v5.operator_injector import OperatorInjector, SIGN as OP_SIGN
        from v5.operator_schema import op_kind_for
        inj_base = model.get_base_model() if hasattr(model, "get_base_model") else model
        op_injector = OperatorInjector(inj_base, tok, layer=op_layer, alpha=op_alpha)

        def _prep(t):
            nodes = [(t["texts"][nid], op_kind_for(t.get("node_types", {}).get(nid, "fact")))
                     for nid in t["node_ids"]]
            _opv["v"] = op_injector.combine(nodes, t["issue"], normalize=True, mean=op_mean)

        def _make_fwd_ctx():
            return op_injector.inject(_opv["v"]) if _opv["v"] is not None else nullcontext()

        print(f"[distill] latent channel = OperatorInjector delta | L{op_injector.L} "
              f"alpha={op_injector.alpha} (op-only, no cross-attn adapter)", flush=True)

    opt = torch.optim.AdamW(trainable, lr=lr)
    print(f"LoRA r={r_lora} layers {layers} | trainable {sum(p.numel() for p in trainable):,} | tasks {len(tasks)}", flush=True)
    print(f"base: quant={qmode} dtype={dtype} device={dev} | K={K} max_new={max_new}", flush=True)
    print(f"staged={staged} | KV-prefix latent state-carry={'ON — DIAGNOSE->FIX via shared KV' if staged else 'OFF — one-shot baseline'}", flush=True)
    if qmode == "none":
        print("WARN: V5_LM_QUANT unset -> full-precision. Prefer V5_LM_QUANT=4bit on rented GPUs.", flush=True)
    if reward_mode == "verifier":
        print("reward_mode=verifier -> exact SWE tests in rollout reward (slow but accurate).", flush=True)

    rng = random.Random(seed)
    n_held = max(4, len(tasks) // 5)
    held, train_tasks = tasks[:n_held], tasks[n_held:]
    train_tasks_easy = sorted(
        train_tasks,
        key=lambda t: (t.get("n_gold_blocks", 99), t.get("gold_sr_lines", 9999), len(t["issue"]))
    )

    # retrieve-or-derive exemplar: inject each task's nearest OTHER resolved gold patch so the
    # model learns to REUSE/adapt a retrieved plan (the binding, rung-2/3).
    _ex_map: dict = {}
    if use_exemplar:
        import numpy as _np
        from v5.training.providers import RealEmbedder as _RE
        def _unit_np(v):
            a = _np.asarray(v, dtype=_np.float32); return a / (_np.linalg.norm(a) + 1e-9)
        _emb = _RE(torch.device("cpu"))
        _ids = [p["iid"] for p in tasks]
        _ev = _emb.embed_nodes({p["iid"]: p["issue"] for p in tasks})
        _mat = _np.stack([_unit_np(_ev[i]) for i in _ids])
        _gold = {p["iid"]: p.get("gold", "") for p in tasks}
        for p in tasks:
            qv = _unit_np(_ev[p["iid"]])
            for j in _np.argsort(-(_mat @ qv)):
                if _ids[j] != p["iid"]:
                    _ex_map[p["iid"]] = _gold[_ids[j]]; break
        print(f"[use-exemplar] map built for {len(_ex_map)} tasks", flush=True)

    def _ex(t):
        return _ex_map.get(t["iid"], "")

    # ── One-shot helpers (used when staged=False OR during eval) ─────────────
    def encode(system, user):
        m = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        kw = dict(add_generation_prompt=True, return_tensors="pt", return_dict=True)
        try:
            enc = tok.apply_chat_template(m, enable_thinking=False, **kw)
        except TypeError:
            enc = tok.apply_chat_template(m, **kw)
        return enc["input_ids"].to(dev)

    def gen_ids(pids, sample):
        with torch.no_grad(), _make_fwd_ctx():
            return model.generate(
                pids, do_sample=sample,
                temperature=temperature if sample else None,
                top_p=0.95 if sample else None,
                max_new_tokens=max_new,
                repetition_penalty=rep_penalty,   # curb greedy degeneration (django-11133 dup-loop);
                pad_token_id=tok.eos_token_id, use_cache=True,   # penalty-only, no no_repeat_ngram (code repeats short n-grams)
            )

    def seq_logprob_oneshot(pids, comp):
        """Log-prob of comp given pids (one-shot path)."""
        full = torch.cat([pids, comp], dim=1)
        with _make_fwd_ctx():
            logp = torch.log_softmax(model(full, use_cache=False).logits[:, :-1].float(), dim=-1)
        start = pids.shape[1] - 1
        span = logp[:, start:start + comp.shape[1]]
        sel = span.gather(-1, comp.unsqueeze(-1)).squeeze(-1).sum(-1)
        ent = -(span.exp() * span).sum(-1).mean()
        return sel, ent

    # ── Staged helpers ────────────────────────────────────────────────────────
    @torch.no_grad()
    def _staged_rollout(t, k_override=None, greedy=False):
        """Run one staged rollout for task t. Returns (fix_comps, rewards, diag_full_ids, glue).

        1. Build ctx_ids (heavy context): prefilled ONCE.
        2. DIAGNOSE greedy as latent continuation — extends the KV cache.
        3. K FIX samples as independent sampled continuations on [ctx + DIAGNOSE].
        No text is re-serialized between stages — latent state via KV.
        """
        ctx_ids = _build_ctx_ids(tok, dev, t["issue"], t["src"])
        fix_glue = _glue_ids(tok, dev)

        # Prime: prefill ctx minus the last token (the last token kicks the first gen).
        sess = PrefixSession(model, tok, dev)
        with _make_fwd_ctx():
            sess.prime(ctx_ids[:, :-1])
        kick = ctx_ids[:, -1:]

        # DIAGNOSE greedy — its tokens extend the KV prefix.
        with _make_fwd_ctx():
            _diag_new = sess.gen(kick, diag_max_new, tok.eos_token_id)

        # Snapshot the post-DIAGNOSE KV state (shared across all K FIX samples).
        diag_past = sess.past
        diag_full = sess.cur_ids  # [1, L_ctx + L_diag]

        fix_comps, rewards = [], []
        k_eval = k_override if k_override is not None else K
        for sample_i in range(k_eval):
            # Fresh session rooted at the post-DIAGNOSE KV — no re-prefill.
            fsess = PrefixSession(model, tok, dev)
            fsess.past = clone_past_key_values(diag_past)
            fsess.cur_ids = diag_full
            with _make_fwd_ctx():
                # GENERATE WITH SAMPLING so K diverse candidates emerge for GRPO
                if greedy:
                    fix_new = fsess.gen(fix_glue, max_new, tok.eos_token_id)
                else:
                    fix_new = fsess.gen_sampled(fix_glue, max_new, tok.eos_token_id, temperature=temperature)
            completion = tok.decode(fix_new, skip_special_tokens=True)
            completion = re.sub(r"<think>.*?</think>", "", completion, flags=re.DOTALL).strip()
            r = _score_task_blocks(t, parse_sr(completion), reward_mode=reward_mode, verifier=verifier)[0]
            if eff_coef > 0 and r >= 1.0:
                r += eff_coef * max(0.0, 1.0 - fix_new.shape[0] / max(1, max_new))
            fix_comps.append(fix_new); rewards.append(r)

        return fix_comps, rewards, diag_full, fix_glue

    def seq_logprob_staged(diag_full, fix_glue, fix_comp):
        """Log-prob of fix_comp given [diag_full + fix_glue] prefix.

        Single forward pass over [diag_full | fix_glue | fix_comp] with use_cache=False
        for correct gradient flow. GRPO gradient flows ONLY through fix_comp logits.
        diag_full carries the entire context + DIAGNOSE tokens — the model attends them
        latently (they are already part of the sequence, NOT re-pasted text).
        """
        glue2 = fix_glue if fix_glue.dim() == 2 else fix_glue.unsqueeze(0)
        comp2 = fix_comp.unsqueeze(0) if fix_comp.dim() == 1 else fix_comp
        full = torch.cat([diag_full, glue2, comp2], dim=1)
        with _make_fwd_ctx():
            logp = torch.log_softmax(model(full, use_cache=False).logits[:, :-1].float(), dim=-1)
        fix_start = diag_full.shape[1] + glue2.shape[1] - 1
        span = logp[:, fix_start:fix_start + comp2.shape[1]]
        sel = span.gather(-1, comp2.unsqueeze(-1)).squeeze(-1).sum(-1)
        ent = -(span.exp() * span).sum(-1).mean()
        return sel, ent

    # ── Eval / emit helpers ───────────────────────────────────────────────────
    @torch.no_grad()
    def evaluate(ts):
        """Held-out evaluation."""
        model.eval(); rs, app, solv = [], 0, 0
        for t in ts:
            if graph: _prep(t)
            if staged:
                fix_comps, _, _, _ = _staged_rollout(t, k_override=1, greedy=True)
                completion = tok.decode(fix_comps[0], skip_special_tokens=True)
            else:
                pids = encode(SR_SYS, fix_user(t["issue"], t["src"], exemplar=_ex(t)))
                out = gen_ids(pids, sample=False)
                completion = tok.decode(out[0, pids.shape[1]:], skip_special_tokens=True)
            r, b = score(completion, t, reward_mode=reward_mode, verifier=verifier)
            rs.append(r); app += int(b.get("grounded", False)); solv += int(b.get("solved", False))
        model.train()
        return sum(rs) / len(ts), app / len(ts), solv / len(ts)

    @torch.no_grad()
    def evaluate_exact(ts, tag):
        if verifier is None: return None
        model.eval()
        task_patches: list[tuple[dict, str]] = []; applyable = 0
        for t in ts:
            if graph: _prep(t)
            if staged:
                fix_comps, _, _, _ = _staged_rollout(t, k_override=1, greedy=True)
                completion = tok.decode(fix_comps[0], skip_special_tokens=True)
            else:
                pids = encode(SR_SYS, fix_user(t["issue"], t["src"], exemplar=_ex(t)))
                out = gen_ids(pids, sample=False)
                completion = tok.decode(out[0, pids.shape[1]:], skip_special_tokens=True)
            blocks = parse_sr(completion)
            applied, patch = materialize_patch(t, blocks)
            if applied > 0 and patch.strip():
                task_patches.append((t, patch)); applyable += 1
        res = verifier.verify_task_batch_unique(task_patches, tag=tag)
        model.train()
        resolved = sum(1 for t in ts if res.get(t["iid"], False))
        return resolved / len(ts), applyable / len(ts)

    def run_gold_sanity(ts, n):
        if verifier is None or n <= 0: return
        ok, total = verifier.run_gold_sanity(ts, n, tag="gold_sanity")
        print(f"[gold-sanity] resolved {ok}/{total} gold patches", flush=True)
        if ok != total:
            raise SystemExit("gold-sanity failed; refusing to trust exact SWE verifier results")

    @torch.no_grad()
    def emit_predictions(ts, tag):
        if not emit_preds_dir: return None
        model.eval(); task_patches: list[tuple[dict, str]] = []
        for t in ts:
            if graph: _prep(t)
            if staged:
                fix_comps, _, _, _ = _staged_rollout(t, k_override=1, greedy=True)
                completion = tok.decode(fix_comps[0], skip_special_tokens=True)
            else:
                pids = encode(SR_SYS, fix_user(t["issue"], t["src"], exemplar=_ex(t)))
                out = gen_ids(pids, sample=False)
                completion = tok.decode(out[0, pids.shape[1]:], skip_special_tokens=True)
            blocks = parse_sr(completion)
            applied, patch = materialize_patch(t, blocks)
            if applied > 0 and patch.strip(): task_patches.append((t, patch))
        out_dir = Path(emit_preds_dir); out_dir.mkdir(parents=True, exist_ok=True)
        pred_path = out_dir / f"{tag}.jsonl"
        n = write_predictions({task["iid"]: patch for task, patch in task_patches}, str(pred_path), model_name="swe_rl")
        model.train()
        print(f"[emit {tag}] wrote {n}/{len(ts)} held predictions -> {pred_path}", flush=True)
        return str(pred_path)

    if verifier is not None and (reward_mode == "verifier" or verify_every > 0):
        run_gold_sanity(held, min(verify_gold_sanity, len(held)))

    bm, ba, bs = evaluate(held)
    print(f"[eval @0] held mean_reward={bm:+.3f} applyable={ba:.0%} gold-solve={bs:.0%}", flush=True)
    if verifier is not None:
        exact0 = evaluate_exact(held, "verify_0")
        if exact0 is not None:
            vr, vp = exact0
            print(f"[verify @0] held exact_resolve={vr:.0%} patch_emission={vp:.0%}", flush=True)

    # ── SFT warm-start ─────────────────────────────────────────────────────────
    # staged SFT: [ctx | gold_diag_tokens | fix_glue | gold_sr_tokens]
    #   loss only on gold_sr_tokens — teaches staged format + code emission format.
    # one-shot SFT: [prompt_ids | gold_sr_tokens] — original path.
    if sft_steps > 0:
        ce = torch.nn.CrossEntropyLoss()
        sft_cap = max(max_new, 640)

        def _gold_diag_text(t: dict) -> str:
            """Minimal synthesized DIAGNOSE text from gold patch metadata."""
            sr = gold_to_sr(t["gold"])
            file = sr[0]["file"] if sr else "module.py"
            return (f"FILE: {file}\nFOCUS: (see patch)\n"
                    "WHY: fix the bug described in the issue\n"
                    "CHANGE: apply the minimal change from the gold patch")

        for s in range(1, sft_steps + 1):
            t = rng.choice(train_tasks)
            if graph: _prep(t)
            if staged:
                ctx_ids = _build_ctx_ids(tok, dev, t["issue"], t["src"])
                gold_diag_ids = tok(
                    _gold_diag_text(t), return_tensors="pt", add_special_tokens=False
                ).input_ids.to(dev)
                fix_glue = _glue_ids(tok, dev)
                gold_fix_ids = tok(
                    t["gold_sr_text"], return_tensors="pt", add_special_tokens=False
                ).input_ids.to(dev)[:, :sft_cap]
                full = torch.cat([ctx_ids, gold_diag_ids, fix_glue, gold_fix_ids], dim=1)
                diag_start = ctx_ids.shape[1]
                fix_start = ctx_ids.shape[1] + gold_diag_ids.shape[1] + fix_glue.shape[1]
                with _make_fwd_ctx():
                    logits = model(full, use_cache=False).logits
                
                loss_diag = ce(
                    logits[:, diag_start - 1:diag_start - 1 + gold_diag_ids.shape[1]].reshape(-1, logits.shape[-1]).float(),
                    gold_diag_ids.reshape(-1),
                )
                loss_fix = ce(
                    logits[:, fix_start - 1:fix_start - 1 + gold_fix_ids.shape[1]].reshape(-1, logits.shape[-1]).float(),
                    gold_fix_ids.reshape(-1),
                )
                loss = loss_diag + loss_fix
            else:
                pids = encode(SR_SYS, fix_user(t["issue"], t["src"], exemplar=_ex(t)))
                gids = tok(t["gold_sr_text"], return_tensors="pt", add_special_tokens=False).input_ids.to(dev)[:, :sft_cap]
                full = torch.cat([pids, gids], dim=1)
                with _make_fwd_ctx():
                    logits = model(full, use_cache=False).logits
                st = pids.shape[1]
                loss = ce(logits[:, st - 1:st - 1 + gids.shape[1]].reshape(-1, logits.shape[-1]).float(), gids.reshape(-1))

            loss.backward(); torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step(); opt.zero_grad()
            if s <= 3 or s % 25 == 0:
                print(f"[sft {s:3}] {t['iid']:26} ce_loss={float(loss.detach()):.3f}", flush=True)

        sm, sa, ss = evaluate(held)
        print(f"[eval after SFT] held mean_reward={sm:+.3f} applyable={sa:.0%} gold-solve={ss:.0%} "
              f"(base {bm:+.3f}/{ba:.0%}/{bs:.0%})\n", flush=True)
        if verifier is not None:
            exact_sft = evaluate_exact(held, "verify_after_sft")
            if exact_sft is not None:
                vr, vp = exact_sft
                print(f"[verify after SFT] held exact_resolve={vr:.0%} patch_emission={vp:.0%}\n", flush=True)
        if emit_preds_dir:
            emit_predictions(held, "held_after_sft")

    # ── Distillation loop (LGGN step 1: text-grounding -> latent-delta grounding) ──
    # Teacher = model WITH plan in prompt TEXT (the path proven to work), no injection.
    # Student = model with plan as latent delta injection ONLY, NO text. KL pulls student->teacher
    # on the gold answer tokens, so the only way to match is to ROUTE the injection into the
    # answer = the model learns to attend to the injected context. The text crutch is removed
    # (exemplar="") so the latent must carry the signal.
    if distill:
        _cap = max(max_new, 256)   # gold-answer token cap for KL (easy-task golds fit; bounds backward mem)

        def _gold_lp(logits, st, gtok):
            n = gtok.shape[0]
            lp = torch.log_softmax(logits[:, st - 1:st - 1 + n].float(), dim=-1)  # [1,n,V]
            return lp, lp[0].gather(-1, gtok.unsqueeze(-1)).squeeze(-1).sum()

        def _distill_step(t):
            gids = tok(t["gold_sr_text"], return_tensors="pt",
                       add_special_tokens=False).input_ids.to(dev)[:, :_cap]
            if gids.shape[1] == 0:
                return None
            gtok = gids[0]
            # TEACHER: plan in TEXT, no injection, no grad -> the distillation target
            pidsT = encode(SR_SYS, fix_user(t["issue"], t["src"], exemplar=_ex(t)))
            with torch.no_grad():
                logT = model(torch.cat([pidsT, gids], 1), use_cache=False).logits
            lpT, goldT = _gold_lp(logT, pidsT.shape[1], gtok)
            lpT = lpT.detach(); pT = lpT.exp()
            # STUDENT: NO text plan, WITH latent delta injection, grad flows
            pidsS = encode(SR_SYS, fix_user(t["issue"], t["src"], exemplar=""))
            _prep(t)
            fullS = torch.cat([pidsS, gids], 1); sS = pidsS.shape[1]
            with _make_fwd_ctx():
                logS = model(fullS, use_cache=False).logits
            lpS, goldS_on = _gold_lp(logS, sS, gtok)
            kl = (pT * (lpT - lpS)).sum(-1).mean()
            # ABLATION: same student prompt, injection OFF, no grad (does delta help recover gold?)
            with torch.no_grad():
                logOff = model(fullS, use_cache=False).logits
                _, goldS_off = _gold_lp(logOff, sS, gtok)
            loss = kl
            if gap_coef > 0:   # optional: explicitly reward injection recovering the gold answer
                loss = kl - gap_coef * (goldS_on - goldS_off.detach())
            return loss, dict(kl=float(kl.detach()), goldT=float(goldT),
                              on=float(goldS_on.detach()), off=float(goldS_off),
                              gap=float((goldS_on - goldS_off).detach()))

        @torch.no_grad()
        def _kill_test(ts, sweep=(0.0,)):
            """does the latent carry signal AND is it the RIGHT content? mean gold-logprob over held.
            sweep = effective-alpha fractions of op_alpha to try (vector scales linearly with alpha, so
            we scale the precomputed v instead of recombining). 0.0 = OFF baseline. Also reports random-
            delta at the strongest scale (content vs perturbation). Calibrates a non-destructive alpha."""
            sweep = tuple(dict.fromkeys((0.0,) + tuple(sweep)))   # always include OFF, dedup
            acc = {f: 0.0 for f in sweep}; rnd = 0.0; n = 0
            for t in ts:
                gids = tok(t["gold_sr_text"], return_tensors="pt",
                           add_special_tokens=False).input_ids.to(dev)[:, :_cap]
                if gids.shape[1] == 0:
                    continue
                gtok = gids[0]
                pidsS = encode(SR_SYS, fix_user(t["issue"], t["src"], exemplar=""))
                fullS = torch.cat([pidsS, gids], 1); sS = pidsS.shape[1]
                _prep(t)
                base_v = _opv["v"]
                for f in sweep:
                    ctx = op_injector.inject(base_v * f) if f != 0.0 else nullcontext()
                    with ctx:
                        _, g = _gold_lp(model(fullS, use_cache=False).logits, sS, gtok)
                    acc[f] += float(g)
                fmax = max(sweep)
                rv = torch.randn_like(base_v); rv = rv / (rv.norm() + 1e-6) * (base_v * fmax).norm()
                with op_injector.inject(rv):
                    _, g_rnd = _gold_lp(model(fullS, use_cache=False).logits, sS, gtok)
                rnd += float(g_rnd); n += 1
            if n:
                off = acc[0.0] / n
                parts = " ".join(f"a{op_alpha*f:g}={acc[f]/n:+.1f}" for f in sweep if f != 0.0)
                fmax = max(sweep)
                v2 = "CONTENT" if acc[fmax] / n > rnd / n else "perturbation-only"
                print(f"[kill-test] gold_lp OFF={off:+.1f} | {parts} | rand(a{op_alpha*fmax:g})={rnd/n:+.1f} "
                      f"| strongest>rand:{v2}", flush=True)
            return None

        print("[distill] kill-test BEFORE training (untrained latent baseline + alpha sweep):", flush=True)
        _kill_test(held, sweep=(0.25, 0.5, 1.0, 2.0, 4.0))
        ds = distill_steps if distill_steps > 0 else steps
        for step in range(1, ds + 1):
            t = rng.choice(train_tasks_easy)
            out = _distill_step(t)
            if out is None:
                continue
            loss, m = out
            if step == 1:
                assert loss.requires_grad, "DISTILL LOSS DETACHED — gradient will not flow"
            opt.zero_grad(set_to_none=True); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()
            if step <= 3 or step % 10 == 0:
                print(f"[distill {step:3}] {t['iid']:24} kl={m['kl']:.3f} goldT={m['goldT']:.1f} "
                      f"on={m['on']:.1f} off={m['off']:.1f} gap={m['gap']:+.1f} gnorm={float(gn):.3f}",
                      flush=True)
            if step % eval_every == 0:
                _kill_test(held, sweep=(1.0,))
        print("[distill] kill-test AFTER training:", flush=True)
        _kill_test(held, sweep=(1.0,))
        if emit_preds_dir:
            emit_predictions(held, "held_after_distill")
        return

    # ── GRPO loop ─────────────────────────────────────────────────────────────
    zero_var = 0
    for step in range(1, steps + 1):
        widen = min(1.0, step / max(1, int(0.4 * steps)))
        pool_frac = 0.25 + 0.75 * widen
        pool_n = max(1, int(len(train_tasks_easy) * pool_frac))
        t = rng.choice(train_tasks_easy[:pool_n])
        if graph: _prep(t)

        if staged:
            # ── STAGED GRPO ──────────────────────────────────────────────────
            # Context prefilled ONCE. DIAGNOSE greedy (latent state in KV).
            # K FIX samples as continuations. Gradient on FIX tokens only.
            fix_comps, rewards, diag_full, fix_glue = _staged_rollout(t)
            mean_r = sum(rewards) / K
            r_std = (sum((r - mean_r) ** 2 for r in rewards) / K) ** 0.5
            if r_std < 1e-9:
                zero_var += 1
                if step % 10 == 0:
                    print(f"[step {step:3}] {t['iid']:24} mean_r={mean_r:+.2f} r_std=0 SKIP", flush=True)
                continue
            advs = advantages(rewards)
            opt.zero_grad(set_to_none=True)
            loss_sum = ent_sum = 0.0
            for fix_comp, adv in zip(fix_comps, advs):
                lp, ent = seq_logprob_staged(diag_full, fix_glue, fix_comp)
                sample_loss = (-adv * lp - ent_coef * ent) / K
                loss_sum += float(sample_loss.detach()); ent_sum += float(ent.detach())
                sample_loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
            if step % 10 == 0:
                print(f"[step {step:3}] {t['iid']:24} pool={pool_n}/{len(train_tasks_easy)} "
                      f"mean_r={mean_r:+.2f} r_std={r_std:.2f} ent={ent_sum/K:.2f} "
                      f"loss={loss_sum:+.3f} gnorm={float(gnorm):.3f} rewards={[round(r,1) for r in rewards]}",
                      flush=True)
        else:
            # ── ONE-SHOT GRPO (A/B baseline) ─────────────────────────────────
            pids = encode(SR_SYS, fix_user(t["issue"], t["src"], exemplar=_ex(t)))
            comps, rewards = [], []
            rollout_modes = [False] + [True] * max(0, K - 1)
            for sample in rollout_modes:
                out = gen_ids(pids, sample=sample)
                comp = out[:, pids.shape[1]:]
                completion = tok.decode(comp[0], skip_special_tokens=True)
                comps.append(comp)
                r = score(completion, t, reward_mode=reward_mode, verifier=verifier)[0]
                if eff_coef > 0 and r >= 1.0:
                    r += eff_coef * max(0.0, 1.0 - comp.shape[1] / max(1, max_new))
                rewards.append(r)
            mean_r = sum(rewards) / K
            r_std = (sum((r - mean_r) ** 2 for r in rewards) / K) ** 0.5
            if r_std < 1e-9:
                zero_var += 1
                if step % 10 == 0:
                    print(f"[step {step:3}] {t['iid']:24} mean_r={mean_r:+.2f} r_std=0 SKIP", flush=True)
                continue
            advs = advantages(rewards)
            opt.zero_grad(set_to_none=True)
            loss_sum = ent_sum = 0.0
            for comp, adv in zip(comps, advs):
                lp, ent = seq_logprob_oneshot(pids, comp)
                sample_loss = (-adv * lp - ent_coef * ent) / K
                loss_sum += float(sample_loss.detach()); ent_sum += float(ent.detach())
                sample_loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
            if step % 10 == 0:
                print(f"[step {step:3}] {t['iid']:24} pool={pool_n}/{len(train_tasks_easy)} "
                      f"mean_r={mean_r:+.2f} r_std={r_std:.2f} ent={ent_sum/K:.2f} "
                      f"loss={loss_sum:+.3f} gnorm={float(gnorm):.3f} rewards={[round(r,1) for r in rewards]}",
                      flush=True)

        if step % eval_every == 0:
            m, ap, sv = evaluate(held)
            print(f"[eval @{step}] held mean_reward={m:+.3f} applyable={ap:.0%} gold-solve={sv:.0%}", flush=True)
        if verifier is not None and verify_every > 0 and step % verify_every == 0:
            exact = evaluate_exact(held, f"verify_{step}")
            if exact is not None:
                vr, vp = exact
                print(f"[verify @{step}] held exact_resolve={vr:.0%} patch_emission={vp:.0%}", flush=True)
        if emit_preds_dir and emit_preds_every > 0 and step % emit_preds_every == 0:
            emit_predictions(held, f"held_step_{step:04d}")

    m, ap, sv = evaluate(held)
    print(f"\n=== SWE-RL DONE === held mean_reward {bm:+.3f}->{m:+.3f} | "
          f"applyable {ba:.0%}->{ap:.0%} | gold-solve {bs:.0%}->{sv:.0%}")
    if verifier is not None:
        exact_final = evaluate_exact(held, "verify_final")
        if exact_final is not None:
            vr, vp = exact_final
            print(f"  final exact_resolve={vr:.0%} | patch_emission={vp:.0%}")
    else:
        print("  exact SWE resolve not run here (no verifier configured).")
    if emit_preds_dir:
        emit_predictions(held, "held_final")
    print(f"  zero-variance groups {zero_var}/{steps}.")
    model.save_pretrained("artifacts/swe_lora")
    print("  LoRA saved -> artifacts/swe_lora")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--traces", default="data/swe/grounded_traces.jsonl")
    ap.add_argument("--nodes", default="artifacts/graph_growth/swe_code_candidates.jsonl")
    ap.add_argument("--dataset", default="lite")
    ap.add_argument("--repo-root", default="data/swe_repos")
    ap.add_argument("--n-tasks", type=int, default=40)
    ap.add_argument("--src-bodies", type=int, default=4)
    ap.add_argument("--src-lines", type=int, default=70)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r-lora", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", type=int, nargs="+", default=[20, 22, 24, 26, 28])
    ap.add_argument("--eval-every", type=int, default=30)
    ap.add_argument("--ent-coef", type=float, default=0.005)
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="lower for SWE: high temp -> unapplyable SEARCH -> -1/SKIP")
    ap.add_argument("--sft-steps", type=int, default=120)
    ap.add_argument("--max-new", type=int, default=512,
                    help="must exceed gold patch length or generation truncates -> unapplyable")
    ap.add_argument("--diag-max-new", type=int, default=200,
                    help="(staged) max tokens for DIAGNOSE stage; kept short to save VRAM")
    ap.add_argument("--staged", action="store_true", default=False,
                    help="enable staged KV rollout: DIAGNOSE->FIX on shared KV prefix. "
                         "Default OFF (one-shot). Use --staged to enable.")
    ap.add_argument("--reward-mode", choices=["proxy", "verifier"], default="proxy",
                    help="proxy = gold-overlap (fast, in-loop); verifier = exact SWE pass/fail")
    ap.add_argument("--verify-every", type=int, default=0,
                    help="exact held-out verify every N GRPO steps (0=off)")
    ap.add_argument("--verify-backend", choices=["docker", "sbcli"], default="docker")
    ap.add_argument("--verify-out-dir", default="artifacts/graph_growth/swe_verify")
    ap.add_argument("--verify-max-workers", type=int, default=4)
    ap.add_argument("--verify-timeout", type=int, default=1800)
    ap.add_argument("--verify-poll-secs", type=int, default=20)
    ap.add_argument("--verify-gold-sanity", type=int, default=5)
    ap.add_argument("--emit-preds-dir", default="")
    ap.add_argument("--emit-preds-every", type=int, default=0)
    ap.add_argument("--use-exemplar", action="store_true")
    ap.add_argument("--eff-coef", type=float, default=0.0,
                    help="efficiency bonus among wins: +eff_coef*(1-len/max_new). Anti-hack: gated by solve.")
    ap.add_argument("--graph", action="store_true",
                    help="enable graph cross-attention + OperatorInjector (activation steering, not prompt)")
    ap.add_argument("--adapter-ckpt", default="")
    ap.add_argument("--gnn-ckpt", default="")
    ap.add_argument("--train-gnn", action="store_true")
    ap.add_argument("--op-layer", type=int, default=26)
    ap.add_argument("--op-alpha", type=float, default=4.0)
    ap.add_argument("--distill", action="store_true",
                    help="LGGN step 1: context-distill text-grounding -> latent-delta grounding. Teacher=plan-in-text, "
                         "student=plan-as-latent-injection (no text); KL trains the model to ATTEND the injection. "
                         "Runs after SFT, skips GRPO. Includes the real-delta-vs-random-delta kill-test.")
    ap.add_argument("--distill-steps", type=int, default=0,
                    help="number of distillation steps (0 -> use --steps).")
    ap.add_argument("--gap-coef", type=float, default=0.0,
                    help="optional reliance term: loss -= gap_coef*(gold_lp_on - gold_lp_off). 0 = pure KL.")
    ap.add_argument("--rep-penalty", type=float, default=1.15,
                    help="generation repetition_penalty (curbs greedy dup-loops). 1.0 = off.")
    ap.add_argument("--op-sum", action="store_true",
                    help="revert to SUMMING op-delta over nodes (uncalibrated, magnitude scales w/ node count). "
                         "Default = MEAN (node-count-invariant alpha; the calibrated distill default).")
    a = ap.parse_args()
    if a.selftest:
        import sys; sys.exit(0 if _selftest() else 1)
    tasks = load_swe_tasks(a.n_tasks, a.traces, a.nodes, a.dataset, a.repo_root, a.src_bodies, a.src_lines)
    print(f"loaded {len(tasks)} SWE tasks", flush=True)
    if len(tasks) < 8:
        raise SystemExit("too few SWE tasks (need traces + nodes + checkouts)")
    need_verifier = (a.reward_mode == "verifier") or (a.verify_every > 0)
    verifier = SWEExactVerifier(
        a.dataset, "test", a.verify_backend, a.verify_out_dir,
        max_workers=a.verify_max_workers, timeout=a.verify_timeout,
        poll_secs=a.verify_poll_secs, model_name="swe_rl",
    ) if need_verifier else None
    train(
        a.model, tasks, a.steps, a.k, a.lr, a.r_lora, a.seed, a.layers, a.eval_every,
        a.ent_coef, a.temperature, a.sft_steps, a.max_new,
        reward_mode=a.reward_mode, verifier=verifier,
        verify_every=a.verify_every, verify_gold_sanity=a.verify_gold_sanity,
        emit_preds_dir=a.emit_preds_dir, emit_preds_every=a.emit_preds_every,
        use_exemplar=a.use_exemplar, eff_coef=a.eff_coef,
        graph=a.graph, adapter_ckpt=a.adapter_ckpt, gnn_ckpt=a.gnn_ckpt, train_gnn=a.train_gnn,
        op_layer=a.op_layer, op_alpha=a.op_alpha,
        staged=a.staged, diag_max_new=a.diag_max_new,
        distill=a.distill, distill_steps=a.distill_steps, gap_coef=a.gap_coef, op_mean=not a.op_sum,
        rep_penalty=a.rep_penalty,
    )


if __name__ == "__main__":
    main()
