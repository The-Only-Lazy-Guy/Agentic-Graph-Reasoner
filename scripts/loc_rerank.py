"""Top-10 LM reranker. The measurement said this is where the remaining accuracy is:
recall@10 = 0.820 on held-INSTANCE against top-1 = 0.527, so 29 points sit in a pure ranking
problem, not an evidence problem.

WHY AN LM FINALLY EARNS ITS TOKENS HERE. Every earlier LM attempt in this work failed on cost or
noise because it was asked to look at a whole repo (1852 files) or to be a retrieval encoder, which
it is measurably worse at than a 22M MiniLM. Judging 10 candidates against an issue is the opposite
shape: tiny budget, and it is the one thing a language model is actually good at.

NOT GENERATION. Each candidate gets ONE forward pass and the score is logit(" yes") - logit(" no")
on a single next-token position -- a cross-encoder read off the LM head. No sampling, no decoding
loop, no parsing.

RESIDUAL ON THE INCUMBENT. Final score is z(fused) + lam * z(lm) with lam fit on TRAIN. At lam = 0
this is exactly the 0.5267 fused scorer, so the incumbent is the floor by construction -- the
discipline two earlier heads in this work skipped and paid for.

EVIDENCE IS REAL FILE TEXT, targeted: the module docstring plus the definitions whose names the
issue actually mentions, with a few lines of body. Reading real source is affordable for 10 files
and was not for 1852, which is the whole reason this stage exists.
"""
import os, sys, json, re, collections, random
os.environ.setdefault("HF_HOME", r"E:\cache\hf")
sys.path.insert(0, r"E:\PROJECT\graph_v5")
import numpy as np
import torch
from v5.runtime.membrane import load_content, content_text, symbol_owners, _z
from embedder import encode_batch

A = r"E:\PROJECT\graph_v5\artifacts"
SRC = r"E:\swebench_src"
DIRS = {r: r.replace("/", "_") for r in
        ["django/django", "sympy/sympy", "matplotlib/matplotlib", "scikit-learn/scikit-learn",
         "pytest-dev/pytest", "sphinx-doc/sphinx", "astropy/astropy", "psf/requests",
         "pylint-dev/pylint", "pydata/xarray", "mwaskom/seaborn", "pallets/flask"]}
IDT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
N_CH = 8

rows = json.load(open(rf"{A}\swebench_loc_big.json", encoding="utf-8"))
cont = load_content()
rich = {}
for l in open(rf"{A}\repo_rich.jsonl", encoding="utf-8"):
    d = json.loads(l)
    rich[(d["repo"], d["path"])] = d

_ev_cache = {}


def evidence(repo, path, issue_toks, cap=700):
    """Real source, targeted at what the issue names. Falls back to the summary if unreadable."""
    key = (repo, path)
    if key in _ev_cache:
        head, defs = _ev_cache[key]
    else:
        fp = os.path.join(SRC, DIRS.get(repo, ""), path.replace("/", os.sep))
        try:
            txt = open(fp, encoding="utf-8", errors="ignore").read()
        except Exception:
            txt = ""
        head = ""
        m = re.match(r'\s*[ru]?"""(.*?)"""', txt, re.S)
        if m:
            head = " ".join(m.group(1).split())[:200]
        defs = {}
        for mm in re.finditer(r"^[ \t]*(?:async +)?(?:def|class)[ \t]+([A-Za-z_]\w*)", txt, re.M):
            body = txt[mm.start():mm.start() + 320]
            defs.setdefault(mm.group(1), " ".join(body.split())[:220])
        _ev_cache[key] = (head, defs)
    hit = [v for k, v in defs.items() if k in issue_toks][:3]
    if not hit:
        d = rich.get(key, {})
        hit = [" ".join(sorted(d.get("ids", []))[:20])]
    return (head + " | " + " ".join(hit))[:cap]


# ── build the 8-channel fused scorer (cached: rebuilding it re-encodes every repo and ate most of
# each run's budget, which made the reranker impossible to iterate on) ──────────────────────────
import pickle
_FC = rf"{A}\loc_feats.pkl"
if os.path.exists(_FC):
    F = pickle.load(open(_FC, "rb"))
    print(f"  features loaded from cache: {len(F)} instances")
else:
    F = {}
    files_by_repo = collections.defaultdict(list)
    for (repo, path) in rich:
        files_by_repo[repo].append(path)
    for repo, files in sorted(files_by_repo.items(), key=lambda kv: -len(kv[1])):
        inst = [r for r in rows if r["repo"] == repo]
        files = sorted(files)
        idx = {f: i for i, f in enumerate(files)}
        ok = [r for r in inst if r["gold"] in idx]
        if not ok:
            continue
        id_own, str_own = collections.Counter(), collections.Counter()
        for f in files:
            for s in rich[(repo, f)]["ids"]:
                id_own[s] += 1
            for s in rich[(repo, f)]["strs"]:
                str_own[s] += 1
        old_own = symbol_owners(cont, repo, files)
        txt = [content_text(cont, repo, f) for f in files]
        Ec = np.concatenate([encode_batch(txt[i:i + 256]) for i in range(0, len(txt), 256)])
        Ep = np.concatenate([encode_batch([f.replace("/", " ").replace("_", " ") for f in files[i:i + 256]])
                             for i in range(0, len(files), 256)])
        qs = np.concatenate([encode_batch([r["problem"][:1000] for r in ok[i:i + 256]])
                             for i in range(0, len(ok), 256)])
        for r, q in zip(ok, qs):
            toks = set(IDT.findall(r["problem"]))
            quoted = {m.strip() for m in re.findall(r"[\"'`]([^\"'`\n]{12,200})[\"'`]", r["problem"])}
            ch = np.zeros((N_CH, len(files)), dtype=np.float32)
            for i, f in enumerate(files):
                d = rich[(repo, f)]
                ch[0, i] = sum(1.0 / old_own.get(s, 1)
                               for s in cont.get((repo, f), {}).get("syms", ()) if s in toks)
                ch[1, i] = 1.0 if f.rsplit("/", 1)[-1][:-3] in toks else 0.0
                ch[4, i] = sum(1.0 / id_own[s] for s in d["ids"] if s in toks)
                ch[5, i] = sum(1.0 / str_own[s] for s in d["strs"] if s in quoted)
                ch[6, i] = sum(1.0 for m in d["imps"] if m in r["problem"])
                ch[7, i] = 1.0 if (f in r["problem"] or f[:-3].replace("/", ".") in r["problem"]) else 0.0
            ch[2] = Ep @ q
            ch[3] = Ec @ q
            F[r["instance_id"]] = (np.stack([_z(c) for c in ch]), idx[r["gold"]], repo, files, toks)
            pass
        print(f"    {repo:<26} {len(ok):>4}", flush=True)
    if not os.path.exists(_FC):
        pickle.dump(F, open(_FC, "wb"))
        print(f"  cached {len(F)} feature matrices")

HELD = ("pytest-dev/pytest", "sphinx-doc/sphinx")
hr = [r for r in rows if r["repo"] in HELD and r["instance_id"] in F]
rest = [r for r in rows if r["repo"] not in HELD and r["instance_id"] in F]
random.Random(0).shuffle(rest)
hi, tr = rest[:300], rest[300:]

import torch.nn as nn
_wt = torch.zeros(N_CH, requires_grad=True)
_o = torch.optim.Adam([_wt], lr=0.05)
_TR = [(torch.tensor(F[r["instance_id"]][0]), F[r["instance_id"]][1]) for r in tr]
for _ in range(120):
    _o.zero_grad()
    torch.stack([nn.functional.cross_entropy((_wt @ Fm).unsqueeze(0), torch.tensor([g]))
                 for Fm, g in _TR]).mean().backward()
    _o.step()
W = _wt.detach().numpy().astype(float)
print(f"\n  fused weights {np.round(W,3)}")

# ── the LM reranker ─────────────────────────────────────────────────────────────────────────────
from v5.runtime.dcpd_latent import WhiteBox
LM = os.environ.get("RERANK_LM", "Qwen/Qwen2.5-0.5B-Instruct")
wb = WhiteBox(LM, quant="4bit")
YES = wb.tok(" yes", add_special_tokens=False).input_ids[-1]
NO = wb.tok(" no", add_special_tokens=False).input_ids[-1]
print(f"  LM {LM} quant={wb.quant} vram={wb.vram_gb:.2f}GB  yes/no ids {YES}/{NO}")


_null_cache = {}
NL = chr(10)
_TAIL = "Question: does this file contain the code the issue is about? Answer:"


def _prompt(issue, repo, cand, toks):
    return (f"Issue: {issue[:600]}" + NL + NL + f"File: {cand}" + NL
            + evidence(repo, cand, toks) + NL + NL + _TAIL)


def _null_prompt(repo, cand, toks):
    """Same prompt shape, issue replaced by a content-free stub."""
    return _prompt("a bug report about this project.", repo, cand, toks)


@torch.no_grad()
def _forward_scores(prompts):
    """One forward per prompt, batched. Score = logit(' yes') - logit(' no') at the last real token."""
    out = []
    for i in range(0, len(prompts), 4):
        enc = wb.tok(prompts[i:i + 4], return_tensors="pt", padding=True, truncation=True,
                     max_length=512).to(wb.device)
        lg = wb.model(**enc).logits
        last = enc["attention_mask"].sum(1) - 1
        for b in range(lg.shape[0]):
            v = lg[b, last[b]]
            out.append(float(v[YES] - v[NO]))
    return np.array(out, dtype=np.float32)


@torch.no_grad()
def null_scores(repo, cands, toks):
    """Query-FREE score for each candidate: the same prompt with the issue replaced by a neutral
    stub. logit(yes)-logit(no) carries a large candidate-specific bias -- long files and common
    paths score high no matter what was asked -- and subtracting this isolates the part that
    actually responds to THIS issue. Cached per file, so the cost amortises to ~zero across the
    instances of a repo."""
    need = [c for c in cands if (repo, c) not in _null_cache]
    if need:
        _q = "Issue: a bug report about this project."
        _tail = "Question: does this file contain the code the issue is about? Answer:"
        ps = [_null_prompt(repo, c, toks) for c in need]
        for c, v in zip(need, _forward_scores(ps)):
            _null_cache[(repo, c)] = v
    return np.array([_null_cache[(repo, c)] for c in cands], dtype=np.float32)


@torch.no_grad()
def lm_scores(issue, repo, cands, toks):
    """One forward per candidate; score = logit(' yes') - logit(' no') at the final position.
    With PMI=1 the query-free baseline is subtracted, which removes the candidate-specific bias
    (long files and common paths score high regardless of what was asked)."""
    raw = _forward_scores([_prompt(issue, repo, c, toks) for c in cands])
    if os.environ.get("PMI", "1") == "1":
        raw = raw - null_scores(repo, cands, toks)
    return raw


def evaluate(split, lam, K=int(os.environ.get("K", "10")), limit=None):
    sp = split[:limit] if limit else split
    base_hit = new_hit = ceil_hit = 0
    for r in sp:
        Fm, gi, repo, files, toks = F[r["instance_id"]]
        s = W @ Fm
        top = list((-s).argsort()[:K])
        base_hit += int(top[0] == gi)
        ceil_hit += int(gi in top)
        ls = lm_scores(r["problem"], repo, [files[i] for i in top], toks)
        fused = _z(s[top]) + lam * _z(ls)
        new_hit += int(top[int(fused.argmax())] == gi)
    n = max(1, len(sp))
    return base_hit / n, new_hit / n, ceil_hit / n


LAM_TR = tr[:70]
best_lam, best_a = 0.0, -1
for lam in (0.0, 0.5, 1.5, 3.0, 6.0):
    _, a, _ = evaluate(LAM_TR, lam)
    print(f"    lam={lam:<5} train-slice rerank {a:.4f}", flush=True)
    if a > best_a:
        best_lam, best_a = lam, a
print(f"\n  lam fit on train = {best_lam}")
print(f"\n  split            fused(base)   +LM rerank   recall@10 (ceiling)")
for tag, sp in (("held-INSTANCE", hi), ("held-REPO", hr)):
    b, nw, c = evaluate(sp, best_lam, limit=int(os.environ.get("LIM","150")))
    print(f"    {tag:<14}  {b:.4f}       {nw:.4f}       {c:.4f}")
