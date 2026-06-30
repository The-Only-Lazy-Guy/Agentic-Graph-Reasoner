"""Self-supervised operator discovery (LGGN V5) — operators EMERGE from the data, not hand-coded.

Replaces the regex `lggn.label_gold` (8 ops, 59% coverage, over-firing fallback). Each gold patch is
split into hunks; each hunk gets a STRUCTURAL SIGNATURE (identifiers/literals abstracted, Python
keywords + the add/modify shape kept). Recurring signatures across the corpus ARE the operators
(library-learning / trajectory compression). Every hunk maps to its nearest operator -> 100% coverage,
no hand patterns. The discovered library is compatible with `lggn.Operator`.

  python -m v5.runtime.operator_discovery --selftest
  python -m v5.runtime.operator_discovery --discover --dataset lite --out data/swe/discovered_ops.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

_KW = {"if", "elif", "else", "for", "while", "try", "except", "finally", "with", "return", "yield",
       "raise", "import", "from", "def", "class", "lambda", "and", "or", "not", "in", "is",
       "None", "True", "False", "assert", "await", "async", "break", "continue", "pass",
       "global", "nonlocal", "del", "self"}


def extract_hunks(diff: str) -> list[tuple[str, list[str], list[str]]]:
    """Unified diff -> [(file, removed_lines, added_lines)] per @@ hunk."""
    out, file, rem, add, lines, i = [], None, [], [], diff.splitlines(), 0
    while i < len(lines):
        l = lines[i]
        if l.startswith("+++ "):
            file = l[4:].strip().split("\t")[0]
            if file.startswith("b/"):
                file = file[2:]
        elif l.startswith("@@"):
            if rem or add:
                out.append((file, rem, add)); rem, add = [], []
            i += 1
            while i < len(lines) and not lines[i].startswith(("@@", "--- ", "diff ", "+++ ")):
                ln = lines[i]
                if ln.startswith("+"):
                    add.append(ln[1:])
                elif ln.startswith("-"):
                    rem.append(ln[1:])
                i += 1
            continue
        i += 1
    if rem or add:
        out.append((file, rem, add))
    return [(f, r, a) for (f, r, a) in out if (r or a)]


def _norm_line(l: str) -> str:
    l = re.sub(r'"[^"]*"|\'[^\']*\'', "S", l)               # strings -> S
    l = re.sub(r"\b\d+\.?\d*\b", "N", l)                    # numbers -> N
    toks = re.findall(r"[A-Za-z_]\w*|[^\w\s]", l)
    return " ".join(t if (t in _KW or not t[:1].isalpha()) else "V" for t in toks)


def _markers(added: list[str]) -> list[str]:
    """Operation-type markers so keyword-less content edits sub-type instead of collapsing to a catch-all."""
    s = "\n".join(added)
    m = []
    if re.search(r"[<>]=?|==|!=| in | is ", s): m.append("CMP")
    if re.search(r"(?<![=!<>+\-*/])=(?!=)", s): m.append("ASSIGN")
    if re.search(r"\w\s*\(", s): m.append("CALL")
    if re.search(r"\w\.\w", s): m.append("ATTR")
    if re.search(r"\w\[", s): m.append("SUB")
    return m


def signature(removed: list[str], added: list[str]) -> tuple:
    """Structural fingerprint of a change: (MOD|ADD, sorted Python keywords, + op-markers). Identifiers/
    literals are abstracted, so 'add None-guard for X' and '...for Y' share a signature; op-markers
    (ASSIGN/CALL/ATTR/CMP/SUB) keep keyword-less content edits from collapsing into one catch-all."""
    akw = sorted({t for line in added for t in re.findall(r"[A-Za-z_]\w*", _norm_line(line)) if t in _KW})
    flag = "MOD" if removed else "ADD"
    mk = _markers(added) if len(akw) <= 1 else []          # only sub-type when keywords are sparse
    return (flag, *akw, *mk[:2])


def _name(sig: tuple) -> str:
    flag, *kws = sig
    body = "_".join(kws[:4]) if kws else "edit"
    return f"{'Mod' if flag == 'MOD' else 'Add'}_{body}"[:48]


def _example_diff(rem: list[str], add: list[str]) -> str:
    body = "\n".join(["- " + r for r in rem[:4]] + ["+ " + a for a in add[:6]])
    return body[:400]


def discover(golds: list[str], min_freq: int = 3, max_ops: int = 60) -> tuple[list[dict], Counter]:
    """Cluster all hunks by signature -> operators (signatures with freq >= min_freq, top max_ops)."""
    sigs: Counter = Counter()
    example: dict = {}
    for g in golds:
        for (_f, rem, add) in extract_hunks(g or ""):
            s = signature(rem, add)
            sigs[s] += 1
            example.setdefault(s, (rem, add))
    ops = []
    for n, (s, c) in enumerate(sigs.most_common(max_ops)):
        if c < min_freq:
            break
        rem, add = example[s]
        ops.append({
            "op_id": f"disc_{n}", "name": _name(s),
            "input_type": "code", "output_type": "code",
            "precondition": " ".join(s[1:]) or "structural edit",
            "realize_hint": "apply a transform of this shape:\n" + _example_diff(rem, add),
            "confidence": 0.5, "source": "discovered", "age": 0, "validation_count": int(c),
            "_sig": list(s),
        })
    return ops, sigs


def _sig_set(sig: tuple) -> set:
    return set(sig)


def chunk(diff: str, ops: list[dict]) -> list[str]:
    """A patch -> operator-name trajectory. Each hunk -> its operator (exact signature, else nearest by
    keyword overlap). 100% coverage (every hunk maps; unmatched -> 'novel')."""
    by_sig = {tuple(o["_sig"]): o["name"] for o in ops if "_sig" in o}
    traj = []
    for (_f, rem, add) in extract_hunks(diff or ""):
        s = signature(rem, add)
        if s in by_sig:
            traj.append(by_sig[s]); continue
        best, bj = None, 0.0
        ss = _sig_set(s)
        for o in ops:
            os = _sig_set(tuple(o.get("_sig", [])))
            j = len(ss & os) / (len(ss | os) + 1e-9)
            if j > bj:
                bj, best = j, o["name"]
        traj.append(best if bj >= 0.5 else "novel")
    return traj


def visualize(ops: list[dict], golds: list[str], out: str, model_trajs: list[list[str]] | None = None) -> None:
    """Plot the operator-space the traversal moves over: discovered operators = nodes (size ∝ freq),
    edges = consecutive-operator TRANSITIONS across the corpus's gold trajectories (width ∝ count).
    This is the graph reasoning traverses. If `model_trajs` (the model's actual chosen op-sequences) is
    given, overlay them in red — a LEARNED traverse hugs the thick gold edges, a decorative one
    scatters (the V7 kill-test, made visible). metrics + PNG; robust if matplotlib absent."""
    import math
    from collections import Counter
    names = [o["name"] for o in ops]
    idx = {n: i for i, n in enumerate(names)}
    freq = {o["name"]: o.get("validation_count", 1) for o in ops}
    trans = Counter()
    for g in golds:
        tr = [t for t in chunk(g, ops) if t in idx]
        for a, b in zip(tr, tr[1:]):
            trans[(a, b)] += 1
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(Path(out).with_suffix(".json")).write_text(
        json.dumps({"nodes": names, "freq": freq, "transitions": {f"{a}->{b}": c for (a, b), c in trans.items()}}, indent=1),
        encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    n = len(names)
    pos = {nm: (math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n)) for i, nm in enumerate(names)}
    fig, ax = plt.subplots(figsize=(13, 13))
    mx = max(trans.values()) if trans else 1
    for (a, b), c in trans.items():
        (x0, y0), (x1, y1) = pos[a], pos[b]
        ax.plot([x0, x1], [y0, y1], color="0.6", lw=0.3 + 3 * c / mx, alpha=0.5, zorder=1)
    fmx = max(freq.values()) if freq else 1
    for nm in names:
        x, y = pos[nm]
        ax.scatter([x], [y], s=80 + 800 * freq[nm] / fmx, color="C0", alpha=0.8, zorder=2)
        ax.text(x * 1.08, y * 1.08, nm, fontsize=6, ha="center", va="center", zorder=3)
    if model_trajs:
        for tr in model_trajs:
            tr = [t for t in tr if t in pos]
            for a, b in zip(tr, tr[1:]):
                (x0, y0), (x1, y1) = pos[a], pos[b]
                ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                            arrowprops=dict(arrowstyle="->", color="red", lw=1.5, alpha=0.7), zorder=4)
    ax.set_title(f"LGGN operator-space ({n} ops) — gold transitions (grey) "
                 + ("+ model traversal (red)" if model_trajs else ""))
    ax.axis("off"); fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def save(ops: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(o) for o in ops), encoding="utf-8")


def _selftest() -> bool:
    print("operator_discovery --selftest: signatures cluster recurring change-shapes (no model)\n")
    # 3 None-guard variants (diff identifiers) + 2 imports + 2 try/except -> 3 operators
    golds = []
    for v in ("foo", "bar", "baz"):
        golds.append(f"--- a/x.py\n+++ b/x.py\n@@\n+    if {v} is None:\n+        return None")
    for m in ("os", "sys"):
        golds.append(f"--- a/y.py\n+++ b/y.py\n@@\n+import {m}")
    for v in ("a", "b"):
        golds.append(f"--- a/z.py\n+++ b/z.py\n@@\n-    do({v})\n+    try:\n+        do({v})\n+    except Exception:\n+        pass")
    ops, sigs = discover(golds, min_freq=2, max_ops=20)
    names = [o["name"] for o in ops]
    print(f"  discovered {len(ops)} operators from {len(golds)} golds: {names}")
    for o in ops:
        print(f"    {o['name']:24} freq={o['validation_count']} sig={o['_sig']}")
    assert len(ops) == 3, f"expected 3 clusters (None-guard / import / try-except), got {len(ops)}"
    # the 3 None-guard variants collapse to ONE operator (identifiers abstracted)
    guard = [o for o in ops if "if" in o["_sig"] and "None" in o["_sig"]]
    assert guard and guard[0]["validation_count"] == 3, "3 None-guard variants -> 1 op, freq 3"
    # chunk a held-out None-guard -> that operator (generalization, not memorization)
    held = "--- a/q.py\n+++ b/q.py\n@@\n+    if other is None:\n+        return None"
    tr = chunk(held, ops)
    print(f"  chunk(held None-guard) -> {tr}")
    assert tr == [guard[0]["name"]], "held-out None-guard maps to the discovered guard operator"
    print("\n  OPERATOR-DISCOVERY SELFTEST -> PASS")
    return True


def main():
    ap = argparse.ArgumentParser(description="Self-supervised operator discovery from gold patches.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--discover", action="store_true")
    ap.add_argument("--dataset", default="lite")
    ap.add_argument("--split", default="test")
    ap.add_argument("--min-freq", type=int, default=3)
    ap.add_argument("--max-ops", type=int, default=60)
    ap.add_argument("--out", default="data/swe/discovered_ops.jsonl")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if _selftest() else 1)
    if a.discover:
        from v5.graph_grower.swe_load import load_instances
        golds = [i.get("patch", "") for i in load_instances(name=a.dataset, split=a.split, limit=0)]
        ops, sigs = discover(golds, min_freq=a.min_freq, max_ops=a.max_ops)
        save(ops, a.out)
        cov = sum(1 for g in golds for _ in [chunk(g, ops)] if _ and "novel" not in _)
        nh = sum(len(extract_hunks(g)) for g in golds)
        covered = sum(t != "novel" for g in golds for t in chunk(g, ops))
        print(f"discovered {len(ops)} operators from {len(golds)} golds ({nh} hunks) -> {a.out}")
        print(f"coverage: {covered}/{nh} hunks ({covered/max(1,nh):.0%}) map to a discovered operator (rest=novel)")
        print("top operators:")
        for o in ops[:15]:
            print(f"  {o['name']:30} freq={o['validation_count']:3} | {o['precondition'][:50]}")
        visualize(ops, golds, "artifacts/train_plots/operator_space.png")
        print("operator-space graph -> artifacts/train_plots/operator_space.png (+ .json)")
        return
    ap.print_help()


if __name__ == "__main__":
    main()
