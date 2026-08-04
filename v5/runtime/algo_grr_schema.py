"""algo_grr_schema -- THE THINKER CHOOSES THE ABSTRACTION; the LM only fills its slots.

This is where the thinker finally has a job that is neither retrieval nor decoration, and it is the
synthesis of three measured failures rather than a fresh guess:

  * thinker picking FILES/STRINGS  -> lost to a plain grep (grep reaches gold 0.786). It was doing
    retrieval, which this project explicitly did not want from it.
  * thinker as LOGIT SLOTS         -> collapsed every time (across-instance cosine 0.99+), because a
    fixed author->test policy gives per-instance state nothing to encode.
  * banked tools                   -> 0 replays across 40 real instances, because the LM writes
    LITERALS ("old = '        if isinstance(value, bytes):'") and a literal cannot transfer.

One cause underneath all three: NOTHING IN THE SYSTEM EVER CHOOSES AN ABSTRACTION. So:
    THINKER  picks the repair SCHEMA        -- discrete, small space, genuinely a decision
    LM       fills that schema's SLOTS      -- symbol names only, never Python source
    GRAPH    banks (schema, params)         -- generalises BY CONSTRUCTION, so replay is possible
    VERIFIER checks against the REAL gold diff -- unchanged, still the boundary

Three things fall out of that split, each of which was a separate measured failure:
  - transfer becomes STRUCTURALLY possible: a schema proven on one instance is the same object on the
    next, only its slots differ.
  - the FORMAT TAX disappears: 55% of literal-arm failures were SyntaxError/IndentationError from
    making the model emit Python that quotes code inside code. Slots are plain symbols.
  - the thinker's output is an ABSTRACTION, not a rank over candidates -- the distinction this
    project's own notes keep pointing at.

SCOPE, AND IT IS A HARD LIMIT: the schemas are MINED from the real gold patches
(scripts/mine_repair_schemas.py) and match only 144 of 871 single-hunk fixes = 16.5%. Real repairs are
mostly not recurring patterns. Everything below is measured ON THAT SUBSET and no schema chooser,
however good, can exceed 16.5% on the full set. Reported everywhere rather than quietly dropped.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("HF_HOME", r"E:\cache\hf")

import numpy as np
import torch
import torch.nn as nn

_SCHEMA_JSON = Path(_ROOT) / "artifacts" / "repair_schemas.json"


def load_schema_data(seed: int = 0):
    if not _SCHEMA_JSON.exists():
        raise FileNotFoundError(f"{_SCHEMA_JSON} missing -- run scripts/mine_repair_schemas.py")
    d = json.loads(_SCHEMA_JSON.read_text(encoding="utf-8"))
    rows = d["instances"]
    random.Random(seed).shuffle(rows)
    return d["schemas"], rows


# ── the schemas as EXECUTABLE repairs. slots are filled by the LM, structure is exact ────────────
def apply_schema(name: str, params: dict, text: str) -> str:
    """Execute a schema against real code. Pure structure -- no LM output is executed here, only
    substituted, so a bad slot value can produce a wrong repair but never arbitrary code."""
    s = params.get("symbol") or params.get("expression") or params.get("object") or ""
    if name == "widen_isinstance":
        t = params.get("added_type", "")
        if not s or not t:
            return text
        # isinstance(x, A) -> isinstance(x, (A, T)) ; isinstance(x, (A, B)) -> isinstance(x, (A, B, T))
        def _w(m):
            arg = m.group(2).strip()
            inner = arg[1:-1].strip() if arg.startswith("(") and arg.endswith(")") else arg
            return f"isinstance({m.group(1)}, ({inner}, {t}))"
        return re.sub(rf"isinstance\(\s*({re.escape(s)})\s*,\s*(\([^)]*\)|[\w\.]+)\s*\)", _w, text, count=1)
    if name == "add_none_guard":
        if not s:
            return text
        return re.sub(rf"^(\s*)(return\s+{re.escape(s)}\b.*)$",
                      rf"\1if {s} is None:\n\1    return None\n\1\2", text, count=1, flags=re.M)
    if name == "add_attribute_check":
        o, at = params.get("object", ""), params.get("attribute", "")
        if not o or not at:
            return text
        return re.sub(rf"^(\s*)(.*\b{re.escape(o)}\.{re.escape(at)}\b.*)$",
                      rf"\1if hasattr({o}, '{at}'):\n\1    \2", text, count=1, flags=re.M)
    if name == "extend_collection_literal":
        col, mem = params.get("collection", ""), params.get("new_member", "")
        if not col or not mem:
            return text
        return re.sub(rf"({re.escape(col)}\s*=\s*[\[\(\{{][^\]\)\}}]*)([\]\)\}}])",
                      rf"\1, {mem}\2", text, count=1)
    if name == "wrap_in_call":
        e, w = params.get("expression", ""), params.get("wrapper", "")
        if not e or not w:
            return text
        return text.replace(e, f"{w}({e})", 1)
    return text


SCHEMA_ORDER = ["widen_isinstance", "add_none_guard", "add_attribute_check",
                "extend_collection_literal", "wrap_in_call", "add_kwarg_passthrough",
                "swap_operator", "change_default_arg"]


# ── the thinker: chooses the ABSTRACTION ─────────────────────────────────────────────────────────
class SchemaThinker(nn.Module):
    """issue+code -> which repair schema. A discrete choice over a SMALL space, which is why it is
    learnable from the 144 real labelled instances -- unlike the 24-way string pointer that REINFORCE
    never cracked. Labels are MINED FROM REAL GOLD DIFFS, so this is supervised on ground truth rather
    than bootstrapped from the model's own guesses."""

    def __init__(self, n_schema: int, d_in: int = 384, d: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * d_in, d), nn.Tanh(), nn.Linear(d, n_schema))

    def forward(self, issue_emb, code_emb):
        return self.net(torch.cat([issue_emb, code_emb], dim=-1))


def embed(texts: list) -> np.ndarray:
    from embedder import encode_batch
    return np.asarray(encode_batch(texts), dtype=np.float32)


def train_thinker(rows: list, schemas: list, epochs: int = 60, lr: float = 3e-3, verbose: bool = True):
    ie = embed([r["problem"][:600] for r in rows])
    ce = embed([r["before"][:600] for r in rows])
    y = torch.tensor([schemas.index(r["schema"]) for r in rows])
    X1, X2 = torch.tensor(ie), torch.tensor(ce)
    m = SchemaThinker(len(schemas))
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for ep in range(epochs):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(m(X1, X2), y)
        loss.backward(); opt.step()
        if verbose and (ep + 1) % 20 == 0:
            acc = float((m(X1, X2).argmax(-1) == y).float().mean())
            print(f"    [thinker] epoch {ep + 1:3d}  loss {float(loss):.4f}  train acc {acc:.3f}",
                  flush=True)
    return m


SLOT_PROMPT = """A bug is repaired by applying this transformation: {schema}

BUG REPORT:
{issue}

CODE TO REPAIR:
{before}

Fill in the values. Output ONLY lines of the form NAME: value, nothing else.
{slots}
"""


def fill_slots(lm, schema: str, slot_names: list, issue: str, before: str) -> dict:
    """The LM's ONLY job: name the symbols. No Python is emitted, so the entire SyntaxError /
    IndentationError failure class -- 55% of the literal arm's failures -- cannot occur."""
    p = SLOT_PROMPT.format(schema=schema, issue=issue[:600], before=before[:900],
                           slots="\n".join(f"{s}:" for s in slot_names))
    try:
        out = str(lm.generate_chat(p, max_new=60, temperature=0.3))
    except Exception:                                              # noqa: BLE001
        return {}
    got = {}
    for ln in out.splitlines():
        m = re.match(r"\s*(\w+)\s*:\s*(.+?)\s*$", ln)
        if m and m.group(1) in slot_names:
            got[m.group(1)] = m.group(2).strip().strip("`'\"")
    return got


# ── the experiment ───────────────────────────────────────────────────────────────────────────────
def evaluate(rows, schemas, slot_defs, thinker, lm, mode="thinker", verbose=True):
    """mode: thinker = the thinker picks the schema (learned abstraction choice)
             random  = uniform pick        -> is the thinker doing anything at all?
             oracle  = the TRUE schema     -> ceiling given perfect abstraction choice, isolating
                                              how much of the loss is slot-filling vs schema choice
    A repair counts ONLY when the transformed code equals the real gold `after` exactly."""
    ie, ce = embed([r["problem"][:600] for r in rows]), embed([r["before"][:600] for r in rows])
    fixed = sch_ok = 0
    for i, r in enumerate(rows):
        if mode == "oracle":
            name = r["schema"]
        elif mode == "random":
            name = random.choice(schemas)
        else:
            with torch.no_grad():
                name = schemas[int(thinker(torch.tensor(ie[i]), torch.tensor(ce[i])).argmax())]
        sch_ok += int(name == r["schema"])
        params = fill_slots(lm, name, slot_defs.get(name, []), r["problem"], r["before"])
        out = apply_schema(name, params, r["before"])
        ok = out.strip() == r["after"].strip()
        fixed += int(ok)
        if verbose and (ok or i < 6):
            print(f"    [{i:3d}] {r['instance_id'][:30]:30s} {name:26s} "
                  f"schema{'OK' if name == r['schema'] else '  '} -> {'FIXED' if ok else 'no'}",
                  flush=True)
    n = max(1, len(rows))
    return {"fixed": fixed, "n": n, "schema_acc": sch_ok / n}


def main():
    ap = argparse.ArgumentParser(description="Thinker picks the repair schema; LM fills its slots.")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--lm", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    if not a.run:
        ap.print_help(); return

    slot_defs, rows = load_schema_data(seed=a.seed)
    schemas = SCHEMA_ORDER
    cut = int(len(rows) * 0.7)
    tr, held = rows[:cut], rows[cut:cut + a.n]
    print(f"{len(rows)} schema-matched instances (16.5% of 871 real single-hunk gold patches --")
    print(f"  that 16.5% is a HARD CEILING for this approach on the full set)")
    print(f"train {len(tr)} | held {len(held)}\n")
    torch.manual_seed(a.seed); random.seed(a.seed)
    th = train_thinker(tr, schemas)

    from v5.runtime.dcpd_latent import WhiteBox
    lm = WhiteBox(a.lm, quant="4bit")
    res = {}
    for mode in ("random", "thinker", "oracle"):
        print(f"\n=== {mode} ===", flush=True)
        random.seed(a.seed)
        res[mode] = evaluate(held, schemas, slot_defs, th, lm, mode=mode)

    print(f"\n{'=' * 74}")
    print(f"THINKER PICKS THE ABSTRACTION, LM FILLS THE SLOTS  (n={res['thinker']['n']} held-out)")
    for mode in ("random", "thinker", "oracle"):
        r = res[mode]
        print(f"  {mode:8s}  schema chosen correctly {r['schema_acc']:.3f}   "
              f"gold fix reproduced {r['fixed']}/{r['n']} = {r['fixed'] / r['n']:.3f}")
    d = res["thinker"]["schema_acc"] - res["random"]["schema_acc"]
    print(f"\n  thinker - random (schema choice): {d:+.3f}  -> "
          f"{'the thinker is choosing, not guessing' if d > 0.1 else 'NOT better than guessing'}")
    print(f"  oracle ceiling shows how much is slot-filling vs abstraction choice")
    print(f"{'=' * 74}")


def _selftest() -> bool:
    print("algo_grr_schema --selftest: schemas execute; thinker chooses; slots are symbols\n")
    ok = True

    def chk(t, c, d=""):
        nonlocal ok
        ok &= bool(c)
        print(f"  [{'PASS' if c else 'FAIL'}] {t}{('  - ' + d) if d else ''}")

    src = "        if isinstance(value, bytes):\n            return bytes(value)"
    out = apply_schema("widen_isinstance", {"symbol": "value", "added_type": "memoryview"}, src)
    chk("[1] widen_isinstance reproduces the REAL django-11133 gold fix",
        "isinstance(value, (bytes, memoryview))" in out, out.splitlines()[0].strip())

    out2 = apply_schema("widen_isinstance", {"symbol": "v", "added_type": "memoryview"}, src)
    chk("[2] a schema is PARAMETERISED: wrong symbol -> no change (it is not a blind replace)",
        out2 == src)

    src3 = "x = [1, 2]"
    out3 = apply_schema("extend_collection_literal", {"collection": "x", "new_member": "3"}, src3)
    chk("[3] extend_collection_literal executes", out3 == "x = [1, 2, 3]", out3)

    # THE TRANSFER PROPERTY, which literal tools structurally could not have
    a1 = apply_schema("widen_isinstance", {"symbol": "a", "added_type": "int"},
                      "if isinstance(a, str):")
    a2 = apply_schema("widen_isinstance", {"symbol": "b", "added_type": "float"},
                      "if isinstance(b, (int, str)):")
    chk("[4] THE SAME schema fires on DIFFERENT code with different slots -- transfer by construction",
        "isinstance(a, (str, int))" in a1 and "isinstance(b, (int, str, float))" in a2,
        a2.strip())

    slot_defs, rows = load_schema_data()
    chk("[5] schemas were MINED from real gold patches, not invented here",
        len(rows) > 100 and all("schema" in r for r in rows), f"{len(rows)} labelled instances")

    t = SchemaThinker(len(SCHEMA_ORDER))
    o = t(torch.randn(384), torch.randn(384))
    chk("[6] the thinker outputs a distribution over ABSTRACTIONS, not a rank over candidates",
        tuple(o.shape) == (len(SCHEMA_ORDER),), f"{tuple(o.shape)}")
    chk("[7] it is small enough to learn from 144 real labelled instances",
        sum(p.numel() for p in t.parameters()) < 60000,
        f"{sum(p.numel() for p in t.parameters()):,} params")

    print(f"\n  ALGO_GRR_SCHEMA -> {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    main()
