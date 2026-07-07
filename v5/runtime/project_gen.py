"""Repo-continuity benchmark generator — parametric PROJECT CHAINS (v3-B).

One instance = a small Python project evolved over N ordered sessions:
  CREATE  self-contained module per spec (all values stated)
  CROSS   new file that must follow CONVENTIONS chosen in earlier sessions
  DEBUG   the target file is corrupted at session start; tests fail; fix it
  EXTEND  add behavior reusing earlier helpers/values

THE MEASUREMENT TRICK — import-proof withholding: dependency-bearing specs reference
earlier choices WITHOUT restating them ("the same line format as inventory receipts",
"reusing pricing's tax helper"). The seeded values are (a) INLINE format strings — not
importable constants, (b) seeded FUNCTION NAMES — can't import what you can't name,
(c) ID schemes. They exist only in the repo the agent itself built. Agent context is
spec + CURRENT TARGET FILE only — a stateless agent must guess (1/4 x 1/4 ...); an agent
with L0 (own-repo symbols) + L1 (own past implementations) has them.

  python -m v5.runtime.project_gen --selftest     # gold chains pass, withholding verified
  python -m v5.runtime.project_gen --stats
"""
from __future__ import annotations

import random

# ── seeded convention variants ───────────────────────────────────────────────────

LINE_FMTS = [                     # (python expr template, human hint used in S1 spec)
    ('"{} x{} @ {:.2f}".format(name, qty, price)', 'like "apple x3 @ 1.50"'),
    ('"{}*{} at {:.2f}".format(name, qty, price)', 'like "apple*3 at 1.50"'),
    ('"{} - {} - {:.2f}".format(name, qty, price)', 'like "apple - 3 - 1.50"'),
    ('"{}: {} units @ {:.2f}".format(name, qty, price)', 'like "apple: 3 units @ 1.50"'),
]
ID_FMTS = [
    ('"ITEM-{:03d}".format(n)', 'like "ITEM-007"'),
    ('"item_{}".format(n)', 'like "item_7"'),
    ('"#{:04d}".format(n)', 'like "#0007"'),
    ('"I{}".format(n)', 'like "I7"'),
]
TAX_NAMES = ["with_tax", "apply_tax", "add_tax", "taxed_total"]
BULK_NAMES = ["bulk_price", "discounted_price", "apply_bulk", "bulk_total"]
TAXES = [0.05, 0.06, 0.07, 0.08, 0.09, 0.11, 0.12]

LOG_FMTS = [                      # parse_line input layouts (inline parsing, no constants)
    ("[{ts}] {level}: {msg}", '"[10:00] WARN: disk low"'),
    ("{ts} | {level} | {msg}", '"10:00 | WARN | disk low"'),
    ("{level} {ts} {msg}", '"WARN 10:00 disk low"'),
]
LEVEL_SETS = [
    ["DEBUG", "INFO", "WARN", "ERROR"],
    ["DEBUG", "INFO", "WARNING", "ERROR"],
    ["TRACE", "INFO", "WARN", "FATAL"],
]
COUNT_NAMES = ["count_by_level", "level_counts", "tally_levels"]


def _fmt_apply(expr_template: str, **kw) -> str:
    """Evaluate a seeded '\"...\".format(...)' expression with sample values."""
    return eval(expr_template, {}, kw)                       # noqa: S307 — generator-side only


# ── archetype: INVENTORY ─────────────────────────────────────────────────────────

def _inventory(rng: random.Random) -> dict:
    low = rng.randint(2, 9)
    tax = rng.choice(TAXES)
    bulk_n = rng.randint(5, 20)
    bulk_d = rng.choice([0.05, 0.10, 0.15, 0.20])
    line_i = rng.randrange(len(LINE_FMTS))
    id_i = rng.randrange(len(ID_FMTS))
    tax_fn = rng.choice(TAX_NAMES)
    bulk_fn = rng.choice(BULK_NAMES)
    line_expr, line_hint = LINE_FMTS[line_i]
    id_expr, id_hint = ID_FMTS[id_i]

    inv_gold = f'''def add_item(inv, name, qty):
    inv[name] = inv.get(name, 0) + qty
    return inv

def remove_item(inv, name, qty):
    inv[name] = max(0, inv.get(name, 0) - qty)
    return inv

def is_low(inv, name):
    return inv.get(name, 0) < {low}

def make_id(n):
    return {id_expr.replace("(n)", "(n)")}

def receipt_line(name, qty, price):
    return {line_expr}
'''
    pricing_gold = f'''TAX_RATE = {tax}

def {tax_fn}(amount):
    return round(amount * (1 + TAX_RATE), 2)

def {bulk_fn}(price, qty):
    total = price * qty
    if qty >= {bulk_n}:
        total = total * (1 - {bulk_d})
    return round(total, 2)
'''
    pricing_buggy = pricing_gold.replace(f"(1 + TAX_RATE)", f"(1 - TAX_RATE)")
    orders_gold = f'''def order_id(n):
    return "ORD-" + {id_expr}

def order_line(name, qty, price):
    return {line_expr}
'''
    orders_gold_s5 = orders_gold + f'''
def order_total(price, qty):
    total = price * qty
    if qty >= {bulk_n}:
        total = total * (1 - {bulk_d})
    return round(total * (1 + {tax}), 2)
'''

    def _bt(price, qty):                                    # seeded bulk-then-tax reference
        t = price * qty
        if qty >= bulk_n:
            t *= (1 - bulk_d)
        return round(round(t, 2) * 0 + t * (1 + tax), 2)

    sessions = [
        dict(kind="create", target_file="inventory.py",
             spec=(f"Create inventory.py. add_item(inv, name, qty) adds qty into dict inv "
                   f"(missing names start at 0) and returns inv. remove_item(inv, name, qty) "
                   f"subtracts but never below 0, returns inv. is_low(inv, name) is True when "
                   f"the quantity is strictly below {low}. make_id(n) returns an id string "
                   f"{id_hint}. receipt_line(name, qty, price) returns one line {line_hint}."),
             tests=[
                 "import inventory\ninv = {}\ninventory.add_item(inv, 'a', 3)\nassert inv['a'] == 3",
                 "import inventory\ninv = {'a': 2}\ninventory.remove_item(inv, 'a', 5)\nassert inv['a'] == 0",
                 f"import inventory\nassert inventory.is_low({{'a': {low - 1}}}, 'a') is True",
                 f"import inventory\nassert inventory.is_low({{'a': {low}}}, 'a') is False",
                 f"import inventory\nassert inventory.make_id(7) == {_fmt_apply(id_expr, n=7)!r}",
                 f"import inventory\nassert inventory.receipt_line('apple', 3, 1.5) == "
                 f"{_fmt_apply(line_expr, name='apple', qty=3, price=1.5)!r}",
             ],
             gold={"inventory.py": inv_gold}, withheld=[], source_session_idx=None),
        dict(kind="create", target_file="pricing.py",
             spec=(f"Create pricing.py. Constant TAX_RATE = {tax}. {tax_fn}(amount) returns "
                   f"amount*(1+TAX_RATE) rounded to 2 decimals. {bulk_fn}(price, qty) returns "
                   f"price*qty with a {bulk_d:.0%} discount when qty >= {bulk_n}, rounded to 2 "
                   f"decimals (no tax)."),
             tests=[
                 f"import pricing\nassert pricing.{tax_fn}(100) == {round(100 * (1 + tax), 2)}",
                 f"import pricing\nassert pricing.{bulk_fn}(2.0, {bulk_n}) == {round(2.0 * bulk_n * (1 - bulk_d), 2)}",
                 f"import pricing\nassert pricing.{bulk_fn}(2.0, {bulk_n - 1}) == {round(2.0 * (bulk_n - 1), 2)}",
             ],
             gold={"pricing.py": pricing_gold}, withheld=[], source_session_idx=None),
        dict(kind="cross", target_file="orders.py",
             spec=("Create orders.py. order_id(n) returns 'ORD-' plus an id in the SAME id "
                   "scheme inventory item ids use. order_line(name, qty, price) returns one "
                   "line in the SAME format as inventory receipt lines."),
             tests=[
                 f"import orders\nassert orders.order_id(7) == 'ORD-' + {_fmt_apply(id_expr, n=7)!r}",
                 f"import orders\nassert orders.order_line('apple', 3, 1.5) == "
                 f"{_fmt_apply(line_expr, name='apple', qty=3, price=1.5)!r}",
             ],
             gold={"orders.py": orders_gold},
             withheld=[_fmt_apply(id_expr, n=7), _fmt_apply(line_expr, name='apple', qty=3, price=1.5),
                       str(low)],
             source_session_idx=0),                    # id/line/low all come from inventory.py (s0)
        dict(kind="debug", target_file="pricing.py",
             spec=("Order totals came out wrong after a recent change to pricing.py — taxed "
                   "amounts are LOWER than the base amount. Fix pricing.py so taxed totals are "
                   "correct again. Keep every function name and the rate unchanged."),
             tests=[
                 f"import pricing\nassert pricing.{tax_fn}(100) == {round(100 * (1 + tax), 2)}",
                 f"import pricing\nassert pricing.{tax_fn}(10) == {round(10 * (1 + tax), 2)}",
                 f"import pricing\nassert pricing.{bulk_fn}(2.0, {bulk_n}) == {round(2.0 * bulk_n * (1 - bulk_d), 2)}",
             ],
             gold={"pricing.py": pricing_gold}, buggy={"pricing.py": pricing_buggy},
             withheld=[], source_session_idx=None),
        dict(kind="extend", target_file="orders.py",
             spec=("Extend orders.py with order_total(price, qty): the order value with the "
                   "SAME bulk discount rule and the SAME tax rate that pricing uses (discount "
                   "before tax), rounded to 2 decimals. Write it self-contained in orders.py."),
             tests=[
                 f"import orders\nassert orders.order_total(2.0, {bulk_n}) == {_bt(2.0, bulk_n)}",
                 f"import orders\nassert orders.order_total(2.0, {bulk_n - 1}) == {_bt(2.0, bulk_n - 1)}",
                 f"import orders\nassert orders.order_total(10.0, 1) == {_bt(10.0, 1)}",
             ],
             gold={"orders.py": orders_gold_s5},
             withheld=[str(tax), str(bulk_n), str(bulk_d), tax_fn, bulk_fn],
             source_session_idx=1),                    # tax/bulk vals+names come from pricing.py (s1)
    ]
    return dict(archetype="inventory", sessions=sessions,
                params=dict(low=low, tax=tax, bulk_n=bulk_n, bulk_d=bulk_d,
                            line_i=line_i, id_i=id_i, tax_fn=tax_fn, bulk_fn=bulk_fn))


# ── archetype: LOGPARSE ──────────────────────────────────────────────────────────

def _logparse(rng: random.Random) -> dict:
    fmt_i = rng.randrange(len(LOG_FMTS))
    levels = rng.choice(LEVEL_SETS)
    warn = levels[2]
    count_fn = rng.choice(COUNT_NAMES)
    layout, sample = LOG_FMTS[fmt_i]

    if fmt_i == 0:
        parse_body = ('    ts = line[1:line.index("]")]\n'
                      '    rest = line[line.index("]") + 2:]\n'
                      '    level, msg = rest.split(": ", 1)\n')
        mk = lambda ts, lv, ms: f"[{ts}] {lv}: {ms}"
    elif fmt_i == 1:
        parse_body = ('    ts, level, msg = [p.strip() for p in line.split("|", 2)]\n')
        mk = lambda ts, lv, ms: f"{ts} | {lv} | {ms}"
    else:
        parse_body = ('    level, ts, msg = line.split(" ", 2)\n')
        mk = lambda ts, lv, ms: f"{lv} {ts} {ms}"

    parser_gold = (f"LEVELS = {levels!r}\n\n"
                   f"def parse_line(line):\n{parse_body}"
                   f"    if level not in LEVELS:\n        return None\n"
                   f"    return (ts, level, msg)\n")
    parser_buggy = parser_gold.replace("return (ts, level, msg)", "return (level, ts, msg)")
    # writer template + matching .format() arg order per layout (mk's f-string POSITION
    # encodes order only when called with distinct values — with identical placeholders
    # "{}" for all three, that information collapses, so the template/arg-order pairing
    # is spelled out explicitly here instead of derived from mk()).
    _WRITER = {
        0: ("[{}] {}: {}", "ts, level, msg"),
        1: ("{} | {} | {}", "ts, level, msg"),
        2: ("{} {} {}", "level, ts, msg"),
    }
    _tmpl, _args = _WRITER[fmt_i]
    writer_gold = f"def format_line(ts, level, msg):\n    return {_tmpl!r}.format({_args})\n"
    # the seeded warn level name + tuple order are conventions for stats
    stats_gold = (f"def {count_fn}(lines):\n"
                  f"    import parser\n"
                  f"    out = {{}}\n"
                  f"    for line in lines:\n"
                  f"        rec = parser.parse_line(line)\n"
                  f"        if rec is None:\n            continue\n"
                  f"        out[rec[1]] = out.get(rec[1], 0) + 1\n"
                  f"    return out\n")

    l1, l2 = mk("10:00", warn, "disk low"), mk("10:01", levels[3], "boom")
    bad = mk("10:02", "NOPE", "x")
    sessions = [
        dict(kind="create", target_file="parser.py",
             spec=(f"Create parser.py. Constant LEVELS = {levels!r}. parse_line(line) parses "
                   f"lines that look like {sample} (layout: {layout!r}) and returns the tuple "
                   f"(ts, level, msg); return None when the level is not in LEVELS."),
             tests=[
                 f"import parser\nassert parser.parse_line({l1!r}) == ('10:00', {warn!r}, 'disk low')",
                 f"import parser\nassert parser.parse_line({bad!r}) is None",
             ],
             gold={"parser.py": parser_gold}, withheld=[], source_session_idx=None),
        dict(kind="cross", target_file="writer.py",
             spec=("Create writer.py. format_line(ts, level, msg) renders one log line in the "
                   "SAME layout parser.parse_line reads (its exact inverse)."),
             tests=[
                 f"import writer\nassert writer.format_line('10:00', {warn!r}, 'disk low') == {l1!r}",
                 f"import writer, parser\nassert parser.parse_line(writer.format_line('10:01', "
                 f"{levels[3]!r}, 'boom')) == ('10:01', {levels[3]!r}, 'boom')",
             ],
             gold={"writer.py": writer_gold}, withheld=[l1, layout],
             source_session_idx=0),                    # layout/sample line come from parser.py (s0)
        dict(kind="debug", target_file="parser.py",
             spec=("Downstream code says parsed records come out scrambled since the last "
                   "change to parser.py — fields are in the wrong order. Fix parse_line so "
                   "records are correct again."),
             tests=[
                 f"import parser\nassert parser.parse_line({l1!r}) == ('10:00', {warn!r}, 'disk low')",
                 f"import parser\nassert parser.parse_line({l2!r}) == ('10:01', {levels[3]!r}, 'boom')",
             ],
             gold={"parser.py": parser_gold}, buggy={"parser.py": parser_buggy}, withheld=[],
             source_session_idx=None),
        dict(kind="extend", target_file="stats.py",
             spec=(f"Create stats.py with {count_fn}(lines): parse each line with the project's "
                   f"parser and return a dict counting records per level name (skip unparseable "
                   f"lines). Use the SAME level names the parser accepts."),
             tests=[
                 f"import stats\nassert stats.{count_fn}([{l1!r}, {l2!r}, {bad!r}]) == "
                 f"{{{warn!r}: 1, {levels[3]!r}: 1}}",
             ],
             gold={"stats.py": stats_gold}, withheld=[warn],
             source_session_idx=0),                    # warn level name comes from parser.py's LEVELS (s0)
    ]
    return dict(archetype="logparse", sessions=sessions,
                params=dict(fmt_i=fmt_i, levels=levels, count_fn=count_fn))


# ── archetype: INVENTORY_INFER (proposed extension, opt-in — see LGGNv3_design.md) ──────────
#
# Every existing dependency session hands the model a fully-specified plan: "the SAME tax
# rate/id scheme/level names that X uses" NAMES the source and the exact rule to apply. The
# model never has to recognize an unstated need or derive how facts combine -- it retrieves
# a named fact and pastes it per explicit instructions. That's why flat cosine keeps winning
# over a trained ranker (see memory_refiner.py's task #25 postmortem): when the task reduces
# to "find the record matching this fully-specified need," similarity search IS the right
# tool, and there's no reasoning residual left for a ranker to capture.
#
# This session tests something different: SELECTIVE composition, not lookup.
#   - No filename cue ("pricing") in the spec -- "the standard bulk-quantity treatment
#     already established in this project" must be recognized implicitly.
#   - The correct answer requires ISOLATING one atomic fact (bulk_fn) from an ADJACENT one
#     (tax) that appears ALONGSIDE it in an existing combined record (order_total, s4) --
#     naively retrieving/copying the nearest-matching record over-includes tax, which this
#     session explicitly requires excluding ("members are tax-exempt").
#
# Registered as a SEPARATE archetype key, not merged into "inventory" -- make_split()'s
# default (archetypes=("inventory","logparse")) is completely unaffected; every existing
# GB1-4 number stays comparable until this is explicitly opted into.

def _inventory_infer(rng: random.Random) -> dict:
    base = _inventory(rng)
    p = base["params"]
    bulk_n, bulk_d = p["bulk_n"], p["bulk_d"]
    orders_gold_s5 = base["sessions"][4]["gold"]["orders.py"]      # s4 = extend (has order_total)
    orders_gold_s6 = orders_gold_s5 + f'''
def apply_member_discount(price, qty):
    total = price * qty
    if qty >= {bulk_n}:
        total = total * (1 - {bulk_d})
    return round(total, 2)
'''
    base["sessions"].append(dict(
        kind="infer", target_file="orders.py",
        spec=("Add apply_member_discount(price, qty) to orders.py: members get the standard "
              "bulk-quantity treatment already established in this project, but NO tax is "
              "applied (members are tax-exempt). Round to 2 decimals."),
        tests=[
            f"import orders\nassert orders.apply_member_discount(2.0, {bulk_n}) == "
            f"{round(2.0 * bulk_n * (1 - bulk_d), 2)}",
            f"import orders\nassert orders.apply_member_discount(2.0, {bulk_n - 1}) == "
            f"{round(2.0 * (bulk_n - 1), 2)}",
        ],
        gold={"orders.py": orders_gold_s6},
        withheld=[str(bulk_n), str(bulk_d)],
        source_session_idx=1,        # bulk_fn's TRUE origin is pricing.py (s1) -- NOT s4's
    ))                                # order_total, which combines it with tax (must be excluded)
    base["archetype"] = "inventory_infer"
    return base


ARCHETYPES = {"inventory": _inventory, "logparse": _logparse, "inventory_infer": _inventory_infer}


def make_instance(archetype: str, seed: int) -> dict:
    rng = random.Random(seed * 1000003 + hash(archetype) % 997)
    inst = ARCHETYPES[archetype](rng)
    inst["instance_id"] = f"{archetype}_{seed}"
    inst["seed"] = seed
    for k, s in enumerate(inst["sessions"]):
        s["sid"] = f"{inst['instance_id']}_s{k}"
        s["depth"] = k
    return inst


def make_split(archetypes=("inventory", "logparse"), seeds=range(0, 20)) -> list[dict]:
    return [make_instance(a, s) for a in archetypes for s in seeds]


def gold_state_after(inst: dict, upto: int) -> dict[str, str]:
    """Gold repo files after sessions 0..upto (inclusive)."""
    files: dict[str, str] = {}
    for s in inst["sessions"][:upto + 1]:
        files.update(s["gold"])
    return files


# ── selftest ────────────────────────────────────────────────────────────────────

def _selftest() -> bool:
    from v5.runtime.sandbox import run_project
    print("project_gen --selftest: gold chains, buggy-fails, withholding, determinism\n")
    n_dep = 0
    for arch in ARCHETYPES:
        for seed in (0, 1, 2):
            inst = make_instance(arch, seed)
            files: dict[str, str] = {}
            for k, s in enumerate(inst["sessions"]):
                files.update(s["gold"])
                res = run_project(files, s["tests"])
                assert res["passed"], (inst["instance_id"], s["sid"], res)
                if s.get("buggy"):
                    bfiles = dict(files)
                    bfiles.update(s["buggy"])
                    bres = run_project(bfiles, s["tests"])
                    assert not bres["passed"], f"buggy state must fail: {s['sid']}"
                for w in s.get("withheld", []):
                    assert str(w) not in s["spec"], \
                        f"withheld leak in spec: {w!r} in {s['sid']}"
                    n_dep += 1
            i2 = make_instance(arch, seed)
            assert i2["sessions"][0]["spec"] == inst["sessions"][0]["spec"], "determinism"
    print(f"  [1] gold chains pass, buggy states fail, {n_dep} withheld tokens verified -> PASS")

    # source_session_idx (Stage 2 ground-truth label): invariant + static per-archetype shape.
    # withheld <-> source_session_idx must move together (a session either has no dependency
    # and no source, or both), and the source must be a REAL earlier session in the same chain.
    for arch in ARCHETYPES:
        inst = make_instance(arch, 0)
        for s in inst["sessions"]:
            idx = s.get("source_session_idx")
            if s.get("withheld"):
                assert idx is not None, f"{s['sid']}: withheld but no source_session_idx"
                assert 0 <= idx < s["depth"], \
                    f"{s['sid']}: source_session_idx {idx} must point to an earlier session"
            else:
                assert idx is None, f"{s['sid']}: no withheld but source_session_idx={idx}"
    exp = {"inventory": {2: 0, 4: 1}, "logparse": {1: 0, 3: 0},
           "inventory_infer": {2: 0, 4: 1, 5: 1}}
    for arch, want in exp.items():
        got = {s["depth"]: s["source_session_idx"] for s in make_instance(arch, 0)["sessions"]
              if s["source_session_idx"] is not None}
        assert got == want, f"{arch}: source_session_idx map {got} != expected {want}"
    print("  [1b] source_session_idx <-> withheld invariant + per-archetype map -> PASS")

    # cross-seed variation: conventions actually vary
    specs = {make_instance("inventory", s)["params"]["line_i"] for s in range(12)}
    assert len(specs) >= 3, "line formats should vary across seeds"
    names = {make_instance("inventory", s)["params"]["tax_fn"] for s in range(12)}
    assert len(names) >= 3, "tax fn names should vary"
    print("  [2] convention variation across seeds -> PASS")
    print("\n  PROJECT_GEN SELFTEST -> PASS")
    return True


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--stats", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if _selftest() else 1)
    if a.stats:
        insts = make_split()
        n_sess = sum(len(i["sessions"]) for i in insts)
        n_dep = sum(1 for i in insts for s in i["sessions"] if s.get("withheld"))
        print(f"{len(insts)} instances, {n_sess} sessions, {n_dep} dependency-bearing")
    else:
        ap.print_help()
