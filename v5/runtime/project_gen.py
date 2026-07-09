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
FEE_RATES = [0.02, 0.03, 0.04, 0.10, 0.15]   # service/handling surcharge — compose's 2nd fact
STRATEGIES = ["nash", "maxmin"]              # game-theory solution concepts — v4 preference
# compose_pool distractors: same "RATE = float + fn(amount)" shape as tax.py/fees.py, so flat
# cosine can't separate the TWO true sources from the crowd on surface similarity -- this is the
# ONLY condition under which a smarter ranker could beat cosine (see LGGNv3_design.md 2026-07-08:
# on the 3-record pool GB4c was 1.000 vs 1.000 because both sources trivially cleared MIN_FIT).
# "handling"/"surcharge" deliberately fee-like to compete with fees.py's service_fee.
RATE_DISTRACTORS = [
    ("discount", "DISCOUNT_RATE", "apply_discount"), ("shipping", "SHIP_RATE", "shipping_cost"),
    ("vat", "VAT_RATE", "add_vat"), ("markup", "MARKUP_RATE", "apply_markup"),
    ("commission", "COMMISSION_RATE", "commission"), ("handling", "HANDLING_RATE", "handling_fee"),
    ("insurance", "INSURANCE_RATE", "insurance_cost"), ("surcharge", "SURCHARGE_RATE", "surcharge"),
    ("rebate", "REBATE_RATE", "rebate"), ("duty", "DUTY_RATE", "customs_duty"),
]
DISTRACTOR_RATES = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.09, 0.11, 0.13, 0.16, 0.18]

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


# ── archetype: COMPOSE (2-hop derivation, opt-in — the Stage 2 payoff case) ─────────────────
#
# Every OTHER archetype withholds a RAW LITERAL that sits verbatim in one earlier gold file:
# the "dependency" is retrieve-the-token-and-paste. Nothing is ever DERIVED, so a task can be
# solved the instant the one source record is delivered -- which is why a better ranker
# (GB4a +0.25 over cosine) produced ZERO solve-rate gain (GB4b flat, 2026-07-08): retrieval
# quality doesn't gate solving when solving is a copy.
#
# COMPOSE breaks that. The answer -- final_price's applied total and the combined rate
# (tax+fee) -- is written in NO file anywhere. It only comes into existence when the model
# retrieves TWO atomic facts from TWO separate files (tax.py's rate, fees.py's rate) and
# COMBINES them by the rule the spec states. New knowledge formed from atomic knowledge --
# exactly the "model derives something that isn't in the graph" property.
#
# Why this finally lets the ranker win: solving now needs BOTH sources AND-ed. The ranker's
# per-source retrieval edge over cosine compounds on a 2-of-N pick (cosine ~0.75/src -> ~0.56
# both; ranker ~1.0/src -> ~1.0 both), so better retrieval FINALLY converts to a solve-rate
# gap (GB4b). Distractor s2 (catalog.py) keeps the pool > 2 so it's a real pick, not 2-of-2.
#
# Opt-in like inventory_infer: NOT in make_split()'s default -- GB1-4 numbers stay comparable.
# source_session_idxs (LIST) is the new multi-hop label; source_session_idx stays as the
# primary (first) source for backward compat with the single-source GB4a path.

def _compose(rng: random.Random, n_distractors: int = 0) -> dict:
    tax = rng.choice(TAXES)
    fee = rng.choice(FEE_RATES)
    while fee == tax:                          # keep the two atomic facts numerically distinct
        fee = rng.choice(FEE_RATES)
    low = rng.randint(2, 9)
    id_i = rng.randrange(len(ID_FMTS))
    id_expr, id_hint = ID_FMTS[id_i]

    tax_gold = f'''TAX_RATE = {tax}

def taxed(amount):
    return round(amount * (1 + TAX_RATE), 2)
'''
    fees_gold = f'''FEE_RATE = {fee}

def service_fee(amount):
    return round(amount * FEE_RATE, 2)
'''
    catalog_gold = f'''def make_sku(n):
    return {id_expr}

def is_low(stock, name):
    return stock.get(name, 0) < {low}
'''
    # the DERIVED answer: p*(1+tax+fee). the combined rate (tax+fee) and every result below
    # exist in NO earlier file -- must be computed by composing tax.py's fact with fees.py's.
    checkout_gold = f'''def final_price(p):
    return round(p * (1 + {tax} + {fee}), 2)
'''

    def _fp(p):                               # test oracle -- SAME arithmetic as checkout_gold
        return round(p * (1 + tax + fee), 2)

    sessions = [
        dict(kind="create", target_file="tax.py",
             spec=(f"Create tax.py. Constant TAX_RATE = {tax}. taxed(amount) returns "
                   f"amount*(1+TAX_RATE) rounded to 2 decimals."),
             tests=[
                 f"import tax\nassert tax.taxed(100) == {round(100 * (1 + tax), 2)}",
                 f"import tax\nassert tax.TAX_RATE == {tax}",
             ],
             gold={"tax.py": tax_gold}, withheld=[], source_session_idx=None,
             source_session_idxs=[]),
        dict(kind="create", target_file="fees.py",
             spec=(f"Create fees.py. Constant FEE_RATE = {fee}. service_fee(amount) returns "
                   f"amount*FEE_RATE rounded to 2 decimals."),
             tests=[
                 f"import fees\nassert fees.service_fee(100) == {round(100 * fee, 2)}",
                 f"import fees\nassert fees.FEE_RATE == {fee}",
             ],
             gold={"fees.py": fees_gold}, withheld=[], source_session_idx=None,
             source_session_idxs=[]),
        dict(kind="create", target_file="catalog.py",
             spec=(f"Create catalog.py. make_sku(n) returns an id string {id_hint}. "
                   f"is_low(stock, name) is True when the stock is strictly below {low}."),
             tests=[
                 f"import catalog\nassert catalog.make_sku(7) == {_fmt_apply(id_expr, n=7)!r}",
                 f"import catalog\nassert catalog.is_low({{'a': {low - 1}}}, 'a') is True",
             ],
             gold={"catalog.py": catalog_gold}, withheld=[], source_session_idx=None,
             source_session_idxs=[]),
    ]

    # RATE-like distractor create sessions (compose_pool): pad the pool with N near-identical
    # "RATE = float + fn(amount)" modules so retrieving the TWO true sources (tax, fees) among
    # many becomes a hard pick, not a 2-of-3. Self-contained + fully specified -> the model
    # still solves them (indep stays ~1.0), they only crowd the RETRIEVAL. source_session_idx
    # None (never a dependency). Inserted BEFORE compose so tax/fees stay at s0/s1.
    for nm, const, fn in rng.sample(RATE_DISTRACTORS, min(n_distractors, len(RATE_DISTRACTORS))):
        drate = rng.choice(DISTRACTOR_RATES)
        body = (f"{const} = {drate}\n\ndef {fn}(amount):\n"
                f"    return round(amount * {const}, 2)\n")
        sessions.append(dict(
            kind="create", target_file=f"{nm}.py",
            spec=(f"Create {nm}.py. Constant {const} = {drate}. {fn}(amount) returns "
                  f"amount*{const} rounded to 2 decimals."),
            tests=[
                f"import {nm}\nassert {nm}.{fn}(100) == {round(100 * drate, 2)}",
                f"import {nm}\nassert {nm}.{const} == {drate}",
            ],
            gold={f"{nm}.py": body}, withheld=[], source_session_idx=None,
            source_session_idxs=[]))

    sessions += [
        dict(kind="compose", target_file="checkout.py",
             spec=("Create checkout.py. final_price(p) returns p plus this project's tax "
                   "charged on p plus this project's service fee charged on p — that is, the "
                   "base price with BOTH established rates applied to it and summed (base + "
                   "tax-on-base + fee-on-base), rounded to 2 decimals. Do NOT restate the "
                   "rates; use the two values this project already established. Write it "
                   "self-contained in checkout.py."),
             tests=[
                 f"import checkout\nassert checkout.final_price(100) == {_fp(100)}",
                 f"import checkout\nassert checkout.final_price(50) == {_fp(50)}",
                 f"import checkout\nassert checkout.final_price(8) == {_fp(8)}",
             ],
             gold={"checkout.py": checkout_gold},
             withheld=[str(tax), str(fee)],
             source_session_idx=0,             # primary = tax.py (s0); full set below
             source_session_idxs=[0, 1]),      # BOTH tax.py (s0) AND fees.py (s1) are required
    ]
    return dict(archetype="compose", sessions=sessions,
                params=dict(tax=tax, fee=fee, low=low, id_i=id_i,
                            n_distractors=n_distractors))


def _compose_pool_n(n: int):
    """compose with n rate-like distractor modules crowding the candidate pool -- the hard-
    retrieval variant that lets the ranker beat cosine (LGGNv3_design.md 2026-07-08). Same
    derived answer + labels as compose; only pool size changes. Varying n gives a hardness
    CURVE (n=0 compose easy -> n=8 hard): the ranker's margin over cosine should scale with n,
    which is what proves the win is retrieval-difficulty-driven, not a fluke of one pool."""
    def _f(rng: random.Random) -> dict:
        inst = _compose(rng, n_distractors=n)
        inst["archetype"] = f"compose_pool{'' if n == 8 else n}"
        return inst
    return _f


# ── archetype: PREFERENCE (conditional multi-hop, v4 traversal benchmark) ──────────────
#
# Every existing archetype retrieves independent parallel facts. compose is 2 independent
# hops (tax + fee). There is no benchmark where you CAN'T know what to retrieve next until
# you've retrieved the first thing — i.e. a CONDITIONAL dependency.
#
# This benchmark fills that gap. Structure:
#   s0: config.py — PREFERRED_STRATEGY = "nash" (or "maxmin"), get_strategy()
#   s1: nash.py   — nash_equilibrium(payoffs) implementation
#   s2: maxmin.py — maxmin_solution(payoffs) implementation
#   s3: utils.py  — helper functions (distractor)
#   s4: solver.py — solve(payoffs) using the project's preferred strategy
#
# The spec for s4 says only "this project's established strategy preference" — does NOT
# name nash or maxmin. The ONLY way to know which solver to use is to retrieve config.py
# first, discover PREFERRED_STRATEGY, then retrieve the matching solver.
#
# For latent traversal (v4 Option B): config.py contains the keyword "nash" or "maxmin",
# which naturally correlates with the matching solver file in mpnet space. h_1 refined
# toward config_emb carries the keyword signal; search_ctx(h_1) at hop 2 retrieves the
# matching solver by cosine similarity.

def _preference(rng: random.Random) -> dict:
    strategy = rng.choice(STRATEGIES)
    utils_i = rng.randrange(len(ID_FMTS))
    id_expr, id_hint = ID_FMTS[utils_i]

    config_gold = f'''PREFERRED_STRATEGY = "{strategy}"

def get_strategy():
    return PREFERRED_STRATEGY
'''
    nash_gold = '''def nash_equilibrium(payoffs):
    """Compute pure-strategy Nash equilibrium for a 2x2 game.
    Both players use the SAME payoff matrix. Returns (row, col) or None."""
    a, b, c, d = payoffs[0][0], payoffs[0][1], payoffs[1][0], payoffs[1][1]
    for r, cj in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        row_pay = payoffs[r][cj]
        col_pay = payoffs[r][cj]
        if row_pay >= payoffs[1 - r][cj] and col_pay >= payoffs[r][1 - cj]:
            return (r, cj)
    return None
'''
    maxmin_gold = '''def maxmin_solution(payoffs):
    """Compute maxmin strategy for a 2x2 game.
    Returns (row_choice, col_choice) guaranteeing the maxmin payoff."""
    a, b, c, d = payoffs[0][0], payoffs[0][1], payoffs[1][0], payoffs[1][1]
    row_min0 = min(a, b)
    row_min1 = min(c, d)
    row_best = 0 if row_min0 >= row_min1 else 1
    col_min0 = min(a, c)
    col_min1 = min(b, d)
    col_best = 0 if col_min0 >= col_min1 else 1
    return (row_best, col_best)
'''
    solver_gold = f'''from config import get_strategy

def solve(payoffs):
    strat = get_strategy()
    if strat == "nash":
        from nash import nash_equilibrium
        return nash_equilibrium(payoffs)
    elif strat == "maxmin":
        from maxmin import maxmin_solution
        return maxmin_solution(payoffs)
    return None
'''
    utils_gold = f'''def make_game_id(n):
    return {id_expr}

def is_valid_payoff(v):
    return isinstance(v, (int, float))
'''

    def _solver_outcome(payoffs, strat):
        if strat == "nash":
            for r, cj in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                if payoffs[r][cj] >= payoffs[1 - r][cj] and payoffs[r][cj] >= payoffs[r][1 - cj]:
                    return (r, cj)
            return None
        else:
            a, b, c, d = payoffs[0][0], payoffs[0][1], payoffs[1][0], payoffs[1][1]
            row_min0 = min(a, b)
            row_min1 = min(c, d)
            row_best = 0 if row_min0 >= row_min1 else 1
            col_min0 = min(a, c)
            col_min1 = min(b, d)
            col_best = 0 if col_min0 >= col_min1 else 1
            return (row_best, col_best)

    payoffs = [[3, 1], [0, 2]] if strategy == "nash" else [[2, 0], [1, 3]]
    gold_out = _solver_outcome(payoffs, strategy)

    # Determine which solver session is the conditional target
    solver_idx = 1 if strategy == "nash" else 2

    sessions = [
        dict(kind="create", target_file="config.py",
             spec=("Create config.py. A constant PREFERRED_STRATEGY (a string naming this "
                   "project's chosen game-theory solution concept). get_strategy() returns it."),
             tests=[
                 "import config\nassert isinstance(config.PREFERRED_STRATEGY, str)",
                 "import config\nassert config.get_strategy() == config.PREFERRED_STRATEGY",
             ],
             gold={"config.py": config_gold}, withheld=[], source_session_idx=None,
             source_session_idxs=[]),
        dict(kind="create", target_file="nash.py",
             spec=("Create nash.py. nash_equilibrium(payoffs) computes the pure-strategy Nash "
                   "equilibrium for a 2x2 payoff matrix [[a,b],[c,d]]. Returns (row_choice, "
                   "col_choice) if a pure NE exists, else None."),
             tests=[
                 "import nash\nassert nash.nash_equilibrium([[3,1],[0,2]]) == (0, 0)",
                 "import nash\nassert nash.nash_equilibrium([[1,3],[2,0]]) == (0, 1)",
                 "import nash\nassert nash.nash_equilibrium([[3,0],[0,2]]) == (0, 0)",
             ],
             gold={"nash.py": nash_gold}, withheld=[], source_session_idx=None,
             source_session_idxs=[]),
        dict(kind="create", target_file="maxmin.py",
             spec=("Create maxmin.py. maxmin_solution(payoffs) computes the maxmin strategy for "
                   "a 2x2 payoff matrix [[a,b],[c,d]]. Returns (row_choice, col_choice)."),
             tests=[
                 "import maxmin\nassert maxmin.maxmin_solution([[2,0],[1,3]]) == (1, 0)",
                 "import maxmin\nassert maxmin.maxmin_solution([[0,2],[3,1]]) == (1, 1)",
             ],
             gold={"maxmin.py": maxmin_gold}, withheld=[], source_session_idx=None,
             source_session_idxs=[]),
        dict(kind="create", target_file="utils.py",
             spec=(f"Create utils.py. make_game_id(n) returns an id string {id_hint}. "
                   f"is_valid_payoff(v) checks if v is an int or float."),
             tests=[
                 f"import utils\nassert utils.make_game_id(7) == {_fmt_apply(id_expr, n=7)!r}",
                 "import utils\nassert utils.is_valid_payoff(5) is True",
                 "import utils\nassert utils.is_valid_payoff('x') is False",
             ],
             gold={"utils.py": utils_gold}, withheld=[], source_session_idx=None,
             source_session_idxs=[]),
        dict(kind="compose", target_file="solver.py",
             spec=("Create solver.py. solve(payoffs) returns the game outcome using the "
                   "strategy preference this project has already established. Import and use "
                   "the matching solver. Return the result directly."),
             tests=[
                 f"import solver\nassert solver.solve({payoffs!r}) == {gold_out!r}",
             ],
             gold={"solver.py": solver_gold},
             withheld=[strategy],
             source_session_idx=0,
             source_session_idxs=[0, solver_idx]),
    ]
    return dict(archetype="preference", sessions=sessions,
                params=dict(strategy=strategy, utils_i=utils_i, payoffs=payoffs))


ARCHETYPES = {"inventory": _inventory, "logparse": _logparse,
              "inventory_infer": _inventory_infer, "compose": _compose,
              "compose_pool4": _compose_pool_n(4), "compose_pool": _compose_pool_n(8),
              "preference": _preference}


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
            # multi-hop label (source_session_idxs, LIST): first entry mirrors the singular
            # source_session_idx; every entry must be a real earlier session.
            idxs = s.get("source_session_idxs")
            if idxs:
                assert idxs[0] == idx, f"{s['sid']}: source_session_idxs[0] != source_session_idx"
                assert all(0 <= j < s["depth"] for j in idxs), \
                    f"{s['sid']}: source_session_idxs {idxs} must all point earlier"
    # For preference, source_session_idx is 0 (config) at depth 4 (solver). The conditional
    # source_session_idxs depends on the seed (nash→[0,1] or maxmin→[0,2]) — checked below.
    exp = {"inventory": {2: 0, 4: 1}, "logparse": {1: 0, 3: 0},
           "inventory_infer": {2: 0, 4: 1, 5: 1}, "compose": {3: 0},
           "compose_pool4": {7: 0},       # compose at depth 3 + 4 distractors = 7
           "compose_pool": {11: 0}}       # compose at depth 3 + 8 distractors = 11
    for arch, want in exp.items():
        got = {s["depth"]: s["source_session_idx"] for s in make_instance(arch, 0)["sessions"]
              if s["source_session_idx"] is not None}
        assert got == want, f"{arch}: source_session_idx map {got} != expected {want}"
    print("  [1b] source_session_idx <-> withheld invariant + per-archetype map -> PASS")

    # compose (2-hop): the model must DERIVE the answer by combining two atomic facts from two
    # separate files -- the derived value must exist in NO earlier gold file (else it's just
    # retrieval, not composition). This is the "derive something that isn't in the graph"
    # requirement, asserted mechanically.
    import re as _re
    for arch in ("compose", "compose_pool4", "compose_pool"):
        for seed in (0, 1, 2, 7):
            inst = make_instance(arch, seed)
            ss = inst["sessions"]
            comp = next(s for s in ss if s["kind"] == "compose")   # index shifts with distractors
            ci = comp["depth"]
            tax, fee = inst["params"]["tax"], inst["params"]["fee"]
            # the two atomic facts genuinely live in their (separate) source files
            assert str(tax) in ss[0]["gold"]["tax.py"], "tax fact must be in tax.py"
            assert str(fee) in ss[1]["gold"]["fees.py"], "fee fact must be in fees.py"
            # neither restated in the compose spec (withholding holds -- forces retrieval)
            assert str(tax) not in comp["spec"] and str(fee) not in comp["spec"], \
                "compose spec must not restate either rate"
            # the DERIVED results (test-asserted values) appear in NO earlier gold file -> the
            # answer cannot be retrieved verbatim, only COMPUTED from the two facts. earlier_src
            # now ALSO spans the distractor modules -- the derived value must dodge all of them.
            earlier_src = "".join(v for s in ss[:ci] for v in s["gold"].values())
            derived = _re.findall(r"==\s*([\d.]+)", "\n".join(comp["tests"]))
            assert len(derived) >= 3, "compose must assert concrete derived values"
            for d in derived:
                assert d not in earlier_src, \
                    f"{arch} seed {seed}: derived value {d} leaks into the graph -- not composition"
            # both sources labelled for the multi-hop ranker
            assert comp["source_session_idxs"] == [0, 1] and comp["source_session_idx"] == 0
            # distractors are non-dependency create sessions; only compose carries withheld
            deps = [s for s in ss if s.get("withheld")]
            assert deps == [comp], f"{arch}: only the compose session may be a dependency"
            # session count = tax+fees+catalog (3) + n distractors + compose (1)
            assert len(ss) == 4 + inst["params"]["n_distractors"], \
                f"{arch}: expected {4 + inst['params']['n_distractors']} sessions, got {len(ss)}"
    print("  [1c] compose(+pool4/pool): 2 atomic facts -> DERIVED answer, absent from graph -> PASS")

    # preference (v4 traversal benchmark): conditional multi-hop. Config contains the
    # keyword; the compose spec does NOT; the correct solver is determined by config.
    for seed in (0, 1, 3, 7, 11):
        inst = make_instance("preference", seed)
        ss = inst["sessions"]
        solver = ss[4]
        strat = inst["params"]["strategy"]
        assert strat in ("nash", "maxmin"), f"strategy must be nash or maxmin, got {strat}"
        assert solver["kind"] == "compose", f"solver session must be compose-kind"
        assert strat not in solver["spec"], f"strategy keyword {strat} leaked into the spec"
        assert solver["source_session_idx"] == 0, "primary source must be config.py (s0)"
        # conditional source list: config + the matching solver
        want_idxs = [0, 1] if strat == "nash" else [0, 2]
        assert solver["source_session_idxs"] == want_idxs, \
            f"seed {seed} strat={strat}: idxs {solver['source_session_idxs']} != {want_idxs}"
        # the derived answer must be genuinely DERIVED: the gold code calls a solver function,
        # not just returns a hardcoded value. Check the gold body for import + function call.
        gold_body = solver["gold"]["solver.py"]
        assert "from config import get_strategy" in gold_body
        assert "from nash import nash_equilibrium" in gold_body
        assert "from maxmin import maxmin_solution" in gold_body
        assert "solve(payoffs)" in gold_body
    print("  [1d] preference: conditional multi-hop, strategy keyword withheld, derived answer absent -> PASS")

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
