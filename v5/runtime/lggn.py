"""LGGN-Coder: operator library + the retrieve-or-derive repair loop (skeleton).

The graph is a growing library of TYPED REPAIR OPERATORS. Traversal composes them into a repair
trajectory; a frozen LLM REALIZES the trajectory as code; success COMPRESSES back into a
(new/strengthened) operator. See LGGN_DESIGN.md.

Model-agnostic: solve() takes a `decode_fn(prompt, task) -> code` and a `verify_fn(task, code) ->
passed`. The library mechanics — retrieve-by-precondition with confidence/validation gating (the
write-back poison gate), add-or-strengthen, trajectory compression (knowledge not artifacts) — are
pure-python and selftested without a model:

    python -m v5.runtime.lggn --selftest
"""
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path

# Reuse gating: below CONF_FLOOR an operator is NOT retrievable for REUSE (only re-validation) —
# this is what stops `wrong op -> write-back -> retrieved again -> reinforced`. Derived ops start
# UNPROVEN (DERIVED_CONF < CONF_FLOOR) so they cannot be reused until an outcome confirms them.
CONF_FLOOR = 0.40
VAL_FLOOR = 0
SEED_CONF = 0.50
DERIVED_CONF = 0.35
# the retrieve-or-derive boundary: reuse an operator ONLY if the precondition match is strong
# enough; below this we DERIVE (synthesize + mint a new operator). Lexical here; cosine on a box.
MATCH_FLOOR = 0.50


def _toks(s: str) -> set:
    return set(re.findall(r"[a-z0-9_]+", (s or "").lower()))


@dataclass
class Operator:
    """A typed repair transform. Carries uncertainty (confidence/source/age/validation) — load-
    bearing: it gates write-back reuse so a wrong derived operator can't poison future tasks."""
    op_id: str
    name: str
    input_type: str
    output_type: str
    precondition: str          # when it applies (soft: NL; matched lexically here, embedding on a box)
    realize_hint: str          # how the decoder emits it (compiler-mode instruction)
    confidence: float = SEED_CONF
    source: str = "seed"
    age: int = 0
    validation_count: int = 0
    embedding: list | None = None

    def match(self, goal: str, evidence: str = "") -> float:
        """Soft precondition match. Lexical overlap here; swap to embedding cosine on a box (the
        Operator.embedding field is reserved for that). Length-normalized so verbose preconditions
        don't dominate."""
        q = _toks(goal) | _toks(evidence)
        p = _toks(self.precondition) | _toks(self.name) | _toks(self.input_type)
        if not q or not p:
            return 0.0
        return len(q & p) / (len(p) ** 0.5 + 1e-9)


class OperatorLibrary:
    def __init__(self, ops: list[Operator] | None = None):
        self.ops: list[Operator] = ops or []

    # ── persistence ────────────────────────────────────────────────────────────
    @classmethod
    def load(cls, path: str) -> "OperatorLibrary":
        p = Path(path)
        if not p.is_file():
            return cls()
        ops = [Operator(**json.loads(l)) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        return cls(ops)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("\n".join(json.dumps(asdict(o)) for o in self.ops), encoding="utf-8")

    # ── retrieval (conf/val gated = the reuse poison gate) ──────────────────────
    def retrieve(self, goal: str, evidence: str = "", k: int = 3,
                 conf_floor: float = CONF_FLOOR, val_floor: int = VAL_FLOOR,
                 match_floor: float = MATCH_FLOOR) -> list[Operator]:
        cand = [o for o in self.ops if o.confidence >= conf_floor and o.validation_count >= val_floor]
        scored = sorted(((o.match(goal, evidence), o) for o in cand), key=lambda x: -x[0])
        return [o for s, o in scored[:k] if s >= match_floor]   # below match_floor -> DERIVE, not reuse

    # ── uncertainty updates (outcome-driven) ────────────────────────────────────
    def strengthen(self, op: Operator) -> None:
        op.confidence = op.confidence + (1.0 - op.confidence) * 0.3   # toward 1.0
        op.validation_count += 1

    def weaken(self, op: Operator) -> None:
        op.confidence *= 0.6                                          # decays below floor if it keeps failing

    def _similar(self, op: Operator) -> Operator | None:
        for o in self.ops:
            if o.name == op.name and o.input_type == op.input_type and o.output_type == op.output_type:
                return o
        return None

    def add_or_strengthen(self, op: Operator) -> Operator:
        ex = self._similar(op)
        if ex is not None:
            self.strengthen(ex)
            return ex
        self.ops.append(op)
        return op

    # ── write-back: compress a trajectory into knowledge (NOT the artifact) ──────
    def compress_trajectory(self, goal: str, traj: list[Operator], code: str,
                            iid: str, success: bool) -> Operator | None:
        """A solved trajectory becomes a new/strengthened operator. A failed one WEAKENS the ops it
        used (so they fall below the reuse floor if they keep failing). Artifacts (code) are never
        stored — only the compressed transform."""
        if not success:
            for o in traj:
                self.weaken(o)
            return None
        if len(traj) == 1:                       # a known op, confirmed -> strengthen it
            self.strengthen(traj[0])
            return traj[0]
        if len(traj) >= 2:                        # multi-op success -> a COMPOSITE operator (logical unit)
            comp = Operator(
                op_id=f"comp_{iid}_{int(time.time())}",
                name="+".join(o.name for o in traj),
                input_type=traj[0].input_type, output_type=traj[-1].output_type,
                precondition=goal[:200],
                realize_hint=" THEN ".join(o.realize_hint for o in traj),
                confidence=DERIVED_CONF, source=f"derived:{iid}", validation_count=1)
            return self.add_or_strengthen(comp)
        # traj empty = DERIVE branch resolved -> mint a NEW operator from the derived repair
        newop = Operator(
            op_id=f"der_{iid}_{int(time.time())}", name=f"Derived_{iid}",
            input_type="any", output_type="any", precondition=goal[:200],
            realize_hint=(code or "").strip()[:200], confidence=DERIVED_CONF,
            source=f"derived:{iid}", validation_count=1)
        self.ops.append(newop)
        return newop


# ── prompts: the decoder REALIZES a trajectory (compiler mode) or DERIVES (synthesis) ──────────
def render_realize_prompt(goal: str, site: str, traj: list[Operator]) -> str:
    steps = "\n".join(
        f"  {i+1}. {o.name} ({o.input_type} -> {o.output_type}): {o.realize_hint}"
        for i, o in enumerate(traj))
    return ("Realize this REPAIR PROGRAM as code. The diagnosis is DONE — emit ONLY the "
            "search/replace block(s) that implement these steps. Do not re-diagnose.\n"
            f"GOAL: {goal}\nSITE: {site}\nREPAIR PROGRAM:\n{steps}\n")


def label_gold(gold_diff: str) -> list[str]:
    """Operator-MINING bootstrap (V5, heuristic first cut): map a gold unified diff -> the seed
    operator name(s) it realizes, by pattern over its +/- lines. Noisy by design — used to build
    realizer-SFT supervision (operator-plan -> gold) at scale, and as the seed for learned discovery.
    Returns op names in a stable order (a trajectory)."""
    add = [l[1:] for l in gold_diff.splitlines() if l.startswith("+") and not l.startswith("+++")]
    rem = [l[1:] for l in gold_diff.splitlines() if l.startswith("-") and not l.startswith("---")]
    A = "\n".join(add)
    ops: list[str] = []
    def has(rx, s=A):
        return re.search(rx, s, re.M) is not None              # MULTILINE: ^ matches each diff line
    if any(re.match(r"\s*(from\s+\S+\s+import\s|import\s+\S)", l) for l in add):
        ops.append("AddImport")
    if has(r"isinstance\s*\("):
        ops.append("GuardType")
    if has(r"\[\s*:\s*\]\s*=") or (has(r"\.(replace|strip|lower|upper)\s*\(") and has(r"=\s*\w+\.(replace|strip)")):
        ops.append("FixInPlaceMutation")
    if has(r"transaction\.atomic"):
        ops.append("WrapTransaction")
    if has(r"^\s*except\s+\w") or has(r"except\s+\("):
        ops.append("AddExceptHandler")
    if has(r"(is\s+None|if\s+not\s+\w)") and has(r"^\s*(if|return)"):
        ops.append("AddNoneGuard")
    if has(r"(\.size\s*==\s*0|len\s*\([^)]*\)\s*==\s*0|0\s+in\s+\w+\.shape)") and has(r"return"):
        ops.append("AddEmptyInputGuard")
    # TightenConditional: a removed condition line reappears with an added `and`/`or` clause
    for r in rem:
        rc = r.strip()
        if rc.startswith(("if ", "elif ", "while ")) or " = " in rc and ("and" in rc or "or" in rc):
            for a in add:
                if a.strip().startswith(rc.split(" and ")[0].split(" or ")[0][:20]) and \
                        (a.count(" and ") > rc.count(" and ") or a.count(" or ") > rc.count(" or ")):
                    ops.append("TightenConditional"); break
    # UseRealValue: a constant assignment becomes a variable assignment
    for r in rem:
        m = re.search(r"=\s*(\d+|None|True|False|'[^']*'|\"[^\"]*\")\s*$", r.strip())
        if m:
            for a in add:
                if a.strip().split("=")[0] == r.strip().split("=")[0] and not re.search(
                        r"=\s*(\d+|None|True|False)\s*$", a.strip()):
                    ops.append("UseRealValue"); break
    # FixFormatString: a quoted string literal changed (both - and + carry quotes, no structural code)
    if any("'" in r or '"' in r for r in rem) and any("'" in a or '"' in a for a in add) and \
            not ops:
        ops.append("FixFormatString")
    # dedup preserving order
    seen = set(); out = []
    for o in ops:
        if o not in seen:
            seen.add(o); out.append(o)
    return out


def render_op_program(traj: list[Operator]) -> str:
    """Operator trajectory -> free-text EDIT PLAN for a realizer that already adds SR-format boilerplate
    (e.g. swe_slot.fix_user(plan=...)). The decoder REALIZES these typed transforms, not infers them."""
    return ("Apply this repair as a sequence of typed transforms (realize each as code):\n" +
            "\n".join(f"  {i+1}. {o.name} ({o.input_type} -> {o.output_type}): {o.realize_hint}"
                      for i, o in enumerate(traj)))


def render_derive_prompt(goal: str, site: str) -> str:
    return ("No known repair operator matched. Diagnose and fix. Emit ONLY the search/replace "
            f"block(s).\nGOAL: {goal}\nSITE: {site}\n")


# ── the retrieve-or-derive solve loop (POMDP-shaped: act -> observe -> update) ──────────────────
def solve(task: dict, lib: OperatorLibrary, decode_fn, verify_fn, budget: int = 4) -> dict:
    goal = task["issue"]
    site = task.get("site", "")
    evidence = task.get("evidence", "")
    obs = []
    for step in range(budget):
        ops = lib.retrieve(goal, evidence)
        if ops:                                   # REUSE: realize a (hand-composed) trajectory
            traj = [ops[0]]                        # skeleton compose = best op; real traverse later
            prompt = render_realize_prompt(goal, site, traj)
            mode = "realize"
        else:                                     # DERIVE: decoder synthesizes (the hard path)
            traj = []
            prompt = render_derive_prompt(goal, site)
            mode = "derive"
        code = decode_fn(prompt, task)
        passed = bool(verify_fn(task, code))
        obs.append({"step": step + 1, "mode": mode, "ops": [o.name for o in traj], "passed": passed})
        wb = lib.compress_trajectory(goal, traj, code, task.get("iid", "?"), passed)
        if passed:
            return {"status": "solved", "mode": mode, "steps": step + 1,
                    "writeback": (wb.name if wb else None), "code": code, "obs": obs}
        # UPDATE = debugging: mark this approach failed, re-traverse with a changed goal
        goal = goal + f"\n[FAILED attempt {step+1} via {mode}({[o.name for o in traj]}); try a different transform]"
    return {"status": "failed", "obs": obs}


# ── seed operator vocabulary (8, from the SWE failure taxonomy) ────────────────────────────────
def seed_library() -> OperatorLibrary:
    S = SEED_CONF
    ops = [
        Operator("seed_guardtype", "GuardType", "unhandled_type", "handled_type",
                 "a value of an unhandled type reaches a conversion/serialization point (e.g. memoryview at HttpResponse content)",
                 "add an isinstance branch that converts the type before the existing conversions", S),
        Operator("seed_inplace", "FixInPlaceMutation", "pure_result", "applied_result",
                 "result of a pure method (.replace/.strip) is discarded instead of assigned/applied in place",
                 "assign the method result back (output[:] = output.replace(...)) so the change takes effect", S),
        Operator("seed_noneguard", "AddNoneGuard", "possibly_none", "checked",
                 "attribute or index access on a value that can be None/missing",
                 "add an early None/empty check guarding the access", S),
        Operator("seed_emptyguard", "AddEmptyInputGuard", "collection", "nonempty_or_returned",
                 "an operation runs on a collection/array that may be empty or zero-size",
                 "early-return the input unchanged when it is empty / 0 in shape", S),
        Operator("seed_tightencond", "TightenConditional", "condition", "condition",
                 "behavior is gated on a boolean condition that is missing a required clause",
                 "add the missing clause to the condition (e.g. `and connection.features.can_rollback_ddl`)", S),
        Operator("seed_wraptxn", "WrapTransaction", "db_mutation", "atomic_mutation",
                 "a multi-row DB mutation runs without rollback safety / IntegrityError handling",
                 "wrap the mutation in transaction.atomic() and handle IntegrityError", S),
        Operator("seed_fmtstring", "FixFormatString", "format_string", "format_string",
                 "a user-facing error/help format string is inconsistent with the actual accepted format",
                 "correct the format string to match the real behavior", S),
        Operator("seed_widenexcept", "WidenExceptHandler", "narrow_except", "correct_except",
                 "a narrow except clause misses an exception type that is actually raised",
                 "broaden/correct the caught exception type", S),
        Operator("seed_userealvalue", "UseRealValue", "placeholder_constant", "actual_value",
                 "a placeholder/constant (e.g. 1, 0, None) is used where the actual passed variable should be",
                 "replace the constant/placeholder with the real variable in scope", S),
        Operator("seed_addimport", "AddImport", "missing_symbol", "imported_symbol",
                 "a symbol used by the fix is not imported in the module",
                 "add the missing import (at module top or locally) so the fix's symbol resolves", S),
        Operator("seed_addexcept", "AddExceptHandler", "unguarded_op", "guarded_op",
                 "an operation can raise an exception that the caller must catch/handle",
                 "wrap the operation in try/except for the specific exception and handle it", S),
    ]
    return OperatorLibrary(ops)


# ── selftest (no model): exercise the library mechanics that make the design work ──────────────
def _selftest() -> bool:
    print("lggn --selftest: operator library mechanics (retrieve-gate / strengthen / compress)\n")
    lib = seed_library()
    assert len(lib.ops) == 11, "seed vocab size"
    print(f"  seed library: {len(lib.ops)} operators")

    # 1) retrieve: a memoryview goal should surface GuardType
    goal = "memoryview value passed as HttpResponse content is not converted to bytes"
    got = lib.retrieve(goal)
    print(f"  retrieve('memoryview...') -> {[o.name for o in got]}")
    assert got and got[0].name == "GuardType", "precondition match picks GuardType"

    # 2) poison gate: a freshly DERIVED op is unproven (conf<floor) -> NOT retrievable for reuse
    der = Operator("der_x", "BogusFix", "memoryview", "bytes", "memoryview content not converted",
                   "do something", confidence=DERIVED_CONF, source="derived:x")
    lib.ops.append(der)
    assert der not in lib.retrieve(goal), "unproven derived op is gated out of reuse"
    print(f"  derived op conf={der.confidence:.2f} < floor {CONF_FLOOR} -> gated out of reuse (poison gate OK)")
    lib.strengthen(der); lib.strengthen(der)   # two outcome-confirmed validations
    assert der.confidence >= CONF_FLOOR and der in lib.retrieve(goal), "validated derived op becomes reusable"
    print(f"  after 2 validations conf={der.confidence:.2f} (val={der.validation_count}) -> now reusable")

    # 3) compress: failure weakens; success strengthens
    g = lib.ops[0]; c0 = g.confidence
    lib.compress_trajectory(goal, [g], "code", "iidA", success=False)
    assert g.confidence < c0, "failed trajectory weakens its op"
    lib.compress_trajectory(goal, [g], "code", "iidA", success=True)
    print(f"  GuardType conf after fail->success: {c0:.2f} -> {g.confidence:.2f}, val={g.validation_count}")

    # 4) end-to-end loop with fake decoder/verifier:
    #    - a task GuardType covers (gold op = GuardType) -> REUSE -> solved -> strengthen
    #    - a task no op covers -> DERIVE -> mint a new operator (library grows)
    lib2 = seed_library()
    def fake_decode(prompt, task):                    # 'realizes' = returns the op it was told to apply
        m = re.search(r"^  1\. (\w+)", prompt, re.M)
        return f"APPLIED:{m.group(1)}" if m else "DERIVED_FIX"
    def fake_verify(task, code):                      # passes iff the applied op == the task's gold op
        return code == f"APPLIED:{task['gold_op']}" or (code == "DERIVED_FIX" and task.get("gold_op") == "__novel__")
    t_reuse = {"iid": "t1", "issue": "memoryview content not converted to bytes", "gold_op": "GuardType"}
    t_novel = {"iid": "t2", "issue": "a brand new bug pattern never seen before zzz", "gold_op": "__novel__"}
    r1 = solve(t_reuse, lib2, fake_decode, fake_verify, budget=3)
    r2 = solve(t_novel, lib2, fake_decode, fake_verify, budget=3)
    print(f"  solve(reuse): {r1['status']} mode={r1.get('mode')} steps={r1.get('steps')} writeback={r1.get('writeback')}")
    print(f"  solve(novel): {r2['status']} mode={r2.get('mode')} writeback={r2.get('writeback')} | lib {len(seed_library().ops)}->{len(lib2.ops)}")
    assert r1["status"] == "solved" and r1["mode"] == "realize", "known repair solved via REUSE"
    assert r2["status"] == "solved" and r2["mode"] == "derive" and len(lib2.ops) == 12, "novel repair DERIVED + minted a new operator"

    # 5) gold->operator mining (V5 bootstrap): patterns map to the right seed ops
    g_mem = "+        if isinstance(value, memoryview):\n+            return bytes(value)"
    g_txn = ("--- a/x.py\n+++ b/x.py\n+from django.db import transaction\n"
             "+        with transaction.atomic():\n+        except IntegrityError:")
    g_const = "-        cright[-1:, -1:] = 1\n+        cright[-1:, -1:] = right"
    print(f"  label_gold(memoryview) -> {label_gold(g_mem)}")
    print(f"  label_gold(txn+except+import) -> {label_gold(g_txn)}")
    print(f"  label_gold(const->var) -> {label_gold(g_const)}")
    assert "GuardType" in label_gold(g_mem), "memoryview isinstance -> GuardType"
    lt = label_gold(g_txn)
    assert {"AddImport", "WrapTransaction", "AddExceptHandler"} <= set(lt), "txn diff -> 3-op trajectory"
    assert "UseRealValue" in label_gold(g_const), "const->var -> UseRealValue"

    print("\n  LGGN SELFTEST -> PASS")
    return True


def main():
    ap = argparse.ArgumentParser(description="LGGN operator library + repair loop (skeleton).")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seed-out", default="", help="write the seed operator library to this jsonl path")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if _selftest() else 1)
    if a.seed_out:
        seed_library().save(a.seed_out)
        print(f"wrote seed library ({len(seed_library().ops)} ops) -> {a.seed_out}")
        return
    ap.print_help()


if __name__ == "__main__":
    main()
