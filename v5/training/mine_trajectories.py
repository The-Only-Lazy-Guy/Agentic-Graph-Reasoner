"""Mine MULTI-STEP operator trajectories from agentic coding sessions -> the POMDP data for the
multi-step traverse + topology kill-test (LGGN_DESIGN T.5/T.9/V7).

Heavy sessions (>= --min-edits edits) -> ORDERED (operator, observation) trajectories. Preserves
Op->Obs->Op (T.9), NOT flattened. Each edit -> soft_assign over the GROWN operator library (seed SWE
ops + novel mined atoms, T.8). Generic over any Claude-Code-format session dir (Fable or other agentic
trace datasets share the transcript schema), so add a dataset by pointing --root at it.

  # Fable heavy sessions (local HF cache):
  python -m v5.training.mine_trajectories --root "<HF_CACHE_SNAPSHOT_DIR>" --min-edits 4 \
      --seed-ops data/swe/discovered_ops_emb.jsonl --out data/traj/fable_heavy.jsonl

  python -m v5.training.mine_trajectories --selftest      # synthetic, no network/model
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import ast

from v5.training.ingest_fable5 import _read_events, _parts, _is_edit, _edit_payload, _goal

_OBS_TYPES = ("tool_result", "function_call_output", "toolresult", "tool_output")


def _args(p: dict) -> dict:
    """arguments as a dict, parsing JSON or Python-repr strings (toolCall format uses single-quote repr)."""
    a = p.get("arguments") or p.get("input") or {}
    if isinstance(a, dict):
        return a
    if isinstance(a, str):
        for parse in (json.loads, ast.literal_eval):
            try:
                v = parse(a)
                if isinstance(v, dict):
                    return v
            except Exception:
                pass
    return {}


def _is_obs(part: dict, role: str) -> bool:
    pt = (part.get("type") or "").lower()
    return any(t in pt for t in _OBS_TYPES) or "result" in pt or role == "tool"


def _obs_text(part: dict) -> str:
    c = part.get("content") or part.get("output") or part.get("text") or part.get("result") or ""
    if isinstance(c, list):                                  # tool_result content can be a parts list
        c = " ".join(str(x.get("text") or x.get("content") or x) if isinstance(x, dict) else str(x) for x in c)
    return str(c)[:600]


def session_trajectory(events: list, sid: str = "") -> dict:
    """Ordered session -> {goal, session_id, steps:[{op_edit, file, obs}]}. Each edit carries the
    observation that FOLLOWED it (tool_result / next tool output) -> the POMDP transition signal."""
    goal = _goal(events)
    steps = []
    for ev in events:
        role = ((ev.get("message") or {}).get("role") or ev.get("role") or "").lower()
        for p in _parts(ev):
            if _is_edit(p):
                args = _args(p)
                steps.append({"edit": _edit_payload(p),
                              "file": args.get("file_path") or args.get("path") or "",
                              "obs": ""})
            elif _is_obs(p, role) and steps and not steps[-1]["obs"]:
                steps[-1]["obs"] = _obs_text(p)              # attach observation to the last edit
    return {"goal": goal, "session_id": sid, "steps": [s for s in steps if s["edit"].strip()]}


def find_sessions(root: str) -> list[Path]:
    return [Path(p) for p in glob.glob(str(Path(root) / "**" / "*.jsonl"), recursive=True)
            if "history" not in Path(p).name]


def _embed_batched(texts, embed_fn, bs=48):
    import numpy as np
    out = []
    for i in range(0, len(texts), bs):
        out.extend(embed_fn(texts[i:i + bs]))
    return np.asarray(out, dtype=float)


def mine(root: str, min_edits: int, seed_ops_path: str, out: str, tau: float = 0.45, k_new: int = 16):
    import numpy as np
    import torch
    from v5.training.providers import RealEmbedder
    from v5.runtime.operator_discovery import grow_library, soft_assign
    emb = RealEmbedder(torch.device("cpu"))

    def embed_fn(ts):
        d = emb.embed_nodes({str(i): t[:1000] for i, t in enumerate(ts)})
        return [d[str(i)] for i in range(len(ts))]

    files = find_sessions(root)
    heavy = []
    for fp in files:
        tr = session_trajectory(_read_events(fp), fp.stem)
        if len(tr["steps"]) >= min_edits:
            heavy.append(tr)
    print(f"[mine] {len(files)} sessions -> {len(heavy)} heavy (>= {min_edits} edits)", flush=True)
    if not heavy:
        print("[mine] no heavy sessions at this threshold; lower --min-edits"); return
    flat = [(h_i, s_i, s["edit"]) for h_i, h in enumerate(heavy) for s_i, s in enumerate(h["steps"])]
    print(f"[mine] embedding {len(flat)} edits...", flush=True)
    X = _embed_batched([e for _h, _s, e in flat], embed_fn)
    seed = [json.loads(l) for l in Path(seed_ops_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    grown, n_novel = grow_library(seed, X, tau=tau, k_new=k_new)
    print(f"[mine] library: {len(seed)} seed -> {len(grown)} grown (+{len(grown)-len(seed)} minted, {n_novel} novel edits)", flush=True)
    for (h_i, s_i, _e), x in zip(flat, X):                   # assign each edit -> soft op
        sa = soft_assign(x, grown, tau=tau)
        heavy[h_i]["steps"][s_i].update(op=sa["best"], op_soft=sa["soft"], op_residual=round(sa["residual"], 3))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for h in heavy:
            f.write(json.dumps({"goal": h["goal"][:1500], "session_id": h["session_id"],
                                "trajectory": [{"op": s["op"], "op_soft": s["op_soft"], "obs": s["obs"][:300],
                                                "file": s["file"]} for s in h["steps"]]}) + "\n")
    Path(out).with_name("grown_ops.jsonl").write_text("\n".join(json.dumps(o) for o in grown), encoding="utf-8")
    lens = [len(h["steps"]) for h in heavy]
    obs_cov = np.mean([1.0 if s["obs"] else 0.0 for h in heavy for s in h["steps"]])
    multistep_ops = np.mean([len({s["op"] for s in h["steps"]}) >= 2 for h in heavy])
    print(f"[mine] -> {out}  ({len(heavy)} trajectories)")
    print(f"  steps/traj: mean={np.mean(lens):.1f} median={int(np.median(lens))} max={max(lens)}")
    print(f"  observation coverage: {obs_cov:.0%} of steps have an obs (the POMDP signal)")
    print(f"  >=2 DISTINCT ops per traj: {multistep_ops:.0%} (real multi-operator decomposition)")
    print(f"  grown library -> {Path(out).with_name('grown_ops.jsonl')}")


def _selftest() -> bool:
    print("mine_trajectories --selftest: synthetic heavy session -> POMDP trajectory (no model)\n")
    events = [
        {"message": {"role": "user", "content": [{"type": "text", "text": "Build a retry decorator + tests."}]}},
        {"message": {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "write the decorator"},
            {"type": "tool_use", "name": "write", "arguments": {"file_path": "retry.py", "content": "def retry(n): ..."}}]}},
        {"message": {"role": "user", "content": [{"type": "tool_result", "content": "created retry.py"}]}},
        {"message": {"role": "assistant", "content": [
            {"type": "tool_use", "name": "write", "arguments": {"file_path": "test_retry.py", "content": "def test(): ..."}}]}},
        {"message": {"role": "user", "content": [{"type": "tool_result", "content": "1 failed: NameError"}]}},
        {"message": {"role": "assistant", "content": [
            {"type": "tool_use", "name": "edit", "arguments": {"file_path": "retry.py", "new_string": "import functools"}}]}},
    ]
    tr = session_trajectory(events, "s1")
    for s in tr["steps"]:
        print(f"  edit file={s['file']:14} obs={s['obs'][:34]!r}")
    assert len(tr["steps"]) == 3, f"expected 3 edits, got {len(tr['steps'])}"
    assert tr["steps"][0]["obs"].startswith("created"), "obs attached to its edit"
    assert "failed" in tr["steps"][1]["obs"], "failure obs preserved (revision signal)"
    assert tr["goal"].startswith("Build a retry"), "goal"
    print("\n  MINE_TRAJECTORIES SELFTEST -> PASS (Op->Obs->Op preserved)")
    return True


def main():
    ap = argparse.ArgumentParser(description="Mine multi-step operator trajectories from agentic sessions.")
    ap.add_argument("--root", help="dir of Claude-Code-format session .jsonl files")
    ap.add_argument("--min-edits", type=int, default=4)
    ap.add_argument("--seed-ops", default="data/swe/discovered_ops_emb.jsonl")
    ap.add_argument("--out", default="data/traj/fable_heavy.jsonl")
    ap.add_argument("--tau", type=float, default=0.45)
    ap.add_argument("--k-new", type=int, default=16)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if _selftest() else 1)
    if not a.root:
        ap.error("--root required (dir of session .jsonl), or use --selftest")
    mine(a.root, a.min_edits, a.seed_ops, a.out, tau=a.tau, k_new=a.k_new)


if __name__ == "__main__":
    main()
