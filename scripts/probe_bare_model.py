"""Probe bare-model uncertainty on a mixed question bank.

For each question:
  1. Ask the bare model (no graph) to answer + self-rate confidence
  2. Parse out: answer, confidence score, uncertain claims
  3. Score hedging language density

Output: data/uncertainty_probe/results.jsonl
        data/uncertainty_probe/summary.json

Usage:
    # opencode serve running on :4096
    python scripts/probe_bare_model.py
    python scripts/probe_bare_model.py --limit 10   # quick test
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Question bank — mixed difficulty, mixed domains, some graph-covered, some not
# ---------------------------------------------------------------------------

QUESTION_BANK = [
    # ---- CS / Algorithms (graph-covered) ----
    {"q": "What is binary search and what is its time complexity?",
     "domain": "cs", "expected_difficulty": "easy", "graph_covered": True},
    {"q": "Explain why Dijkstra's algorithm fails on graphs with negative edge weights.",
     "domain": "cs", "expected_difficulty": "medium", "graph_covered": True},
    {"q": "What is a Fenwick tree and how does the lowbit trick (i & -i) work?",
     "domain": "cs", "expected_difficulty": "medium", "graph_covered": True},
    {"q": "Explain the difference between a segment tree and a Fenwick tree. When would you choose one over the other?",
     "domain": "cs", "expected_difficulty": "medium", "graph_covered": True},
    {"q": "What is Kadane's algorithm for maximum subarray sum? Walk through it on [-2, 1, -3, 4, -1, 2, 1, -5, 4].",
     "domain": "cs", "expected_difficulty": "medium", "graph_covered": True},
    {"q": "Design a system that can answer 'what is my rank?' queries in O(log n) for 500K concurrent users with real-time score updates.",
     "domain": "cs", "expected_difficulty": "hard", "graph_covered": True},

    # ---- CS / Systems (partially graph-covered) ----
    {"q": "What is a race condition? Give a concrete example with shared mutable state.",
     "domain": "cs_systems", "expected_difficulty": "medium", "graph_covered": True},
    {"q": "Explain the CAP theorem and give a real-world example of each trade-off.",
     "domain": "cs_systems", "expected_difficulty": "medium", "graph_covered": False},
    {"q": "What is consistent hashing and why is it used in distributed systems?",
     "domain": "cs_systems", "expected_difficulty": "medium", "graph_covered": False},

    # ---- Math (partially graph-covered) ----
    {"q": "Prove that the square root of 2 is irrational.",
     "domain": "math", "expected_difficulty": "medium", "graph_covered": True},
    {"q": "What is the difference between pointwise and uniform convergence of a sequence of functions?",
     "domain": "math", "expected_difficulty": "hard", "graph_covered": True},
    {"q": "Explain the Banach fixed-point theorem and give an example application.",
     "domain": "math", "expected_difficulty": "hard", "graph_covered": False},

    # ---- Physics (partially graph-covered) ----
    {"q": "Why can light travel through space but sound cannot?",
     "domain": "physics", "expected_difficulty": "easy", "graph_covered": True},
    {"q": "Explain the photoelectric effect and why it supports the particle theory of light.",
     "domain": "physics", "expected_difficulty": "medium", "graph_covered": True},
    {"q": "What is renormalization in quantum field theory and why is it necessary?",
     "domain": "physics", "expected_difficulty": "hard", "graph_covered": False},

    # ---- Chemistry (partially graph-covered) ----
    {"q": "What determines whether a chemical reaction is exothermic or endothermic?",
     "domain": "chemistry", "expected_difficulty": "easy", "graph_covered": True},
    {"q": "Explain Le Chatelier's principle with a concrete equilibrium example.",
     "domain": "chemistry", "expected_difficulty": "medium", "graph_covered": True},

    # ---- Biology (graph-covered) ----
    {"q": "What is the difference between mitosis and meiosis?",
     "domain": "biology", "expected_difficulty": "easy", "graph_covered": True},

    # ---- Software Engineering (graph-covered) ----
    {"q": "What is the difference between unit tests and integration tests? When would you skip one?",
     "domain": "software_eng", "expected_difficulty": "easy", "graph_covered": True},
    {"q": "Explain SOLID principles. Which one is most frequently violated in practice?",
     "domain": "software_eng", "expected_difficulty": "medium", "graph_covered": True},

    # ---- Out-of-domain (NOT in graph) ----
    {"q": "Explain how a transformer neural network's self-attention mechanism works, including the Q/K/V matrices.",
     "domain": "ml", "expected_difficulty": "hard", "graph_covered": False},
    {"q": "What is the difference between REST and GraphQL? When would you choose one over the other?",
     "domain": "web", "expected_difficulty": "medium", "graph_covered": False},
    {"q": "Explain how Bitcoin's proof-of-work consensus mechanism works and its energy implications.",
     "domain": "crypto", "expected_difficulty": "medium", "graph_covered": False},
    {"q": "What is the Riemann hypothesis and why does it matter for prime number distribution?",
     "domain": "math", "expected_difficulty": "hard", "graph_covered": False},
    {"q": "Explain the Byzantine Generals Problem and how PBFT solves it.",
     "domain": "distributed", "expected_difficulty": "hard", "graph_covered": False},
]


# ---------------------------------------------------------------------------
# Bare-model probe prompt
# ---------------------------------------------------------------------------

PROBE_SYSTEM = """\
Answer the question below. Then, after your answer, add a self-assessment block:

<confidence>
score: [1-10 integer, where 1=guessing, 5=somewhat sure, 10=certain]
uncertain_claims:
- [list any specific claims in your answer you're not fully confident about]
- [leave empty if you're confident in everything]
hedges: [count of times you used words like "might", "possibly", "I think", "not sure", "approximately", "roughly"]
</confidence>

Be honest in the self-assessment. It's better to flag uncertainty than to fake confidence.
"""


# ---------------------------------------------------------------------------
# Hedging detection
# ---------------------------------------------------------------------------

_HEDGE_WORDS = re.compile(
    r"\b(might|may|maybe|possibly|perhaps|probably|could be|"
    r"i think|i believe|not sure|not certain|uncertain|"
    r"approximately|roughly|around|about|seems|appears|"
    r"likely|unlikely|arguably|debatable|unclear)\b",
    re.IGNORECASE,
)


def count_hedges(text: str) -> int:
    return len(_HEDGE_WORDS.findall(text or ""))


def parse_confidence_block(text: str) -> Dict[str, Any]:
    """Extract the <confidence>...</confidence> block."""
    m = re.search(r"<confidence>(.*?)</confidence>", text or "", re.DOTALL | re.IGNORECASE)
    if not m:
        return {"score": None, "uncertain_claims": [], "hedges": count_hedges(text)}
    block = m.group(1)
    score_m = re.search(r"score\s*:\s*(\d+)", block)
    score = int(score_m.group(1)) if score_m else None
    uncertain = re.findall(r"^-\s+(.+)$", block, re.MULTILINE)
    uncertain = [u.strip() for u in uncertain if u.strip() and u.strip() != "[]" and "empty" not in u.lower()]
    return {
        "score": score,
        "uncertain_claims": uncertain,
        "hedges": count_hedges(text),
    }


# ---------------------------------------------------------------------------
# Model caller
# ---------------------------------------------------------------------------

def _find_opencode() -> str:
    for cand in ("opencode", "opencode.cmd"):
        p = shutil.which(cand)
        if p:
            return p
    raise FileNotFoundError("opencode not on PATH")


def call_bare_model(question: str, *, server_url: str, config_dir: str, model: str, timeout: float = 120) -> Dict[str, Any]:
    exe = _find_opencode()
    use_shell = sys.platform == "win32" and exe.endswith(".cmd")
    env = dict(os.environ)
    env["OPENCODE_CONFIG_DIR"] = config_dir
    cmd = [exe, "run", "--model", model, "--format", "json"]
    if server_url:
        cmd += ["--attach", server_url]

    prompt = f"{PROBE_SYSTEM}\n\n---\n\nQuestion: {question}"
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        timeout=timeout,
        encoding="utf-8",
        errors="replace",
        shell=use_shell,
        env=env,
    )
    elapsed = time.time() - t0

    text_parts: List[str] = []
    for raw in proc.stdout.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            evt = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if evt.get("type") == "text":
            text_parts.append(evt.get("part", {}).get("text", ""))
    answer = "".join(text_parts).strip()
    if not answer and proc.returncode != 0:
        raise RuntimeError(f"opencode failed: {proc.stderr[:300]}")
    return {"answer": answer, "elapsed_sec": round(elapsed, 1)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0, help="max questions to probe (0=all)")
    ap.add_argument("--server", default="http://127.0.0.1:4096")
    ap.add_argument("--model", default="opencode/big-pickle")
    ap.add_argument("--config-dir", default=r"C:\Users\Ace\AppData\Local\Temp\opencode-empty-config")
    ap.add_argument("--output-dir", default="data/uncertainty_probe")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"

    bank = QUESTION_BANK
    if args.limit > 0:
        bank = bank[:args.limit]

    print(f"Probing {len(bank)} questions with bare model ({args.model})")
    print(f"Output: {results_path}\n")

    results: List[Dict[str, Any]] = []
    for i, item in enumerate(bank):
        q = item["q"]
        print(f"[{i+1}/{len(bank)}] {q[:70]}{'...' if len(q) > 70 else ''}")
        try:
            resp = call_bare_model(q, server_url=args.server, config_dir=args.config_dir, model=args.model)
            conf = parse_confidence_block(resp["answer"])
            hedge_count = count_hedges(resp["answer"])
            row = {
                **item,
                "answer": resp["answer"],
                "elapsed_sec": resp["elapsed_sec"],
                "self_score": conf["score"],
                "uncertain_claims": conf["uncertain_claims"],
                "hedge_count": hedge_count,
                "answer_length": len(resp["answer"]),
            }
            print(f"  confidence={conf['score']}/10  hedges={hedge_count}  "
                  f"uncertain_claims={len(conf['uncertain_claims'])}  "
                  f"elapsed={resp['elapsed_sec']}s")
        except Exception as e:
            row = {**item, "answer": "", "error": str(e), "self_score": None, "hedge_count": 0}
            print(f"  ERROR: {e}")
        results.append(row)
        with results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Summary
    scores = [r["self_score"] for r in results if r.get("self_score") is not None]
    hedges = [r["hedge_count"] for r in results]

    summary = {
        "total_questions": len(results),
        "answered": sum(1 for r in results if r.get("answer")),
        "mean_confidence": round(sum(scores) / len(scores), 2) if scores else None,
        "low_confidence_count": sum(1 for s in scores if s <= 5),
        "high_confidence_count": sum(1 for s in scores if s >= 8),
        "mean_hedges": round(sum(hedges) / len(hedges), 2) if hedges else 0,
        "by_domain": {},
        "by_graph_covered": {"covered": [], "not_covered": []},
    }
    for r in results:
        domain = r.get("domain", "?")
        summary["by_domain"].setdefault(domain, []).append(r.get("self_score"))
        bucket = "covered" if r.get("graph_covered") else "not_covered"
        summary["by_graph_covered"][bucket].append(r.get("self_score"))

    # Aggregate per domain
    for domain, scores_list in summary["by_domain"].items():
        valid = [s for s in scores_list if s is not None]
        summary["by_domain"][domain] = {
            "count": len(scores_list),
            "mean_confidence": round(sum(valid) / len(valid), 2) if valid else None,
        }
    for bucket in ("covered", "not_covered"):
        valid = [s for s in summary["by_graph_covered"][bucket] if s is not None]
        summary["by_graph_covered"][bucket] = {
            "count": len(valid),
            "mean_confidence": round(sum(valid) / len(valid), 2) if valid else None,
        }

    # Top uncertain questions
    uncertain = sorted(
        [r for r in results if r.get("self_score") is not None],
        key=lambda r: (r["self_score"], -r["hedge_count"]),
    )
    summary["most_uncertain"] = [
        {"q": r["q"][:80], "score": r["self_score"], "hedges": r["hedge_count"], "domain": r["domain"]}
        for r in uncertain[:5]
    ]

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"\n=== SUMMARY ===")
    print(f"  answered: {summary['answered']}/{summary['total_questions']}")
    print(f"  mean confidence: {summary['mean_confidence']}/10")
    print(f"  low confidence (<=5): {summary['low_confidence_count']}")
    print(f"  high confidence (>=8): {summary['high_confidence_count']}")
    print(f"  mean hedges: {summary['mean_hedges']}")
    print(f"\n  Graph-covered: mean={summary['by_graph_covered']['covered']['mean_confidence']}")
    print(f"  Not covered:   mean={summary['by_graph_covered']['not_covered']['mean_confidence']}")
    print(f"\n  Most uncertain:")
    for u in summary["most_uncertain"]:
        print(f"    [{u['score']}/10] ({u['domain']}) {u['q']}")
    print(f"\nFull results: {results_path}")
    print(f"Summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
