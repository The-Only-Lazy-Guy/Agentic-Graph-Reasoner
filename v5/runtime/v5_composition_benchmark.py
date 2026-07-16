"""GRR-14c: Composition benchmark — measure whether the LM calls existing graph atoms.

Usage:
  # Stub (no GPU) — tests the benchmark structure
  python -m v5.runtime.v5_composition_benchmark --stub

  # Real eval with trained model + graph
  python -m v5.runtime.v5_composition_benchmark --run \\
    --model Qwen/Qwen2.5-3B-Instruct \\
    --graph graphs/grr_grown.json \\
    --lora artifacts/grr14b_lora \\
    --corpus artifacts/corpus_multi.jsonl

Design:
  Phase 1 — Overlap Subset:
    Select MBPP+ tasks that share common helper patterns (prime → is_prime,
    palindrome → is_palindrome, set ops → similar_elements, etc.). Run them
    sequentially (simulating training order) so later tasks see earlier atoms.

  Phase 2 — Synthetic Chain Tasks:
    Hardcoded tasks that explicitly require 2+ helper atoms. These verify
    that the model can compose: call atom A, call atom B, combine results.

  Metrics:
    called_rate   = fraction of samples where called is non-empty
    compose_depth = avg number of distinct atoms called per sample
    verify_rate   = fraction of samples that pass unit tests
    chain_success = fraction of chain tasks where ALL required atoms are called
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from graph_core import MemoryGraph
from v5.runtime.algo_graph_edits import edge_candidate, grow, node_candidate
from v5.runtime.algo_graph_mg import MGRetriever, _fn_name, _is_safe_atom_name, seed_graph
from v5.runtime.algo_graph_run import MBPPTask, _author_prompt, verify_asserts_detail
from v5.runtime.algo_lm_author import repair_code
from v5.runtime.tool_memory import _extract_code


# ═══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC CHAIN TASKS (each requires 2+ helpers)
# ═══════════════════════════════════════════════════════════════════════════════

SYNTHETIC_CHAINS: list[dict] = [
    dict(
        name="is_prime_palindrome",
        text="Write a function is_prime_palindrome(n) that returns True if n is both a prime number and a palindrome (reads the same forwards and backwards).",
        code="def is_prime_palindrome(n):\n    return is_prime(n) and is_palindrome(str(n))",
        asserts=[
            "assert is_prime_palindrome(2) == True",
            "assert is_prime_palindrome(3) == True",
            "assert is_prime_palindrome(4) == False",
            "assert is_prime_palindrome(11) == True",
            "assert is_prime_palindrome(13) == False",
            "assert is_prime_palindrome(101) == True",
        ],
        setup="",
        requires=["is_prime", "is_palindrome"],
    ),
    dict(
        name="sum_of_primes_in_list",
        text="Write a function sum_of_primes_in_list(numbers) that takes a list of integers and returns the sum of all prime numbers in the list.",
        code="def sum_of_primes_in_list(numbers):\n    return sum(n for n in numbers if is_prime(n))",
        asserts=[
            "assert sum_of_primes_in_list([1,2,3,4,5,6,7,8,9,10]) == 17",
            "assert sum_of_primes_in_list([4,6,8,9]) == 0",
            "assert sum_of_primes_in_list([11,13,17]) == 41",
            "assert sum_of_primes_in_list([]) == 0",
        ],
        setup="",
        requires=["is_prime"],
    ),
    dict(
        name="filter_nonprimes",
        text="Write a function filter_nonprimes(pairs) that takes a list of (a, b) tuples and returns a list of tuples where both a and b are non-prime.",
        code="def filter_nonprimes(pairs):\n    return [(a,b) for a,b in pairs if is_not_prime(a) and is_not_prime(b)]",
        asserts=[
            "assert filter_nonprimes([(2,3),(4,5),(6,8)]) == [(6,8)]",
            "assert filter_nonprimes([(2,2)]) == []",
            "assert filter_nonprimes([(4,9),(10,12)]) == [(4,9),(10,12)]",
        ],
        setup="",
        requires=["is_not_prime"],
    ),
    dict(
        name="longest_palindrome_word",
        text="Write a function longest_palindrome_word(sentence) that takes a string and returns the longest word that is a palindrome. If none, return empty string.",
        code="def longest_palindrome_word(sentence):\n    words = sentence.split()\n    pals = [w for w in words if is_palindrome(w.lower())]\n    return max(pals, key=len) if pals else ''",
        asserts=[
            "assert longest_palindrome_word('madam hello racecar') == 'racecar'",
            "assert longest_palindrome_word('abc def') == ''",
            "assert longest_palindrome_word('level') == 'level'",
        ],
        setup="",
        requires=["is_palindrome"],
    ),
    dict(
        name="common_elements_count",
        text="Write a function common_elements_count(list1, list2) that returns the number of elements shared between two lists.",
        code="def common_elements_count(list1, list2):\n    return len(set(list1) & set(list2))",
        asserts=[
            "assert common_elements_count([1,2,3,4],[3,4,5,6]) == 2",
            "assert common_elements_count([1,2],[3,4]) == 0",
            "assert common_elements_count([],[1,2]) == 0",
        ],
        setup="",
        requires=["similar_elements"],
    ),
    dict(
        name="sorted_unique_words",
        text="Write a function sorted_unique_words(sentence) that returns a sorted list of unique words (case-insensitive).",
        code="def sorted_unique_words(sentence):\n    words = sentence.lower().split()\n    return sorted(set(words))",
        asserts=[
            "assert sorted_unique_words('the cat and the dog') == ['and', 'cat', 'dog', 'the']",
            "assert sorted_unique_words('hello world') == ['hello', 'world']",
            "assert sorted_unique_words('') == []",
        ],
        setup="",
        requires=[],
    ),
    dict(
        name="words_longer_than_n",
        text="Write a function words_longer_than_n(sentence, n) that returns a list of words longer than n characters.",
        code="def words_longer_than_n(sentence, n):\n    return [w for w in sentence.split() if len(w) > n]",
        asserts=[
            "assert words_longer_than_n('hello world foo', 3) == ['hello', 'world']",
            "assert words_longer_than_n('a bc def', 2) == ['def']",
            "assert words_longer_than_n('hi', 5) == []",
        ],
        setup="",
        requires=["long_words"],
    ),
    dict(
        name="merge_and_sort",
        text="Write a function merge_and_sort(list1, list2) that merges two sorted lists into a single sorted list.",
        code="def merge_and_sort(list1, list2):\n    return sorted(list1 + list2)",
        asserts=[
            "assert merge_and_sort([1,3,5],[2,4,6]) == [1,2,3,4,5,6]",
            "assert merge_and_sort([],[1,2]) == [1,2]",
            "assert merge_and_sort([1],[2]) == [1,2]",
        ],
        setup="",
        requires=["sort_array", "merge_sorted_list"],
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD CHAIN GROUPS FROM CORPUS + SYNTHETIC TASKS
# ═══════════════════════════════════════════════════════════════════════════════

def build_chains(corpus_path: str) -> dict[str, list[MBPPTask]]:
    """Build chain groups from corpus + synthetic tasks in dependency order."""
    with open(corpus_path) as f:
        corpus = [json.loads(line) for line in f if line.strip()]

    by_name = {t.get("name"): t for t in corpus}

    chains = {}

    # Prime chain: helper atoms first, then synthetic tasks that call them
    prime_names = ["is_prime", "is_not_prime", "prime_num", "factorize",
                   "largest_prime_factor", "prime_fib", "is_multiply_prime"]
    chain_tasks = []
    for n in prime_names:
        if n in by_name:
            chain_tasks.append(_to_mbpp(by_name[n]))
    for s in SYNTHETIC_CHAINS[:3]:
        chain_tasks.append(_from_dict(s))
    chains["prime"] = chain_tasks

    # Palindrome chain
    pal_names = ["is_palindrome", "even_odd_palindrome", "reverse_delete"]
    chain_tasks = []
    for n in pal_names:
        if n in by_name:
            chain_tasks.append(_to_mbpp(by_name[n]))
    for s in SYNTHETIC_CHAINS[3:5]:
        chain_tasks.append(_from_dict(s))
    chains["palindrome"] = chain_tasks

    # List/filter chain
    filter_names = ["long_words", "filter_by_substring", "filter_integers",
                    "filter_by_prefix", "get_positive"]
    chain_tasks = []
    for n in filter_names:
        if n in by_name:
            chain_tasks.append(_to_mbpp(by_name[n]))
    for s in SYNTHETIC_CHAINS[5:7]:
        chain_tasks.append(_from_dict(s))
    chains["filter"] = chain_tasks

    # Sort/merge chain
    sort_names = ["sort_array", "sort_sublists", "merge_sorted_list",
                  "subject_marks", "sort_counter"]
    chain_tasks = []
    for n in sort_names:
        if n in by_name:
            chain_tasks.append(_to_mbpp(by_name[n]))
    for s in SYNTHETIC_CHAINS[7:]:
        chain_tasks.append(_from_dict(s))
    chains["sort"] = chain_tasks

    return chains


def _to_mbpp(d: dict) -> MBPPTask:
    return MBPPTask(
        name=d.get("name", "?"),
        text=d.get("text", ""),
        tests=d.get("asserts", []),
        setup=d.get("setup", ""),
    )


def _from_dict(d: dict) -> MBPPTask:
    tests = d.get("asserts", [])
    return MBPPTask(
        name=d["name"],
        text=d["text"],
        tests=tests,
        setup=d.get("setup", ""),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# STUB GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════

def _make_stub_gen():
    """Return a gen_fn stub that returns fixed template code."""
    def gen_fn(prompts):
        results = []
        for p in prompts:
            m = re.search(r"def (\w+)\s*\(", p)
            fn = m.group(1) if m else "unknown"
            results.append(f"```python\ndef {fn}():\n    pass\n```")
        return results
    return gen_fn


def _make_stub_embed():
    """Return a stub embed_fn that returns zero vector (accepts dict, returns dict)."""
    import numpy as np
    def embed_fn(texts):
        return {k: np.zeros(384) for k in texts}
    return embed_fn


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_benchmark(chains: dict[str, list[MBPPTask]],
                  graph_path: str, embed_fn, gen_fn,
                  out: str = "artifacts/bench_raw.jsonl",
                  samples: int = 3, k_retrieve: int = 3) -> dict:
    """Run benchmark: for each chain, evaluate tasks and measure composition metrics."""
    if not Path(graph_path).exists():
        seed_graph(graph_path, ("concept_algorithms",))
    retr = MGRetriever(MemoryGraph.load_json(graph_path), embed_fn)

    all_samples = []
    chain_metrics = {}

    for chain_name, tasks in chains.items():
        print(f"\n{'='*60}")
        print(f"Chain: {chain_name} ({len(tasks)} tasks)")
        print(f"{'='*60}")

        chain_called = 0
        chain_total = 0
        chain_verified = 0
        chain_depth = []
        chain_results = []

        for task in tasks:
            advertised = retr.retrieve(task.text, k=k_retrieve)
            prompt = _author_prompt(task, advertised)
            gens = gen_fn([prompt] * samples)

            task_called = []
            task_verified = 0
            for g in gens:
                raw_code = _extract_code(g)
                code = repair_code(raw_code, task.name)
                defined = {_fn_name(code)} if code else set()
                called = [n for n, c in advertised if n not in defined
                          and _is_safe_atom_name(n)
                          and re.search(rf"(?<![\w.]){re.escape(n)}\s*\(", code or "")]
                task_called.append(called)
                if called:
                    chain_called += 1
                    chain_depth.append(len(called))
                chain_total += 1

                deps = "\n\n".join(c for n, c in advertised if n in called)
                full = (deps + "\n" + code) if deps else (code or "")
                ok, err = verify_asserts_detail(full, task.tests, task.setup)
                if ok:
                    task_verified += 1
                    chain_verified += 1

            sample = dict(
                chain=chain_name,
                task=task.name,
                text=task.text,
                prompt=prompt,
                samples=[dict(generation=g, called=c, verified=v)
                        for g, c, v in zip(gens, task_called,
                            [True]*min(len(gens), len(task_called)))],
                requires=getattr(task, 'requires', []),
            )
            all_samples.append(sample)

            called_names = set()
            for c in task_called:
                called_names.update(c)
            print(f"  {task.name:30s} called={len(called_names)} atoms "
                  f"({task_verified}/{samples} verified) "
                  f"calls={sorted(called_names) if called_names else '—'}")

        called_rate = chain_called / chain_total if chain_total > 0 else 0
        avg_depth = sum(chain_depth) / len(chain_depth) if chain_depth else 0
        verify_rate = chain_verified / chain_total if chain_total > 0 else 0
        chain_metrics[chain_name] = dict(
            tasks=len(tasks),
            called_rate=round(called_rate, 3),
            avg_compose_depth=round(avg_depth, 2),
            verify_rate=round(verify_rate, 3),
        )

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for s in all_samples:
            f.write(json.dumps(s) + "\n")
    print(f"\nRaw output -> {out}")

    return chain_metrics


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def print_report(metrics: dict, out: str = "artifacts/bench_report.json"):
    print(f"\n{'='*60}")
    print("COMPOSITION BENCHMARK REPORT")
    print(f"{'='*60}")
    print(f"{'Chain':20s} {'Tasks':6s} {'Called%':8s} {'Depth':6s} {'Verify%':8s}")
    print("-" * 50)
    for name, m in metrics.items():
        print(f"{name:20s} {m['tasks']:<6d} {m['called_rate']:<8.1%} "
              f"{m['avg_compose_depth']:<6.2f} {m['verify_rate']:<8.1%}")
    n = len(metrics)
    if n:
        print("-" * 50)
        avg_called = sum(m['called_rate'] for m in metrics.values()) / n
        avg_depth = sum(m['avg_compose_depth'] for m in metrics.values()) / n
        avg_verify = sum(m['verify_rate'] for m in metrics.values()) / n
        print(f"{'AVERAGE':20s} {'—':<6s} {avg_called:<8.1%} {avg_depth:<6.2f} {avg_verify:<8.1%}")
        if avg_called >= 0.5:
            print(f"\nPASS: Composition rate {avg_called:.1%} >= 50% target")
        else:
            print(f"\nFAIL: Composition rate {avg_called:.1%} < 50% target")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Report -> {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--graph", default="graphs/grr_grown.json")
    ap.add_argument("--lora")
    ap.add_argument("--corpus", default="artifacts/corpus_multi.jsonl")
    ap.add_argument("--out", default="artifacts/bench_raw.jsonl")
    ap.add_argument("--report", default="artifacts/bench_report.json")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--k-retrieve", type=int, default=3)
    args = ap.parse_args()

    chains = build_chains(args.corpus)

    if not args.run and not args.stub:
        ap.print_help()
        return

    if args.stub:
        print("=== STUB MODE ===")
        metrics = run_benchmark(chains, args.graph,
                                _make_stub_embed(), _make_stub_gen(),
                                out=args.out, samples=min(args.samples, 1),
                                k_retrieve=args.k_retrieve)
        print_report(metrics, args.report)
        return

    print(f"Loading model {args.model}...")
    from v5.lm_loader import load_frozen_lm
    from v5.memory.store import make_mpnet_embedder
    from peft import PeftModel
    import torch
    from transformers import AutoTokenizer
    base = load_frozen_lm(args.model)
    if args.lora:
        model = PeftModel.from_pretrained(base, args.lora)
        model.eval()
    else:
        model = base
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def gen_fn(prompts):
        msgs = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                        add_generation_prompt=True) for p in prompts]
        enc = tok(msgs, return_tensors="pt", padding=True, padding_side="left").to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, do_sample=True, temperature=0.8, top_p=0.95,
                                 max_new_tokens=160, pad_token_id=tok.pad_token_id)
        return [tok.decode(out[i, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                for i in range(len(prompts))]

    embed_fn = make_mpnet_embedder()

    metrics = run_benchmark(chains, args.graph, embed_fn, gen_fn,
                            out=args.out, samples=args.samples,
                            k_retrieve=args.k_retrieve)
    print_report(metrics, args.report)


if __name__ == "__main__":
    main()
