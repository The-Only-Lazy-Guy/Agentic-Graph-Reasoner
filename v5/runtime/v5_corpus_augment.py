"""GRR-14d: Corpus augmenter — generate diverse task variants to prevent memorization.

Strategy:
  For each concept atom in MBPP+ (is_prime, is_palindrome, sort_array, etc.),
  generate 30-50 semantically unique tasks that REQUIRE that atom as a helper.
  Each variant has UNIQUE text + UNIQUE asserts so the model cannot memorise
  individual task<->solution pairs.  The only path to solve is:  retrieve the
  known atom from the graph and call it.

Usage:
  # Generate augmented corpus from existing corpus_multi.jsonl
  python -m v5.runtime.v5_corpus_augment --corpus artifacts/corpus_multi.jsonl \\
    --out artifacts/corpus_augmented.jsonl --variants 40

  # Train with augmented corpus
  python -m v5.runtime.algo_lm_train --run \\
    --model Qwen/Qwen2.5-3B-Instruct \\
    --graph graphs/grr_grown.json \\
    --corpus artifacts/corpus_augmented.jsonl \\
    --rounds 20 --batch 16
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER DEFINITIONS — each helper is an MBPP+ atom we want the model to CALL
# ═══════════════════════════════════════════════════════════════════════════════

# Each helper: (name, code_to_import, template generators)
# A template generator yields (text, solution_code, asserts_list) per variant.

HELPER_TEMPLATES: dict[str, dict] = {}


def _register(name: str, code: str):
    """Decorator to register template generators."""
    def wrap(fn):
        HELPER_TEMPLATES[name] = {"code": code, "gen": fn}
        return fn
    return wrap


# ── is_prime ─────────────────────────────────────────────────────────────

_IS_PRIME_CODE = """def is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True"""


@_register("is_prime", _IS_PRIME_CODE)
def _gen_is_prime_variants(n: int) -> list[tuple[str, str, list[str]]]:
    """Generate n variants of tasks that check if numbers are prime."""
    rng = random.Random(42)
    variants = []

    def _pairs(v):
        return [(v[i], v[i+1]) for i in range(0, len(v)-1, 2)]

    prime_nums = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
                  59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
                  127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181,
                  191, 193, 197, 199]
    non_primes_below_200 = [n for n in range(0, 201)
                            if n not in prime_nums and n not in (0, 1)]
    non_primes_below_200 = [0, 1, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21,
                            22, 24, 25, 26, 27, 28, 30, 32, 33, 34, 35, 36, 38,
                            39, 40, 42, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55,
                            56, 57, 58, 60, 62, 63, 64, 65, 66, 68, 69, 70, 72,
                            74, 75, 76, 77, 78, 80, 81, 82, 84, 85, 86, 87, 88,
                            90, 91, 92, 93, 94, 95, 96, 98, 99, 100]

    templates = [
        "Write a function {name}(n) that returns True if n is a prime number and False otherwise.",
        "Write a function {name}(x) to check whether x is a prime number.",
        "Create a function {name}(num) that determines if a given positive integer num is prime.",
        "Write a function {name}(value) that returns True when value is a prime, False if not.",
        "Implement {name}(number) which returns True for prime numbers and False for composite numbers.",
        "Write a function {name}(a) to test if a is a prime number.",
        "Create a boolean function {name}(p) that checks if p is a prime.",
        "Write a function {name}(n) that returns True if and only if n is prime.",
        "Implement a primality test function {name}(candidate) that returns True if candidate is prime.",
        "Write a function {name}(integer) that returns True for prime integers, False otherwise.",
    ]

    for i in range(n):
        idx = i % len(templates)
        fn = f"check_prime_{i}"
        tmpl = templates[idx].replace("{name}", fn)

        # Pick test cases
        rng.shuffle(prime_nums)
        rng.shuffle(non_primes_below_200)
        test_primes = prime_nums[:3 + (i % 4)]
        test_nonprimes = non_primes_below_200[:3 + (i % 3)]
        asserts = []
        for p in test_primes:
            asserts.append(f"assert {fn}({p}) == True")
        for np_val in test_nonprimes:
            asserts.append(f"assert {fn}({np_val}) == False")

        code = f"def {fn}(n):\n    return is_prime(n)"
        variants.append((tmpl, code, asserts))

    return variants


# ── is_not_prime ─────────────────────────────────────────────────────────

_IS_NOT_PRIME_CODE = """def is_not_prime(n):
    return not is_prime(n)"""


@_register("is_not_prime", _IS_NOT_PRIME_CODE)
def _gen_is_not_prime_variants(n: int) -> list[tuple[str, str, list[str]]]:
    rng = random.Random(43)
    variants = []

    templates = [
        "Write a function {name}(n) that returns True if n is NOT a prime number.",
        "Write a function {name}(x) that identifies composite (non-prime) numbers.",
        "Create a function {name}(num) that returns True when num is not prime.",
        "Write a function {name}(value) that checks if a number is composite.",
        "Implement {name}(number) that returns True for non-prime numbers.",
    ]

    for i in range(n):
        fn = f"is_composite_{i}"
        tmpl = templates[i % len(templates)].replace("{name}", fn)

        rng = random.Random(100 + i)
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        composites = [0, 1, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22,
                      24, 25, 26, 27, 28, 30]
        rng.shuffle(primes)
        rng.shuffle(composites)

        asserts = []
        for c in composites[:3 + (i % 3)]:
            asserts.append(f"assert {fn}({c}) == True")
        for p in primes[:2 + (i % 2)]:
            asserts.append(f"assert {fn}({p}) == False")

        code = f"def {fn}(n):\n    return is_not_prime(n)"
        variants.append((tmpl, code, asserts))

    return variants


# ── is_palindrome ────────────────────────────────────────────────────────

_IS_PALINDROME_CODE = """def is_palindrome(s):
    s = str(s)
    return s == s[::-1]"""


@_register("is_palindrome", _IS_PALINDROME_CODE)
def _gen_palindrome_variants(n: int) -> list[tuple[str, str, list[str]]]:
    rng = random.Random(44)
    variants = []

    templates = [
        "Write a function {name}(s) that returns True if string s is a palindrome.",
        "Write a function {name}(text) to check if text reads the same forwards and backwards.",
        "Create a function {name}(word) that tests whether a word is a palindrome.",
        "Write a function {name}(value) that checks if a string is a palindrome.",
        "Implement {name}(input_str) that returns True for palindromic strings.",
        "Write a function {name}(phrase) to determine if a phrase is a palindrome.",
        "Create a function {name}(token) that returns True when token is a palindrome.",
        "Write a function {name}(x) that checks if x reads the same in both directions.",
    ]

    palindromes = ["racecar", "level", "madam", "radar", "civic", "refer",
                   "tenet", "kayak", "deified", "rotator", "repaper", "solos",
                   "stats", "redder", "xxx", "xxxx", "xxxxx"]
    non_palindromes = ["hello", "world", "python", "test", "code", "data",
                       "algorithm", "function", "variable", "recursion",
                       "iteration", "composition"]

    for i in range(n):
        fn = f"check_palindrome_{i}"
        tmpl = templates[i % len(templates)].replace("{name}", fn)

        rng.shuffle(palindromes)
        rng.shuffle(non_palindromes)
        asserts = []
        for p in palindromes[:2 + (i % 2)]:
            asserts.append(f"assert {fn}('{p}') == True")
        for np_val in non_palindromes[:2 + (i % 3)]:
            asserts.append(f"assert {fn}('{np_val}') == False")

        code = f"def {fn}(s):\n    return is_palindrome(str(s))"
        variants.append((tmpl, code, asserts))

    return variants


# ── similar_elements ─────────────────────────────────────────────────────

_SIMILAR_ELEMENTS_CODE = """def similar_elements(t1, t2):
    return tuple(set(t1) & set(t2))"""


@_register("similar_elements", _SIMILAR_ELEMENTS_CODE)
def _gen_similar_variants(n: int) -> list[tuple[str, str, list[str]]]:
    rng = random.Random(45)
    variants = []

    templates = [
        "Write a function {name}(list1, list2) that returns the common elements between two lists.",
        "Write a function {name}(a, b) that finds elements present in both sequences.",
        "Create a function {name}(xs, ys) that returns the intersection of two lists.",
        "Write a function {name}(first, second) to find shared elements.",
        "Implement {name}(collection1, collection2) that returns elements appearing in both collections.",
        "Write a function {name}(seq1, seq2) that returns the overlapping elements.",
        "Create a function {name}(left, right) to compute set intersection.",
        "Write a function {name}(data1, data2) that finds values common to both lists.",
    ]

    word_pools = [
        (["apple", "banana", "cherry", "date"], ["banana", "date", "elderberry"]),
        (["red", "blue", "green", "yellow"], ["blue", "yellow", "purple"]),
        (["cat", "dog", "fish"], ["dog", "bird", "fish"]),
        (["x", "y", "z"], ["y", "z"]),
        (["one", "two", "three"], ["two", "three", "four"]),
        (["alpha", "beta", "gamma"], ["gamma", "delta", "epsilon"]),
    ]

    for i in range(n):
        fn = f"common_elements_{i}"
        tmpl = templates[i % len(templates)].replace("{name}", fn)

        if i < len(word_pools):
            a, b = word_pools[i]
        else:
            rng = random.Random(200 + i)
            pool = [chr(ord('a') + j) for j in range(10)]
            rng.shuffle(pool)
            a = pool[:4 + (i % 3)]
            rng.shuffle(pool)
            b = pool[:3 + (i % 4)]

        common = list(set(a) & set(b))
        code = f"def {fn}(a, b):\n    return similar_elements(a, b)"
        asserts = [
            f"assert set({fn}({a!r}, {b!r})) == set({common!r})",
        ]

        variants.append((tmpl, code, asserts))

    return variants


# ── long_words (filter by word length) ───────────────────────────────────

_LONG_WORDS_CODE = """def long_words(words, n):
    return [w for w in words if len(w) > n]"""


@_register("long_words", _LONG_WORDS_CODE)
def _gen_long_words_variants(n: int) -> list[tuple[str, str, list[str]]]:
    rng = random.Random(46)
    variants = []

    templates = [
        "Write a function {name}(words, min_len) that returns words longer than min_len characters.",
        "Write a function {name}(lst, threshold) filtering elements by minimum length.",
        "Create a function {name}(items, size) that keeps only items with length > size.",
        "Write a function {name}(strings, k) returning strings with more than k characters.",
        "Implement {name}(tokens, limit) that filters tokens longer than limit.",
        "Write a function {name}(entries, min) to get entries with length exceeding min.",
    ]

    word_lists = [
        (["hello", "world", "a", "of", "the", "python", "code"], 3),
        (["cat", "elephant", "dog", "bird", "ant"], 3),
        (["short", "tiny", "medium", "longer", "extralong"], 4),
        (["a", "ab", "abc", "abcd", "abcde"], 2),
        (["x", "xx", "xxx", "xxxx", "xxxxx"], 1),
    ]

    for i in range(n):
        fn = f"words_longer_{i}"
        tmpl = templates[i % len(templates)].replace("{name}", fn)

        if i < len(word_lists):
            words, k = word_lists[i]
        else:
            rng = random.Random(300 + i)
            word_pool = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta",
                         "eta", "theta", "iota", "kappa"]
            rng.shuffle(word_pool)
            words = word_pool[:5 + (i % 3)]
            k = 3 + (i % 4)

        expected = [w for w in words if len(w) > k]
        code = f"def {fn}(words, n):\n    return long_words(words, n)"
        asserts = [
            f"assert {fn}({words!r}, {k}) == {expected!r}",
        ]

        variants.append((tmpl, code, asserts))

    return variants


# ── sort_array ───────────────────────────────────────────────────────────

_SORT_ARRAY_CODE = """def sort_array(arr):
    return sorted(arr)"""


@_register("sort_array", _SORT_ARRAY_CODE)
def _gen_sort_variants(n: int) -> list[tuple[str, str, list[str]]]:
    rng = random.Random(47)
    variants = []

    templates = [
        "Write a function {name}(numbers) that returns the sorted version of a list of numbers.",
        "Write a function {name}(arr) that sorts an array of integers ascending.",
        "Create a function {name}(items) that sorts a list in non-decreasing order.",
        "Write a function {name}(data) that orders a sequence from smallest to largest.",
        "Implement {name}(values) to sort a list of numbers in ascending order.",
        "Write a function {name}(lst) returning a sorted copy of the input list.",
    ]

    for i in range(n):
        fn = f"sort_asc_{i}"
        tmpl = templates[i % len(templates)].replace("{name}", fn)

        rng = random.Random(400 + i)
        length = 3 + (i % 6)
        arr = [rng.randint(-20, 50) for _ in range(length)]

        code = f"def {fn}(arr):\n    return sort_array(arr)"
        asserts = [
            f"assert {fn}({arr!r}) == {sorted(arr)!r}",
        ]

        variants.append((tmpl, code, asserts))

    return variants


# ── merge_sorted_list ────────────────────────────────────────────────────

_MERGE_SORTED_CODE = """def merge_sorted_list(l1, l2):
    return sorted(l1 + l2)"""


@_register("merge_sorted_list", _MERGE_SORTED_CODE)
def _gen_merge_variants(n: int) -> list[tuple[str, str, list[str]]]:
    rng = random.Random(48)
    variants = []

    templates = [
        "Write a function {name}(a, b) that merges two sorted lists into one sorted list.",
        "Write a function {name}(left, right) to merge two sorted sequences.",
        "Create a function {name}(first, second) that combines two sorted arrays into sorted output.",
        "Write a function {name}(list1, list2) returning a merged sorted list.",
        "Implement {name}(xs, ys) that takes two sorted lists and returns their sorted union.",
        "Write a function {name}(arr1, arr2) to merge two sorted lists maintaining sort order.",
    ]

    for i in range(n):
        fn = f"merge_sorted_{i}"
        tmpl = templates[i % len(templates)].replace("{name}", fn)

        rng = random.Random(500 + i)
        a = sorted([rng.randint(-10, 30) for _ in range(2 + (i % 4))])
        b = sorted([rng.randint(-10, 30) for _ in range(2 + (i % 3))])

        code = f"def {fn}(a, b):\n    return merge_sorted_list(a, b)"
        asserts = [
            f"assert {fn}({a!r}, {b!r}) == {sorted(a + b)!r}",
        ]

        variants.append((tmpl, code, asserts))

    return variants


# ═══════════════════════════════════════════════════════════════════════════════
# AUGMENTER
# ═══════════════════════════════════════════════════════════════════════════════

def augment_corpus(corpus_path: str, out_path: str, variants_per_helper: int = 40):
    """Load corpus, augment with diverse variants, save augmented corpus."""
    with open(corpus_path) as f:
        corpus = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(corpus)} corpus tasks")

    # Index existing tasks by name
    existing_names = {t.get("name"): t for t in corpus}

    augmented = list(corpus)  # start with original corpus

    for helper_name, helper_def in HELPER_TEMPLATES.items():
        code = helper_def["code"]
        gen_fn = helper_def["gen"]

        # Generate variants
        variants = gen_fn(variants_per_helper)
        print(f"  {helper_name}: generating {len(variants)} variants")

        for text, solution_code, asserts in variants:
            # Create a unique task record
            fn_name = re.search(r"def (\w+)\s*\(", solution_code).group(1)
            rec = {
                "text": text,
                "code": solution_code,
                "asserts": asserts,
                "plus_test": "",  # no plus tests for variants
                "setup": "",
                "n_plus": 0,
                "name": fn_name,
                "source": "augmented",
                "pipeline_shaped": False,
            }
            augmented.append(rec)

    # Write
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for rec in augmented:
            f.write(json.dumps(rec) + "\n")

    n_new = len(augmented) - len(corpus)
    print(f"\nAugmented corpus: {len(corpus)} original + {n_new} new = {len(augmented)} total -> {out_path}")
    print(f"  Helpers used: {', '.join(HELPER_TEMPLATES.keys())}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="artifacts/corpus_multi.jsonl")
    ap.add_argument("--out", default="artifacts/corpus_augmented.jsonl")
    ap.add_argument("--variants", type=int, default=40)
    a = ap.parse_args()

    augment_corpus(a.corpus, a.out, variants_per_helper=a.variants)


if __name__ == "__main__":
    main()
