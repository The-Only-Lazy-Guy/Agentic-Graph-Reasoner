"""Probe: what node/edge types does the model naturally generate?
Run on remote: python -m v5.v5_probe_graph_schema --real
Local: python -m v5.v5_probe_graph_schema (stub mode)
"""
import json, sys, os, re

# ── Probe prompts ──
# Each probe: which schema type, what to ask the model, what patterns to look for
PROBES = [
    # --- Node type probes ---
    {
        "schema": "node:implementation",
        "prompt": "Write a function `unique_elements(lst)` that returns unique elements preserving insertion order. Use existing library functions if helpful.",
        "key_patterns": ["def unique_elements", "return"],
        "anti_patterns": ["def similar_elements", "def is_not_prime"],
    },
    {
        "schema": "node:trial_knowledge",
        "prompt": 'I tried two approaches for deduplication: (1) using set() which is O(n) but loses order, (2) using dict.fromkeys() which preserves order. The dict approach is better when order matters. Consider this tradeoff when choosing an approach.',
        "key_patterns": ["approach", "tried", "tradeoff", "instead of"],
        "anti_patterns": [],
    },
    {
        "schema": "node:explanation",
        "prompt": "Explain how the `similar_elements` function works. It takes two lists, converts them to sets, finds the intersection with the & operator, and returns the result as a list.",
        "key_patterns": ["list", "set", "intersection", "convert", "return"],
        "anti_patterns": ["def "],
    },
    {
        "schema": "node:syntax_explain",
        "prompt": "Explain the Python syntax for finding the intersection of two sets: `set1 & set2` uses the & operator which is overloaded for sets to mean intersection. This is O(min(len(s1), len(s2))) average case.",
        "key_patterns": ["&", "operator", "overload", "set"],
        "anti_patterns": [],
    },
    {
        "schema": "node:warning",
        "prompt": "WARNING: `is_not_prime(n)` only works reliably for n >= 0. For negative n, it returns True (negative numbers are not prime). The sqrt-based implementation may overflow for very large n (>10^15). Use a deterministic Miller-Rabin for cryptographic applications.",
        "key_patterns": ["WARNING", "edge case", "only works", "may", "careful"],
        "anti_patterns": [],
    },
    {
        "schema": "node:recipe",
        "prompt": "RECIPE for finding the top N elements:\n1. Sort the list in descending order\n2. Take the first N elements with slicing [:n]\n3. Return the result\nFor large lists, use heapq.nlargest(n, lst) instead (O(n log k) vs O(n log n)).",
        "key_patterns": ["1.", "2.", "3.", "step", "first", "then"],
        "anti_patterns": [],
    },
    {
        "schema": "node:comparison",
        "prompt": "COMPARE: heap_queue_largest vs sorted(nums, reverse=True)[:n].\n- heap_queue_largest: O(n log k), uses heapq, memory efficient for large n\n- sorted: O(n log n), simpler code, better for small lists\n- Use heap_queue_largest when k << n, use sorted when code simplicity matters more.",
        "key_patterns": ["COMPARE", "vs", "instead", "better", "faster", "simpler"],
        "anti_patterns": [],
    },
    # --- Edge type probes ---
    {
        "schema": "edge:depend",
        "prompt": "Consider two functions in a codebase:\n- `unique_elements(lst)`: returns unique elements preserving order\n- `sort_list(lst)`: returns sorted list\n\nDoes `unique_elements` depend on `sort_list`? Only if it calls sort_list internally. If they are independent, there is no dependency.",
        "key_patterns": ["depend", "calls", "uses", "independent", "not depend", "no dependency"],
        "anti_patterns": [],
    },
    {
        "schema": "edge:related",
        "prompt": "Consider two functions:\n- `is_prime(n)`: checks if n is prime\n- `is_not_prime(n)`: checks if n is NOT prime\n\nAre these related? They work on the same concept (primality) but from opposite directions. If they share conceptual grounding, they are related even without calling each other.",
        "key_patterns": ["related", "concept", "same", "opposite", "complementary", "similar"],
        "anti_patterns": [],
    },
    {
        "schema": "edge:part_of",
        "prompt": "Consider:\n- `similar_elements(list1, list2)`: finds elements common to both lists\n- Concept: list_operations\n\nIs `similar_elements` part of the list_operations concept? It operates on list inputs and produces list outputs, fitting naturally under list operations.",
        "key_patterns": ["part of", "belongs", "category", "concept", "under", "fits"],
        "anti_patterns": [],
    },
    {
        "schema": "edge:contradicts",
        "prompt": "Two approaches to deduplication:\n- `unique_via_set(lst)`: uses set() — O(n) but loses order\n- `unique_via_dict(lst)`: uses dict.fromkeys() — O(n) and preserves order in Python 3.7+\n\nDo these contradict each other as approaches? They solve the same problem with different tradeoffs. If they give different results for the same input (one preserves order, one doesn't), they represent alternative/contradicting strategies.",
        "key_patterns": ["alternative", "contradict", "different", "tradeoff", "instead of", "vs"],
        "anti_patterns": [],
    },
]

def detect_schema(text: str) -> str:
    """Classify what schema type the generated text represents."""
    lower = text.lower()
    lines = [l.strip() for l in text.splitlines()]
    
    # Check for implementation (has a function definition)
    if any(l.startswith("def ") or l.startswith("```python") for l in lines[:5]):
        if any("return" in text for _ in [1]):
            return "implementation"
    
    # Check for warnings
    if "warning" in lower[:200]:
        return "warning"
    
    # Check for recipes
    if any(l.startswith(("1.", "2.", "3.", "step")) for l in lines[:10]):
        return "recipe"
    
    # Check for comparisons
    if "compare" in lower[:200] or " vs " in lower[:300]:
        return "comparison"
    
    # Check for trial knowledge
    if any(x in lower for x in ["i tried", "attempt", "approach", "attempted", "experiment"]):
        return "trial_knowledge"
    
    # Check for edge classifications
    if any(x in lower for x in ["depends on", "dependency", "calls", "uses", "independent", "no dependency"]):
        return "edge:depend"
    if any(x in lower for x in ["related", "complementary", "conceptual"]):
        return "edge:related"
    if any(x in lower for x in ["part of", "belongs to", "fits under", "category"]):
        return "edge:part_of"
    if any(x in lower for x in ["contradict", "alternative", "tradeoff", "instead of"]):
        return "edge:contradicts"
    
    # Check for explanation
    if any(x in lower for x in ["explain", "how it works", "this means", "because"]):
        return "explanation"
    
    # Check for syntax explain
    if any(x in lower for x in ["operator", "syntax", "overload"]):
        return "syntax_explain"
    
    return "unclear"


def stub_gen(prompts):
    """Stub generator for testing without GPU."""
    known = {
        "unique": "def unique_elements(lst):\n    return list(dict.fromkeys(lst))",
        "approach": "I tried two approaches. The first uses set() but loses order. The second uses dict.fromkeys() and preserves order. The tradeoff is between simplicity and correctness.",
        "similar_elements": "similar_elements converts both lists to sets, finds the intersection with &, and returns list(set(list1) & set(list2)).",
        "&": "The & operator on sets performs intersection. It's syntactic sugar for set1.intersection(set2). Python overloads & for sets only.",
        "WARNING": "WARNING: is_not_prime assumes non-negative input. For n < 0 it returns True. For large n > 10^15 the sqrt may overflow.",
        "RECIPE": "RECIPE:\n1. Sort descending\n2. Slice [:n]\n3. Return\nFor large lists, use heapq.nlargest instead.",
        "COMPARE": "COMPARE: heap_queue_largest (O(n log k)) vs sorted (O(n log n)). Use heap for large lists with small k, sorted for simplicity.",
        "depend": "unique_elements does NOT depend on sort_list. They are independent functions operating on lists.",
        "related": "is_prime and is_not_prime are related — they check opposite conditions on the same primality concept.",
        "part of": "similar_elements is part of list_operations — it operates on lists and returns list results.",
        "contradict": "unique_via_set and unique_via_dict contradict each other: one sacrifices order for speed, the other preserves order with the same O(n) complexity.",
    }
    results = []
    for p in prompts:
        matched = None
        for key, val in known.items():
            if key.lower() in p.lower():
                matched = val
                break
        results.append(matched or "I processed your request.")
    return results


def run_probes(probes, gen_fn, out_path="artifacts/schema_probe.jsonl"):
    """Run all probes, classify outputs, report results."""
    prompts = [p["prompt"] for p in probes]
    responses = gen_fn(prompts) if not isinstance(gen_fn("x"), list) else gen_fn(prompts)
    
    results = []
    print(f"{'#':>2} {'Schema Type':28s} {'Generated → Detected':30s} Result")
    print("-" * 80)
    
    for i, (probe, resp) in enumerate(zip(probes, responses)):
        detected = detect_schema(resp)
        expected = probe["schema"]
        
        # Check key patterns
        keys_ok = sum(1 for k in probe["key_patterns"] if k.lower() in resp.lower())
        anti_ok = sum(1 for a in probe["anti_patterns"] if a.lower() not in resp.lower())
        pattern_score = keys_ok + anti_ok
        total_patterns = len(probe["key_patterns"]) + len(probe["anti_patterns"])
        pattern_match = pattern_score >= max(1, total_patterns * 0.5)
        
        schema_ok = detected == expected.split(":")[1] if ":" in expected else detected == expected
        
        passed = schema_ok and pattern_match
        
        status = "PASS" if passed else "FAIL"
        note = ""
        if not schema_ok:
            note = f" (expected={expected.split(':')[-1]}, got={detected})"
        elif not pattern_match:
            note = " (pattern mismatch)"
        
        print(f"{i:>2} {expected:28s} {detected:30s} [{status}]{note}")
        
        results.append({
            "i": i, "schema": expected, "prompt": probe["prompt"],
            "generation": resp, "detected": detected,
            "passed": passed, "note": note.strip(),
        })
    
    # Save
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    print("-" * 80)
    passed = sum(1 for r in results if r["passed"])
    print(f"  {passed}/{len(results)} probes passed -> {out_path}")
    print(f"  Verdict: ", end="")
    if passed >= len(results) * 0.6:
        print("Model CAN produce diverse graph schema")
    elif passed >= len(results) * 0.3:
        print("Partial - may need prompt tuning for some types")
    else:
        print("Model struggles - need stronger prompt engineering")
    
    return results


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Probe graph schema capabilities")
    ap.add_argument("--real", action="store_true", help="use real HF model (GPU)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--lora", help="LoRA adapter path")
    ap.add_argument("--out", default="artifacts/schema_probe.jsonl")
    a = ap.parse_args()
    
    if a.real:
        # Lazy import only on GPU
        from v5.runtime.algo_lm_proposer import make_hf_gen
        gen_fn = make_hf_gen(a.model, temperature=0.7, max_new_tokens=300, lora_path=a.lora)
    else:
        gen_fn = stub_gen
    
    run_probes(PROBES, gen_fn, out_path=a.out)


if __name__ == "__main__":
    main()
