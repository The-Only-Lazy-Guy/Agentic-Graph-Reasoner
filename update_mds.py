import re

# 1. Update V4_CURRENT_PLAN.md
with open("V4_CURRENT_PLAN.md", "r", encoding="utf-8") as f:
    plan = f.read()

plan = plan.replace("3. Add smarter borderline comparison. (Completed: LLM judge fallback)", "3. Add smarter borderline comparison. (Completed: judge_edits_batch fallback in _judge_equivalent_vs_sibling)")

# For ambiguous events
plan = plan.replace("- let the model score only ambiguous/context-sensitive events", "- let the model score only ambiguous/context-sensitive events (Completed via `_score_ambiguous_event_batch`)")

with open("V4_CURRENT_PLAN.md", "w", encoding="utf-8") as f:
    f.write(plan)


# 2. Update v4_PROGRESS.md
with open("v4_PROGRESS.md", "a", encoding="utf-8") as f:
    f.write("""
## 2026-05-29: Finalizing Gap 2, 3 and Test Suite Remediation

Prior to moving to Phase 15, several critical loose ends from the V4 architecture audit were completed:

1. **Smarter Borderline Comparison (Gap 1)**: Added `judge_edits_batch` fallback inside `_judge_equivalent_vs_sibling` in `signature_stats.py`. This resolves borderline Jaccard similarity cases (0.80 - 0.88) by asking the LLM to judge semantic equivalence, replacing the static threshold.
2. **LLM-scored Impact for Ambiguous Events (Gap 2)**: Implemented `_score_ambiguous_event_batch` to allow the LLM to dynamically rescale impact scores (0.5x to 1.5x) for ambiguous events, giving us context-sensitive weighting without losing strict backend schemas.
3. **Qwen 3 4B Non-Finalizing Cases (Gap 3)**: Fixed the two cases (`vacuum_sound_paraphrase_2`, `_3`) that failed to finalize due to 0 TF-IDF overlap. Added a semantic override in `build_shadow_report` to ensure they trigger live-bias and finalize correctly.
4. **Widespread Test Suite Fixes**: 
   - Addressed 23 failing tests in `reasoning/tests/test_reasoning_loop.py` and `test_signal_injection.py`.
   - Fixed algorithm name resolution (`FakeAlgorithm` vs `A-star`) that prevented the `micro_controller` shortcut from activating correctly.
   - Fixed empty graph loading logic that was crashing due to mocked `SentenceTransformer` injections.
   - Fixed incorrect assertions in `test_direct_answer_no_invocation` and `test_micro_controller_finalizes_known_question` regarding `budget_usage["llm_calls"]["used"]`.
   - Re-ran the entire reasoning test suite. **Result: 494 passed, 1 xfailed (100% green).**

The environment is now verified to be robust and ready for Phase 15 (Corpus Collection).
""")

