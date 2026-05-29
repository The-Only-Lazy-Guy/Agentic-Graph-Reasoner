"""Debug checker plugin behavior for specific answers."""
import sys; sys.path.insert(0, ".")
from reasoning.substrate_v2 import (
    CheckerRegistry, StepContextPacket, StepResult, parse_step_result,
    _haystack, DeltaTransaction, StateDelta,
    _check_zero_downtime_migration,
    _check_payment_crash_recovery,
    _check_inventory_reservation,
    _check_dynamic_max_subarray,
    _check_shortest_path_safety,
    _check_generic_step_format,
)

checks = [
    ("zero_downtime_migration", "unsafe_cutover_claim",
     "Take a maintenance window over the weekend, stop writes, copy all data to the new database, then switch the connection string.",
     "Migrate a monolith orders table from MySQL to a new sharded Spanner-like database with zero downtime."),
    ("inventory_reservation", "cache_authority_claim",
     "Use Redis as the source of truth for inventory counts. It is fast enough to handle the flash sale volume.",
     "Design an inventory reservation system for a flash sale where thousands of concurrent requests compete for limited units."),
    ("inventory_reservation", "global_mutex_claim",
     "Use a global distributed mutex to serialize all inventory reservation requests.",
     "Design an inventory reservation system for a flash sale."),
    ("inventory_reservation", "reservation_lifecycle_missing",
     "Assign each SKU a dedicated partition owner. The owner serializes reservations and decrements inventory atomically. If a payment fails, the inventory is not released.",
     "Design an inventory reservation system for a flash sale."),
    ("dynamic_connectivity_deletions", "per_query_traversal",
     "For each connectivity query after edge deletions, run a BFS or DFS from the source node to see if the target is reachable.",
     "For offline dynamic connectivity with edge deletions and connectivity queries."),
    ("dynamic_max_subarray", "segment_tree_merge_missing",
     "Use a segment tree where each node stores the sum, prefix max, suffix max, and best subarray sum. Merge by combining left sum with right sum etc.",
     "Maintain an array under online point updates and report the maximum non-empty subarray sum after each update."),
    ("dynamic_max_subarray", "long_long_missing",
     "Use a segment tree where each node stores sum, prefix, suffix, and best. Merge with best = max(left.best, right.best, left.suffix + right.prefix). All fields are integers.",
     "Maintain an array under online point updates and report the maximum non-empty subarray sum after each update. Values can be up to 1e9."),
    ("shortest_path_safety", "dijkstra_negative_edge",
     "Dijkstra's algorithm is safe. It maintains a priority queue of the closest unprocessed node and processes each edge exactly once.",
     "Can Dijkstra be trusted on a directed graph with edges s->a weight 2, a->b weight -4, and s->b weight 5?"),
    ("generic_step_format", "missing_required",
     "STEP_RESULT\nstatus: need_info\nresult: I need more information.\ndelta:\n  decisions:\n    - ask clarification\nEND_STEP_RESULT",
     "What is the time complexity of merge sort?"),
    ("generic_step_format", "delta_dropped",
     "STEP_RESULT\nstatus: resolved\nresult: Merge sort is O(n log n).\ndelta:\n  decisions:\n    - none\nEND_STEP_RESULT",
     "What is the time complexity of merge sort?"),
    ("payment_crash_recovery", "exactly_once_claim",
     "Use exactly-once delivery semantics to ensure the PSP is never double-charged.",
     "A payment worker may crash after sending a charge request but before persisting the response."),
    ("payment_crash_recovery", "psp_2pc_claim",
     "Use a distributed two-phase commit protocol between the payment worker and the PSP.",
     "A payment worker may crash after sending a charge request."),
]

for plugin, expected, answer, question in checks:
    packet = StepContextPacket(
        task_summary=question, focus=question, looking_for="answer",
        active_signals=[], hard_constraints=[],
        parent_decisions=[], open_gaps=[],
    )
    step_result = parse_step_result(answer)
    hay = _haystack(step_result)
    if step_result.delta_transaction.status == "dropped":
        step_result = StepResult(
            status="resolved", result=answer,
            delta_transaction=DeltaTransaction(status="parsed", delta=StateDelta()),
        )

    registry = CheckerRegistry([plugin])
    check = registry.verify(step_result, packet)
    codes = [v.code for v in check.violations]
    passed = expected in codes
    status = "PASS" if passed else f"FAIL (expected={expected}, got={codes})"
    print(f"{plugin:35s} {status}")
