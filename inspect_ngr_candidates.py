from __future__ import annotations

import argparse
import collections
import json

from train_ngr_v1 import V1ProgressDataset, tuple_key


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--sample", type=int, default=5000)
    args = ap.parse_args()

    rows = []
    with open(args.jsonl, encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    ds = V1ProgressDataset(rows)

    cand_len = collections.Counter()
    allowed_len = collections.Counter()
    cand_action_counts = collections.Counter()
    allowed_action_counts = collections.Counter()
    same_action_set = collections.Counter()
    same_tuple_set = collections.Counter()
    phase_stats = collections.defaultdict(lambda: collections.Counter())
    same_action_wrong_tuple = collections.Counter()

    n = min(len(ds), args.sample)
    for i in range(n):
        item = ds[i]
        row = rows[i]
        phase = row.get("phase")

        allowed = item["allowed_next"]
        candidates = item["candidate_next"]

        a_keys = {tuple_key(x) for x in allowed}
        c_keys = {tuple_key(x) for x in candidates}
        a_actions = {x.get("action") for x in allowed}
        c_actions = {x.get("action") for x in candidates}

        allowed_len[len(allowed)] += 1
        cand_len[len(candidates)] += 1

        for x in allowed:
            allowed_action_counts[x.get("action")] += 1
        for x in candidates:
            cand_action_counts[x.get("action")] += 1

        same_action_set[a_actions == c_actions] += 1
        same_tuple_set[a_keys == c_keys] += 1

        allowed_actions = {str(a.get("action", "")) for a in allowed}
        has_same_action_negative = any(
            str(c.get("action", "")) in allowed_actions and tuple_key(c) not in a_keys
            for c in candidates
        )
        same_action_wrong_tuple[has_same_action_negative] += 1

        phase_stats[phase]["rows"] += 1
        phase_stats[phase][f"cand_len_{len(candidates)}"] += 1
        phase_stats[phase][f"allowed_len_{len(allowed)}"] += 1
        phase_stats[phase][f"same_action_{a_actions == c_actions}"] += 1
        phase_stats[phase][f"same_tuple_{a_keys == c_keys}"] += 1
        phase_stats[phase][f"same_action_negative_{has_same_action_negative}"] += 1

    print("sampled", n)
    print("allowed_len", dict(allowed_len))
    print("candidate_len", dict(cand_len))
    print("same_action_set", dict(same_action_set))
    print("same_tuple_set", dict(same_tuple_set))
    print("same_action_wrong_tuple", dict(same_action_wrong_tuple))
    print("allowed_action_counts", dict(allowed_action_counts))
    print("candidate_action_counts", dict(cand_action_counts))

    print("\nper phase:")
    for phase, c in sorted(phase_stats.items()):
        print(phase, dict(c))

    print("\nexpected v1a5.3 shape:")
    print("  create/link/attach/add/cover: same_action_negative_True should be high")
    print("  noop: candidate_len_1 should dominate")
    print("  stop: candidate_len_1 should dominate")
    print("  candidate_action_counts should no longer be dominated by CREATE/LINK everywhere")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
