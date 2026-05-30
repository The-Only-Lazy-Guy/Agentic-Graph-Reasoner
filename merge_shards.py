"""Merge corpus shards (local + pushed from other machines) into one corpus.

After other machines push data/corpus_shards/<run-id>.jsonl and you `git pull`,
run this to concatenate all shards (+ optional extra files) into a single corpus,
de-duplicated, ready for v5.training.corpus_scaling.

    python merge_shards.py --out data/corpus_merged.jsonl
    python -m v5.training.corpus_scaling --corpus data/corpus_merged.jsonl
"""
import argparse, glob, json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards-dir", default="data/corpus_shards")
    ap.add_argument("--extra", nargs="*", default=["artifacts/phase15_50/corpus50.jsonl"],
                    help="extra corpus files to fold in (e.g. the original local run)")
    ap.add_argument("--out", default="data/corpus_merged.jsonl")
    a = ap.parse_args()

    files = sorted(glob.glob(f"{a.shards_dir}/*.jsonl"))
    files += [f for f in a.extra if Path(f).exists()]
    print(f"merging {len(files)} files:")
    seen, rows, dup = set(), [], 0
    for f in files:
        n = 0
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            # de-dup by (session_id, task_id) so re-pushed/overlapping rows collapse
            key = (r.get("session_id"), (r.get("input", {}) or {}).get("question"))
            if key in seen:
                dup += 1; continue
            seen.add(key); rows.append(line); n += 1
        print(f"  {f}: +{n}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as w:
        w.write("\n".join(rows) + "\n")
    print(f"\nmerged {len(rows)} unique traces ({dup} dups dropped) -> {a.out}")


if __name__ == "__main__":
    main()
