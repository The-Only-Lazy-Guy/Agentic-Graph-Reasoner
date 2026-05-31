"""Project a corpus JSONL into V5-native supervision targets.

Usage:
    python project_corpus_to_v5_targets.py --corpus data/corpus_merged.jsonl
    python project_corpus_to_v5_targets.py --corpus data/corpus_merged.jsonl --out data/corpus_merged_v5proj.jsonl
    python project_corpus_to_v5_targets.py --corpus data/corpus_merged.jsonl --in-place
"""
from __future__ import annotations

import argparse
from pathlib import Path

from v5.training.projection import project_corpus_file


def _default_out(path: Path) -> Path:
    return path.with_name(f"{path.stem}_v5proj{path.suffix}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Attach V5-native projection targets to a corpus JSONL.")
    ap.add_argument("--corpus", required=True, help="Input corpus JSONL.")
    ap.add_argument("--out", default=None, help="Output JSONL. Default: <stem>_v5proj.jsonl")
    ap.add_argument("--in-place", action="store_true", help="Rewrite the input file in place.")
    args = ap.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        raise SystemExit(f"Corpus not found: {corpus_path}")
    if args.in_place and args.out:
        raise SystemExit("Use either --out or --in-place, not both.")

    out_path = corpus_path if args.in_place else Path(args.out) if args.out else _default_out(corpus_path)
    stats = project_corpus_file(corpus_path, out_path)

    print("V5 corpus projection complete:")
    print(f"  input               : {stats['corpus_path']}")
    print(f"  output              : {stats['out_path']}")
    print(f"  rows                : {stats['rows']}")
    print(f"  planning rows       : {stats['planning_rows']}")
    print(f"  evidence rows       : {stats['evidence_rows']}")
    print(f"  support rows        : {stats['support_rows']}")
    print(f"  loop-supervised rows: {stats['loop_rows']}")
    print(f"  mean candidate nodes: {stats['mean_candidate_nodes']:.1f}")


if __name__ == "__main__":
    main()
