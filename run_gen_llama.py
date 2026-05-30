"""Generate V4 corpus traces via a local llama-server (no opencode/cloud).

Mirror of run_phase15_corpus.py but uses V4LlamaServerController pointed at a
local llama-server (LOCAL GGUF, no external provider/cost).

    python run_gen_llama.py --dataset artifacts/phase15_test_50.json \
        --out-dir artifacts/phase15_50 --corpus-file corpus50.jsonl \
        --base-url http://127.0.0.1:6768 [--limit N] [--start K]
"""
import argparse, json, sys, time
from pathlib import Path

from answerer_v4 import V4LlamaServerController, V4ControllerConfig, answer_query_v4
from graph_core import MemoryGraph
from reasoning.distillation_corpus import append_session_to_corpus

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/question_bank.json")
    ap.add_argument("--graph", default="graphs/merged_graph.json")
    ap.add_argument("--out-dir", default="data/corpus_shards")
    ap.add_argument("--corpus-file", default=None, help="default: <run-id>.jsonl")
    ap.add_argument("--base-url", default="http://127.0.0.1:6768")
    ap.add_argument("--limit", type=int, default=0)     # 0 = all
    ap.add_argument("--start", type=int, default=0)
    # multi-machine sharding: each machine takes a disjoint slice (idx % num == index)
    ap.add_argument("--run-id", default="local", help="machine label (e.g. local, vast1) -> filename + metadata tag")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--openai-mode", action="store_true",
                    help="generic OpenAI server (e.g. llama-cpp-python python -m llama_cpp.server): "
                         "skips the /health probe + llama.cpp-only body fields. Use on vast.ai.")
    a = ap.parse_args()
    if a.num_shards < 1 or not (0 <= a.shard_index < a.num_shards):
        ap.error(f"need 0 <= shard-index < num-shards (got {a.shard_index}/{a.num_shards})")

    all_tasks = json.load(open(a.dataset, encoding="utf-8"))["tasks"]
    # disjoint shard so machines generate UNIQUE data
    tasks = [t for i, t in enumerate(all_tasks) if i % a.num_shards == a.shard_index]
    if a.start:
        tasks = tasks[a.start:]
    if a.limit:
        tasks = tasks[:a.limit]
    corpus_file = a.corpus_file or f"{a.run_id}.jsonl"
    a.corpus_file = corpus_file
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"run-id={a.run_id}  shard {a.shard_index}/{a.num_shards}  "
          f"-> {len(tasks)}/{len(all_tasks)} questions")
    cfg = V4ControllerConfig(base_url=a.base_url, temperature=0.2, max_tokens=2400,
                             timeout=600.0, llamacpp_mode=not a.openai_mode)
    print(f"backend={a.base_url}  tasks={len(tasks)}  out={out_dir/a.corpus_file}")

    ok = fail = 0
    for i, task in enumerate(tasks):
        controller = V4LlamaServerController(cfg)
        tid = task.get("id", f"task_{i}"); q = task.get("question", "")
        diff = task.get("difficulty", "?"); max_steps = task.get("max_steps", 8)
        print(f"\n[{i+1}/{len(tasks)}] {tid} ({diff}): {q[:70]}")
        g = MemoryGraph.load_json(a.graph)
        t0 = time.time()
        try:
            pkt = answer_query_v4(question=q, graph=g, controller=controller,
                                  max_steps=max_steps, collect_corpus=True)
            dt = time.time() - t0
            print(f"  done {dt:.1f}s finalized={pkt.finalized} steps={pkt.steps}/{max_steps}")
            append_session_to_corpus(pkt=pkt, graph=g, corpus_root=out_dir,
                                     corpus_file=a.corpus_file, controller_label=f"llama-{a.run_id}",
                                     extra_metadata={"task_id": tid, "difficulty": diff,
                                                     "expected": task.get("expected", ""), "elapsed_sec": dt,
                                                     "run_id": a.run_id, "shard": f"{a.shard_index}/{a.num_shards}"})
            ok += 1
        except Exception as e:
            print(f"  ERROR {tid}: {e}")
            fail += 1
    print(f"\n--- done: {ok} ok / {fail} fail -> {out_dir/a.corpus_file}")


if __name__ == "__main__":
    main()
