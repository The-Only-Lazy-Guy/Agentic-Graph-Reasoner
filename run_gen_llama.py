"""Generate V4 corpus traces — sharded for multi-machine, two backends.

  --backend llama    : local llama-server / GGUF (free, GPU; lower quality 4B)
  --backend opencode : opencode CLI -> a big cloud model (higher quality;
                       API cost; no GPU needed). Requires `opencode auth` set up.

For a clean corpus use ONE backend throughout (mixing 4B-local + big-cloud halves
confounds held-out calibration).

    # local GGUF:
    python run_gen_llama.py --backend llama --run-id local --shard-index 0 --num-shards 2
    # opencode big model:
    python run_gen_llama.py --backend opencode --run-id vast1 --shard-index 1 --num-shards 2
"""
import argparse, json, sys, time
from pathlib import Path

from answerer_v4 import (V4LlamaServerController, V4ControllerConfig,
                         V4OpencodeController, answer_query_v4)
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
    ap.add_argument("--backend", choices=["llama", "opencode"], default="llama",
                    help="llama=local GGUF server; opencode=big cloud model via opencode CLI")
    ap.add_argument("--opencode-config-dir", default="pure-opencode")
    ap.add_argument("--opencode-model", default=None,
                    help="opencode model alias (e.g. opencode/big-pickle). Default = controller default; "
                         "set this to a BIG model for higher-quality traces.")
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
    print(f"backend={a.backend}  tasks={len(tasks)}  out={out_dir/a.corpus_file}")

    def make_controller():
        if a.backend == "opencode":
            # model=None -> omit --model -> use opencode's own configured default
            # (invocation: `opencode run --format json <msg>`). Pass --opencode-model
            # only if you specifically want to override it.
            return V4OpencodeController(config_dir=a.opencode_config_dir,
                                        model=a.opencode_model, server_url=None)
        return V4LlamaServerController(cfg)
    if a.backend == "opencode":
        print(f"opencode model = {a.opencode_model or '(opencode default, no --model)'}")

    ok = fail = 0
    for i, task in enumerate(tasks):
        controller = make_controller()
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
                                     corpus_file=a.corpus_file, controller_label=f"{a.backend}-{a.run_id}",
                                     extra_metadata={"task_id": tid, "difficulty": diff,
                                                     "expected": task.get("expected", ""), "elapsed_sec": dt,
                                                     "run_id": a.run_id, "shard": f"{a.shard_index}/{a.num_shards}",
                                                     "backend": a.backend})
            ok += 1
        except Exception as e:
            print(f"  ERROR {tid}: {e}")
            fail += 1
    print(f"\n--- done: {ok} ok / {fail} fail -> {out_dir/a.corpus_file}")


if __name__ == "__main__":
    main()
