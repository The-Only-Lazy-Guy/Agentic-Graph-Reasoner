# MoE Inference Optimization Plan

Date: 2026-05-17

Target runtime: quantized GLM-4.7-Flash GGUF through llama.cpp / llama-cpp-python.

## Verified Model Facts

- GLM-4.7-Flash uses `Glm4MoeLiteForCausalLM`.
- It has `num_hidden_layers=47`, `first_k_dense_replace=1`, `n_routed_experts=64`, `n_shared_experts=1`, and `num_experts_per_tok=4`.
- Hidden size is 2048 and routed expert FFN size is 1536.
- The max trained context is 202,752 tokens, but this repo currently runs 4,096 context. Keep context small unless the task needs more; KV cache memory scales with context.
- Unsloth documents GLM-4.7-Flash as a local 30B MoE with about 3.6B active parameters and recommends llama.cpp/llama-server for GGUF deployment.

## Immediate Recommendation

Do not start with a custom MoE kernel. Start by moving production inference to a recent `llama-server` build and run a sweep:

```powershell
llama-server `
  -m cache/models/GLM-4.7-Flash-Q2_K.gguf `
  --alias glm47-flash `
  -c 4096 `
  -ngl 99 `
  -fa on `
  -b 2048 `
  -ub 512 `
  --cache-prompt `
  --cache-reuse 256 `
  -ctk q8_0 `
  -ctv q8_0 `
  --n-cpu-moe 47 `
  --port 8001
```

Then call it through an OpenAI-compatible client from the controller. A recent upstream llama.cpp has `--cpu-moe` / `--n-cpu-moe` paths, while the installed `llama-cpp-python==0.3.22` wrapper here does not expose that high-level option. This means the best MoE offload path is probably outside the current Python wrapper unless we rebuild or patch it.

Sweep `--n-cpu-moe` instead of assuming 47 is best:

- `47`: keep all routed MoE expert tensors on CPU/RAM; maximize VRAM saved.
- `32`: keep early MoE layers on CPU, later MoE layers GPU-resident.
- `16`: useful if VRAM can hold more experts and CPU bandwidth is the bottleneck.
- `0`: all offload decided by normal layer/GPU strategy; fastest only if it fits.

For this repo's RTX 4050-class target, `--n-cpu-moe 47` or an override-tensor equivalent is likely the space unlock. It may not maximize token/s, but it can allow more attention/shared/dense tensors to stay GPU-resident.

## Repo Changes Added

- `answerer_v2.py` now exposes llama-cpp-python tuning knobs: `n_ctx`, `n_batch`, `n_ubatch`, `n_threads`, `n_threads_batch`, `use_mmap`, `use_mlock`, `offload_kqv`, `op_offload`, and `swa_full`.
- `bench_glm_inference_configs.py` benchmarks repeatable presets and writes JSONL to `artifacts/glm_inference_bench.jsonl`.

Example:

```powershell
python bench_glm_inference_configs.py `
  --model-path cache/models/GLM-4.7-Flash-Q2_K.gguf `
  --presets baseline,flash,flash_op,batch1024,ubatch256,kv_q8,kv_q4 `
  --n-gpu-layers 20 `
  --n-ctx 4096 `
  --runs 2 `
  --warmup 1
```

Run it once with the current Q2_K model and once with `MXFP4_MOE` or `UD-IQ2_M` if downloaded. Do not choose by file size only; select by pass rate plus controller action latency.

## Space Optimization Design: LHER

If current llama.cpp offload is not enough, implement Layerwise Hot Expert Residency (LHER) in a llama.cpp fork.

Core idea:

- Keep router, attention, shared experts, dense layers, embeddings, and output tensors GPU-resident.
- For each MoE layer, keep only a hot cache of H routed experts on GPU. Start with `H=8`, because GLM activates 4 routed experts per token and recent routing-locality research suggests cache sizes around 2x active experts are often a good first point.
- Keep cold routed experts in pinned CPU RAM, preferably mmap-backed and preloaded.
- During decode, run the gate first, obtain top-4 expert IDs, and check the per-layer GPU expert cache.
- For misses, asynchronously copy only missing expert tensors into staging slots and evict by segmented LRU / LFU.
- Prefetch the previous token's selected expert IDs for the same layer and optionally the top experts from a short rolling segment profile.
- Measure cache hit rate per layer, not just global hit rate. If some layers have poor locality, pin those layers to CPU or GPU statically and use LHER only where it works.

Why this is plausible:

- MoE decode is sparse: GLM uses 4 routed experts out of 64 per token.
- Personal-machine decode is often batch size 1, where expert reuse can be high.
- Published MoE-Infinity results show expert caching/prefetch can improve per-token latency when sparse activations repeat.
- Recent local-routing-consistency work warns that not all MoE models cache well, so we must trace GLM routes before investing deeply.

Minimum fork instrumentation:

- Emit `(token_index, layer_index, selected_expert_ids)` during decode.
- Compute per-layer hit rate for cache sizes `H={4,8,12,16}`.
- Compute segment cache hit rate on windows of `S={16,32,64}` tokens.
- Only implement GPU expert cache if trace replay predicts at least 70 percent hit rate at H<=12 for the controller workload.

Failure modes:

- If GLM-4.7-Flash routing has low local consistency, dynamic expert cache will thrash and lose to static CPU MoE offload.
- If PCIe copy latency dominates, prefetch must be accurate or the design will be slower than CPU expert execution.
- If batch size grows, per-token expert diversity rises; dynamic caching helps less than continuous batching and static placement.

## Speed/Throughput Path If Space Does Not Move

1. Use server mode for production. `llama-server` gives continuous batching, prompt cache, cache reuse, metrics, and OpenAI-compatible calls.
2. Cache the controller prefix. The system prompt, tool descriptions, and JSON schema are repeated each step. Prefix/prompt caching directly attacks time-to-first-token and repeated prompt eval.
3. Reduce controller output tokens. Keep `thought_summary` capped, use short action IDs, and lower `max_tokens` from 512 once the action JSON is reliable.
4. Sweep `-b`/`-ub`. Prompt eval benefits from larger batch; decode can regress if ubatch is too large for VRAM.
5. Sweep KV cache types. Try `q8_0` first; try `q4_0` only if memory pressure is still high and quality is stable.
6. Prefer current llama.cpp binaries for MoE deployment. The Python wrapper here is usable, but it lags upstream MoE placement features.
7. Evaluate KTransformers only if llama.cpp cannot hit target token/s. It is specifically designed for CPU/GPU hybrid MoE, but integration cost is higher than swapping server runtime.

## Source Pointers

- Z.ai/Hugging Face config: https://huggingface.co/zai-org/GLM-4.7-Flash/blob/main/config.json
- Unsloth GLM-4.7-Flash local guide: https://unsloth.ai/docs/models/glm-4.7-flash
- Unsloth GLM-4.7 full guide with MoE offload and KV cache guidance: https://unsloth.ai/docs/models/tutorials/glm-4.7
- llama.cpp server docs: https://www.mintlify.com/ggml-org/llama.cpp/inference/server
- llama.cpp argument source: https://github.com/ggml-org/llama.cpp/blob/master/common/arg.cpp
- MoE-Infinity: https://arxiv.org/abs/2401.14361
- Local routing consistency: https://arxiv.org/abs/2505.16056
- Prompt Cache: https://arxiv.org/abs/2311.04934
- KTransformers paper: https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf
