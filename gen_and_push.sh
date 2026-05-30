#!/usr/bin/env bash
# ============================================================================
#  gen_and_push.sh  -  generate a UNIQUE corpus shard on a remote box (vast.ai)
#                       and push it to GitHub for the local trainer to merge.
#
#  Each machine runs a DISJOINT shard of data/question_bank.json (idx % N == K),
#  writes data/corpus_shards/<run-id>.jsonl, and pushes ONLY that file (distinct
#  filename per machine -> no merge conflicts). Local pulls + merges.
#
#  Usage (on vast.ai, repo cloned with push access):
#    RUN_ID=vast1 SHARD_INDEX=1 NUM_SHARDS=2 bash gen_and_push.sh
#
#  Env knobs:
#    RUN_ID        machine label (default vast1) -> shard filename + metadata
#    SHARD_INDEX   this machine's shard (default 1)         local usually uses 0
#    NUM_SHARDS    total machines (default 2)
#    GGUF_REPO     HF repo for the GGUF  (default unsloth/Qwen3-4B-Instruct-2507-GGUF)
#    GGUF_FILE     GGUF filename         (default Qwen3-4B-Instruct-2507-Q4_K_M.gguf)
#    PORT          llama-server port     (default 6768)
#    GIT_REMOTE    push remote           (default origin)
#    GIT_BRANCH    push branch           (default main)
#    NO_PUSH=1     generate only, skip the git push
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

RUN_ID="${RUN_ID:-vast1}"
SHARD_INDEX="${SHARD_INDEX:-1}"
NUM_SHARDS="${NUM_SHARDS:-2}"
GGUF_REPO="${GGUF_REPO:-unsloth/Qwen3-4B-Instruct-2507-GGUF}"
GGUF_FILE="${GGUF_FILE:-Qwen3-4B-Instruct-2507-Q4_K_M.gguf}"
PORT="${PORT:-6768}"
GIT_REMOTE="${GIT_REMOTE:-origin}"
GIT_BRANCH="${GIT_BRANCH:-main}"
SHARD_PATH="data/corpus_shards/${RUN_ID}.jsonl"

echo "=== [0] env: RUN_ID=$RUN_ID shard=$SHARD_INDEX/$NUM_SHARDS gguf=$GGUF_REPO/$GGUF_FILE ==="
export KMP_DUPLICATE_LIB_OK=TRUE PYTHONUNBUFFERED=1
export LOCAL_LLM_BASE_URL="http://127.0.0.1:${PORT}"

# 1. python env + runtime deps (reuse the shared setup)
echo "=== [1] python env + deps ==="
bash setup_datagen_env.sh ${GPU:+--gpu} || true
# shellcheck disable=SC1091
source .venv-datagen/bin/activate 2>/dev/null || true
pip install -q huggingface_hub llama-cpp-python 2>/dev/null || true

# 2. fetch the GGUF (skip if present)
echo "=== [2] fetch GGUF ==="
mkdir -p models
GGUF_PATH="models/${GGUF_FILE}"
if [ ! -f "$GGUF_PATH" ]; then
  python - "$GGUF_REPO" "$GGUF_FILE" "$GGUF_PATH" <<'PY'
import sys; from huggingface_hub import hf_hub_download; import shutil
repo, fname, dst = sys.argv[1], sys.argv[2], sys.argv[3]
p = hf_hub_download(repo_id=repo, filename=fname)
shutil.copy(p, dst); print("GGUF ->", dst)
PY
fi

# 3. start llama-server (llama-cpp-python server is OpenAI-compatible)
echo "=== [3] start llama-server on :$PORT ==="
python -m llama_cpp.server --model "$GGUF_PATH" --host 127.0.0.1 --port "$PORT" \
  --n_gpu_layers ${N_GPU_LAYERS:-999} --n_ctx 8192 > llama_server.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT
echo "  waiting for server..."
for _ in $(seq 1 60); do
  curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && break || sleep 3
done

# 4. generate this machine's unique shard
echo "=== [4] generate shard $SHARD_INDEX/$NUM_SHARDS -> $SHARD_PATH ==="
python run_gen_llama.py --dataset data/question_bank.json --graph graphs/merged_graph.json \
  --out-dir data/corpus_shards --run-id "$RUN_ID" \
  --shard-index "$SHARD_INDEX" --num-shards "$NUM_SHARDS" \
  --base-url "http://127.0.0.1:${PORT}"

# 5. push the shard
if [ "${NO_PUSH:-0}" = "1" ]; then
  echo "NO_PUSH=1 -> skipping git push. Shard at $SHARD_PATH"
  exit 0
fi
echo "=== [5] commit + push $SHARD_PATH ==="
git add -f "$SHARD_PATH"
git -c user.name="${GIT_USER_NAME:-vast-datagen}" -c user.email="${GIT_USER_EMAIL:-datagen@vast.ai}" \
    commit -m "data($RUN_ID): corpus shard $SHARD_INDEX/$NUM_SHARDS ($(wc -l < "$SHARD_PATH") traces)" || {
      echo "nothing to commit"; exit 0; }
git pull --rebase "$GIT_REMOTE" "$GIT_BRANCH" || true   # absorb other machines' shard pushes
git push "$GIT_REMOTE" "$GIT_BRANCH"
echo "pushed $SHARD_PATH"
