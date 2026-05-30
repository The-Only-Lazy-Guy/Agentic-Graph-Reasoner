#!/usr/bin/env bash
# ============================================================================
#  gen_and_push.sh  —  fresh vast.ai box -> UNIQUE corpus shard -> push to GitHub
#
#  Self-contained: assumes ONLY a Linux box with a GPU, python3, and git, and the
#  repo already cloned (with push creds). Does everything else: venv, deps, a
#  CUDA llama-cpp server, GGUF download, sharded generation, and a clean push of
#  just this machine's shard file.
#
#  Clone first (one-liner, on the box):
#    git clone https://<TOKEN>@github.com/The-Only-Lazy-Guy/Agentic-Graph-Reasoner.git
#    cd Agentic-Graph-Reasoner
#
#  Run (each machine MUST use a different SHARD_INDEX; NUM_SHARDS = #machines):
#    SHARD_INDEX=1 NUM_SHARDS=2 bash gen_and_push.sh           # 2nd machine
#    SHARD_INDEX=2 NUM_SHARDS=3 RUN_ID=vastB bash gen_and_push.sh
#
#  Uniqueness: questions are sharded by (idx % NUM_SHARDS == SHARD_INDEX) -> each
#  machine generates a DISJOINT set; RUN_ID (default vast-<host>-<shard>) names a
#  DISTINCT shard file so pushes never collide.
#
#  Env knobs (all optional except SHARD_INDEX/NUM_SHARDS for >1 machine):
#    BACKEND (llama)  llama = local GGUF (this script downloads + serves it);
#                     opencode = big cloud model via opencode CLI (HIGHER QUALITY,
#                     API cost, NO GPU needed -> use a cheap CPU instance). For
#                     opencode you must `npm i -g opencode-ai` + `opencode auth`
#                     beforehand (set a provider key); GGUF/server steps are skipped.
#    SHARD_INDEX (0)  NUM_SHARDS (1)  RUN_ID (vast-<host>-<shard>)
#    GGUF_REPO (unsloth/Qwen3-4B-Instruct-2507-GGUF)  GGUF_FILE (..Q4_K_M.gguf)
#    PORT (6768)  CUDA_TAG (cu124)  N_GPU_LAYERS (999)  N_CTX (8192)
#    OPENCODE_CONFIG_DIR (pure-opencode)
#    HF_TOKEN (for gated models)  GIT_REMOTE (origin)  GIT_BRANCH (main)
#    NO_PUSH=1 (generate only)    LIMIT (cap #questions, for a smoke test)
#    NOTE: for a clean corpus use the SAME BACKEND across all machines.
#    SAFE: never echoes HF_TOKEN or git creds; only `git add -f` the shard file.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# ---- config ----------------------------------------------------------------
BACKEND="${BACKEND:-llama}"
SHARD_INDEX="${SHARD_INDEX:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
HOST_TAG="$(hostname 2>/dev/null | tr -cd 'a-z0-9' | cut -c1-12)"; HOST_TAG="${HOST_TAG:-box}"
RUN_ID="${RUN_ID:-vast-${HOST_TAG}-${SHARD_INDEX}}"
GGUF_REPO="${GGUF_REPO:-unsloth/Qwen3-4B-Instruct-2507-GGUF}"
GGUF_FILE="${GGUF_FILE:-Qwen3-4B-Instruct-2507-Q4_K_M.gguf}"
PORT="${PORT:-6768}"
CUDA_TAG="${CUDA_TAG:-cu124}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"
N_CTX="${N_CTX:-8192}"
GIT_REMOTE="${GIT_REMOTE:-origin}"
GIT_BRANCH="${GIT_BRANCH:-main}"
VENV=".venv-datagen"
SHARD_PATH="data/corpus_shards/${RUN_ID}.jsonl"
export KMP_DUPLICATE_LIB_OK=TRUE PYTHONUNBUFFERED=1
export LOCAL_LLM_BASE_URL="http://127.0.0.1:${PORT}"

log(){ echo "[gen_and_push] $*"; }
die(){ echo "[gen_and_push] ERROR: $*" >&2; exit 1; }

# ---- 0. safety checks ------------------------------------------------------
[ -f graphs/merged_graph.json ] || die "run from the repo root (graphs/merged_graph.json not found)."
[ -f data/question_bank.json ]  || die "data/question_bank.json missing (pull latest main)."
[ "$NUM_SHARDS" -ge 1 ] 2>/dev/null || die "NUM_SHARDS must be >=1"
[ "$SHARD_INDEX" -ge 0 ] && [ "$SHARD_INDEX" -lt "$NUM_SHARDS" ] 2>/dev/null \
    || die "need 0 <= SHARD_INDEX < NUM_SHARDS (got $SHARD_INDEX/$NUM_SHARDS)"
command -v python3 >/dev/null || die "python3 not found"
command -v git >/dev/null || die "git not found"
case "$BACKEND" in llama|opencode) ;; *) die "BACKEND must be llama or opencode (got $BACKEND)";; esac
log "BACKEND=$BACKEND  RUN_ID=$RUN_ID  shard $SHARD_INDEX/$NUM_SHARDS  skip_existing=${SKIP_EXISTING:-1}"

# Pull existing shards FIRST so --skip-existing sees what every machine has
# already generated -> this run only makes questions nobody has done yet.
if [ "${NO_PUSH:-0}" != "1" ]; then
  git pull --rebase --autostash "$GIT_REMOTE" "$GIT_BRANCH" 2>/dev/null || log "WARN: initial pull skipped"
fi
if [ "$BACKEND" = "opencode" ]; then
  # Resolve the opencode executable (nvm installs outside the default PATH).
  if [ -n "${OPENCODE_EXE_PATH:-}" ]; then
    :  # caller provided it
  elif command -v opencode >/dev/null; then
    OPENCODE_EXE_PATH="$(command -v opencode)"
  else
    OPENCODE_EXE_PATH="$(ls -1 /opt/nvm/versions/node/*/bin/opencode "$HOME"/.nvm/versions/node/*/bin/opencode 2>/dev/null | head -1 || true)"
  fi
  [ -n "${OPENCODE_EXE_PATH:-}" ] && [ -x "$OPENCODE_EXE_PATH" ] \
    || die "opencode not found. Install (npm i -g opencode-ai) or set OPENCODE_EXE_PATH=/opt/nvm/versions/node/<ver>/bin/opencode"
  export OPENCODE_EXE_PATH
  log "opencode: $OPENCODE_EXE_PATH"
  "$OPENCODE_EXE_PATH" auth list 2>/dev/null | grep -qiE "key|token|oauth|credential|provider" \
    || log "WARN: 'opencode auth list' shows no credential — run 'opencode auth login' first or generation will fail."
fi

# ---- 1. system deps (best-effort; needs sudo or root) ----------------------
if ! python3 -c 'import venv' 2>/dev/null; then
  log "installing python3-venv (apt)..."
  (sudo apt-get update -y && sudo apt-get install -y python3-venv python3-pip) 2>/dev/null \
    || apt-get install -y python3-venv python3-pip 2>/dev/null || log "WARN: could not apt-get; assuming venv works"
fi

# ---- 2. venv (inherit system CUDA torch if present) ------------------------
log "python venv ($VENV, --system-site-packages to reuse preinstalled torch)..."
[ -d "$VENV" ] || python3 -m venv --system-site-packages "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -q --upgrade pip wheel >/dev/null

# torch: reuse the box's torch if present; else install (CUDA for llama GPU box,
# CPU for opencode since only the MiniLM embedder needs torch there).
if python -c 'import torch' 2>/dev/null; then
  log "using preinstalled torch ($(python -c 'import torch;print(torch.__version__)'))"
elif [ "$BACKEND" = "opencode" ]; then
  log "installing CPU torch (opencode: torch only for the MiniLM embedder)"
  pip install -q torch --index-url https://download.pytorch.org/whl/cpu || die "torch install failed"
else
  log "installing torch ($CUDA_TAG)"
  pip install -q torch --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" \
    || die "torch install failed; set CUDA_TAG to your CUDA (cu121/cu122/cu124)"
fi

log "runtime deps..."
pip install -q -r requirements-datagen.txt || die "pip install -r requirements-datagen.txt failed"

if [ "$BACKEND" = "llama" ]; then
  # ---- 3. llama-cpp-python server (CUDA prebuilt wheel) --------------------
  if ! python -c 'import llama_cpp' 2>/dev/null; then
    log "installing llama-cpp-python[server] CUDA wheel ($CUDA_TAG)..."
    pip install -q "llama-cpp-python[server]" \
        --extra-index-url "https://abetlen.github.io/llama-cpp-python/whl/${CUDA_TAG}" \
      || die "llama-cpp-python install failed; try a different CUDA_TAG"
  fi
  # ---- 4. fetch GGUF (idempotent) -----------------------------------------
  mkdir -p models
  GGUF_PATH="models/${GGUF_FILE}"
  if [ ! -f "$GGUF_PATH" ]; then
    log "downloading GGUF (this can take a few minutes)..."
    HF_TOKEN="${HF_TOKEN:-}" python - "$GGUF_REPO" "$GGUF_FILE" "$GGUF_PATH" <<'PY'
import os, sys, shutil
from huggingface_hub import hf_hub_download
repo, fname, dst = sys.argv[1], sys.argv[2], sys.argv[3]
tok = os.environ.get("HF_TOKEN") or None
p = hf_hub_download(repo_id=repo, filename=fname, token=tok)
shutil.copy(p, dst); print("GGUF ->", dst, round(os.path.getsize(dst)/1e9, 2), "GB")
PY
  else
    log "GGUF already present: $GGUF_PATH"
  fi
  # ---- 5. start OpenAI-compatible server, wait until ready ----------------
  log "starting llama_cpp.server on :$PORT ..."
  python -m llama_cpp.server --model "$GGUF_PATH" --host 127.0.0.1 --port "$PORT" \
    --n_gpu_layers "$N_GPU_LAYERS" --n_ctx "$N_CTX" > llama_server.log 2>&1 &
  SRV=$!
  trap 'kill $SRV 2>/dev/null || true' EXIT
  for _ in $(seq 1 80); do
    curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && { log "server up"; break; }
    kill -0 "$SRV" 2>/dev/null || die "server died on startup — see llama_server.log"
    sleep 3
  done
  curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 || die "server not ready (see llama_server.log)"
fi

# ---- 6. generate this machine's UNIQUE shard ------------------------------
# SKIP_EXISTING (default 1): skip any question already in data/corpus_shards/*.jsonl
# (incl. shards pulled from other machines) -> never regenerate the same question.
SKIP_FLAG=""; [ "${SKIP_EXISTING:-1}" = "1" ] && SKIP_FLAG="--skip-existing"
log "generating shard ($BACKEND) -> $SHARD_PATH"
if [ "$BACKEND" = "opencode" ]; then
  python run_gen_llama.py --backend opencode --opencode-config-dir "${OPENCODE_CONFIG_DIR:-pure-opencode}" \
    ${OPENCODE_MODEL:+--opencode-model "$OPENCODE_MODEL"} $SKIP_FLAG \
    --dataset data/question_bank.json --graph graphs/merged_graph.json \
    --out-dir data/corpus_shards --run-id "$RUN_ID" \
    --shard-index "$SHARD_INDEX" --num-shards "$NUM_SHARDS" ${LIMIT:+--limit "$LIMIT"}
else
  python run_gen_llama.py --backend llama $SKIP_FLAG \
    --dataset data/question_bank.json --graph graphs/merged_graph.json \
    --out-dir data/corpus_shards --run-id "$RUN_ID" \
    --shard-index "$SHARD_INDEX" --num-shards "$NUM_SHARDS" \
    --base-url "http://127.0.0.1:${PORT}" --openai-mode ${LIMIT:+--limit "$LIMIT"}
fi

# ---- 7. safety: must have produced traces ----------------------------------
N=$( [ -f "$SHARD_PATH" ] && wc -l < "$SHARD_PATH" | tr -d ' ' || echo 0 )
[ "$N" -gt 0 ] || die "no traces generated (server/model issue?) — not pushing."
log "generated $N traces."

# ---- 8. push ONLY the shard file -------------------------------------------
if [ "${NO_PUSH:-0}" = "1" ]; then
  log "NO_PUSH=1 -> done. Shard at $SHARD_PATH ($N traces)."; exit 0
fi
log "committing + pushing $SHARD_PATH ..."
git add -f "$SHARD_PATH"
git -c user.name="${GIT_USER_NAME:-vast-datagen}" -c user.email="${GIT_USER_EMAIL:-datagen@vast.ai}" \
    commit -m "data($RUN_ID): corpus shard $SHARD_INDEX/$NUM_SHARDS ($N traces)" \
  || { log "nothing new to commit"; exit 0; }
git pull --rebase --autostash "$GIT_REMOTE" "$GIT_BRANCH" || log "WARN: rebase had issues; attempting push anyway"
git push "$GIT_REMOTE" "$GIT_BRANCH" && log "pushed $SHARD_PATH" \
  || die "push failed — check the box has push credentials (token in clone URL)."

log "DONE. On local:  git pull && python merge_shards.py && python -m v5.training.corpus_scaling --corpus data/corpus_merged.jsonl"
