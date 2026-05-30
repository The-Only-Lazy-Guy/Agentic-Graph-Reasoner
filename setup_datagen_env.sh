#!/usr/bin/env bash
# ============================================================================
#  setup_datagen_env.sh  -  fresh-environment setup for V4 data generation (Linux)
#
#  Linux/cloud companion to setup_datagen_env.bat. Stands up everything needed to
#  run answerer_v4 / run_phase15_corpus.py and produce traces + scoped_patches.
#
#  Usage:
#    bash setup_datagen_env.sh            # full setup, no generation
#    bash setup_datagen_env.sh --run      # setup + kick off a generation run
#    bash setup_datagen_env.sh --gpu      # CUDA torch (default: CPU)
#
#  Backend (pick one, configured after this script):
#    A) opencode CLI : npm install -g opencode-ai ; opencode auth login
#    B) llama-server : serve a GGUF on http://127.0.0.1:6768  (LOCAL_LLM_BASE_URL)
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

VENV=".venv-datagen"
DATASET="artifacts/phase15_test_20.json"
GRAPH="graphs/merged_graph.json"
OUT_DIR="artifacts/phase15"
CORPUS_FILE="phase15_corpus.jsonl"
DO_RUN=0; GPU=0
for a in "$@"; do
  [ "$a" = "--run" ] && DO_RUN=1
  [ "$a" = "--gpu" ] && GPU=1
done

echo "[1/6] Python venv: $VENV"
command -v python3 >/dev/null || { echo "ERROR: python3 not found"; exit 1; }
[ -d "$VENV" ] || python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip setuptools wheel

echo "[2/6] PyTorch (GPU=$GPU)"
if [ "$GPU" = "1" ]; then
  pip install torch --index-url https://download.pytorch.org/whl/cu124
else
  pip install torch --index-url https://download.pytorch.org/whl/cpu
fi

echo "[3/6] Runtime deps (requirements-datagen.txt)"
pip install -r requirements-datagen.txt

echo "[4/6] Warm the embedder cache (all-MiniLM-L6-v2)"
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2'); print('embedder cached')" || echo "WARN: embedder warm failed"

echo "[5/6] LLM backend"
if command -v opencode >/dev/null; then
  echo "  opencode found."
elif command -v npm >/dev/null; then
  echo "  installing opencode-ai via npm..."
  npm install -g opencode-ai || echo "WARN: opencode install failed - use llama-server backend."
else
  echo "  Node/npm not found. Install Node 18+ then 'npm install -g opencode-ai', OR use llama-server."
fi

echo "[6/6] Environment variables (export into your shell):"
cat <<'ENVS'
  export LOCAL_LLM_BASE_URL=http://127.0.0.1:6768
  export LOCAL_LLM_MAX_TOKENS=2400
  export LOCAL_LLM_TEMPERATURE=0.2
  export KMP_DUPLICATE_LIB_OK=TRUE
  export PYTHONUNBUFFERED=1
ENVS
export LOCAL_LLM_BASE_URL=http://127.0.0.1:6768
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONUNBUFFERED=1

cat <<EOF

==========================================================
 SETUP COMPLETE. Next:
==========================================================
 Backend A (opencode):  opencode auth login
 Backend B (llama-server, local GGUF):
    llama-server -m <path>.gguf --port 6768 --host 127.0.0.1 -c 8192

 Generate corpus:
    python run_phase15_corpus.py --dataset $DATASET --graph $GRAPH \\
       --out-dir $OUT_DIR --mode harvest --corpus-file $CORPUS_FILE

 Build V5 corpus + held-out metrics:
    python -m v5.training.corpus_scaling --corpus $OUT_DIR/$CORPUS_FILE
==========================================================
EOF

if [ "$DO_RUN" = "1" ]; then
  echo "--run set: starting generation (ensure a backend is serving first)..."
  python run_phase15_corpus.py --dataset "$DATASET" --graph "$GRAPH" \
     --out-dir "$OUT_DIR" --mode harvest --corpus-file "$CORPUS_FILE"
fi
