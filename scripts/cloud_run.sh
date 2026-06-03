#!/usr/bin/env bash
# ============================================================================
# vast.ai cloud run — V5 v2 Track-B experiments (INFERENCE ONLY, no training).
#
# Decides the graph embedder by numbers + validates Qwen3.5 hybrid injection.
# Everything it needs is in the repo (graphs/grown_graph4.json + data/retrieval_gold.jsonl).
#
# On the rented box:
#   git clone <repo> && cd graph_v5
#   export HF_TOKEN=hf_xxx          # for model downloads
#   export PUSH_RESULTS=1           # optional: commit results back (needs git creds)
#   bash scripts/cloud_run.sh
#
# Overridable: GRAPH, QWEN_LM, QWEN_EMB, GOLD
# ============================================================================
set -uo pipefail          # NOT -e: each step runs even if a prior one fails

cd "$(dirname "$0")/.."   # repo root
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONUNBUFFERED=1
export V5_LM_TRUST_REMOTE_CODE=1          # Qwen3.5 ships a custom (hybrid) modeling file

GRAPH="${GRAPH:-graphs/grown_graph4.json}"
GOLD="${GOLD:-data/retrieval_gold.jsonl}"
QWEN_LM="${QWEN_LM:-Qwen/Qwen3.5-4B}"
QWEN_EMB="${QWEN_EMB:-Qwen/Qwen3-Embedding-0.6B}"
RES="artifacts/graph_growth/cloud_results"
mkdir -p "$RES"

echo "================ V5 v2 cloud run ================"
echo "graph=$GRAPH  gold=$GOLD  lm=$QWEN_LM  emb=$QWEN_EMB"
[ -z "${HF_TOKEN:-}" ] && echo "WARN: HF_TOKEN unset -> downloads may rate-limit/fail"
[ -f "$GRAPH" ] || { echo "FATAL: $GRAPH missing (did you git clone the full repo?)"; exit 1; }
[ -f "$GOLD" ]  || echo "WARN: $GOLD missing -> will fall back to scanning data/corpus_shards"

echo "=== GPU ==="
python -c "import torch;print('cuda',torch.cuda.is_available(),'|',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')" || true

echo "=== installing deps ==="
DEPS="transformers accelerate bitsandbytes sentence-transformers datasets numpy torch_geometric"
for i in 1 2 3; do
  pip install -q --no-cache-dir -U $DEPS 2>&1 | tail -2 && break || { echo "pip attempt $i failed (network?); retrying"; sleep 5; }
done
# hard check: a broken install would just fail every step below -> abort loudly instead.
python -c "import transformers, sentence_transformers, torch" 2>/dev/null || {
  echo "FATAL: core deps missing after install. Run: pip install -U $DEPS  then re-run."; exit 1; }

step() {  # label, cmd...
  local label="$1"; shift
  echo; echo "=== [$label] $* ==="
  if "$@"; then echo "[$label] OK"; else echo "[$label] FAILED (exit $?) -- continuing"; fi
}

# ---- 1. embedder A/B (the decisive number; same gold+graph for all three) ----
step mpnet        python -m v5.graph_grower.retrieval_eval --graph "$GRAPH" --gold-file "$GOLD" \
                    --embedder mpnet         --out "$RES/retrieval_mpnet.json"
step qwen-hidden  python -m v5.graph_grower.retrieval_eval --graph "$GRAPH" --gold-file "$GOLD" \
                    --embedder causal-hidden --model "$QWEN_LM"  --out "$RES/retrieval_qwen35hidden.json"
step qwen-embed   python -m v5.graph_grower.retrieval_eval --graph "$GRAPH" --gold-file "$GOLD" \
                    --embedder st-embed      --model "$QWEN_EMB" --out "$RES/retrieval_qwenembed.json"

# ---- 1b. CODE retrieval (SWE symbols) — st-embed segfaults on Windows, runs here ----
# Combine Lite + Verified into the full code pool (799 queries / ~21.6k symbols).
CODE_NODES="${CODE_NODES:-$RES/code_nodes_all.jsonl}"
CODE_GOLD="${CODE_GOLD:-$RES/code_gold_all.jsonl}"
cat artifacts/graph_growth/swe_code_candidates.jsonl \
    artifacts/graph_growth/swe_code_candidates_verified.jsonl 2>/dev/null > "$CODE_NODES" || true
cat data/swe/retrieval_gold_code.jsonl \
    data/swe/retrieval_gold_code_verified.jsonl 2>/dev/null > "$CODE_GOLD" || true
[ -s "$CODE_NODES" ] || CODE_NODES="artifacts/graph_growth/swe_code_candidates.jsonl"
[ -s "$CODE_GOLD" ]  || CODE_GOLD="data/swe/retrieval_gold_code.jsonl"
if [ -f "$CODE_NODES" ] && [ -f "$CODE_GOLD" ]; then
  step code-mpnet python -m v5.graph_grower.retrieval_eval --nodes-file "$CODE_NODES" \
                    --gold-file "$CODE_GOLD" --embedder mpnet    --out "$RES/retrieval_code_mpnet.json"
  step code-qwen  python -m v5.graph_grower.retrieval_eval --nodes-file "$CODE_NODES" \
                    --gold-file "$CODE_GOLD" --embedder st-embed --model "$QWEN_EMB" \
                    --out "$RES/retrieval_code_qwen.json"
fi

# ---- 2. TRAIN the ranker (#2; opt-in). Self-contained: internal held-out split ----
# train_ranker holds out a query fraction -> leakage-free before/after on the SAME
# held-out queries (no Verified needed). Lift = code-ranker vs code-base-held.
HELDOUT="${HELDOUT:-data/swe/retrieval_gold_heldout.jsonl}"
if [ "${TRAIN_RANKER:-0}" = "1" ] && [ -f "$CODE_GOLD" ]; then
  step train-ranker   python -m v5.graph_grower.train_ranker \
                        --gold "$CODE_GOLD" --nodes "$CODE_NODES" --base "$QWEN_EMB" \
                        --out models/ranker-code --epochs 2 --heldout-out "$HELDOUT"
  step code-base-held python -m v5.graph_grower.retrieval_eval --nodes-file "$CODE_NODES" \
                        --gold-file "$HELDOUT" --embedder st-embed --model "$QWEN_EMB" \
                        --out "$RES/retrieval_code_heldout_base.json"
  step code-ranker    python -m v5.graph_grower.retrieval_eval --nodes-file "$CODE_NODES" \
                        --gold-file "$HELDOUT" --embedder st-embed --model models/ranker-code \
                        --out "$RES/retrieval_code_heldout_ranker.json"
fi

# ---- 2. Qwen3.5 hybrid injection validation (frozen, 4-bit) ----
echo; echo "=== [realstack] Qwen3.5 injection-into-hybrid (4-bit) ==="
( export V5_LM_QUANT=4bit
  python -m v5.realstack_test --graph "$GRAPH" --model "$QWEN_LM" 2>&1 | tee "$RES/realstack_qwen35.log" | tail -25 ) \
  || echo "[realstack] FAILED -- check $RES/realstack_qwen35.log"

# ---- 3. leaderboard summary ----
echo; echo "================ RESULTS ================"
python - "$RES" <<'PY'
import json, os, sys
res = sys.argv[1]
order = [("mpnet (baseline)","retrieval_mpnet.json"),
         ("qwen3.5-hidden","retrieval_qwen35hidden.json"),
         ("qwen-embedding","retrieval_qwenembed.json")]
print(f"{'embedder':18} {'Hit@1':>6} {'Hit@5':>6} {'MRR':>6} {'Recall@5':>8}  n")
for label, f in order:
    p = os.path.join(res, f)
    if not os.path.exists(p):
        print(f"{label:18}   (missing — step failed)"); continue
    d = json.load(open(p))
    print(f"{label:18} {d['Hit@k']['1']:>6} {d['Hit@k']['5']:>6} {d['MRR']:>6} {d['Recall@k']['5']:>8}  {d['queries_scored']}")
print("\nDecide: a Qwen embedder beating mpnet's Hit@5/MRR by a real margin -> adopt.")
print("(gold is mildly mpnet-biased, so a tie likely means the Qwen option is actually better.)")
PY

# ---- 4. optional: push results back ----
if [ "${PUSH_RESULTS:-0}" = "1" ]; then
  echo; echo "=== pushing results ==="
  git config user.email "${GIT_EMAIL:-cloud@vast.ai}"
  git config user.name  "${GIT_NAME:-vast-cloud}"
  git add -f "$RES"/*.json "$RES"/*.log 2>/dev/null
  git commit -m "results(cloud): embedder A/B + Qwen3.5 realstack (vast.ai)" 2>&1 | tail -2 && \
    git push origin HEAD 2>&1 | tail -2 || echo "push skipped/failed (set git creds + branch)"
fi

echo; echo "=== DONE. results in $RES ==="
