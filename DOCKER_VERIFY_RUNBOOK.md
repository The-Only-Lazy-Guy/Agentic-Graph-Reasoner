# Docker-box verification runbook — the real resolve-rate

**Goal:** turn the loop's applyable patches into the headline **test-pass** number using the
official `swebench` harness — the ONE reliable verifier. (sb-cli's hosted SWE-bench-M lite eval
marks correct GOLD patches as "failed" → untrustworthy; see READ_THIS 2026-06-07. Use Docker.)

**Two machines** (verification is CPU-only — no GPU):
- **GPU box** (Blackwell/etc): runs the 4B → generates the loop's patches.
- **Docker box** (cheap CPU, NO GPU): runs the tests in per-instance containers.

---

## 0. Docker box requirements
- A **real Docker daemon** (`docker info` succeeds). Rented *GPU* containers usually can't run
  Docker (no privileged/DinD) → use **bare-metal / a VPS** (Hetzner, DO, …) or a provider that
  grants privileged Docker. **CPU-only is fine; NO GPU needed.**
- **Disk:** SWE-bench images are several GB each → budget **50–100+ GB**.
- Python 3.11 + git.

## 1. GPU box — generate the loop's predictions
```bash
git pull
V5_LM_TRUST_REMOTE_CODE=1 python -m v5.runtime.verifier_retry \
  --adapter-ckpt artifacts/stage_cache/adapter_code_s3.pt \
  --n-eval 50 --max-retries 4 \
  --emit-predictions artifacts/loop_predictions.jsonl
git add -f artifacts/loop_predictions.jsonl && git commit -m "chore: loop preds" && git push
```
(`--n-eval 50 --max-retries 4` = a meaningful sample. Only **applyable** patches are emitted.)

## 2. Docker box — setup + PROVE the harness (gold-sanity)
```bash
git clone <repo> && cd Agentic-Graph-Reasoner
pip install swebench datasets
docker info        # MUST succeed
# gold patches MUST (nearly all) resolve -> proves the harness + env
python -m v5.graph_grower.swe_verify --backend docker --gold-sanity --dataset lite --limit 5 --max-workers 4
```
→ **gold ~5/5 resolved** = harness OK, trust the numbers. If gold does NOT resolve → the
Docker env is broken; fix it before trusting any model number. (Unlike sb-cli, the official
harness *will* pass gold — that's the point.)

## 3. Docker box — the real resolve-rate
```bash
git pull   # get loop_predictions.jsonl from step 1
python -m v5.graph_grower.swe_verify --backend docker \
  --predictions artifacts/loop_predictions.jsonl --dataset lite --run-id v5_loop --max-workers 4
```
→ prints **`resolved N / total`**. THAT is the headline:
> the graph-grounded 4B loop resolves **N%** of SWE-bench Lite.

## Reading the number
- **resolved** = patches that made FAIL_TO_PASS go green (+ PASS_TO_PASS stay green) = **real fixes**.
- Denominator = the loop's **applyable** patches (~53% of attempted); resolved ≤ applyable.
- Expect **low absolute** (weak 4B; many applyable patches are no-op/wrong) — but **any
  resolved > 0 = a real bug fixed end-to-end** by graph→read→edit→retry. That's the milestone.

## The grounding LIFT (the real claim) — ablation
Absolute resolve-rate is one thing; the **lift from grounding** is the thesis. Run the loop
**with** vs **without** the graph (`--no-graph` = cold: no source-read, no injection) and verify both:
```bash
# GPU box — grounded (default) AND cold baseline:
V5_LM_TRUST_REMOTE_CODE=1 python -m v5.runtime.verifier_retry --adapter-ckpt artifacts/stage_cache/adapter_code_s3.pt \
  --n-eval 50 --max-retries 4 --emit-predictions artifacts/grounded_predictions.jsonl
V5_LM_TRUST_REMOTE_CODE=1 python -m v5.runtime.verifier_retry --adapter-ckpt artifacts/stage_cache/adapter_code_s3.pt \
  --n-eval 50 --max-retries 4 --no-graph --emit-predictions artifacts/cold_predictions.jsonl
git add -f artifacts/*_predictions.jsonl && git commit -m "preds" && git push
# Docker box — verify both:
python -m v5.graph_grower.swe_verify --backend docker --predictions artifacts/grounded_predictions.jsonl --dataset lite --run-id v5_grounded
python -m v5.graph_grower.swe_verify --backend docker --predictions artifacts/cold_predictions.jsonl --dataset lite --run-id v5_cold
```
`resolved_grounded` − `resolved_cold` = **"grounding resolves X% more SWE tasks"** — the headline
claim, not just the absolute.

## First-run notes
- swebench builds a **Docker image per repo** (django/astropy/…) → slow + GBs the first time;
  cached after. `--limit 5` gold-sanity keeps it small.
- `--max-workers N` parallelizes — set to the box's CPU cores.
- Time: first gold-sanity ~10–30 min (image builds); subsequent runs much faster (cached).
