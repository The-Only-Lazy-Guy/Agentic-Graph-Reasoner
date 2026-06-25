"""Colab training driver for swe_rl --use-exemplar (the retrieve-or-derive policy).

NOTEBOOK-SAFE: no terminal. Every shell action goes through `subprocess` from Python cells.
Paste each `# %% CELL` block into its own Colab cell (or run this file top-to-bottom).

Strategy (matches TRAINING_DESIGN.md): TRAIN on Colab with the cheap PROXY reward (gold-overlap, no
Docker — Colab has no easy Docker-in-Docker), emit held predictions, then VERIFY real resolve on a
Docker box (local/rented) with `v5.graph_grower.swe_verify`.
"""
import subprocess, sys, os, json, shlex


def sh(args, **kw):
    """Run a command (list) and stream output — the only way to 'use the terminal' from a notebook."""
    print("$", " ".join(shlex.quote(a) for a in args), flush=True)
    return subprocess.run(args, check=False, **kw)


# %% CELL 1 — install deps (subprocess pip; no `!pip`)
def install():
    pkgs = ["torch", "transformers", "bitsandbytes", "accelerate", "peft", "datasets",
            "sentence-transformers", "numpy"]
    sh([sys.executable, "-m", "pip", "install", "-q", *pkgs])
    # swebench is only needed on the VERIFY box, not for proxy-reward training. Skip here.


# %% CELL 2 — get the repo + data (EDIT for your setup)
# The training needs, under the repo root:
#   v5/ reasoning/ graph_core.py            (the code)
#   data/swe/grounded_traces.jsonl          (--traces)
#   data/swe/exemplars.jsonl                (exemplar pool; or reuse train-task golds)
#   artifacts/graph_growth/swe_code_candidates.jsonl   (--nodes)
#   data/swe_repos/<repo>/                  (per-task checkouts — LARGE, GBs)
# Easiest: mount Google Drive that holds a prepared copy, then chdir into it.
REPO_ROOT = "/content/graph_v5"   # <- set to your mounted/cloned path

def get_repo():
    if not os.path.isdir(REPO_ROOT):
        # option A: clone (set your remote); option B: copy from a mounted Drive.
        # from google.colab import drive; drive.mount('/content/drive')
        # sh(["cp", "-r", "/content/drive/MyDrive/graph_v5", REPO_ROOT])
        raise SystemExit(f"set REPO_ROOT to your repo path (with v5/, data/swe/, data/swe_repos/). got {REPO_ROOT!r}")
    os.chdir(REPO_ROOT)
    print("cwd =", os.getcwd())


# %% CELL 3 — sanity (no-GPU selftest of the training logic)
def selftest():
    env = {**os.environ, "PYTHONPATH": REPO_ROOT}
    sh([sys.executable, "-m", "v5.runtime.swe_rl", "--selftest"], env=env, cwd=REPO_ROOT)


# %% CELL 4 — TRAIN: swe_rl + the retrieve-or-derive policy, proxy reward, emit held preds
def train(n_tasks=80, sft_steps=200, steps=300, k=6, max_new=512, eff_coef=0.15):
    env = {**os.environ,
           "PYTHONPATH": REPO_ROOT,
           "V5_LM_TRUST_REMOTE_CODE": "1",
           "V5_LM_QUANT": "4bit"}                 # 4-bit base; LoRA trains on top. Bigger Colab GPU -> drop quant.
    sh([sys.executable, "-m", "v5.runtime.swe_rl",
        "--n-tasks", str(n_tasks),
        "--use-exemplar",                          # <- the policy: retrieved plan in the rollout (binding)
        "--eff-coef", str(eff_coef),               # <- efficiency: among WINS, prefer the cheaper/shorter fix
        "--reward-mode", "proxy",                  # gold-overlap; no Docker needed on Colab
        "--sft-steps", str(sft_steps),
        "--steps", str(steps),
        "--k", str(k),
        "--max-new", str(max_new),
        "--lr", "1e-4", "--r-lora", "8",
        "--eval-every", "30",
        "--emit-preds-dir", "artifacts/colab_preds",  # held predictions for the Docker-verify box
        "--emit-preds-every", "0"],
       env=env, cwd=REPO_ROOT)


# %% CELL 5 — (on a DOCKER box, not Colab) verify the emitted predictions for REAL resolve
#   python -m v5.graph_grower.swe_verify --predictions artifacts/colab_preds/held_after_sft.jsonl \
#       --dataset lite --split test --run-id colab_check --verify-backend docker
# (gold-sanity first; that's the apex resolve number the proxy reward is a stand-in for.)


if __name__ == "__main__":
    install()
    get_repo()
    selftest()
    train()
