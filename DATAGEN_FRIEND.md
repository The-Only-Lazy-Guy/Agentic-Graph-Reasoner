# Generate a unique corpus shard (for a helper machine)

Run these on a fresh Ubuntu box (a **cheap CPU instance** is fine — opencode uses
a cloud model, no GPU needed). You generate a DISJOINT half of the questions and
push just your shard file; nothing else is touched.

Replace the **2 placeholders**: `<GH_USER>` and `<GH_TOKEN>` (a GitHub token with
repo push access). The model is whatever your opencode is already configured to
use — **no `--model` needed**.

```bash
# 1. system packages
sudo apt-get update -y
sudo apt-get install -y git python3 python3-venv python3-pip curl

# 2. opencode CLI (skip if already installed via nvm). Then log in to your provider:
#    (opencode is invoked as:  opencode run --format json <msg>  — no --model)
opencode auth login
opencode auth list                    # confirm a credential shows

# 3. clone the repo WITH push access + set identity
git clone https://<GH_USER>:<GH_TOKEN>@github.com/The-Only-Lazy-Guy/Agentic-Graph-Reasoner.git
cd Agentic-Graph-Reasoner
git config user.name  "<GH_USER>"
git config user.email "<GH_USER>@users.noreply.github.com"

# 4. point at the opencode binary (nvm path example) if it isn't on PATH:
export OPENCODE_EXE_PATH=/opt/nvm/versions/node/v24.14.1/bin/opencode

# 5. SMOKE TEST — 2 questions, no push (confirms opencode + auth work)
BACKEND=opencode NUM_SHARDS=2 SHARD_INDEX=1 LIMIT=2 NO_PUSH=1 bash gen_and_push.sh

# 6. FULL run — your unique half (shard 1 of 2) + auto-push your shard
BACKEND=opencode NUM_SHARDS=2 SHARD_INDEX=1 bash gen_and_push.sh
```

That pushes `data/corpus_shards/vast-<host>-1.jsonl` to `main`. Done.

## Notes
- **No `--model`**: opencode uses its own configured default model; the script does
  not pass `--model`. (If you ever want to force one, add `OPENCODE_MODEL=<alias>`.)
- **Uniqueness is automatic**: questions split by `idx % NUM_SHARDS == SHARD_INDEX`.
  You are `SHARD_INDEX=1`; the owner is `SHARD_INDEX=0`. Different shard = different
  questions = unique data. Your shard filename is distinct too, so pushes never collide.
- **More than 2 machines?** Set `NUM_SHARDS` to the total, give each a different
  `SHARD_INDEX` (0,1,2,...).
- **Safe by design**: only `git add -f` your one shard file (never venv/keys/logs),
  and it skips the push if 0 traces were produced.
- **No GPU needed** for opencode — use a cheap CPU instance.

## Owner, after shards are pushed
```bash
git pull
python merge_shards.py
python -m v5.training.corpus_scaling --corpus data/corpus_merged.jsonl
```
