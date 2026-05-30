# Generate a unique corpus shard (for a helper machine)

Run these on a fresh Ubuntu box (a **cheap CPU instance** is fine — opencode uses
a cloud model, no GPU needed). You generate a DISJOINT half of the questions and
push just your shard file; nothing else is touched.

Replace the **3 placeholders**: `<GH_USER>`, `<GH_TOKEN>` (GitHub token with repo
push), and `<BIG_MODEL>` (the opencode model alias to use, e.g. `opencode/big-pickle`).

```bash
# 1. system packages
sudo apt-get update -y
sudo apt-get install -y git python3 python3-venv python3-pip curl

# 2. Node + opencode CLI
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs
npm install -g opencode-ai

# 3. log in to your model provider (this is where the API cost lives)
opencode auth login                 # follow prompts; pick provider + paste key
opencode auth list                  # confirm a credential shows up

# 4. clone the repo WITH push access
git clone https://<GH_USER>:<GH_TOKEN>@github.com/The-Only-Lazy-Guy/Agentic-Graph-Reasoner.git
cd Agentic-Graph-Reasoner
git config user.name  "<GH_USER>"
git config user.email "<GH_USER>@users.noreply.github.com"

# 5. SMOKE TEST first — 2 questions, no push (confirms model + auth work)
BACKEND=opencode OPENCODE_MODEL="<BIG_MODEL>" \
  NUM_SHARDS=2 SHARD_INDEX=1 LIMIT=2 NO_PUSH=1 bash gen_and_push.sh

# 6. FULL run — your unique half (shard 1 of 2) + auto-push your shard
BACKEND=opencode OPENCODE_MODEL="<BIG_MODEL>" \
  NUM_SHARDS=2 SHARD_INDEX=1 bash gen_and_push.sh
```

That pushes `data/corpus_shards/vast-<host>-1.jsonl` to `main`. Done.

## Notes
- **Uniqueness is automatic**: questions are split by `idx % NUM_SHARDS == SHARD_INDEX`.
  You are `SHARD_INDEX=1`; the owner is `SHARD_INDEX=0`. Different shard = different
  questions = unique data. Your shard filename (`vast-<host>-1.jsonl`) is also distinct,
  so pushes never collide.
- **More than 2 machines?** Set `NUM_SHARDS` to the total and give each box a different
  `SHARD_INDEX` (0,1,2,...).
- **Same model everywhere**: for a clean corpus, every machine should use the SAME
  `OPENCODE_MODEL`. Tell the owner which `<BIG_MODEL>` you used.
- **Safe by design**: the script only `git add -f` your one shard file, never your
  venv/keys/logs; it skips the push if 0 traces were produced.
- **If push fails**: the clone URL needs a token with push rights (step 4).

## Owner, after shards are pushed
```bash
git pull
python merge_shards.py
python -m v5.training.corpus_scaling --corpus data/corpus_merged.jsonl
```
