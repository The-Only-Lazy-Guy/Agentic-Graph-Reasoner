import os
import shutil

src_dir = r"E:\PROJECT\graph_final"
dest_dir = r"E:\PROJECT\graph_v5"

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

# Directories to copy entirely (ignoring __pycache__ and temp files)
dirs_to_copy = [
    "reasoning",
    "scripts",
    "graphs",
    ".agents"
]

def ignore_files(dir_path, contents):
    ignored = []
    for item in contents:
        # Ignore cache and logs
        if item == "__pycache__":
            ignored.append(item)
        elif item.endswith(".pyc") or item.endswith(".log"):
            ignored.append(item)
        elif item == ".pytest_cache":
            ignored.append(item)
    return ignored

for d in dirs_to_copy:
    src_d = os.path.join(src_dir, d)
    dest_d = os.path.join(dest_dir, d)
    if os.path.exists(src_d):
        print(f"Copying {src_d} to {dest_d}...")
        if os.path.exists(dest_d):
            shutil.rmtree(dest_d)
        shutil.copytree(src_d, dest_d, ignore=ignore_files)

# Files to copy at root level
print("Copying root files...")
for item in os.listdir(src_dir):
    src_item = os.path.join(src_dir, item)
    if os.path.isfile(src_item):
        # Allow .md and .txt
        if item.endswith(".md") or item.endswith(".txt"):
            shutil.copy2(src_item, os.path.join(dest_dir, item))
            continue
        
        # Allow .py files BUT ignore _*.py, patch*.py, tmp*.py, debug*.py
        if item.endswith(".py"):
            if item.startswith("_") or item.startswith("patch") or item.startswith("tmp") or item.startswith("debug"):
                continue
            shutil.copy2(src_item, os.path.join(dest_dir, item))
            continue

print("Done copying.")
