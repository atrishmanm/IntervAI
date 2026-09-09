"""
IntervAI Kaggle Entry Point
============================
Upload this file as a Kaggle Code kernel, or paste into a Kaggle notebook cell.

Usage in Kaggle notebook cell:
    !python /kaggle/working/run_kaggle.py --smoke-test
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/atrishmanm/IntervAI.git"
REPO_DIR = Path("/kaggle/working/IntervAI")
SCRIPT = REPO_DIR / "kaggle" / "intervai_research_notebook.py"

# Step 1: Clone repo if not present, otherwise refresh it — /kaggle/working can
# persist across runs of the same kernel, so an existing clone may predate
# recent fixes. reset --hard keeps untracked outputs (models/, data/, results/).
if not REPO_DIR.exists():
    print("[setup] Cloning IntervAI repo...")
    subprocess.run(["git", "clone", REPO_URL, str(REPO_DIR)], check=True)
else:
    print("[setup] Repo exists — refreshing to origin/main...")
    subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "origin"], check=False)
    r = subprocess.run(["git", "-C", str(REPO_DIR), "reset", "--hard", "origin/main"], check=False)
    if r.returncode != 0:
        print("[setup] WARN: refresh failed (offline or not a git clone) — using existing code.")

# Step 2: Run the notebook
args = " ".join(sys.argv[1:])
cmd = f"python {SCRIPT} {args}"
print(f"[run] {cmd}")
os.system(cmd)
