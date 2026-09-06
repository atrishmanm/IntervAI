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

# Step 1: Clone repo if not present
if not REPO_DIR.exists():
    print("[setup] Cloning IntervAI repo...")
    subprocess.run(["git", "clone", REPO_URL, str(REPO_DIR)], check=True)

# Step 2: Run the notebook
args = " ".join(sys.argv[1:])
cmd = f"python {SCRIPT} {args}"
print(f"[run] {cmd}")
os.system(cmd)
