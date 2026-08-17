"""
kaggle/train_on_kaggle.py
==========================
Run INTERVUE training on Kaggle (T4/P100 16GB).

Copy this file into a Kaggle Notebook cell, or run:
    !python /kaggle/working/IntervAI/kaggle/train_on_kaggle.py --stage all

REQUIRED SETUP ON KAGGLE (before running):
  1. New Notebook → Settings → Accelerator = GPU T4 x2 (or P100)
  2. Add Data → your `intervai-data` Kaggle dataset (upload data/raw/*.jsonl)
     It mounts at /kaggle/input/intervai-data/<files>
  3. (Optional) Upload the pretrained tokenizer if you trained it locally:
     tokenizer/saved/tokenizer.json

The unified trainer (models/generator/train.py) auto-detects Kaggle and uses
batch=16, accum=4, medium model, FP16. Checkpoints land in
/kaggle/working/IntervAI/models/generator/saved/ — download them at the end.
"""

import os
import sys
import time
from pathlib import Path

# ── 0. Clone the repo if not already present ──────────────────
WORK = Path("/kaggle/working")
REPO = WORK / "IntervAI"
if not REPO.exists():
    print("Cloning IntervAI repo...")
    os.system("git clone https://github.com/<YOUR_USER>/IntervAI.git /kaggle/working/IntervAI")

sys.path.insert(0, str(REPO))
os.chdir(REPO)

# ── 1. Environment summary ────────────────────────────────────
from env_config import ENV, ROOT, SAVE_ROOT, MODEL_SIZE, print_env_summary
print_env_summary()
assert ENV == "kaggle", f"Expected Kaggle env, got {ENV}"

# ── 2. Install deps (idempotent) ──────────────────────────────
def _pip(*pkgs):
    os.system(f"{sys.executable} -m pip install -q {' '.join(pkgs)}")

_pip("torch", "tokenizers", "numpy", "scikit-learn", "tqdm", "safetensors")

# ── 3. Train tokenizer on Kaggle input (if not already saved) ─
tok_path = REPO / "tokenizer" / "saved" / "tokenizer.json"
if not tok_path.exists():
    print("\nTraining tokenizer on Kaggle data...")
    os.system(f"{sys.executable} tokenizer/train_tokenizer.py")

# ── 4. Run training ───────────────────────────────────────────
STAGES = ["pretrain", "domain", "instruction", "interview", "evaluator", "followup"]
stage = sys.argv[sys.argv.index("--stage") + 1] if "--stage" in sys.argv else "all"
if stage != "all":
    STAGES = [stage]

total_t0 = time.time()
for s in STAGES:
    print(f"\n{'='*60}\n  RUNNING STAGE: {s}\n{'='*60}")
    t0 = time.time()
    rc = os.system(f"{sys.executable} models/generator/train.py --stage {s}")
    if rc != 0:
        print(f"!! Stage '{s}' FAILED (exit {rc}) — stopping.")
        sys.exit(rc)
    print(f"  Stage {s} took {(time.time()-t0)/60:.1f} min")

print(f"\nALL STAGES COMPLETE in {(time.time()-total_t0)/60:.1f} min")
print(f"Checkpoints in: {SAVE_ROOT}")
for p in sorted(SAVE_ROOT.glob("*.pt")):
    print(f"  - {p.name} ({p.stat().st_size/1e6:.1f} MB)")

print("\nDownload /kaggle/working/IntervAI/models/generator/saved/*.pt to your laptop!")
