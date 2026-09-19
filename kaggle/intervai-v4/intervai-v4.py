"""
IntervAI v4 — Full Pipeline Kaggle Kernel (80%+ Accuracy Target)
================================================================
Runs the COMPLETE 8-stage curriculum training pipeline with:
  - Fresh-start detection (stale tokenizer/checkpoint cleanup)
  - Tokenizer trained AFTER data copy (fixes 267-vocab bug)
  - Assistant-only loss masking (labels=-100 for user/system tokens)
  - Rebalanced time budgets (interview gets 25% of total)
  - ML classifier + regressor (Section 3)
  - Comparative analysis + final report (Sections 4+6)

Usage on Kaggle:
  Simply run this kernel with the intervai-data dataset attached.
  Total runtime: ~8 hours on 2x T4.
"""

import os
import sys
import subprocess
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# ── 1. Clone the latest repo (always fresh to get all code fixes) ──
REPO = "/kaggle/working/IntervAI"
if os.path.exists(REPO):
    subprocess.run(["rm", "-rf", REPO], check=False)
print("Cloning repo...")
subprocess.run(
    ["git", "clone", "https://github.com/atrishmanm/IntervAI.git", REPO],
    check=True,
)
print(f"Repo cloned to {REPO}")
os.chdir(REPO)

# ── GPU check ─────────────────────────────────────────────────
import torch
print(f"\n  CUDA available: {torch.cuda.is_available()}")
print(f"  GPU count:      {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("  WARNING: No GPU detected!")

# ── 2. Install dependencies ───────────────────────────────────
print("\nInstalling requirements...")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
    check=False,
)

# ── 3. Run the full pipeline via train_on_kaggle.py ───────────
# This handles EVERYTHING:
#   - Dataset copy from /kaggle/input → data/raw/
#   - Stale tokenizer/checkpoint detection + deletion
#   - Tokenizer training (AFTER data copy → 16K BPE vocab)
#   - ML classifier + regressor training (Section 3)
#   - 8-stage generative curriculum (Section 5)
#   - Comparative analysis + final report (Sections 4+6)
print("\n" + "=" * 60)
print("  FULL PIPELINE — TARGET: 80%+ ACCURACY")
print("  GPUs: 2x Tesla T4 (14.6 GB each)")
print("  Total budget: 480 min (8 hours)")
print("=" * 60)

ret = subprocess.run(
    [
        sys.executable,
        os.path.join("kaggle", "train_on_kaggle.py"),
        "--part", "all",
        "--time-budget", "480",
    ],
    cwd=REPO,
)

if ret.returncode != 0:
    print(f"\n  !! Pipeline exited with code {ret.returncode}")
    print("  Check logs above for errors.")
else:
    print("\n  Pipeline completed successfully!")

# ── 4. Final summary ─────────────────────────────────────────
CKPT_DIR = os.path.join(REPO, "models", "generator", "saved")
print(f"\n{'=' * 60}")
print(f"  FINAL SUMMARY")
print(f"{'=' * 60}")

print(f"\n  Checkpoints in {CKPT_DIR}:")
if os.path.isdir(CKPT_DIR):
    for f in sorted(os.listdir(CKPT_DIR)):
        if f.endswith(".pt"):
            size_mb = os.path.getsize(os.path.join(CKPT_DIR, f)) / (1024 * 1024)
            print(f"    {f}: {size_mb:.1f} MB")
else:
    print("    No checkpoints found!")

print(f"\n  Results in {os.path.join(REPO, 'results')}:")
results_dir = os.path.join(REPO, "results")
if os.path.isdir(results_dir):
    for f in sorted(os.listdir(results_dir)):
        print(f"    {f}")

# ── 5. Print key accuracy metrics ─────────────────────────────
import json
ml_results_path = os.path.join(REPO, "results", "ml_results.json")
if os.path.exists(ml_results_path):
    with open(ml_results_path) as f:
        ml = json.load(f)
    if ml.get("cls_results"):
        best = max(ml["cls_results"], key=lambda x: x["val_acc"])
        print(f"\n  CLASSIFIER: Best val_acc = {best['val_acc']:.4f} (epoch {best['epoch']})")
    if ml.get("reg_results"):
        best = min(ml["reg_results"], key=lambda x: x["val_rmse"])
        print(f"  REGRESSOR:  Best val_rmse = {best['val_rmse']:.4f} (epoch {best['epoch']})")

# Print generative model metrics from logs
for stage in ["pretrain", "domain", "instruction", "interview", "evaluator", "followup"]:
    log_path = os.path.join(REPO, "results", f"{stage}.log")
    if os.path.exists(log_path):
        last_line = None
        with open(log_path) as f:
            for line in f:
                if line.strip():
                    last_line = line
        if last_line:
            try:
                rec = json.loads(last_line)
                print(f"  {stage:>15s}: tok_acc={rec.get('tok_acc', 0):.4f}  "
                      f"top5={rec.get('top5_acc', 0):.4f}  "
                      f"val_loss={rec.get('val_loss', 0):.4f}")
            except json.JSONDecodeError:
                pass

print("\n  Done. Download checkpoints from /kaggle/working/IntervAI/models/generator/saved/")
