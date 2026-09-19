"""
IntervAI v3 — Staged Kaggle Training Kernel
Runs PRETRAIN first, verifies checkpoint, then proceeds to next stages.
Uses 2x T4 GPUs with the val_loss fix.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# ── 1. Clone the latest repo ──────────────────────────────────
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
print(f"  CUDA available: {torch.cuda.is_available()}")
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

# ── 3. Verify the checkpoint fix is present ───────────────────
train_py = os.path.join(REPO, "models", "generator", "train.py")
with open(train_py) as f:
    content = f.read()

# Check that the first-epoch save is present (the fix)
if "best_val_loss == float(\"inf\")" not in content:
    print("FATAL: Checkpoint fix not found in train.py!")
    sys.exit(1)
print("Checkpoint fix verified: first-epoch save is present")

# ── 3b. Train tokenizer if not present ────────────────────────
tok_path = os.path.join(REPO, "tokenizer", "saved", "tokenizer.json")
if not os.path.exists(tok_path):
    print("\nTraining tokenizer (16K vocab, ByteLevel BPE)...")
    ret = subprocess.run(
        [sys.executable, os.path.join("tokenizer", "train_tokenizer.py")],
        cwd=REPO,
    )
    if ret.returncode != 0:
        print("FATAL: Tokenizer training failed!")
        sys.exit(1)
    if not os.path.exists(tok_path):
        print(f"FATAL: Tokenizer not found at {tok_path} after training!")
        sys.exit(1)
    print(f"Tokenizer trained and saved: {tok_path}")
else:
    print(f"Tokenizer already present: {tok_path}")

# ── 4. Run PRETRAIN ONLY, then stop for verification ───────────
STAGES = [
    ("pretrain", 168),      # ~168 min (35% of 480)
]

STAGE_CKPT = {
    "pretrain": "pretrained.pt",
    "domain": "domain_tuned.pt",
    "instruction": "instruction_tuned.pt",
    "interview": "interview_tuned.pt",
    "evaluator": "evaluator.pt",
    "followup": "final_model.pt",
    "resume_finetune": "resume_finetuned.pt",
    "negotiation": "negotiation_tuned.pt",
}

CKPT_DIR = os.path.join(REPO, "models", "generator", "saved")
os.makedirs(CKPT_DIR, exist_ok=True)

print("\n" + "=" * 60)
print("  STAGED TRAINING PIPELINE")
print("  GPUs: 2x Tesla T4 (14.6 GB each)")
print("  Total budget: 480 min")
print("=" * 60)

import time
START = time.time()

for stage_name, budget in STAGES:
    elapsed_min = (time.time() - START) / 60
    remaining = max(0, 480 - elapsed_min)
    print(f"\n{'='*60}")
    print(f"  STAGE: {stage_name.upper()}")
    print(f"  Budget: {budget} min | Elapsed: {elapsed_min:.1f} min | Remaining: {remaining:.1f} min")
    print(f"{'='*60}")

    if remaining < 10:
        print("  TIME BUDGET EXCEEDED — stopping")
        break

    t0 = time.time()
    ret = subprocess.run(
        [
            sys.executable,
            os.path.join("models", "generator", "train.py"),
            f"--stage", stage_name,
            f"--time-budget", str(budget),
        ],
        cwd=REPO,
    )
    stage_elapsed = (time.time() - t0) / 60

    # ── Verify checkpoint was saved ──
    ckpt_name = STAGE_CKPT[stage_name]
    ckpt_path = os.path.join(CKPT_DIR, ckpt_name)

    if ret.returncode != 0:
        print(f"\n  !! Stage '{stage_name}' FAILED (exit {ret.returncode}) after {stage_elapsed:.1f} min")
        print(f"  Stopping pipeline. Fix the error and re-run.")
        break

    if not os.path.exists(ckpt_path):
        print(f"\n  !! CHECKPOINT NOT SAVED: {ckpt_name}")
        print(f"  This means downstream stages will train from scratch.")
        print(f"  Stopping pipeline. Fix the checkpoint bug and re-run.")
        break

    ckpt_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
    print(f"\n  CHECKPOINT VERIFIED: {ckpt_name} ({ckpt_size_mb:.1f} MB)")
    print(f"  Stage '{stage_name}' completed in {stage_elapsed:.1f} min")

    # ── Check training progress file ──
    progress_file = os.path.join(CKPT_DIR, "training_progress.json")
    if os.path.exists(progress_file):
        with open(progress_file) as f:
            progress = json.load(f)
        if stage_name in progress:
            print(f"  Status: {progress[stage_name].get('status', 'unknown')}")

# ── 5. Final summary ──
total_min = (time.time() - START) / 60
print(f"\n{'='*60}")
print(f"  PIPELINE SUMMARY")
print(f"{'='*60}")
print(f"  Total time: {total_min:.1f} min ({total_min/60:.1f} hours)")

print(f"\n  Checkpoints in {CKPT_DIR}:")
if os.path.isdir(CKPT_DIR):
    for f in sorted(os.listdir(CKPT_DIR)):
        if f.endswith(".pt"):
            size_mb = os.path.getsize(os.path.join(CKPT_DIR, f)) / (1024 * 1024)
            print(f"    {f}: {size_mb:.1f} MB")

print(f"\n  Results in {os.path.join(REPO, 'results')}:")
results_dir = os.path.join(REPO, "results")
if os.path.isdir(results_dir):
    for f in sorted(os.listdir(results_dir)):
        print(f"    {f}")

print("\n  Done.")
