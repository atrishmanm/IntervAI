"""
kaggle/train_on_kaggle.py
==========================
Run INTERVUE research-grade training on Kaggle (1x or 2x T4/P100) — fully
automatic, resume-safe.

What it does:
  1. Clones the IntervAI repo (if not present)
  2. Finds your `intervai-data` dataset in /kaggle/input and copies it
     into the repo's data/raw/ folder (the trainer + tokenizer read from there)
  3. Installs dependencies
  4. Retrains the tokenizer (it is NOT in the repo — it's gitignored)
  5. Runs all 6 curriculum stages with the research-grade trainer:
       - Multi-GPU (DataParallel on 2x T4)
       - Auto LR-finder + batch-size profiler per stage
       - Early stopping + best-checkpoint tracking
       - Crash-safe mid-epoch checkpoints (resume within an epoch)
       - Token accuracy / top-5 accuracy metrics
       - JSON training logs in results/<stage>.log

REQUIRED SETUP ON KAGGLE (before running):
  1. New Notebook → Settings → Accelerator = GPU T4 x2 (or P100)
     (2 GPUs are used automatically; 1 GPU also works fine)
  2. Add Data → your `intervai-data` dataset
     (must contain: starcoder_large.jsonl, opencodeinstruct.jsonl,
      conversations.jsonl, codesearchnet.jsonl, codefeedback.jsonl,
      oasst_coding.jsonl, codealpaca.jsonl, mohler_asag.jsonl,
      mmlu_cs.json, cruxeval/cruxeval.jsonl)
  3. Run this cell (paste the whole file, or run the line below):

     !python /kaggle/working/IntervAI/kaggle/train_on_kaggle.py --stage all

  Optional: set INTERVUE_MODEL=large|medium|small to override model size
  (default on T4 is large = 24.3M params).

Checkpoints land in /kaggle/working/IntervAI/models/generator/saved/.
Download them when done. Re-run the same cell to resume after a timeout.
"""

import os
import shutil
import sys
import time
from pathlib import Path

# ── 0. Clone the repo if not already present ──────────────────
WORK = Path("/kaggle/working")
REPO = WORK / "IntervAI"
if not REPO.exists():
    print("Cloning IntervAI repo...")
    os.system("git clone https://github.com/atrishmanm/IntervAI.git /kaggle/working/IntervAI")

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

# ── 3. Copy your dataset into data/raw ────────────────────────
RAW_DIR = REPO / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

def copy_dataset_into_raw():
    """Find the intervai-data dataset in /kaggle/input and copy into data/raw.

    Kaggle preserves whatever folder structure your upload had (e.g. files
    nested under raw/ or data/raw/). We flatten ALL files into data/raw/ by
    basename, keeping the cruxeval/ subfolder (train.py expects it there).
    """
    print("\nLocating dataset in /kaggle/input...")
    input_dir = Path("/kaggle/input")
    if not input_dir.exists():
        raise RuntimeError("/kaggle/input not found — did you add your dataset to the notebook?")

    print("  Contents of /kaggle/input:")
    for d in input_dir.iterdir():
        print(f"    - {d.name}")

    # Find the dataset dir (intervai-data, or whichever has our files)
    dataset_dirs = []
    for d in input_dir.iterdir():
        if not d.is_dir():
            continue
        names = {p.name for p in d.rglob("*") if p.is_file()}
        if {"starcoder_large.jsonl", "conversations.jsonl"} & names:
            dataset_dirs.append(d)
    if not dataset_dirs:
        print("  WARNING: no dataset with expected files found in /kaggle/input.")
        print("  Skipping copy. Training will skip missing files.")
        return

    src = dataset_dirs[0]
    print(f"  Found dataset: {src}")

    # Flatten every file into data/raw/ by basename, but keep files that live
    # in a 'cruxeval' folder under data/raw/cruxeval/.
    copied = 0
    for item in sorted(src.rglob("*")):
        if not item.is_file():
            continue
        parts = item.relative_to(src).parts
        if len(parts) >= 2 and parts[0].lower() == "cruxeval":
            dest = RAW_DIR / "cruxeval" / item.name
        else:
            dest = RAW_DIR / item.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(item, dest)
            copied += 1
            print(f"    + {item.name}")
    print(f"  Copied {copied} files into {RAW_DIR}")

    # Sanity check: find each required file anywhere under data/raw
    required = ["starcoder_large.jsonl", "opencodeinstruct.jsonl", "conversations.jsonl",
                "codesearchnet.jsonl", "oasst_coding.jsonl",
                "codealpaca.jsonl", "mohlerasag_hf.jsonl", "scientsbank.jsonl",
                "beetle.jsonl", "asap_aes.jsonl", "cruxeval/cruxeval.jsonl",
                "fineweb_edu_sample.jsonl", "openwebtext_sample.jsonl"]
    print("\n  Sanity check:")
    for name in required:
        found = None
        for p in RAW_DIR.rglob(Path(name).name):
            if Path(name).name == p.name:
                found = p
                break
        if found:
            print(f"    OK  {name} ({found.stat().st_size/1e6:.1f} MB)")
        else:
            print(f"    !!  {name} MISSING")

    # Show final tree of data/raw
    print("\n  data/raw tree:")
    for p in sorted(RAW_DIR.rglob("*")):
        if p.is_file():
            print(f"    {p.relative_to(RAW_DIR)}  ({p.stat().st_size/1e6:.1f} MB)")

copy_dataset_into_raw()

# ── 4. Train tokenizer (not in repo — gitignored) ─────────────
tok_path = REPO / "tokenizer" / "saved" / "tokenizer.json"
if not tok_path.exists():
    print("\nTraining tokenizer on Kaggle data (16K vocab)...")
    rc = os.system(f"{sys.executable} tokenizer/train_tokenizer.py")
    if rc != 0:
        print("!! Tokenizer training failed — stopping.")
        sys.exit(rc)
else:
    print("\nTokenizer already present, skipping training.")

# ── 5. Run training ───────────────────────────────────────────
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
