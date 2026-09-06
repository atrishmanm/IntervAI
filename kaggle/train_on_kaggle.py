"""
kaggle/train_on_kaggle.py
=========================
Run INTERVUE research-grade training on Kaggle (1x or 2x T4/P100) — fully
automatic, resume-safe, checkpoint-enabled.

What it does:
  1. Clones the IntervAI repo (if not present)
  2. Finds your `intervai-data` dataset in /kaggle/input and copies it
     into the repo's data/raw/ folder
  3. Installs dependencies
  4. Retrains the tokenizer (if not present)
  5. Runs all 8 curriculum stages with checkpoint resume:
       - pretrain → domain → instruction → interview → evaluator → followup → resume_finetune → negotiation
       - Multi-GPU (DataParallel on 2x T4)
       - Auto LR-finder + batch-size profiler per stage
       - Early stopping + best-checkpoint tracking
       - Crash-safe mid-epoch checkpoints (resume within an epoch)
       - Token accuracy / top-5 accuracy metrics
       - JSON training logs in results/<stage>.log

TIME BUDGET: <8 hours total on T4/P100
"""

import os
import shutil
import sys
import time
import json
import signal
from pathlib import Path

# ── Curriculum Stages & Proportional Time Budgeting ──────────
ALL_STAGES = [
    "pretrain", "domain", "instruction", "interview",
    "evaluator", "followup", "resume_finetune", "negotiation"
]
STAGE_NAMES = ALL_STAGES

# Stage-specific budget allocation (proportional to data volume and task importance)
STAGE_BUDGET_WEIGHTS = {
    "pretrain": 0.35,        # ~168 min (largest dataset)
    "domain": 0.15,          # ~72 min
    "instruction": 0.12,     # ~58 min
    "interview": 0.18,       # ~86 min (crucial for interview dialog)
    "evaluator": 0.08,       # ~38 min
    "followup": 0.04,        # ~20 min
    "resume_finetune": 0.04, # ~20 min
    "negotiation": 0.04,     # ~20 min
}

def get_stage_time_limit_minutes(stage_idx: int, total_budget_minutes: float = 480.0) -> float:
    """Calculate proportional time allocation for a curriculum stage."""
    if 0 <= stage_idx < len(STAGE_NAMES):
        stage_name = STAGE_NAMES[stage_idx]
        weight = STAGE_BUDGET_WEIGHTS.get(stage_name, 0.1)
        return round(total_budget_minutes * weight, 1)
    return round(total_budget_minutes * 0.1, 1)

# ── Time budget tracking ──────────────────────────────────────
START_TIME = time.time()
TIME_BUDGET_MINUTES = 480  # 8 hours default

def time_remaining_minutes():
    elapsed = (time.time() - START_TIME) / 60
    return max(0, TIME_BUDGET_MINUTES - elapsed)

def check_time_budget(stage_name):
    remaining = time_remaining_minutes()
    if remaining < 10:
        print(f"\n{'='*60}")
        print(f"  TIME BUDGET EXCEEDED ({TIME_BUDGET_MINUTES} min)")
        print(f"  Elapsed: {(time.time()-START_TIME)/60:.1f} min")
        print(f"  Stopping before stage: {stage_name}")
        print(f"  Use --resume to continue from checkpoint")
        print(f"{'='*60}")
        sys.exit(0)


def parse_args():
    args = {
        "stage": "all",
        "resume": False,
        "time_budget": 480,
    }
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--stage" and i < len(sys.argv) - 1:
            args["stage"] = sys.argv[i + 1]
        elif arg == "--resume":
            args["resume"] = True
        elif arg == "--time-budget" and i < len(sys.argv) - 1:
            args["time_budget"] = int(sys.argv[i + 1])
    return args


def _pip(*pkgs):
    os.system(f"{sys.executable} -m pip install -q {' '.join(pkgs)}")


def copy_dataset_into_raw(raw_dir: Path):
    """Find the intervai-data dataset in /kaggle/input and copy into data/raw."""
    print("\nLocating dataset in /kaggle/input...")
    input_dir = Path("/kaggle/input")
    if not input_dir.exists():
        raise RuntimeError("/kaggle/input not found — did you add your dataset to the notebook?")

    print("  Contents of /kaggle/input:")
    for d in input_dir.iterdir():
        print(f"    - {d.name}")

    # Find the dataset dir
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

    # Only copy files we actually need for training
    REQUIRED_FILES = {
        "starcoder_large.jsonl",
        "codesearchnet.jsonl",
        "codefeedback.jsonl",
        "opencodeinstruct.jsonl",
        "oasst_coding.jsonl",
        "codealpaca.jsonl",
        "conversations.jsonl",
        "interview_sft_100k.jsonl",
        "mohler_asag.jsonl",
        "resumes_54k.jsonl",
        "resumes_54k.json",
        "negotiation_sft_100k.jsonl",
        "kodcode_verified.jsonl",
        "cruxeval.jsonl",
        "cruxeval.json",
    }

    copied = 0
    skipped = 0
    for item in sorted(src.rglob("*")):
        if not item.is_file():
            continue
        # Check if this file is needed
        if item.name not in REQUIRED_FILES:
            skipped += 1
            continue
        parts = item.relative_to(src).parts
        if len(parts) >= 2 and parts[0].lower() == "cruxeval":
            dest = raw_dir / "cruxeval" / item.name
        else:
            dest = raw_dir / item.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(item, dest)
            copied += 1
            print(f"    + {item.name}")
        else:
            print(f"    = {item.name} (exists)")

    # Ensure aliases exist for json vs jsonl
    for base in ["resumes_54k", "cruxeval"]:
        f_json = raw_dir / f"{base}.json"
        f_jsonl = raw_dir / f"{base}.jsonl"
        if f_json.exists() and not f_jsonl.exists():
            shutil.copy2(f_json, f_jsonl)
            print(f"    + alias: created {f_jsonl.name} from {f_json.name}")
        elif f_jsonl.exists() and not f_json.exists():
            shutil.copy2(f_jsonl, f_json)
            print(f"    + alias: created {f_json.name} from {f_jsonl.name}")

    print(f"  Copied {copied} files, skipped {skipped} unnecessary files")


def main():
    global TIME_BUDGET_MINUTES, START_TIME
    START_TIME = time.time()

    # ── 0. Clone the repo if not already present on Kaggle ─────────
    work = Path("/kaggle/working")
    repo = work / "IntervAI"
    if work.exists() and not repo.exists():
        print("Cloning IntervAI repo...")
        os.system("git clone https://github.com/atrishmanm/IntervAI.git /kaggle/working/IntervAI")

    if repo.exists():
        sys.path.insert(0, str(repo))
        os.chdir(repo)

    # ── 1. Environment summary ────────────────────────────────────
    from env_config import ENV, ROOT, SAVE_ROOT, MODEL_SIZE, print_env_summary
    print_env_summary()
    assert ENV == "kaggle", f"Expected Kaggle env, got {ENV}"

    # ── 2. Parse arguments ────────────────────────────────────────
    args = parse_args()
    TIME_BUDGET_MINUTES = args["time_budget"]
    resume = args["resume"]

    print(f"\n  Configuration:")
    print(f"    Stage: {args['stage']}")
    print(f"    Resume: {resume}")
    print(f"    Time budget: {TIME_BUDGET_MINUTES} min")
    print(f"    Model: {MODEL_SIZE}")

    # ── 3. Install deps (idempotent) ──────────────────────────────
    _pip("torch", "tokenizers", "numpy", "scikit-learn", "tqdm", "safetensors", "datasets")

    # ── 4. Copy dataset ───────────────────────────────────────────
    raw_dir = ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    copy_dataset_into_raw(raw_dir)

    # ── 5. Train tokenizer (if not present) ───────────────────────
    tok_path = ROOT / "tokenizer" / "saved" / "tokenizer.json"
    if not tok_path.exists():
        print("\nTraining tokenizer on Kaggle data (16K vocab)...")
        rc = os.system(f"{sys.executable} tokenizer/train_tokenizer.py")
        if rc != 0:
            print("!! Tokenizer training failed — stopping.")
            sys.exit(rc)
    else:
        print("\nTokenizer already present, skipping training.")

    # ── 6. Check available checkpoints for resume ─────────────────
    ckpt_dir = SAVE_ROOT
    print(f"\n  Checkpoint directory: {ckpt_dir}")
    existing_ckpts = list(ckpt_dir.glob("*.pt")) if ckpt_dir.exists() else []
    if existing_ckpts:
        print(f"  Found {len(existing_ckpts)} existing checkpoint(s):")
        for p in sorted(existing_ckpts):
            print(f"    - {p.name} ({p.stat().st_size/1e6:.1f} MB)")
    else:
        print("  No existing checkpoints found.")

    # ── 7. Run training with time budget ──────────────────────────
    stages = [args["stage"]] if args["stage"] != "all" else ALL_STAGES

    progress_file = ckpt_dir / "training_progress.json"

    def save_progress(stage, status, elapsed_min):
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        progress = {}
        if progress_file.exists():
            with open(progress_file) as f:
                progress = json.load(f)
        progress[stage] = {
            "status": status,
            "elapsed_min": round(elapsed_min, 2),
            "timestamp": time.time(),
        }
        with open(progress_file, "w") as f:
            json.dump(progress, f, indent=2)

    if resume and progress_file.exists():
        with open(progress_file) as f:
            progress = json.load(f)
        completed_stages = [s for s, p in progress.items() if p.get("status") == "completed"]
        print(f"\n  Resume mode: skipping completed stages: {completed_stages}")
        stages = [s for s in stages if s not in completed_stages]
        if not stages:
            print("  All stages already completed!")
            return

    total_t0 = time.time()
    completed = []
    failed = []

    for s in stages:
        check_time_budget(s)
        expected_stage_budget = round(TIME_BUDGET_MINUTES * STAGE_BUDGET_WEIGHTS.get(s, 0.1), 1)

        print(f"\n{'='*60}")
        print(f"  RUNNING STAGE: {s}")
        print(f"  Target stage budget: ~{expected_stage_budget} min | Total time remaining: {time_remaining_minutes():.1f} min")
        print(f"{'='*60}")

        t0 = time.time()
        try:
            rc = os.system(f"{sys.executable} models/generator/train.py --stage {s} --time-budget {expected_stage_budget}")
            elapsed = (time.time() - t0) / 60

            if rc != 0:
                print(f"!! Stage '{s}' FAILED (exit {rc})")
                failed.append(s)
                save_progress(s, "failed", elapsed)
                continue

            print(f"  Stage {s} completed in {elapsed:.1f} min")
            completed.append(s)
            save_progress(s, "completed", elapsed)

        except Exception as e:
            elapsed = (time.time() - t0) / 60
            print(f"!! Stage '{s}' EXCEPTION: {e}")
            failed.append(s)
            save_progress(s, "failed", elapsed)
            continue

    # ── 8. Final summary ─────────────────────────────────────────
    total_all = (time.time() - START_TIME) / 60
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Total time: {total_all:.1f} min ({total_all/60:.1f} hours)")
    print(f"  Stages completed: {len(completed)}/{len(stages)}")
    if completed:
        print(f"    Completed: {', '.join(completed)}")
    if failed:
        print(f"    Failed: {', '.join(failed)}")

    # Post-training quality benchmark check
    print(f"\n{'='*60}")
    print("  POST-TRAINING GENERATIVE BENCHMARK")
    print(f"{'='*60}")
    try:
        from backend.model_service import get_model_service
        ms = get_model_service()
        prompt = ms._format_prompt(
            system="You are an expert technical interviewer at a top tech company.",
            user="Tell me about a time you designed a scalable distributed system."
        )
        res = ms.generate("interview", prompt, max_new_tokens=50)
        print(f"Sample Question/Response Output:\n\"{res}\"\n")
    except Exception as e:
        print(f"Generative benchmark note: {e}")

    print(f"\n  Download /kaggle/working/IntervAI/models/generator/saved/*.pt to your laptop!")


if __name__ == "__main__":
    main()
