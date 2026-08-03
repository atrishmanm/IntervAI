"""
train_all.py
============
Master training pipeline for AI Interview Prep V3.

Steps:
  1. Download datasets (stindardlogic/coding-interview-sft-100k)
  2. Clean data (extract theoretical/output/concept questions)
  3. Build question bank (SQLite)

Usage:
    python train_all.py [--skip-download]
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def banner(msg: str):
    print("\n" + "=" * 60)
    print(f"  {msg}")
    print("=" * 60)


def step(n: int, total: int, msg: str):
    print(f"\n[Step {n}/{total}] {msg}")
    print("-" * 50)


def run_step(fn, step_name: str):
    t0 = time.time()
    try:
        fn()
        elapsed = time.time() - t0
        print(f"\n+ '{step_name}' completed in {elapsed:.1f}s")
        return True
    except SystemExit:
        raise
    except Exception as e:
        print(f"\nX '{step_name}' failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse
    p = argparse.ArgumentParser(description="AI Interview Prep V3 -- Training Pipeline")
    p.add_argument("--skip-download", action="store_true", help="Skip download if data exists")
    args = p.parse_args()

    banner("AI Interview Prep V3 -- Training Pipeline")
    TOTAL = 3

    # 1. Download
    step(1, TOTAL, "Downloading datasets")
    raw_exists = (ROOT / "data" / "raw" / "conversations.jsonl").exists()
    if args.skip_download and raw_exists:
        print("  Skipping download (data exists).")
    else:
        from data_pipeline.download_datasets import main as dl_main
        run_step(dl_main, "Download")

    # 2. Clean
    step(2, TOTAL, "Extracting questions (theoretical, output, concept)")
    from data_pipeline.clean_data import main as clean_main
    run_step(clean_main, "Clean data")

    # 3. Build question bank
    step(3, TOTAL, "Building SQLite question bank")
    from data_pipeline.build_question_bank import main as bank_main
    run_step(bank_main, "Question bank")


    banner("Training Pipeline Complete!")
    print("\nNext steps:")
    print("  1. Start the backend:")
    print("       python backend/main.py")
    print("  2. Open http://localhost:8000 in your browser")
    print()


if __name__ == "__main__":
    main()
