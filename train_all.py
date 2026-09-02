"""
train_all.py
============
Master training pipeline for INTERVUE — ChatGPT-Like Coding Interview Model.

Steps:
  Phase 0: Data Engineering
    1. Download evaluator data (real scoring datasets)
    2. Download pretraining data
    3. Extract questions, build question bank, transform training data, build concept graph

  Phase 1: Tokenizer
    4. Train tokenizer (16K vocab, ByteLevel, case-sensitive)

  Phase 2-3: Model Training (120M params, <8h total)
    5. Train generator (pretrain → domain → instruction → interview → evaluator → followup)

Usage:
    python train_all.py [--skip-download] [--stage all|data|tokenizer|train]
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
    p = argparse.ArgumentParser(description="INTERVUE — Full Training Pipeline")
    p.add_argument("--skip-download", action="store_true", help="Skip download if data exists")
    p.add_argument("--stage", default="all", choices=["all", "data", "tokenizer", "train"],
                    help="Which stage to run")
    p.add_argument("--skip-train", action="store_true", help="Skip model training stages")
    args = p.parse_args()

    banner("INTERVUE — Training Pipeline")

    # ── Phase 0: Data Engineering ──
    if args.stage in ("all", "data"):
        banner("Phase 0: Data Engineering")

        TOTAL_DATA = 5

        # 1. Download evaluator data (real scoring datasets)
        step(1, TOTAL_DATA, "Downloading evaluator data")
        from scripts.download_evaluator_data import main as eval_dl_main
        run_step(eval_dl_main, "Evaluator data download")

        # 2. Download pretraining data
        step(2, TOTAL_DATA, "Downloading pretraining data")
        from scripts.download_pretrain_data import main as pretrain_dl_main
        run_step(pretrain_dl_main, "Pretraining data download")

        # 3. Extract questions (existing pipeline)
        step(3, TOTAL_DATA, "Extracting questions")
        from data_pipeline.clean_data import main as clean_main
        run_step(clean_main, "Clean data")

        # 4. Build question bank
        step(4, TOTAL_DATA, "Building SQLite question bank")
        from data_pipeline.build_question_bank import main as bank_main
        run_step(bank_main, "Question bank")

        # 5. Transform data into training formats
        step(5, TOTAL_DATA, "Transforming data into training formats")
        from data_pipeline.transform_training_data import main as transform_main
        run_step(transform_main, "Transform training data")

    # ── Phase 1: Tokenizer ──
    if args.stage in ("all", "tokenizer"):
        banner("Phase 1: Tokenizer Training")
        from tokenizer.train_tokenizer import main as tok_main
        run_step(tok_main, "Train tokenizer")

    # ── Phase 2-3: Model Training ──
    if args.stage in ("all", "train") and not args.skip_train:
        banner("Phase 2-3: Model Training (120M params)")
        print("\nNote: Model training requires a GPU (or Colab/Kaggle).")
        print("Each stage will save checkpoints. You can resume if interrupted.\n")

        import os
        import subprocess

        # Run the unified train.py
        train_script = ROOT / "models" / "generator" / "train.py"
        rc = subprocess.run(
            [sys.executable, str(train_script), "--stage", "all"],
            cwd=str(ROOT),
        )
        if rc.returncode != 0:
            print(f"\nX Training failed (exit {rc.returncode}). Check logs above.")

    # ── Done ──
    banner("Pipeline Complete!")
    print("\nNext steps:")
    print("  1. Download checkpoints from models/generator/saved/")
    print("  2. Start the backend:")
    print("       python backend/main.py")
    print("  3. Open http://localhost:8000 in your browser")
    print()


if __name__ == "__main__":
    main()
