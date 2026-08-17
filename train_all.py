"""
train_all.py
============
Master training pipeline for INTERVUE — ChatGPT-Like Coding Interview Model.

Steps:
  Phase 0: Data Engineering
    1. Download all datasets
    2. Extract questions (existing)
    3. Build question bank (existing)
    4. Transform data into training formats
    5. Build concept graph

  Phase 1: Tokenizer
    6. Train tokenizer (16K vocab, ByteLevel)

  Phase 2-3: Model Training
    7. Train generator (pretrain → domain → instruction → interview → evaluator → followup)

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

        # 1. Download all datasets
        step(1, TOTAL_DATA, "Downloading all datasets")
        raw_exists = (ROOT / "data" / "raw" / "conversations.jsonl").exists()
        if args.skip_download and raw_exists:
            print("  Skipping download (data exists).")
        else:
            from data_pipeline.download_datasets import main as dl_main
            run_step(dl_main, "Download all datasets")

        # 2. Extract questions (existing pipeline)
        step(2, TOTAL_DATA, "Extracting questions")
        from data_pipeline.clean_data import main as clean_main
        run_step(clean_main, "Clean data")

        # 3. Build question bank
        step(3, TOTAL_DATA, "Building SQLite question bank")
        from data_pipeline.build_question_bank import main as bank_main
        run_step(bank_main, "Question bank")

        # 4. Transform data into training formats
        step(4, TOTAL_DATA, "Transforming data into training formats")
        from data_pipeline.transform_training_data import main as transform_main
        run_step(transform_main, "Transform training data")

        # 5. Build concept graph
        step(5, TOTAL_DATA, "Building concept graph")
        from data_pipeline.build_concept_graph import main as concept_main
        run_step(concept_main, "Concept graph")

    # ── Phase 1: Tokenizer ──
    if args.stage in ("all", "tokenizer"):
        banner("Phase 1: Tokenizer Training")
        from tokenizer.train_tokenizer import main as tok_main
        run_step(tok_main, "Train tokenizer")

    # ── Phase 2-3: Model Training ──
    if args.stage in ("all", "train") and not args.skip_train:
        banner("Phase 2-3: Model Training")
        print("\nNote: Model training requires a GPU (or Colab).")
        print("Each stage will save checkpoints. You can resume if interrupted.\n")

        stages = [
            ("Stage 1: Code Pretraining", "models.generator.train_pretrain"),
            ("Stage 2: CS Domain Training", "models.generator.train_domain"),
            ("Stage 3: Instruction Tuning", "models.generator.train_instruction"),
            ("Stage 4: Interview Dialogue", "models.generator.train_interview"),
            ("Stage 5: Answer Evaluation", "models.generator.train_evaluator"),
            ("Stage 6: Follow-up Generation", "models.generator.train_followup"),
        ]

        for i, (name, module_path) in enumerate(stages, 1):
            step(i, len(stages), name)
            try:
                import importlib
                mod = importlib.import_module(module_path)
                run_step(mod.main, name)
            except Exception as e:
                print(f"\nX {name} failed: {e}")
                print("  You can re-run this stage individually.")
                import traceback
                traceback.print_exc()

    # ── Done ──
    banner("Pipeline Complete!")
    print("\nNext steps:")
    print("  1. Start the backend:")
    print("       python backend/main.py")
    print("  2. Open http://localhost:8000 in your browser")
    print()
    print("To train model stages individually:")
    print("  python models/generator/train_pretrain.py")
    print("  python models/generator/train_domain.py")
    print("  python models/generator/train_instruction.py")
    print("  python models/generator/train_interview.py")
    print("  python models/generator/train_evaluator.py")
    print("  python models/generator/train_followup.py")
    print()


if __name__ == "__main__":
    main()
