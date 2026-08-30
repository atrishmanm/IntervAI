"""
data_pipeline/download_datasets.py
==================================
Downloads all datasets for INTERVUE.

Datasets:
  1. stindardlogic/coding-interview-sft-100k  (100K multi-turn coding interviews)
  2. sahil2801/CodeAlpaca-20k                 (20K code instruction/response)
  3. m-a-p/CodeFeedback-Filtered-Instruction   (156K high-quality code instructions)
  4. nkazi/MohlerASAG                          (2,442 real student answers with grades)
  5. cruxeval-org/cruxeval                     (800 code tracing examples)
  6. MMLU CS                                   (already in project)
  7. bigcode/starcoderdata                     (streaming sample for pretraining)
"""

import json
import sys
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"


def download_standardlogic(max_records=100_000):
    """Download the full StandardLogic coding interview dataset (up to 100K)."""
    out_path = RAW_DIR / "conversations.jsonl"
    if out_path.exists():
        existing = sum(1 for _ in open(out_path, encoding="utf-8"))
        if existing >= max_records * 0.9:
            print(f"  [SKIP] StandardLogic already exists ({existing} records)")
            return
    print("[1/6] Downloading StandardLogic (coding-interview-sft-100k)...")
    try:
        ds = load_dataset("stindardlogic/coding-interview-sft-100k", split="train")
    except Exception as e:
        print(f"  Error: {e}")
        return

    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for item in ds:
            if count >= max_records:
                break
            convs = item.get("conversations", [])
            if not convs:
                continue
            record = {
                "id": item.get("id", f"conv_{count}"),
                "conversations": convs,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    print(f"  Saved {count} conversations to {out_path}")


def download_codealpaca():
    """Download CodeAlpaca 20K instruction dataset."""
    out_path = RAW_DIR / "codealpaca.jsonl"
    if out_path.exists():
        print(f"  [SKIP] CodeAlpaca already exists")
        return
    print("[2/6] Downloading CodeAlpaca-20k...")
    try:
        ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    except Exception as e:
        print(f"  Error: {e}")
        return

    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for item in ds:
            record = {
                "instruction": item.get("instruction", ""),
                "input": item.get("input", ""),
                "output": item.get("output", ""),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    print(f"  Saved {count} records to {out_path}")


def download_codefeedback(max_records=156_000):
    """Download CodeFeedback filtered instruction dataset."""
    out_path = RAW_DIR / "codefeedback.jsonl"
    if out_path.exists():
        print(f"  [SKIP] CodeFeedback already exists")
        return
    print("[3/6] Downloading CodeFeedback-Filtered-Instruction...")
    try:
        ds = load_dataset("m-a-p/CodeFeedback-Filtered-Instruction", split="train")
    except Exception as e:
        print(f"  Error: {e}")
        return

    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for item in ds:
            if count >= max_records:
                break
            record = {
                "instruction": item.get("instruction", item.get("query", "")),
                "response": item.get("response", item.get("output", "")),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    print(f"  Saved {count} records to {out_path}")


def download_mohler():
    """Download Mohler ASAG dataset — real student answers with human grades."""
    out_path = RAW_DIR / "mohler_asag.jsonl"
    if out_path.exists():
        print(f"  [SKIP] Mohler ASAG already exists")
        return
    print("[4/6] Downloading Mohler ASAG...")
    try:
        ds = load_dataset("nkazi/MohlerASAG", name="raw", split="open_ended")
    except Exception as e:
        print(f"  Error: {e}")
        return

    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for item in ds:
            record = {
                "id": item.get("id", ""),
                "question": item.get("question", ""),
                "instructor_answer": item.get("instructor_answer", ""),
                "student_answer": item.get("student_answer", ""),
                "score_grader_1": item.get("score_grader_1", 0),
                "score_grader_2": item.get("score_grader_2", 0),
                "score_avg": item.get("score_avg", 0),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    print(f"  Saved {count} student answers to {out_path}")

    # Also download the curated version if available
    try:
        ds_curated = load_dataset("nkazi/MohlerASAG-Curated", split="train")
        curated_path = RAW_DIR / "mohler_curated.jsonl"
        count_c = 0
        with open(curated_path, "w", encoding="utf-8") as f:
            for item in ds_curated:
                f.write(json.dumps(dict(item), ensure_ascii=False) + "\n")
                count_c += 1
        print(f"  Saved {count_c} curated records to {curated_path}")
    except Exception:
        pass  # Curated version may not exist yet


def download_cruxeval():
    """Download CRUXEval code tracing dataset."""
    out_dir = RAW_DIR / "cruxeval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cruxeval.jsonl"
    if out_path.exists():
        print(f"  [SKIP] CRUXEval already exists")
        return
    print("[5/6] Downloading CRUXEval...")
    try:
        ds = load_dataset("cruxeval-org/cruxeval", split="test")
        ds.to_json(str(out_path))
        print(f"  Saved CRUXEval to {out_path}")
    except Exception as e:
        print(f"  Error: {e}")


def download_starcoder_sample(max_lines=50_000):
    """Download a sample of StarCoder data for code pretraining."""
    out_path = RAW_DIR / "starcoder_sample.jsonl"
    if out_path.exists():
        print(f"  [SKIP] StarCoder sample already exists")
        return
    print("[6/6] Downloading StarCoder sample (streaming)...")
    try:
        ds = load_dataset(
            "bigcode/starcoderdata",
            data_dir="python",
            split="train",
            streaming=True,
        )
    except Exception:
        try:
            ds = load_dataset("bigcode/starcoderdata", split="train", streaming=True)
        except Exception as e:
            print(f"  Error: {e}")
            return

    count = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for item in ds:
            if count >= max_lines:
                break
            # StarCoder has 'content' field with raw code
            content = item.get("content", "")
            if content and len(content) > 100:
                record = {
                    "content": content[:10000],  # Cap per file at 10K chars
                    "repo": item.get("repo", ""),
                    "path": item.get("path", ""),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    print(f"  Saved {count} code samples to {out_path}")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("INTERVUE — Dataset Download Pipeline")
    print("=" * 60)
    print(f"Output directory: {RAW_DIR}\n")

    download_standardlogic()
    download_codealpaca()
    download_codefeedback()
    download_mohler()
    download_cruxeval()
    download_starcoder_sample()

    # Summary
    print("\n" + "=" * 60)
    print("Download Summary:")
    print("=" * 60)
    for f in sorted(RAW_DIR.rglob("*.jsonl")):
        lines = sum(1 for _ in open(f, encoding="utf-8"))
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name:40s} {lines:>8,} records  ({size_mb:.1f} MB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
