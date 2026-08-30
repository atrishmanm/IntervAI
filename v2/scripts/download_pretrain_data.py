#!/usr/bin/env python3
"""
scripts/download_pretrain_data.py
===================================
Downloads additional pretraining data to supplement existing starcoder/codesearchnet.
Downloads are sampled to keep total manageable (~100K-200K examples).

Usage:
    python scripts/download_pretrain_data.py

Output (in data/raw/):
    nemotron_code_sample.jsonl  - Sampled Nemotron code (100K)
    fineweb_edu_sample.jsonl    - Sampled FineWeb-Edu (50K)
"""

import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"


def ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def save_jsonl(records, filename):
    path = DATA_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records)} records -> {path}")
    return len(records)


def download_nemotron_code(max_samples=100000):
    """Download and sample from Nemotron-Pretraining-Code-v1."""
    print(f"\n[1/2] Nemotron Code (sampling {max_samples} examples)...")
    try:
        from datasets import load_dataset
        ds = load_dataset("nvidia/Nemotron-Pretraining-Code-v1", split="train",
                          streaming=True, trust_remote_code=True)
        records = []
        for i, row in enumerate(ds):
            if i >= max_samples:
                break
            content = row.get("content", "") or row.get("text", "")
            lang = row.get("language", "") or row.get("lang", "")
            if content and len(content.strip()) > 100:
                records.append({
                    "content": content.strip(),
                    "language": lang,
                    "source": "nemotron_code",
                })
            if (i + 1) % 10000 == 0:
                print(f"    Processed {i+1} examples, kept {len(records)}")
        return save_jsonl(records, "nemotron_code_sample.jsonl")
    except Exception as e:
        print(f"  [ERROR] Nemotron Code download failed: {e}")
        print("  [INFO] This dataset is large (~16GB). May need manual download.")
        return 0


def download_fineweb_edu(max_samples=50000):
    """Download and sample from FineWeb-Edu (educational text)."""
    print(f"\n[2/2] FineWeb-Edu (sampling {max_samples} examples)...")
    try:
        from datasets import load_dataset
        # FineWeb-Edu is very large, use streaming
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-100BT",
                          split="train", streaming=True, trust_remote_code=True)
        records = []
        for i, row in enumerate(ds):
            if i >= max_samples:
                break
            text = row.get("text", "") or row.get("content", "")
            # Quality filter: educational content should be substantial
            if text and len(text.strip()) > 200 and len(text.strip()) < 50000:
                records.append({
                    "content": text.strip(),
                    "source": "fineweb_edu",
                })
            if (i + 1) % 10000 == 0:
                print(f"    Processed {i+1} examples, kept {len(records)}")
        return save_jsonl(records, "fineweb_edu_sample.jsonl")
    except Exception as e:
        print(f"  [ERROR] FineWeb-Edu download failed: {e}")
        return 0


def main():
    print("=" * 60)
    print("Pretraining Data Download")
    print("=" * 60)
    print(f"Output directory: {DATA_DIR}")
    print("Note: Existing starcoder_large.jsonl, codesearchnet.jsonl, cruxeval/ are unchanged.")

    ensure_data_dir()

    total = 0
    total += download_nemotron_code()
    total += download_fineweb_edu()

    print(f"\n{'=' * 60}")
    print(f"Total new pretraining examples: {total}")
    print(f"{'=' * 60}")

    if total == 0:
        print("\n[WARNING] No new data downloaded.")
        print("The existing pretrain data (starcoder + codesearchnet + cruxeval) is still available.")


if __name__ == "__main__":
    main()
