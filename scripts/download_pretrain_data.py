#!/usr/bin/env python3
"""
scripts/download_pretrain_data.py
==================================
Downloads additional pretraining data to supplement existing starcoder/codesearchnet.

Usage:
    python scripts/download_pretrain_data.py

Output (in data/raw/):
    fineweb_edu_sample.jsonl  - Sampled FineWeb-Edu (50K educational text)
    openwebtext_sample.jsonl  - Sampled OpenWebText (100K web text)
"""

import json
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


def download_fineweb_edu(max_samples=50000):
    print(f"\n[1/2] FineWeb-Edu (sampling {max_samples} examples)...")
    try:
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-100BT",
                          split="train", streaming=True, trust_remote_code=True)
        records = []
        for i, row in enumerate(ds):
            if i >= max_samples:
                break
            text = row.get("text", "") or row.get("content", "")
            if text and 200 < len(text.strip()) < 50000:
                records.append({"content": text.strip(), "source": "fineweb_edu"})
            if (i + 1) % 10000 == 0:
                print(f"    Processed {i+1} examples, kept {len(records)}")
        return save_jsonl(records, "fineweb_edu_sample.jsonl")
    except Exception as e:
        print(f"  [ERROR] FineWeb-Edu download failed: {e}")
        return 0


def download_openwebtext(max_samples=100000):
    print(f"\n[2/2] OpenWebText (sampling {max_samples} examples)...")
    try:
        from datasets import load_dataset
        ds = load_dataset("Skylion007/openwebtext", split="train",
                          streaming=True, trust_remote_code=True)
        records = []
        for i, row in enumerate(ds):
            if i >= max_samples:
                break
            text = row.get("text", "")
            if text and 200 < len(text.strip()) < 50000:
                records.append({"content": text.strip(), "source": "openwebtext"})
            if (i + 1) % 10000 == 0:
                print(f"    Processed {i+1} examples, kept {len(records)}")
        return save_jsonl(records, "openwebtext_sample.jsonl")
    except Exception as e:
        print(f"  [ERROR] OpenWebText download failed: {e}")
        return 0


def main():
    print("=" * 60)
    print("Pretraining Data Download")
    print("=" * 60)
    ensure_data_dir()
    total = 0
    total += download_fineweb_edu()
    total += download_openwebtext()
    print(f"\n{'=' * 60}")
    print(f"Total new pretraining examples: {total}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
