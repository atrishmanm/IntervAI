"""
data_pipeline/download_datasets.py
==================================
Downloads large-scale HuggingFace datasets for AI Interview Prep V2.

Datasets:
  - stindardlogic/coding-interview-sft-100k: 100k conversational coding interviews.
    Provides deep, multi-turn dialogues for training the generative model.
"""

import json
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download
from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / "conversations.jsonl"

    print("Downloading 'stindardlogic/coding-interview-sft-100k'...")
    try:
        ds = load_dataset("stindardlogic/coding-interview-sft-100k", split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    print(f"Loaded {len(ds)} conversations. Saving to JSONL...")

    # We will sample 10,000 conversations for speed, 
    # but the pipeline handles arbitrary sizes.
    max_records = 10000
    count = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for item in ds:
            if count >= max_records:
                break
            # The dataset has 'conversations' which is a list of dicts.
            # Usually format is [{'from': 'human', 'value': '...'}, {'from': 'gpt', 'value': '...'}]
            convs = item.get("conversations", [])
            if not convs:
                continue
                
            record = {
                "id": item.get("id", f"conv_{count}"),
                "conversations": convs
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"✓ Saved {count} conversations to {out_path}")

    # 2. Download CRUXEval
    CRUX_DIR = RAW_DIR / "cruxeval"
    CRUX_DIR.mkdir(parents=True, exist_ok=True)
    crux_file = CRUX_DIR / "cruxeval.jsonl"
    
    if not crux_file.exists():
        print("  Downloading cruxeval-org/cruxeval...")
        try:
            ds = load_dataset("cruxeval-org/cruxeval", split="test")
            ds.to_json(str(crux_file))
            print(f"  Saved cruxeval to {crux_file}")
        except Exception as e:
            print(f"  Failed to download cruxeval: {e}")
            print("  Make sure `datasets` and `pyarrow` are installed.")
    else:
        print("  cruxeval dataset already exists.")

    print("\n+ Downloads complete.")


if __name__ == "__main__":
    main()
