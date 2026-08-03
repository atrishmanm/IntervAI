"""
models/ranker/evaluate.py
=========================
Evaluates the QuestionRanker on the held-out test split.

Metrics:
  - Precision@1 (exact top match)
  - Precision@5 (target in top-5)

Usage:
    python models/ranker/evaluate.py
"""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from models.ranker.model import QuestionRanker
from models.ranker.train import (
    build_pairs, eval_recall_at_k, load_split, RankerDataset
)
from tokenizers import Tokenizer

TOK_PATH  = ROOT / "tokenizer" / "saved" / "tokenizer.json"
CKPT_PATH = ROOT / "models" / "ranker" / "saved" / "best_ranker.pt"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate():
    if not TOK_PATH.exists():
        print(f"✗ Tokenizer not found: {TOK_PATH}")
        return
    if not CKPT_PATH.exists():
        print(f"✗ Checkpoint not found: {CKPT_PATH}")
        return

    tokenizer = Tokenizer.from_file(str(TOK_PATH))
    ckpt      = torch.load(CKPT_PATH, map_location=DEVICE)
    model     = QuestionRanker(vocab_size=ckpt["vocab_size"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"✓ Loaded checkpoint from epoch {ckpt['epoch']}")

    test_records = load_split("test")
    if not test_records:
        print("No test records.")
        return

    test_pairs = build_pairs(test_records, queries_per_record=1)
    print(f"Test pairs: {len(test_pairs):,}")

    for k in [1, 5, 10]:
        recall = eval_recall_at_k(model, test_pairs, tokenizer, k=k)
        print(f"Recall@{k}: {recall:.4f}")

    # Qualitative examples
    print("\n--- Qualitative Retrieval Examples ---")
    sample = test_pairs[:5]
    for i, (q_text, d_text) in enumerate(sample, 1):
        print(f"\nExample {i}:")
        print(f"  Query: {q_text}")
        print(f"  Expected doc: {d_text[:100]}...")


if __name__ == "__main__":
    print("=" * 60)
    print("AI Interview Prep — Ranker Evaluation")
    print("=" * 60)
    evaluate()
