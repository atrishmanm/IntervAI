"""
models/classifier/evaluate.py
=============================
Evaluates the trained AnswerClassifier on the held-out test split.

Metrics:
  - Overall accuracy
  - Per-class F1, precision, recall (scikit-learn)
  - Confusion matrix (printed to console)

Usage:
    python models/classifier/evaluate.py
"""

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from models.classifier.model import AnswerClassifier, LABEL2ID, ID2LABEL, NUM_CLASSES
from models.classifier.train import ClassifierDataset, make_examples, load_split
from tokenizers import Tokenizer

TOK_PATH  = ROOT / "tokenizer" / "saved" / "tokenizer.json"
CKPT_PATH = ROOT / "models" / "classifier" / "saved" / "best_classifier.pt"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate():
    # Load tokenizer
    if not TOK_PATH.exists():
        print(f"✗ Tokenizer not found: {TOK_PATH}")
        return
    tokenizer = Tokenizer.from_file(str(TOK_PATH))

    # Load checkpoint
    if not CKPT_PATH.exists():
        print(f"✗ Checkpoint not found: {CKPT_PATH}")
        return
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model = AnswerClassifier(vocab_size=ckpt["vocab_size"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"✓ Loaded checkpoint from epoch {ckpt['epoch']} (val_loss={ckpt['val_loss']:.4f})")

    # Load test data
    test_records = load_split("test")
    if not test_records:
        print("No test records found.")
        return
    test_ex = make_examples(test_records)
    test_ds = ClassifierDataset(test_ex, tokenizer)
    loader  = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
    print(f"Test examples: {len(test_ex):,}")

    # Run inference
    all_preds, all_labels = [], []
    with torch.no_grad():
        for ids, labels in loader:
            ids = ids.to(DEVICE)
            logits = model(ids)
            preds  = logits.argmax(dim=-1).cpu().tolist()
            all_preds  .extend(preds)
            all_labels .extend(labels.tolist())

    # Metrics
    try:
        from sklearn.metrics import (
            classification_report, confusion_matrix, accuracy_score
        )
        label_names = [ID2LABEL[i] for i in range(NUM_CLASSES)]

        acc = accuracy_score(all_labels, all_preds)
        print(f"\nOverall Accuracy: {acc:.4f}")

        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=label_names))

        cm = confusion_matrix(all_labels, all_preds)
        print("Confusion Matrix (rows=true, cols=pred):")
        header = f"{'':20s}" + " ".join(f"{n[:8]:>10s}" for n in label_names)
        print(header)
        for i, row in enumerate(cm):
            row_str = " ".join(f"{v:>10d}" for v in row)
            print(f"{label_names[i][:20]:20s} {row_str}")
    except ImportError:
        # Manual accuracy if sklearn not available
        correct = sum(p == l for p, l in zip(all_preds, all_labels))
        print(f"\nAccuracy: {correct / len(all_labels):.4f}")
        print("Install scikit-learn for full classification report: pip install scikit-learn")


if __name__ == "__main__":
    print("=" * 60)
    print("AI Interview Prep — Classifier Evaluation")
    print("=" * 60)
    evaluate()
