"""
models/classifier/train.py
==========================
Trains the AnswerClassifier from scratch.

What it does:
  1. Loads the trained tokenizer
  2. Loads train/val splits from data/splits/
  3. Synthesises labelled (student_answer, reference_answer, topic) triples
  4. Trains with AdamW + linear warmup + cosine decay
  5. Saves the best checkpoint (by val loss) to models/classifier/saved/

Label synthesis strategy:
  correct          → reference_answer itself (ground truth)
  partially_correct → first half of reference_answer
  incorrect        → a randomly sampled reference from a *different* question
  off_topic        → a completely unrelated sentence

Usage:
    python models/classifier/train.py [--epochs 20] [--batch-size 32]
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from models.classifier.model import AnswerClassifier, LABEL2ID, NUM_CLASSES
from tokenizers import Tokenizer

SPLITS_DIR = ROOT / "data" / "splits"
TOK_PATH   = ROOT / "tokenizer" / "saved" / "tokenizer.json"
SAVE_DIR   = ROOT / "models" / "classifier" / "saved"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED   = 42

OFF_TOPIC_SENTENCES = [
    "The weather today is sunny and warm.",
    "I enjoy cooking Italian food on weekends.",
    "The French revolution began in 1789.",
    "Photosynthesis converts sunlight into glucose.",
    "The Amazon river is the world's largest river.",
    "Basketball was invented in 1891 by James Naismith.",
    "Mount Everest is the highest mountain on Earth.",
    "The speed of light is approximately 300,000 km/s.",
    "Shakespeare wrote Hamlet in the early 17th century.",
    "Water freezes at 0 degrees Celsius at standard pressure.",
]


# ────────────────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────────────────

def make_examples(records: list, num_per_question: int = 4) -> list:
    """
    Synthesise labelled examples from the question bank records.
    Returns list of (student_answer, reference_answer, topic, label_id)
    """
    examples = []
    refs = [r["reference_answer"] for r in records]

    for r in records:
        ref   = r["reference_answer"]
        topic = r.get("topic", "")

        # 1. Correct: student echoes the reference answer
        examples.append((ref, ref, topic, LABEL2ID["correct"]))

        # 2. Partially correct: first ~half of reference
        words = ref.split()
        half  = max(5, len(words) // 2)
        partial_ans = " ".join(words[:half])
        examples.append((partial_ans, ref, topic, LABEL2ID["partially_correct"]))

        # 3. Incorrect: random reference from a different question
        other_ref = random.choice(refs)
        while other_ref == ref and len(refs) > 1:
            other_ref = random.choice(refs)
        examples.append((other_ref, ref, topic, LABEL2ID["incorrect"]))

        # 4. Off-topic: completely unrelated sentence
        off = random.choice(OFF_TOPIC_SENTENCES)
        examples.append((off, ref, topic, LABEL2ID["off_topic"]))

    random.shuffle(examples)
    return examples


class ClassifierDataset(Dataset):
    def __init__(self, examples: list, tokenizer: Tokenizer, max_len: int = 256):
        self.examples  = examples
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        student_ans, ref_ans, topic, label = self.examples[idx]
        # Encode as pair: student_answer + reference_answer (topic appended)
        text_a = student_ans[:500]
        text_b = ref_ans[:300] + " " + topic
        enc = self.tokenizer.encode(text_a, text_b)
        ids = enc.ids[: self.max_len]
        # Pad to max_len
        pad = self.max_len - len(ids)
        ids = ids + [0] * pad
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


# ────────────────────────────────────────────────────────────
# LR scheduler
# ────────────────────────────────────────────────────────────

def get_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ────────────────────────────────────────────────────────────
# Training loop
# ────────────────────────────────────────────────────────────

def load_split(name: str) -> list:
    path = SPLITS_DIR / f"{name}.jsonl"
    records = []
    if not path.exists():
        print(f"  ⚠ {path} not found")
        return records
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def evaluate(model, loader, loss_fn, device) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for ids, labels in loader:
            ids, labels = ids.to(device), labels.to(device)
            logits = model(ids)
            loss   = loss_fn(logits, labels)
            total_loss += loss.item() * ids.size(0)
            preds      = logits.argmax(dim=-1)
            correct    += (preds == labels).sum().item()
            total      += ids.size(0)
    return total_loss / max(1, total), correct / max(1, total)


def train(args):
    random.seed(SEED)
    torch.manual_seed(SEED)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    if not TOK_PATH.exists():
        print(f"✗ Tokenizer not found at {TOK_PATH}. Run train_tokenizer.py first.")
        sys.exit(1)
    tokenizer = Tokenizer.from_file(str(TOK_PATH))
    vocab_size = tokenizer.get_vocab_size()
    print(f"  Tokenizer loaded. Vocab size: {vocab_size:,}")

    # Load data
    train_records = load_split("train")
    val_records   = load_split("val")
    print(f"  Train questions: {len(train_records):,} | Val questions: {len(val_records):,}")

    # Synthesise examples
    train_ex = make_examples(train_records)
    val_ex   = make_examples(val_records)
    print(f"  Synthesised {len(train_ex):,} train examples, {len(val_ex):,} val examples")

    train_ds = ClassifierDataset(train_ex, tokenizer)
    val_ds   = ClassifierDataset(val_ex,   tokenizer)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    model = AnswerClassifier(vocab_size=vocab_size).to(DEVICE)
    print(f"  Model: {model.num_params:,} parameters | Device: {DEVICE}")

    # Loss with class weighting (off_topic and partial are harder)
    weights = torch.tensor([1.0, 1.5, 1.0, 1.5]).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps  = len(train_loader) * args.epochs
    warmup_steps = max(1, total_steps // 10)
    scheduler    = get_scheduler(optimizer, warmup_steps, total_steps)

    best_val_loss = float("inf")
    best_path     = SAVE_DIR / "best_classifier.pt"

    print(f"\n{'Epoch':>6} {'Train Loss':>12} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>10}")
    print("-" * 55)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for ids, labels in train_loader:
            ids, labels = ids.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(ids)
            loss   = loss_fn(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * ids.size(0)
            preds       = logits.argmax(dim=-1)
            correct    += (preds == labels).sum().item()
            total      += ids.size(0)

        train_loss = total_loss / max(1, total)
        train_acc  = correct    / max(1, total)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, DEVICE)

        print(f"{epoch:>6}  {train_loss:>12.4f}  {train_acc:>10.4f}  {val_loss:>10.4f}  {val_acc:>10.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "vocab_size":  vocab_size,
                "val_loss":    val_loss,
                "val_acc":     val_acc,
            }, best_path)
            print(f"         ↑ Best model saved (val_loss={val_loss:.4f})")

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: {best_path}")


# ────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train AnswerClassifier")
    p.add_argument("--epochs",     type=int,   default=20)
    p.add_argument("--batch-size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=2e-4)
    return p.parse_args()


if __name__ == "__main__":
    print("=" * 60)
    print("AI Interview Prep — Classifier Training")
    print("=" * 60)
    args = parse_args()
    print(f"Config: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    train(args)
