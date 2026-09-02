"""
models/ranker/train.py
======================
Trains the QuestionRanker bi-encoder using InfoNCE contrastive loss.

Synthetic training data construction:
  For each question in the training bank:
    - Positive doc:  the question itself (question + subtopic + difficulty)
    - Query:         a synthetically generated state descriptor, e.g.
                     "topic=OS; subtopic=threads; last_result=correct; difficulty=medium"
    - Negatives:     in-batch negatives (all other docs in the same batch)

  Multiple queries per question simulate different arrival contexts.

Usage:
    python models/ranker/train.py [--epochs 15] [--batch-size 32]
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from models.ranker.model import QuestionRanker, InfoNCELoss
from tokenizers import Tokenizer

SPLITS_DIR = ROOT / "data" / "splits"
TOK_PATH   = ROOT / "tokenizer" / "saved" / "tokenizer.json"
SAVE_DIR   = ROOT / "models" / "ranker" / "saved"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED   = 42

LAST_RESULTS = ["correct", "partially_correct", "incorrect", "off_topic"]
DIFFICULTIES = ["easy", "medium", "hard"]


# ────────────────────────────────────────────────────────────
# Synthetic query generation
# ────────────────────────────────────────────────────────────

def make_state_descriptor(record: dict, last_result: str = None, target_diff: str = None) -> str:
    """
    Generate a natural state descriptor string for a question.
    This is the 'query' in the retrieval task.
    """
    topic   = record.get("topic", "")
    sub     = record.get("subtopic", "")
    diff    = target_diff or record.get("difficulty", "medium")
    lr      = last_result or random.choice(LAST_RESULTS)
    return f"topic={topic}; subtopic={sub}; last_result={lr}; difficulty={diff}"


def make_doc_text(record: dict) -> str:
    """Encode a question bank entry as a document string."""
    return (
        f"{record.get('question', '')[:300]} "
        f"[{record.get('topic','')}] "
        f"[{record.get('subtopic','')}] "
        f"[{record.get('difficulty','')}]"
    )


def build_pairs(records: list, queries_per_record: int = 3) -> list[tuple[str, str]]:
    """
    Returns list of (query_text, doc_text) positive pairs.
    """
    pairs = []
    for r in records:
        doc = make_doc_text(r)
        for lr in random.sample(LAST_RESULTS, min(queries_per_record, len(LAST_RESULTS))):
            query = make_state_descriptor(r, last_result=lr)
            pairs.append((query, doc))
    random.shuffle(pairs)
    return pairs


# ────────────────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────────────────

class RankerDataset(Dataset):
    def __init__(self, pairs: list[tuple[str, str]], tokenizer: Tokenizer, max_len: int = 128):
        self.pairs     = pairs
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.pairs)

    def _encode(self, text: str) -> torch.Tensor:
        enc = self.tokenizer.encode(text)
        ids = enc.ids[: self.max_len]
        ids += [0] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        q_text, d_text = self.pairs[idx]
        return self._encode(q_text), self._encode(d_text)


# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────

def load_split(name: str) -> list:
    path = SPLITS_DIR / f"{name}.jsonl"
    records = []
    if not path.exists():
        return records
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ────────────────────────────────────────────────────────────
# Evaluate: Precision@k on val pairs
# ────────────────────────────────────────────────────────────

def eval_recall_at_k(model, val_pairs: list, tokenizer: Tokenizer, k: int = 5, max_len: int = 128) -> float:
    """
    Approximate recall@k: for each val query, check if its positive doc
    appears in the top-k retrieved documents (from the val pool).
    """
    model.eval()

    def encode_batch(texts, batch=64):
        all_embs = []
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            ids = []
            for t in chunk:
                enc = tokenizer.encode(t)
                seq = enc.ids[:max_len] + [0] * (max_len - len(enc.ids[:max_len]))
                ids.append(seq)
            t = torch.tensor(ids, dtype=torch.long).to(DEVICE)
            with torch.no_grad():
                emb = model.encode_query(t)
            all_embs.append(emb.cpu())
        return torch.cat(all_embs, dim=0)

    q_texts = [p[0] for p in val_pairs]
    d_texts = [p[1] for p in val_pairs]

    q_embs = encode_batch(q_texts)
    d_embs = encode_batch(d_texts)

    sim = torch.matmul(q_embs, d_embs.T)   # (N, N)
    hits = 0
    for i in range(len(val_pairs)):
        topk = sim[i].topk(k).indices.tolist()
        if i in topk:
            hits += 1
    return hits / max(1, len(val_pairs))


# ────────────────────────────────────────────────────────────
# Training
# ────────────────────────────────────────────────────────────

def train(args):
    random.seed(SEED)
    torch.manual_seed(SEED)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    if not TOK_PATH.exists():
        print(f"✗ Tokenizer not found: {TOK_PATH}. Run train_tokenizer.py first.")
        sys.exit(1)
    tokenizer  = Tokenizer.from_file(str(TOK_PATH))
    vocab_size = tokenizer.get_vocab_size()
    print(f"  Tokenizer loaded. Vocab size: {vocab_size:,}")

    train_records = load_split("train")
    val_records   = load_split("val")
    print(f"  Train questions: {len(train_records):,} | Val: {len(val_records):,}")

    train_pairs = build_pairs(train_records, queries_per_record=3)
    val_pairs   = build_pairs(val_records,   queries_per_record=1)
    print(f"  Train pairs: {len(train_pairs):,} | Val pairs: {len(val_pairs):,}")

    train_ds = RankerDataset(train_pairs, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model   = QuestionRanker(vocab_size=vocab_size).to(DEVICE)
    loss_fn = InfoNCELoss(temperature=0.07)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    total_steps  = len(train_loader) * args.epochs
    warmup_steps = max(1, total_steps // 10)
    scheduler    = get_scheduler(optimizer, warmup_steps, total_steps)

    best_recall = 0.0
    best_path   = SAVE_DIR / "best_ranker.pt"

    print(f"\n{'Epoch':>6} {'Train Loss':>12} {'Val Recall@5':>14}")
    print("-" * 35)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total = 0.0, 0

        for q_ids, d_ids in train_loader:
            q_ids = q_ids.to(DEVICE)
            d_ids = d_ids.to(DEVICE)

            optimizer.zero_grad()
            q_emb = model.encode_query(q_ids)
            d_emb = model.encode_doc(d_ids)
            loss  = loss_fn(q_emb, d_emb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * q_ids.size(0)
            total      += q_ids.size(0)

        avg_loss = total_loss / max(1, total)
        recall   = eval_recall_at_k(model, val_pairs[:200], tokenizer, k=5)

        print(f"{epoch:>6}  {avg_loss:>12.4f}  {recall:>14.4f}")

        if recall > best_recall:
            best_recall = recall
            torch.save({
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "vocab_size":  vocab_size,
                "val_recall":  recall,
            }, best_path)
            print(f"         ↑ Best model saved (recall@5={recall:.4f})")

    print(f"\nTraining complete. Best val recall@5: {best_recall:.4f}")
    print(f"Best checkpoint: {best_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Train QuestionRanker")
    p.add_argument("--epochs",     type=int,   default=15)
    p.add_argument("--batch-size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=2e-4)
    return p.parse_args()


if __name__ == "__main__":
    print("=" * 60)
    print("AI Interview Prep — Ranker Training")
    print("=" * 60)
    args = parse_args()
    print(f"Config: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    train(args)
