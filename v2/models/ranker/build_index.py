"""
models/ranker/build_index.py
============================
Precomputes embeddings for every question in the question bank
and saves them as a numpy matrix for fast cosine-similarity lookup at runtime.

Output:
    models/ranker/saved/question_embeddings.npy   — shape (N, embed_dim)
    models/ranker/saved/question_ids.json         — ordered list of question IDs

Usage:
    python models/ranker/build_index.py
"""

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from models.ranker.model import QuestionRanker
from tokenizers import Tokenizer

DB_PATH   = ROOT / "data" / "question_bank.db"
TOK_PATH  = ROOT / "tokenizer" / "saved" / "tokenizer.json"
CKPT_PATH = ROOT / "models" / "ranker" / "saved" / "best_ranker.pt"
SAVE_DIR  = ROOT / "models" / "ranker" / "saved"

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
BATCH    = 64
MAX_LEN  = 128


def load_all_questions() -> list[dict]:
    if not DB_PATH.exists():
        print(f"✗ Question bank not found: {DB_PATH}")
        sys.exit(1)
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("SELECT id, topic, subtopic, difficulty, question FROM questions")
    rows = cur.fetchall()
    conn.close()
    return [
        {"id": r[0], "topic": r[1], "subtopic": r[2], "difficulty": r[3], "question": r[4]}
        for r in rows
    ]


def make_doc_text(r: dict) -> str:
    return (
        f"{r['question'][:300]} "
        f"[{r['topic']}] [{r['subtopic']}] [{r['difficulty']}]"
    )


def encode_batch(model: QuestionRanker, texts: list[str], tokenizer: Tokenizer) -> np.ndarray:
    all_embs = []
    for i in range(0, len(texts), BATCH):
        chunk = texts[i: i + BATCH]
        ids = []
        for t in chunk:
            enc = tokenizer.encode(t)
            seq = enc.ids[:MAX_LEN] + [0] * (MAX_LEN - len(enc.ids[:MAX_LEN]))
            ids.append(seq)
        t_ids = torch.tensor(ids, dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            emb = model.encode_doc(t_ids)  # (B, D)
        all_embs.append(emb.cpu().numpy())
        if (i // BATCH + 1) % 10 == 0:
            print(f"  Encoded {i + len(chunk):,} / {len(texts):,} documents...")
    return np.vstack(all_embs)


def build_index():
    print("=" * 60)
    print("AI Interview Prep — Building Question Embedding Index")
    print("=" * 60)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    if not TOK_PATH.exists():
        print(f"✗ Tokenizer not found: {TOK_PATH}")
        sys.exit(1)
    tokenizer = Tokenizer.from_file(str(TOK_PATH))

    # Load ranker checkpoint
    if not CKPT_PATH.exists():
        print(f"✗ Ranker checkpoint not found: {CKPT_PATH}. Run ranker/train.py first.")
        sys.exit(1)
    ckpt  = torch.load(CKPT_PATH, map_location=DEVICE)
    model = QuestionRanker(vocab_size=ckpt["vocab_size"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"✓ Loaded ranker from epoch {ckpt['epoch']} (val_recall@5={ckpt['val_recall']:.4f})")

    # Load questions
    questions = load_all_questions()
    print(f"✓ Loaded {len(questions):,} questions from question bank")

    # Encode
    texts = [make_doc_text(q) for q in questions]
    print(f"\nEncoding {len(texts):,} documents (batch={BATCH})...")
    embeddings = encode_batch(model, texts, tokenizer)

    # Save
    ids_path  = SAVE_DIR / "question_ids.json"
    emb_path  = SAVE_DIR / "question_embeddings.npy"

    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump([q["id"] for q in questions], f)

    np.save(str(emb_path), embeddings)

    print(f"\n✓ Embeddings saved: {emb_path}  shape={embeddings.shape}")
    print(f"✓ Question IDs saved: {ids_path}")
    print("=" * 60)
    print("Index built successfully. Ready for inference.")
    print("=" * 60)


if __name__ == "__main__":
    build_index()
