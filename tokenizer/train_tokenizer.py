"""
tokenizer/train_tokenizer.py
============================
Trains a BPE tokenizer from scratch on the cleaned conversational corpus.

Input:  data/processed/dialogues.jsonl
Output: tokenizer/saved/  (vocab.json, merges.txt, tokenizer.json)

Usage:
    python tokenizer/train_tokenizer.py
"""

import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import Lowercase, Sequence as NormSequence, Strip
from tokenizers.processors import TemplateProcessing

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
SAVE_DIR      = Path(__file__).resolve().parent / "saved"

VOCAB_SIZE   = 8_000
MIN_FREQ     = 2
SPECIAL_TOKS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<|user|>", "<|ai|>", "<|end|>"]

PAD_ID = 0
UNK_ID = 1
CLS_ID = 2
SEP_ID = 3


# ────────────────────────────────────────────────────────────
# Corpus iterator
# ────────────────────────────────────────────────────────────

def iter_corpus(jsonl_path: Path):
    """Yield all text fields from the processed dataset."""
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            yield r.get("text", "")


def build_corpus_file(jsonl_path: Path, corpus_path: Path):
    """Write corpus to a temp text file for tokenizer training."""
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    with open(corpus_path, "w", encoding="utf-8") as out:
        for text in iter_corpus(jsonl_path):
            out.write(text + "\n")
    lines = corpus_path.read_text(encoding="utf-8").count("\n")
    print(f"  + Corpus: {lines:,} lines -> {corpus_path}")
    return corpus_path


# ────────────────────────────────────────────────────────────
# Tokenizer training
# ────────────────────────────────────────────────────────────

def train(corpus_path: Path) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    # Normaliser: lowercase + strip whitespace
    tokenizer.normalizer = NormSequence([Strip(), Lowercase()])

    # Pre-tokenise on whitespace
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQ,
        special_tokens=SPECIAL_TOKS,
        show_progress=True,
    )

    print(f"  Training BPE tokenizer (vocab_size={VOCAB_SIZE})...")
    tokenizer.train(files=[str(corpus_path)], trainer=trainer)

    # Post-processing: automatically add [CLS] at start and [SEP] at end
    tokenizer.post_processor = TemplateProcessing(
        single=f"[CLS]:0 $A:0 [SEP]:0",
        pair=f"[CLS]:0 $A:0 [SEP]:0 $B:0 [SEP]:0",
        special_tokens=[("[CLS]", CLS_ID), ("[SEP]", SEP_ID)],
    )

    # Enable padding
    tokenizer.enable_padding(pad_id=PAD_ID, pad_token="[PAD]")
    tokenizer.enable_truncation(max_length=512)

    return tokenizer


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("AI Interview Prep — Tokenizer Training")
    print("=" * 60)

    jsonl_path   = PROCESSED_DIR / "dialogues.jsonl"
    corpus_path  = PROCESSED_DIR / "corpus.txt"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    if not jsonl_path.exists():
        print(f"X {jsonl_path} not found. Run clean_data.py first.")
        return

    print("\n[1/3] Building corpus text file...")
    build_corpus_file(jsonl_path, corpus_path)

    print("\n[2/3] Training BPE tokenizer...")
    tok = train(corpus_path)

    print("\n[3/3] Saving tokenizer...")
    save_path = SAVE_DIR / "tokenizer.json"
    tok.save(str(save_path))
    print(f"  + Tokenizer saved to {save_path}")
    print(f"  Vocab size: {tok.get_vocab_size():,}")

    # Quick sanity check
    enc = tok.encode("<|user|> What is a binary search tree? <|end|>")
    print(f"\nSanity check:")
    print(f"  Input : '<|user|> What is a binary search tree? <|end|>'")
    print(f"  Tokens: {enc.tokens[:10]} ...")
    print(f"  IDs   : {enc.ids[:10]} ...")

    print("\n" + "=" * 60)
    print("Tokenizer training complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
