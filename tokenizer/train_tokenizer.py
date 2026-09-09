"""
tokenizer/train_tokenizer.py
=============================
Trains a BPE tokenizer from scratch on all coding/CS datasets.

Changes from V1:
  - Vocab size: 8K → 16K
  - Pre-tokenizer: Whitespace → ByteLevel (better for code)
  - Special tokens: added <|system|>, <|code|>, <|/code|>
  - Max length: 512 → 1024
  - Training corpus: all datasets combined (not just dialogues)

Input:  data/processed/ (multiple files)
Output: tokenizer/saved/tokenizer.json

Usage:
    python tokenizer/train_tokenizer.py
"""

import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import Lowercase, Sequence as NormSequence, Strip
from tokenizers.processors import TemplateProcessing
from tokenizers.implementations import BaseTokenizer

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DIR = ROOT / "data" / "raw"
SAVE_DIR = ROOT / "tokenizer" / "saved"

VOCAB_SIZE = 16_000
MIN_FREQ = 2
SPECIAL_TOKS = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
    "<|system|>",
    "<|code|>",
    "<|/code|>",
]

PAD_ID = 0
UNK_ID = 1
CLS_ID = 2
SEP_ID = 3


# ────────────────────────────────────────────────────────────
# Corpus iterators
# ────────────────────────────────────────────────────────────

def iter_all_texts():
    """Yield all text from all available datasets."""

    # 1. StandardLogic conversations
    conv_path = RAW_DIR / "conversations.jsonl"
    if conv_path.exists():
        print("  + StandardLogic conversations...")
        with open(conv_path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    for turn in rec.get("conversations", []):
                        text = turn.get("value", "").strip()
                        if text:
                            yield text
                except json.JSONDecodeError:
                    continue

    # 2. CodeAlpaca
    alpaca_path = RAW_DIR / "codealpaca.jsonl"
    if alpaca_path.exists():
        print("  + CodeAlpaca...")
        with open(alpaca_path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    for field in ["instruction", "input", "output"]:
                        text = rec.get(field, "").strip()
                        if text:
                            yield text
                except json.JSONDecodeError:
                    continue

    # 3. CodeFeedback
    feedback_path = RAW_DIR / "codefeedback.jsonl"
    if feedback_path.exists():
        print("  + CodeFeedback...")
        with open(feedback_path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    for field in ["instruction", "response"]:
                        text = rec.get(field, "").strip()
                        if text:
                            yield text
                except json.JSONDecodeError:
                    continue

    # 4. StarCoder samples (code)
    star_path = RAW_DIR / "starcoder_large.jsonl"
    if not star_path.exists():
        star_path = RAW_DIR / "starcoder_sample.jsonl"
    if star_path.exists():
        print(f"  + StarCoder code ({star_path.name})...")
        with open(star_path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    content = rec.get("content", "").strip()
                    if content:
                        yield content
                except json.JSONDecodeError:
                    continue

    # 4b. CodeSearchNet
    csn_path = RAW_DIR / "codesearchnet.jsonl"
    if csn_path.exists():
        print("  + CodeSearchNet...")
        with open(csn_path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    for field in ["code", "doc", "name"]:
                        text = rec.get(field, "").strip()
                        if text:
                            yield text
                except json.JSONDecodeError:
                    continue

    # 4c. OpenCodeInstruct
    oci_path = RAW_DIR / "opencodeinstruct.jsonl"
    if oci_path.exists():
        print("  + OpenCodeInstruct...")
        with open(oci_path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    for field in ["input", "output"]:
                        text = str(rec.get(field, "")).strip()
                        if text:
                            yield text
                except json.JSONDecodeError:
                    continue

    # 4d. OASST coding
    oasst_path = RAW_DIR / "oasst_coding.jsonl"
    if oasst_path.exists():
        print("  + OASST coding...")
        with open(oasst_path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    text = str(rec.get("text", "")).strip()
                    if text:
                        yield text
                except json.JSONDecodeError:
                    continue

    # 4e. Mohler ASAG (real student answers)
    mohler_path = RAW_DIR / "mohler_asag.jsonl"
    if mohler_path.exists():
        print("  + Mohler ASAG...")
        with open(mohler_path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    for field in ["question", "student_answer", "instructor_answer"]:
                        text = str(rec.get(field, "")).strip()
                        if text:
                            yield text
                except json.JSONDecodeError:
                    continue

    # 4f. CRUXEval
    crux_path = RAW_DIR / "cruxeval" / "cruxeval.jsonl"
    if not crux_path.exists():
        crux_path = RAW_DIR / "cruxeval.jsonl"
    if crux_path.exists():
        print(f"  + CRUXEval ({crux_path.name})...")
        with open(crux_path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    for field in ["code", "input", "output"]:
                        text = str(rec.get(field, "")).strip()
                        if text:
                            yield text
                except json.JSONDecodeError:
                    continue

    # 5. Processed dialogues
    dialogue_path = PROCESSED_DIR / "dialogues.jsonl"
    if dialogue_path.exists():
        print("  + Processed dialogues...")
        with open(dialogue_path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    text = rec.get("text", "").strip()
                    if text:
                        yield text
                except json.JSONDecodeError:
                    continue

    # 6. MMLU CS
    mmlu_path = ROOT / "data" / "raw" / "mmlu_cs.json"
    if mmlu_path.exists():
        print("  + MMLU CS...")
        with open(mmlu_path, encoding="utf-8") as f:
            try:
                data = json.load(f)
                for rec in data:
                    for field in ["question", "answer"]:
                        text = str(rec.get(field, "")).strip()
                        if text:
                            yield text
            except json.JSONDecodeError:
                pass


def build_corpus_file(corpus_path: Path):
    """Write all text to a single corpus file for tokenizer training."""
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(corpus_path, "w", encoding="utf-8") as out:
        for text in iter_all_texts():
            out.write(text + "\n")
            count += 1
    print(f"  Corpus: {count:,} lines -> {corpus_path}")
    return corpus_path


# ────────────────────────────────────────────────────────────
# Tokenizer training
# ────────────────────────────────────────────────────────────

def train(corpus_path: Path) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    # Normalizer: strip only (case-sensitive — preserves code semantics)
    tokenizer.normalizer = NormSequence([Strip()])

    # Pre-tokenizer: ByteLevel (handles code syntax, special chars)
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQ,
        special_tokens=SPECIAL_TOKS,
        # BpeTrainer reads files line-by-line, so '\n' never appears in the corpus
        # and Ċ would be missing from the vocab — every newline would encode as
        # [UNK]. Seed the full ByteLevel alphabet (all 256 bytes) to prevent this.
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )

    print(f"  Training BPE tokenizer (vocab_size={VOCAB_SIZE})...")
    tokenizer.train(files=[str(corpus_path)], trainer=trainer)

    # Post-processing: add [CLS] at start, [SEP] at end
    tokenizer.post_processor = TemplateProcessing(
        single=f"[CLS]:0 $A:0 [SEP]:0",
        pair=f"[CLS]:0 $A:0 [SEP]:0 $B:0 [SEP]:0",
        special_tokens=[("[CLS]", CLS_ID), ("[SEP]", SEP_ID)],
    )

    # Enable padding and truncation
    tokenizer.enable_padding(pad_id=PAD_ID, pad_token="[PAD]")
    tokenizer.enable_truncation(max_length=1024)

    return tokenizer


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("INTERVUE — Tokenizer Training (V2)")
    print("=" * 60)
    print(f"  Vocab size: {VOCAB_SIZE}")
    print(f"  Pre-tokenizer: ByteLevel")
    print(f"  Max length: 1024")

    corpus_path = PROCESSED_DIR / "tokenizer_corpus.txt"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] Building corpus from all datasets...")
    build_corpus_file(corpus_path)

    print("\n[2/3] Training BPE tokenizer...")
    tok = train(corpus_path)

    print("\n[3/3] Saving tokenizer...")
    save_path = SAVE_DIR / "tokenizer.json"
    tok.save(str(save_path))
    print(f"  Saved to {save_path}")
    print(f"  Vocab size: {tok.get_vocab_size():,}")

    # Sanity checks
    print("\n" + "=" * 60)
    print("Sanity Checks:")
    print("=" * 60)

    test_cases = [
        "<|system|> You are an expert technical interviewer.<|end|>",
        "<|user|> What is a binary search tree?<|end|>",
        "<|assistant|> A binary search tree is a binary tree where...",
        "def binary_search(arr, target):\n    low, high = 0, len(arr) - 1",
        "Time complexity: O(log n), Space complexity: O(1)",
    ]

    for text in test_cases:
        enc = tok.encode(text)
        print(f"\n  Input: {text[:60]}...")
        print(f"  Tokens: {enc.tokens[:12]}...")
        print(f"  IDs: {enc.ids[:12]}...")

    print("\n" + "=" * 60)
    print("Tokenizer training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
