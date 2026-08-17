"""
models/generator/train.py
==========================
UNIFIED training script for ALL curriculum stages.

Runs the same on:
  - Local laptop  (GTX 1650 4GB  → small model, batch 2, accum 8)
  - Kaggle        (T4/P100 16GB  → medium model, batch 16, accum 4)
  - Colab         (T4 16GB       → medium model)

Usage:
    python models/generator/train.py --stage pretrain     # Stage 1
    python models/generator/train.py --stage domain       # Stage 2
    python models/generator/train.py --stage instruction  # Stage 3
    python models/generator/train.py --stage interview    # Stage 4
    python models/generator/train.py --stage evaluator    # Stage 5
    python models/generator/train.py --stage followup     # Stage 6
    python models/generator/train.py --stage all          # Run all in order

Every stage:
  - Auto-detects environment & GPU (env_config)
  - Saves checkpoints with resume support (epoch/step/optimizer/scheduler)
  - Uses FP16 + gradient accumulation
  - Validates on a holdout split
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader, random_split

from env_config import (
    ENV, ROOT as ENV_ROOT, DATA_DIR, SAVE_ROOT, DEVICE, BASE_BATCH_SIZE,
    GRAD_ACCUM_STEPS, USE_FP16, VOCAB_SIZE, MODEL_SIZE, print_env_summary,
)
from models.generator.model import create_small_model, create_medium_model, create_large_model
from models.generator.train_utils import (
    TextDataset, ChatDataset, get_cosine_schedule_with_warmup,
    train_epoch, evaluate, save_checkpoint, load_checkpoint,
    load_tokenizer, make_training_configs,
)

# ─────────────────────────────────────────────────────────────
# Stage definitions: which data file feeds each stage
# ─────────────────────────────────────────────────────────────

# Data file selection per stage. Paths are relative to ROOT.
STAGE_DATA = {
    "pretrain": [
        "data/raw/starcoder_large.jsonl",
        "data/raw/codesearchnet.jsonl",
        "data/raw/cruxeval/cruxeval.jsonl",
    ],
    "domain": [
        "data/raw/conversations.jsonl",
        "data/raw/opencodeinstruct.jsonl",
        "data/raw/codefeedback.jsonl",
        "data/raw/oasst_coding.jsonl",
        "data/processed/dialogues.jsonl",
    ],
    "instruction": [
        "data/raw/codealpaca.jsonl",
        "data/raw/opencodeinstruct.jsonl",
    ],
    "interview": [
        "data/raw/conversations.jsonl",
        "data/raw/opencodeinstruct.jsonl",
    ],
    "evaluator": [
        "data/raw/mohler_asag.jsonl",
        "data/raw/codealpaca.jsonl",
    ],
    "followup": [
        "data/raw/oasst_coding.jsonl",
        "data/raw/conversations.jsonl",
    ],
}

# Where the checkpoint goes for each stage (relative to SAVE_ROOT)
STAGE_CKPT = {
    "pretrain": "pretrained.pt",
    "domain": "domain_tuned.pt",
    "instruction": "instruction_tuned.pt",
    "interview": "interview_tuned.pt",
    "evaluator": "evaluator.pt",
    "followup": "final_model.pt",
}

# Which checkpoint each stage initializes FROM (None = from scratch / random)
STAGE_INIT = {
    "pretrain": None,          # starts from scratch
    "domain": "pretrained.pt",
    "instruction": "domain_tuned.pt",
    "interview": "domain_tuned.pt",     # instruction not strictly required before interview
    "evaluator": "interview_tuned.pt",
    "followup": "evaluator.pt",
}

# Which model factory to use
def _factory():
    if MODEL_SIZE == "large":
        return create_large_model
    if MODEL_SIZE == "medium":
        return create_medium_model
    return create_small_model


# ─────────────────────────────────────────────────────────────
# Data resolution
# ─────────────────────────────────────────────────────────────

def resolve_data_path(rel: str):
    """Resolve a stage data file. Local: <ROOT>/<rel>. Kaggle: /kaggle/input/<rel>."""
    candidates = []
    if ENV == "kaggle":
        # Datasets uploaded to Kaggle appear under /kaggle/input/<dataset-name>/<file>
        # Try direct path first, then scan /kaggle/input for the basename.
        candidates.append(Path("/kaggle/input") / rel)
    candidates.append(ENV_ROOT / rel)
    # Scan /kaggle/input recursively for the basename as last resort
    if ENV == "kaggle":
        base = Path(rel).name
        for d in Path("/kaggle/input").rglob(base):
            candidates.append(d)
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def build_dataset_for_stage(stage, tokenizer, config):
    """Build train + val datasets for the stage."""
    global _LIMIT
    files = STAGE_DATA[stage]
    max_len = config["max_len"]
    examples = []

    for rel in files:
        path = resolve_data_path(rel)
        if not path.exists():
            print(f"  [skip] missing {rel} ({path})")
            continue
        # All our training files are JSONL. Use ChatDataset (handles messages + text).
        try:
            ds = ChatDataset(str(path), tokenizer, max_len=max_len, limit=_LIMIT)
            print(f"  + {path.name}: {len(ds)} examples")
            examples.extend(ds.examples)
        except Exception as e:
            print(f"  [skip] error loading {path.name}: {e}")

    if not examples:
        raise RuntimeError(
            f"No usable data found for stage '{stage}'. Check files in {DATA_DIR}"
        )

    # Deduplicate (exact duplicates waste compute)
    seen = set()
    uniq = []
    for m in examples:
        key = json.dumps(m, ensure_ascii=False)[:200]
        if key not in seen:
            seen.add(key)
            uniq.append(m)
    print(f"  Deduplicated: {len(examples)} -> {len(uniq)}")

    # Build a wrapper dataset from the raw message lists
    class _Wrapper(torch.utils.data.Dataset):
        def __init__(self, msgs):
            self.msgs = msgs
        def __len__(self):
            return len(self.msgs)
        def __getitem__(self, i):
            parts = []
            for msg in self.msgs[i]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                parts.append(f"<|{role}|> {content} <|end|>")
            full = "\n".join(parts)
            enc = tokenizer.encode(full)
            ids = enc.ids[:max_len]
            pad_id = tokenizer.token_to_id("[PAD]") or 0
            if len(ids) < max_len:
                ids = ids + [pad_id] * (max_len - len(ids))
            am = [1 if t != pad_id else 0 for t in ids]
            return {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(am, dtype=torch.long),
                "labels": torch.tensor(ids, dtype=torch.long),
            }

    full = _Wrapper(uniq)
    val_size = min(500, max(1, len(full) // 30))
    train_size = len(full) - val_size
    train_ds, val_ds = random_split(full, [train_size, val_size])
    return train_ds, val_ds


# ─────────────────────────────────────────────────────────────
# Model loading (with curriculum init + resume)
# ─────────────────────────────────────────────────────────────

def load_model_for_stage(stage, vocab_size):
    """Create model, initialize from a prior checkpoint if the stage requires it."""
    factory = _factory()
    model = factory(vocab_size=vocab_size)

    init_ckpt = STAGE_INIT[stage]
    if init_ckpt:
        init_path = SAVE_ROOT / init_ckpt
        if init_path.exists():
            print(f"  Init weights from: {init_path}")
            ckpt = torch.load(init_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            print(f"  WARN: init checkpoint {init_path} missing — starting fresh.")
    return model


# ─────────────────────────────────────────────────────────────
# Single stage runner
# ─────────────────────────────────────────────────────────────

def run_stage(stage, config):
    print(f"\n{'='*60}")
    print(f"  STAGE: {stage.upper()}")
    print(f"{'='*60}")

    t0 = time.time()

    # Tokenizer
    tok_path = ENV_ROOT / "tokenizer" / "saved" / "tokenizer.json"
    if not tok_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {tok_path}. Train it first:\n"
            "  python tokenizer/train_tokenizer.py"
        )
    tokenizer = load_tokenizer(tok_path)
    vocab_size = tokenizer.get_vocab_size()
    print(f"Tokenizer: vocab={vocab_size}")

    # Data
    train_ds, val_ds = build_dataset_for_stage(stage, tokenizer, config)
    bs = config["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Batch: {bs}")

    # Model
    model = load_model_for_stage(stage, vocab_size)
    model = model.to(DEVICE)
    print(f"Model params: {model.num_params_millions:.1f}M")

    # Optimizer + scheduler
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        betas=config["betas"],
        weight_decay=config["weight_decay"],
    )
    total_steps = len(train_loader) // config["grad_accum_steps"] * config["epochs"]
    warmup = max(1, int(total_steps * config["warmup_ratio"]))
    sched = get_cosine_schedule_with_warmup(opt, warmup, total_steps)

    # Checkpoint / resume
    ckpt_path = SAVE_ROOT / STAGE_CKPT[stage]
    start_epoch = 0
    best_val_loss = float("inf")
    if ckpt_path.exists():
        ep, loss, step = load_checkpoint(ckpt_path, model, opt, sched)
        start_epoch = ep + 1
        best_val_loss = loss
        print(f"  Resuming from epoch {ep} (val_loss={loss:.4f})")

    # Training loop
    for epoch in range(start_epoch, config["epochs"]):
        print(f"\n--- Epoch {epoch+1}/{config['epochs']} ---")
        train_loss, steps = train_epoch(
            model, train_loader, opt, sched, DEVICE,
            grad_clip=config["grad_clip"],
            grad_accum_steps=config["grad_accum_steps"],
            use_fp16=USE_FP16,
        )
        val_loss, val_ppl = evaluate(model, val_loader, DEVICE, use_fp16=USE_FP16)
        print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  ppl={val_ppl:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, opt, sched, epoch, val_loss, ckpt_path, step=steps,
                            extra={"stage": stage, "best_val_loss": best_val_loss})

    print(f"\n  {stage} done in {(time.time()-t0)/60:.1f} min. Best val_loss={best_val_loss:.4f}")
    print(f"  Checkpoint: {ckpt_path}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="INTERVUE unified training")
    ap.add_argument("--stage", default="all",
                    choices=["all", "pretrain", "domain", "instruction",
                             "interview", "evaluator", "followup"])
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit examples per file (for smoke tests)")
    args = ap.parse_args()

    print_env_summary()

    configs = make_training_configs({
        "BASE_BATCH_SIZE": BASE_BATCH_SIZE,
        "GRAD_ACCUM_STEPS": GRAD_ACCUM_STEPS,
    })

    # Apply limit for smoke tests
    if args.limit:
        for k in configs:
            pass  # limit handled at dataset level — see below
        STAGE_LIMIT = args.limit
    else:
        STAGE_LIMIT = None

    stages = ["pretrain", "domain", "instruction", "interview", "evaluator", "followup"]
    if args.stage != "all":
        stages = [args.stage]

    # Wire the limit into the dataset builder via a module-level hook
    global _LIMIT
    _LIMIT = STAGE_LIMIT

    for stage in stages:
        try:
            run_stage(stage, configs[stage])
        except Exception as e:
            print(f"\n  X Stage '{stage}' failed: {e}")
            import traceback; traceback.print_exc()
            if args.stage != "all":
                raise
            print("  Continuing to next stage...")

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)


_LIMIT = None

if __name__ == "__main__":
    main()
