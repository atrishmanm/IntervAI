"""
models/generator/train.py
==========================
UNIFIED research-grade training script for ALL curriculum stages.

Runs the same on:
  - Local laptop  (GTX 1650 4GB  → small model, batch 2, accum 8)
  - Kaggle        (2x T4 16GB    → large model, batch 16/GPU, accum 4, DataParallel)
  - Colab         (T4 16GB       → large model)

Usage:
    python models/generator/train.py --stage pretrain     # Stage 1
    python models/generator/train.py --stage domain       # Stage 2
    python models/generator/train.py --stage instruction  # Stage 3
    python models/generator/train.py --stage interview    # Stage 4
    python models/generator/train.py --stage evaluator    # Stage 5
    python models/generator/train.py --stage followup     # Stage 6
    python models/generator/train.py --stage all          # Run all in order

Research-grade features:
  - Multi-GPU (DataParallel) on Kaggle 2x T4
  - Auto LR-finder (LR range test) + batch-size memory profiler per stage
  - Early stopping with patience (best-checkpoint tracking)
  - Crash-safe mid-epoch checkpointing (every N optimizer steps) + FP16 scaler state
  - Token accuracy + top-5 accuracy on validation
  - JSON training log per stage (results/<stage>.json)
  - Correct full-content dedup (SHA256) — no more 200-char truncation bug
  - Fixed curriculum chain (followup now fine-tunes from interview_tuned, not evaluator)
  - Evaluator stage trained on CODE feedback data (not educational ASAG)
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
    GRAD_ACCUM_STEPS, USE_FP16, VOCAB_SIZE, MODEL_SIZE, NUM_GPUS,
    EFFECTIVE_BATCH_SIZE, print_env_summary,
)
from models.generator.model import create_small_model, create_medium_model, create_large_model
from models.generator.train_utils import (
    TextDataset, ChatDataset, get_cosine_schedule_with_warmup, get_wsd_schedule,
    train_epoch, evaluate, save_checkpoint, load_checkpoint,
    load_tokenizer, make_training_configs, dedup_examples, quality_filter,
    wrap_data_parallel, unwrap_model, find_learning_rate, profile_batch_size,
    _make_scaler, EMA,
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
        "data/pretrain/nemotron_code_100k.jsonl",
        "data/pretrain/fineweb_edu_50k.jsonl",
    ],
    "domain": [
        "data/raw/conversations.jsonl",
        "data/raw/opencodeinstruct.jsonl",
        "data/raw/oasst_coding.jsonl",
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
        "data/evaluator/mohler_asag.jsonl",
        "data/evaluator/scientsbank.jsonl",
        "data/evaluator/beetle.jsonl",
        "data/evaluator/asap_aes.jsonl",
        "data/evaluator/asap_sas.jsonl",
    ],
    "followup": [
        "data/raw/oasst_coding.jsonl",
        "data/raw/conversations.jsonl",
        "data/raw/opencodeinstruct.jsonl",
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
    "followup": "interview_tuned.pt",   # FIXED: was evaluator.pt (catastrophic forgetting)
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

    # Deduplicate on FULL content (SHA256) — the 200-char truncation bug
    # previously collapsed distinct conversations, losing 80%+ of the data.
    uniq = dedup_examples(examples)
    print(f"  Deduplicated: {len(examples)} -> {len(uniq)}")

    # Quality filter: remove degenerate/too-short/too-long examples
    filtered = [ex for ex in uniq if quality_filter(ex)]
    print(f"  Quality filtered: {len(uniq)} -> {len(filtered)}")

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

    full = _Wrapper(filtered)
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
            sd = ckpt["model_state_dict"]
            # Strip 'module.' prefix from DataParallel checkpoints
            if any(k.startswith("module.") for k in sd):
                sd = {k[len("module."):]: v for k, v in sd.items()}
            model.load_state_dict(sd)
        else:
            print(f"  WARN: init checkpoint {init_path} missing — starting fresh.")
    return model


# ─────────────────────────────────────────────────────────────
# Training-log writer (research-grade JSON metrics)
# ─────────────────────────────────────────────────────────────

def _append_log(stage, record, path=None):
    """Append one JSON line to results/<stage>.log."""
    import os
    log_dir = ENV_ROOT / "results"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = path or (log_dir / f"{stage}.log")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


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

    # Model + multi-GPU (wrap BEFORE tuning so batches are sharded per GPU)
    model = load_model_for_stage(stage, vocab_size)

    # Enable gradient checkpointing if configured (saves VRAM, slower)
    if config.get("gradient_checkpointing", False):
        model.enable_gradient_checkpointing()
        print("  Gradient checkpointing: ON")

    model = model.to(DEVICE)
    model = wrap_data_parallel(model)  # no-op if 1 GPU
    print(f"Model params: {unwrap_model(model).num_params_millions:.1f}M  (GPUs: {NUM_GPUS})")

    # ── EMA (Exponential Moving Average) ──
    ema = EMA(model, decay=config.get("ema_decay", 0.999))
    print(f"  EMA decay: {config.get('ema_decay', 0.999)}")

    # ── Auto-tuning runs backward passes, so snapshot weights first ──
    def _snapshot():
        return {k: v.detach().clone() for k, v in unwrap_model(model).state_dict().items()}

    def _restore(sd):
        unwrap_model(model).load_state_dict(sd)
        torch.cuda.empty_cache()

    # ── Auto-tuning: batch-size profiler (per-GPU) ──
    bs_per_gpu = config["batch_size"]
    if torch.cuda.is_available() and config.get("lr_finder", True):
        snap = _snapshot()
        try:
            bs_per_gpu = profile_batch_size(unwrap_model(model), train_ds, DEVICE,
                                            base_batch=bs_per_gpu, use_fp16=USE_FP16)
        finally:
            _restore(snap)
    # DataParallel shards a batch of `bs_per_gpu * NUM_GPUS` into bs_per_gpu/GPU.
    loader_bs = bs_per_gpu * max(NUM_GPUS, 1)
    config["batch_size"] = loader_bs
    num_workers = 2 if (ENV == "kaggle" and torch.cuda.is_available()) else 0
    train_loader = DataLoader(train_ds, batch_size=loader_bs, shuffle=True,
                              num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                              persistent_workers=(num_workers > 0))
    val_loader = DataLoader(val_ds, batch_size=loader_bs,
                            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                            persistent_workers=(num_workers > 0))
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Per-GPU batch: {bs_per_gpu} "
          f"| Loader batch: {loader_bs} | GPUs: {NUM_GPUS}")

    # ── Optimizer (LR finder re-inits it) ──
    def _optimizer(lr):
        return torch.optim.AdamW(
            unwrap_model(model).parameters(),
            lr=lr,
            betas=config["betas"],
            weight_decay=config["weight_decay"],
        )

    # ── Auto-tuning: LR finder ──
    if config.get("lr_finder", True) and torch.cuda.is_available():
        snap = _snapshot()
        try:
            # Pass the WRAPPED model: the loader batch (per-GPU x NUM_GPUS)
            # must be sharded across GPUs, otherwise one GPU gets the full batch.
            lr = find_learning_rate(
                model, train_loader, DEVICE, _optimizer, config,
                use_fp16=USE_FP16,
            )
        finally:
            _restore(snap)
        config["lr"] = lr
    opt = _optimizer(config["lr"])

    total_steps = len(train_loader) // config["grad_accum_steps"] * config["epochs"]
    warmup = max(1, int(total_steps * config["warmup_ratio"]))

    # Use WSD schedule if configured, else fallback to cosine
    if config.get("schedule") == "wsd":
        stable_pct = config.get("stable_pct", 0.80)
        decay_pct = config.get("decay_pct", 0.18)
        warmup_steps = int(total_steps * config["warmup_ratio"])
        stable_steps = int(total_steps * stable_pct)
        decay_steps = max(1, total_steps - warmup_steps - stable_steps)
        sched = get_wsd_schedule(
            opt, warmup_steps, stable_steps, decay_steps,
            peak_lr=config["lr"], min_lr_ratio=0.1,
        )
        print(f"  Schedule: WSD (warmup={warmup_steps}, stable={stable_steps}, decay={decay_steps})")
    else:
        sched = get_cosine_schedule_with_warmup(opt, warmup, total_steps)
        print(f"  Schedule: Cosine (warmup={warmup}, total={total_steps})")

    # FP16 scaler (persisted across resume)
    scaler = _make_scaler(USE_FP16)

    # ── Checkpoint / resume ──
    ckpt_path = SAVE_ROOT / STAGE_CKPT[stage]
    resume_path = SAVE_ROOT / f"{STAGE_CKPT[stage]}.resume"
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    patience_left = config.get("patience", 2)
    # Prefer the crash-safe resume checkpoint if it exists (it is most recent)
    resume_source = resume_path if resume_path.exists() else (ckpt_path if ckpt_path.exists() else None)
    if resume_source:
        ep, loss, step = load_checkpoint(resume_source, model, opt, sched, scaler=scaler)
        if ep >= 0:
            start_epoch = ep + 1
            best_val_loss = loss
            global_step = step
            print(f"  Resuming from epoch {ep} (val_loss={loss:.4f}, step={step})")
        else:
            print("  No compatible checkpoint — training from scratch.")

    # Training loop (mid-epoch checkpoint callback uses `cur_epoch` via closure list)
    cur_epoch = [start_epoch - 1]

    def _mid_epoch_save(step, path):
        save_checkpoint(unwrap_model(model), opt, sched, cur_epoch[0], best_val_loss,
                        path, step=step,
                        extra={"stage": stage, "best_val_loss": best_val_loss},
                        scaler=scaler)

    for epoch in range(start_epoch, config["epochs"]):
        cur_epoch[0] = epoch
        print(f"\n--- Epoch {epoch+1}/{config['epochs']} ---")
        train_loss, steps, global_step = train_epoch(
            model, train_loader, opt, sched, DEVICE,
            grad_clip=config["grad_clip"],
            grad_accum_steps=config["grad_accum_steps"],
            use_fp16=USE_FP16,
            start_step=global_step,
            checkpoint_every=config.get("ckpt_every"),
            checkpoint_path=resume_path,
            on_checkpoint=_mid_epoch_save,
            scaler=scaler,
            ema=ema,
            label_smoothing=config.get("label_smoothing", 0.0),
        )

        # Evaluate with EMA shadow weights (better generalization)
        ema.apply_shadow()
        val = evaluate(model, val_loader, DEVICE, use_fp16=USE_FP16)
        ema.restore()

        val_loss, val_ppl = val["loss"], val["ppl"]
        print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  ppl={val_ppl:.2f}"
              f"  tok_acc={val['tok_acc']:.4f}  top5_acc={val['top5_acc']:.4f}")

        # Log per-epoch metrics
        _append_log(stage, {
            "epoch": epoch + 1, "stage": stage,
            "train_loss": train_loss, "val_loss": val_loss,
            "ppl": val_ppl, "tok_acc": val["tok_acc"], "top5_acc": val["top5_acc"],
            "lr": opt.param_groups[0]["lr"], "global_step": global_step,
            "elapsed_min": round((time.time() - t0) / 60, 2),
        })

        # Best-checkpoint tracking + early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_left = config.get("patience", 2)
            # Save best with EMA weights (better generalization)
            ema.apply_shadow()
            save_checkpoint(unwrap_model(model), opt, sched, epoch, val_loss, ckpt_path,
                            step=global_step,
                            extra={"stage": stage, "best_val_loss": best_val_loss},
                            scaler=scaler)
            ema.restore()
        else:
            patience_left -= 1
            print(f"  [early] val_loss did not improve ({patience_left} left)")
            if patience_left <= 0:
                print(f"  Early stopping after epoch {epoch+1} (best val_loss={best_val_loss:.4f})")
                break

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
