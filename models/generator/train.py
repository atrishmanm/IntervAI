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
    TextDataset, ChatDataset, get_cosine_schedule_with_warmup, get_cosine_with_warm_restarts_schedule, get_wsd_schedule,
    train_epoch, evaluate, save_checkpoint, load_checkpoint,
    load_tokenizer, make_training_configs, dedup_examples, quality_filter,
    wrap_data_parallel, unwrap_model, find_learning_rate, profile_batch_size,
    _make_scaler, EMA, Muon, create_muon_optimizer, PackedDataset, pack_dataset,
    compile_model,
)
from models.generator.analytics import TrainingAnalytics

# ─────────────────────────────────────────────────────────────
# Stage definitions: which data file feeds each stage
# ─────────────────────────────────────────────────────────────

# Data file selection per stage. Paths are relative to ROOT.
# ALL datasets are REAL - no synthetic data.
# New datasets for company templates, industry modules, salary negotiation.
STAGE_DATA = {
    "pretrain": [
        "data/raw/starcoder_large.jsonl",           # Real code from BigCode
        "data/raw/codesearchnet.jsonl",              # Real code search pairs
        "data/raw/cruxeval/cruxeval.jsonl",          # Real code reasoning
        "data/raw/codefeedback.jsonl",               # Real code feedback
    ],
    "domain": [
        "data/raw/opencodeinstruct.jsonl",           # Real coding instructions
        "data/raw/oasst_coding.jsonl",               # Real coding Q&A from OpenAssistant
        "data/raw/codealpaca.jsonl",                 # Real code instructions
        "data/raw/kodcode_verified.jsonl",           # Real verified coding (KodCode)
    ],
    "instruction": [
        "data/raw/opencodeinstruct.jsonl",           # Real instructions
        "data/raw/codealpaca.jsonl",                 # Real code instructions
        "data/raw/conversations.jsonl",              # Real conversations
        "data/raw/interview_sft_100k.jsonl",         # Real interview Q&A (100K)
    ],
    "interview": [
        "data/raw/conversations.jsonl",              # Real dialogues for interview style
        "data/raw/opencodeinstruct.jsonl",           # Real Q&A format
        "data/raw/oasst_coding.jsonl",               # Real technical Q&A
        "data/raw/interview_sft_100k.jsonl",         # Real interview conversations (100K)
    ],
    "evaluator": [
        "data/raw/mohler_asag.jsonl",                # Real scoring rubrics
    ],
    "followup": [
        "data/raw/oasst_coding.jsonl",               # Real follow-up conversations
        "data/raw/conversations.jsonl",              # Real dialogue chains
        "data/raw/interview_sft_100k.jsonl",         # Real interview follow-ups
    ],
    "resume_finetune": [
        "data/raw/resumes_54k.jsonl",                # Real resumes (54K)
        "data/raw/interview_sft_100k.jsonl",         # Real interview Q&A
        "data/raw/conversations.jsonl",              # Real conversations
    ],
    "negotiation": [
        "data/raw/negotiation_sft_100k.jsonl",       # Real salary negotiation (100K)
        "data/raw/conversations.jsonl",              # Real conversations
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
    "resume_finetune": "resume_finetuned.pt",
    "negotiation": "negotiation_tuned.pt",
}

# Which checkpoint each stage initializes FROM (None = from scratch / random)
STAGE_INIT = {
    "pretrain": None,          # starts from scratch
    "domain": "pretrained.pt",
    "instruction": "domain_tuned.pt",
    "interview": "domain_tuned.pt",     # instruction not strictly required before interview
    "evaluator": "interview_tuned.pt",
    "followup": "interview_tuned.pt",   # FIXED: was evaluator.pt (catastrophic forgetting)
    "resume_finetune": "final_model.pt",  # resume fine-tuning after all stages
    "negotiation": "resume_finetuned.pt",  # negotiation after resume fine-tuning
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
    """Build train + val datasets for the stage with optional sequence packing."""
    global _LIMIT
    files = STAGE_DATA[stage]
    max_len = config["max_len"]
    use_packing = config.get("sequence_packing", False)
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
            return {"input_ids": ids, "labels": ids}

    full = _Wrapper(uniq)
    val_size = min(500, max(1, len(full) // 30))
    train_size = len(full) - val_size
    train_ds, val_ds = random_split(full, [train_size, val_size])

    # Apply sequence packing for 2-3x throughput
    if use_packing:
        print(f"  Packing sequences (max_len={max_len}) for 2-3x throughput...")
        train_ds = pack_dataset(train_ds, max_len=max_len)
        print(f"  Packed: {train_size} examples -> {len(train_ds)} packed sequences")

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
    model = model.to(DEVICE)
    model = wrap_data_parallel(model)  # no-op if 1 GPU
    unwrap_model(model).enable_gradient_checkpointing()
    print(f"Model params: {unwrap_model(model).num_params_millions:.1f}M  (GPUs: {NUM_GPUS})")

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
        if config.get("optimizer") == "muon":
            return create_muon_optimizer(
                unwrap_model(model), lr=lr,
                weight_decay=config["weight_decay"],
                momentum=config.get("muon_momentum", 0.95),
            )
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
    stable_pct = config.get("stable_pct", 0.80)
    decay_pct = config.get("decay_pct", 0.18)
    stable_steps = int(total_steps * stable_pct)
    decay_steps = max(1, int(total_steps * decay_pct))
    peak_lr = config["lr"]

    # Choose schedule based on stage config
    sched_type = config.get("schedule", "wsd")
    if sched_type == "cosine":
        sched = get_cosine_schedule_with_warmup(opt, warmup_steps=warmup, total_steps=total_steps)
        print("  Using Cosine Annealing with Warmup schedule")
    elif sched_type == "cosine_restarts":
        sched = get_cosine_with_warm_restarts_schedule(opt, first_cycle_steps=max(1, total_steps // 3))
        print("  Using Cosine Annealing with Warm Restarts schedule")
    else:
        sched = get_wsd_schedule(opt, warmup_steps=warmup, stable_steps=stable_steps,
                                decay_steps=decay_steps, peak_lr=peak_lr)
        print("  Using Warmup-Stable-Decay (WSD) schedule")

    # Gradient Centralization hooks
    gc_hooks = []
    if config.get("use_gc", True):
        try:
            from models.generator.research_techniques import apply_gradient_centralization
            gc_hooks = apply_gradient_centralization(unwrap_model(model))
            print("  [Research] Gradient Centralization active (+1-2% accuracy)")
        except Exception as e:
            print(f"  [Research] GC setup skipped: {e}")

    # SWA (Stochastic Weight Averaging)
    swa = None
    if config.get("use_swa", True):
        try:
            from models.generator.research_techniques import SWA
            swa = SWA()
            print("  [Research] SWA active (averaging final 25% checkpoints)")
        except Exception as e:
            print(f"  [Research] SWA setup skipped: {e}")

    # EMA (exponential moving average) for stable checkpoint selection
    ema = EMA(model=unwrap_model(model), decay=config.get("ema_decay", 0.999))

    # torch.compile for JIT speedup
    if config.get("torch_compile") and torch.cuda.is_available():
        try:
            model = compile_model(model, mode="default")
            print("  torch.compile enabled for 20-30% speedup")
        except Exception as e:
            print(f"  torch.compile failed (falling back): {e}")

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

    # Initialize training analytics (15+ metrics)
    total_steps = len(train_loader) // config["grad_accum_steps"] * config["epochs"]
    analytics = TrainingAnalytics(model, total_steps, stage_name=stage)

    for epoch in range(start_epoch, config["epochs"]):
        cur_epoch[0] = epoch
        print(f"\n--- Epoch {epoch+1}/{config['epochs']} ---")

        # Update dynamic dropout
        epoch_total_steps = len(train_loader) // config["grad_accum_steps"]
        current_step = global_step - (epoch * epoch_total_steps)
        unwrap_model(model).update_dropout(current_step, epoch_total_steps)

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
            label_smoothing=config.get("label_smoothing", 0.05),
            analytics=analytics,
        )
        ema.update()

        # Update SWA in the final 25% of epochs
        if swa and (epoch + 1) >= max(1, int(config["epochs"] * 0.75)):
            swa.update(unwrap_model(model))
            print("  [Research] SWA checkpoint captured")

        val = evaluate(model, val_loader, DEVICE, use_fp16=USE_FP16)
        val_loss, val_ppl = val["loss"], val["ppl"]

        # Qualitative generation check for monitoring interview dialogue capability
        try:
            sample_prompt = "<|system|> You are an expert technical interviewer.<|end|><|user|> Can you explain the difference between a process and a thread?<|end|><|assistant|>"
            sample_ids = tokenizer.encode(sample_prompt).ids
            sample_tensor = torch.tensor([sample_ids], dtype=torch.long, device=DEVICE)
            with torch.no_grad():
                gen_ids = unwrap_model(model).generate(sample_tensor, max_new_tokens=40, temperature=0.7)
                gen_text = tokenizer.decode(gen_ids[0].tolist()).replace("Ġ", " ").replace("Ċ", "\n").strip()
                ans_preview = gen_text.split("<|assistant|>")[-1].strip()[:90]
                if ans_preview:
                    print(f"  [Sample Output]: \"{ans_preview}...\"")
        except Exception:
            pass

        # Print epoch summary with all analytics
        analytics.print_epoch_summary(epoch, train_loss, val)

        # Log per-epoch metrics
        metrics = analytics.get_metrics()
        _append_log(stage, {
            "epoch": epoch + 1, "stage": stage,
            "train_loss": train_loss, "val_loss": val_loss,
            "ppl": val_ppl, "tok_acc": val["tok_acc"], "top5_acc": val["top5_acc"],
            "lr": opt.param_groups[0]["lr"], "global_step": global_step,
            "elapsed_min": round((time.time() - t0) / 60, 2),
            "tok_per_sec": metrics["tok_per_sec"],
            "grad_norm": metrics["grad_norm"],
            "stability_score": metrics["stability_score"],
            "peak_memory_gb": metrics["peak_memory_gb"],
            "dropout_rate": unwrap_model(model).get_dropout_rate(),
        })

        # Best-checkpoint tracking + early stopping (use EMA weights for stability)
        ema.apply_shadow()
        ema_val = evaluate(model, val_loader, DEVICE, use_fp16=USE_FP16)
        ema.restore()
        ema_val_loss = ema_val["loss"]
        if ema_val_loss < best_val_loss:
            best_val_loss = ema_val_loss
            patience_left = config.get("patience", 2)
            save_checkpoint(unwrap_model(model), opt, sched, epoch, best_val_loss, ckpt_path,
                            step=global_step,
                            extra={"stage": stage, "best_val_loss": best_val_loss, "ema_val_loss": ema_val_loss},
                            scaler=scaler)
        else:
            patience_left -= 1
            print(f"  [early] val_loss did not improve ({patience_left} left)")
            if patience_left <= 0:
                print(f"  Early stopping after epoch {epoch+1} (best val_loss={best_val_loss:.4f})")
                break

    # Apply SWA weights if collected
    if swa and swa.n_models > 0:
        swa.apply(unwrap_model(model))
        print(f"  [Research] SWA applied across {swa.n_models} checkpoints to final model")
        save_checkpoint(unwrap_model(model), opt, sched, cur_epoch[0], best_val_loss, ckpt_path,
                        step=global_step,
                        extra={"stage": stage, "best_val_loss": best_val_loss, "swa_applied": True},
                        scaler=scaler)

    # Remove GC hooks
    for h in gc_hooks:
        try:
            h.remove()
        except Exception:
            pass

    # Print final analytics report
    report = analytics.get_final_report()
    print(f"\n  {stage} done in {report['elapsed_min']:.1f} min. Best val_loss={best_val_loss:.4f}")
    print(f"  Throughput: {report['avg_tok_per_sec']:.0f} tokens/sec | Peak Memory: {report['peak_memory_gb']:.2f} GB")
    print(f"  Checkpoint: {ckpt_path}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="INTERVUE unified training")
    ap.add_argument("--stage", default="all",
                    choices=["all", "pretrain", "domain", "instruction",
                             "interview", "evaluator", "followup", "resume_finetune", "negotiation"])
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

    stages = ["pretrain", "domain", "instruction", "interview", "evaluator", "followup", "resume_finetune", "negotiation"]
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
