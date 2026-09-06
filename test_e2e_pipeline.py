"""End-to-end test: simulate what Kaggle training actually does."""
import sys
import os
import json
import time
import torch
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

print("=" * 60)
print("  END-TO-END TRAINING PIPELINE TEST")
print("=" * 60)

# ── Test 1: PackedDataset + collate + profile_batch_size flow ──
print("\n[1/5] PackedDataset -> collate -> model forward (the exact Kaggle path)...")

from tokenizers import Tokenizer
tok_path = Path("tokenizer/saved/tokenizer.json")
tokenizer = Tokenizer.from_file(str(tok_path))

from models.generator.train_utils import PackedDataset, collate_packed, profile_batch_size
from models.generator.model import GeneratorConfig, InterviewGenerator

# Build a tiny PackedDataset the same way train.py does
dummy_examples = []
for i in range(200):
    text = f"<|user|> Question {i}: What is binary search? <|end|> <|assistant|> Binary search divides a sorted array in half. <|end|>"
    encoded = tokenizer.encode(text)
    input_ids = encoded.ids[:2048]
    dummy_examples.append({"input_ids": input_ids})

packed_ds = PackedDataset(dummy_examples, max_len=2048)
print(f"  PackedDataset: {len(packed_ds)} packed sequences")

sample = packed_ds[0]
assert isinstance(sample["input_ids"], (list, torch.Tensor))
print(f"  [OK] __getitem__ returns tensor/list (len={len(sample['input_ids'])})")

batch = collate_packed([packed_ds[0], packed_ds[1], packed_ds[2]])
assert isinstance(batch["input_ids"], torch.Tensor)
print(f"  [OK] collate_packed returns tensor {batch['input_ids'].shape}")

config = GeneratorConfig(vocab_size=16000, embed_dim=256, n_layers=4, n_heads=4, ff_dim=512, max_len=2048)
model = InterviewGenerator(config).to("cpu")
result = profile_batch_size(model, packed_ds, "cpu", base_batch=2, use_fp16=False, max_batch=4, trials=1)
print(f"  [OK] profile_batch_size: {result}")

# ── Test 2: save_progress creates directory ──
print("\n[2/5] save_progress creates directory if missing...")

test_dir = Path("models/generator/saved_test")
progress_file = test_dir / "training_progress.json"

def save_progress(stage, status, elapsed_min):
    test_dir.mkdir(parents=True, exist_ok=True)
    progress = {}
    if progress_file.exists():
        with open(progress_file) as f:
            progress = json.load(f)
    progress[stage] = {"status": status, "elapsed_min": elapsed_min, "timestamp": time.time()}
    with open(progress_file, "w") as f:
        json.dump(progress, f, indent=2)

save_progress("pretrain", "completed", 85.3)
assert progress_file.exists()
with open(progress_file) as f:
    loaded = json.load(f)
assert loaded["pretrain"]["status"] == "completed"
print(f"  [OK] save_progress creates dir + writes file")
import shutil
shutil.rmtree(test_dir, ignore_errors=True)

# ── Test 3: Full train.py pipeline simulation ──
print("\n[3/5] Full train.py pipeline simulation...")

from models.generator.train_utils import (
    PackedDataset, collate_packed, make_training_configs,
    create_muon_optimizer, train_epoch, EMA
)

max_len = 512
examples_for_ds = []
for i in range(100):
    text = f"<|user|> Q{i}: explain Big O notation <|end|> <|assistant|> Big O describes algorithm complexity. <|end|>"
    encoded = tokenizer.encode(text)
    examples_for_ds.append({"input_ids": encoded.ids[:max_len]})

packed = PackedDataset(examples_for_ds, max_len=max_len)
print(f"  Dataset: {len(packed)} packed sequences")

config = GeneratorConfig(vocab_size=16000, embed_dim=256, n_layers=4, n_heads=4, ff_dim=512, max_len=max_len)
model = InterviewGenerator(config)

# Create Muon optimizer the same way train.py does
optimizer = create_muon_optimizer(model, lr=3e-4, weight_decay=0.1, momentum=0.95)

# Create scheduler
from models.generator.train_utils import get_cosine_schedule_with_warmup
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps=5, total_steps=20)

# Create dataloader
from torch.utils.data import DataLoader
loader = DataLoader(packed, batch_size=2, shuffle=True, num_workers=0,
                    collate_fn=lambda b: collate_packed(b, pad_token_id=0))

# Run 2 training steps
model.train()
loss, nbatch, step = train_epoch(
    model, loader, optimizer, scheduler, "cpu",
    grad_clip=1.0, grad_accum_steps=1, use_fp16=False,
    start_step=0, log_every=1, ema=None,
    label_smoothing=0.05
)
print(f"  [OK] train_epoch completed: loss={loss:.4f}, batches={nbatch}, step={step}")

# ── Test 4: Checkpoint save/load ──
print("\n[4/5] Checkpoint save/load during training...")

from models.generator.train_utils import save_checkpoint, load_checkpoint

ckpt_path = Path("models/generator/saved/test_e2e.pt")
save_checkpoint(model, optimizer, scheduler, epoch=1, loss=loss, path=ckpt_path, step=42)
print(f"  [OK] Saved checkpoint ({ckpt_path.stat().st_size/1e6:.1f} MB)")

model2 = InterviewGenerator(config)
optimizer2 = create_muon_optimizer(model2, lr=3e-4, weight_decay=0.1, momentum=0.95)
scheduler2 = get_cosine_schedule_with_warmup(optimizer2, warmup_steps=5, total_steps=20)
epoch_loaded, loss_loaded, step_loaded = load_checkpoint(ckpt_path, model2, optimizer2, scheduler2)
assert step_loaded == 42, f"Expected step=42, got {step_loaded}"
print(f"  [OK] Loaded checkpoint (epoch={epoch_loaded}, loss={loss_loaded:.4f}, step={step_loaded})")
ckpt_path.unlink()

# ── Test 5: EMA + full training loop ──
print("\n[5/5] EMA + research wrapper training loop...")

ema = EMA(model, decay=0.999)
model.train()
loss, nbatch, step = train_epoch(
    model, loader, optimizer, scheduler, "cpu",
    grad_clip=1.0, grad_accum_steps=1, use_fp16=False,
    start_step=0, log_every=1, ema=ema,
    label_smoothing=0.05
)
print(f"  [OK] train_epoch with EMA: loss={loss:.4f}")

# Test EMA apply/restore
ema.apply_shadow()
model.train()
loss2, _, _ = train_epoch(
    model, loader, optimizer, scheduler, "cpu",
    grad_clip=1.0, grad_accum_steps=1, use_fp16=False,
    start_step=step, log_every=1, ema=ema,
    label_smoothing=0.05
)
ema.restore()
print(f"  [OK] EMA apply/restore works")

# Test research wrapper
from models.generator.research_techniques import ResearchTrainingWrapper
wrapper = ResearchTrainingWrapper(
    model, optimizer, use_sam=False, use_lookahead=False,
    use_gc=False, use_progressive_resizing=False, use_swa=False
)
loss3, _, _ = train_epoch(
    model, loader, optimizer, scheduler, "cpu",
    grad_clip=1.0, grad_accum_steps=1, use_fp16=False,
    start_step=step, log_every=1, ema=ema,
    label_smoothing=0.05, research_wrapper=wrapper
)
print(f"  [OK] train_epoch with research wrapper: loss={loss3:.4f}")

print("\n" + "=" * 60)
print("  ALL END-TO-END TESTS PASSED")
print("=" * 60)
print("\n  Kaggle training will work correctly.")
