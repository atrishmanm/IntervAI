"""
test_training_fixes.py
=======================
Comprehensive dry-run test for all training-pipeline fixes.
Run with: python test_training_fixes.py
"""

import sys
import os
sys.path.insert(0, '.')

import torch
from torch.utils.data import DataLoader
import random

# ── 1. Test imports ──────────────────────────────────────────
print('=== 1. Testing imports ===')
from models.generator.train_utils import (
    TextDataset, ChatDataset, PackedDataset, pack_dataset,
    collate_packed, pad_collate, dedup_examples, make_training_configs,
    train_epoch, evaluate, save_checkpoint, load_checkpoint,
    load_tokenizer, wrap_data_parallel, unwrap_model,
    find_learning_rate, profile_batch_size, _make_scaler, EMA,
    Muon, create_muon_optimizer, compile_model,
    get_cosine_schedule_with_warmup, get_wsd_schedule,
)
print('  train_utils: OK')

from models.generator.model import create_small_model
print('  model: OK')

from models.generator.analytics import TrainingAnalytics
print('  analytics: OK')

import importlib
train_mod = importlib.import_module('models.generator.train')
assert hasattr(train_mod, 'pad_collate'), 'pad_collate not exported from train.py'
print('  train.pad_collate: OK')

# ── 2. Test pad_collate with variable-length lists ────────────
print()
print('=== 2. Testing pad_collate (list inputs) ===')
batch_lists = [
    {'input_ids': [1, 2, 3],       'labels': [1, 2, 3]},
    {'input_ids': [4, 5, 6, 7, 8], 'labels': [4, 5, 6, 7, 8]},
    {'input_ids': [9, 10],         'labels': [9, 10]},
]
out = pad_collate(batch_lists)
assert out['input_ids'].shape == torch.Size([3, 5]), f"Bad shape: {out['input_ids'].shape}"
assert out['labels'].shape == torch.Size([3, 5]),    f"Bad shape: {out['labels'].shape}"
assert out['input_ids'][0, 3].item() == 0, "Expected pad=0"
assert out['input_ids'][2, 2].item() == 0, "Expected pad=0"
print(f'  shape={tuple(out["input_ids"].shape)}  padding=OK  PASS')

# ── 3. Test pad_collate with tensor inputs ─────────────────────
print()
print('=== 3. Testing pad_collate (tensor inputs) ===')
batch_tensors = [
    {'input_ids': torch.tensor([1, 2, 3]),       'labels': torch.tensor([1, 2, 3])},
    {'input_ids': torch.tensor([4, 5, 6, 7, 8]), 'labels': torch.tensor([4, 5, 6, 7, 8])},
]
out2 = pad_collate(batch_tensors)
assert out2['input_ids'].shape == torch.Size([2, 5]), f"Bad shape: {out2['input_ids'].shape}"
print(f'  shape={tuple(out2["input_ids"].shape)}  PASS')

# ── 4. Test PackedDataset returns tensors ─────────────────────
print()
print('=== 4. Testing PackedDataset ===')
raw_examples = [{'input_ids': list(range(i, i+10)), 'labels': list(range(i, i+10))}
                for i in range(50)]
pds = PackedDataset(raw_examples, max_len=30)
assert len(pds) > 0, "PackedDataset is empty"
item = pds[0]
assert isinstance(item['input_ids'], torch.Tensor), \
    f"PackedDataset must return tensor, got {type(item['input_ids'])}"
assert isinstance(item['labels'], torch.Tensor), \
    f"PackedDataset must return tensor, got {type(item['labels'])}"
print(f'  len={len(pds)}  item[0] type={type(item["input_ids"]).__name__}  PASS')

# ── 5. DataLoader + pad_collate with PackedDataset ────────────
print()
print('=== 5. Testing DataLoader + pad_collate (PackedDataset) ===')
loader = DataLoader(pds, batch_size=4, collate_fn=pad_collate)
b = next(iter(loader))
assert isinstance(b['input_ids'], torch.Tensor), "Expected tensor from DataLoader"
print(f'  DataLoader batch shape={tuple(b["input_ids"].shape)}  PASS')

# ── 6. DataLoader + pad_collate with variable-length dataset ──
print()
print('=== 6. Testing DataLoader + pad_collate (variable-length) ===')

class FakeVarLenDataset(torch.utils.data.Dataset):
    def __init__(self, n):
        self.data = [
            {'input_ids': list(range(random.randint(3, 25))),
             'labels':    list(range(random.randint(3, 25)))}
            for _ in range(n)
        ]
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]

var_ds = FakeVarLenDataset(80)
var_dl = DataLoader(var_ds, batch_size=8, shuffle=True, collate_fn=pad_collate)
try:
    vb = next(iter(var_dl))
    print(f'  Variable-length DataLoader batch shape={tuple(vb["input_ids"].shape)}  PASS')
except RuntimeError as e:
    print(f'  FAIL: {e}')
    sys.exit(1)

# ── 7. Test TrainingAnalytics (no AttributeError) ─────────────
print()
print('=== 7. Testing TrainingAnalytics API ===')
model_an = create_small_model(vocab_size=100)
an = TrainingAnalytics(model_an, total_steps=100, stage_name='test')
an.start_epoch()       # must NOT raise AttributeError
an.start_step()
an.end_data_loading()
an.end_forward()
an.end_backward()
an.end_optimizer()
an.update_loss(2.3)
an.update_grad_norm(0.5)   # correct method name (not update_gradient_norm)
an.update_throughput(512)
an.update_memory()
m = an.get_metrics()
required_keys = ['loss', 'ppl', 'tok_per_sec', 'grad_norm', 'stability_score', 'peak_memory_gb']
for k in required_keys:
    assert k in m, f"Missing analytics key: {k}"
print(f'  All required metrics present  PASS')

# ── 8. Model forward/backward ─────────────────────────────────
print()
print('=== 8. Testing model forward/backward ===')
model = create_small_model(vocab_size=100)
ids  = torch.randint(0, 100, (2, 16))
lbls = torch.randint(0, 100, (2, 16))
result = model(input_ids=ids, labels=lbls)
assert 'loss' in result and 'logits' in result
loss_val = result['loss'].item()
assert loss_val > 0, "Loss should be positive"
result['loss'].backward()
grad_count = sum(1 for p in model.parameters() if p.grad is not None)
assert grad_count > 0, "No gradients computed"
print(f'  loss={loss_val:.4f}  logits={tuple(result["logits"].shape)}  grads={grad_count}  PASS')

# ── 9. Test train_epoch on CPU ────────────────────────────────
print()
print('=== 9. Testing train_epoch (CPU smoke test) ===')
model2 = create_small_model(vocab_size=100)
opt    = torch.optim.AdamW(model2.parameters(), lr=1e-3)
sched  = get_cosine_schedule_with_warmup(opt, warmup_steps=1, total_steps=10)
an2    = TrainingAnalytics(model2, total_steps=10, stage_name='smoke')

class TinyDS(torch.utils.data.Dataset):
    def __init__(self):
        self.data = [
            {'input_ids': list(range(i % 8 + 5)),
             'labels':    list(range(i % 8 + 5))}
            for i in range(20)
        ]
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]

tiny_ds = TinyDS()
tiny_dl = DataLoader(tiny_ds, batch_size=4, shuffle=True, collate_fn=pad_collate)
avg_loss, n_batches, global_step = train_epoch(
    model2, tiny_dl, opt, sched, device='cpu',
    grad_clip=1.0, grad_accum_steps=2, use_fp16=False,
    start_step=0, log_every=999, analytics=an2,
    label_smoothing=0.05,
)
assert avg_loss > 0, "Expected positive loss"
assert n_batches > 0, "Expected >0 batches"
assert global_step > 0, "Expected global_step to advance"
print(f'  avg_loss={avg_loss:.4f}  batches={n_batches}  global_step={global_step}  PASS')

# ── 10. Test dedup_examples ───────────────────────────────────
print()
print('=== 10. Testing dedup_examples ===')
dupes = [
    [{'role': 'user', 'content': 'hello'}],
    [{'role': 'user', 'content': 'hello'}],
    [{'role': 'user', 'content': 'world'}],
]
uniq = dedup_examples(dupes)
assert len(uniq) == 2, f"Expected 2 unique, got {len(uniq)}"
print(f'  {len(dupes)} -> {len(uniq)} after dedup  PASS')

# ── 11. profile_batch_size with tensor-returning dataset ──────
print()
print('=== 11. Testing profile_batch_size with tensor dataset ===')

class TensorDS(torch.utils.data.Dataset):
    def __init__(self):
        self.data = [
            {'input_ids': torch.randint(0, 100, (random.randint(4, 16),)),
             'labels':    torch.randint(0, 100, (random.randint(4, 16),))}
            for _ in range(20)
        ]
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]

model3 = create_small_model(vocab_size=100)
result_bs = profile_batch_size(model3, TensorDS(), device='cpu',
                               base_batch=4, use_fp16=False)
print(f'  profile_batch_size (CPU) returned {result_bs}  PASS')

# ── 12. Test unwrap_model with torch.compile + DataParallel wrappers ──
print()
print('=== 12. Testing unwrap_model unwrapping ===')
raw_model = create_small_model(vocab_size=100)

class FakeDataParallel(torch.nn.Module):
    def __init__(self, mod):
        super().__init__()
        self.module = mod

class FakeCompiledModule(torch.nn.Module):
    def __init__(self, mod):
        super().__init__()
        self._orig_mod = mod

# Test nested wrappers (e.g. torch.compile around DataParallel)
wrapped = FakeCompiledModule(FakeDataParallel(raw_model))
unwrapped = unwrap_model(wrapped)
assert unwrapped is raw_model, "Failed to unwrap compiled DataParallel wrapper"
assert hasattr(unwrapped, 'update_dropout'), "Unwrapped model missing update_dropout"
print('  unwrap_model strips FakeCompiledModule(FakeDataParallel) -> raw model  PASS')

print()
print('=' * 55)
print('  ALL 12 TESTS PASSED')
print('  Training pipeline is healthy and ready for Kaggle!')
print('=' * 55)
