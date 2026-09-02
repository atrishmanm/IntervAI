"""
models/generator/train_utils.py — Research-Grade
=================================================
Shared training utilities for all training stages.

Features:
  - ChatML / raw-text datasets
  - WSD (Warmup-Stable-Decay) LR schedule + cosine fallback
  - EMA (Exponential Moving Average) of model weights
  - Label smoothing cross-entropy
  - Data quality filtering
  - Gradient accumulation (for small VRAM)
  - FP16 mixed precision (auto-enabled on GPU)
  - Gradient checkpointing support
  - Robust checkpoint save/load with resume
  - Hardware-aware batch sizing (see env_config)
"""

import hashlib
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────────────────────
# Multi-GPU helpers
# ─────────────────────────────────────────────────────────────

def wrap_data_parallel(model, device_ids=None):
    """Wrap a model in DataParallel if >1 GPU is available."""
    import torch
    n = torch.cuda.device_count()
    if n > 1:
        model = nn.DataParallel(model, device_ids=device_ids or list(range(n)))
    return model


def unwrap_model(model):
    """Return the underlying module (strip DataParallel wrapper)."""
    return model.module if isinstance(model, nn.DataParallel) else model


# ─────────────────────────────────────────────────────────────
# EMA (Exponential Moving Average) of model weights
# ─────────────────────────────────────────────────────────────

class EMA:
    """Maintains an exponential moving average of model parameters.

    Usage:
        ema = EMA(model, decay=0.999)
        # ... train step ...
        ema.update()
        # Before eval:
        ema.apply_shadow()
        # ... evaluate ...
        ema.restore()
    """
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in unwrap_model(model).named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self):
        """Update shadow weights with current model weights."""
        m = unwrap_model(self.model)
        for name, param in m.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data

    def apply_shadow(self):
        """Replace model weights with shadow weights (for eval)."""
        m = unwrap_model(self.model)
        self.backup = {}
        for name, param in m.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()

    def restore(self):
        """Restore original model weights (after eval)."""
        m = unwrap_model(self.model)
        for name, param in m.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name].clone()
        self.backup = {}


# ─────────────────────────────────────────────────────────────
# Data Quality Filter
# ─────────────────────────────────────────────────────────────

def quality_filter(example, min_len=50, max_len=10000):
    """Filter out low-quality examples."""
    if not example or not isinstance(example, list):
        return False
    text_parts = []
    for msg in example:
        if isinstance(msg, dict):
            text_parts.append(msg.get("content", ""))
    text = " ".join(text_parts)
    if len(text) < min_len:
        return False
    if len(text) > max_len:
        return False
    unique_chars = len(set(text))
    if unique_chars / max(len(text), 1) < 0.05:
        return False
    lines = text.split("\n")
    blank_ratio = sum(1 for l in lines if not l.strip()) / max(len(lines), 1)
    if blank_ratio > 0.5:
        return False
    return True


# ─────────────────────────────────────────────────────────────
# Environment-aware helpers
# ─────────────────────────────────────────────────────────────

def import_env():
    """Import env_config once and return the module."""
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from env_config import (
        ENV, ROOT, DATA_DIR, SAVE_ROOT, DEVICE, BASE_BATCH_SIZE,
        GRAD_ACCUM_STEPS, USE_FP16, VOCAB_SIZE, MODEL_SIZE,
        NUM_GPUS, EFFECTIVE_BATCH_SIZE, print_env_summary,
    )
    return dict(
        ENV=ENV, ROOT=ROOT, DATA_DIR=DATA_DIR, SAVE_ROOT=SAVE_ROOT,
        DEVICE=DEVICE, BASE_BATCH_SIZE=BASE_BATCH_SIZE,
        GRAD_ACCUM_STEPS=GRAD_ACCUM_STEPS, USE_FP16=USE_FP16,
        VOCAB_SIZE=VOCAB_SIZE, MODEL_SIZE=MODEL_SIZE,
        NUM_GPUS=NUM_GPUS, EFFECTIVE_BATCH_SIZE=EFFECTIVE_BATCH_SIZE,
        print_env_summary=print_env_summary,
    )


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────

class TextDataset(Dataset):
    """Simple dataset that tokenizes text files line-by-line."""
    def __init__(self, file_path: str, tokenizer, max_len: int = 512, limit: int = None):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples = []
        path = Path(file_path)
        if path.suffix == ".jsonl":
            with open(path, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if limit and i >= limit:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if "messages" in rec:
                            text = self._format_chatml(rec["messages"])
                        elif "text" in rec:
                            text = rec["text"]
                        elif "content" in rec:
                            text = rec["content"]
                        elif "code" in rec:
                            text = rec["code"]
                        elif "output" in rec:
                            text = rec["output"]
                        else:
                            text = str(rec)
                        if text and len(text.strip()) > 10:
                            self.examples.append(text)
                    except json.JSONDecodeError:
                        continue
        elif path.suffix == ".txt":
            with open(path, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if limit and i >= limit:
                        break
                    line = line.strip()
                    if line and len(line) > 20:
                        self.examples.append(line)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    def _format_chatml(self, messages):
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|{role}|> {content} <|end|>")
        return "\n".join(parts)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]
        encoded = self.tokenizer.encode(text)
        input_ids = encoded.ids[:self.max_len]
        pad_id = self.tokenizer.token_to_id("[PAD]") or 0
        if len(input_ids) < self.max_len:
            input_ids = input_ids + [pad_id] * (self.max_len - len(input_ids))
        attention_mask = [1 if tid != pad_id else 0 for tid in input_ids]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(input_ids, dtype=torch.long),
        }


class ChatDataset(Dataset):
    """Dataset for training that handles ALL data schemas used in this project."""
    def __init__(self, jsonl_path: str, tokenizer, max_len: int = 512, limit: int = None,
                 wrap_code=True):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.examples = []
        with open(jsonl_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    msgs = self._messages_from_record(rec)
                    if msgs:
                        self.examples.append(msgs)
                except json.JSONDecodeError:
                    continue

    @staticmethod
    def _messages_from_record(rec: dict):
        if not isinstance(rec, dict):
            return None
        if "messages" in rec and isinstance(rec["messages"], list):
            msgs = []
            for m in rec["messages"]:
                role = m.get("role", "user")
                content = m.get("content", "")
                if content:
                    msgs.append({"role": role, "content": content})
            return msgs or None
        if "conversations" in rec and isinstance(rec["conversations"], list):
            msgs = []
            for m in rec["conversations"]:
                role = m.get("from", "user")
                role = "assistant" if role == "gpt" else ("user" if role == "human" else role)
                content = m.get("value", "")
                if content:
                    msgs.append({"role": role, "content": content})
            return msgs or None
        if "instruction" in rec or "input" in rec or "query" in rec:
            instr = rec.get("instruction") or rec.get("query") or rec.get("input") or ""
            out = rec.get("response") or rec.get("output") or rec.get("answer") or ""
            if instr and out:
                return [{"role": "user", "content": instr},
                        {"role": "assistant", "content": out}]
            if instr:
                return [{"role": "user", "content": instr}]
        if "code" in rec:
            code = rec["code"] or ""
            doc = rec.get("doc") or ""
            if code:
                code_text = f"<|code|>\n{code}\n<|/code|>"
                if doc:
                    return [{"role": "user", "content": f"Explain: {doc}"},
                            {"role": "assistant", "content": code_text}]
                return [{"role": "user", "content": code_text}]
        if "content" in rec:
            content = rec["content"]
            if isinstance(content, str) and content.strip():
                return [{"role": "user", "content": f"<|code|>\n{content}\n<|/code|>"}]
            if isinstance(content, list):
                try:
                    text = bytes(content).decode("utf-8", errors="replace")
                    if text.strip():
                        return [{"role": "user", "content": f"<|code|>\n{text}\n<|/code|>"}]
                except Exception:
                    return None
            return None
        if "student_answer" in rec:
            q = rec.get("question", "")
            sa = rec.get("student_answer", "")
            ia = rec.get("instructor_answer", "")
            score = rec.get("score_avg", 0) or rec.get("score", 0)
            if sa:
                return [{"role": "user",
                         "content": f"Question: {q}\nStudent answer: {sa}\n"
                                    f"Reference answer: {ia}\n\nEvaluate the student answer."},
                        {"role": "assistant",
                         "content": f"Score: {score}/5. The answer "
                                    f"{'covers the key points well' if float(score) >= 3 else 'is incomplete or incorrect'}."}]
            return None
        if "text" in rec:
            text = rec["text"]
            if isinstance(text, str) and text.strip():
                return [{"role": "user", "content": text}]
            return None
        if rec:
            return [{"role": "user", "content": str(rec)[:2000]}]
        return None

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        messages = self.examples[idx]
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|{role}|> {content} <|end|>")
        full_text = "\n".join(parts)
        encoded = self.tokenizer.encode(full_text)
        input_ids = encoded.ids[:self.max_len]
        pad_id = self.tokenizer.token_to_id("[PAD]") or 0
        if len(input_ids) < self.max_len:
            input_ids = input_ids + [pad_id] * (self.max_len - len(input_ids))
        attention_mask = [1 if tid != pad_id else 0 for tid in input_ids]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(input_ids, dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────
# Deduplication
# ─────────────────────────────────────────────────────────────

def dedup_examples(examples):
    """Deduplicate example message-lists using a full-content SHA256 hash."""
    seen = set()
    uniq = []
    for m in examples:
        key = hashlib.sha256(
            json.dumps(m, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        if key not in seen:
            seen.add(key)
            uniq.append(m)
    return uniq


# ─────────────────────────────────────────────────────────────
# Learning Rate Schedules
# ─────────────────────────────────────────────────────────────

def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    """Cosine decay with linear warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_wsd_schedule(optimizer, warmup_steps, stable_steps, decay_steps, peak_lr, min_lr_ratio=0.1):
    """Warmup-Stable-Decay (WSD) learning rate schedule.

    Phase 1 (Warmup): Linear warmup from 0 to peak_lr
    Phase 2 (Stable): Hold at peak_lr
    Phase 3 (Decay): Linear decay from peak_lr to peak_lr * min_lr_ratio
    """
    min_lr = peak_lr * min_lr_ratio

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        elif step < warmup_steps + stable_steps:
            return 1.0
        else:
            decay_progress = (step - warmup_steps - stable_steps) / max(1, decay_steps)
            decay_progress = min(decay_progress, 1.0)
            return 1.0 - (1.0 - min_lr_ratio) * decay_progress

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────
# Training Loop (with grad accumulation + optional FP16)
# ─────────────────────────────────────────────────────────────

def _make_scaler(use_fp16):
    try:
        return torch.amp.GradScaler("cuda", enabled=use_fp16) if torch.cuda.is_available() else None
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=use_fp16) if torch.cuda.is_available() else None


def train_epoch(model, dataloader, optimizer, scheduler, device,
                grad_clip=1.0, grad_accum_steps=1, use_fp16=False,
                start_step=0, log_every=50, on_checkpoint=None,
                checkpoint_every=None, checkpoint_path=None, scaler=None,
                ema=None, label_smoothing=0.0):
    """Train for one epoch. Returns (avg_loss, num_batches, global_step)."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    global_step = start_step
    optimizer.zero_grad()

    if scaler is None:
        scaler = _make_scaler(use_fp16)

    def _optimizer_step():
        nonlocal global_step
        if grad_clip > 0:
            if use_fp16 and scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if use_fp16 and scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.update()
        global_step += 1

    for i, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_fp16):
            result = model(input_ids=input_ids, labels=labels,
                          label_smoothing=label_smoothing)
            loss = result["loss"]
            if getattr(loss, "dim", lambda: 0)() > 0:
                loss = loss.mean()
            loss = loss / grad_accum_steps

        if use_fp16 and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        num_batches += 1

        if (i + 1) % grad_accum_steps == 0:
            _optimizer_step()
            if checkpoint_every and checkpoint_path and global_step % checkpoint_every == 0:
                if on_checkpoint:
                    on_checkpoint(global_step, checkpoint_path)

        total_loss += loss.item() * grad_accum_steps

        if (num_batches % log_every) == 0:
            print(f"    batch {num_batches}/{len(dataloader)}  loss={loss.item()*grad_accum_steps:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}")

    if num_batches % grad_accum_steps != 0:
        _optimizer_step()

    return total_loss / max(num_batches, 1), num_batches, global_step


@torch.no_grad()
def evaluate(model, dataloader, device, use_fp16=False):
    """Evaluate model. Returns dict with avg loss, perplexity, token accuracy, top-5 accuracy."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    correct_tok = 0
    correct_top5 = 0
    total_tok = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast("cuda", enabled=use_fp16):
            result = model(input_ids=input_ids, labels=labels)
            logits = result["logits"]
            loss = result["loss"]
            if getattr(loss, "dim", lambda: 0)() > 0:
                loss = loss.mean()

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        preds = shift_logits.argmax(dim=-1)
        top5 = shift_logits.topk(5, dim=-1).indices

        mask = shift_labels != unwrap_model(model).pad_id

        correct_tok += ((preds == shift_labels) & mask).sum().item()
        correct_top5 += ((shift_labels.unsqueeze(-1) == top5) & mask.unsqueeze(-1)).any(dim=-1).sum().item()
        total_tok += mask.sum().item()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    perplexity = math.exp(min(avg_loss, 20))
    tok_acc = correct_tok / max(total_tok, 1)
    top5_acc = correct_top5 / max(total_tok, 1)
    return {
        "loss": avg_loss,
        "ppl": perplexity,
        "tok_acc": tok_acc,
        "top5_acc": top5_acc,
    }


# ─────────────────────────────────────────────────────────────
# Checkpoint save/load with resume support
# ─────────────────────────────────────────────────────────────

def _state_dict(model):
    """Get state dict from a model, stripping DataParallel's 'module.' prefix."""
    m = unwrap_model(model)
    return m.state_dict()


def save_checkpoint(model, optimizer, scheduler, epoch, loss, path, step=0,
                    extra=None, scaler=None, tokenizer=None):
    """Save training checkpoint (safe, robust). Includes scaler + global step."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    m = unwrap_model(model)
    state = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": _state_dict(model),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "loss": loss,
        "config": m.config.__dict__ if hasattr(m, "config") else {},
    }
    if extra:
        state.update(extra)
    tmp_path = path.with_suffix(".tmp")
    torch.save(state, tmp_path)
    tmp_path.replace(path)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None):
    """Load training checkpoint. Returns (epoch, loss, step)."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    sd = checkpoint["model_state_dict"]

    target = unwrap_model(model)
    if any(k.startswith("module.") for k in sd):
        sd = {k[len("module."):]: v for k, v in sd.items()}

    target_sd = target.state_dict()
    shape_ok = True
    for k, v in sd.items():
        if k in target_sd and tuple(target_sd[k].shape) != tuple(v.shape):
            shape_ok = False
            break
    if not shape_ok:
        print(f"  WARN: checkpoint at {path} was saved from a different model "
              f"size/shape than the current model. Starting fresh instead of resuming.")
        return -1, 0.0, 0

    target.load_state_dict(sd)
    if optimizer and checkpoint.get("optimizer_state_dict"):
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception as e:
            print(f"  WARN: Could not load optimizer state: {e}")
    if scheduler and checkpoint.get("scheduler_state_dict"):
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except Exception as e:
            print(f"  WARN: Could not load scheduler state: {e}")
    if scaler is not None and checkpoint.get("scaler_state_dict"):
        try:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        except Exception as e:
            print(f"  WARN: Could not load scaler state: {e}")
    return (
        checkpoint.get("epoch", 0),
        checkpoint.get("loss", 0.0),
        checkpoint.get("step", 0),
    )


# ─────────────────────────────────────────────────────────────
# Tokenizer Loading
# ─────────────────────────────────────────────────────────────

def load_tokenizer(path):
    """Load a trained BPE tokenizer."""
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(path))


# ─────────────────────────────────────────────────────────────
# Config Presets (hardware-aware, <8h total)
# ─────────────────────────────────────────────────────────────

def make_training_configs(env):
    """Build hardware-aware training configs (<8h total on T4x2).

    Research-grade defaults:
      - WSD schedule
      - EMA weight averaging
      - Label smoothing
      - Gradient checkpointing for large model
    """
    batch = env["BASE_BATCH_SIZE"]
    accum = env["GRAD_ACCUM_STEPS"]
    return {
        "pretrain": {
            "lr": 3e-4, "warmup_ratio": 0.02, "weight_decay": 0.1,
            "betas": (0.9, 0.95), "epochs": 3, "batch_size": batch,
            "grad_clip": 1.0, "max_len": 1024, "grad_accum_steps": accum,
            "patience": 2, "ckpt_every": 2000, "lr_finder": True,
            "ema_decay": 0.999, "label_smoothing": 0.05,
            "schedule": "wsd", "stable_pct": 0.80, "decay_pct": 0.18,
            "gradient_checkpointing": False,
        },
        "domain": {
            "lr": 1e-4, "warmup_ratio": 0.05, "weight_decay": 0.1,
            "betas": (0.9, 0.95), "epochs": 3, "batch_size": batch,
            "grad_clip": 1.0, "max_len": 1024, "grad_accum_steps": accum,
            "patience": 2, "ckpt_every": 2000, "lr_finder": True,
            "ema_decay": 0.999, "label_smoothing": 0.05,
            "schedule": "wsd", "stable_pct": 0.80, "decay_pct": 0.18,
            "gradient_checkpointing": False,
        },
        "instruction": {
            "lr": 5e-5, "warmup_ratio": 0.05, "weight_decay": 0.05,
            "betas": (0.9, 0.99), "epochs": 2, "batch_size": batch,
            "grad_clip": 1.0, "max_len": 1024, "grad_accum_steps": accum,
            "patience": 2, "ckpt_every": 1000, "lr_finder": True,
            "ema_decay": 0.999, "label_smoothing": 0.05,
            "schedule": "wsd", "stable_pct": 0.80, "decay_pct": 0.18,
            "gradient_checkpointing": False,
        },
        "interview": {
            "lr": 2e-5, "warmup_ratio": 0.1, "weight_decay": 0.05,
            "betas": (0.9, 0.99), "epochs": 3, "batch_size": batch,
            "grad_clip": 1.0, "max_len": 1024, "grad_accum_steps": accum,
            "patience": 2, "ckpt_every": 1000, "lr_finder": True,
            "ema_decay": 0.999, "label_smoothing": 0.05,
            "schedule": "wsd", "stable_pct": 0.80, "decay_pct": 0.18,
            "gradient_checkpointing": True,
        },
        "evaluator": {
            "lr": 2e-5, "warmup_ratio": 0.1, "weight_decay": 0.05,
            "betas": (0.9, 0.99), "epochs": 5, "batch_size": batch,
            "grad_clip": 1.0, "max_len": 1024, "grad_accum_steps": accum,
            "patience": 2, "ckpt_every": 1000, "lr_finder": True,
            "ema_decay": 0.999, "label_smoothing": 0.1,
            "schedule": "wsd", "stable_pct": 0.75, "decay_pct": 0.20,
            "gradient_checkpointing": True,
        },
        "followup": {
            "lr": 2e-5, "warmup_ratio": 0.1, "weight_decay": 0.05,
            "betas": (0.9, 0.99), "epochs": 3, "batch_size": batch,
            "grad_clip": 1.0, "max_len": 1024, "grad_accum_steps": accum,
            "patience": 2, "ckpt_every": 1000, "lr_finder": True,
            "ema_decay": 0.999, "label_smoothing": 0.05,
            "schedule": "wsd", "stable_pct": 0.80, "decay_pct": 0.18,
            "gradient_checkpointing": False,
        },
    }


# ─────────────────────────────────────────────────────────────
# Auto-tuning: Learning Rate Finder (LR range test)
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def _lr_smoke_loss(model, batch, device, use_fp16):
    """One forward pass for the LR finder — no backward, no graph kept."""
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    with torch.amp.autocast("cuda", enabled=use_fp16):
        result = model(input_ids=input_ids, labels=labels, return_logits=False)
    loss = result["loss"]
    if getattr(loss, "dim", lambda: 0)() > 0:
        loss = loss.mean()
    return loss.item()


def find_learning_rate(model, dataloader, device, optimizer_factory, config,
                       lr_min=1e-6, lr_max=1e-2, num_steps=40, use_fp16=False):
    """Run a classic LR range test and return the suggested peak LR."""
    model.train()
    print("\n  [auto] Running LR range test...")
    if use_fp16:
        scaler = _make_scaler(True)
    else:
        scaler = None

    it = iter(dataloader)
    opt = optimizer_factory(lr=lr_min)
    losses = []
    lrs = []
    best_loss = float("inf")
    best_lr = lr_min

    for step in range(num_steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dataloader)
            batch = next(it)
        lr = lr_min * (lr_max / lr_min) ** (step / max(1, num_steps - 1))
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_fp16):
            result = model(input_ids=batch["input_ids"].to(device),
                           labels=batch["labels"].to(device), return_logits=False)
            loss = result["loss"]
            if getattr(loss, "dim", lambda: 0)() > 0:
                loss = loss.mean()
        if use_fp16 and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if use_fp16 and scaler is not None:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if use_fp16 and scaler is not None:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()

        l = loss.item()
        losses.append(l)
        lrs.append(lr)
        if l < best_loss:
            best_loss = l
            best_lr = lr
        if step % 10 == 0:
            print(f"    lr={lr:.2e}  loss={l:.4f}")

    suggested = max(lr_min, best_lr / 3.0)
    print(f"  [auto] LR finder done: best_loss={best_loss:.4f} at lr={best_lr:.2e} -> suggest {suggested:.2e}")
    if suggested > config["lr"]:
        suggested = config["lr"]
        print(f"  [auto] Capped to configured peak LR {suggested:.2e}")
    return suggested


# ─────────────────────────────────────────────────────────────
# Auto-tuning: Batch-size profiler (memory-aware)
# ─────────────────────────────────────────────────────────────

def profile_batch_size(model, dataset, device, base_batch, use_fp16=False,
                       max_batch=64, trials=2):
    """Try larger per-GPU batch sizes and return the largest that fits memory."""
    if not torch.cuda.is_available() or len(dataset) == 0:
        return base_batch
    print(f"\n  [auto] Profiling batch size (base={base_batch})...")
    model.train()

    def _try(bs):
        idx = torch.randint(len(dataset), (min(bs, len(dataset)),)).tolist()
        rows = [dataset[i] for i in idx]
        batch = {
            "input_ids": torch.stack([r["input_ids"] for r in rows]).to(device),
            "labels": torch.stack([r["labels"] for r in rows]).to(device),
        }
        try:
            opt = torch.optim.SGD(model.parameters(), lr=1e-6)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_fp16):
                loss = model(input_ids=batch["input_ids"], labels=batch["labels"],
                             return_logits=False)["loss"]
            if getattr(loss, "dim", lambda: 0)() > 0:
                loss = loss.mean()
            if use_fp16:
                scaler = _make_scaler(True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            torch.cuda.synchronize()
            return True
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return False
        except RuntimeError:
            torch.cuda.empty_cache()
            return False

    best = base_batch
    bs = base_batch
    while bs <= max_batch:
        ok = all(_try(bs) for _ in range(trials))
        if ok:
            best = bs
            print(f"    batch {bs}: OK")
            bs *= 2
        else:
            print(f"    batch {bs}: OOM — stopping")
            break
    print(f"  [auto] Profiling done: recommended per-GPU batch = {best}")
    return best
