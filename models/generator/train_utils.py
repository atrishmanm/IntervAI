"""
models/generator/train_utils.py
================================
Shared training utilities for all training stages.

Features:
  - ChatML / raw-text datasets
  - Cosine LR schedule with warmup
  - Gradient accumulation (for small VRAM)
  - FP16 mixed precision (auto-enabled on GPU)
  - Robust checkpoint save/load with resume
  - Hardware-aware batch sizing (see env_config)
"""

import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


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
        GRAD_ACCUM_STEPS, USE_FP16, VOCAB_SIZE, MODEL_SIZE, print_env_summary,
    )
    return dict(
        ENV=ENV, ROOT=ROOT, DATA_DIR=DATA_DIR, SAVE_ROOT=SAVE_ROOT,
        DEVICE=DEVICE, BASE_BATCH_SIZE=BASE_BATCH_SIZE,
        GRAD_ACCUM_STEPS=GRAD_ACCUM_STEPS, USE_FP16=USE_FP16,
        VOCAB_SIZE=VOCAB_SIZE, MODEL_SIZE=MODEL_SIZE,
        print_env_summary=print_env_summary,
    )


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────

class TextDataset(Dataset):
    """
    Simple dataset that tokenizes text files line-by-line.
    Used for pretraining (raw text) and instruction tuning (ChatML).
    """
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
                        # Handle ChatML format (list of messages)
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
        """Convert messages list to ChatML string."""
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
    """
    Dataset for training that handles ALL data schemas used in this project:

      - messages:   [{"role", "content"}, ...]                    (ChatML)
      - conversations: [{"from": "human|gpt", "value": text}, ...] (stindardlogic)
      - input/output, instruction/response                        (codealpaca, codefeedback)
      - content                                                   (starcoder)
      - code/doc, language/name/code                              (codesearchnet)
      - text                                                      (oasst, raw text)
      - question/student_answer/instructor_answer/score           (mohler ASAG)

    Raw code is wrapped in <|code|> ... <|/code|>; conversations become ChatML.
    """
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
        """Convert any record schema into a ChatML messages list (or None)."""
        if not isinstance(rec, dict):
            return None

        # ChatML
        if "messages" in rec and isinstance(rec["messages"], list):
            msgs = []
            for m in rec["messages"]:
                role = m.get("role", "user")
                content = m.get("content", "")
                if content:
                    msgs.append({"role": role, "content": content})
            return msgs or None

        # stindardlogic conversations: [{"from": "human|gpt", "value": ...}]
        if "conversations" in rec and isinstance(rec["conversations"], list):
            msgs = []
            for m in rec["conversations"]:
                role = m.get("from", "user")
                role = "assistant" if role == "gpt" else ("user" if role == "human" else role)
                content = m.get("value", "")
                if content:
                    msgs.append({"role": role, "content": content})
            return msgs or None

        # Instruction pairs
        if "instruction" in rec or "input" in rec or "query" in rec:
            instr = rec.get("instruction") or rec.get("query") or rec.get("input") or ""
            out = rec.get("response") or rec.get("output") or rec.get("answer") or ""
            if instr and out:
                return [{"role": "user", "content": instr},
                        {"role": "assistant", "content": out}]
            if instr:
                return [{"role": "user", "content": instr}]

        # Code + doc (codesearchnet)
        if "code" in rec:
            code = rec["code"] or ""
            doc = rec.get("doc") or ""
            if code:
                code_text = f"<|code|>\n{code}\n<|/code|>"
                if doc:
                    return [{"role": "user", "content": f"Explain: {doc}"},
                            {"role": "assistant", "content": code_text}]
                return [{"role": "user", "content": code_text}]

        # Raw code (starcoder)
        if "content" in rec:
            content = rec["content"]
            if isinstance(content, str) and content.strip():
                return [{"role": "user", "content": f"<|code|>\n{content}\n<|/code|>"}]
            if isinstance(content, list):  # some files store bytes as list
                try:
                    text = bytes(content).decode("utf-8", errors="replace")
                    if text.strip():
                        return [{"role": "user", "content": f"<|code|>\n{text}\n<|/code|>"}]
                except Exception:
                    return None
            return None

        # Mohler ASAG — candidate Q/A with score (evaluator stage)
        if "student_answer" in rec:
            q = rec.get("question", "")
            sa = rec.get("student_answer", "")
            ia = rec.get("instructor_answer", "")
            score = rec.get("score_avg", 0)
            if sa:
                return [{"role": "user",
                         "content": f"Question: {q}\nStudent answer: {sa}\n"
                                    f"Reference answer: {ia}\n\nEvaluate the student answer."},
                        {"role": "assistant",
                         "content": f"Score: {score}/5. The answer "
                                    f"{'covers the key points well' if score >= 3 else 'is incomplete or incorrect'}."}]
            return None

        # Plain text / oasst
        if "text" in rec:
            text = rec["text"]
            if isinstance(text, str) and text.strip():
                return [{"role": "user", "content": text}]
            return None

        # Fallback: stringify simple dict
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
# Learning Rate Schedule
# ─────────────────────────────────────────────────────────────

def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    """Cosine decay with linear warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────
# Training Loop (with grad accumulation + optional FP16)
# ─────────────────────────────────────────────────────────────

def train_epoch(model, dataloader, optimizer, scheduler, device,
                grad_clip=1.0, grad_accum_steps=1, use_fp16=False, start_step=0):
    """Train for one epoch. Returns average loss and total steps run."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()

    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_fp16) if torch.cuda.is_available() else None
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_fp16) if torch.cuda.is_available() else None

    for i, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_fp16):
            result = model(input_ids=input_ids, labels=labels)
            loss = result["loss"] / grad_accum_steps

        if use_fp16 and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        num_batches += 1

        if (i + 1) % grad_accum_steps == 0:
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

        total_loss += loss.item() * grad_accum_steps

        if (num_batches % 50) == 0:
            print(f"    batch {num_batches}/{len(dataloader)}  loss={loss.item()*grad_accum_steps:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}")

    # Flush any remaining gradient accum
    if num_batches % grad_accum_steps != 0:
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if use_fp16 and scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()

    return total_loss / max(num_batches, 1), num_batches


@torch.no_grad()
def evaluate(model, dataloader, device, use_fp16=False):
    """Evaluate model. Returns average loss and perplexity."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast("cuda", enabled=use_fp16):
            result = model(input_ids=input_ids, labels=labels)
            loss = result["loss"]

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    perplexity = math.exp(min(avg_loss, 20))  # Cap to avoid overflow
    return avg_loss, perplexity


# ─────────────────────────────────────────────────────────────
# Checkpoint save/load with resume support
# ─────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scheduler, epoch, loss, path, step=0, extra=None):
    """Save training checkpoint (safe, robust)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "loss": loss,
        "config": model.config.__dict__ if hasattr(model, "config") else {},
    }
    if extra:
        state.update(extra)
    # Atomic write: save to temp then rename (prevents corrupt checkpoints on crash)
    tmp_path = path.with_suffix(".tmp")
    torch.save(state, tmp_path)
    tmp_path.replace(path)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """Load training checkpoint. Returns (epoch, loss, step)."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
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
# Config Presets (hardware-aware)
# ─────────────────────────────────────────────────────────────

def make_training_configs(env):
    """Build hardware-aware training configs."""
    batch = env["BASE_BATCH_SIZE"]
    accum = env["GRAD_ACCUM_STEPS"]
    return {
        "pretrain": {
            "lr": 3e-4, "warmup_ratio": 0.01, "weight_decay": 0.1,
            "betas": (0.9, 0.95), "epochs": 2, "batch_size": batch,
            "grad_clip": 1.0, "max_len": 512, "grad_accum_steps": accum,
        },
        "domain": {
            "lr": 1e-4, "warmup_ratio": 0.05, "weight_decay": 0.1,
            "betas": (0.9, 0.95), "epochs": 3, "batch_size": batch,
            "grad_clip": 1.0, "max_len": 512, "grad_accum_steps": accum,
        },
        "instruction": {
            "lr": 5e-5, "warmup_ratio": 0.05, "weight_decay": 0.05,
            "betas": (0.9, 0.99), "epochs": 2, "batch_size": batch,
            "grad_clip": 1.0, "max_len": 512, "grad_accum_steps": accum,
        },
        "interview": {
            "lr": 2e-5, "warmup_ratio": 0.1, "weight_decay": 0.05,
            "betas": (0.9, 0.99), "epochs": 3, "batch_size": batch,
            "grad_clip": 1.0, "max_len": 768, "grad_accum_steps": accum,
        },
        "evaluator": {
            "lr": 2e-5, "warmup_ratio": 0.1, "weight_decay": 0.05,
            "betas": (0.9, 0.99), "epochs": 4, "batch_size": batch,
            "grad_clip": 1.0, "max_len": 768, "grad_accum_steps": accum,
        },
        "followup": {
            "lr": 2e-5, "warmup_ratio": 0.1, "weight_decay": 0.05,
            "betas": (0.9, 0.99), "epochs": 3, "batch_size": batch,
            "grad_clip": 1.0, "max_len": 512, "grad_accum_steps": accum,
        },
    }