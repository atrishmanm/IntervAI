"""
env_config.py
==============
Single source of truth for environment detection and hardware config.

Auto-detects:
  - Local laptop (GTX 1650, 4GB VRAM)  → data/ dir, small batch
  - Kaggle notebook (T4/P100, 16GB)    → /kaggle/working, large batch
  - Colab (T4, 16GB)                   → /content, large batch

Import this in EVERY training script so the same code runs on
both machines with zero modifications.

Usage:
    from env_config import ENV, ROOT, DEVICE, BATCH_SIZE, DATA_DIR, ...
"""

import os
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Environment detection
# ─────────────────────────────────────────────────────────────

IS_KAGGLE = Path("/kaggle/working").exists()
IS_COLAB = Path("/content").exists()

if IS_KAGGLE:
    ENV = "kaggle"
    ROOT = Path("/kaggle/working/IntervAI")
    DATA_DIR = Path("/kaggle/input")  # datasets mounted here by Kaggle
    SAVE_ROOT = ROOT / "models" / "generator" / "saved"
    RESULTS_DIR = Path("/kaggle/working")
elif IS_COLAB:
    ENV = "colab"
    ROOT = Path("/content/IntervAI")
    DATA_DIR = ROOT / "data"
    SAVE_ROOT = ROOT / "models" / "generator" / "saved"
    RESULTS_DIR = Path("/content")
else:
    ENV = "local"
    ROOT = Path(__file__).resolve().parent
    DATA_DIR = ROOT / "data"
    SAVE_ROOT = ROOT / "models" / "generator" / "saved"
    RESULTS_DIR = ROOT

# Add root to path so imports work everywhere
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────
# GPU detection
# ─────────────────────────────────────────────────────────────

def _detect_gpu():
    """Return a dict describing the available GPU."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"available": False, "name": "CPU", "vram_gb": 0}
        props = torch.cuda.get_device_properties(0)
        vram_gb = round(props.total_memory / (1024 ** 3), 1)
        return {"available": True, "name": props.name, "vram_gb": vram_gb}
    except Exception:
        return {"available": False, "name": "CPU", "vram_gb": 0}


GPU = _detect_gpu()
DEVICE = "cuda" if GPU["available"] else "cpu"
GPU_NAME = GPU["name"]
VRAM_GB = GPU["vram_gb"]

# ─────────────────────────────────────────────────────────────
# Hardware-aware training config
# ─────────────────────────────────────────────────────────────
# Small VRAM (4GB GTX 1650): batch 2, grad accumulation 8
# Large VRAM (16GB T4/P100): batch 16, grad accumulation 4
# CPU: batch 1 (for smoke tests / tiny debugging)
# ─────────────────────────────────────────────────────────────

if VRAM_GB >= 12:
    BASE_BATCH_SIZE = 16
    GRAD_ACCUM_STEPS = 4
    USE_FP16 = True
elif VRAM_GB >= 4:
    BASE_BATCH_SIZE = 2
    GRAD_ACCUM_STEPS = 8
    USE_FP16 = True
else:
    BASE_BATCH_SIZE = 1
    GRAD_ACCUM_STEPS = 1
    USE_FP16 = False

# Effective batch = BASE_BATCH_SIZE * GRAD_ACCUM_STEPS
EFFECTIVE_BATCH_SIZE = BASE_BATCH_SIZE * GRAD_ACCUM_STEPS

# Which model size to use per GPU:
#   small VRAM  → Small model (5.9M) for safety
#   large VRAM  → Medium model (10.7M)
MODEL_SIZE = "small" if VRAM_GB < 8 else "medium"

# ─────────────────────────────────────────────────────────────
# Tokens / vocab
# ─────────────────────────────────────────────────────────────

VOCAB_SIZE = 16_000
MAX_LEN = 1024

# ─────────────────────────────────────────────────────────────
# Friendly summary
# ─────────────────────────────────────────────────────────────

def print_env_summary():
    """Print what was detected (call at the start of each training script)."""
    print("=" * 60)
    print(f"  Environment:  {ENV.upper()}")
    print(f"  GPU:          {GPU_NAME} ({VRAM_GB} GB)" if GPU["available"] else "  GPU:          NONE (CPU)")
    print(f"  Device:       {DEVICE}")
    print(f"  Base batch:   {BASE_BATCH_SIZE}  (grad accum x{GRAD_ACCUM_STEPS} = eff {EFFECTIVE_BATCH_SIZE})")
    print(f"  FP16:         {USE_FP16}")
    print(f"  Model size:   {MODEL_SIZE}")
    print(f"  ROOT:         {ROOT}")
    print(f"  Data:         {DATA_DIR}")
    print(f"  Checkpoints:  {SAVE_ROOT}")
    print("=" * 60)