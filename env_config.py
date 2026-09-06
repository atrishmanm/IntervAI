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
# GPU detection (multi-GPU aware)
# ─────────────────────────────────────────────────────────────

def _detect_gpus():
    """Return a dict describing the available GPU(s)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"available": False, "names": ["CPU"], "vram_gb": 0.0, "count": 0}
        count = torch.cuda.device_count()
        props = torch.cuda.get_device_properties(0)
        vram_gb = round(props.total_memory / (1024 ** 3), 1)
        names = [torch.cuda.get_device_properties(i).name for i in range(count)]
        return {"available": True, "names": names, "vram_gb": vram_gb, "count": count}
    except Exception:
        return {"available": False, "names": ["CPU"], "vram_gb": 0.0, "count": 0}


GPU = _detect_gpus()
DEVICE = "cuda" if GPU["available"] else "cpu"
GPU_NAME = GPU["names"][0]
VRAM_GB = GPU["vram_gb"]
NUM_GPUS = GPU["count"]

# ─────────────────────────────────────────────────────────────
# Hardware-aware training config
# ─────────────────────────────────────────────────────────────
# Hardware-aware training config (120M model, <8h total)
# ─────────────────────────────────────────────────────────────
# Small VRAM (4GB GTX 1650): batch 1, grad accumulation 16
# Large VRAM (16GB T4/P100): batch 8, grad accumulation 8
# CPU: batch 1 (for smoke tests / tiny debugging)
# ─────────────────────────────────────────────────────────────

if VRAM_GB >= 12:
    BASE_BATCH_SIZE = 8
    GRAD_ACCUM_STEPS = 8
    USE_FP16 = True
elif VRAM_GB >= 8:
    BASE_BATCH_SIZE = 4
    GRAD_ACCUM_STEPS = 8
    USE_FP16 = True
elif VRAM_GB >= 4:
    BASE_BATCH_SIZE = 1
    GRAD_ACCUM_STEPS = 16
    USE_FP16 = True
else:
    BASE_BATCH_SIZE = 1
    GRAD_ACCUM_STEPS = 1
    USE_FP16 = False

# Effective batch = BASE_BATCH_SIZE * NUM_GPUS * GRAD_ACCUM_STEPS
EFFECTIVE_BATCH_SIZE = BASE_BATCH_SIZE * NUM_GPUS * GRAD_ACCUM_STEPS

# Which model size to use per GPU tier:
#   CPU/small VRAM (<4GB)  → Small model (14M)
#   4-8GB VRAM (GTX 1650)  → Small model (14M)
#   >=8GB VRAM (RTX 3060)  → Medium model (47M)
#   >=12GB VRAM (T4/P100)  → Large model (126M)
MODEL_SIZE = "large" if VRAM_GB >= 12 else ("medium" if VRAM_GB >= 8 else "small")

# Allow override via env var (e.g. INTERVUE_MODEL=small for fast smoke tests)
_MODEL_OVERRIDE = os.environ.get("INTERVUE_MODEL", "").strip().lower()
if _MODEL_OVERRIDE in ("small", "medium", "large"):
    MODEL_SIZE = _MODEL_OVERRIDE

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
    if GPU["available"]:
        gpu_str = " + ".join(GPU["names"]) + f" ({len(GPU['names'])}x)"
    else:
        gpu_str = "NONE (CPU)"
    print(f"  GPU:          {gpu_str} ({VRAM_GB} GB each)" if GPU["available"] else "  GPU:          NONE (CPU)")
    print(f"  Device:       {DEVICE}  (GPUs: {NUM_GPUS})")
    print(f"  Base batch:   {BASE_BATCH_SIZE}  x{NUM_GPUS} GPU(s) x{GRAD_ACCUM_STEPS} accum = eff {EFFECTIVE_BATCH_SIZE}")
    print(f"  FP16:         {USE_FP16}")
    print(f"  Model size:   {MODEL_SIZE}")
    print(f"  ROOT:         {ROOT}")
    print(f"  Data:         {DATA_DIR}")
    print(f"  Checkpoints:  {SAVE_ROOT}")
    print("=" * 60)