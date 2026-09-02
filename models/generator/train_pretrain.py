"""
models/generator/train_pretrain.py
==================================
Stage 1: Code Pretraining.
Thin wrapper — delegates to unified train.py (env-aware, checkpointed).

Run:
    python models/generator/train_pretrain.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from models.generator.train import main

if __name__ == "__main__":
    # Inject --stage pretrain
    sys.argv = [sys.argv[0], "--stage", "pretrain"] + [a for a in sys.argv[1:] if a not in ("pretrain",)]
    main()
